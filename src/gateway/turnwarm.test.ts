// Tests for the turn warmer against a mock vLLM server (chat completions,
// /tokenize and /v1/completions). Run: npm run test:gateway

import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { Gateway } from './server.js';
import { defaultConfig, loadConfig } from './config.js';
import type { GatewayConfig } from './config.js';
import { sseData } from './sse.js';
import { setLogLevel, setLogSink } from './log.js';
import type { Obj } from './translate.js';
import { defaultTurnWarmConfig, planTurnWarm, tokenizeRequest, turnWarmPayload, warmStages } from './turnwarm.js';
import type { TurnWarmConfig } from './turnwarm.js';

setLogLevel('error');

const BLOCK = 1728;

interface Usage { prompt: number; cached: number; completion: number }

interface Mock {
  url: string;
  /** Chat-completions bodies. */
  chats: Obj[];
  /** /tokenize bodies. */
  tokenizes: Obj[];
  /** /v1/completions bodies. */
  completions: Obj[];
  /** The usage of the next chat answers, in order. The last one repeats. */
  usages: Usage[];
  /** The /tokenize count minus the prompt tokens of the chat answer with the same messages. */
  countOffset: number;
  /** Keep /v1/completions requests open until `releaseWarm` is called or the client closes them. */
  holdWarm: boolean;
  /** Let the oldest held /v1/completions request answer. */
  releaseWarm: () => void;
  /** Keep /tokenize requests open until the client closes them. */
  holdTokenize: boolean;
  closedTokenize: number;
  /** Keep chat requests open until `releaseOne` is called. */
  holdChat: boolean;
  /** Let the oldest held chat request answer. */
  releaseOne: () => void;
  closedWarm: number;
  /** The order of the requests: 'chat', 'chat-end', 'tokenize', 'completions'. */
  order: string[];
  close: () => Promise<void>;
}

async function mockVllm(): Promise<Mock> {
  /** Prompt tokens of each chat answer, by its messages. */
  const prompts = new Map<string, number>();
  let served = 0;
  const waiting: Array<() => void> = [];
  const warms: Array<() => void> = [];
  const mock = {
    chats: [] as Obj[], tokenizes: [] as Obj[], completions: [] as Obj[], order: [] as string[],
    usages: [{ prompt: 12200, cached: 10368, completion: 20 }],
    countOffset: 0, holdWarm: false, holdChat: false, closedWarm: 0, holdTokenize: false, closedTokenize: 0,
    releaseOne: () => waiting.shift()?.(),
    releaseWarm: () => warms.shift()?.(),
  } as unknown as Mock;
  const server = http.createServer(async (req, res) => {
    if (req.url === '/health') { res.writeHead(200); res.end(); return; }
    let raw = '';
    for await (const c of req) raw += c;
    const body = JSON.parse(raw);
    if (req.url === '/tokenize') {
      mock.tokenizes.push(body);
      mock.order.push('tokenize');
      if (mock.holdTokenize) {
        res.on('close', () => { if (!res.writableEnded) mock.closedTokenize++; });
        return;
      }
      // Like vLLM, /tokenize renders the messages as they are.
      const count = (prompts.get(JSON.stringify(body.messages)) ?? 0) + mock.countOffset;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ count, max_model_len: 524288, tokens: Array.from({ length: count }, (_, i) => i % 1000) }));
      return;
    }
    if (req.url === '/v1/completions') {
      mock.completions.push(body);
      mock.order.push('completions');
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      if (mock.holdWarm) {
        let released = false;
        res.on('close', () => { if (!res.writableEnded && !released) mock.closedWarm++; });
        await new Promise<void>((r) => warms.push(() => { released = true; r(); }));
        if (res.destroyed) return;
      }
      res.write(`data: ${JSON.stringify({ choices: [{ index: 0, text: 'x', finish_reason: 'length' }] })}\n\n`);
      // vLLM limits the hit to the prompt minus one token: the boundary before the end.
      const usage = { prompt_tokens: body.prompt.length, completion_tokens: 1, prompt_tokens_details: { cached_tokens: body.prompt.length - BLOCK } };
      res.write(`data: ${JSON.stringify({ choices: [], usage })}\n\n`);
      res.end('data: [DONE]\n\n');
      return;
    }
    mock.chats.push(body);
    mock.order.push('chat');
    // Like vLLM, the chat request renames reasoning_content to reasoning before the render.
    for (const m of body.messages ?? []) {
      if (m.reasoning_content != null) {
        if (m.reasoning == null) m.reasoning = m.reasoning_content;
        delete m.reasoning_content;
      }
    }
    const u = mock.usages[Math.min(served++, mock.usages.length - 1)];
    if (mock.holdChat) await new Promise<void>((r) => waiting.push(r));
    prompts.set(JSON.stringify(body.messages), u.prompt);
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const chunks = [
      { choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      { choices: [], usage: { prompt_tokens: u.prompt, completion_tokens: u.completion, total_tokens: u.prompt + u.completion, prompt_tokens_details: { cached_tokens: u.cached } } },
    ];
    for (const c of chunks) res.write(`data: ${JSON.stringify(c)}\n\n`);
    mock.order.push('chat-end');
    res.end('data: [DONE]\n\n');
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address() as AddressInfo;
  mock.url = `http://127.0.0.1:${port}/v1`;
  mock.close = () => new Promise((r) => { server.closeAllConnections(); server.close(() => r()); });
  return mock;
}

async function startGateway(mock: Mock, turnWarm: Partial<TurnWarmConfig> = { enabled: true }) {
  const config: GatewayConfig = {
    ...defaultConfig(),
    host: '127.0.0.1',
    port: 0,
    healthIntervalMs: 0,
    models: [{ id: 'qwen3.8-flash-next', profile: 'qwen38', endpoints: [{ name: 'spark2', baseUrl: mock.url }] }],
    turnWarm: { ...defaultTurnWarmConfig(), ...turnWarm },
  };
  const gateway = new Gateway(config);
  await gateway.start();
  return { gateway, base: `http://127.0.0.1:${gateway.address!.port}` };
}

const codexRequest = (session: string, extra: Obj = {}): Obj => ({
  model: 'qwen3.8-flash-next',
  instructions: 'You are Codex.',
  input: [
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: `List the files of ${session}.` }] },
    { type: 'function_call', call_id: 'c1', name: 'exec_command', arguments: '{"cmd":"ls"}' },
    { type: 'function_call_output', call_id: 'c1', output: 'a.txt\nb.txt' },
  ],
  tools: [{ type: 'function', name: 'exec_command', description: 'Run a command.', parameters: { type: 'object', properties: {} } }],
  reasoning: { effort: 'medium' },
  stream: true,
  store: false,
  prompt_cache_key: session,
  ...extra,
});

async function send(base: string, body: Obj): Promise<Obj[]> {
  const res = await fetch(`${base}/v1/responses`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (body.stream === false) return [await res.json() as Obj];
  const out: Obj[] = [];
  for await (const data of sseData(res.body!)) out.push(JSON.parse(data));
  return out;
}

async function until(check: () => boolean, ms = 3000): Promise<void> {
  const end = Date.now() + ms;
  while (!check()) {
    if (Date.now() > end) throw new Error('condition not met in time');
    await new Promise((r) => setTimeout(r, 10));
  }
}

test('planTurnWarm: warm to the last boundary that the prefill crossed', () => {
  // The prefill ran from 10368 (6 blocks) to 12200 and crossed 12096.
  assert.deepEqual(planTurnWarm({ promptTokens: 12200, cachedTokens: 10368, completionTokens: 20 }, BLOCK, 256), { warm: true, target: 12096, from: 10368 });
  // A prefill that crossed three boundaries: the target is the last one.
  assert.deepEqual(planTurnWarm({ promptTokens: 15700, cachedTokens: 10368, completionTokens: 5 }, BLOCK, 256), { warm: true, target: 15552, from: 10368 });
  // A cold prompt.
  assert.deepEqual(planTurnWarm({ promptTokens: 11213, cachedTokens: 0, completionTokens: 1 }, BLOCK, 256), { warm: true, target: 10368, from: 0 });
});

test('planTurnWarm: the boundaries that the previous prefill crossed are reusable', () => {
  const turn = (promptTokens: number, cachedTokens: number, completionTokens = 20) => ({ promptTokens, cachedTokens, completionTokens });
  // The chain of the A/B: turn 3 (17376, 13824 cached) after turn 2 (16337).
  // Turn 2 crossed 15552, so turn 3 made it reusable; only 17280 is new.
  assert.deepEqual(planTurnWarm(turn(17376, 13824), BLOCK, 256, turn(16337, 12096)), { warm: true, target: 17280, from: 15552 });
  // The previous prefill crossed B too: nothing to warm.
  assert.deepEqual(planTurnWarm(turn(16400, 12096), BLOCK, 256, turn(16337, 10368)), { warm: false, reason: 'crossed_before', boundary: 15552 });
  // A shorter prompt (a compaction) or a smaller cached part (a new effort level) does not continue the previous prompt.
  assert.deepEqual(planTurnWarm(turn(15700, 10368), BLOCK, 256, turn(16337, 10368)), { warm: true, target: 15552, from: 10368 });
  assert.deepEqual(planTurnWarm(turn(17376, 1728), BLOCK, 256, turn(16337, 12096)), { warm: true, target: 17280, from: 1728 });
  // A previous prompt that ended before C changes nothing.
  assert.deepEqual(planTurnWarm(turn(15700, 10368), BLOCK, 256, turn(11000, 8640)), { warm: true, target: 15552, from: 10368 });
});

test('planTurnWarm: skip cases', () => {
  // No boundary between the cached part and the end of the prompt.
  assert.deepEqual(planTurnWarm({ promptTokens: 12000, cachedTokens: 10368, completionTokens: 30 }, BLOCK, 256), { warm: false, reason: 'no_boundary' });
  // The decode crossed 12096, a later boundary that is reusable.
  assert.deepEqual(planTurnWarm({ promptTokens: 12000, cachedTokens: 8640, completionTokens: 200 }, BLOCK, 256), { warm: false, reason: 'decode_crossed', boundary: 12096 });
  // The last output token is not fed back, so it does not cross a boundary.
  assert.deepEqual(planTurnWarm({ promptTokens: 12095, cachedTokens: 8640, completionTokens: 1 }, BLOCK, 256), { warm: true, target: 10368, from: 8640 });
  assert.deepEqual(planTurnWarm({ promptTokens: 12094, cachedTokens: 8640, completionTokens: 2 }, BLOCK, 256), { warm: true, target: 10368, from: 8640 });
  assert.deepEqual(planTurnWarm({ promptTokens: 12095, cachedTokens: 8640, completionTokens: 2 }, BLOCK, 256), { warm: false, reason: 'decode_crossed', boundary: 12096 });
  // A gain less than the minimum (possible only with a small block).
  assert.deepEqual(planTurnWarm({ promptTokens: 300, cachedTokens: 128, completionTokens: 1 }, 128, 256), { warm: false, reason: 'small_gain' });
  assert.deepEqual(planTurnWarm({ promptTokens: 0, cachedTokens: 0, completionTokens: 0 }, BLOCK, 256), { warm: false, reason: 'no_usage' });
  // The prefill ended on the boundary, so the state there is kept.
  assert.deepEqual(planTurnWarm({ promptTokens: 12096, cachedTokens: 10368, completionTokens: 1 }, BLOCK, 256), { warm: false, reason: 'no_boundary' });
});

test('tokenizeRequest renders the chat payload with the effort as a template argument', () => {
  const payload = {
    model: 'm', messages: [{ role: 'user', content: 'x' }], tools: [{ type: 'function', function: { name: 'f' } }],
    chat_template_kwargs: { enable_thinking: true }, reasoning_effort: 'medium', stream: true, max_tokens: 9,
  };
  assert.deepEqual(tokenizeRequest(payload), {
    model: 'm', messages: payload.messages, tools: payload.tools, add_generation_prompt: true,
    chat_template_kwargs: { enable_thinking: true, reasoning_effort: 'medium' },
  });
  assert.deepEqual(tokenizeRequest({ model: 'm', messages: [], tools: [] }), { model: 'm', messages: [], add_generation_prompt: true });
  // vLLM renames reasoning_content to reasoning for a chat request only.
  const messages = [
    { role: 'user', content: 'x' },
    { role: 'assistant', content: null, reasoning_content: 'plan' },
    { role: 'assistant', content: 'y', reasoning_content: 'old', reasoning: 'new' },
  ];
  assert.deepEqual(tokenizeRequest({ model: 'm', messages }).messages, [
    { role: 'user', content: 'x' },
    { role: 'assistant', content: null, reasoning: 'plan' },
    { role: 'assistant', content: 'y', reasoning: 'new' },
  ]);
  assert.equal(messages[1].reasoning_content, 'plan');
  const plain = [{ role: 'user', content: 'x' }];
  assert.equal(tokenizeRequest({ model: 'm', messages: plain }).messages, plain);
  const warm = turnWarmPayload('m', [1, 2, 3, 4, 5], 3);
  assert.deepEqual(warm.prompt, [1, 2, 3]);
  assert.equal(warm.max_tokens, 1);
});

test('after a turn the gateway renders the payload and prefills the first B tokens', async () => {
  const mock = await mockVllm();
  const { gateway, base } = await startGateway(mock);
  try {
    const evs = await send(base, codexRequest('s1'));
    assert.equal(evs.at(-1)!.type, 'response.completed');
    await until(() => gateway.turnWarmer.stats.completed === 1);
    // The render has the same messages, tools and template arguments.
    const chat = mock.chats[0];
    assert.equal(mock.tokenizes.length, 1);
    assert.deepEqual(mock.tokenizes[0].messages, chat.messages);
    assert.deepEqual(mock.tokenizes[0].tools, chat.tools);
    assert.deepEqual(mock.tokenizes[0].chat_template_kwargs, { ...chat.chat_template_kwargs, reasoning_effort: 'medium' });
    assert.equal(mock.tokenizes[0].add_generation_prompt, true);
    // The warm prompt ends on the boundary.
    assert.equal(mock.completions.length, 1);
    assert.equal(mock.completions[0].prompt.length, 12096);
    assert.equal(mock.completions[0].max_tokens, 1);
    assert.equal(mock.completions[0].model, chat.model);
    assert.deepEqual(mock.order, ['chat', 'chat-end', 'tokenize', 'completions']);

    // A non-streaming turn also triggers a warm.
    await send(base, codexRequest('s1', { stream: false }));
    await until(() => gateway.turnWarmer.stats.completed === 2);

    const metrics = (await (await fetch(`${base}/metrics`)).json() as Obj).turn_warm;
    assert.equal(metrics.enabled, true);
    assert.equal(metrics.block_tokens, BLOCK);
    assert.equal(metrics.triggers, 2);
    assert.equal(metrics.requests, 2);
    assert.equal(metrics.completed, 2);
    assert.equal(metrics.prompt_tokens, 2 * 12096);
    assert.equal(metrics.cached_tokens, 2 * 10368);
    assert.equal(metrics.warmed_tokens, 2 * BLOCK);
    assert.equal(metrics.gain_tokens, 2 * BLOCK);
    assert.equal(metrics.last.warm_prompt_tokens, 12096);
    assert.equal(typeof metrics.time_ms, 'number');
    // The second turn had 10368 cached, less than the boundary of the first.
    assert.deepEqual(metrics.next_turn.after_warm, { hit: 0, miss: 1 });
  } finally { await gateway.stop(); await mock.close(); }
});

test('the next turn counts a hit at the warmed boundary', async () => {
  const mock = await mockVllm();
  mock.usages = [{ prompt: 12200, cached: 10368, completion: 20 }, { prompt: 12500, cached: 12096, completion: 20 }];
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => gateway.turnWarmer.stats.completed === 1);
    await send(base, codexRequest('s1'));
    await gateway.turnWarmer.idle();
    const s = gateway.turnWarmer.snapshot();
    assert.deepEqual(s.next_turn.after_warm, { hit: 1, miss: 0 });
    // The second turn crossed no new boundary.
    assert.equal(s.skipped.no_boundary, 1);
    assert.equal(mock.completions.length, 1);
  } finally { await gateway.stop(); await mock.close(); }
});

test('a turn that crosses again the boundaries of the previous prefill is not warmed', async () => {
  const mock = await mockVllm();
  // Turn 1 crosses 12096 and 13824; the warm makes them reusable. Turn 2 has no
  // warm hit (for example the warm could not run) and crosses them again.
  mock.usages = [{ prompt: 14000, cached: 10368, completion: 5 }, { prompt: 14100, cached: 10368, completion: 5 }];
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => gateway.turnWarmer.stats.completed === 1);
    await send(base, codexRequest('s1'));
    await gateway.turnWarmer.idle();
    const m = gateway.turnWarmer.snapshot();
    assert.equal(m.skipped.crossed_before, 1);
    assert.deepEqual(mock.completions.map((c) => c.prompt.length), [12096, 13824]);
    // Another session does not use the prompt of s1.
    await send(base, codexRequest('s2'));
    await until(() => gateway.turnWarmer.stats.completed === 2);
    assert.deepEqual(mock.completions.map((c) => c.prompt.length), [12096, 13824, 12096, 13824]);
  } finally { await gateway.stop(); await mock.close(); }
});

test('a render with another token count is not warmed', async () => {
  const mock = await mockVllm();
  mock.countOffset = 3;
  const lines: string[] = [];
  setLogLevel('info');
  setLogSink((_level, line) => lines.push(line));
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => gateway.turnWarmer.stats.skipped.count_mismatch === 1);
    await gateway.turnWarmer.idle();
    assert.equal(mock.tokenizes.length, 1);
    assert.equal(mock.completions.length, 0);
    assert.equal(gateway.turnWarmer.stats.completed, 0);
    const line = lines.map((l) => JSON.parse(l)).find((l) => l.msg === 'turn warm skipped; the render does not match the prompt');
    assert.equal(line.count, 12203);
    assert.equal(line.prompt_tokens, 12200);
  } finally {
    setLogSink(null);
    setLogLevel('error');
    await gateway.stop(); await mock.close();
  }
});

test('turns that cross no boundary, or cross one in the decode, send no warm request', async () => {
  const mock = await mockVllm();
  mock.usages = [{ prompt: 12000, cached: 10368, completion: 30 }, { prompt: 12000, cached: 8640, completion: 200 }];
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await send(base, codexRequest('s2'));
    await gateway.turnWarmer.idle();
    assert.equal(mock.tokenizes.length, 0);
    assert.equal(mock.completions.length, 0);
    assert.deepEqual(gateway.turnWarmer.stats.skipped, { no_boundary: 1, decode_crossed: 1 });
  } finally { await gateway.stop(); await mock.close(); }
});

test('warmStages: one stage per step up to the target', () => {
  assert.deepEqual(warmStages(10368, 12096, BLOCK, 1), [12096]);
  assert.deepEqual(warmStages(10368, 15552, BLOCK, 1), [12096, 13824, 15552]);
  assert.deepEqual(warmStages(10368, 15552, BLOCK, 2), [13824, 15552]);
  assert.deepEqual(warmStages(10368, 15552, BLOCK, 3), [15552]);
  assert.deepEqual(warmStages(0, 3456, BLOCK, 1), [1728, 3456]);
  // A cached count that is not on a boundary starts at the boundary before it.
  assert.deepEqual(warmStages(10080, 12096, BLOCK, 1), [10368, 12096]);
  assert.deepEqual(warmStages(12096, 12096, BLOCK, 1), []);
});

test('a warm over several boundaries prefills one block per stage', async () => {
  const mock = await mockVllm();
  mock.usages = [{ prompt: 15700, cached: 10368, completion: 5 }];
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => gateway.turnWarmer.stats.completed === 1);
    assert.equal(mock.tokenizes.length, 1);
    assert.deepEqual(mock.completions.map((c) => c.prompt.length), [12096, 13824, 15552]);
    const m = gateway.turnWarmer.snapshot();
    assert.equal(m.requests, 3);
    assert.equal(m.stages, 3);
    assert.equal(m.warmed_tokens, 3 * BLOCK);
    assert.equal(m.gain_tokens, 15552 - 10368);
    assert.equal(m.last.reached, 15552);
    assert.equal(m.last.stages, 3);
    assert.equal(m.last.warm_cached_tokens, 10368);
  } finally { await gateway.stop(); await mock.close(); }
});

test('a real request does not abort the stage in flight; the next stage waits until the gateway is idle', async () => {
  const mock = await mockVllm();
  // s1 crosses three boundaries; s2 crosses none.
  mock.usages = [{ prompt: 15700, cached: 10368, completion: 5 }, { prompt: 12000, cached: 10368, completion: 5 }];
  mock.holdWarm = true;
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => mock.completions.length === 1);
    mock.holdChat = true;
    const second = send(base, codexRequest('s2'));
    await until(() => mock.chats.length === 2);
    // The stage completes while s2 is in flight, and no new stage starts.
    mock.holdWarm = false;
    mock.releaseWarm();
    await until(() => gateway.turnWarmer.stats.yielded === 1);
    await new Promise((r) => setTimeout(r, 200));
    assert.equal(mock.completions.length, 1);
    mock.releaseOne();
    await second;
    await until(() => gateway.turnWarmer.stats.completed === 1);
    const s = gateway.turnWarmer.stats;
    assert.equal(mock.closedWarm, 0);
    assert.equal(s.aborted, 0);
    assert.equal(s.overlaps, 1);
    // The warm goes on from the stage that completed, with the same render.
    assert.equal(mock.tokenizes.length, 1);
    assert.deepEqual(mock.completions.map((c) => c.prompt.length), [12096, 13824, 15552]);
  } finally { await gateway.stop(); await mock.close(); }
});

test('a real request of the same session ends the warm after the stage in flight', async () => {
  const mock = await mockVllm();
  mock.usages = [{ prompt: 15700, cached: 10368, completion: 5 }, { prompt: 17400, cached: 12096, completion: 1 }];
  mock.holdWarm = true;
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => mock.completions.length === 1);
    mock.holdChat = true;
    const second = send(base, codexRequest('s1'));
    await until(() => mock.chats.length === 2);
    mock.holdWarm = false;
    mock.releaseWarm();
    await until(() => gateway.turnWarmer.stats.yielded === 1);
    mock.releaseOne();
    await second;
    // The second turn found the boundary of the first stage and makes its own
    // warm. It crossed 13824 and 15552 for the second time, so only 17280 is new.
    await until(() => gateway.turnWarmer.stats.completed === 1);
    await gateway.turnWarmer.idle();
    const s = gateway.turnWarmer.stats;
    assert.equal(s.skipped.superseded, 1);
    assert.equal(mock.tokenizes.length, 2);
    assert.deepEqual(mock.completions.map((c) => c.prompt.length), [12096, 17280]);
    assert.equal(gateway.turnWarmer.snapshot().last.from, 15552);
    assert.equal(mock.closedWarm, 0);
  } finally { await gateway.stop(); await mock.close(); }
});

test('a real request aborts the /tokenize request; the warm goes again when the gateway is idle', async () => {
  const mock = await mockVllm();
  mock.usages = [{ prompt: 12200, cached: 10368, completion: 20 }, { prompt: 12000, cached: 10368, completion: 20 }];
  mock.holdTokenize = true;
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => mock.tokenizes.length === 1);
    mock.holdTokenize = false;
    await send(base, codexRequest('s2'));
    await until(() => gateway.turnWarmer.stats.completed === 1);
    await gateway.turnWarmer.idle();
    assert.equal(mock.closedTokenize, 1);
    assert.equal(gateway.turnWarmer.stats.aborted, 1);
    assert.equal(mock.tokenizes.length, 2);
    assert.deepEqual(mock.completions.map((c) => c.prompt.length), [12096]);
  } finally { await gateway.stop(); await mock.close(); }
});

test('the shutdown aborts the stage in flight', async () => {
  const mock = await mockVllm();
  mock.holdWarm = true;
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => mock.completions.length === 1);
  } finally { await gateway.stop(); }
  await until(() => mock.closedWarm === 1);
  assert.equal(gateway.turnWarmer.stats.timeouts, 0);
  await mock.close();
});

test('a turn with replayed reasoning renders the same count at /tokenize', async () => {
  const mock = await mockVllm();
  const { gateway, base } = await startGateway(mock);
  try {
    const request = codexRequest('s1');
    request.input.splice(1, 0, { type: 'reasoning', summary: [{ type: 'summary_text', text: 'I list the files.' }] });
    await send(base, request);
    await until(() => gateway.turnWarmer.stats.completed === 1);
    const messages = mock.tokenizes[0].messages as Obj[];
    const assistant = messages.find((m) => m.reasoning != null)!;
    assert.equal(assistant.reasoning, 'I list the files.');
    assert.equal('reasoning_content' in assistant, false);
    assert.equal(gateway.turnWarmer.stats.skipped.count_mismatch, undefined);
  } finally { await gateway.stop(); await mock.close(); }
});

test('no warm request while a real request is in flight', async () => {
  const mock = await mockVllm();
  // s1 crosses a boundary in the prefill; s2 does not.
  mock.usages = [{ prompt: 12200, cached: 10368, completion: 20 }, { prompt: 12000, cached: 10368, completion: 20 }];
  const { gateway, base } = await startGateway(mock);
  try {
    mock.holdChat = true;
    const first = send(base, codexRequest('s1'));
    await until(() => mock.chats.length === 1);
    const second = send(base, codexRequest('s2'));
    await until(() => mock.chats.length === 2);
    // s1 ends; s2 is still in flight, so the warm of s1 waits.
    mock.releaseOne();
    await first;
    await new Promise((r) => setTimeout(r, 300));
    assert.equal(mock.tokenizes.length, 0);
    assert.equal(gateway.turnWarmer.snapshot().queued, 1);
    mock.releaseOne();
    await second;
    await until(() => gateway.turnWarmer.stats.completed === 1);
    assert.deepEqual(mock.order, ['chat', 'chat', 'chat-end', 'chat-end', 'tokenize', 'completions']);
    assert.equal(mock.completions[0].prompt.length, 12096);
  } finally { await gateway.stop(); await mock.close(); }
});

test('the turn warmer is off by default', async () => {
  const mock = await mockVllm();
  const { gateway, base } = await startGateway(mock, {});
  try {
    assert.equal(defaultConfig().turnWarm.enabled, false);
    await send(base, codexRequest('s1'));
    await new Promise((r) => setTimeout(r, 100));
    assert.equal(mock.tokenizes.length, 0);
    assert.equal(mock.completions.length, 0);
    const metrics = (await (await fetch(`${base}/metrics`)).json() as Obj).turn_warm;
    assert.equal(metrics.enabled, false);
    assert.equal(metrics.triggers, 0);
  } finally { await gateway.stop(); await mock.close(); }
});

test('a warm request that takes too long stops at the time limit', async () => {
  const mock = await mockVllm();
  mock.holdWarm = true;
  const { gateway, base } = await startGateway(mock, { enabled: true, timeoutMs: 100 });
  try {
    await send(base, codexRequest('s1'));
    await until(() => gateway.turnWarmer.stats.timeouts === 1);
    await gateway.turnWarmer.idle();
    await until(() => mock.closedWarm === 1);
    assert.equal(gateway.turnWarmer.stats.aborted, 0);
    assert.equal(mock.completions.length, 1);
  } finally { await gateway.stop(); await mock.close(); }
});

test('cached tokens that are not a multiple of the block are counted', async () => {
  const mock = await mockVllm();
  mock.usages = [{ prompt: 12200, cached: 10080, completion: 20 }];
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await gateway.turnWarmer.idle();
    assert.equal(gateway.turnWarmer.stats.blockMismatch, 1);
  } finally { await gateway.stop(); await mock.close(); }
});

test('config: the turn warmer settings come from the environment and are checked', () => {
  const config = loadConfig({
    PULSE_GATEWAY_TURN_WARM: '1',
    PULSE_GATEWAY_PREFIX_BLOCK_TOKENS: '1680',
    PULSE_GATEWAY_TURN_WARM_MIN_GAIN_TOKENS: '512',
    PULSE_GATEWAY_TURN_WARM_TIMEOUT_MS: '5000',
    PULSE_GATEWAY_TURN_WARM_STEP_BLOCKS: '2',
  }, []);
  assert.deepEqual(config.turnWarm, { ...defaultTurnWarmConfig(), enabled: true, blockTokens: 1680, minGainTokens: 512, timeoutMs: 5000, stepBlocks: 2 });
  assert.equal(defaultTurnWarmConfig().stepBlocks, 1);
  assert.throws(() => loadConfig({ PULSE_GATEWAY_TURN_WARM_STEP_BLOCKS: '0' }, []), /turn warm step/);
  assert.equal(loadConfig({}, []).turnWarm.enabled, false);
  assert.equal(loadConfig({ PULSE_GATEWAY_TURN_WARM: '0' }, []).turnWarm.enabled, false);
  assert.throws(() => loadConfig({ PULSE_GATEWAY_PREFIX_BLOCK_TOKENS: '0' }, []), /prefix block/);
});
