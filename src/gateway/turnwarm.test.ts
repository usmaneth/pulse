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
import { defaultTurnWarmConfig, planTurnWarm, tokenizeRequest, turnWarmPayload } from './turnwarm.js';
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
  /** cached_tokens of a warm request. */
  warmCached: number;
  /** Keep /v1/completions requests open until the client closes them. */
  holdWarm: boolean;
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
  const mock = {
    chats: [] as Obj[], tokenizes: [] as Obj[], completions: [] as Obj[], order: [] as string[],
    usages: [{ prompt: 12200, cached: 10368, completion: 20 }],
    countOffset: 0, warmCached: 10368, holdWarm: false, holdChat: false, closedWarm: 0,
    releaseOne: () => waiting.shift()?.(),
  } as unknown as Mock;
  const server = http.createServer(async (req, res) => {
    if (req.url === '/health') { res.writeHead(200); res.end(); return; }
    let raw = '';
    for await (const c of req) raw += c;
    const body = JSON.parse(raw);
    if (req.url === '/tokenize') {
      mock.tokenizes.push(body);
      mock.order.push('tokenize');
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
        res.on('close', () => { if (!res.writableEnded) mock.closedWarm++; });
        return;
      }
      res.write(`data: ${JSON.stringify({ choices: [{ index: 0, text: 'x', finish_reason: 'length' }] })}\n\n`);
      const usage = { prompt_tokens: body.prompt.length, completion_tokens: 1, prompt_tokens_details: { cached_tokens: mock.warmCached } };
      res.write(`data: ${JSON.stringify({ choices: [], usage })}\n\n`);
      res.end('data: [DONE]\n\n');
      return;
    }
    mock.chats.push(body);
    mock.order.push('chat');
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
  assert.deepEqual(planTurnWarm({ promptTokens: 12200, cachedTokens: 10368, completionTokens: 20 }, BLOCK, 256), { warm: true, target: 12096 });
  // A prefill that crossed three boundaries: the target is the last one.
  assert.deepEqual(planTurnWarm({ promptTokens: 15700, cachedTokens: 10368, completionTokens: 5 }, BLOCK, 256), { warm: true, target: 15552 });
  // A cold prompt.
  assert.deepEqual(planTurnWarm({ promptTokens: 11213, cachedTokens: 0, completionTokens: 1 }, BLOCK, 256), { warm: true, target: 10368 });
});

test('planTurnWarm: skip cases', () => {
  // No boundary between the cached part and the end of the prompt.
  assert.deepEqual(planTurnWarm({ promptTokens: 12000, cachedTokens: 10368, completionTokens: 30 }, BLOCK, 256), { warm: false, reason: 'no_boundary' });
  // The decode crossed 12096, a later boundary that is reusable.
  assert.deepEqual(planTurnWarm({ promptTokens: 12000, cachedTokens: 8640, completionTokens: 200 }, BLOCK, 256), { warm: false, reason: 'decode_crossed', boundary: 12096 });
  // The last output token is not fed back, so it does not cross a boundary.
  assert.deepEqual(planTurnWarm({ promptTokens: 12095, cachedTokens: 8640, completionTokens: 1 }, BLOCK, 256), { warm: true, target: 10368 });
  assert.deepEqual(planTurnWarm({ promptTokens: 12094, cachedTokens: 8640, completionTokens: 2 }, BLOCK, 256), { warm: true, target: 10368 });
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

test('a real request of the same session aborts the warm request, and the warm is not sent again', async () => {
  const mock = await mockVllm();
  mock.usages = [{ prompt: 12200, cached: 10368, completion: 20 }, { prompt: 12300, cached: 10368, completion: 1 }];
  mock.holdWarm = true;
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => mock.completions.length === 1);
    mock.holdWarm = false;
    const evs = await send(base, codexRequest('s1'));
    assert.equal(evs.at(-1)!.type, 'response.completed');
    await until(() => mock.closedWarm === 1);
    // The second turn crossed 12096 again, so it gets its own warm.
    await until(() => gateway.turnWarmer.stats.completed === 1);
    await gateway.turnWarmer.idle();
    const s = gateway.turnWarmer.stats;
    assert.equal(s.aborted, 1);
    assert.equal(mock.completions.length, 2);
    assert.deepEqual(mock.order.slice(0, 5), ['chat', 'chat-end', 'tokenize', 'completions', 'chat']);
    // The second turn did not find the boundary that the aborted warm was for.
    assert.deepEqual(s.nextTurn.without_warm, { hit: 0, miss: 1 });
  } finally { await gateway.stop(); await mock.close(); }
});

test('a real request of another session aborts the warm request; the warm goes again when the gateway is idle', async () => {
  const mock = await mockVllm();
  mock.usages = [{ prompt: 12200, cached: 10368, completion: 20 }, { prompt: 12000, cached: 10368, completion: 20 }];
  mock.holdWarm = true;
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('s1'));
    await until(() => mock.completions.length === 1);
    mock.holdWarm = false;
    await send(base, codexRequest('s2'));
    await until(() => gateway.turnWarmer.stats.completed === 1);
    await gateway.turnWarmer.idle();
    assert.equal(mock.closedWarm, 1);
    assert.equal(gateway.turnWarmer.stats.aborted, 1);
    assert.equal(mock.completions.length, 2);
    assert.equal(mock.completions[1].prompt.length, 12096);
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
  }, []);
  assert.deepEqual(config.turnWarm, { ...defaultTurnWarmConfig(), enabled: true, blockTokens: 1680, minGainTokens: 512, timeoutMs: 5000 });
  assert.equal(loadConfig({}, []).turnWarm.enabled, false);
  assert.equal(loadConfig({ PULSE_GATEWAY_TURN_WARM: '0' }, []).turnWarm.enabled, false);
  assert.throws(() => loadConfig({ PULSE_GATEWAY_PREFIX_BLOCK_TOKENS: '0' }, []), /prefix block/);
});
