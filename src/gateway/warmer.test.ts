// Tests for the prefix warmer against a mock chat-completions backend.
// Run: npm run test:gateway

import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, readdirSync, rmSync, statSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { AddressInfo } from 'node:net';
import { Gateway } from './server.js';
import { defaultConfig } from './config.js';
import type { GatewayConfig } from './config.js';
import { sseData } from './sse.js';
import { setLogLevel } from './log.js';
import type { Obj } from './translate.js';
import { defaultWarmupConfig, prefixMessages, variantKey, warmPayload } from './warmer.js';

setLogLevel('error');

interface Mock {
  url: string;
  requests: Obj[];
  /** /health answers 200 when true and 503 when false. */
  healthy: boolean;
  /** Keep requests with max_tokens 1 open until the client closes them. */
  holdWarm: boolean;
  /** Warm requests that the client closed before an answer. */
  closedWarm: number;
  close: () => Promise<void>;
}

async function mockBackend(): Promise<Mock> {
  const mock = { requests: [] as Obj[], healthy: true, holdWarm: false, closedWarm: 0 } as Mock;
  const server = http.createServer(async (req, res) => {
    if (req.url === '/health') { res.writeHead(mock.healthy ? 200 : 503); res.end(); return; }
    let raw = '';
    for await (const c of req) raw += c;
    const body = JSON.parse(raw);
    mock.requests.push(body);
    if (body.max_tokens === 1 && mock.holdWarm) {
      res.on('close', () => { if (!res.writableEnded) mock.closedWarm++; });
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      return;
    }
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const chunks = [
      { choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      { choices: [], usage: { prompt_tokens: 100, completion_tokens: 1, total_tokens: 101, prompt_tokens_details: { cached_tokens: 64 } } },
    ];
    for (const c of chunks) res.write(`data: ${JSON.stringify(c)}\n\n`);
    res.end('data: [DONE]\n\n');
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address() as AddressInfo;
  mock.url = `http://127.0.0.1:${port}/v1`;
  mock.close = () => new Promise((r) => { server.closeAllConnections(); server.close(() => r()); });
  return mock;
}

async function startGateway(mock: Mock, warmup: Partial<GatewayConfig['warmup']> = {}) {
  const config: GatewayConfig = {
    ...defaultConfig(),
    host: '127.0.0.1',
    port: 0,
    healthIntervalMs: 0,
    models: [{ id: 'qwen3.8-flash-next', profile: 'qwen38', endpoints: [{ name: 'spark1', baseUrl: mock.url }] }],
    warmup: { ...defaultWarmupConfig(), ...warmup },
  };
  const gateway = new Gateway(config);
  await gateway.start();
  return { gateway, base: `http://127.0.0.1:${gateway.address!.port}` };
}

/** A request in the shape that Codex sends for the first turn of a session. */
const codexRequest = (task: string, session?: string): Obj => ({
  model: 'qwen3.8-flash-next',
  instructions: 'You are Codex.',
  input: [
    { type: 'message', role: 'developer', content: [{ type: 'input_text', text: 'Rules.' }] },
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: '<environment_context>cwd</environment_context>' }] },
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: task }] },
  ],
  tools: [{ type: 'function', name: 'exec_command', description: 'Run a command.', parameters: { type: 'object', properties: {} } }],
  reasoning: { effort: 'low' },
  stream: true,
  store: false,
  ...(session ? { prompt_cache_key: session } : {}),
});

async function send(base: string, body: Obj): Promise<Obj[]> {
  const res = await fetch(`${base}/v1/responses`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
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

/** Mark the backend down and then up again, as after a restart. */
async function restart(gateway: Gateway, mock: Mock): Promise<void> {
  mock.healthy = false;
  await gateway.checkHealth();
  assert.equal(gateway.routes[0].endpoints[0].healthy, false);
  mock.healthy = true;
  await gateway.checkHealth();
}

test('prefixMessages keeps the system message and the environment message', () => {
  const sys = { role: 'system', content: 's' };
  const env = { role: 'user', content: 'env' };
  const task = { role: 'user', content: 'task' };
  assert.deepEqual(prefixMessages([sys, env, task]), [sys, env]);
  assert.deepEqual(prefixMessages([sys, task]), [sys]);
  assert.deepEqual(prefixMessages([sys, task, { role: 'assistant', content: 'a' }]), [sys]);
  assert.equal(prefixMessages([task]), null);
  assert.equal(prefixMessages(undefined), null);
});

test('warmPayload asks for one token and keeps the prompt fields', () => {
  const payload = { model: 'm', messages: [{ role: 'user', content: 'x' }], tools: [{ type: 'function' }], max_tokens: 900, stream: false, chat_template_kwargs: { enable_thinking: false } };
  const warm = warmPayload(payload);
  assert.equal(warm.max_tokens, 1);
  assert.equal(warm.stream, true);
  assert.deepEqual(warm.stream_options, { include_usage: true });
  assert.deepEqual(warm.messages, payload.messages);
  assert.deepEqual(warm.tools, payload.tools);
  assert.deepEqual(warm.chat_template_kwargs, payload.chat_template_kwargs);
  assert.notEqual(variantKey({ chat_template_kwargs: { enable_thinking: false } }), variantKey({ reasoning_effort: 'high' }));
  // The JSON schema of a forced tool_choice does not change the prefill.
  assert.equal(warmPayload({ ...payload, response_format: { type: 'json_schema' } }).response_format, undefined);
});

test('after a backend restart the warmer sends the prefix and the last session, each twice', async () => {
  const mock = await mockBackend();
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('first task', 'session-1'));
    const real = mock.requests[0];
    await restart(gateway, mock);
    await until(() => mock.requests.length === 5);
    const warm = mock.requests.slice(1);
    for (const w of warm) assert.equal(w.max_tokens, 1);
    // Prefix: system and environment messages, same tools and template arguments.
    assert.deepEqual(warm[0].messages, real.messages.slice(0, 2));
    assert.deepEqual(warm[0].tools, real.tools);
    assert.deepEqual(warm[0].chat_template_kwargs, real.chat_template_kwargs);
    assert.deepEqual(warm[1].messages, warm[0].messages);
    // Session: the full last payload.
    assert.deepEqual(warm[2].messages, real.messages);
    assert.deepEqual(warm[3].messages, real.messages);
    await until(() => gateway.warmer.stats.completed === 4);
    const metrics = await (await fetch(`${base}/metrics`)).json() as Obj;
    assert.equal(metrics.warmup.completed, 4);
    assert.equal(metrics.warmup.prompt_tokens, 400);
    assert.equal(metrics.warmup.cached_tokens, 256);
    assert.equal(metrics.warmup.last.kind, 'session');
  } finally { await gateway.stop(); await mock.close(); }
});

test('the first health check does not warm when no prefix is known', async () => {
  const mock = await mockBackend();
  const { gateway } = await startGateway(mock);
  try {
    await gateway.checkHealth();
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(mock.requests.length, 0);
    assert.equal(gateway.warmer.stats.triggers, 0);
  } finally { await gateway.stop(); await mock.close(); }
});

test('a real request aborts the warm request in flight, and the warmer skips what it covers', async () => {
  const mock = await mockBackend();
  const { gateway, base } = await startGateway(mock, { sessions: 0 });
  try {
    await send(base, codexRequest('first task'));
    mock.holdWarm = true;
    const route = gateway.routes[0];
    const warming = gateway.warmer.warm(route, route.endpoints[0], 'test');
    await until(() => mock.requests.length === 2);
    const evs = await send(base, codexRequest('second task'));
    assert.equal(evs.at(-1)!.type, 'response.completed');
    await warming;
    await until(() => mock.closedWarm === 1);
    assert.equal(gateway.warmer.stats.aborted, 1);
    assert.equal(gateway.warmer.stats.skipped, 1);
    // The real request came after the warm request; no warm request followed.
    assert.equal(mock.requests.length, 3);
    assert.notEqual(mock.requests[2].max_tokens, 1);
  } finally { await gateway.stop(); await mock.close(); }
});

test('the real request that finds the backend healthy again is not warmed again', async () => {
  const mock = await mockBackend();
  const { gateway, base } = await startGateway(mock);
  try {
    await send(base, codexRequest('task', 'session-1'));
    mock.healthy = false;
    await gateway.checkHealth();
    mock.healthy = true;
    // No health check ran, so this request marks the endpoint healthy.
    const evs = await send(base, codexRequest('task', 'session-1'));
    assert.equal(evs.at(-1)!.type, 'response.completed');
    await until(() => gateway.warmer.stats.skipped === 2);
    assert.equal(gateway.warmer.stats.triggers, 1);
    assert.equal(gateway.warmer.stats.requests, 0);
    assert.equal(mock.requests.length, 2);
  } finally { await gateway.stop(); await mock.close(); }
});

test('a disabled warmer sends nothing after a restart', async () => {
  const mock = await mockBackend();
  const { gateway, base } = await startGateway(mock, { enabled: false });
  try {
    await send(base, codexRequest('task', 'session-1'));
    await restart(gateway, mock);
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(mock.requests.length, 1);
    assert.equal((await (await fetch(`${base}/metrics`)).json() as Obj).warmup.enabled, false);
  } finally { await gateway.stop(); await mock.close(); }
});

test('sessions older than the age limit are not warmed, and confirm can be off', async () => {
  const mock = await mockBackend();
  const { gateway, base } = await startGateway(mock, { sessionMaxAgeMs: 0, confirm: false });
  try {
    await send(base, codexRequest('task', 'session-1'));
    await new Promise((r) => setTimeout(r, 5));
    await restart(gateway, mock);
    await until(() => gateway.warmer.stats.completed === 1);
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(mock.requests.length, 2);
    assert.equal(mock.requests[1].messages.length, 2);
  } finally { await gateway.stop(); await mock.close(); }
});

test('the state file keeps the prefix for a new gateway process, with mode 0600', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'pulse-warm-'));
  const stateFile = join(dir, 'warm.json');
  const mock = await mockBackend();
  try {
    const first = await startGateway(mock, { stateFile, sessions: 0 });
    try {
      await send(first.base, codexRequest('task'));
      await until(() => { try { return statSync(stateFile).size > 0; } catch { return false; } }, 5000);
      assert.equal(statSync(stateFile).mode & 0o777, 0o600);
      assert.equal(JSON.parse(readFileSync(stateFile, 'utf8')).prefixes.length, 1);
    } finally { await first.gateway.stop(); }
    const real = mock.requests[0];
    mock.requests.length = 0;
    const second = await startGateway(mock, { stateFile, sessions: 0 });
    try {
      // The first health check of the new process warms the stored prefix.
      await until(() => second.gateway.warmer.stats.completed === 2);
      assert.deepEqual(mock.requests[0].messages, real.messages.slice(0, 2));
      assert.equal(mock.requests[0].max_tokens, 1);
    } finally { await second.gateway.stop(); }
  } finally { await mock.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('stop writes a state file that is due, and leaves no temporary file', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'pulse-warm-'));
  const stateFile = join(dir, 'warm.json');
  const mock = await mockBackend();
  try {
    const { gateway, base } = await startGateway(mock, { stateFile, sessions: 0 });
    await send(base, codexRequest('task'));
    // The write is due in 2 s. The stop writes it now.
    await gateway.stop();
    assert.equal(statSync(stateFile).mode & 0o777, 0o600);
    assert.equal(JSON.parse(readFileSync(stateFile, 'utf8')).prefixes.length, 1);
    assert.deepEqual(readdirSync(dir), ['warm.json']);
  } finally { await mock.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('with two endpoints, a real request aborts both warm requests and skips only its own endpoint', async () => {
  const a = await mockBackend();
  const b = await mockBackend();
  const config: GatewayConfig = {
    ...defaultConfig(),
    host: '127.0.0.1',
    port: 0,
    healthIntervalMs: 0,
    models: [{
      id: 'qwen3.8-flash-next', profile: 'qwen38',
      endpoints: [{ name: 'spark1', baseUrl: a.url }, { name: 'spark2', baseUrl: b.url }],
    }],
    warmup: { ...defaultWarmupConfig(), sessions: 0 },
  };
  const gateway = new Gateway(config);
  await gateway.start();
  const base = `http://127.0.0.1:${gateway.address!.port}`;
  try {
    await send(base, codexRequest('first task'));
    assert.equal(a.requests.length, 1);
    a.holdWarm = true;
    b.holdWarm = true;
    const route = gateway.routes[0];
    const warming = Promise.all(route.endpoints.map((e) => gateway.warmer.warm(route, e, 'test')));
    await until(() => a.requests.length === 2 && b.requests.length === 1);
    // The held requests stay open. The next warm requests to spark2 get an answer.
    b.holdWarm = false;
    const evs = await send(base, codexRequest('second task'));
    assert.equal(evs.at(-1)!.type, 'response.completed');
    await warming;
    await until(() => a.closedWarm === 1 && b.closedWarm === 1);
    assert.equal(gateway.warmer.stats.aborted, 2);
    // spark1 got the real request, so its cache has the prefix.
    assert.equal(gateway.warmer.stats.skipped, 1);
    assert.equal(a.requests.length, 3);
    assert.notEqual(a.requests[2].max_tokens, 1);
    // spark2 did not get the real request, so the warmer warmed it again.
    assert.equal(b.requests.length, 3);
    for (const r of b.requests) assert.equal(r.max_tokens, 1);
    assert.equal(gateway.warmer.stats.completed, 2);
  } finally { await gateway.stop(); await a.close(); await b.close(); }
});
