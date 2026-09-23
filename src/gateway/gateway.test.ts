// HTTP tests for the gateway against mock chat-completions backends.
// No GPU and no real model server is needed. Run: npm run test:gateway

import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { readFileSync, rmSync, statSync } from 'node:fs';
import type { AddressInfo } from 'node:net';
import { Gateway } from './server.js';
import { defaultConfig, loadConfig, parseEndpointList } from './config.js';
import type { GatewayConfig } from './config.js';
import { sseData } from './sse.js';
import { setLogLevel } from './log.js';
import type { Obj } from './translate.js';

setLogLevel('error');

type Handler = (body: Obj, req: http.IncomingMessage, res: http.ServerResponse) => void;

async function mockBackend(handler: Handler): Promise<{ url: string; requests: Obj[]; close: () => Promise<void> }> {
  const requests: Obj[] = [];
  const server = http.createServer(async (req, res) => {
    if (req.url === '/health') { res.writeHead(200); res.end(); return; }
    let raw = '';
    for await (const c of req) raw += c;
    const body = JSON.parse(raw);
    requests.push(body);
    handler(body, req, res);
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}/v1`,
    requests,
    close: () => new Promise((r) => { server.closeAllConnections(); server.close(() => r()); }),
  };
}

/** A port with nothing on it. */
async function deadPort(): Promise<string> {
  const server = http.createServer();
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address() as AddressInfo;
  await new Promise<void>((r) => server.close(() => r()));
  return `http://127.0.0.1:${port}/v1`;
}

function sse(res: http.ServerResponse, chunks: Obj[], done = true): void {
  res.writeHead(200, { 'Content-Type': 'text/event-stream' });
  for (const c of chunks) res.write(`data: ${JSON.stringify(c)}\n\n`);
  res.end(done ? 'data: [DONE]\n\n' : undefined);
}

const reply = (text: string) => [
  { choices: [{ index: 0, delta: { content: text }, finish_reason: null }] },
  { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
  { choices: [], usage: { prompt_tokens: 12, completion_tokens: 3, total_tokens: 15 } },
];

async function startGateway(endpoints: Array<{ name: string; baseUrl: string; enabled?: boolean }>, extra: Partial<GatewayConfig> = {}) {
  const config: GatewayConfig = {
    ...defaultConfig(),
    host: '127.0.0.1',
    port: 0,
    healthIntervalMs: 0,
    models: [{ id: 'qwen3.8-flash-next', profile: 'qwen38', endpoints }],
    ...extra,
  };
  const gateway = new Gateway(config);
  await gateway.start();
  const base = `http://127.0.0.1:${gateway.address!.port}`;
  return { gateway, base };
}

const post = (base: string, body: Obj) => fetch(`${base}/v1/responses`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

async function events(response: Response): Promise<Obj[]> {
  const out: Obj[] = [];
  for await (const data of sseData(response.body!)) out.push(JSON.parse(data));
  return out;
}

test('streams a Responses answer and maps the effort for Qwen3.8', async () => {
  const backend = await mockBackend((_b, _q, res) => sse(res, reply('hi there')));
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }]);
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'hello', stream: true, reasoning: { effort: 'low' } });
    assert.equal(res.status, 200);
    assert.equal(res.headers.get('x-pulse-backend'), 'spark1');
    const evs = await events(res);
    assert.equal(evs[0].type, 'response.created');
    assert.equal(evs.at(-1)!.type, 'response.completed');
    assert.equal(evs.at(-1)!.response.output[0].content[0].text, 'hi there');
    assert.equal(evs.at(-1)!.response.usage.total_tokens, 15);
    assert.deepEqual(backend.requests[0].chat_template_kwargs, { enable_thinking: false });
    assert.equal(backend.requests[0].model, 'qwen3.8-flash-next');
    const metrics = await (await fetch(`${base}/metrics`)).json() as Obj;
    assert.equal(metrics.requests.completed, 1);
    assert.equal(metrics.tokens.output, 3);
    assert.equal(metrics.backends[0].requests, 1);
    assert.equal(metrics.backends[0].latency.total.count, 1);
  } finally { await gateway.stop(); await backend.close(); }
});

test('non-streaming request returns one Responses object with a tool call', async () => {
  const backend = await mockBackend((_b, _q, res) => sse(res, [
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'apply_patch', arguments: '{"input":"*** Begin Patch"}' } }] }, finish_reason: null }] },
    { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
  ]));
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }]);
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'edit', tools: [{ type: 'custom', name: 'apply_patch' }] });
    assert.equal(res.status, 200);
    const body = await res.json() as Obj;
    assert.equal(body.status, 'completed');
    assert.equal(body.output[0].type, 'custom_tool_call');
    assert.equal(body.output[0].input, '*** Begin Patch');
  } finally { await gateway.stop(); await backend.close(); }
});

test('tool call arguments reach the client before the backend finishes', async () => {
  let finish!: () => void;
  const backend = await mockBackend((_b, _q, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'shell_command', arguments: '{"command":' } }] }, finish_reason: null }] })}\n\n`);
    finish = () => {
      res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"ls"}' } }] }, finish_reason: null }] })}\n\n`);
      res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] })}\n\n`);
      res.end('data: [DONE]\n\n');
    };
  });
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }]);
  try {
    const tools = [{ type: 'function', name: 'shell_command', parameters: { type: 'object', properties: {} } }];
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'ls', stream: true, tools });
    const seen: Obj[] = [];
    for await (const data of sseData(res.body!)) {
      const event = JSON.parse(data);
      seen.push(event);
      // The backend holds the rest of the call until the first delta arrives here.
      if (event.type === 'response.function_call_arguments.delta' && event.delta === '{"command":') finish();
    }
    assert.deepEqual(seen.filter((e) => e.type === 'response.function_call_arguments.delta').map((e) => e.delta), ['{"command":', '"ls"}']);
    const final = seen.at(-1)!;
    assert.equal(final.type, 'response.completed');
    assert.equal(final.response.output[0].arguments, '{"command":"ls"}');
  } finally { await gateway.stop(); await backend.close(); }
});

test('reasoning replay can be turned off, and the trace file gets the payload', async () => {
  const backend = await mockBackend((_b, _q, res) => sse(res, reply('ok')));
  const trace = `${process.env.TMPDIR ?? '/tmp'}/pulse-gateway-trace-${process.pid}.jsonl`;
  const input = [
    { type: 'message', role: 'user', content: 'q' },
    { type: 'reasoning', summary: [{ type: 'summary_text', text: 'plan' }] },
    { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'a' }] },
    { type: 'message', role: 'user', content: 'q2' },
  ];
  for (const replayReasoning of [true, false]) {
    const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { replayReasoning, traceFile: trace });
    try {
      await events(await post(base, { model: 'qwen3.8-flash-next', input, stream: true }));
    } finally { await gateway.stop(); }
  }
  try {
    assert.equal(backend.requests[0].messages[1].reasoning_content, 'plan');
    assert.equal(backend.requests[1].messages[1].reasoning_content, undefined);
    const lines = readFileSync(trace, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
    assert.equal(lines.length, 2);
    assert.equal(lines[0].payload.messages[1].reasoning_content, 'plan');
    assert.equal(lines[0].request.input.length, 4);
    assert.equal(statSync(trace).mode & 0o777, 0o600);
  } finally { rmSync(trace, { force: true }); await backend.close(); }
});

test('unsupported input returns 400 before any backend call', async () => {
  const backend = await mockBackend((_b, _q, res) => sse(res, reply('no')));
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }]);
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', store: true, stream: true });
    assert.equal(res.status, 400);
    assert.match((await res.json() as Obj).error.message, /store=true/);
    assert.equal(backend.requests.length, 0);
  } finally { await gateway.stop(); await backend.close(); }
});

test('codex exec --output-schema: a json_schema text.format reaches the backend as an instruction', async () => {
  const backend = await mockBackend((_b, _q, res) => sse(res, reply('{"answer": 5}')));
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }]);
  try {
    const schema = { type: 'object', properties: { answer: { type: 'integer' } }, required: ['answer'] };
    const res = await post(base, {
      model: 'qwen3.8-flash-next', input: 'What is 2+3?', stream: true, store: false,
      text: { verbosity: 'low', format: { type: 'json_schema', name: 'codex_output_schema', strict: true, schema } },
    });
    assert.equal(res.status, 200);
    assert.equal((await events(res)).at(-1)!.type, 'response.completed');
    assert.equal(backend.requests[0].messages[0].role, 'system');
    assert(backend.requests[0].messages[0].content.endsWith(JSON.stringify(schema)));
    assert.equal(backend.requests[0].response_format, undefined);
  } finally { await gateway.stop(); await backend.close(); }
});

test('fails over from a dead endpoint and from a 503 endpoint, in order', async () => {
  const busy = await mockBackend((_b, _q, res) => { res.writeHead(503); res.end('loading'); });
  const good = await mockBackend((_b, _q, res) => sse(res, reply('from spark3')));
  const dead = await deadPort();
  const { gateway, base } = await startGateway([
    { name: 'spark1', baseUrl: dead },
    { name: 'spark2', baseUrl: busy.url },
    { name: 'spark3', baseUrl: good.url },
    { name: 'off', baseUrl: good.url, enabled: false },
  ]);
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    assert.equal(res.headers.get('x-pulse-backend'), 'spark3');
    const evs = await events(res);
    assert.equal(evs.at(-1)!.response.output[0].content[0].text, 'from spark3');
    const metrics = await (await fetch(`${base}/metrics`)).json() as Obj;
    const by = Object.fromEntries(metrics.backends.map((b: Obj) => [b.name, b]));
    // The start-up health check can mark spark1 down before the request. Then
    // the request skips it. Either way spark1 ends unhealthy and spark3 serves.
    assert.equal(by.spark1.healthy, false);
    assert.equal(by.spark2.failovers, 1);
    assert.equal(by.spark3.requests, 1);
    assert.equal(by.off.requests, 0);
    assert.equal(good.requests.length, 1);
  } finally { await gateway.stop(); await busy.close(); await good.close(); }
});

test('a health check moves an unhealthy endpoint to the end of the order', async () => {
  const dead = await deadPort();
  const good = await mockBackend((_b, _q, res) => sse(res, reply('ok')));
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: dead }, { name: 'spark2', baseUrl: good.url }]);
  try {
    await gateway.checkHealth();
    const order = gateway.routes[0].attemptOrder().map((e) => e.name);
    assert.deepEqual(order, ['spark2', 'spark1']);
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    assert.equal(res.headers.get('x-pulse-backend'), 'spark2');
    await events(res);
    const metrics = await (await fetch(`${base}/metrics`)).json() as Obj;
    assert.equal(metrics.backends[0].requests, 0);
    const health = await (await fetch(`${base}/health`)).json() as Obj;
    assert.equal(health.status, 'ok');
  } finally { await gateway.stop(); await good.close(); }
});

test('returns 503 when no endpoint can serve, and /health reports it', async () => {
  const dead = await deadPort();
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: dead }]);
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    assert.equal(res.status, 503);
    const body = await res.json() as Obj;
    assert.equal(body.error.type, 'backend_unavailable');
    const health = await fetch(`${base}/health`);
    assert.equal(health.status, 503);
    const h = await health.json() as Obj;
    assert.equal(h.models[0].backends[0].healthy, false);
  } finally { await gateway.stop(); }
});

test('passes a backend 400 through without failover', async () => {
  const bad = await mockBackend((_b, _q, res) => { res.writeHead(400, { 'Content-Type': 'application/json' }); res.end('{"error":{"message":"prompt too long"}}'); });
  const other = await mockBackend((_b, _q, res) => sse(res, reply('no')));
  const { gateway, base } = await startGateway([{ name: 'a', baseUrl: bad.url }, { name: 'b', baseUrl: other.url }]);
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    assert.equal(res.status, 400);
    assert.match((await res.json() as Obj).error.message, /prompt too long/);
    assert.equal(other.requests.length, 0);
  } finally { await gateway.stop(); await bad.close(); await other.close(); }
});

test('rejects unknown models and bad JSON with 400', async () => {
  const { gateway, base } = await startGateway([{ name: 'a', baseUrl: 'http://127.0.0.1:9/v1' }]);
  try {
    assert.equal((await post(base, { model: 'gpt-5', input: 'x' })).status, 400);
    const bad = await fetch(`${base}/v1/responses`, { method: 'POST', body: '{nope' });
    assert.equal(bad.status, 400);
    const models = await (await fetch(`${base}/v1/models`)).json() as Obj;
    assert.equal(models.data[0].id, 'qwen3.8-flash-next');
  } finally { await gateway.stop(); }
});

test('a stream cut before the finish reason ends with response.failed', async () => {
  const backend = await mockBackend((_b, _q, res) => sse(res, [{ choices: [{ index: 0, delta: { content: 'part' }, finish_reason: null }] }], false));
  const { gateway, base } = await startGateway([{ name: 'a', baseUrl: backend.url }]);
  try {
    const evs = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true }));
    assert.equal(evs.at(-1)!.type, 'response.failed');
    assert(!evs.some((e) => e.type === 'response.completed'));
    const metrics = await (await fetch(`${base}/metrics`)).json() as Obj;
    assert.equal(metrics.requests.failed, 1);
  } finally { await gateway.stop(); await backend.close(); }
});

test('backend idle timeout fails the stream', async () => {
  const backend = await mockBackend((_b, _q, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'a' }, finish_reason: null }] })}\n\n`);
    // then nothing
  });
  const { gateway, base } = await startGateway([{ name: 'a', baseUrl: backend.url }], { idleTimeoutMs: 200 });
  try {
    const evs = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true }));
    assert.equal(evs.at(-1)!.type, 'response.failed');
    assert.match(evs.at(-1)!.response.error.message, /idle/);
  } finally { await gateway.stop(); await backend.close(); }
});

test('client disconnect cancels the backend request', async () => {
  let backendClosed!: () => void;
  const closed = new Promise<void>((r) => { backendClosed = r; });
  const backend = await mockBackend((_b, _q, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'a' }, finish_reason: null }] })}\n\n`);
    res.on('close', backendClosed);
  });
  const { gateway, base } = await startGateway([{ name: 'a', baseUrl: backend.url }]);
  try {
    const abort = new AbortController();
    const res = await fetch(`${base}/v1/responses`, {
      method: 'POST', body: JSON.stringify({ model: 'qwen3.8-flash-next', input: 'x', stream: true }), signal: abort.signal,
    });
    const reader = res.body!.getReader();
    await reader.read();
    abort.abort();
    await closed;
    await new Promise((r) => setTimeout(r, 50));
    const metrics = await (await fetch(`${base}/metrics`)).json() as Obj;
    assert.equal(metrics.requests.cancelled, 1);
    assert.equal(metrics.requests.in_flight, 0);
  } finally { await gateway.stop(); await backend.close(); }
});

test('graceful shutdown lets an in-flight stream finish', async () => {
  const backend = await mockBackend((_b, _q, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    setTimeout(() => { for (const c of reply('late')) res.write(`data: ${JSON.stringify(c)}\n\n`); res.end('data: [DONE]\n\n'); }, 200);
  });
  const { gateway, base } = await startGateway([{ name: 'a', baseUrl: backend.url }], { shutdownGraceMs: 5_000 });
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    const stopped = gateway.stop();
    const evs = await events(res);
    await stopped;
    assert.equal(evs.at(-1)!.type, 'response.completed');
  } finally { await backend.close(); }
});

test('SSE parser handles split UTF-8, CRLF, comments and multi-line data', async () => {
  const bytes = new TextEncoder().encode(': heartbeat\r\ndata: hé\r\ndata: llo\r\n\r\ndata: [DONE]\n\n');
  const body = new ReadableStream<Uint8Array>({ start(c) { for (const b of bytes) c.enqueue(Uint8Array.of(b)); c.close(); } });
  const out: string[] = [];
  for await (const d of sseData(body)) out.push(d);
  assert.deepEqual(out, ['hé\nllo', '[DONE]']);
});

test('config: env endpoint list sets the failover order', () => {
  assert.deepEqual(parseEndpointList('spark1=http://127.0.0.1:8888/v1, spark2=http://10.99.0.2:8888/v1'), [
    { name: 'spark1', baseUrl: 'http://127.0.0.1:8888/v1' },
    { name: 'spark2', baseUrl: 'http://10.99.0.2:8888/v1' },
  ]);
  const config = loadConfig({ PULSE_QWEN_BACKENDS: 'http://a:1/v1/,http://b:2/v1', PULSE_GATEWAY_PORT: '18800' }, []);
  assert.equal(config.port, 18800);
  assert.deepEqual(config.models[0].endpoints.map((e) => e.baseUrl), ['http://a:1/v1', 'http://b:2/v1']);
  assert.throws(() => loadConfig({ PULSE_GATEWAY_PORT: 'abc' }, []));
  const defaults = loadConfig({}, []);
  assert.equal(defaults.models[0].id, 'qwen3.8-flash-next');
  assert.equal(defaults.models[0].endpoints[0].baseUrl, 'http://127.0.0.1:8888/v1');
});
