// HTTP tests for the gateway against mock chat-completions backends.
// No GPU and no real model server is needed. Run: npm run test:gateway

import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { chmodSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import type { AddressInfo } from 'node:net';
import { Gateway, privateAppendStream } from './server.js';
import { defaultConfig, loadConfig, parseEndpointList } from './config.js';
import type { GatewayConfig } from './config.js';
import { sseData } from './sse.js';
import { setLogLevel, setLogSink } from './log.js';
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

test('the trace file is made private also when it already exists', async () => {
  const backend = await mockBackend((_b, _q, res) => sse(res, reply('ok')));
  const trace = `${process.env.TMPDIR ?? '/tmp'}/pulse-gateway-trace-mode-${process.pid}.jsonl`;
  writeFileSync(trace, '', { mode: 0o644 });
  chmodSync(trace, 0o644);
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { traceFile: trace });
  try {
    await events(await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true }));
    await gateway.stop();
    assert.equal(statSync(trace).mode & 0o777, 0o600);
    assert.equal(readFileSync(trace, 'utf8').trim().split('\n').length, 1);
  } finally { rmSync(trace, { force: true }); await backend.close(); }
});

test('the private append stream sets the mode before the first write', async () => {
  const file = `${process.env.TMPDIR ?? '/tmp'}/pulse-gateway-private-${process.pid}.jsonl`;
  writeFileSync(file, 'old\n', { mode: 0o644 });
  chmodSync(file, 0o644);
  try {
    const stream = privateAppendStream(file);
    // The stream writes only after the open event, so the mode at that event
    // is the mode of the first row.
    const modeAtOpen = await new Promise<number>((resolve, reject) => {
      stream.once('open', () => resolve(statSync(file).mode & 0o777));
      stream.once('error', reject);
    });
    await new Promise<void>((resolve) => stream.end('new\n', () => resolve()));
    assert.equal(modeAtOpen, 0o600);
    assert.equal(readFileSync(file, 'utf8'), 'old\nnew\n');
  } finally { rmSync(file, { force: true }); }
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

test('rejected requests and backend errors get a log line with the reason', async () => {
  const lines: Obj[] = [];
  setLogLevel('info');
  setLogSink((_level, line) => lines.push(JSON.parse(line)));
  const trace = `${process.env.TMPDIR ?? '/tmp'}/pulse-gateway-reject-${process.pid}.jsonl`;
  const bad = await mockBackend((body, _q, res) => {
    if (body.messages.at(-1).content === 'cut') {
      sse(res, [
        { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'shell_command', arguments: '{"command": "ls' } }] }, finish_reason: null }] },
        { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
      ]);
      return;
    }
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end('{"error":{"message":"Unterminated string starting at: line 1 column 9"}}');
  });
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: bad.url }], { traceFile: trace });
  try {
    const tools = [{ type: 'function', name: 'shell_command', parameters: { type: 'object', properties: {} } }];
    assert.equal((await post(base, { model: 'qwen3.8-flash-next', input: 'x', store: true, stream: true })).status, 400);
    assert.equal((await post(base, { model: 'gpt-5', input: 'x' })).status, 400);
    assert.equal((await fetch(`${base}/v1/responses`, { method: 'POST', body: '{nope' })).status, 400);
    assert.equal((await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true })).status, 400);
    await events(await post(base, { model: 'qwen3.8-flash-next', input: 'cut', stream: true, tools }));
    const rejected = lines.filter((l) => l.msg === 'request rejected');
    assert.deepEqual(rejected.map((l) => [l.status, l.error]), [
      [400, 'store=true is not supported; the gateway keeps no state, so set store=false'],
      [400, 'unknown model: gpt-5; this gateway serves qwen3.8-flash-next'],
      [400, 'invalid JSON'],
    ]);
    for (const l of rejected) assert.match(l.request_id, /^req_/);
    const responses = lines.filter((l) => l.msg === 'response');
    assert.equal(responses[0].status, 'client_error');
    assert.match(responses[0].error, /Unterminated string/);
    const warned = lines.find((l) => l.msg === 'tool call arguments are not a JSON object');
    assert.deepEqual([warned?.name, warned?.arguments], ['shell_command', '{"command": "ls']);
    // The trace file keeps the rejected request with its reason.
    await gateway.flushTrace();
    const rows = readFileSync(trace, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
    assert.equal(rows[0].request_id, rejected[0].request_id);
    assert.equal(rows[0].request.store, true);
    assert.match(rows[0].error, /store=true/);
    assert.equal(rows[0].payload, undefined);
  } finally {
    setLogSink(null);
    setLogLevel('error');
    rmSync(trace, { force: true });
    await gateway.stop();
    await bad.close();
  }
});

test('backend requests on a keep-alive socket do not pile up timeout listeners', async () => {
  const warnings: Error[] = [];
  const onWarning = (warning: Error) => { if (warning.name === 'MaxListenersExceededWarning') warnings.push(warning); };
  process.on('warning', onWarning);
  const sockets = new Set<unknown>();
  const backend = await mockBackend((body, req, res) => {
    sockets.add(req.socket);
    if (body.messages.at(-1).content === 'stall') {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'a' }, finish_reason: null }] })}\n\n`);
      return;
    }
    sse(res, reply('ok'));
  });
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { idleTimeoutMs: 300 });
  try {
    for (let i = 0; i < 25; i++) {
      const evs = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true }));
      assert.equal(evs.at(-1)!.type, 'response.completed');
    }
    // The idle timeout still works on a socket that served many requests.
    const stalled = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'stall', stream: true }));
    assert.equal(stalled.at(-1)!.type, 'response.failed');
    assert.match(stalled.at(-1)!.response.error.message, /idle/);
    await new Promise((r) => setImmediate(r));
    assert(sockets.size <= 2, `sockets: ${sockets.size}`);
    assert.deepEqual(warnings.map((w) => w.message), []);
  } finally {
    process.off('warning', onWarning);
    await gateway.stop();
    await backend.close();
  }
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
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: dead }], { retryWindowMs: 0 });
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

// ------------------------------------------------------------ reliability

const chunkFrame = (c: Obj) => `data: ${JSON.stringify(c)}\n\n`;

test('keepalive: a long prefill gets response.in_progress events, not only SSE comments', async () => {
  const backend = await mockBackend((_b, _q, res) => {
    // vLLM sends the headers at once and no byte during the prefill.
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.flushHeaders();
    setTimeout(() => { for (const c of reply('after prefill')) res.write(chunkFrame(c)); res.end('data: [DONE]\n\n'); }, 450);
  });
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { keepaliveMs: 100 });
  try {
    const evs = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true }));
    const progress = evs.filter((e) => e.type === 'response.in_progress');
    // One from the start, and at least two keepalives during the 450 ms wait.
    assert(progress.length >= 3, `in_progress events: ${progress.length}`);
    assert.equal(evs.at(-1)!.type, 'response.completed');
    evs.forEach((e, i) => assert.equal(e.sequence_number, i));
    assert.equal(new Set(evs.map((e) => e.response?.id).filter(Boolean)).size, 1);
  } finally { await gateway.stop(); await backend.close(); }
});

test('keepalive: the stream starts before slow backend headers arrive', async () => {
  const backend = await mockBackend((_b, _q, res) => { setTimeout(() => sse(res, reply('slow')), 500); });
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { keepaliveMs: 100, streamStartMs: 50 });
  try {
    const t0 = performance.now();
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    assert(performance.now() - t0 < 400, 'the headers must not wait for the backend');
    assert.equal(res.status, 200);
    const evs = await events(res);
    assert.equal(evs[0].type, 'response.created');
    assert(evs.filter((e) => e.type === 'response.in_progress').length >= 3);
    assert.equal(evs.at(-1)!.response.output[0].content[0].text, 'slow');
  } finally { await gateway.stop(); await backend.close(); }
});

test('retry: a backend that comes back within the retry window serves the request', async () => {
  const dead = await deadPort();
  const port = Number(new URL(dead).port);
  let late: http.Server | null = null;
  const lateStart = setTimeout(() => {
    late = http.createServer(async (req, res) => {
      if (req.url === '/health') { res.writeHead(200); res.end(); return; }
      for await (const _ of req) { /* drain */ }
      sse(res, reply('back again'));
    });
    late.listen(port, '127.0.0.1');
  }, 700);
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: dead }], { retryWindowMs: 10_000, streamStartMs: 100, keepaliveMs: 100 });
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    assert.equal(res.status, 200);
    const evs = await events(res);
    assert.equal(evs.at(-1)!.type, 'response.completed');
    assert.equal(evs.at(-1)!.response.output[0].content[0].text, 'back again');
    assert.equal(evs.filter((e) => e.type === 'response.created').length, 1);
    const metrics = await (await fetch(`${base}/metrics`)).json() as Obj;
    assert(metrics.requests.retries >= 1);
  } finally {
    clearTimeout(lateStart);
    await gateway.stop();
    const server = late as http.Server | null;
    if (server) await new Promise<void>((r) => { server.closeAllConnections(); server.close(() => r()); });
  }
});

test('retry: the last attempt comes at the end of the full retry window', async () => {
  // The waits are 250 ms and 500 ms, then the rest of the window. A backend
  // that comes back after 850 ms of a 1000 ms window still gets the request.
  const dead = await deadPort();
  const port = Number(new URL(dead).port);
  let late: http.Server | null = null;
  const lateStart = setTimeout(() => {
    late = http.createServer(async (req, res) => {
      if (req.url === '/health') { res.writeHead(200); res.end(); return; }
      for await (const _ of req) { /* drain */ }
      sse(res, reply('just in time'));
    });
    late.listen(port, '127.0.0.1');
  }, 850);
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: dead }], { retryWindowMs: 1_000 });
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x' });
    assert.equal(res.status, 200);
    assert.equal((await res.json() as Obj).output[0].content[0].text, 'just in time');
  } finally {
    clearTimeout(lateStart);
    await gateway.stop();
    const server = late as http.Server | null;
    if (server) await new Promise<void>((r) => { server.closeAllConnections(); server.close(() => r()); });
  }
});

test('retry: a backend that stays down gets attempts until the window ends', async () => {
  const dead = await deadPort();
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: dead }], { retryWindowMs: 1_000 });
  try {
    const started = Date.now();
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x' });
    const elapsed = Date.now() - started;
    assert.equal(res.status, 503);
    assert(elapsed >= 950, `the gateway gave up after ${elapsed} ms of a 1000 ms window`);
    assert(elapsed < 2_000, `the gateway waited ${elapsed} ms for a 1000 ms window`);
  } finally { await gateway.stop(); }
});

test('retry: a stream that breaks before output is sent again until the end of the window', async () => {
  const started = Date.now();
  const backend = await mockBackend((_b, _q, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    if (Date.now() - started < 850) { res.destroy(); return; }
    for (const c of reply('after the restart')) res.write(chunkFrame(c));
    res.end('data: [DONE]\n\n');
  });
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { retryWindowMs: 1_000 });
  try {
    const evs = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true }));
    assert.equal(evs.at(-1)!.type, 'response.completed');
    assert.equal(evs.at(-1)!.response.output[0].content[0].text, 'after the restart');
    assert(backend.requests.length >= 4);
  } finally { await gateway.stop(); await backend.close(); }
});

test('retry: 503 while the backend loads, then 200, is one completed response', async () => {
  let calls = 0;
  const backend = await mockBackend((_b, _q, res) => {
    if (++calls <= 2) { res.writeHead(503); res.end('loading'); return; }
    sse(res, reply('loaded'));
  });
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { retryWindowMs: 10_000 });
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x' });
    assert.equal(res.status, 200);
    assert.equal((await res.json() as Obj).output[0].content[0].text, 'loaded');
    assert.equal(backend.requests.length, 3);
  } finally { await gateway.stop(); await backend.close(); }
});

test('retry: a stream that breaks before any output is sent again; after output it fails', async () => {
  let calls = 0;
  const backend = await mockBackend((body, _q, res) => {
    calls++;
    const cut = body.messages.at(-1).content === 'cut after output';
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    if (cut) {
      res.write(chunkFrame({ choices: [{ index: 0, delta: { content: 'half' }, finish_reason: null }] }));
      setTimeout(() => res.destroy(), 20);
      return;
    }
    if (calls === 1) {
      // A role chunk and usage are not output. Then the backend restarts.
      res.write(chunkFrame({ choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
      setTimeout(() => res.destroy(), 20);
      return;
    }
    for (const c of reply('second try')) res.write(chunkFrame(c));
    res.end('data: [DONE]\n\n');
  });
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { retryWindowMs: 10_000 });
  try {
    const evs = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true }));
    assert.equal(evs.at(-1)!.type, 'response.completed');
    assert.equal(evs.at(-1)!.response.output.length, 1);
    assert.equal(evs.at(-1)!.response.output[0].content[0].text, 'second try');
    assert.equal(evs.filter((e) => e.type === 'response.created').length, 1);
    evs.forEach((e, i) => assert.equal(e.sequence_number, i));
    assert.equal(backend.requests.length, 2);

    const cut = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'cut after output', stream: true }));
    assert.equal(cut.at(-1)!.type, 'response.failed');
    assert.equal(cut.at(-1)!.response.error.code, 'backend_error');
    assert.equal(backend.requests.length, 3);
  } finally { await gateway.stop(); await backend.close(); }
});

test('retry: when the window ends after the stream started, the client gets response.failed', async () => {
  const dead = await deadPort();
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: dead }], { retryWindowMs: 700, streamStartMs: 50 });
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    assert.equal(res.status, 200);
    const evs = await events(res);
    assert.equal(evs[0].type, 'response.created');
    assert.equal(evs.at(-1)!.type, 'response.failed');
    assert.equal(evs.at(-1)!.response.error.code, 'backend_unavailable');
  } finally { await gateway.stop(); }
});

test('retry: a backend 4xx after the stream started keeps its meaning for Codex', async () => {
  const backend = await mockBackend((_b, _q, res) => setTimeout(() => {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end('{"error":{"message":"This model\'s maximum context length is 262144 tokens."}}');
  }, 200));
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { streamStartMs: 50 });
  try {
    const evs = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true }));
    assert.equal(evs.at(-1)!.type, 'response.failed');
    assert.equal(evs.at(-1)!.response.error.code, 'context_length_exceeded');
    assert.equal(backend.requests.length, 1);
  } finally { await gateway.stop(); await backend.close(); }
});

test('no total request limit by default: a long generation that streams tokens completes', async () => {
  assert.equal(defaultConfig().requestTimeoutMs, 0);
  assert.equal(loadConfig({}, []).requestTimeoutMs, 0);
  const backend = await mockBackend((_b, _q, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    let n = 0;
    const timer = setInterval(() => {
      if (++n <= 12) { res.write(chunkFrame({ choices: [{ index: 0, delta: { content: 't' }, finish_reason: null }] })); return; }
      clearInterval(timer);
      res.write(chunkFrame({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] }));
      res.end('data: [DONE]\n\n');
    }, 50);
  });
  // The whole stream takes about 650 ms. The idle limit is 200 ms.
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }], { idleTimeoutMs: 200 });
  try {
    const evs = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true }));
    assert.equal(evs.at(-1)!.type, 'response.completed');
    assert.equal(evs.at(-1)!.response.output[0].content[0].text, 't'.repeat(12));
  } finally { await gateway.stop(); await backend.close(); }
});

test('drain: a request during shutdown gets 503 with Retry-After', async () => {
  let release!: () => void;
  const backend = await mockBackend((_b, _q, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.flushHeaders();
    release = () => { for (const c of reply('late')) res.write(chunkFrame(c)); res.end('data: [DONE]\n\n'); };
  });
  const { gateway, base } = await startGateway([{ name: 'a', baseUrl: backend.url }], { shutdownGraceMs: 5_000 });
  try {
    const port = gateway.address!.port;
    // This request sends its headers now and its body after the drain starts.
    let sendRest!: () => void;
    const late = new Promise<{ status: number; retryAfter: string | undefined; body: Obj }>((resolve, reject) => {
      const req = http.request({ host: '127.0.0.1', port, path: '/v1/responses', method: 'POST', agent: false, headers: { 'Content-Type': 'application/json' } }, (res) => {
        let text = '';
        res.on('data', (c) => { text += c; });
        res.on('end', () => resolve({ status: res.statusCode ?? 0, retryAfter: res.headers['retry-after'] as string | undefined, body: JSON.parse(text) }));
      });
      req.on('error', reject);
      req.write('{"model":"qwen3.8-flash-next",');
      sendRest = () => req.end('"input":"x","stream":true}');
    });
    const first = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    assert.equal(first.status, 200);
    const stopped = gateway.stop();
    sendRest();
    const refused = await late;
    assert.equal(refused.status, 503);
    assert.equal(refused.retryAfter, '2');
    assert.equal(refused.body.error.message, 'the gateway is shutting down');
    release();
    const evs = await events(first);
    await stopped;
    assert.equal(evs.at(-1)!.type, 'response.completed');
  } finally { await backend.close(); }
});

test('drain: a stream that outlives the grace time ends with response.failed', async () => {
  const backend = await mockBackend((_b, _q, res) => {
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.write(chunkFrame({ choices: [{ index: 0, delta: { content: 'a' }, finish_reason: null }] }));
  });
  const { gateway, base } = await startGateway([{ name: 'a', baseUrl: backend.url }], { shutdownGraceMs: 200 });
  try {
    const res = await post(base, { model: 'qwen3.8-flash-next', input: 'x', stream: true });
    const stopped = gateway.stop();
    const evs = await events(res);
    await stopped;
    assert.equal(evs.at(-1)!.type, 'response.failed');
    assert.match(evs.at(-1)!.response.error.message, /shutting down/);
  } finally { await backend.close(); }
});

test('/health reports the backend state, the in-flight count and the retry window', async () => {
  const good = await mockBackend((_b, _q, res) => sse(res, reply('ok')));
  const dead = await deadPort();
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: good.url }, { name: 'spark2', baseUrl: dead }]);
  try {
    await gateway.checkHealth();
    const res = await fetch(`${base}/health`);
    assert.equal(res.status, 200);
    const h = await res.json() as Obj;
    assert.equal(h.status, 'ok');
    assert.equal(h.in_flight, 0);
    assert.equal(h.retry_window_ms, 180_000);
    const [one, two] = h.models[0].backends;
    assert.equal(one.healthy, true);
    assert.match(one.last_ok_at, /^\d{4}-/);
    assert.equal(one.consecutive_failures, 0);
    assert.equal(two.healthy, false);
    assert.equal(two.last_ok_at, null);
    assert(two.consecutive_failures >= 1);
    assert(two.last_error);
  } finally { await gateway.stop(); await good.close(); }
  const down = await startGateway([{ name: 'spark1', baseUrl: dead }]);
  try {
    await down.gateway.checkHealth();
    const res = await fetch(`${down.base}/health`);
    assert.equal(res.status, 503);
    assert.equal(res.headers.get('retry-after'), '2');
    assert.equal((await res.json() as Obj).status, 'degraded');
  } finally { await down.gateway.stop(); }
});

test('config: retry window, keepalive and stream start come from the environment', () => {
  const config = loadConfig({ PULSE_GATEWAY_RETRY_WINDOW_MS: '60000', PULSE_GATEWAY_KEEPALIVE_MS: '5000', PULSE_GATEWAY_STREAM_START_MS: '0' }, []);
  assert.deepEqual([config.retryWindowMs, config.keepaliveMs, config.streamStartMs], [60_000, 5_000, 0]);
  const defaults = defaultConfig();
  assert.deepEqual([defaults.retryWindowMs, defaults.keepaliveMs, defaults.streamStartMs], [180_000, 10_000, 3_000]);
});

test('forced tool_choice: the backend gets a JSON schema and the client gets the call; a patch repair is counted', async () => {
  const backend = await mockBackend((body, _q, res) => {
    const forced = Boolean(body.response_format);
    const content = forced
      ? '[{"name": "calculator", "parameters": {"expression": "7*8"}}]'
      : '';
    sse(res, forced ? [
      { choices: [{ index: 0, delta: { content }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
    ] : [
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'p1', function: { name: 'apply_patch', arguments: JSON.stringify({ input: '*** Add File: a.txt\n+hi\n' }) } }] }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
    ]);
  });
  const lines: Obj[] = [];
  setLogSink((_level, line) => lines.push(JSON.parse(line)));
  setLogLevel('info');
  const { gateway, base } = await startGateway([{ name: 'spark1', baseUrl: backend.url }]);
  try {
    const tools = [
      { type: 'function', name: 'calculator', parameters: { type: 'object', properties: { expression: { type: 'string' } } } },
      { type: 'custom', name: 'apply_patch' },
    ];
    const forced = await (await post(base, { model: 'qwen3.8-flash-next', input: '7*8?', tools, tool_choice: 'required' })).json() as Obj;
    assert.equal(backend.requests[0].tool_choice, undefined);
    assert.equal(backend.requests[0].response_format.json_schema.name, 'tool_calls');
    assert.equal(forced.status, 'completed');
    assert.deepEqual([forced.output[0].type, forced.output[0].name, forced.output[0].arguments], ['function_call', 'calculator', '{"expression":"7*8"}']);

    const evs = await events(await post(base, { model: 'qwen3.8-flash-next', input: 'add a.txt', tools, stream: true }));
    const done = evs.find((e) => e.type === 'response.custom_tool_call_input.done')!;
    assert.equal(done.input, '*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch\n');
    const metrics = await (await fetch(`${base}/metrics`)).json() as Obj;
    assert.equal(metrics.forced_tool_choice, 1);
    assert.equal(metrics.tool_call_repairs, 1);
    const repair = lines.find((l) => l.msg === 'repaired tool call')!;
    assert.deepEqual([repair.name, repair.added], ['apply_patch', ['begin', 'end']]);
  } finally {
    setLogSink(null);
    setLogLevel('error');
    await gateway.stop(); await backend.close();
  }
});

test('config: forced tool_choice mode comes from the environment and is checked', () => {
  assert.equal(loadConfig({}, []).forcedToolChoice, undefined);
  assert.equal(loadConfig({ PULSE_GATEWAY_FORCED_TOOL_CHOICE: 'native' }, []).forcedToolChoice, 'native');
  assert.throws(() => loadConfig({ PULSE_GATEWAY_FORCED_TOOL_CHOICE: 'always' }, []), /forcedToolChoice must be/);
});
