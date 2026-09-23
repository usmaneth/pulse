import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { handleResponses, sseData, toChat } from './responses.js';

test('history preserves instructions, reasoning, parallel calls, and tool results', () => {
  const { payload } = toChat({ instructions: 'first', input: [
    { role: 'developer', content: 'second' }, { role: 'user', content: 'go' },
    { type: 'reasoning', summary: [{ text: 'plan' }] },
    { type: 'function_call', call_id: 'a', name: 'one', arguments: '{}' },
    { type: 'function_call', call_id: 'b', name: 'two', arguments: '{}' },
    { type: 'function_call_output', call_id: 'a', output: 'ok' },
    { type: 'function_call_output', call_id: 'b', output: [{ type: 'input_text', text: 'fine' }] },
  ] });
  assert.deepEqual(payload.messages[0], { role: 'system', content: 'first\n\nsecond' });
  assert.equal(payload.messages[2].reasoning_content, 'plan');
  assert.equal(payload.messages[2].tool_calls.length, 2);
  assert.equal(payload.messages[4].tool_call_id, 'b');
});

test('custom and namespaced tools retain their input format', () => {
  const { payload, custom } = toChat({ input: [
    { type: 'custom_tool_call', call_id: 'c', name: 'patch', input: '*** patch' },
    { type: 'custom_tool_call_output', call_id: 'c', output: 'ok' },
  ], tools: [{ type: 'custom', name: 'patch', format: { type: 'grammar', syntax: 'lark', definition: 'start: PATCH' } }, { type: 'namespace', name: 'ns', tools: [{ type: 'function', name: 'read' }] }] });
  assert(custom.has('patch'));
  assert(payload.tools[0].function.description.includes('start: PATCH'));
  assert.equal(payload.tools[1].function.name, 'ns.read');
  assert.equal(payload.messages[0].tool_calls[0].function.arguments, '{"input":"*** patch"}');
});

test('unsupported state, media, and hosted tools fail explicitly', () => {
  for (const request of [null, { input: '', previous_response_id: 'x' }, { input: '', store: true },
    { input: '', tools: [{ type: 'web_search' }] }, { input: [{ role: 'user', content: [{ type: 'input_image' }] }] },
    { input: [{ role: 'user', content: null }] }, { input: [{ type: 'reasoning', encrypted_content: 'opaque' }] }]) assert.throws(() => toChat(request as any));
});

test('SSE parses split UTF-8, CRLF, comments, and multiline data', async () => {
  const bytes = new TextEncoder().encode(': heartbeat\r\ndata: hé\r\ndata: llo\r\n\r\ndata: [DONE]\n\n');
  const body = new ReadableStream<Uint8Array>({ start(controller) {
    for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
    controller.close();
  } });
  const events: string[] = [];
  for await (const event of sseData(body)) events.push(event);
  assert.deepEqual(events, ['hé\nllo', '[DONE]']);
});

async function listen(server: http.Server): Promise<string> {
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${(server.address() as any).port}`;
}
const close = (server: http.Server) => new Promise<void>(resolve => { server.closeAllConnections(); server.close(() => resolve()); });

test('HTTP stream emits ordered lifecycle, tools, continuation, usage and incomplete status', async () => {
  const requests: any[] = [];
  const backend = http.createServer(async (req, res) => {
    let raw = ''; for await (const chunk of req) raw += chunk;
    requests.push(JSON.parse(raw));
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    const send = (delta: any, finish_reason: any = null) => res.write(`data: ${JSON.stringify({ choices: [{ delta, finish_reason }] })}\n\n`);
    if (requests.length === 1) {
      send({ reasoning_content: 'plan' });
      send({ tool_calls: [{ index: 0, id: 'call1', function: { name: 'pa', arguments: '{"in' } }] });
      send({ tool_calls: [{ index: 0, function: { name: 'tch', arguments: 'put":"edit"}' } }] });
      send({}, 'tool_calls');
    } else { send({ content: 'done' }); send({}, 'length'); }
    res.write('data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24}}\n\n');
    res.end('data: [DONE]\n\n');
  });
  const backendUrl = await listen(backend);
  const adapter = http.createServer((req, res) => void handleResponses(req, res, backendUrl));
  const url = await listen(adapter);
  try {
    const response = await fetch(url, { method: 'POST', headers: { 'X-Pulse-Reasoning-Budget': '512' }, body: JSON.stringify({ input: 'go', stream: true, tools: [{ type: 'custom', name: 'patch' }] }) });
    const events = (await response.text()).split('\n').filter(line => line.startsWith('data:')).map(line => JSON.parse(line.slice(5)));
    assert.equal(requests[0].reasoning_budget_tokens, 512);
    events.forEach((event, index) => assert.equal(event.sequence_number, index));
    assert.equal(events.at(-1).type, 'response.completed');
    assert.equal(events.at(-1).response.usage.total_tokens, 24);
    const output = events.at(-1).response.output;
    assert.equal(output[1].type, 'custom_tool_call');
    assert.equal(output[1].input, 'edit');
    assert(!events.some(event => event.type === 'response.function_call_arguments.delta'));
    const next = await fetch(url, { method: 'POST', body: JSON.stringify({ input: [
      { role: 'user', content: 'go' }, ...output, { type: 'custom_tool_call_output', call_id: 'call1', output: 'ok' },
    ], tools: [{ type: 'custom', name: 'patch' }] }) });
    const result: any = await next.json();
    assert.equal(result.status, 'incomplete');
    assert.equal(result.incomplete_details.reason, 'max_output_tokens');
    assert.equal(requests[1].messages.at(-1).tool_call_id, 'call1');
    assert.equal(requests[1].messages[1].reasoning_content, 'plan');
  } finally { await close(adapter); await close(backend); }
});

test('HTTP truncated stream fails instead of reporting completion', async () => {
  const backend = http.createServer((req, res) => { req.resume(); res.end('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'); });
  const backendUrl = await listen(backend);
  const adapter = http.createServer((req, res) => void handleResponses(req, res, backendUrl));
  const url = await listen(adapter);
  try {
    const response = await fetch(url, { method: 'POST', body: JSON.stringify({ input: 'go', stream: true }) });
    const text = await response.text();
    assert(text.includes('response.failed'));
    assert(!text.includes('response.completed'));
  } finally { await close(adapter); await close(backend); }
});

test('client cancellation closes the backend request', async () => {
  let closed!: () => void;
  const cancelled = new Promise<void>(resolve => { closed = resolve; });
  const backend = http.createServer((req, res) => {
    req.resume(); res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.write('data: {"choices":[{"delta":{"content":"hello"}}]}\n\n');
    res.on('close', closed);
  });
  const backendUrl = await listen(backend);
  const adapter = http.createServer((req, res) => void handleResponses(req, res, backendUrl));
  const url = await listen(adapter);
  try {
    const abort = new AbortController();
    const response = await fetch(url, { method: 'POST', body: JSON.stringify({ input: 'go', stream: true }), signal: abort.signal });
    await response.body!.getReader().read(); abort.abort();
    await Promise.race([cancelled, new Promise((_, reject) => setTimeout(() => reject(new Error('backend did not cancel')), 1500).unref())]);
  } finally { await close(adapter); await close(backend); }
});

test('namespaced forced choice matches the declared backend name', () => {
  const { payload } = toChat({ input: 'go', tools: [{ type: 'namespace', name: 'ns', tools: [{ type: 'function', name: 'read' }] }],
    tool_choice: { type: 'function', namespace: 'ns', name: 'read' } });
  assert.equal(payload.tool_choice.function.name, payload.tools[0].function.name);
});

test('truncated custom tool input returns incomplete without an executable call', async () => {
  const backend = http.createServer((req, res) => {
    req.resume();
    res.end('data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x","function":{"name":"patch","arguments":"{\\\"input\\\":"}}]},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n');
  });
  const backendUrl = await listen(backend);
  const adapter = http.createServer((req, res) => void handleResponses(req, res, backendUrl));
  const url = await listen(adapter);
  try {
    const response = await fetch(url, { method: 'POST', body: JSON.stringify({ input: 'go', tools: [{ type: 'custom', name: 'patch' }] }) });
    const result: any = await response.json();
    assert.equal(result.status, 'incomplete');
    assert.deepEqual(result.output, []);
  } finally { await close(adapter); await close(backend); }
});
