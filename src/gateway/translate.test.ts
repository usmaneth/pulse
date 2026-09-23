// Unit tests for the Responses <-> Chat Completions translation.
// Run: npm run test:gateway

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ChatStreamTranslator, CUSTOM_TOOL_HINT, RequestError, arrangeSystem, flattenTools, grammarHint, mapToolChoice,
  mergeAssistantTurns, normalizeToolOutput, responsesToChat,
} from './translate.js';
import type { Obj, ResponseEvent } from './translate.js';

const qwen = { profile: 'qwen38' as const, upstreamModel: 'qwen3.8-flash-next', maxToolOutputChars: 12_000 };
const llama = { profile: 'llamacpp' as const, upstreamModel: 'bonsai', maxToolOutputChars: 12_000 };

// Tool definitions in the shape that Codex CLI sends.
const CODEX_TOOLS = [
  { type: 'function', name: 'shell_command', description: 'Run a shell command.', parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] } },
  { type: 'custom', name: 'apply_patch', description: 'Use the apply_patch tool to edit files.', format: { type: 'grammar', syntax: 'lark', definition: 'start: begin_patch' } },
  { type: 'namespace', name: 'mcp__github', tools: [{ type: 'function', name: 'get_issue', description: 'Get an issue.', parameters: { type: 'object', properties: { number: { type: 'integer' } } } }] },
  { type: 'tool_search' },
  { type: 'web_search' },
];

// ------------------------------------------------------------ reasoning

for (const effort of ['none', 'minimal', 'low']) {
  test(`qwen38: effort ${effort} turns thinking off and sends no reasoning_effort`, () => {
    const { payload } = responsesToChat({ model: 'm', input: 'hi', reasoning: { effort } }, qwen);
    assert.deepEqual(payload.chat_template_kwargs, { enable_thinking: false });
    assert.equal(payload.reasoning_effort, undefined);
  });
}

for (const effort of ['medium', 'high', 'xhigh']) {
  test(`qwen38: effort ${effort} turns thinking on and passes the effort`, () => {
    const { payload } = responsesToChat({ model: 'm', input: 'hi', reasoning: { effort } }, qwen);
    assert.deepEqual(payload.chat_template_kwargs, { enable_thinking: true });
    assert.equal(payload.reasoning_effort, effort);
  });
}

test('qwen38: no effort leaves the template default', () => {
  const { payload } = responsesToChat({ model: 'm', input: 'hi' }, qwen);
  assert.equal(payload.chat_template_kwargs, undefined);
  assert.equal(payload.reasoning_effort, undefined);
  assert.equal(payload.cache_prompt, undefined);
});

test('llamacpp: effort passes through and prompt caching is on', () => {
  const { payload } = responsesToChat({ model: 'm', input: 'hi', reasoning: { effort: 'low' } }, llama);
  assert.equal(payload.reasoning_effort, 'low');
  assert.equal(payload.chat_template_kwargs, undefined);
  assert.equal(payload.cache_prompt, true);
});

// ---------------------------------------------------------------- request

test('payload streams from the backend, asks for usage and uses the upstream model', () => {
  const { payload } = responsesToChat({ model: 'alias', input: 'hi', max_output_tokens: 99, temperature: 0.2, stream: false }, qwen);
  assert.equal(payload.model, 'qwen3.8-flash-next');
  assert.equal(payload.stream, true);
  assert.deepEqual(payload.stream_options, { include_usage: true });
  assert.equal(payload.max_tokens, 99);
  assert.equal(payload.temperature, 0.2);
  assert.deepEqual(payload.messages, [{ role: 'user', content: 'hi' }]);
});

test('tools: function kept, custom wrapped, namespace flattened, tool_search and others dropped', () => {
  const { tools, map } = flattenTools(CODEX_TOOLS);
  assert.deepEqual(tools.map((t) => t.function.name), ['shell_command', 'apply_patch', 'mcp__github__get_issue']);
  const patch = tools[1].function;
  assert.equal(patch.description, 'Use the apply_patch tool to edit files.' + CUSTOM_TOOL_HINT +
    '\nThe input string must follow this lark grammar:\nstart: begin_patch');
  assert.deepEqual(patch.parameters, { type: 'object', properties: { input: { type: 'string' } }, required: ['input'] });
  assert(map.custom.has('apply_patch'));
  assert.deepEqual(map.namespaced.get('mcp__github__get_issue'), ['mcp__github', 'get_issue']);
  assert.deepEqual(tools[0].function.parameters, CODEX_TOOLS[0].parameters);
});

test('history: system merge, tool calls, custom calls, namespaces, parallel calls', () => {
  const { payload } = responsesToChat({
    model: 'm',
    instructions: 'base instructions',
    tools: CODEX_TOOLS,
    input: [
      { type: 'message', role: 'developer', content: [{ type: 'input_text', text: 'dev note' }] },
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: '<recommended_plugins>x</recommended_plugins>' }] },
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'fix the bug' }, { type: 'input_image', image_url: 'data:' }] },
      { type: 'reasoning', summary: [{ type: 'summary_text', text: 'thinking' }] },
      { type: 'function_call', call_id: 'c1', name: 'shell_command', arguments: '{"command":"ls"}' },
      { type: 'function_call', call_id: 'c2', name: 'get_issue', namespace: 'mcp__github', arguments: '{"number":1}' },
      { type: 'function_call_output', call_id: 'c1', output: 'a.ts' },
      { type: 'function_call_output', call_id: 'c2', output: [{ type: 'input_text', text: 'issue body' }] },
      { type: 'custom_tool_call', call_id: 'c3', name: 'apply_patch', input: '*** Begin Patch\n*** End Patch' },
      { type: 'custom_tool_call_output', call_id: 'c3', output: 'Success' },
      { type: 'function_call', call_id: 'dangling', name: 'shell_command', arguments: '{}' },
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'done' }] },
    ],
  }, qwen);
  assert.deepEqual(payload.messages, [
    { role: 'system', content: 'base instructions\n\ndev note' },
    { role: 'user', content: 'fix the bug' },
    { role: 'assistant', content: null, reasoning_content: 'thinking', tool_calls: [
      { id: 'c1', type: 'function', function: { name: 'shell_command', arguments: '{"command":"ls"}' } },
      { id: 'c2', type: 'function', function: { name: 'mcp__github__get_issue', arguments: '{"number":1}' } },
    ] },
    { role: 'tool', tool_call_id: 'c1', content: 'a.ts' },
    { role: 'tool', tool_call_id: 'c2', content: 'issue body' },
    { role: 'assistant', content: null, tool_calls: [
      { id: 'c3', type: 'function', function: { name: 'apply_patch', arguments: JSON.stringify({ input: '*** Begin Patch\n*** End Patch' }) } },
    ] },
    { role: 'tool', tool_call_id: 'c3', content: 'Success' },
    { role: 'assistant', content: 'done' },
  ]);
});

test('tool output keeps head and tail when it exceeds the cap', () => {
  const text = 'a'.repeat(50) + 'b'.repeat(50);
  const out = normalizeToolOutput(text, 20);
  assert(out.startsWith('a'.repeat(10)));
  assert(out.endsWith('b'.repeat(10)));
  assert(out.includes('100 chars total'));
  assert.equal(normalizeToolOutput(text, 0), text);
});

test('invalid requests raise RequestError', () => {
  assert.throws(() => responsesToChat([], qwen), RequestError);
  assert.throws(() => responsesToChat({ input: 3 }, qwen), RequestError);
  assert.throws(() => responsesToChat({ input: 'x', previous_response_id: 'r' }, qwen), RequestError);
});

test('unsupported input gets a clear RequestError instead of a silent drop', () => {
  const bad: Array<[Obj, RegExp]> = [
    [{ input: 'x', conversation: 'c' }, /conversation is not supported/],
    [{ input: 'x', background: true }, /background is not supported/],
    [{ input: 'x', store: true }, /store=true is not supported/],
    [{ input: 'x', text: { format: { type: 'json_schema', schema: {} } } }, /text.format json_schema/],
    [{ input: [{ type: 'web_search_call', id: 'w' }] }, /unsupported input item type: web_search_call/],
    [{ input: [{ type: 'message', role: 'tool', content: 'x' }] }, /unsupported message role: tool/],
  ];
  for (const [request, message] of bad) assert.throws(() => responsesToChat(request, qwen), message);
  // Codex sends these. They must pass.
  responsesToChat({ input: 'x', store: false, text: { format: { type: 'text' }, verbosity: 'low' } }, qwen);
});

test('grammar hint: only custom tools with a grammar format get one', () => {
  assert.equal(grammarHint(undefined), '');
  assert.equal(grammarHint({ type: 'text' }), '');
  assert.equal(grammarHint({ type: 'grammar', syntax: 'lark', definition: '  ' }), '');
  assert.equal(grammarHint({ type: 'grammar', syntax: 'regex', definition: 'a+\n' }), '\nThe input string must follow this regex grammar:\na+');
  const { tools } = flattenTools([{ type: 'custom', name: 'free', description: 'd' }]);
  assert.equal(tools[0].function.description, 'd' + CUSTOM_TOOL_HINT);
});

test('reasoning replay: reasoning, text and calls of one turn become one assistant message', () => {
  const input = [
    { type: 'message', role: 'user', content: 'go' },
    { type: 'reasoning', summary: [{ type: 'summary_text', text: 'plan' }] },
    { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Running ls.' }] },
    { type: 'function_call', call_id: 'c1', name: 'shell_command', arguments: '{"command":"ls"}' },
    { type: 'function_call_output', call_id: 'c1', output: 'a.ts' },
    { type: 'reasoning', summary: [], encrypted_content: 'opaque' },
    { type: 'reasoning', summary: [], content: [{ type: 'reasoning_text', text: 'from content' }] },
    { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Done.' }] },
  ];
  const { payload } = responsesToChat({ model: 'm', input }, qwen);
  assert.deepEqual(payload.messages, [
    { role: 'user', content: 'go' },
    { role: 'assistant', content: 'Running ls.', reasoning_content: 'plan', tool_calls: [
      { id: 'c1', type: 'function', function: { name: 'shell_command', arguments: '{"command":"ls"}' } },
    ] },
    { role: 'tool', tool_call_id: 'c1', content: 'a.ts' },
    { role: 'assistant', content: 'Done.', reasoning_content: 'from content' },
  ]);
  const off = responsesToChat({ model: 'm', input }, { ...qwen, replayReasoning: false }).payload;
  assert.equal(off.messages[1].reasoning_content, undefined);
  assert.equal(off.messages[1].content, 'Running ls.');
});

test('merge: a new model turn starts at reasoning or text that follows calls', () => {
  const call = (id: string) => ({ id, type: 'function', function: { name: 'f', arguments: '{}' } });
  assert.deepEqual(mergeAssistantTurns([
    { role: 'assistant', content: null, tool_calls: [call('a')] },
    { role: 'assistant', content: null, tool_calls: [call('b')] },
    { role: 'assistant', content: null, reasoning_content: 'r2' },
    { role: 'assistant', content: 'x' },
    { role: 'assistant', content: null, tool_calls: [call('c')] },
    { role: 'assistant', content: 'after calls' },
  ]), [
    { role: 'assistant', content: null, tool_calls: [call('a'), call('b')] },
    { role: 'assistant', content: 'x', reasoning_content: 'r2', tool_calls: [call('c')] },
    { role: 'assistant', content: 'after calls' },
  ]);
});

test('system: qwen38 keeps a late developer message in place; llamacpp moves it to the start', () => {
  const messages = [
    { role: 'system', content: 'a' },
    { role: 'system', content: 'b' },
    { role: 'user', content: 'u1' },
    { role: 'system', content: 'late' },
    { role: 'system', content: '' },
    { role: 'user', content: 'u2' },
  ];
  assert.deepEqual(arrangeSystem(messages.map((m) => ({ ...m })), 'qwen38'), [
    { role: 'system', content: 'a\n\nb' },
    { role: 'user', content: 'u1' },
    { role: 'system', content: 'late' },
    { role: 'user', content: 'u2' },
  ]);
  assert.deepEqual(arrangeSystem(messages.map((m) => ({ ...m })), 'llamacpp'), [
    { role: 'system', content: 'a\n\nb\n\nlate' },
    { role: 'user', content: 'u1' },
    { role: 'user', content: 'u2' },
  ]);
});

test('prefix stability: the next turn starts with the same messages', () => {
  const turn1 = [
    { type: 'message', role: 'developer', content: [{ type: 'input_text', text: 'rules' }] },
    { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'task' }] },
  ];
  const turn2 = [
    ...turn1,
    { type: 'reasoning', summary: [{ type: 'summary_text', text: 'plan' }] },
    { type: 'function_call', call_id: 'c1', name: 'shell_command', arguments: '{"command":"ls"}' },
    { type: 'function_call_output', call_id: 'c1', output: 'a.ts' },
    { type: 'message', role: 'developer', content: [{ type: 'input_text', text: 'approval mode changed' }] },
  ];
  const body = { model: 'm', instructions: 'base', tools: CODEX_TOOLS, reasoning: { effort: 'high' } };
  const a = responsesToChat({ ...body, input: turn1 }, qwen).payload;
  const b = responsesToChat({ ...body, input: turn2 }, qwen).payload;
  assert.equal(JSON.stringify(b.tools), JSON.stringify(a.tools));
  assert.equal(JSON.stringify(b.messages.slice(0, a.messages.length)), JSON.stringify(a.messages));
});

test('tool_choice: names map to the flat function name; parallel_tool_calls passes', () => {
  const names = new Set(['shell_command', 'apply_patch', 'mcp__github__get_issue']);
  assert.equal(mapToolChoice('auto', names), 'auto');
  assert.equal(mapToolChoice('required', names), 'required');
  assert.deepEqual(mapToolChoice({ type: 'custom', name: 'apply_patch' }, names), { type: 'function', function: { name: 'apply_patch' } });
  assert.deepEqual(mapToolChoice({ type: 'function', name: 'get_issue', namespace: 'mcp__github' }, names),
    { type: 'function', function: { name: 'mcp__github__get_issue' } });
  assert.throws(() => mapToolChoice({ type: 'function', name: 'nope' }, names), /not in tools/);
  assert.throws(() => mapToolChoice({ type: 'allowed_tools', tools: [] }, names), /unsupported tool_choice type/);
  assert.throws(() => mapToolChoice('sometimes', names), /unsupported tool_choice/);
  const { payload } = responsesToChat({ model: 'm', input: 'x', tools: CODEX_TOOLS, tool_choice: 'auto', parallel_tool_calls: true }, qwen);
  assert.equal(payload.tool_choice, 'auto');
  assert.equal(payload.parallel_tool_calls, true);
  // vLLM rejects tool_choice without tools, so it is not sent.
  const bare = responsesToChat({ model: 'm', input: 'x', tool_choice: 'auto', parallel_tool_calls: false }, qwen).payload;
  assert.equal(bare.tool_choice, undefined);
  assert.equal(bare.parallel_tool_calls, undefined);
});

// ----------------------------------------------------------------- stream

const chunk = (delta: Obj, finish_reason: string | null = null) => ({ choices: [{ index: 0, delta, finish_reason }] });

function run(chunks: Obj[], tools: unknown = CODEX_TOOLS, emitReasoning = true): { events: ResponseEvent[]; t: ChatStreamTranslator } {
  const { map } = flattenTools(tools);
  const t = new ChatStreamTranslator('qwen3.8-flash-next', map, emitReasoning);
  const events = [...t.start()];
  for (const c of chunks) events.push(...t.push(c));
  events.push(...t.finishStream());
  return { events, t };
}

const types = (events: ResponseEvent[]) => events.map((e) => e.type);

test('stream: reasoning then text, in the Responses event order', () => {
  const { events, t } = run([
    chunk({ role: 'assistant', content: '' }),
    chunk({ reasoning_content: 'Let me ' }),
    chunk({ reasoning: 'think.' }),
    chunk({ content: 'Hello' }),
    chunk({ content: ' world' }),
    chunk({}, 'stop'),
    { choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15, prompt_tokens_details: { cached_tokens: 4 } } },
  ]);
  assert.deepEqual(types(events), [
    'response.created',
    'response.in_progress',
    'response.output_item.added',
    'response.reasoning_summary_part.added',
    'response.reasoning_summary_text.delta',
    'response.reasoning_summary_text.delta',
    'response.reasoning_summary_text.done',
    'response.reasoning_summary_part.done',
    'response.output_item.done',
    'response.output_item.added',
    'response.content_part.added',
    'response.output_text.delta',
    'response.output_text.delta',
    'response.output_text.done',
    'response.content_part.done',
    'response.output_item.done',
    'response.completed',
  ]);
  events.forEach((e, i) => assert.equal(e.sequence_number, i));
  const final = t.response;
  assert.equal(final.status, 'completed');
  assert.equal(final.output[0].type, 'reasoning');
  assert.equal(final.output[0].summary[0].text, 'Let me think.');
  assert.equal(final.output[1].content[0].text, 'Hello world');
  assert.equal(final.output[1].status, 'completed');
  assert.deepEqual(final.usage, {
    input_tokens: 10, output_tokens: 5, total_tokens: 15,
    input_tokens_details: { cached_tokens: 4 }, output_tokens_details: { reasoning_tokens: 0 },
  });
  assert.equal((events.at(-1) as Obj).response.output.length, 2);
});

test('stream: reasoning can be hidden', () => {
  const { events, t } = run([chunk({ reasoning_content: 'x' }), chunk({ content: 'y' }), chunk({}, 'stop')], [], false);
  assert(!types(events).some((e) => e.includes('reasoning')));
  assert.equal(t.response.output.length, 1);
});

test('stream: qwen3_xml tool call deltas become a function_call', () => {
  const { events, t } = run([
    chunk({ reasoning_content: 'need ls' }),
    chunk({ tool_calls: [{ index: 0, id: 'chatcmpl-tool-1', type: 'function', function: { name: 'shell_command', arguments: '' } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: '{"command": ' } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: '"ls -la"}' } }] }),
    chunk({}, 'tool_calls'),
  ]);
  const call = t.response.output[1];
  assert.deepEqual(
    { type: call.type, call_id: call.call_id, name: call.name, arguments: call.arguments, status: call.status },
    { type: 'function_call', call_id: 'chatcmpl-tool-1', name: 'shell_command', arguments: '{"command": "ls -la"}', status: 'completed' },
  );
  assert.deepEqual(types(events).slice(-7), [
    'response.output_item.done', // reasoning closes when the tool call starts
    'response.output_item.added',
    'response.function_call_arguments.delta',
    'response.function_call_arguments.delta',
    'response.function_call_arguments.done',
    'response.output_item.done',
    'response.completed',
  ]);
  const deltas = events.filter((e) => e.type === 'response.function_call_arguments.delta').map((e) => (e as Obj).delta);
  assert.deepEqual(deltas, ['{"command": ', '"ls -la"}']);
  const added = events.find((e) => e.type === 'response.output_item.added' && (e as Obj).item.type === 'function_call') as Obj;
  assert.equal(added.output_index, 1);
  assert.equal(added.item.status, 'in_progress');
});

test('stream: custom tool call is unwrapped to raw input', () => {
  const patch = '*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch';
  const args = JSON.stringify({ input: patch });
  const { events, t } = run([
    chunk({ tool_calls: [{ index: 0, id: 'call_p', function: { name: 'apply_patch', arguments: args.slice(0, 10) } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: args.slice(10) } }] }),
    chunk({}, 'tool_calls'),
  ]);
  const item = t.response.output[0];
  assert.equal(item.type, 'custom_tool_call');
  assert.equal(item.name, 'apply_patch');
  assert.equal(item.input, patch);
  assert.equal(item.call_id, 'call_p');
  assert.equal(item.arguments, undefined);
  assert.deepEqual(types(events).slice(2), [
    'response.output_item.added',
    'response.custom_tool_call_input.delta',
    'response.custom_tool_call_input.done',
    'response.output_item.done',
    'response.completed',
  ]);
  assert(!types(events).includes('response.function_call_arguments.delta'));
});

test('stream: custom tool call without the JSON wrapper keeps the raw text', () => {
  const { t } = run([
    chunk({ tool_calls: [{ index: 0, id: 'c', function: { name: 'apply_patch', arguments: '*** Begin Patch' } }] }),
    chunk({}, 'tool_calls'),
  ]);
  assert.equal(t.response.output[0].input, '*** Begin Patch');
});

test('stream: namespaced call gets its namespace back; parallel calls keep order', () => {
  const { t } = run([
    chunk({ content: 'Checking.' }),
    chunk({ tool_calls: [
      { index: 0, id: 'a', function: { name: 'mcp__github__get_issue', arguments: '{"number":7}' } },
      { index: 1, id: 'b', function: { name: 'shell_command', arguments: '{"command":"pwd"}' } },
    ] }),
    chunk({}, 'tool_calls'),
  ]);
  const [msg, first, second] = t.response.output;
  assert.equal(msg.type, 'message');
  assert.deepEqual([first.name, first.namespace, first.call_id], ['get_issue', 'mcp__github', 'a']);
  assert.deepEqual([second.name, second.namespace, second.call_id], ['shell_command', undefined, 'b']);
});

test('stream: finish_reason length is incomplete and drops partial tool calls', () => {
  const { events, t } = run([
    chunk({ content: 'partial' }),
    chunk({ tool_calls: [{ index: 0, id: 'x', function: { name: 'shell_command', arguments: '{"comm' } }] }),
    chunk({}, 'length'),
  ]);
  assert.equal(t.response.status, 'incomplete');
  assert.deepEqual(t.response.incomplete_details, { reason: 'max_output_tokens' });
  // The text before the call is complete. The call is not, and it leaves the output.
  assert.equal(t.response.output.length, 1);
  assert.equal(t.response.output[0].type, 'message');
  assert.equal(t.response.output[0].status, 'completed');
  assert.equal(events.at(-1)!.type, 'response.incomplete');
  assert(!events.some((e) => e.type === 'response.function_call_arguments.done'));
  assert(!events.some((e) => e.type === 'response.output_item.done' && (e as Obj).item.type === 'function_call'));
});

test('stream: finish_reason length marks a text-only message incomplete', () => {
  const { t } = run([chunk({ content: 'partial' }), chunk({}, 'length')]);
  assert.equal(t.response.output[0].status, 'incomplete');
});

test('stream: finish_reason length drops every call of the turn, also the complete ones', () => {
  const { events, t } = run([
    chunk({ tool_calls: [{ index: 0, id: 'a', function: { name: 'shell_command', arguments: '{"command":"ls"}' } }] }),
    chunk({ tool_calls: [{ index: 1, id: 'b', function: { name: 'shell_command', arguments: '{"comm' } }] }),
    chunk({}, 'length'),
  ]);
  assert.deepEqual(t.response.output, []);
  assert(!events.some((e) => e.type === 'response.output_item.done'));
});

test('stream: function call arguments go out before the finish reason', () => {
  const { map } = flattenTools(CODEX_TOOLS);
  const t = new ChatStreamTranslator('m', map);
  t.start();
  const first = t.push(chunk({ tool_calls: [{ index: 0, id: 'c', function: { name: 'shell_command', arguments: '{"comm' } }] }));
  assert.deepEqual(types(first), ['response.output_item.added', 'response.function_call_arguments.delta']);
  assert.equal((first[0] as Obj).item.status, 'in_progress');
  assert.equal((first[0] as Obj).item.arguments, '');
  const second = t.push(chunk({ tool_calls: [{ index: 0, function: { arguments: 'and":"ls"}' } }] }));
  assert.deepEqual(types(second), ['response.function_call_arguments.delta']);
  assert.equal((second[0] as Obj).delta, 'and":"ls"}');
  t.push(chunk({}, 'tool_calls'));
  const end = t.finishStream();
  assert.deepEqual(types(end), ['response.function_call_arguments.done', 'response.output_item.done', 'response.completed']);
  assert.equal((end[0] as Obj).arguments, '{"command":"ls"}');
  assert.equal((end[1] as Obj).item.status, 'completed');
});

test('stream: text closes before the first call; white space between calls is dropped', () => {
  const { events, t } = run([
    chunk({ content: 'I will list.' }),
    chunk({ tool_calls: [{ index: 0, id: 'a', function: { name: 'shell_command', arguments: '{"command":"ls"}' } }] }),
    chunk({ content: '\n' }),
    chunk({ tool_calls: [{ index: 1, id: 'b', function: { name: 'shell_command', arguments: '{"command":"pwd"}' } }] }),
    chunk({}, 'tool_calls'),
  ]);
  assert.deepEqual(t.response.output.map((i: Obj) => i.type), ['message', 'function_call', 'function_call']);
  assert.equal(t.response.output[0].content[0].text, 'I will list.');
  const doneMsg = events.findIndex((e) => e.type === 'response.output_item.done' && (e as Obj).item.type === 'message');
  const addedCall = events.findIndex((e) => e.type === 'response.output_item.added' && (e as Obj).item.type === 'function_call');
  assert(doneMsg < addedCall);
  const doneIds = events.filter((e) => e.type === 'response.output_item.done').map((e) => (e as Obj).item.call_id);
  assert.deepEqual(doneIds, [undefined, 'a', 'b']);
});

test('stream: arguments that arrive before the name go out when the name arrives', () => {
  const { events, t } = run([
    chunk({ tool_calls: [{ index: 0, id: 'a', function: { arguments: '{"command":' } }] }),
    chunk({ tool_calls: [{ index: 0, function: { name: 'shell_command', arguments: '"ls"}' } }] }),
    chunk({}, 'tool_calls'),
  ]);
  const deltas = events.filter((e) => e.type === 'response.function_call_arguments.delta').map((e) => (e as Obj).delta);
  assert.equal(deltas.join(''), '{"command":"ls"}');
  assert.equal(t.response.output[0].arguments, '{"command":"ls"}');
});

test('stream: a call without arguments gets {}', () => {
  const { events, t } = run([
    chunk({ tool_calls: [{ index: 0, id: 'a', function: { name: 'shell_command' } }] }),
    chunk({}, 'tool_calls'),
  ]);
  assert.equal(t.response.output[0].arguments, '{}');
  assert.equal(events.filter((e) => e.type === 'response.function_call_arguments.delta').map((e) => (e as Obj).delta).join(''), '{}');
});

test('stream: a stream without a finish reason fails', () => {
  const { events, t } = run([chunk({ content: 'cut' })]);
  assert.equal(events.at(-1)!.type, 'response.failed');
  assert.equal(t.response.status, 'failed');
  assert(!types(events).includes('response.completed'));
});

test('stream: a backend error chunk throws', () => {
  const t = new ChatStreamTranslator('m', flattenTools([]).map);
  assert.throws(() => t.push({ error: { message: 'boom' } }), /boom/);
});
