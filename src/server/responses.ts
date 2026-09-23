import type { IncomingMessage, ServerResponse } from 'node:http';
import { randomUUID } from 'node:crypto';
import { appendFile } from 'node:fs/promises';
import { routeReasoning, BoundaryMetadata } from '../jev/boundary.js';

type Obj = Record<string, any>;
export class RequestError extends Error {}
const id = (prefix: string) => `${prefix}_${randomUUID().replaceAll('-', '')}`;
const requireString = (value: unknown, field: string): string => {
  if (typeof value !== 'string') throw new RequestError(`${field} must be a string`);
  return value;
};

function content(value: any): string {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) throw new RequestError('message content must be text');
  return value.map(part => {
    if (!part || typeof part !== 'object') throw new RequestError('invalid content part');
    if (!['input_text', 'output_text', 'text'].includes(part.type)) {
      throw new RequestError(`unsupported content type: ${part.type}`);
    }
    return requireString(part.text, 'content text');
  }).join('');
}

/** Translate full-history, stateless Responses requests into Chat Completions. */
export function toChat(request: Obj): { payload: Obj; custom: Set<string> } {
  if (!request || Array.isArray(request) || typeof request !== 'object') throw new RequestError('request must be an object');
  for (const key of ['previous_response_id', 'conversation', 'background']) {
    if (request[key]) throw new RequestError(`${key} is not supported; send the full input history`);
  }
  if (request.store === true) throw new RequestError('stored responses are not supported; set store=false');
  if (request.text?.format && request.text.format.type !== 'text') throw new RequestError('structured text output is not supported');
  const messages: Obj[] = [];
  if (request.instructions) messages.push({ role: 'system', content: requireString(request.instructions, 'instructions') });
  const input = typeof request.input === 'string' ? [{ role: 'user', content: request.input }] : request.input;
  if (!Array.isArray(input)) throw new RequestError('input must be a string or an array');
  for (const item of input) {
    if (!item || typeof item !== 'object') throw new RequestError('invalid input item');
    if (item.type === 'reasoning') {
      if (!Array.isArray(item.summary ?? [])) throw new RequestError('reasoning summary must be an array');
      const text = (item.summary ?? []).map((p: Obj) => requireString(p?.text, 'reasoning text')).join('');
      if (item.encrypted_content && !text) throw new RequestError('encrypted reasoning without a text summary is not supported');
      if (text) messages.push({ role: 'assistant', content: null, reasoning_content: text });
      continue;
    }
    if (['function_call', 'custom_tool_call'].includes(item.type)) {
      const custom = item.type === 'custom_tool_call';
      messages.push({ role: 'assistant', content: null, tool_calls: [{
        id: requireString(item.call_id, 'call_id'), type: 'function', function: {
          name: item.namespace ? `${item.namespace}.${requireString(item.name, 'tool name')}` : requireString(item.name, 'tool name'),
          arguments: custom ? JSON.stringify({ input: requireString(item.input, 'tool input') }) : requireString(item.arguments, 'arguments'),
        },
      }] });
    } else if (['function_call_output', 'custom_tool_call_output'].includes(item.type)) {
      messages.push({ role: 'tool', tool_call_id: requireString(item.call_id, 'call_id'), content: content(item.output) });
    } else if (!item.type || item.type === 'message') {
      if (!['system', 'developer', 'user', 'assistant'].includes(item.role)) throw new RequestError('unsupported message role');
      messages.push({ role: item.role === 'developer' ? 'system' : item.role, content: content(item.content) });
    } else throw new RequestError(`unsupported input item: ${item.type}`);
  }
  // Bonsai's template permits one system message at the start.
  const system = messages.filter(message => message.role === 'system');
  if (system.length) {
    for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === 'system') messages.splice(i, 1);
    messages.unshift({ role: 'system', content: system.map(message => message.content).join('\n\n') });
  }
  for (let i = 1; i < messages.length;) {
    const previous = messages[i - 1];
    const current = messages[i];
    if (previous.role === 'assistant' && current.role === 'assistant') {
      previous.content = (previous.content ?? '') + (current.content ?? '') || null;
      if (current.reasoning_content) previous.reasoning_content = (previous.reasoning_content ?? '') + current.reasoning_content;
      if (current.tool_calls) previous.tool_calls = [...(previous.tool_calls ?? []), ...current.tool_calls];
      messages.splice(i, 1);
    } else i++;
  }
  const custom = new Set<string>();
  if (request.tools != null && !Array.isArray(request.tools)) throw new RequestError('tools must be an array');
  const flatTools: Obj[] = [];
  for (const tool of request.tools ?? []) {
    if (!tool || typeof tool !== 'object') throw new RequestError('invalid tool');
    if (tool.type === 'namespace') {
      if (!Array.isArray(tool.tools)) throw new RequestError('namespace tools must be an array');
      const namespace = requireString(tool.name, 'namespace name');
      for (const nested of tool.tools) {
        if (!nested || typeof nested !== 'object') throw new RequestError('invalid namespace tool');
        flatTools.push({ ...nested, name: `${namespace}.${requireString(nested.name, 'tool name')}` });
      }
    } else flatTools.push(tool);
  }
  const tools = flatTools.map((tool: Obj) => {
    if (!tool || typeof tool !== 'object') throw new RequestError('invalid tool');
    if (!['function', 'custom'].includes(tool.type)) throw new RequestError(`unsupported tool type: ${tool.type}`);
    const name = requireString(tool.name, 'tool name');
    if (tool.type === 'custom') custom.add(name);
    const grammar = tool.type === 'custom' && tool.format?.type === 'grammar'
      ? `\nThe input string must follow this ${tool.format.syntax ?? ''} grammar:\n${tool.format.definition ?? ''}` : '';
    const wrapper = tool.type === 'custom' ? 'This function wraps a text tool. Supply a JSON object with an input string. Apply the original tool instructions to that string, without JSON or code fences inside it.\nOriginal text-tool instructions:\n' : '';
    return { type: 'function', function: { name, description: wrapper + (tool.description ?? '') + grammar,
      parameters: tool.type === 'custom' ? {
        type: 'object', properties: { input: { type: 'string', description: 'The complete tool input as plain text.' } },
        required: ['input'], additionalProperties: false,
      } : tool.parameters ?? { type: 'object', properties: {} },
    } };
  });
  let toolChoice = request.tool_choice;
  if (toolChoice && typeof toolChoice === 'object') {
    if (!['function', 'custom'].includes(toolChoice.type)) throw new RequestError('unsupported tool_choice');
    toolChoice = { type: 'function', function: { name: toolChoice.namespace ? `${toolChoice.namespace}.${requireString(toolChoice.name, 'tool_choice name')}` : requireString(toolChoice.name, 'tool_choice name') } };
  }
  const payload: Obj = { model: request.model, messages, stream: true,
    stream_options: { include_usage: true }, cache_prompt: true };
  if (tools.length) payload.tools = tools;
  if (toolChoice != null) payload.tool_choice = toolChoice;
  for (const key of ['temperature', 'top_p', 'parallel_tool_calls']) if (request[key] != null) payload[key] = request[key];
  if (request.max_output_tokens != null) payload.max_tokens = request.max_output_tokens;
  // These are backend hints, not an OpenAI reasoning-token guarantee.
  if (request.reasoning?.effort) payload.reasoning_effort = request.reasoning.effort;
  return { payload, custom };
}

/** Parse SSE frames across arbitrary byte and UTF-8 boundaries. */
export async function* sseData(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let match: RegExpExecArray | null;
      while ((match = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const lines = frame.split(/\r?\n/).filter(line => line.startsWith('data:'));
        if (lines.length) yield lines.map(line => line.slice(5).replace(/^ /, '')).join('\n');
      }
      if (buffer.length > 8 * 1024 * 1024) throw new Error('backend SSE frame exceeds 8 MiB');
      if (done) break;
    }
    if (buffer.trim()) throw new Error('backend stream ended inside an SSE frame');
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}

async function write(res: ServerResponse, text: string): Promise<void> {
  if (res.destroyed) throw new Error('client disconnected');
  if (res.write(text)) return;
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => { res.off('drain', drain); res.off('close', close); res.off('error', error); };
    const drain = () => { cleanup(); resolve(); };
    const close = () => { cleanup(); reject(new Error('client disconnected')); };
    const error = (err: Error) => { cleanup(); reject(err); };
    res.once('drain', drain); res.once('close', close); res.once('error', error);
  });
}

export async function handleResponses(req: IncomingMessage, res: ServerResponse, backendUrl: string, activeRequests = 1): Promise<void> {
  const started = performance.now();
  let firstTokenMs: number | null = null;
  let firstVisibleMs: number | null = null;
  let backendTimings: Obj | null = null;
  let jevDecision: BoundaryMetadata | null = null;
  const abort = new AbortController();
  const disconnected = () => { if (!res.writableEnded) abort.abort(); };
  res.on('close', disconnected);
  const response: Obj = { id: id('resp'), object: 'response', created_at: Math.floor(Date.now() / 1000),
    status: 'in_progress', error: null, incomplete_details: null, output: [], usage: null, store: false };
  let sequence = 0;
  let streaming = false;
  let heartbeat: ReturnType<typeof setInterval> | undefined;
  const emit = async (type: string, fields: Obj) => {
    if (firstVisibleMs === null && ['response.output_text.delta', 'response.function_call_arguments.delta', 'response.custom_tool_call_input.delta'].includes(type)) firstVisibleMs = performance.now() - started;
    if (streaming) await write(res, `event: ${type}\ndata: ${JSON.stringify({ type, sequence_number: sequence++, ...fields })}\n\n`);
  };
  const timeout = setTimeout(() => abort.abort(new Error('backend request timed out')),
    Number(process.env.PULSE_BACKEND_TIMEOUT_MS ?? 600_000));
  try {
    let size = 0;
    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      size += chunk.length;
      if (size > Number(process.env.PULSE_MAX_BODY_BYTES ?? 16_777_216)) throw new RequestError('request body exceeds limit');
      chunks.push(chunk);
    }
    let request: Obj;
    try { request = JSON.parse(Buffer.concat(chunks).toString('utf8')); }
    catch { throw new RequestError('invalid JSON'); }
    const { payload, custom } = toChat(request);
    response.model = request.model;
    const budgetHeader = req.headers['x-pulse-reasoning-budget'];
    if (budgetHeader !== undefined) {
      const budget = Number(budgetHeader);
      if (!Number.isSafeInteger(budget) || budget < -1) throw new RequestError('X-Pulse-Reasoning-Budget must be an integer of -1 or greater');
      payload.reasoning_budget_tokens = budget;
    }
    if (process.env.PULSE_JEV_MODE === 'request') {
      jevDecision = await routeReasoning(payload, budgetHeader !== undefined, abort.signal);
      response.metadata = { pulse_jev: JSON.stringify(jevDecision) };
    }
    if (process.env.PULSE_RESPONSES_TRACE_FILE) {
      await appendFile(process.env.PULSE_RESPONSES_TRACE_FILE, JSON.stringify({ response_id: response.id, payload }) + '\n', { mode: 0o600 });
    }
    if (process.env.PULSE_RESPONSES_SPEC_POLICY === '1') {
      const approxTokens = JSON.stringify({ messages: payload.messages, tools: payload.tools }).length / 3.6;
      const lastUser = payload.messages.filter((m: Obj) => m.role === 'user').at(-1)?.content ?? '';
      let k = /\bjson\b|schema|interface |dataclass|struct |enum |typedef |\btable\b/i.test(lastUser) ? 3 : 4;
      const cutoff = Number(process.env.PULSE_SPEC_CTX_CUTOFF ?? (activeRequests > 1 ? 4096 : 14336));
      if (approxTokens >= cutoff) k = 0;
      else if (activeRequests === 1 && approxTokens >= 10240) k = Math.min(k, 2);
      else if (activeRequests === 1 && approxTokens >= 8192) k = Math.min(k, 3);
      payload.spec_draft_n_max = k;
      res.setHeader('X-Pulse-Speculation-K', String(k));
    }
    streaming = request.stream === true;
    if (streaming) {
      res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no' });
      await emit('response.created', { response: { ...response, output: [] } });
      await emit('response.in_progress', { response: { ...response, output: [] } });
      // Heartbeats keep the connection active during cold prompt processing.
      heartbeat = setInterval(() => {
        if (!res.destroyed && res.writableLength === 0) res.write(': pulse heartbeat\n\n');
      }, 10_000).unref();
    }
    const backend = await fetch(`${backendUrl.replace(/\/$/, '')}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: abort.signal,
    });
    if (!backend.ok) {
      const error = await backend.text();
      if (streaming) throw new Error(`backend HTTP ${backend.status}: ${error}`);
      res.writeHead(backend.status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: 'backend_error', message: error } }));
      return;
    }
    if (!backend.body) throw new Error('backend response has no body');
    let reasoning: Obj | undefined;
    let reasoningIndex = -1;
    let message: Obj | undefined;
    let messageIndex = -1;
    let finish: string | undefined;
    let done = false;
    const calls = new Map<number, { item: Obj; index: number; raw: string }>();
    for await (const data of sseData(backend.body)) {
      if (data === '[DONE]') { done = true; break; }
      const chunk = JSON.parse(data);
      if (chunk.error) throw new Error(chunk.error.message ?? 'backend stream error');
      if (chunk.usage) response.usage = {
        input_tokens: chunk.usage.prompt_tokens ?? 0,
        output_tokens: chunk.usage.completion_tokens ?? 0,
        total_tokens: chunk.usage.total_tokens ?? (chunk.usage.prompt_tokens ?? 0) + (chunk.usage.completion_tokens ?? 0),
        input_tokens_details: { cached_tokens: chunk.usage.prompt_tokens_details?.cached_tokens ?? 0 },
        output_tokens_details: { reasoning_tokens: chunk.usage.completion_tokens_details?.reasoning_tokens ?? 0 },
      };
      if (chunk.timings) backendTimings = chunk.timings;
      const choice = chunk.choices?.[0];
      if (!choice) continue;
      if (choice.finish_reason) finish = choice.finish_reason;
      const delta = choice.delta ?? {};
      if (firstTokenMs === null && (delta.content || delta.reasoning_content || delta.tool_calls?.length)) firstTokenMs = performance.now() - started;
      if (delta.reasoning_content) {
        if (!reasoning) {
          reasoning = { id: id('rs'), type: 'reasoning', summary: [{ type: 'summary_text', text: '' }] };
          reasoningIndex = response.output.push(reasoning) - 1;
          await emit('response.output_item.added', { output_index: reasoningIndex, item: { ...reasoning, summary: [] } });
          await emit('response.reasoning_summary_part.added', { item_id: reasoning.id, output_index: reasoningIndex, summary_index: 0, part: { ...reasoning.summary[0] } });
        }
        reasoning.summary[0].text += delta.reasoning_content;
        await emit('response.reasoning_summary_text.delta', { item_id: reasoning.id, output_index: reasoningIndex, summary_index: 0, delta: delta.reasoning_content });
      }
      if (delta.content) {
        if (!message) {
          message = { id: id('msg'), type: 'message', role: 'assistant', status: 'in_progress', content: [] };
          messageIndex = response.output.push(message) - 1;
          await emit('response.output_item.added', { output_index: messageIndex, item: { ...message, content: [] } });
          message.content.push({ type: 'output_text', text: '', annotations: [] });
          await emit('response.content_part.added', { item_id: message.id, output_index: messageIndex, content_index: 0, part: { ...message.content[0] } });
        }
        message.content[0].text += delta.content;
        await emit('response.output_text.delta', { item_id: message.id, output_index: messageIndex, content_index: 0, delta: delta.content });
      }
      for (const tool of delta.tool_calls ?? []) {
        const slot = tool.index ?? 0;
        let call = calls.get(slot);
        if (!call) {
          const item: Obj = { id: id('fc'), type: 'function_call', status: 'in_progress', call_id: tool.id ?? id('call'), name: '', arguments: '' };
          call = { item, index: response.output.push(item) - 1, raw: '' };
          calls.set(slot, call);
        }
        if (tool.id) call.item.call_id = tool.id;
        if (tool.function?.name) call.item.name += tool.function.name;
        call.raw += tool.function?.arguments ?? '';

      }
    }
    if (!done || !finish) throw new Error('backend stream ended before completion');
    if (!['stop', 'tool_calls', 'length'].includes(finish)) throw new Error(`unsupported backend finish reason: ${finish}`);
    if (reasoning) {
      await emit('response.reasoning_summary_text.done', { item_id: reasoning.id, output_index: reasoningIndex, summary_index: 0, text: reasoning.summary[0].text });
      await emit('response.reasoning_summary_part.done', { item_id: reasoning.id, output_index: reasoningIndex, summary_index: 0, part: reasoning.summary[0] });
      await emit('response.output_item.done', { output_index: reasoningIndex, item: reasoning });
    }
    if (message) {
      message.status = finish === 'length' ? 'incomplete' : 'completed';
      await emit('response.output_text.done', { item_id: message.id, output_index: messageIndex, content_index: 0, text: message.content[0].text });
      await emit('response.content_part.done', { item_id: message.id, output_index: messageIndex, content_index: 0, part: message.content[0] });
      await emit('response.output_item.done', { output_index: messageIndex, item: message });
    }
    for (const call of calls.values()) {
      // Never expose a partial tool call as an executable output item.
      if (finish === 'length') {
        response.output = response.output.filter((item: Obj) => item !== call.item);
        continue;
      }
      if (!call.item.name) throw new Error('backend tool call has no name');
      if (!payload.tools?.some((tool: Obj) => tool.function.name === call.item.name)) throw new Error('backend called an unknown tool');
      const backendName = call.item.name;
      if (backendName.includes('.')) {
        const dot = backendName.indexOf('.');
        call.item.namespace = backendName.slice(0, dot);
        call.item.name = backendName.slice(dot + 1);
      }
      call.item.status = 'completed';
      if (custom.has(backendName)) {
        let value: Obj;
        try { value = JSON.parse(call.raw); } catch { throw new Error('backend custom tool arguments are not JSON'); }
        call.item.type = 'custom_tool_call';
        call.item.input = requireString(value.input, 'custom tool input');
        delete call.item.arguments;
        await emit('response.output_item.added', { output_index: call.index, item: { ...call.item, status: 'in_progress', input: '' } });
        await emit('response.custom_tool_call_input.delta', { item_id: call.item.id, output_index: call.index, delta: call.item.input });
        await emit('response.custom_tool_call_input.done', { item_id: call.item.id, output_index: call.index, input: call.item.input });
      } else {
        await emit('response.output_item.added', { output_index: call.index, item: { ...call.item, status: 'in_progress' } });
        await emit('response.function_call_arguments.delta', { item_id: call.item.id, output_index: call.index, delta: call.raw });
        call.item.arguments = call.raw;
        JSON.parse(call.raw);
        await emit('response.function_call_arguments.done', { item_id: call.item.id, output_index: call.index, arguments: call.raw });
      }
      await emit('response.output_item.done', { output_index: call.index, item: call.item });
    }
    response.status = finish === 'length' ? 'incomplete' : 'completed';
    if (finish === 'length') response.incomplete_details = { reason: 'max_output_tokens' };
    await emit(`response.${response.status}`, { response });
    if (!streaming) res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(streaming ? undefined : JSON.stringify(response));
  } catch (error) {
    if (res.destroyed) { response.status = 'cancelled'; return; }
    const message = error instanceof Error ? error.message : String(error);
    response.status = 'failed';
    response.error = { code: error instanceof RequestError ? 'invalid_request_error' : 'backend_error', message };
    if (res.headersSent) {
      await emit('response.failed', { response }).catch(() => {});
      res.end();
    } else {
      res.writeHead(error instanceof RequestError ? 400 : 502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: { type: response.error.code, message } }));
    }
  } finally {
    clearTimeout(timeout);
    if (heartbeat) clearInterval(heartbeat);
    abort.abort();
    res.off('close', disconnected);
    if (process.env.PULSE_RESPONSES_METRICS_FILE) {
      await appendFile(process.env.PULSE_RESPONSES_METRICS_FILE, JSON.stringify({
        response_id: response.id, status: response.status, model: response.model,
        first_backend_token_ms: firstTokenMs, first_visible_delta_ms: firstVisibleMs,
        elapsed_ms: performance.now() - started, usage: response.usage, backend_timings: backendTimings, jev_decision: jevDecision,
      }) + '\n', { mode: 0o600 }).catch(() => console.error('Could not write Responses metrics'));
    }
  }
}
