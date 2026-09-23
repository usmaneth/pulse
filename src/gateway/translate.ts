// Translation between the OpenAI Responses API (what Codex CLI sends) and the
// OpenAI Chat Completions API (what vLLM and llama.cpp serve).
//
// The rules here follow the router that already serves Codex in production
// (~/.codex/model-router.py): responses_to_chat, responses_to_qwen_chat,
// flatten_tools, chat_to_response and _stream_bonsai. Where this module does
// something different, a comment says why.
//
// Everything in this file is pure. It does no I/O, so the tests can drive it
// directly.

import { randomUUID } from 'node:crypto';

export type Obj = Record<string, any>;

export class RequestError extends Error {}

/** Model-specific request shaping. One profile per backend family. */
export type ProfileName = 'qwen38' | 'llamacpp';

export interface TranslateOptions {
  profile: ProfileName;
  /** The model name that the backend serves (vLLM --served-model-name). */
  upstreamModel: string;
  /** Cap for one tool output in characters. 0 disables the cap. */
  maxToolOutputChars: number;
}

/**
 * Tool-name bookkeeping for one request. The backend sees only flat function
 * tools. The response translator uses this map to restore the Codex names.
 */
export interface ToolMap {
  /** Names of Codex custom (freeform) tools, for example apply_patch. */
  custom: Set<string>;
  /** Flat backend name -> [namespace, tool name]. */
  namespaced: Map<string, [string, string]>;
}

export interface ChatRequest {
  payload: Obj;
  tools: ToolMap;
}

/** Codex effort levels that turn Qwen thinking off. Same set as the router. */
export const QWEN_NO_THINK_EFFORTS = new Set(['none', 'minimal', 'low']);

/** The router appends this sentence to every custom tool description. */
export const CUSTOM_TOOL_HINT = " Pass the full freeform input as the 'input' string argument.";

/** Namespace tools become `<namespace>__<name>`, as in the router. */
export const NAMESPACE_SEPARATOR = '__';

export const newId = (prefix: string, length = 24) =>
  `${prefix}_${randomUUID().replaceAll('-', '').slice(0, length)}`;

const CALL_TYPES = new Set(['function_call', 'custom_tool_call']);
const OUTPUT_TYPES = new Set(['function_call_output', 'custom_tool_call_output']);
const TEXT_PARTS = new Set(['input_text', 'output_text', 'text']);

/** Join the text parts of a Responses message. Image parts are dropped. */
export function messageText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((part) => part && typeof part === 'object' && TEXT_PARTS.has(part.type))
    .map((part) => (typeof part.text === 'string' ? part.text : ''))
    .join('');
}

/**
 * Convert a tool output to plain text and cap its length. The cap keeps the
 * head and the tail, because the error at the end of a build log matters as
 * much as the command at the start.
 */
export function normalizeToolOutput(output: unknown, maxChars: number): string {
  let text: string;
  if (typeof output === 'string') text = output;
  else if (Array.isArray(output)) text = messageText(output);
  else text = output == null ? '' : String(output);
  if (maxChars > 0 && text.length > maxChars) {
    const half = Math.floor(maxChars / 2);
    text = text.slice(0, half) +
      `\n\n... [output truncated: ${text.length} chars total, showing first ${half} and last ${half} chars] ...\n\n` +
      text.slice(-half);
  }
  return text;
}

/**
 * Drop tool calls that have no output and outputs that have no call. An
 * interrupted Codex turn leaves such items, and chat templates reject them.
 */
export function sanitizePairs(items: Obj[]): Obj[] {
  const outputs = new Set(items.filter((i) => OUTPUT_TYPES.has(i?.type)).map((i) => i.call_id));
  const calls = new Set(items.filter((i) => CALL_TYPES.has(i?.type)).map((i) => i.call_id ?? i.id));
  return items.filter((i) => {
    if (CALL_TYPES.has(i?.type)) return outputs.has(i.call_id ?? i.id);
    if (OUTPUT_TYPES.has(i?.type)) return calls.has(i.call_id);
    return true;
  });
}

/**
 * Flatten Codex tools into chat-completions function tools.
 *
 * - function: kept.
 * - namespace: each nested function becomes `<namespace>__<name>`.
 * - custom (freeform, for example apply_patch): becomes a function with one
 *   string argument `input`. The response translator unwraps it again.
 * - tool_search: dropped. Codex answers a tool_search call only in its own
 *   Responses item shape, and rejects a plain function call with "tool_search
 *   handler received unsupported payload". The router drops it for the same
 *   reason on its Grok path.
 * - other types (web_search, local_shell, image_generation): dropped. The
 *   backend cannot run them.
 */
export function flattenTools(tools: unknown): { tools: Obj[]; map: ToolMap } {
  const map: ToolMap = { custom: new Set(), namespaced: new Map() };
  const out: Obj[] = [];
  const fn = (name: string, description: string, parameters: Obj | undefined) => out.push({
    type: 'function',
    function: { name, description, parameters: parameters ?? { type: 'object', properties: {} } },
  });
  if (!Array.isArray(tools)) return { tools: out, map };
  for (const tool of tools) {
    if (!tool || typeof tool !== 'object') continue;
    if (tool.type === 'function' && typeof tool.name === 'string') {
      fn(tool.name, tool.description ?? '', tool.parameters);
    } else if (tool.type === 'namespace' && Array.isArray(tool.tools)) {
      const ns = String(tool.name ?? '');
      for (const sub of tool.tools) {
        if (sub?.type !== 'function' || typeof sub.name !== 'string') continue;
        const flat = `${ns}${NAMESPACE_SEPARATOR}${sub.name}`;
        map.namespaced.set(flat, [ns, sub.name]);
        fn(flat, sub.description ?? '', sub.parameters);
      }
    } else if (tool.type === 'custom' && typeof tool.name === 'string') {
      map.custom.add(tool.name);
      fn(tool.name, (tool.description ?? '') + CUSTOM_TOOL_HINT, {
        type: 'object', properties: { input: { type: 'string' } }, required: ['input'],
      });
    }
  }
  return { tools: out, map };
}

/** Apply the reasoning-effort rules for one profile to a chat payload. */
export function applyReasoning(payload: Obj, profile: ProfileName, effort: unknown): void {
  if (typeof effort !== 'string' || !effort) return;
  const value = effort.toLowerCase();
  if (profile === 'qwen38') {
    // none/minimal/low: thinking off. Tool loops at these levels do not need a
    // reasoning block, and the block costs decode time on every turn.
    // medium/high/xhigh: thinking on. The server chat template maps the effort
    // aliases to its own levels, so the value goes through unchanged.
    if (QWEN_NO_THINK_EFFORTS.has(value)) {
      payload.chat_template_kwargs = { ...payload.chat_template_kwargs, enable_thinking: false };
    } else {
      payload.chat_template_kwargs = { ...payload.chat_template_kwargs, enable_thinking: true };
      payload.reasoning_effort = value;
    }
    return;
  }
  // llama.cpp: pass the hint through. The template decides what it means.
  payload.reasoning_effort = value;
}

/** Translate a stateless, full-history Responses request to Chat Completions. */
export function responsesToChat(request: unknown, options: TranslateOptions): ChatRequest {
  if (!request || typeof request !== 'object' || Array.isArray(request)) {
    throw new RequestError('request must be a JSON object');
  }
  const body = request as Obj;
  if (body.previous_response_id) {
    throw new RequestError('previous_response_id is not supported; send the full input history');
  }
  const messages: Obj[] = [];
  if (typeof body.instructions === 'string' && body.instructions) {
    messages.push({ role: 'system', content: body.instructions });
  }
  if (typeof body.input === 'string') {
    messages.push({ role: 'user', content: body.input });
  } else if (Array.isArray(body.input)) {
    const flatName = (item: Obj) => {
      const name = String(item.name ?? '');
      return item.namespace ? `${item.namespace}${NAMESPACE_SEPARATOR}${name}` : name;
    };
    for (const item of sanitizePairs(body.input.filter((i: unknown) => i && typeof i === 'object'))) {
      const type = item.type;
      if (type === 'message' || (type === undefined && 'role' in item)) {
        let role = item.role ?? 'user';
        if (role === 'developer') role = 'system';
        if (!['system', 'user', 'assistant'].includes(role)) continue;
        const text = messageText(item.content);
        // Codex adds a plugin advertisement block. The router drops it; it
        // only adds prompt tokens for a model that cannot use the plugins.
        if (text.includes('<recommended_plugins>')) continue;
        messages.push({ role, content: text });
      } else if (CALL_TYPES.has(type)) {
        const args = typeof item.arguments === 'string'
          ? item.arguments
          : JSON.stringify({ input: typeof item.input === 'string' ? item.input : '' });
        const call = { id: item.call_id ?? item.id, type: 'function', function: { name: flatName(item), arguments: args } };
        // Parallel calls from one model turn arrive as consecutive items.
        // Merge them into one assistant message, which is the shape the
        // model produced and the shape the chat template expects.
        const last = messages.at(-1);
        if (last?.role === 'assistant' && Array.isArray(last.tool_calls) && last.content == null) {
          last.tool_calls.push(call);
        } else {
          messages.push({ role: 'assistant', content: null, tool_calls: [call] });
        }
      } else if (OUTPUT_TYPES.has(type)) {
        messages.push({
          role: 'tool',
          tool_call_id: item.call_id,
          content: normalizeToolOutput(item.output, options.maxToolOutputChars),
        });
      }
      // Other items (reasoning, web_search_call, compaction markers) carry
      // nothing that a chat template can use. They are dropped, as in the router.
    }
  } else if (body.input != null) {
    throw new RequestError('input must be a string or an array');
  }

  // Chat templates accept one system message at the start.
  const system = messages.filter((m) => m.role === 'system').map((m) => m.content).filter(Boolean);
  const rest = messages.filter((m) => m.role !== 'system');
  if (system.length) rest.unshift({ role: 'system', content: system.join('\n\n') });

  const payload: Obj = {
    model: options.upstreamModel,
    messages: rest.length ? rest : [{ role: 'user', content: ' ' }],
    stream: true,
    stream_options: { include_usage: true },
  };
  const { tools, map } = flattenTools(body.tools);
  if (tools.length) payload.tools = tools;
  if (typeof body.max_output_tokens === 'number') payload.max_tokens = body.max_output_tokens;
  for (const key of ['temperature', 'top_p']) {
    if (typeof body[key] === 'number') payload[key] = body[key];
  }
  applyReasoning(payload, options.profile, body.reasoning?.effort);
  if (options.profile === 'llamacpp') {
    // Codex re-sends a growing context every turn. llama.cpp reuses the
    // cached prefix only when this flag is set.
    payload.cache_prompt = true;
  }
  return { payload, tools: map };
}

// ---------------------------------------------------------------- responses

export interface ResponseEvent {
  type: string;
  [key: string]: unknown;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_tokens_details: { cached_tokens: number };
  output_tokens_details: { reasoning_tokens: number };
}

export function chatUsageToResponses(usage: Obj): Usage {
  const input = Number(usage.prompt_tokens ?? 0);
  const output = Number(usage.completion_tokens ?? 0);
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: Number(usage.total_tokens ?? input + output),
    input_tokens_details: { cached_tokens: Number(usage.prompt_tokens_details?.cached_tokens ?? 0) },
    output_tokens_details: { reasoning_tokens: Number(usage.completion_tokens_details?.reasoning_tokens ?? 0) },
  };
}

/** Restore the Codex name of a backend tool call. */
export function codexToolName(rawName: string, map: ToolMap): { name: string; namespace?: string; custom: boolean } {
  const ns = map.namespaced.get(rawName);
  const name = ns ? ns[1] : rawName;
  return { name, namespace: ns?.[0], custom: map.custom.has(rawName) || map.custom.has(name) };
}

/** Unwrap the `{"input": ...}` wrapper of a custom tool call. */
export function customToolInput(args: string): string {
  try {
    const value = JSON.parse(args);
    if (value && typeof value === 'object' && typeof value.input === 'string') return value.input;
  } catch {
    // The router falls back to the raw text. A model that writes the patch
    // without the JSON wrapper still produces a usable call.
  }
  return args;
}

interface PendingCall {
  id?: string;
  name: string;
  args: string;
}

/**
 * Turn a stream of chat-completions chunks into Responses stream events.
 *
 * Text and reasoning go out as deltas when they arrive. Tool calls go out
 * complete after the backend finishes, because the arguments of a custom tool
 * are JSON that must be unwrapped as a whole. This matches the router and the
 * Responses event order that Codex expects:
 *
 *   response.created, response.in_progress,
 *   [reasoning item events], [message item events], [tool call item events],
 *   response.completed | response.incomplete | response.failed
 */
export class ChatStreamTranslator {
  readonly response: Obj;
  private sequence = 0;
  private reasoning: { item: Obj; index: number; open: boolean } | null = null;
  private message: { item: Obj; index: number } | null = null;
  private readonly calls = new Map<number, PendingCall>();
  private finish: string | null = null;
  private sawVisibleOutput = false;

  constructor(model: string, private readonly tools: ToolMap, private readonly emitReasoning = true) {
    this.response = {
      id: newId('resp'),
      object: 'response',
      created_at: Math.floor(Date.now() / 1000),
      status: 'in_progress',
      model,
      output: [],
      error: null,
      incomplete_details: null,
      usage: null,
      store: false,
    };
  }

  get finishReason(): string | null { return this.finish; }
  get hasOutput(): boolean { return this.sawVisibleOutput; }

  private event(type: string, fields: Obj): ResponseEvent {
    return { type, sequence_number: this.sequence++, ...fields };
  }

  private snapshot(): Obj {
    return { ...this.response, output: [] };
  }

  start(): ResponseEvent[] {
    return [
      this.event('response.created', { response: this.snapshot() }),
      this.event('response.in_progress', { response: this.snapshot() }),
    ];
  }

  /** Feed one parsed chat-completions chunk. Returns the events to send. */
  push(chunk: Obj): ResponseEvent[] {
    const events: ResponseEvent[] = [];
    if (chunk.error) {
      throw new Error(`backend stream error: ${chunk.error.message ?? JSON.stringify(chunk.error)}`);
    }
    if (chunk.usage) this.response.usage = chatUsageToResponses(chunk.usage);
    const choice = chunk.choices?.[0];
    if (!choice) return events;
    const delta = choice.delta ?? {};
    // vLLM names the field `reasoning_content` in older builds and
    // `reasoning` in newer ones. llama.cpp uses `reasoning_content`.
    const thought = delta.reasoning_content ?? delta.reasoning;
    if (typeof thought === 'string' && thought) {
      this.sawVisibleOutput = true;
      if (this.emitReasoning) events.push(...this.reasoningDelta(thought));
    }
    if (typeof delta.content === 'string' && delta.content) {
      this.sawVisibleOutput = true;
      events.push(...this.closeReasoning());
      events.push(...this.textDelta(delta.content));
    }
    if (Array.isArray(delta.tool_calls)) {
      events.push(...this.closeReasoning());
      for (const tc of delta.tool_calls) {
        const slot = typeof tc.index === 'number' ? tc.index : 0;
        let call = this.calls.get(slot);
        if (!call) {
          call = { name: '', args: '' };
          this.calls.set(slot, call);
        }
        this.sawVisibleOutput = true;
        if (tc.id && !call.id) call.id = tc.id;
        // Some servers repeat the full name in every chunk, others send it
        // once. Keep the first non-empty name, as the router does.
        if (tc.function?.name && !call.name) call.name = tc.function.name;
        if (typeof tc.function?.arguments === 'string') call.args += tc.function.arguments;
      }
    }
    if (choice.finish_reason) this.finish = choice.finish_reason;
    return events;
  }

  private reasoningDelta(text: string): ResponseEvent[] {
    const events: ResponseEvent[] = [];
    if (!this.reasoning || !this.reasoning.open) {
      const item = { id: newId('rs'), type: 'reasoning', summary: [{ type: 'summary_text', text: '' }] };
      const index = this.response.output.push(item) - 1;
      this.reasoning = { item, index, open: true };
      events.push(this.event('response.output_item.added', { output_index: index, item: { ...item, summary: [] } }));
      events.push(this.event('response.reasoning_summary_part.added', {
        item_id: item.id, output_index: index, summary_index: 0, part: { type: 'summary_text', text: '' },
      }));
    }
    const r = this.reasoning;
    r.item.summary[0].text += text;
    events.push(this.event('response.reasoning_summary_text.delta', {
      item_id: r.item.id, output_index: r.index, summary_index: 0, delta: text,
    }));
    return events;
  }

  private closeReasoning(): ResponseEvent[] {
    const r = this.reasoning;
    if (!r || !r.open) return [];
    r.open = false;
    const part = r.item.summary[0];
    return [
      this.event('response.reasoning_summary_text.done', { item_id: r.item.id, output_index: r.index, summary_index: 0, text: part.text }),
      this.event('response.reasoning_summary_part.done', { item_id: r.item.id, output_index: r.index, summary_index: 0, part: { ...part } }),
      this.event('response.output_item.done', { output_index: r.index, item: r.item }),
    ];
  }

  private textDelta(text: string): ResponseEvent[] {
    const events: ResponseEvent[] = [];
    if (!this.message) {
      const item = { id: newId('msg', 16), type: 'message', role: 'assistant', status: 'in_progress', content: [] as Obj[] };
      const index = this.response.output.push(item) - 1;
      this.message = { item, index };
      events.push(this.event('response.output_item.added', { output_index: index, item: { ...item, content: [] } }));
      item.content.push({ type: 'output_text', text: '', annotations: [] });
      events.push(this.event('response.content_part.added', {
        item_id: item.id, output_index: index, content_index: 0, part: { type: 'output_text', text: '', annotations: [] },
      }));
    }
    const m = this.message;
    m.item.content[0].text += text;
    events.push(this.event('response.output_text.delta', {
      item_id: m.item.id, output_index: m.index, content_index: 0, delta: text,
    }));
    return events;
  }

  /**
   * Close all open items and emit the terminal event. Call this after the
   * backend sends `[DONE]`. A stream that ends without a finish reason is a
   * truncated stream, and it fails.
   */
  finishStream(): ResponseEvent[] {
    if (!this.finish) return this.fail('backend stream ended before a finish reason');
    const events: ResponseEvent[] = [...this.closeReasoning()];
    const truncated = this.finish === 'length';
    if (this.message) {
      const m = this.message;
      m.item.status = truncated ? 'incomplete' : 'completed';
      const part = m.item.content[0];
      events.push(this.event('response.output_text.done', { item_id: m.item.id, output_index: m.index, content_index: 0, text: part.text }));
      events.push(this.event('response.content_part.done', { item_id: m.item.id, output_index: m.index, content_index: 0, part: { ...part } }));
      events.push(this.event('response.output_item.done', { output_index: m.index, item: m.item }));
    }
    // A call cut off by the token limit has partial arguments. Codex would
    // run it, so it is dropped instead.
    if (!truncated) {
      for (const slot of [...this.calls.keys()].sort((a, b) => a - b)) {
        const call = this.calls.get(slot)!;
        if (!call.name) continue;
        events.push(...this.emitCall(call));
      }
    }
    this.response.status = truncated ? 'incomplete' : 'completed';
    if (truncated) this.response.incomplete_details = { reason: 'max_output_tokens' };
    events.push(this.event(`response.${this.response.status}`, { response: this.response }));
    return events;
  }

  private emitCall(call: PendingCall): ResponseEvent[] {
    const { name, namespace, custom } = codexToolName(call.name, this.tools);
    const callId = call.id ?? newId('call', 16);
    const args = call.args || '{}';
    const events: ResponseEvent[] = [];
    if (custom) {
      const input = customToolInput(args);
      const item: Obj = { id: newId('ctc', 16), type: 'custom_tool_call', status: 'completed', call_id: callId, name, input };
      const index = this.response.output.push(item) - 1;
      events.push(this.event('response.output_item.added', { output_index: index, item: { ...item, status: 'in_progress', input: '' } }));
      events.push(this.event('response.custom_tool_call_input.delta', { item_id: item.id, output_index: index, delta: input }));
      events.push(this.event('response.custom_tool_call_input.done', { item_id: item.id, output_index: index, input }));
      events.push(this.event('response.output_item.done', { output_index: index, item }));
    } else {
      const item: Obj = { id: newId('fc', 16), type: 'function_call', status: 'completed', call_id: callId, name, arguments: args };
      if (namespace) item.namespace = namespace;
      const index = this.response.output.push(item) - 1;
      events.push(this.event('response.output_item.added', { output_index: index, item: { ...item, status: 'in_progress', arguments: '' } }));
      events.push(this.event('response.function_call_arguments.delta', { item_id: item.id, output_index: index, delta: args }));
      events.push(this.event('response.function_call_arguments.done', { item_id: item.id, output_index: index, arguments: args }));
      events.push(this.event('response.output_item.done', { output_index: index, item }));
    }
    return events;
  }

  /** Emit `response.failed`. The response keeps the output that arrived. */
  fail(message: string, code = 'backend_error'): ResponseEvent[] {
    this.response.status = 'failed';
    this.response.error = { code, message };
    return [this.event('response.failed', { response: this.response })];
  }
}

/** Format one Responses event as an SSE frame. */
export function sseFrame(event: ResponseEvent): string {
  return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
}
