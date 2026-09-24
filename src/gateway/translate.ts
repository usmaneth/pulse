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

/** The backend sent an error chunk in its stream. A retry gets the same error. */
export class BackendStreamError extends Error {}

/** Model-specific request shaping. One profile per backend family. */
export type ProfileName = 'qwen38' | 'llamacpp';

export interface TranslateOptions {
  profile: ProfileName;
  /** The model name that the backend serves (vLLM --served-model-name). */
  upstreamModel: string;
  /** Cap for one tool output in characters. 0 disables the cap. */
  maxToolOutputChars: number;
  /** Send the text of earlier reasoning items back to the model. Default true. */
  replayReasoning?: boolean;
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

const PLUGIN_BLOCK = /<recommended_plugins>[\s\S]*?<\/recommended_plugins>\n*/g;

/**
 * Remove the plugin advertisement block of Codex from a message text. The
 * block only adds prompt tokens for a model that cannot use the plugins.
 * Codex sends it as one part of the user message that also holds the AGENTS.md
 * instructions and the environment context (cwd, shell, date). Those parts
 * stay, because without the cwd the model searches the whole disk for files.
 */
export function dropPluginBlock(text: string): string {
  return text.replace(PLUGIN_BLOCK, '');
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
      fn(tool.name, (tool.description ?? '') + CUSTOM_TOOL_HINT + grammarHint(tool.format), {
        type: 'object', properties: { input: { type: 'string' } }, required: ['input'],
      });
    }
  }
  return { tools: out, map };
}

/**
 * Text that gives the model the grammar of a freeform tool. Codex sends the
 * Lark grammar of apply_patch in `format`. A function tool cannot carry a
 * grammar, so the grammar goes into the description instead.
 */
export function grammarHint(format: unknown): string {
  if (!format || typeof format !== 'object') return '';
  const f = format as Obj;
  if (f.type !== 'grammar' || typeof f.definition !== 'string' || !f.definition.trim()) return '';
  const syntax = typeof f.syntax === 'string' && f.syntax ? `${f.syntax} ` : '';
  return `\nThe input string must follow this ${syntax}grammar:\n${f.definition.trim()}`;
}

/**
 * Map a Responses tool_choice to Chat Completions. Named function and custom
 * tools become a named function. Namespaced tools use the flat name.
 */
export function mapToolChoice(choice: unknown, available: Set<string>): unknown {
  if (typeof choice === 'string') {
    if (['auto', 'none', 'required'].includes(choice)) return choice;
    throw new RequestError(`unsupported tool_choice: ${choice}`);
  }
  if (!choice || typeof choice !== 'object') throw new RequestError('tool_choice must be a string or an object');
  const c = choice as Obj;
  if ((c.type !== 'function' && c.type !== 'custom') || typeof c.name !== 'string') {
    throw new RequestError(`unsupported tool_choice type: ${String(c.type)}`);
  }
  const name = c.namespace ? `${c.namespace}${NAMESPACE_SEPARATOR}${c.name}` : c.name;
  if (!available.has(name)) throw new RequestError(`tool_choice names a tool that is not in tools: ${name}`);
  return { type: 'function', function: { name } };
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

/**
 * The system prompt text for a structured `text.format` (`codex exec
 * --output-schema` sends `json_schema`). vLLM can force a format with
 * `response_format`, but the constraint applies to every answer. Codex sends
 * the format on each request of the task, so tool calls would become
 * impossible. The model gets the schema as an instruction instead. The
 * gateway does not check the answer against the schema.
 */
export function outputFormatNote(format: unknown): string {
  if (format == null) return '';
  if (typeof format !== 'object' || Array.isArray(format)) throw new RequestError('text.format must be an object');
  const f = format as Obj;
  if (f.type === 'text') return '';
  const bare = 'The final answer must be one JSON object, with no other text and no code fence.';
  if (f.type === 'json_object') return bare;
  if (f.type === 'json_schema') {
    if (!f.schema || typeof f.schema !== 'object' || Array.isArray(f.schema)) {
      throw new RequestError('text.format json_schema needs a schema object');
    }
    return 'The final answer must be one JSON object that matches this JSON schema, with no other text and no code fence:\n' +
      JSON.stringify(f.schema);
  }
  throw new RequestError(`text.format ${String(f.type)} is not supported; use text, json_schema or json_object`);
}

/** True when the text is a JSON object, which is what a function call needs as arguments. */
export function isJsonObjectText(text: unknown): boolean {
  if (typeof text !== 'string') return false;
  try {
    const value = JSON.parse(text);
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  } catch {
    return false;
  }
}

/** Top-level request fields that need server-side state. */
const STATEFUL_KEYS = ['previous_response_id', 'conversation', 'background'];

/** Join the summary text of a reasoning item. */
export function reasoningText(item: Obj): string {
  const parts = (list: unknown) => Array.isArray(list)
    ? list.filter((p) => p && typeof p.text === 'string').map((p) => p.text as string)
    : [];
  const summary = parts(item.summary);
  // The summary is what this gateway emits. `content` (reasoning_text) is the
  // fallback for items from other servers.
  return (summary.length ? summary : parts(item.content)).join('\n\n');
}

/**
 * Merge consecutive assistant messages into one turn. Codex sends one model
 * turn as separate items (reasoning, message, tool calls). The chat template
 * expects them in one assistant message, in the order the model wrote them.
 */
export function mergeAssistantTurns(messages: Obj[]): Obj[] {
  const out: Obj[] = [];
  for (const message of messages) {
    const last = out.at(-1);
    if (message.role !== 'assistant' || last?.role !== 'assistant') {
      out.push(message);
      continue;
    }
    // Reasoning that follows text or tool calls starts a new model turn.
    if (message.reasoning_content && (last.content != null || last.tool_calls)) {
      out.push(message);
      continue;
    }
    if (message.reasoning_content) last.reasoning_content = (last.reasoning_content ?? '') + message.reasoning_content;
    if (message.content != null) {
      // Text after tool calls also starts a new model turn.
      if (last.tool_calls) { out.push(message); continue; }
      last.content = (last.content ?? '') + message.content;
    }
    if (message.tool_calls) last.tool_calls = [...(last.tool_calls ?? []), ...message.tool_calls];
  }
  return out;
}

/** Translate a stateless, full-history Responses request to Chat Completions. */
export function responsesToChat(request: unknown, options: TranslateOptions): ChatRequest {
  if (!request || typeof request !== 'object' || Array.isArray(request)) {
    throw new RequestError('request must be a JSON object');
  }
  const body = request as Obj;
  for (const key of STATEFUL_KEYS) {
    if (body[key]) throw new RequestError(`${key} is not supported; send the full input history`);
  }
  if (body.store === true) throw new RequestError('store=true is not supported; the gateway keeps no state, so set store=false');
  const formatNote = outputFormatNote(body.text?.format);
  const replayReasoning = options.replayReasoning ?? true;
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
        if (!['system', 'user', 'assistant'].includes(role)) {
          throw new RequestError(`unsupported message role: ${String(item.role)}`);
        }
        let text = messageText(item.content);
        if (text.includes('<recommended_plugins>')) {
          text = dropPluginBlock(text);
          if (!text.trim()) continue;
        }
        messages.push({ role, content: text });
      } else if (type === 'reasoning') {
        // The Qwen3.8 template renders each assistant turn as
        // `<think>reasoning</think>content`. With the reasoning replayed, the
        // history is the same text that the model wrote, so the model sees
        // its own plan in a tool loop. Items without text (for example
        // encrypted reasoning from another provider) carry nothing usable.
        const text = replayReasoning ? reasoningText(item) : '';
        if (text) messages.push({ role: 'assistant', content: null, reasoning_content: text });
      } else if (CALL_TYPES.has(type)) {
        // vLLM and llama.cpp parse the arguments of each earlier call to render
        // the template, and refuse the request when they are not a JSON
        // object. One call that the stream cut short (for example
        // '{"cmd": "ls') would then fail every later request of the session.
        // Codex already gave the model the parse error as the call output, so
        // the call goes back with empty arguments. A valid object keeps its
        // bytes, so the prompt prefix does not change.
        const args = typeof item.arguments === 'string'
          ? (isJsonObjectText(item.arguments) ? item.arguments : '{}')
          : JSON.stringify({ input: typeof item.input === 'string' ? item.input : '' });
        const call = { id: item.call_id ?? item.id, type: 'function', function: { name: flatName(item), arguments: args } };
        messages.push({ role: 'assistant', content: null, tool_calls: [call] });
      } else if (OUTPUT_TYPES.has(type)) {
        messages.push({
          role: 'tool',
          tool_call_id: item.call_id,
          content: normalizeToolOutput(item.output, options.maxToolOutputChars),
        });
      } else {
        throw new RequestError(`unsupported input item type: ${String(type)}`);
      }
    }
  } else if (body.input != null) {
    throw new RequestError('input must be a string or an array');
  }

  const arranged = arrangeSystem(mergeAssistantTurns(messages), options.profile);
  if (formatNote) {
    // At the end of the leading system message, the note does not change the
    // prompt prefix that a session without a format shares.
    if (arranged[0]?.role === 'system') arranged[0] = { ...arranged[0], content: `${arranged[0].content}\n\n${formatNote}` };
    else arranged.unshift({ role: 'system', content: formatNote });
  }
  const payload: Obj = {
    model: options.upstreamModel,
    messages: arranged,
    stream: true,
    stream_options: { include_usage: true },
  };
  if (!payload.messages.length) payload.messages = [{ role: 'user', content: ' ' }];
  const { tools, map } = flattenTools(body.tools);
  if (tools.length) {
    payload.tools = tools;
    if (body.tool_choice != null) {
      payload.tool_choice = mapToolChoice(body.tool_choice, new Set(tools.map((t) => t.function.name)));
    }
    if (typeof body.parallel_tool_calls === 'boolean') payload.parallel_tool_calls = body.parallel_tool_calls;
  }
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

/**
 * Put the system messages where the chat template accepts them.
 *
 * qwen38: the leading system messages merge into one. A later system message
 * (Codex sends a developer message when a setting changes during a session)
 * stays where it is. The Qwen3.8 template renders it in place. If it moved to
 * the start, the prompt prefix would change and the vLLM prefix cache would
 * miss for the full history.
 *
 * llamacpp: the Bonsai template accepts one system message at the start
 * only, so all system messages merge into it.
 */
export function arrangeSystem(messages: Obj[], profile: ProfileName): Obj[] {
  const isSystem = (m: Obj) => m.role === 'system';
  let head = 0;
  if (profile === 'qwen38') while (head < messages.length && isSystem(messages[head])) head++;
  const leading = profile === 'qwen38' ? messages.slice(0, head) : messages.filter(isSystem);
  const rest = profile === 'qwen38'
    ? messages.slice(head).filter((m) => !isSystem(m) || m.content)
    : messages.filter((m) => !isSystem(m));
  const system = leading.map((m) => m.content).filter(Boolean);
  if (system.length) rest.unshift({ role: 'system', content: system.join('\n\n') });
  return rest;
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
  /** The Responses item, once `output_item.added` went out. */
  item?: Obj;
  index?: number;
  /** True for a Codex custom (freeform) tool, for example apply_patch. */
  custom?: boolean;
}

/**
 * Turn a stream of chat-completions chunks into Responses stream events.
 *
 * Text and reasoning go out as deltas when they arrive. Each tool call gets
 * its `output_item.added` event and its place in the output when its name
 * arrives. Function-call arguments go out as deltas when they arrive. The
 * input of a custom tool call goes out after the finish reason, because the
 * backend sends it as JSON that must be unwrapped as a whole. The done events
 * of all calls wait for the finish reason, in slot order. Codex runs a call
 * when its `output_item.done` arrives, and a call that the token limit cut off
 * must not run. The event order:
 *
 *   response.created, response.in_progress,
 *   [reasoning item events], [message item events],
 *   [function call: output_item.added, function_call_arguments.delta...],
 *   [custom call: output_item.added],
 *   ...finish reason...
 *   [function call: function_call_arguments.done, output_item.done],
 *   [custom call: custom_tool_call_input.delta, .done, output_item.done],
 *   response.completed | response.incomplete | response.failed
 */
export class ChatStreamTranslator {
  readonly response: Obj;
  private sequence = 0;
  private reasoning: { item: Obj; index: number; open: boolean } | null = null;
  private message: { item: Obj; index: number; open: boolean } | null = null;
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

  /**
   * A `response.in_progress` event for a stream that has nothing else to send,
   * for example during a long prefill. Codex ignores the event, but it resets
   * the stream idle timer of Codex. An SSE comment line does not.
   */
  keepalive(): ResponseEvent {
    return this.event('response.in_progress', { response: this.snapshot() });
  }

  /**
   * Forget all backend output, to send the request to the backend again. The
   * response id and the sequence numbers continue. Use it only when no event
   * after start() went to the client.
   */
  restart(): void {
    this.response.output = [];
    this.response.usage = null;
    this.reasoning = null;
    this.message = null;
    this.calls.clear();
    this.finish = null;
    this.sawVisibleOutput = false;
  }

  /** Feed one parsed chat-completions chunk. Returns the events to send. */
  push(chunk: Obj): ResponseEvent[] {
    const events: ResponseEvent[] = [];
    if (chunk.error) {
      throw new BackendStreamError(`backend stream error: ${chunk.error.message ?? JSON.stringify(chunk.error)}`);
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
      // After the tool calls start, the tool parser can pass on white space
      // between calls. It is not part of the answer.
      if (!(this.calls.size && !delta.content.trim())) {
        this.sawVisibleOutput = true;
        events.push(...this.closeReasoning());
        events.push(...this.textDelta(delta.content));
      }
    }
    if (Array.isArray(delta.tool_calls)) {
      events.push(...this.closeReasoning());
      events.push(...this.closeMessage());
      for (const tc of delta.tool_calls) {
        const slot = typeof tc.index === 'number' ? tc.index : 0;
        let call = this.calls.get(slot);
        if (!call) {
          call = { name: '', args: '' };
          this.calls.set(slot, call);
        }
        if (tc.id && !call.id) call.id = tc.id;
        // Some servers repeat the full name in every chunk, others send it
        // once. Keep the first non-empty name, as the router does.
        if (tc.function?.name && !call.name) call.name = tc.function.name;
        const piece = typeof tc.function?.arguments === 'string' ? tc.function.arguments : '';
        call.args += piece;
        events.push(...this.streamCall(call, piece));
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
    if (!this.message || !this.message.open) {
      const item = { id: newId('msg', 16), type: 'message', role: 'assistant', status: 'in_progress', content: [] as Obj[] };
      const index = this.response.output.push(item) - 1;
      this.message = { item, index, open: true };
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
   * Close the message item. The text before a tool call is complete, so the
   * message closes when the first tool call starts, and the Responses items
   * stay in sequence.
   */
  private closeMessage(status = 'completed'): ResponseEvent[] {
    const m = this.message;
    if (!m || !m.open) return [];
    m.open = false;
    m.item.status = status;
    const part = m.item.content[0];
    return [
      this.event('response.output_text.done', { item_id: m.item.id, output_index: m.index, content_index: 0, text: part.text }),
      this.event('response.content_part.done', { item_id: m.item.id, output_index: m.index, content_index: 0, part: { ...part } }),
      this.event('response.output_item.done', { output_index: m.index, item: m.item }),
    ];
  }

  /**
   * Open the item of a call when its name is known. The item takes its place
   * in the output then, for a function call and for a custom call, so the
   * output keeps the order in which the model made the calls. A function call
   * also sends the new argument text as a delta.
   */
  private streamCall(call: PendingCall, piece: string): ResponseEvent[] {
    if (!call.name) return [];
    if (!call.item) {
      const { name, namespace, custom } = codexToolName(call.name, this.tools);
      call.id ??= newId('call', 16);
      call.custom = custom;
      const item: Obj = custom
        ? { id: newId('ctc', 16), type: 'custom_tool_call', status: 'in_progress', call_id: call.id, name, input: '' }
        : { id: newId('fc', 16), type: 'function_call', status: 'in_progress', call_id: call.id, name, arguments: '' };
      if (namespace && !custom) item.namespace = namespace;
      call.item = item;
      call.index = this.response.output.push(item) - 1;
      this.sawVisibleOutput = true;
      const events = [this.event('response.output_item.added', { output_index: call.index, item: { ...item } })];
      // Arguments that arrived before the name go out now, in one delta.
      if (call.args && !custom) events.push(this.argsDelta(call, call.args));
      return events;
    }
    return piece && !call.custom ? [this.argsDelta(call, piece)] : [];
  }

  /**
   * Send argument text of a function call. The item keeps the text that went
   * out, so a `response.failed` snapshot agrees with the deltas.
   */
  private argsDelta(call: PendingCall, delta: string): ResponseEvent {
    call.item!.arguments = call.args;
    return this.event('response.function_call_arguments.delta', { item_id: call.item!.id, output_index: call.index, delta });
  }

  /**
   * Close all open items and emit the terminal event. Call this after the
   * backend sends `[DONE]`. A stream that ends without a finish reason is a
   * truncated stream, and it fails.
   */
  finishStream(): ResponseEvent[] {
    if (!this.finish) return this.fail('backend stream ended before a finish reason');
    if (!this.sawVisibleOutput && this.finish !== 'length') {
      // vLLM can generate tokens and stream none of them (seen live: 156
      // output tokens and no delta). A completed response without output
      // ends the Codex task with no answer. A failed one tells the user, and
      // Codex can retry the request.
      const tokens = this.response.usage?.output_tokens;
      return this.fail(`the backend finished (${this.finish}) without any text, reasoning or tool call` +
        (tokens ? ` after ${tokens} output tokens` : ''));
    }
    const events: ResponseEvent[] = [...this.closeReasoning()];
    const truncated = this.finish === 'length';
    events.push(...this.closeMessage(truncated ? 'incomplete' : 'completed'));
    const slots = [...this.calls.keys()].sort((a, b) => a - b).map((slot) => this.calls.get(slot)!);
    if (truncated) {
      // A call cut off by the token limit has partial arguments. Codex would
      // run it, so it gets no done event and leaves the final output. The
      // other calls of the turn go too: Codex retries an incomplete turn, and
      // a call that already ran would run again.
      const open = new Set(slots.map((call) => call.item).filter(Boolean));
      this.response.output = this.response.output.filter((item: Obj) => !open.has(item));
    } else {
      // A call without a name has no item. It is not in the output.
      for (const call of slots) {
        if (call.item) events.push(...(call.custom ? this.closeCustomCall(call) : this.closeCall(call)));
      }
    }
    this.response.status = truncated ? 'incomplete' : 'completed';
    if (truncated) this.response.incomplete_details = { reason: 'max_output_tokens' };
    events.push(this.event(`response.${this.response.status}`, { response: this.response }));
    return events;
  }

  private closeCall(call: PendingCall): ResponseEvent[] {
    const item = call.item!;
    const events: ResponseEvent[] = [];
    if (!call.args) {
      call.args = '{}';
      events.push(this.argsDelta(call, call.args));
    }
    item.arguments = call.args;
    item.status = 'completed';
    events.push(this.event('response.function_call_arguments.done', { item_id: item.id, output_index: call.index, arguments: call.args }));
    events.push(this.event('response.output_item.done', { output_index: call.index, item }));
    return events;
  }

  /** Send the unwrapped input of a custom tool call and close it. */
  private closeCustomCall(call: PendingCall): ResponseEvent[] {
    const item = call.item!;
    const input = customToolInput(call.args || '{}');
    item.input = input;
    item.status = 'completed';
    return [
      this.event('response.custom_tool_call_input.delta', { item_id: item.id, output_index: call.index, delta: input }),
      this.event('response.custom_tool_call_input.done', { item_id: item.id, output_index: call.index, input }),
      this.event('response.output_item.done', { output_index: call.index, item }),
    ];
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
