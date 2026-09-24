// Prefix warmer. When a backend becomes healthy again (for example after a
// vLLM restart), the warmer sends the most recent Codex prompt prefix to it as
// a prefill-only request, so that the first real turn hits the prefix cache.
//
// A restarted vLLM server has an empty prefix cache. Without the warmer, the
// first turn of each new Codex session pays the prefill of the system prompt,
// the tools and the environment message (about 11k tokens, 5 to 6 s on one
// GB10), and the next turn of a session that was active before the restart
// pays the prefill of its full history.
//
// Rules:
// - A warm request has the same messages, tools and template arguments as the
//   real request, and max_tokens 1. The rendered prompt then starts with the
//   same tokens, and the server caches its full blocks.
// - The warmer sends each item twice. The second request confirms the cache
//   hit, and its log line shows the cached token count. With the Mamba
//   "align" cache of vLLM, the state at the end of the first prefill chunk
//   of a request can be missing after the first pass. The second pass
//   computes it again and the server keeps it.
// - Warm requests go one at a time, and only when the gateway has no real
//   request in flight. A real request aborts the warm request in flight. The
//   server keeps the blocks that the aborted request completed.
// - The warmer skips a prefix or a session when a real request with it reached
//   the backend after the trigger, because that request fills the cache.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { open, rename, rm } from 'node:fs/promises';
import { postChat } from './backends.js';
import type { Endpoint, ModelRoute } from './backends.js';
import type { GatewayConfig } from './config.js';
import { log } from './log.js';
import { sseData } from './sse.js';
import type { Obj } from './translate.js';

export interface WarmupConfig {
  /** Warm the backends when they become healthy. */
  enabled: boolean;
  /** Also warm the last payload of this many recent sessions. 0 turns it off. */
  sessions: number;
  /** Skip a session when its last request is older than this. */
  sessionMaxAgeMs: number;
  /** Skip a payload when its JSON text is longer than this. */
  maxPayloadChars: number;
  /** Stop when the gateway is not idle for this time after the trigger. */
  idleWaitMs: number;
  /** Send each item a second time to confirm the cache hit. */
  confirm: boolean;
  /**
   * Optional JSON file for the prefixes, so that the warmer can also warm
   * after a gateway restart. The file holds the system prompt, the tools and
   * the environment message, so only the owner can read it.
   */
  stateFile?: string;
}

export function defaultWarmupConfig(): WarmupConfig {
  return {
    enabled: true,
    sessions: 1,
    sessionMaxAgeMs: 30 * 60_000,
    maxPayloadChars: 4_000_000,
    idleWaitMs: 10 * 60_000,
    confirm: true,
  };
}

/** Keep the prefixes of this many template variants (effort levels) per model. */
const MAX_VARIANTS = 4;
const PERSIST_DELAY_MS = 2_000;
const IDLE_POLL_MS = 250;

type Kind = 'prefix' | 'session';

export interface WarmEntry {
  key: string;
  kind: Kind;
  payload: Obj;
  /** Time of the last real request that this entry came from. 0 for entries from the state file. */
  at: number;
  /**
   * For each endpoint, the tick of the last real request with this prefix or
   * session that reached it.
   */
  servedAt: Map<string, number>;
}

interface Stats {
  triggers: number;
  requests: number;
  completed: number;
  aborted: number;
  errors: number;
  skipped: number;
  promptTokens: number;
  cachedTokens: number;
}

/**
 * The messages that every session of one project starts with: the system
 * message and, when Codex sends one, the environment message. Codex sends the
 * environment (cwd, shell, AGENTS.md) as the first user message and the task
 * as the next user message. Returns null when there is no system message.
 */
export function prefixMessages(messages: unknown): Obj[] | null {
  if (!Array.isArray(messages) || messages[0]?.role !== 'system') return null;
  const head: Obj[] = [messages[0]];
  if (messages[1]?.role === 'user' && messages[2]?.role === 'user') head.push(messages[1]);
  return head;
}

/**
 * The template arguments change the head of the rendered prompt (the Qwen3.8
 * template writes the effort and the thinking mode before the tools), so
 * each variant has its own cached prefix.
 */
export function variantKey(payload: Obj): string {
  return JSON.stringify([payload.chat_template_kwargs ?? null, payload.reasoning_effort ?? null]);
}

/** The payload of a warm request: prefill only, one output token. */
export function warmPayload(payload: Obj): Obj {
  const { max_tokens: _m, max_completion_tokens: _c, stream: _s, stream_options: _o, ...rest } = payload;
  return { ...rest, max_tokens: 1, stream: true, stream_options: { include_usage: true } };
}

const digest = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex').slice(0, 16);

export class Warmer {
  readonly stats: Stats = {
    triggers: 0, requests: 0, completed: 0, aborted: 0, errors: 0, skipped: 0, promptTokens: 0, cachedTokens: 0,
  };
  private readonly prefixes = new Map<string, Map<string, WarmEntry>>();
  private readonly sessions = new Map<string, Map<string, WarmEntry>>();
  private readonly running = new Set<Endpoint>();
  /** The warm requests in flight, one per endpoint that the warmer warms. */
  private readonly current = new Set<AbortController>();
  private persistTimer: ReturnType<typeof setTimeout> | null = null;
  private persistedDigest = '';
  private stopped = false;
  private last: Obj | null = null;
  /** Orders served requests and triggers. Two events in one millisecond still get an order. */
  private tick = 0;

  constructor(
    private readonly config: WarmupConfig,
    private readonly timeouts: Pick<GatewayConfig, 'connectTimeoutMs' | 'headersTimeoutMs' | 'idleTimeoutMs'>,
    /** The number of real requests in flight. */
    private readonly busy: () => number,
  ) {
    if (config.enabled && config.stateFile) this.load(config.stateFile);
  }

  /** Record the payload of a real request. Call it before the backend call. */
  observe(route: ModelRoute, payload: Obj, sessionKey?: unknown): void {
    if (!this.config.enabled) return;
    const now = Date.now();
    const head = prefixMessages(payload.messages);
    if (head) {
      const variants = this.slot(this.prefixes, route.config.id);
      const key = variantKey(payload);
      const servedAt = variants.get(key)?.servedAt ?? new Map<string, number>();
      variants.delete(key);
      variants.set(key, { key, kind: 'prefix', payload: { ...payload, messages: head }, at: now, servedAt });
      while (variants.size > MAX_VARIANTS) variants.delete(variants.keys().next().value!);
      this.schedulePersist();
    }
    if (this.config.sessions > 0 && typeof sessionKey === 'string' && sessionKey) {
      const list = this.slot(this.sessions, route.config.id);
      const servedAt = list.get(sessionKey)?.servedAt ?? new Map<string, number>();
      list.delete(sessionKey);
      list.set(sessionKey, { key: sessionKey, kind: 'session', payload, at: now, servedAt });
      while (list.size > this.config.sessions) list.delete(list.keys().next().value!);
    }
  }

  /**
   * Record that a real request reached the endpoint. Its prefix and its
   * session are then in the cache of that endpoint, so the warmer does not
   * send them to it again.
   */
  served(route: ModelRoute, endpoint: Endpoint, payload: Obj, sessionKey?: unknown): void {
    if (!this.config.enabled) return;
    const tick = ++this.tick;
    const prefix = this.prefixes.get(route.config.id)?.get(variantKey(payload));
    if (prefix) prefix.servedAt.set(endpoint.name, tick);
    if (typeof sessionKey === 'string') {
      const session = this.sessions.get(route.config.id)?.get(sessionKey);
      if (session) session.servedAt.set(endpoint.name, tick);
    }
  }

  /** Abort the warm request in flight. Call it when a real request starts. */
  yieldToRealRequest(): void {
    for (const controller of this.current) controller.abort(new Error('a real request arrived'));
  }

  /**
   * Called when an endpoint becomes healthy. `previous` is false after a
   * failure and null at the first health check of the process.
   *
   * After a failure, the server can have lost its cache, so every known item
   * is warmed. At the first check of the process, only the prefixes from the
   * state file are warmed: a request of this process that made the endpoint
   * healthy fills the cache itself.
   */
  onHealthy(route: ModelRoute, endpoint: Endpoint, previous: boolean | null): void {
    if (!this.config.enabled || this.stopped) return;
    if (previous === false) void this.warm(route, endpoint, 'recovered');
    else void this.warm(route, endpoint, 'startup', (item) => item.at === 0);
  }

  /** Warm one endpoint now. Resolves when the warmer is done or gives up. */
  async warm(route: ModelRoute, endpoint: Endpoint, reason: string, accept: (item: WarmEntry) => boolean = () => true): Promise<void> {
    if (this.running.has(endpoint)) return;
    const since = ++this.tick;
    const items = this.items(route, Date.now()).filter(accept);
    if (!items.length) return;
    this.running.add(endpoint);
    this.stats.triggers++;
    log('info', 'warmup started', { backend: endpoint.name, model: route.config.id, reason, items: items.length });
    const deadline = Date.now() + this.config.idleWaitMs;
    try {
      for (const item of items) {
        const passes = this.config.confirm ? ['warm', 'confirm'] : ['warm'];
        for (const pass of passes) {
          let result: 'done' | 'aborted' | 'error' | 'skip' = 'aborted';
          while (result === 'aborted') {
            if (this.stopped || endpoint.healthy === false) return;
            if (this.superseded(route, endpoint, item, since)) { result = 'skip'; break; }
            if (!(await this.waitIdle(deadline))) {
              log('info', 'warmup stopped; the gateway was not idle', { backend: endpoint.name });
              return;
            }
            if (this.superseded(route, endpoint, item, since)) { result = 'skip'; break; }
            // A real request that failed after the trigger can have a newer payload.
            result = await this.send(endpoint, this.latest(route, item), reason, pass);
          }
          if (result === 'skip') { this.stats.skipped++; break; }
          if (result === 'error') break;
        }
      }
    } finally {
      this.running.delete(endpoint);
    }
  }

  snapshot(): Obj {
    const s = this.stats;
    return {
      enabled: this.config.enabled,
      triggers: s.triggers,
      requests: s.requests,
      completed: s.completed,
      aborted: s.aborted,
      errors: s.errors,
      skipped: s.skipped,
      prompt_tokens: s.promptTokens,
      cached_tokens: s.cachedTokens,
      last: this.last,
    };
  }

  /** Abort the warm requests, and write the state file when a write is due. */
  async stop(): Promise<void> {
    this.stopped = true;
    for (const controller of this.current) controller.abort(new Error('the gateway is shutting down'));
    if (!this.persistTimer) return;
    clearTimeout(this.persistTimer);
    this.persistTimer = null;
    await this.persist(this.config.stateFile!);
  }

  private slot(map: Map<string, Map<string, WarmEntry>>, model: string): Map<string, WarmEntry> {
    let inner = map.get(model);
    if (!inner) map.set(model, (inner = new Map()));
    return inner;
  }

  /** Prefixes first (most recent first), then recent sessions (most recent first). */
  private items(route: ModelRoute, now: number): WarmEntry[] {
    const id = route.config.id;
    const prefixes = [...(this.prefixes.get(id)?.values() ?? [])].reverse();
    const sessions = [...(this.sessions.get(id)?.values() ?? [])].reverse()
      .filter((s) => now - s.at <= this.config.sessionMaxAgeMs);
    return [...prefixes, ...sessions].filter((item) => {
      if (JSON.stringify(item.payload).length <= this.config.maxPayloadChars) return true;
      log('info', 'warmup skips a payload over the size limit', { kind: item.kind });
      return false;
    });
  }

  private latest(route: ModelRoute, item: WarmEntry): WarmEntry {
    const map = item.kind === 'prefix' ? this.prefixes : this.sessions;
    return map.get(route.config.id)?.get(item.key) ?? item;
  }

  /** True when a real request for this item reached the endpoint after the trigger. */
  private superseded(route: ModelRoute, endpoint: Endpoint, item: WarmEntry, since: number): boolean {
    const map = item.kind === 'prefix' ? this.prefixes : this.sessions;
    const latest = map.get(route.config.id)?.get(item.key);
    return latest !== undefined && (latest.servedAt.get(endpoint.name) ?? 0) > since;
  }

  private async waitIdle(deadline: number): Promise<boolean> {
    while (this.busy() > 0) {
      if (this.stopped || Date.now() > deadline) return false;
      await new Promise((r) => setTimeout(r, IDLE_POLL_MS));
    }
    return !this.stopped;
  }

  private async send(endpoint: Endpoint, item: WarmEntry, reason: string, pass: string): Promise<'done' | 'aborted' | 'error'> {
    const controller = new AbortController();
    this.current.add(controller);
    this.stats.requests++;
    const started = performance.now();
    const fields = { backend: endpoint.name, kind: item.kind, pass, reason };
    try {
      const stream = await postChat(endpoint, warmPayload(item.payload), this.timeouts, controller.signal);
      if (stream.status < 200 || stream.status >= 300 || !stream.body) {
        this.stats.errors++;
        log('warn', 'warmup request failed', { ...fields, status: stream.status, error: stream.errorText?.slice(0, 300) });
        return 'error';
      }
      let usage: Obj | undefined;
      for await (const data of sseData(stream.body)) {
        if (data === '[DONE]') break;
        try {
          const chunk = JSON.parse(data);
          if (chunk.usage) usage = chunk.usage;
        } catch {
          // Not JSON. A warm request uses only the usage.
        }
      }
      const promptTokens = Number(usage?.prompt_tokens ?? 0);
      const cachedTokens = Number(usage?.prompt_tokens_details?.cached_tokens ?? 0);
      this.stats.completed++;
      this.stats.promptTokens += promptTokens;
      this.stats.cachedTokens += cachedTokens;
      this.last = { ...fields, prompt_tokens: promptTokens, cached_tokens: cachedTokens, at: new Date().toISOString() };
      log('info', 'warmup', {
        ...fields, prompt_tokens: promptTokens, cached_tokens: cachedTokens,
        elapsed_ms: Math.round(performance.now() - started),
      });
      return 'done';
    } catch (error) {
      if (controller.signal.aborted) {
        this.stats.aborted++;
        log('info', 'warmup yielded to a real request', { ...fields, elapsed_ms: Math.round(performance.now() - started) });
        return 'aborted';
      }
      this.stats.errors++;
      log('warn', 'warmup request failed', { ...fields, error: (error as Error).message });
      return 'error';
    } finally {
      this.current.delete(controller);
    }
  }

  /** Entries from the state file have at = 0. */
  private load(file: string): void {
    let raw: string;
    try {
      raw = readFileSync(file, 'utf8');
    } catch {
      return; // No state yet.
    }
    try {
      const state = JSON.parse(raw) as { version?: number; prefixes?: Array<{ model: string; key: string; payload: Obj; at: number }> };
      if (state.version !== 1 || !Array.isArray(state.prefixes)) throw new Error('unknown format');
      for (const p of state.prefixes) {
        this.slot(this.prefixes, p.model).set(p.key, { key: p.key, kind: 'prefix', payload: p.payload, at: 0, servedAt: new Map() });
      }
      this.persistedDigest = digest(state.prefixes.map((p) => [p.model, p.key, p.payload]));
      log('info', 'warmup state loaded', { file, prefixes: state.prefixes.length });
    } catch (error) {
      log('warn', 'could not read the warmup state file', { file, error: String(error) });
    }
  }

  private schedulePersist(): void {
    if (!this.config.stateFile || this.persistTimer) return;
    this.persistTimer = setTimeout(() => {
      this.persistTimer = null;
      void this.persist(this.config.stateFile!);
    }, PERSIST_DELAY_MS);
    this.persistTimer.unref();
  }

  private async persist(file: string): Promise<void> {
    const prefixes = [...this.prefixes].flatMap(([model, variants]) =>
      [...variants.values()].map((e) => ({ model, key: e.key, payload: e.payload, at: e.at })));
    const hash = digest(prefixes.map((p) => [p.model, p.key, p.payload]));
    if (hash === this.persistedDigest) return;
    // Write a new file and rename it, so that a stop in the middle of the
    // write does not leave a partial state file.
    const temp = `${file}.${process.pid}.tmp`;
    try {
      const handle = await open(temp, 'w', 0o600);
      try {
        // The mode of open() applies only to a new file. Set it before the
        // write, so that the text is never in a file that others can read.
        await handle.chmod(0o600);
        await handle.writeFile(JSON.stringify({ version: 1, prefixes }));
      } finally {
        await handle.close();
      }
      await rename(temp, file);
      this.persistedDigest = hash;
    } catch (error) {
      await rm(temp, { force: true }).catch(() => {});
      log('warn', 'could not write the warmup state file', { file, error: String(error) });
    }
  }
}
