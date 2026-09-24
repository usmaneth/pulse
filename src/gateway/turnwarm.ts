// Turn warmer. After a response, while Codex runs the tool and the GPU is
// idle, the gateway sends one prefill-only request that ends exactly on the
// last prefix-cache block boundary of that turn. The next turn of the session
// then gets the cache hit at that boundary.
//
// Why: with the Mamba "align" prefix cache of vLLM (TTFT.md, finding F4), a
// block boundary is reusable only when a scheduler step ended on it. The
// prompt's last prefill chunk ends at the end of the prompt, so a boundary
// that this chunk crosses is not reusable. A boundary that the decode crosses
// is reusable. Most Codex turns come after a tool output, and the prefill of
// that output usually crosses a boundary. Without the warm, the next turn
// prefills again from the older boundary: up to one block, about 0.85 s at
// 1728 tokens (finding F1: about 0.49 ms per uncached token).
//
// Rules:
// - With the usage of the response (prompt P, cached C, completion D) and the
//   block size, the target is B = floor(P / block) * block. The warmer skips
//   the turn when B <= C or B = P (no boundary crossed in the prefill), when
//   the decode crossed a later boundary (that one is reusable), or when B - C
//   is less than the minimum gain.
// - The warm request renders the same chat payload with vLLM /tokenize. When
//   the token count is not P, the render is not the same, and the warmer
//   skips the turn. Else it sends the first B tokens to /v1/completions with
//   max_tokens 1. That prefill ends on B, so vLLM keeps the state at B.
// - The warmer sends one warm request at a time, only when the gateway has no
//   real request in flight and the startup warmer does not warm the endpoint.
//   A real request aborts the warm request (Warmer.yieldToRealRequest). A new
//   request of the same session makes its queued warm obsolete. A time limit
//   applies to each warm request.
// - vLLM limits a cache hit to the prompt length minus one token. So the warm
//   request of B tokens gets at most the boundary before B from the cache, and
//   it prefills at least one block. If a later vLLM keeps these boundaries, the
//   `next_turn.without_warm.hit` counter shows it, and the warmer is then not
//   necessary.

import { postJson, serverUrl } from './backends.js';
import type { BackendStream, Endpoint, ModelRoute } from './backends.js';
import type { GatewayConfig } from './config.js';
import { log } from './log.js';
import { sseData } from './sse.js';
import type { Obj } from './translate.js';
import type { Warmer } from './warmer.js';

export interface TurnWarmConfig {
  /** Warm the boundary of each turn. Off by default. */
  enabled: boolean;
  /**
   * The prefix-cache block of the server in tokens: 1728 at MTP K=6, 1680 at
   * K=4 (the vLLM log line "Setting attention block size to N tokens"). The
   * block_size of the vLLM cache_config_info metric is a different value.
   */
  blockTokens: number;
  /** Skip a turn when the warm saves the next turn fewer tokens than this. */
  minGainTokens: number;
  /** Time limit for one warm request (tokenize and prefill). */
  timeoutMs: number;
  /** Skip a payload when its /tokenize JSON text is longer than this. */
  maxPayloadChars: number;
}

export function defaultTurnWarmConfig(): TurnWarmConfig {
  return { enabled: false, blockTokens: 1728, minGainTokens: 256, timeoutMs: 30_000, maxPayloadChars: 4_000_000 };
}

export interface TurnUsage {
  promptTokens: number;
  cachedTokens: number;
  completionTokens: number;
}

export type TurnWarmSkip = 'no_usage' | 'no_boundary' | 'decode_crossed' | 'small_gain';

export type TurnWarmPlan =
  | { warm: true; target: number }
  | { warm: false; reason: TurnWarmSkip; /** The boundary that the decode crossed. */ boundary?: number };

/**
 * The decision for one completed turn. The decode feeds each output token
 * except the last one back into the model, so the decode materializes the
 * states up to position P + D - 1.
 */
export function planTurnWarm(usage: TurnUsage, blockTokens: number, minGainTokens: number): TurnWarmPlan {
  const { promptTokens: p, cachedTokens: c, completionTokens: d } = usage;
  if (!(p > 0) || !(blockTokens > 0)) return { warm: false, reason: 'no_usage' };
  const target = Math.floor(p / blockTokens) * blockTokens;
  // A prefill that ends on the boundary keeps the state there.
  if (target <= c || target === p) return { warm: false, reason: 'no_boundary' };
  const decoded = d > 1 ? Math.floor((p + d - 1) / blockTokens) * blockTokens : target;
  if (decoded > target) return { warm: false, reason: 'decode_crossed', boundary: decoded };
  if (target - c < minGainTokens) return { warm: false, reason: 'small_gain' };
  return { warm: true, target };
}

/** Read the usage of a Responses object. */
export function turnUsage(usage: Obj | null | undefined): TurnUsage | null {
  if (!usage) return null;
  return {
    promptTokens: Number(usage.input_tokens ?? 0),
    cachedTokens: Number(usage.input_tokens_details?.cached_tokens ?? 0),
    completionTokens: Number(usage.output_tokens ?? 0),
  };
}

/**
 * The /tokenize body for a chat payload. vLLM gives `reasoning_effort` of a
 * chat request to the chat template as a template argument, and /tokenize has
 * no such field, so it goes into `chat_template_kwargs`.
 */
export function tokenizeRequest(payload: Obj): Obj {
  const kwargs: Obj = { ...payload.chat_template_kwargs };
  if (payload.reasoning_effort != null) kwargs.reasoning_effort = payload.reasoning_effort;
  const body: Obj = { model: payload.model, messages: payload.messages, add_generation_prompt: true };
  if (Array.isArray(payload.tools) && payload.tools.length) body.tools = payload.tools;
  if (Object.keys(kwargs).length) body.chat_template_kwargs = kwargs;
  return body;
}

/** The /v1/completions body of a warm request: the first `target` tokens, one output token. */
export function turnWarmPayload(model: unknown, tokens: number[], target: number): Obj {
  return { model, prompt: tokens.slice(0, target), max_tokens: 1, stream: true, stream_options: { include_usage: true } };
}

/** Keep the pending warms of this many sessions. */
const MAX_JOBS = 8;
/** Remember the request count and the last boundary of this many sessions. */
const MAX_SESSIONS = 256;
/** Try a warm again after this many aborts at most. */
const MAX_ATTEMPTS = 3;
/** Drop a queued warm that could not start in this time. */
const JOB_MAX_AGE_MS = 5 * 60_000;
const IDLE_POLL_MS = 100;

class TurnWarmTimeout extends Error {}

interface Job {
  session: string;
  endpoint: Endpoint;
  payload: Obj;
  promptTokens: number;
  cachedTokens: number;
  target: number;
  /** The request count of the session when the job was made. */
  seq: number;
  at: number;
  attempts: number;
}

type Evidence = { boundary: number; kind: 'after_warm' | 'without_warm' | 'after_decode' };

interface Stats {
  triggers: number;
  requests: number;
  completed: number;
  aborted: number;
  timeouts: number;
  errors: number;
  skipped: Record<string, number>;
  /** Tokens that the warm requests prefilled. */
  warmedTokens: number;
  /** Tokens that the next turns do not prefill again after a completed warm (B - C). */
  gainTokens: number;
  promptTokens: number;
  cachedTokens: number;
  timeMs: number;
  blockMismatch: number;
  nextTurn: Record<Evidence['kind'], { hit: number; miss: number }>;
}

type WarmerHooks = Pick<Warmer, 'track' | 'isWarming'>;

export class TurnWarmer {
  readonly stats: Stats = {
    triggers: 0, requests: 0, completed: 0, aborted: 0, timeouts: 0, errors: 0, skipped: {},
    warmedTokens: 0, gainTokens: 0, promptTokens: 0, cachedTokens: 0, timeMs: 0, blockMismatch: 0,
    nextTurn: { after_warm: { hit: 0, miss: 0 }, without_warm: { hit: 0, miss: 0 }, after_decode: { hit: 0, miss: 0 } },
  };
  private readonly jobs = new Map<string, Job>();
  private readonly seq = new Map<string, number>();
  private readonly evidence = new Map<string, Evidence>();
  private pumping = false;
  private stopped = false;
  private last: Obj | null = null;

  constructor(
    private readonly config: TurnWarmConfig,
    private readonly timeouts: Pick<GatewayConfig, 'connectTimeoutMs' | 'headersTimeoutMs' | 'idleTimeoutMs'>,
    /** The number of real requests in flight. */
    private readonly busy: () => number,
    private readonly warmer: WarmerHooks,
  ) {}

  /** Call it when a real request starts. A queued warm of its session is then obsolete. */
  onRequest(sessionKey: unknown): void {
    if (!this.config.enabled) return;
    const key = sessionOf(sessionKey);
    bump(this.seq, key, (this.seq.get(key) ?? 0) + 1);
    if (this.jobs.delete(key)) this.skip('superseded');
  }

  /** Call it when a real request completed, with the Responses usage. */
  afterResponse(route: ModelRoute, endpoint: Endpoint, payload: Obj, sessionKey: unknown, responseUsage: Obj | null | undefined): void {
    if (!this.config.enabled || this.stopped || route.config.profile !== 'qwen38') return;
    this.stats.triggers++;
    const key = sessionOf(sessionKey);
    const usage = turnUsage(responseUsage);
    if (!usage) return this.skip('no_usage');
    this.checkBlock(usage.cachedTokens);
    this.settle(key, usage);

    const plan = planTurnWarm(usage, this.config.blockTokens, this.config.minGainTokens);
    if (!plan.warm) {
      if (plan.boundary) bump(this.evidence, key, { boundary: plan.boundary, kind: 'after_decode' });
      return this.skip(plan.reason);
    }
    if (endpoint.healthy === false) return this.skip('unhealthy');
    bump(this.evidence, key, { boundary: plan.target, kind: 'without_warm' });
    if (this.jobs.delete(key)) this.skip('superseded');
    this.jobs.set(key, {
      session: key, endpoint, payload,
      promptTokens: usage.promptTokens, cachedTokens: usage.cachedTokens, target: plan.target,
      seq: this.seq.get(key) ?? 0, at: Date.now(), attempts: 0,
    });
    while (this.jobs.size > MAX_JOBS) {
      this.jobs.delete(this.jobs.keys().next().value!);
      this.skip('queue_full');
    }
    void this.pump();
  }

  snapshot(): Obj {
    const s = this.stats;
    return {
      enabled: this.config.enabled,
      block_tokens: this.config.blockTokens,
      min_gain_tokens: this.config.minGainTokens,
      triggers: s.triggers,
      requests: s.requests,
      completed: s.completed,
      aborted: s.aborted,
      timeouts: s.timeouts,
      errors: s.errors,
      skipped: { ...s.skipped },
      queued: this.jobs.size,
      warmed_tokens: s.warmedTokens,
      gain_tokens: s.gainTokens,
      prompt_tokens: s.promptTokens,
      cached_tokens: s.cachedTokens,
      time_ms: Math.round(s.timeMs),
      block_mismatch: s.blockMismatch,
      next_turn: {
        after_warm: { ...s.nextTurn.after_warm },
        without_warm: { ...s.nextTurn.without_warm },
        after_decode: { ...s.nextTurn.after_decode },
      },
      last: this.last,
    };
  }

  /** Drop the queue. The Warmer aborts the warm request in flight. */
  stop(): void {
    this.stopped = true;
    this.jobs.clear();
  }

  /** Resolves when no warm is queued or in flight. For tests. */
  async idle(): Promise<void> {
    while (this.pumping || this.jobs.size) await delay(10);
  }

  private skip(reason: string): void {
    this.stats.skipped[reason] = (this.stats.skipped[reason] ?? 0) + 1;
  }

  /**
   * Cached token counts are multiples of the block. Another value means that
   * the block setting does not agree with the server.
   */
  private checkBlock(cached: number): void {
    if (cached <= 0 || cached % this.config.blockTokens === 0) return;
    if (this.stats.blockMismatch++ === 0) {
      log('warn', 'cached tokens are not a multiple of the prefix block; check PULSE_GATEWAY_PREFIX_BLOCK_TOKENS', {
        cached_tokens: cached, block_tokens: this.config.blockTokens,
      });
    }
  }

  /** Count if the turn found the boundary that the previous turn of its session left. */
  private settle(key: string, usage: TurnUsage): void {
    const previous = this.evidence.get(key);
    if (!previous) return;
    this.evidence.delete(key);
    // A shorter prompt (for example after a compaction) does not tell.
    if (usage.promptTokens <= previous.boundary) return;
    const counter = this.stats.nextTurn[previous.kind];
    if (usage.cachedTokens >= previous.boundary) counter.hit++;
    else counter.miss++;
  }

  private ready(endpoint: Endpoint): boolean {
    return this.busy() === 0 && !this.warmer.isWarming(endpoint);
  }

  private async pump(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    try {
      while (!this.stopped && this.jobs.size) {
        // The most recent turn first.
        const job = [...this.jobs.values()].pop()!;
        if (Date.now() - job.at > JOB_MAX_AGE_MS || job.endpoint.healthy === false) {
          // After a failure the server can have lost its cache, and the
          // startup warmer warms the recent session.
          this.jobs.delete(job.session);
          this.skip(job.endpoint.healthy === false ? 'unhealthy' : 'expired');
          continue;
        }
        if (!this.ready(job.endpoint)) {
          await delay(IDLE_POLL_MS);
          continue;
        }
        this.jobs.delete(job.session);
        const result = await this.run(job);
        // A request of another session aborted the warm. Try again when the
        // gateway is idle, unless the session sent a new request.
        if (result === 'aborted' && !this.stopped && job.attempts < MAX_ATTEMPTS
          && this.seq.get(job.session) === job.seq && !this.jobs.has(job.session)) {
          this.jobs.set(job.session, job);
        }
      }
    } finally {
      this.pumping = false;
    }
  }

  private async run(job: Job): Promise<'done' | 'aborted' | 'skip' | 'error'> {
    const controller = new AbortController();
    const untrack = this.warmer.track(controller);
    const timer = setTimeout(() => controller.abort(new TurnWarmTimeout(`turn warm exceeded ${this.config.timeoutMs} ms`)), this.config.timeoutMs);
    const { endpoint, target } = job;
    const fields = { backend: endpoint.name, prompt_tokens: job.promptTokens, cached_tokens: job.cachedTokens, target };
    const started = performance.now();
    job.attempts++;
    this.stats.requests++;
    try {
      const body = JSON.stringify(tokenizeRequest(job.payload));
      if (body.length > this.config.maxPayloadChars) {
        this.skip('too_large');
        return 'skip';
      }
      controller.signal.throwIfAborted();
      const rendered = await readJson(await postJson(serverUrl(endpoint, '/tokenize'), body, this.timeouts, controller.signal));
      const tokens = rendered.tokens;
      if (!Array.isArray(tokens) || Number(rendered.count) !== job.promptTokens || tokens.length !== job.promptTokens) {
        this.skip('count_mismatch');
        log('info', 'turn warm skipped; the render does not match the prompt', { ...fields, count: rendered.count ?? null });
        return 'skip';
      }
      const tokenizeMs = performance.now() - started;
      controller.signal.throwIfAborted();
      const stream = await postJson(
        new URL(`${endpoint.config.baseUrl}/completions`),
        turnWarmPayload(job.payload.model, tokens as number[], target), this.timeouts, controller.signal,
      );
      if (stream.status < 200 || stream.status >= 300 || !stream.body) {
        throw new Error(`HTTP ${stream.status}: ${(stream.errorText ?? '').slice(0, 300)}`);
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
      const warmPrompt = Number(usage?.prompt_tokens ?? 0);
      const warmCached = Number(usage?.prompt_tokens_details?.cached_tokens ?? 0);
      const elapsed = performance.now() - started;
      const s = this.stats;
      s.completed++;
      s.promptTokens += warmPrompt;
      s.cachedTokens += warmCached;
      s.warmedTokens += Math.max(0, warmPrompt - warmCached);
      s.gainTokens += target - job.cachedTokens;
      s.timeMs += elapsed;
      const evidence = this.evidence.get(job.session);
      if (evidence?.boundary === target) evidence.kind = 'after_warm';
      const result = {
        ...fields, warm_prompt_tokens: warmPrompt, warm_cached_tokens: warmCached,
        tokenize_ms: Math.round(tokenizeMs), elapsed_ms: Math.round(elapsed),
      };
      this.last = { ...result, at: new Date().toISOString() };
      log('info', 'turn warm', result);
      return 'done';
    } catch (error) {
      const elapsed = Math.round(performance.now() - started);
      if (controller.signal.aborted) {
        const reason = controller.signal.reason;
        if (reason instanceof TurnWarmTimeout) {
          this.stats.timeouts++;
          log('warn', 'turn warm timed out', { ...fields, elapsed_ms: elapsed });
          return 'error';
        }
        this.stats.aborted++;
        log('info', 'turn warm yielded to a real request', { ...fields, elapsed_ms: elapsed });
        return 'aborted';
      }
      this.stats.errors++;
      log('warn', 'turn warm failed', { ...fields, error: (error as Error).message, elapsed_ms: elapsed });
      return 'error';
    } finally {
      clearTimeout(timer);
      untrack();
    }
  }
}

/** Codex sends the conversation id as prompt_cache_key. Requests without one share one key. */
function sessionOf(sessionKey: unknown): string {
  return typeof sessionKey === 'string' ? sessionKey : '';
}

/** Set a key as the most recent one, and keep at most MAX_SESSIONS keys. */
function bump<T>(map: Map<string, T>, key: string, value: T): void {
  map.delete(key);
  map.set(key, value);
  while (map.size > MAX_SESSIONS) map.delete(map.keys().next().value!);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms).unref());
}

async function readJson(stream: BackendStream): Promise<Obj> {
  if (stream.status < 200 || stream.status >= 300 || !stream.body) {
    throw new Error(`HTTP ${stream.status}: ${(stream.errorText ?? '').slice(0, 300)}`);
  }
  let text = '';
  const decoder = new TextDecoder();
  for await (const chunk of stream.body) text += decoder.decode(chunk, { stream: true });
  text += decoder.decode();
  return JSON.parse(text) as Obj;
}
