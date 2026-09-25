// Turn warmer. After a response, while Codex runs the tool and the GPU is
// idle, the gateway sends prefill-only requests that end exactly on the
// prefix-cache block boundaries that the prefill of that turn crossed. The
// next turn of the session then gets the cache hit at the last boundary.
//
// Why: with the Mamba "align" prefix cache of vLLM (TTFT.md, finding F4), a
// block boundary is reusable only when a scheduler step ended on it. The
// prompt's last prefill chunk ends at the end of the prompt, so a boundary
// that this chunk crosses is not reusable. A boundary that the decode crosses
// is reusable. Most Codex turns come after a tool output, and the prefill of
// that output usually crosses a boundary. Without the warm, the next turn
// prefills again from the older boundary: up to one block, about 0.85 s at
// 1728 tokens (finding F1: about 0.49 ms per uncached token), and more when
// the output crossed more than one boundary.
//
// Rules:
// - With the usage of the response (prompt P, cached C, completion D) and the
//   block size, the target is B = floor(P / block) * block. The warmer skips
//   the turn when B <= C or B = P (no boundary crossed in the prefill), when
//   the decode crossed a later boundary (that one is reusable), or when the
//   gain is less than the minimum.
// - A boundary that a prefill crosses for the second time is reusable. The
//   A/B chains without the warm agree with this rule for each turn: turn k
//   hits the last boundary that turns k-1 and k-2 both crossed. So when the
//   prompt continues the prompt P' of the previous turn, the boundaries up to
//   floor(P' / block) * block are reusable after this turn, and the warm
//   starts at the later of C and that boundary. When it is B, the warmer
//   skips the turn (`crossed_before`).
// - The warm renders the same chat payload with vLLM /tokenize. When the token
//   count is not P, the render is not the same, and the warmer skips the turn.
//   Else it prefills to B in stages. Each stage is one /v1/completions request
//   with max_tokens 1 and a prompt of the first X tokens, where X is the next
//   boundary (see stepBlocks), and B last. A stage ends on X, so vLLM keeps
//   the state at X, and the next stage gets the cache hit at X.
// - Stages bound the time that a real request can wait. vLLM does not stop a
//   scheduler step that it started, so an abort of the client does not give
//   the GPU back sooner: a real request that arrives during a warm step waits
//   for the end of that step. In the first A/B a warm of two blocks in one
//   step added 1.8 s to a request with another prefix that arrived 0.3 s
//   after the warm started. With stages of one block, the wait is at most one
//   block (about 1 s on spark1). The aborted warms of that A/B still gave the
//   next turn of the session the hit at B, so vLLM completes the step and
//   keeps the state. Thus a real request does not abort a stage, and the
//   warmer starts no new stage while a real request is in flight.
// - The /tokenize request is CPU work in the API server, and a real request
//   aborts it (Warmer.yieldToRealRequest).
// - The warmer runs one warm at a time, only when the gateway has no real
//   request in flight and the startup warmer does not warm the endpoint. A
//   new request of the same session makes its warm obsolete. A time limit
//   applies to each request of a warm.
// - vLLM limits a cache hit to the prompt length minus one token. So a stage
//   of X tokens gets at most the boundary before X from the cache, and it
//   prefills at least one block. If a later vLLM keeps these boundaries, the
//   `next_turn.without_warm.hit` counter shows it, and the warmer is then not
//   necessary.
// - The prefill of a real request that took more than one scheduler step
//   (more than max-num-batched-tokens) ended steps on boundaries that are
//   reusable already. The stages prefill these blocks again. The GPU is idle
//   at that time, but the warm takes longer to reach B.

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
  /** Time limit for one request of a warm (the tokenize or one stage). */
  timeoutMs: number;
  /**
   * The blocks that one stage prefills. One block is one scheduler step of
   * about 1 s, which is the longest time that a real request waits.
   */
  stepBlocks: number;
  /** Skip a payload when its /tokenize JSON text is longer than this. */
  maxPayloadChars: number;
}

export function defaultTurnWarmConfig(): TurnWarmConfig {
  return { enabled: false, blockTokens: 1728, minGainTokens: 256, timeoutMs: 30_000, stepBlocks: 1, maxPayloadChars: 4_000_000 };
}

export interface TurnUsage {
  promptTokens: number;
  cachedTokens: number;
  completionTokens: number;
}

export type TurnWarmSkip = 'no_usage' | 'no_boundary' | 'decode_crossed' | 'crossed_before' | 'small_gain';

export type TurnWarmPlan =
  | { warm: true; target: number; /** The warm starts here: C, or a later boundary that is reusable already. */ from: number }
  | { warm: false; reason: TurnWarmSkip; /** The boundary that the decode or an earlier prefill made reusable. */ boundary?: number };

/**
 * The decision for one completed turn. The decode feeds each output token
 * except the last one back into the model, so the decode materializes the
 * states up to position P + D - 1.
 *
 * `previous` is the usage of the previous turn of the session. A boundary
 * that a prefill crosses for the second time is reusable (TTFT.md, F4: B2
 * after B hit the boundary; the A/B chains agree with this for each turn).
 * The previous prefill crossed each boundary up to floor(P' / block) * block.
 * When this prompt continues the previous one, these boundaries are reusable
 * after this turn, and the warm starts after them. A longer prompt and a
 * cached part that did not shrink show that the prompt continues the previous
 * one (a new effort level or a compaction changes the prefix, and then C
 * shrinks or P shrinks).
 */
export function planTurnWarm(usage: TurnUsage, blockTokens: number, minGainTokens: number, previous?: TurnUsage | null): TurnWarmPlan {
  const { promptTokens: p, cachedTokens: c, completionTokens: d } = usage;
  if (!(p > 0) || !(blockTokens > 0)) return { warm: false, reason: 'no_usage' };
  const target = Math.floor(p / blockTokens) * blockTokens;
  // A prefill that ends on the boundary keeps the state there.
  if (target <= c || target === p) return { warm: false, reason: 'no_boundary' };
  const decoded = d > 1 ? Math.floor((p + d - 1) / blockTokens) * blockTokens : target;
  if (decoded > target) return { warm: false, reason: 'decode_crossed', boundary: decoded };
  let from = c;
  if (previous && p > previous.promptTokens && c >= previous.cachedTokens) {
    from = Math.max(c, Math.min(target, Math.floor(previous.promptTokens / blockTokens) * blockTokens));
  }
  if (from >= target) return { warm: false, reason: 'crossed_before', boundary: target };
  if (target - from < minGainTokens) return { warm: false, reason: 'small_gain' };
  return { warm: true, target, from };
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
 * The /tokenize body for a chat payload. It does the two changes that vLLM
 * does to a chat request and not to a /tokenize request:
 * - vLLM gives `reasoning_effort` to the chat template as a template
 *   argument. /tokenize has no such field, so it goes into
 *   `chat_template_kwargs`.
 * - The chat request renames the `reasoning_content` field of a message to
 *   `reasoning` (ChatCompletionRequest._normalize_messages_before), and the
 *   template renderer reads only `reasoning`. /tokenize does not rename it, so
 *   without this change the render drops the replayed reasoning, the count is
 *   less than P, and every turn after a reasoning turn is skipped.
 */
export function tokenizeRequest(payload: Obj): Obj {
  const kwargs: Obj = { ...payload.chat_template_kwargs };
  if (payload.reasoning_effort != null) kwargs.reasoning_effort = payload.reasoning_effort;
  const body: Obj = { model: payload.model, messages: renameReasoning(payload.messages), add_generation_prompt: true };
  if (Array.isArray(payload.tools) && payload.tools.length) body.tools = payload.tools;
  if (Object.keys(kwargs).length) body.chat_template_kwargs = kwargs;
  return body;
}

/** Copy the messages with `reasoning_content` renamed to `reasoning`, as vLLM does for a chat request. */
function renameReasoning(messages: unknown): unknown {
  if (!Array.isArray(messages) || !messages.some((m) => m?.reasoning_content != null)) return messages;
  return messages.map((message) => {
    if (message?.reasoning_content == null) return message;
    const { reasoning_content: reasoning, ...rest } = message as Obj;
    return rest.reasoning == null ? { ...rest, reasoning } : rest;
  });
}

/**
 * The ends of the stages of a warm from `from` (a reusable point) to
 * `target` (a boundary): each boundary `stepBlocks` blocks after the previous
 * one, and `target` last.
 */
export function warmStages(from: number, target: number, blockTokens: number, stepBlocks: number): number[] {
  const step = Math.max(1, Math.floor(stepBlocks)) * blockTokens;
  const stages: number[] = [];
  for (let end = Math.floor(from / blockTokens) * blockTokens + step; end < target; end += step) stages.push(end);
  if (target > from) stages.push(target);
  return stages;
}

/** The /v1/completions body of a warm request: the first `target` tokens, one output token. */
export function turnWarmPayload(model: unknown, tokens: number[], target: number): Obj {
  return { model, prompt: tokens.slice(0, target), max_tokens: 1, stream: true, stream_options: { include_usage: true } };
}

/** Keep the pending warms of this many sessions. */
const MAX_JOBS = 8;
/** Remember the request count and the last boundary of this many sessions. */
const MAX_SESSIONS = 256;
/** Stop a warm after it yielded to a real request this many times. */
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
  /** The reusable point where the warm starts. */
  from: number;
  /** The rendered prompt, after the /tokenize request. */
  tokens?: number[];
  /** The last boundary that a stage reached (at first, the cached part C). */
  reached: number;
  stages: number;
  /** Totals of the stages, for the log. */
  warmPromptTokens: number;
  warmCachedTokens: number;
  elapsedMs: number;
  tokenizeMs: number;
  /** The request count of the session when the job was made. */
  seq: number;
  at: number;
  attempts: number;
}

type Evidence = { boundary: number; kind: 'after_warm' | 'without_warm' | 'after_decode' | 'crossed_before' };

interface Stats {
  triggers: number;
  /** Stage requests sent. */
  requests: number;
  /** Stage requests that completed. */
  stages: number;
  /** Warms that reached their target B. */
  completed: number;
  /** /tokenize requests that a real request aborted. */
  aborted: number;
  /** Warms that stopped between two stages for a real request. */
  yielded: number;
  /** Real requests that started while a stage was in flight (they wait at most for that stage). */
  overlaps: number;
  timeouts: number;
  errors: number;
  skipped: Record<string, number>;
  /** Tokens that the stages prefilled. */
  warmedTokens: number;
  /** Tokens between the start of each warm and the boundary that its stages reached. */
  gainTokens: number;
  promptTokens: number;
  cachedTokens: number;
  timeMs: number;
  blockMismatch: number;
  nextTurn: Record<Evidence['kind'], { hit: number; miss: number }>;
}

type WarmerHooks = Pick<Warmer, 'track' | 'isWarming'>;

type RunResult = 'done' | 'yielded' | 'skip' | 'error';

export class TurnWarmer {
  readonly stats: Stats = {
    triggers: 0, requests: 0, stages: 0, completed: 0, aborted: 0, yielded: 0, overlaps: 0,
    timeouts: 0, errors: 0, skipped: {},
    warmedTokens: 0, gainTokens: 0, promptTokens: 0, cachedTokens: 0, timeMs: 0, blockMismatch: 0,
    nextTurn: {
      after_warm: { hit: 0, miss: 0 }, without_warm: { hit: 0, miss: 0 },
      after_decode: { hit: 0, miss: 0 }, crossed_before: { hit: 0, miss: 0 },
    },
  };
  private readonly jobs = new Map<string, Job>();
  private readonly seq = new Map<string, number>();
  private readonly evidence = new Map<string, Evidence>();
  /** The usage of the last turn of each session. */
  private readonly previous = new Map<string, TurnUsage>();
  private pumping = false;
  private stopped = false;
  /** The controller of the stage request in flight. */
  private stage: AbortController | null = null;
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
    if (this.stage) this.stats.overlaps++;
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

    const plan = planTurnWarm(usage, this.config.blockTokens, this.config.minGainTokens, this.previous.get(key));
    bump(this.previous, key, usage);
    if (!plan.warm) {
      if (plan.boundary) {
        bump(this.evidence, key, { boundary: plan.boundary, kind: plan.reason === 'crossed_before' ? 'crossed_before' : 'after_decode' });
      }
      return this.skip(plan.reason);
    }
    if (endpoint.healthy === false) return this.skip('unhealthy');
    bump(this.evidence, key, { boundary: plan.target, kind: 'without_warm' });
    if (this.jobs.delete(key)) this.skip('superseded');
    this.jobs.set(key, {
      session: key, endpoint, payload,
      promptTokens: usage.promptTokens, cachedTokens: usage.cachedTokens, target: plan.target,
      from: plan.from, reached: plan.from, stages: 0, warmPromptTokens: 0, warmCachedTokens: 0, elapsedMs: 0, tokenizeMs: 0,
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
      step_blocks: this.config.stepBlocks,
      triggers: s.triggers,
      requests: s.requests,
      stages: s.stages,
      completed: s.completed,
      aborted: s.aborted,
      yielded: s.yielded,
      overlaps: s.overlaps,
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
        crossed_before: { ...s.nextTurn.crossed_before },
      },
      last: this.last,
    };
  }

  /** Drop the queue and abort the stage in flight. The Warmer aborts the /tokenize request in flight. */
  stop(): void {
    this.stopped = true;
    this.jobs.clear();
    this.stage?.abort(new Error('the gateway is shutting down'));
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
        if (result !== 'yielded' || this.stopped) continue;
        // A real request stopped the warm. When it came from the same
        // session, its own response makes a new warm. Else the warm goes on
        // from the last stage when the gateway is idle again.
        if (this.seq.get(job.session) !== job.seq) this.skip('superseded');
        else if (++job.attempts >= MAX_ATTEMPTS) this.skip('attempts');
        else if (!this.jobs.has(job.session)) this.jobs.set(job.session, job);
      }
    } finally {
      this.pumping = false;
    }
  }

  private fields(job: Job): Obj {
    return {
      backend: job.endpoint.name, prompt_tokens: job.promptTokens, cached_tokens: job.cachedTokens,
      from: job.from, target: job.target, reached: job.reached, stages: job.stages,
    };
  }

  private async run(job: Job): Promise<RunResult> {
    if (!job.tokens) {
      const tokenized = await this.tokenize(job);
      if (tokenized !== 'done') return tokenized;
    }
    const stages = warmStages(job.reached, job.target, this.config.blockTokens, this.config.stepBlocks);
    for (const end of stages) {
      // No new stage while a real request is in flight or after the session
      // sent a new request. The stage in flight is not aborted (see the top
      // of this file).
      if (this.stopped || !this.ready(job.endpoint) || this.seq.get(job.session) !== job.seq) {
        this.stats.yielded++;
        log('info', 'turn warm yielded to a real request', { ...this.fields(job), elapsed_ms: Math.round(job.elapsedMs) });
        return 'yielded';
      }
      const result = await this.prefill(job, end);
      if (result !== 'done') return result;
    }
    const s = this.stats;
    s.completed++;
    const evidence = this.evidence.get(job.session);
    if (evidence?.boundary === job.target) evidence.kind = 'after_warm';
    const result = {
      ...this.fields(job), warm_prompt_tokens: job.warmPromptTokens, warm_cached_tokens: job.warmCachedTokens,
      tokenize_ms: Math.round(job.tokenizeMs), elapsed_ms: Math.round(job.elapsedMs),
    };
    this.last = { ...result, at: new Date().toISOString() };
    log('info', 'turn warm', result);
    return 'done';
  }

  /** Render the payload. A real request aborts this request. */
  private async tokenize(job: Job): Promise<RunResult> {
    const controller = new AbortController();
    const untrack = this.warmer.track(controller);
    const timer = setTimeout(() => controller.abort(new TurnWarmTimeout(`turn warm exceeded ${this.config.timeoutMs} ms`)), this.config.timeoutMs);
    const started = performance.now();
    try {
      const body = JSON.stringify(tokenizeRequest(job.payload));
      if (body.length > this.config.maxPayloadChars) {
        this.skip('too_large');
        return 'skip';
      }
      controller.signal.throwIfAborted();
      const rendered = await readJson(await postJson(serverUrl(job.endpoint, '/tokenize'), body, this.timeouts, controller.signal));
      const tokens = rendered.tokens;
      if (!Array.isArray(tokens) || Number(rendered.count) !== job.promptTokens || tokens.length !== job.promptTokens) {
        this.skip('count_mismatch');
        log('info', 'turn warm skipped; the render does not match the prompt', { ...this.fields(job), count: rendered.count ?? null });
        return 'skip';
      }
      job.tokens = tokens as number[];
      return 'done';
    } catch (error) {
      return this.failed(job, controller, error, performance.now() - started);
    } finally {
      const elapsed = performance.now() - started;
      job.tokenizeMs += elapsed;
      job.elapsedMs += elapsed;
      this.stats.timeMs += elapsed;
      clearTimeout(timer);
      untrack();
    }
  }

  /** One stage: prefill the first `end` tokens. Only the time limit and the shutdown abort it. */
  private async prefill(job: Job, end: number): Promise<RunResult> {
    const controller = new AbortController();
    this.stage = controller;
    const timer = setTimeout(() => controller.abort(new TurnWarmTimeout(`turn warm exceeded ${this.config.timeoutMs} ms`)), this.config.timeoutMs);
    const started = performance.now();
    this.stats.requests++;
    try {
      const stream = await postJson(
        new URL(`${job.endpoint.config.baseUrl}/completions`),
        turnWarmPayload(job.payload.model, job.tokens!, end), this.timeouts, controller.signal,
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
      const s = this.stats;
      s.stages++;
      s.promptTokens += warmPrompt;
      s.cachedTokens += warmCached;
      s.warmedTokens += Math.max(0, warmPrompt - warmCached);
      s.gainTokens += Math.max(0, end - job.reached);
      job.warmPromptTokens += warmPrompt;
      if (job.stages === 0) job.warmCachedTokens = warmCached;
      job.stages++;
      job.reached = end;
      return 'done';
    } catch (error) {
      return this.failed(job, controller, error, performance.now() - started);
    } finally {
      const elapsed = performance.now() - started;
      job.elapsedMs += elapsed;
      this.stats.timeMs += elapsed;
      clearTimeout(timer);
      if (this.stage === controller) this.stage = null;
    }
  }

  private failed(job: Job, controller: AbortController, error: unknown, elapsedMs: number): RunResult {
    const fields = { ...this.fields(job), elapsed_ms: Math.round(elapsedMs) };
    if (controller.signal.aborted) {
      if (controller.signal.reason instanceof TurnWarmTimeout) {
        this.stats.timeouts++;
        log('warn', 'turn warm timed out', fields);
        return 'error';
      }
      if (this.stopped) return 'error';
      this.stats.aborted++;
      log('info', 'turn warm yielded to a real request', fields);
      return 'yielded';
    }
    this.stats.errors++;
    log('warn', 'turn warm failed', { ...fields, error: (error as Error).message });
    return 'error';
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
