// Pulse gateway: an OpenAI Responses endpoint for Codex CLI in front of one
// or more OpenAI-compatible chat-completions backends (vLLM for Qwen3.8,
// llama.cpp for Bonsai).
//
// Endpoints:
//   POST /v1/responses   Responses API, streaming and non-streaming
//   GET  /v1/models      configured model ids
//   GET  /health         200 when every model has an available backend
//   GET  /metrics        JSON counters and per-backend latency

import http from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { createWriteStream, fchmod, readFileSync } from 'node:fs';
import type { WriteStream } from 'node:fs';
import type { GatewayConfig } from './config.js';
import { Endpoint, HealthChecker, ModelRoute, RetryableBackendError, postChat, closeAgents } from './backends.js';
import type { BackendStream } from './backends.js';
import { GatewayMetrics } from './metrics.js';
import { log } from './log.js';
import { sseData } from './sse.js';
import {
  BackendStreamError, ChatStreamTranslator, RequestError, isJsonObjectText, newId, responsesToChat, sseFrame,
} from './translate.js';
import type { ChatRequest, Obj, ResponseEvent } from './translate.js';

const VERSION = '1.0.0';
/** Seconds in the Retry-After header of a 503 response. */
const RETRY_AFTER_S = 2;

class HttpError extends Error {
  /** The id of the Responses request that failed, when there is one. */
  requestId?: string;
  /** True when the `response` log line of the request already has this error. */
  logged = false;
  constructor(
    readonly status: number,
    message: string,
    readonly code = 'invalid_request_error',
    readonly headers: Record<string, string> = {},
  ) { super(message); }
}

/** Wait before the next backend attempt: 250 ms, doubled on each attempt, up to 3 s. */
export function retryDelayMs(attempt: number): number {
  return Math.min(250 * 2 ** attempt, 3_000);
}

/** Resolve after `ms`. Reject when the signal aborts. */
function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason);
    const onAbort = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', onAbort); resolve(); }, ms);
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * The Responses error code for a backend 4xx message. Codex treats
 * `context_length_exceeded` as a full context window, and `invalid_prompt`
 * as a request error that a retry does not repair.
 */
export function backendErrorCode(message: string): string {
  return /context length|context window|too many tokens|reduce the length|prompt is too long/i.test(message)
    ? 'context_length_exceeded'
    : 'invalid_prompt';
}

export class Gateway {
  readonly metrics = new GatewayMetrics();
  readonly routes: ModelRoute[];
  private readonly health: HealthChecker;
  private server: http.Server | null = null;
  private readonly inflight = new Set<AbortController>();
  private draining = false;
  private catalog: Obj[] | null = null;
  private traceStream: WriteStream | null = null;
  private traceFailed = false;

  constructor(readonly config: GatewayConfig) {
    this.routes = config.models.map((m) => new ModelRoute(m));
    this.health = new HealthChecker(this.routes, config);
    const catalogPath = process.env.PULSE_GATEWAY_MODEL_CATALOG;
    if (catalogPath) {
      const raw = JSON.parse(readFileSync(catalogPath, 'utf8'));
      const ids = new Set(config.models.map((m) => m.id));
      this.catalog = (Array.isArray(raw) ? raw : raw.models ?? []).filter((m: Obj) => ids.has(m.slug));
    }
  }

  /** Run one health check round now. */
  checkHealth(): Promise<void> {
    return this.health.checkAll();
  }

  get address(): { host: string; port: number } | null {
    const addr = this.server?.address();
    return addr && typeof addr === 'object' ? { host: addr.address, port: addr.port } : null;
  }

  start(): Promise<void> {
    this.health.start();
    this.server = http.createServer((req, res) => void this.handle(req, res));
    // Node closes idle keep-alive sockets after 5 s by default, and a request
    // that waits for a long prefill must not hit the server request timeout.
    this.server.requestTimeout = 0;
    this.server.headersTimeout = 60_000;
    this.server.keepAliveTimeout = 65_000;
    return new Promise((resolve, reject) => {
      this.server!.once('error', reject);
      this.server!.listen(this.config.port, this.config.host, () => {
        this.server!.off('error', reject);
        log('info', 'pulse gateway listening', {
          address: this.address,
          models: this.config.models.map((m) => ({
            id: m.id, profile: m.profile, endpoints: m.endpoints.map((e) => `${e.name}=${e.baseUrl}${e.enabled === false ? ' (disabled)' : ''}`),
          })),
        });
        resolve();
      });
    });
  }

  /**
   * Stop accepting connections, let in-flight requests finish for up to
   * `shutdownGraceMs`, then cancel the rest. Cancelled streams get a
   * `response.failed` event, so Codex shows an error instead of hanging.
   */
  async stop(): Promise<void> {
    this.draining = true;
    this.health.stop();
    const closed = new Promise<void>((resolve) => (this.server ? this.server.close(() => resolve()) : resolve()));
    this.server?.closeIdleConnections();
    const deadline = Date.now() + this.config.shutdownGraceMs;
    while (this.inflight.size && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 100));
    }
    if (this.inflight.size) {
      log('warn', 'shutdown grace ended; cancelling in-flight requests', { in_flight: this.inflight.size });
      for (const controller of this.inflight) controller.abort(new Error('the gateway is shutting down'));
      // Give each cancelled stream the time to write its response.failed
      // event before the sockets close.
      const cancelDeadline = Date.now() + 1_000;
      while (this.inflight.size && Date.now() < cancelDeadline) {
        await new Promise((r) => setTimeout(r, 10));
      }
    }
    this.server?.closeAllConnections();
    await closed;
    closeAgents();
    await this.closeTrace();
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const path = (req.url ?? '/').split('?')[0].replace(/\/+$/, '') || '/';
    try {
      if (req.method === 'GET' && (path === '/health' || path === '/ready')) return this.sendHealth(res);
      if (req.method === 'GET' && path === '/metrics') return sendJson(res, 200, this.snapshot());
      if (!this.authorized(req)) throw new HttpError(401, 'missing or wrong API key', 'unauthorized');
      if (req.method === 'GET' && (path === '/v1/models' || path === '/models')) return this.sendModels(res);
      if (req.method === 'POST' && (path === '/v1/responses' || path === '/responses')) return await this.handleResponses(req, res);
      throw new HttpError(404, `no route for ${req.method} ${path}`, 'not_found');
    } catch (error) {
      const status = error instanceof HttpError ? error.status : 500;
      const code = error instanceof HttpError ? error.code : 'internal_error';
      if (!(error instanceof HttpError)) {
        log('error', 'request failed', { path, error: String(error) });
      } else if (!error.logged) {
        // An error before the backend call (bad JSON, unknown model,
        // unsupported input, auth, route, shutdown) has no `response` line.
        log('warn', 'request rejected', {
          request_id: error.requestId ?? null, method: req.method, path, status, error: error.message,
        });
      }
      if (!res.headersSent) {
        sendJson(res, status, { error: { type: code, message: (error as Error).message } }, error instanceof HttpError ? error.headers : {});
      } else res.end();
    }
  }

  private authorized(req: IncomingMessage): boolean {
    const key = process.env.PULSE_GATEWAY_API_KEY;
    if (!key) return true;
    return req.headers.authorization === `Bearer ${key}`;
  }

  /**
   * The gateway state and the state of each backend. The status is 200 when
   * every model has a backend that is not marked unhealthy, else 503.
   */
  private sendHealth(res: ServerResponse): void {
    const iso = (t: number | null) => (t ? new Date(t).toISOString() : null);
    const models = this.routes.map((route) => ({
      id: route.config.id,
      available: route.available,
      backends: route.endpoints.map((e) => ({
        name: e.name,
        base_url: e.config.baseUrl,
        enabled: e.enabled,
        healthy: e.healthy,
        in_flight: e.stats.inFlight,
        consecutive_failures: e.consecutiveFailures,
        last_check_at: iso(e.lastCheckAt),
        last_ok_at: iso(e.lastOkAt),
        last_error: e.lastError,
      })),
    }));
    const ok = !this.draining && this.routes.every((r) => r.available);
    const headers: Record<string, string> = ok ? {} : { 'Retry-After': String(RETRY_AFTER_S) };
    sendJson(res, ok ? 200 : 503, {
      status: this.draining ? 'draining' : ok ? 'ok' : 'degraded',
      version: VERSION,
      uptime_s: Math.round((Date.now() - this.metrics.startedAt) / 1000),
      in_flight: this.metrics.inFlight,
      retries: this.metrics.retries,
      retry_window_ms: this.config.retryWindowMs,
      models,
    }, headers);
  }

  private sendModels(res: ServerResponse): void {
    const created = Math.floor(this.metrics.startedAt / 1000);
    // `data` is the OpenAI list shape. `models` is the Codex catalog shape,
    // filled when PULSE_GATEWAY_MODEL_CATALOG names a catalog file.
    sendJson(res, 200, {
      object: 'list',
      data: this.routes.map((r) => ({ id: r.config.id, object: 'model', created, owned_by: 'pulse' })),
      models: this.catalog ?? [],
    });
  }

  snapshot(): Obj {
    const m = this.metrics;
    return {
      version: VERSION,
      uptime_s: Math.round((Date.now() - m.startedAt) / 1000),
      requests: {
        total: m.requests, in_flight: m.inFlight, completed: m.completed, incomplete: m.incomplete,
        failed: m.failed, cancelled: m.cancelled, client_errors: m.clientErrors, retries: m.retries,
      },
      tokens: {
        input: m.inputTokens, cached_input: m.cachedInputTokens, output: m.outputTokens,
        reasoning: m.reasoningTokens,
      },
      tool_calls: m.toolCalls,
      backends: this.routes.flatMap((route) => route.endpoints.map((e) => ({
        model: route.config.id,
        name: e.name,
        base_url: e.config.baseUrl,
        enabled: e.enabled,
        healthy: e.healthy,
        requests: e.stats.requests,
        errors: e.stats.errors,
        failovers: e.stats.failovers,
        in_flight: e.stats.inFlight,
        latency: {
          headers: e.stats.headers.summary(),
          first_token: e.stats.firstToken.summary(),
          total: e.stats.total.summary(),
        },
      }))),
    };
  }

  private readBody(req: IncomingMessage): Promise<string> {
    const limit = this.config.maxBodyBytes;
    return new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      let size = 0;
      req.on('data', (chunk: Buffer) => {
        size += chunk.length;
        if (size > limit) {
          reject(new HttpError(413, `request body exceeds ${limit} bytes`));
          req.resume();
          return;
        }
        chunks.push(chunk);
      });
      req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
      req.on('error', reject);
    });
  }

  /**
   * Try each endpoint in failover order until one returns response headers.
   * When no endpoint can take the request, wait and try again until the
   * deadline. So a short backend restart delays the request, but does not
   * fail it.
   */
  private async openBackend(
    route: ModelRoute, payload: string, signal: AbortSignal, requestId: string, deadline: number,
  ): Promise<{ endpoint: Endpoint; stream: BackendStream }> {
    for (let attempt = 0; ; attempt++) {
      const errors: string[] = [];
      for (const endpoint of route.attemptOrder()) {
        if (signal.aborted) break;
        endpoint.stats.requests++;
        try {
          const stream = await postChat(endpoint, payload, this.config, signal);
          endpoint.stats.headers.add(stream.headersMs);
          // 502/503/504 mean the backend itself is not able to serve: loading,
          // overloaded or behind a dead proxy. The next endpoint can serve.
          if ([502, 503, 504].includes(stream.status)) {
            endpoint.stats.errors++;
            endpoint.stats.failovers++;
            endpoint.markDown(`HTTP ${stream.status}`);
            errors.push(`${endpoint.name}: HTTP ${stream.status}`);
            log('warn', 'backend refused request; trying next', { request_id: requestId, backend: endpoint.name, status: stream.status });
            continue;
          }
          endpoint.markUp();
          return { endpoint, stream };
        } catch (error) {
          endpoint.stats.errors++;
          const message = (error as Error).message;
          if (!(error instanceof RetryableBackendError)) {
            throw new HttpError(502, `${endpoint.name}: ${message}`, 'backend_error');
          }
          endpoint.stats.failovers++;
          endpoint.markDown(message);
          errors.push(`${endpoint.name}: ${message}`);
          log('warn', 'backend unreachable; trying next', { request_id: requestId, backend: endpoint.name, error: message });
        }
      }
      if (signal.aborted) throw new HttpError(499, 'client closed the request', 'cancelled');
      const reason = errors.join('; ') || 'no enabled backends';
      const wait = retryDelayMs(attempt);
      if (Date.now() + wait > deadline) {
        throw new HttpError(503, `no backend could take the request (${reason})`, 'backend_unavailable', { 'Retry-After': String(RETRY_AFTER_S) });
      }
      this.metrics.retries++;
      log('warn', 'no backend could take the request; retrying', { request_id: requestId, attempt: attempt + 1, wait_ms: wait, error: reason });
      try {
        await sleep(wait, signal);
      } catch {
        throw new HttpError(499, 'client closed the request', 'cancelled');
      }
    }
  }

  /**
   * Append one row to the trace file, when one is set. For debugging only.
   * The file holds full prompts, so only the owner can read it. The write is
   * asynchronous, so a trace does not stop other streams. `row` can be JSON
   * text, to use a payload that is already serialized.
   */
  private trace(row: Obj | string): void {
    if (!this.config.traceFile || this.traceFailed) return;
    if (!this.traceStream) {
      const path = this.config.traceFile;
      const stream = createWriteStream(path, { flags: 'a', mode: 0o600 });
      // The mode of open(2) applies only to a new file. An existing file
      // gets the private mode when the stream opens it.
      stream.on('open', (fd: number) => fchmod(fd, 0o600, (error) => {
        if (error) log('warn', 'could not make the trace file private', { error: String(error) });
      }));
      stream.on('error', (error) => {
        log('warn', 'could not write the trace file; tracing stops', { error: String(error) });
        this.traceFailed = true;
        this.traceStream = null;
      });
      this.traceStream = stream;
    }
    this.traceStream.write((typeof row === 'string' ? row : JSON.stringify(row)) + '\n');
  }

  /** Resolve when all trace rows so far are in the file. */
  flushTrace(): Promise<void> {
    const stream = this.traceStream;
    if (!stream) return Promise.resolve();
    return new Promise((resolve) => stream.write('', () => resolve()));
  }

  private closeTrace(): Promise<void> {
    const stream = this.traceStream;
    this.traceStream = null;
    if (!stream) return Promise.resolve();
    return new Promise((resolve) => stream.end(() => resolve()));
  }

  private async handleResponses(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const requestId = newId('req', 16);
    const started = performance.now();
    const m = this.metrics;
    m.requests++;
    // A request that fails before the backend call. handle() writes its log
    // line, and the trace keeps the request with the reason.
    const reject = (error: unknown, request?: Obj): never => {
      m.clientErrors++;
      if (error instanceof HttpError) error.requestId = requestId;
      if (request !== undefined) this.trace({ request_id: requestId, request, error: (error as Error).message });
      throw error;
    };
    // Codex retries a 503. The next attempt reaches the new gateway process.
    const shuttingDown = () => new HttpError(503, 'the gateway is shutting down', 'unavailable', {
      'Retry-After': String(RETRY_AFTER_S), Connection: 'close',
    });
    if (this.draining) return reject(shuttingDown());

    let request: Obj;
    try {
      request = JSON.parse(await this.readBody(req));
    } catch (error) {
      return reject(error instanceof HttpError ? error : new HttpError(400, 'invalid JSON'));
    }
    // The body can arrive after the drain started.
    if (this.draining) return reject(shuttingDown());
    const route = this.routes.find((r) => r.config.id === request?.model);
    if (!route) {
      return reject(new HttpError(400, `unknown model: ${String(request?.model)}; this gateway serves ${this.routes.map((r) => r.config.id).join(', ')}`), request);
    }
    let chat: ChatRequest;
    try {
      chat = responsesToChat(request, {
        profile: route.config.profile,
        upstreamModel: route.upstreamModel,
        maxToolOutputChars: this.config.maxToolOutputChars,
        replayReasoning: this.config.replayReasoning,
      });
    } catch (error) {
      return reject(error instanceof RequestError ? new HttpError(400, error.message) : error, request);
    }
    // Serialize the payload once. Each backend attempt and the trace use it.
    const payload = JSON.stringify(chat.payload);
    if (this.config.traceFile) {
      this.trace(`{"request_id":${JSON.stringify(requestId)},"request":${JSON.stringify(request)},"payload":${payload}}`);
    }

    const streaming = request.stream === true;
    const controller = new AbortController();
    const requestTimer = this.config.requestTimeoutMs > 0
      ? setTimeout(() => controller.abort(new Error(`request exceeded ${this.config.requestTimeoutMs} ms`)), this.config.requestTimeoutMs)
      : undefined;
    const onClose = () => { if (!res.writableFinished) controller.abort(new Error('client disconnected')); };
    res.on('close', onClose);
    this.inflight.add(controller);
    m.inFlight++;

    const translator = new ChatStreamTranslator(String(request.model), chat.tools, this.config.emitReasoning);
    const deadline = Date.now() + this.config.retryWindowMs;
    let endpoint: Endpoint | null = null;
    /** The endpoint whose in-flight count this request holds. */
    let active: Endpoint | null = null;
    let firstTokenMs: number | null = null;
    let keepalive: ReturnType<typeof setInterval> | undefined;
    let startTimer: ReturnType<typeof setTimeout> | undefined;
    let outcome = 'failed';
    /** The reason of a failure that the translator does not hold. */
    let failure: string | undefined;
    /** True when an event after response.in_progress went to the client. */
    let sentOutput = false;
    let lastWriteAt = Date.now();
    const write = (events: ResponseEvent[]) => {
      if (!streaming || !events.length) return Promise.resolve();
      lastWriteAt = Date.now();
      return writeAll(res, events.map(sseFrame).join(''));
    };
    const release = () => {
      if (active) active.stats.inFlight--;
      active = null;
    };
    // The stream starts (headers, response.created, keepalives) when the
    // backend answers, or after streamStartMs when it does not answer yet.
    // After the start, an error goes to the client as response.failed.
    let committed = false;
    const commit = (): Promise<void> => {
      if (!streaming || committed || res.destroyed) return Promise.resolve();
      committed = true;
      if (startTimer) clearTimeout(startTimer);
      const headers: Record<string, string> = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'X-Pulse-Request-Id': requestId,
      };
      if (endpoint) headers['X-Pulse-Backend'] = endpoint.name;
      res.writeHead(200, headers);
      if (this.config.keepaliveMs > 0) {
        // Codex fails a stream after stream_idle_timeout_ms without an SSE
        // event, and SSE comment lines do not count. A long prefill sends no
        // token, so the gateway sends response.in_progress events.
        keepalive = setInterval(() => {
          if (res.destroyed || res.writableLength > 0) return;
          if (Date.now() - lastWriteAt < this.config.keepaliveMs) return;
          lastWriteAt = Date.now();
          res.write(sseFrame(translator.keepalive()));
        }, Math.max(50, Math.floor(this.config.keepaliveMs / 2)));
        keepalive.unref();
      }
      return write(translator.start());
    };
    if (streaming) {
      startTimer = setTimeout(() => void commit().catch(() => {}), this.config.streamStartMs);
      startTimer.unref();
    }

    try {
      for (let attempt = 0; ; attempt++) {
        const opened = await this.openBackend(route, payload, controller.signal, requestId, deadline);
        endpoint = opened.endpoint;
        active = endpoint;
        endpoint.stats.inFlight++;
        const backend = opened.stream;
        if (backend.status < 200 || backend.status >= 300 || !backend.body) {
          // A 4xx from the backend describes the request (for example a prompt
          // longer than the context window). Pass it to the client unchanged.
          outcome = 'client_error';
          m.clientErrors++;
          const message = backend.errorText || `backend HTTP ${backend.status}`;
          failure = message.slice(0, 1000);
          if (committed) {
            await write(translator.fail(message, backendErrorCode(message)));
            res.end();
          } else {
            sendJson(res, backend.status || 502, { error: { type: 'backend_error', code: backendErrorCode(message), message } });
          }
          return;
        }
        await commit();

        // Some servers close the stream after the finish reason without
        // [DONE]. That is still a complete answer. A stream without a finish
        // reason is not, and finishStream() fails it.
        let broken: string | null = null;
        try {
          for await (const data of sseData(backend.body)) {
            if (data === '[DONE]') break;
            let chunk: Obj;
            try { chunk = JSON.parse(data); } catch { continue; }
            const events = translator.push(chunk);
            if (firstTokenMs === null && translator.hasOutput) {
              firstTokenMs = performance.now() - started;
              endpoint.stats.firstToken.add(firstTokenMs);
            }
            if (events.length) {
              sentOutput = streaming;
              await write(events);
            }
          }
        } catch (error) {
          if (controller.signal.aborted || error instanceof BackendStreamError || sentOutput) throw error;
          broken = (error as Error).message;
        }
        if (broken === null && !translator.finishReason && !sentOutput) broken = 'backend stream ended before a finish reason';
        if (broken !== null) {
          // The client has no output of this attempt yet, so the gateway can
          // send the request again. It is stateless, and the prefix cache of
          // the backend makes the new prefill short.
          const wait = retryDelayMs(attempt);
          endpoint.stats.errors++;
          endpoint.markDown(broken);
          release();
          if (Date.now() + wait > deadline) throw new Error(broken);
          m.retries++;
          log('warn', 'backend stream broke before any output; retrying', {
            request_id: requestId, backend: endpoint.name, attempt: attempt + 1, wait_ms: wait, error: broken,
          });
          translator.restart();
          firstTokenMs = null;
          await sleep(wait, controller.signal);
          continue;
        }
        break;
      }

      const final = translator.finishStream();
      for (const item of translator.response.output) {
        // Codex answers such a call with a parse error. The warning shows how
        // often the backend sends one.
        if (item.type === 'function_call' && !isJsonObjectText(item.arguments)) {
          log('warn', 'tool call arguments are not a JSON object', {
            request_id: requestId, name: item.name, arguments: String(item.arguments).slice(0, 300),
          });
        }
      }
      await write(final);
      const response = translator.response;
      outcome = response.status;
      if (!streaming) {
        if (response.status === 'failed') sendJson(res, 502, { error: { type: 'backend_error', message: response.error?.message } });
        else sendJson(res, 200, response);
      } else {
        res.end();
      }
    } catch (error) {
      const aborted = controller.signal.aborted;
      const reason = aborted && controller.signal.reason instanceof Error ? controller.signal.reason.message : (error as Error).message;
      const clientGone = res.destroyed || reason === 'client disconnected';
      outcome = clientGone ? 'cancelled' : 'failed';
      failure = reason;
      if (error instanceof HttpError && !res.headersSent) {
        outcome = error.status === 499 ? 'cancelled' : error.status >= 500 ? 'failed' : 'client_error';
        error.requestId = requestId;
        error.logged = true;
        throw error;
      }
      if (active && !clientGone) active.stats.errors++;
      if (clientGone) return;
      if (res.headersSent) {
        // Codex retries a stream that ends with response.failed.
        const code = error instanceof HttpError ? error.code : 'backend_error';
        await write(translator.fail(reason, code)).catch(() => {});
        res.end();
      } else if (this.draining) {
        sendJson(res, 503, { error: { type: 'unavailable', message: reason } }, { 'Retry-After': String(RETRY_AFTER_S), Connection: 'close' });
      } else {
        sendJson(res, 502, { error: { type: 'backend_error', message: reason } });
      }
    } finally {
      if (requestTimer) clearTimeout(requestTimer);
      if (startTimer) clearTimeout(startTimer);
      if (keepalive) clearInterval(keepalive);
      res.off('close', onClose);
      controller.abort();
      this.inflight.delete(controller);
      m.inFlight--;
      release();
      if (endpoint) endpoint.stats.total.add(performance.now() - started);
      this.record(outcome, translator.response);
      log(outcome === 'failed' ? 'warn' : 'info', 'response', {
        request_id: requestId,
        response_id: translator.response.id,
        model: request.model,
        backend: endpoint?.name ?? null,
        stream: streaming,
        effort: request.reasoning?.effort ?? null,
        status: outcome,
        error: translator.response.error?.message ?? failure,
        first_token_ms: firstTokenMs === null ? null : Math.round(firstTokenMs),
        elapsed_ms: Math.round(performance.now() - started),
        usage: translator.response.usage,
        output_items: translator.response.output.map((item: Obj) => item.type),
      });
    }
  }

  private record(outcome: string, response: Obj): void {
    const m = this.metrics;
    if (outcome === 'completed') m.completed++;
    else if (outcome === 'incomplete') m.incomplete++;
    else if (outcome === 'cancelled') m.cancelled++;
    else if (outcome === 'failed') m.failed++;
    const usage = response.usage;
    if (usage) {
      m.inputTokens += usage.input_tokens ?? 0;
      m.cachedInputTokens += usage.input_tokens_details?.cached_tokens ?? 0;
      m.outputTokens += usage.output_tokens ?? 0;
      m.reasoningTokens += usage.output_tokens_details?.reasoning_tokens ?? 0;
    }
    m.toolCalls += response.output.filter((item: Obj) => item.type === 'function_call' || item.type === 'custom_tool_call').length;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { ...headers, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(text) });
  res.end(text);
}

/** Write with backpressure. Rejects when the client goes away. */
async function writeAll(res: ServerResponse, text: string): Promise<void> {
  if (!text) return;
  if (res.destroyed) throw new Error('client disconnected');
  if (res.write(text)) return;
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => { res.off('drain', onDrain); res.off('close', onClose); };
    const onDrain = () => { cleanup(); resolve(); };
    const onClose = () => { cleanup(); reject(new Error('client disconnected')); };
    res.once('drain', onDrain);
    res.once('close', onClose);
  });
}
