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
import { appendFileSync, readFileSync } from 'node:fs';
import type { GatewayConfig } from './config.js';
import { Endpoint, HealthChecker, ModelRoute, RetryableBackendError, postChat, closeAgents } from './backends.js';
import type { BackendStream } from './backends.js';
import { GatewayMetrics } from './metrics.js';
import { log } from './log.js';
import { sseData } from './sse.js';
import {
  ChatStreamTranslator, RequestError, newId, responsesToChat, sseFrame,
} from './translate.js';
import type { Obj, ResponseEvent } from './translate.js';

const VERSION = '1.0.0';
const HEARTBEAT_MS = 10_000;

class HttpError extends Error {
  constructor(readonly status: number, message: string, readonly code = 'invalid_request_error') { super(message); }
}

export class Gateway {
  readonly metrics = new GatewayMetrics();
  readonly routes: ModelRoute[];
  private readonly health: HealthChecker;
  private server: http.Server | null = null;
  private readonly inflight = new Set<AbortController>();
  private draining = false;
  private catalog: Obj[] | null = null;

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
    }
    this.server?.closeAllConnections();
    await closed;
    closeAgents();
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
      // HttpErrors from the Responses path are already in the `response` log line.
      if (!(error instanceof HttpError)) log('error', 'request failed', { path, error: String(error) });
      if (!res.headersSent) sendJson(res, status, { error: { type: code, message: (error as Error).message } });
      else res.end();
    }
  }

  private authorized(req: IncomingMessage): boolean {
    const key = process.env.PULSE_GATEWAY_API_KEY;
    if (!key) return true;
    return req.headers.authorization === `Bearer ${key}`;
  }

  private sendHealth(res: ServerResponse): void {
    const models = this.routes.map((route) => ({
      id: route.config.id,
      available: route.available,
      backends: route.endpoints.map((e) => ({
        name: e.name,
        base_url: e.config.baseUrl,
        enabled: e.enabled,
        healthy: e.healthy,
        last_check_at: e.lastCheckAt ? new Date(e.lastCheckAt).toISOString() : null,
        last_error: e.lastError,
      })),
    }));
    const ok = !this.draining && this.routes.every((r) => r.available);
    sendJson(res, ok ? 200 : 503, { status: this.draining ? 'draining' : ok ? 'ok' : 'degraded', version: VERSION, models });
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
        failed: m.failed, cancelled: m.cancelled, client_errors: m.clientErrors,
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

  /** Try each endpoint in failover order until one returns response headers. */
  private async openBackend(route: ModelRoute, payload: Obj, signal: AbortSignal, requestId: string): Promise<{ endpoint: Endpoint; stream: BackendStream }> {
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
    throw new HttpError(503, `no backend could take the request (${errors.join('; ') || 'no enabled backends'})`, 'backend_unavailable');
  }

  private async handleResponses(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const requestId = newId('req', 16);
    const started = performance.now();
    const m = this.metrics;
    m.requests++;
    if (this.draining) {
      m.clientErrors++;
      throw new HttpError(503, 'the gateway is shutting down', 'unavailable');
    }

    let request: Obj;
    try {
      request = JSON.parse(await this.readBody(req));
    } catch (error) {
      m.clientErrors++;
      if (error instanceof HttpError) throw error;
      throw new HttpError(400, 'invalid JSON');
    }
    const route = this.routes.find((r) => r.config.id === request?.model);
    if (!route) {
      m.clientErrors++;
      throw new HttpError(400, `unknown model: ${String(request?.model)}; this gateway serves ${this.routes.map((r) => r.config.id).join(', ')}`);
    }
    let chat;
    try {
      chat = responsesToChat(request, {
        profile: route.config.profile,
        upstreamModel: route.upstreamModel,
        maxToolOutputChars: this.config.maxToolOutputChars,
        replayReasoning: this.config.replayReasoning,
      });
    } catch (error) {
      m.clientErrors++;
      if (error instanceof RequestError) throw new HttpError(400, error.message);
      throw error;
    }

    if (this.config.traceFile) {
      // For debugging only. The file holds full prompts, so only the owner can read it.
      try {
        appendFileSync(this.config.traceFile, JSON.stringify({ request_id: requestId, request, payload: chat.payload }) + '\n', { mode: 0o600 });
      } catch (error) {
        log('warn', 'could not write the trace file', { error: String(error) });
      }
    }

    const streaming = request.stream === true;
    const controller = new AbortController();
    const requestTimer = setTimeout(
      () => controller.abort(new Error(`request exceeded ${this.config.requestTimeoutMs} ms`)),
      this.config.requestTimeoutMs,
    );
    const onClose = () => { if (!res.writableFinished) controller.abort(new Error('client disconnected')); };
    res.on('close', onClose);
    this.inflight.add(controller);
    m.inFlight++;

    const translator = new ChatStreamTranslator(String(request.model), chat.tools, this.config.emitReasoning);
    let endpoint: Endpoint | null = null;
    let firstTokenMs: number | null = null;
    let heartbeat: ReturnType<typeof setInterval> | undefined;
    let outcome = 'failed';
    const write = (events: ResponseEvent[]) => streaming ? writeAll(res, events.map(sseFrame).join('')) : Promise.resolve();

    try {
      const opened = await this.openBackend(route, chat.payload, controller.signal, requestId);
      endpoint = opened.endpoint;
      endpoint.stats.inFlight++;
      const backend = opened.stream;
      if (backend.status < 200 || backend.status >= 300 || !backend.body) {
        // A 4xx from the backend describes the request (for example a prompt
        // longer than the context window). Pass it to the client unchanged.
        outcome = 'client_error';
        m.clientErrors++;
        sendJson(res, backend.status || 502, {
          error: { type: 'backend_error', message: backend.errorText || `backend HTTP ${backend.status}` },
        });
        return;
      }

      if (streaming) {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'X-Accel-Buffering': 'no',
          'X-Pulse-Backend': endpoint.name,
          'X-Pulse-Request-Id': requestId,
        });
        // SSE comments keep proxies and the client socket active while the
        // backend runs a long prefill.
        heartbeat = setInterval(() => {
          if (!res.destroyed && res.writableLength === 0) res.write(': pulse heartbeat\n\n');
        }, HEARTBEAT_MS);
        heartbeat.unref();
      }
      await write(translator.start());

      // Some servers close the stream after the finish reason without [DONE].
      // That is still a complete answer. A stream without a finish reason is
      // not, and finishStream() fails it.
      for await (const data of sseData(backend.body)) {
        if (data === '[DONE]') break;
        let chunk: Obj;
        try { chunk = JSON.parse(data); } catch { continue; }
        const events = translator.push(chunk);
        if (firstTokenMs === null && translator.hasOutput) {
          firstTokenMs = performance.now() - started;
          endpoint.stats.firstToken.add(firstTokenMs);
        }
        if (events.length) await write(events);
      }
      const final = translator.finishStream();
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
      if (error instanceof HttpError && !res.headersSent) {
        outcome = error.status === 499 ? 'cancelled' : error.status >= 500 ? 'failed' : 'client_error';
        throw error;
      }
      if (endpoint && !clientGone) endpoint.stats.errors++;
      if (clientGone) return;
      if (res.headersSent) {
        await write(translator.fail(reason)).catch(() => {});
        res.end();
      } else {
        sendJson(res, 502, { error: { type: 'backend_error', message: reason } });
      }
    } finally {
      clearTimeout(requestTimer);
      if (heartbeat) clearInterval(heartbeat);
      res.off('close', onClose);
      controller.abort();
      this.inflight.delete(controller);
      m.inFlight--;
      if (endpoint) {
        endpoint.stats.inFlight--;
        endpoint.stats.total.add(performance.now() - started);
      }
      this.record(outcome, translator.response);
      log(outcome === 'failed' ? 'warn' : 'info', 'response', {
        request_id: requestId,
        response_id: translator.response.id,
        model: request.model,
        backend: endpoint?.name ?? null,
        stream: streaming,
        effort: request.reasoning?.effort ?? null,
        status: outcome,
        error: translator.response.error?.message ?? undefined,
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

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(text) });
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
