// Backend endpoints: health checks, failover order and the HTTP client for
// chat-completions streams.
//
// The client uses node:http directly instead of fetch. fetch cannot set a
// connect timeout, and a host that drops packets (for example spark2 when its
// server binds loopback only) would hold each request for the full default
// connect timeout before the gateway could try the next endpoint.

import http from 'node:http';
import https from 'node:https';
import { Readable } from 'node:stream';
import type { EndpointConfig, GatewayConfig, ModelConfig } from './config.js';
import { healthUrlFor } from './config.js';
import { EndpointStats } from './metrics.js';
import { log } from './log.js';
import type { Obj } from './translate.js';

export class Endpoint {
  readonly stats = new EndpointStats();
  /** null until the first health check finishes. */
  healthy: boolean | null = null;
  lastCheckAt: number | null = null;
  /** Time of the last health check or request that the endpoint answered. */
  lastOkAt: number | null = null;
  lastError: string | null = null;
  consecutiveFailures = 0;

  constructor(readonly config: EndpointConfig) {}

  get name(): string { return this.config.name; }
  get enabled(): boolean { return this.config.enabled !== false; }

  markUp(): void {
    if (this.healthy === false) log('info', 'backend is healthy again', { backend: this.name });
    this.healthy = true;
    this.lastOkAt = Date.now();
    this.consecutiveFailures = 0;
    this.lastError = null;
  }

  markDown(reason: string): void {
    if (this.healthy !== false) log('warn', 'backend is unhealthy', { backend: this.name, reason });
    this.healthy = false;
    this.consecutiveFailures++;
    this.lastError = reason;
  }
}

/** One model id with its ordered endpoint list. */
export class ModelRoute {
  readonly endpoints: Endpoint[];

  constructor(readonly config: ModelConfig) {
    this.endpoints = config.endpoints.map((e) => new Endpoint(e));
  }

  get upstreamModel(): string { return this.config.upstreamModel ?? this.config.id; }

  /**
   * The order in which to try endpoints for one request: healthy and unknown
   * endpoints first in config order, then unhealthy ones as a last resort.
   * A health check can be up to one interval old, so an endpoint marked down
   * can already be back.
   */
  attemptOrder(): Endpoint[] {
    const enabled = this.endpoints.filter((e) => e.enabled);
    return [...enabled.filter((e) => e.healthy !== false), ...enabled.filter((e) => e.healthy === false)];
  }

  get available(): boolean {
    return this.endpoints.some((e) => e.enabled && e.healthy !== false);
  }
}

/** A backend error that means "try the next endpoint". */
export class RetryableBackendError extends Error {}

export interface BackendStream {
  status: number;
  /** Error body text when status is not 2xx. */
  errorText?: string;
  body?: ReadableStream<Uint8Array>;
  headersMs: number;
}

interface RequestTimeouts {
  connectTimeoutMs: number;
  headersTimeoutMs: number;
  idleTimeoutMs: number;
}

const agentHttp = new http.Agent({ keepAlive: true, maxSockets: 64 });
const agentHttps = new https.Agent({ keepAlive: true, maxSockets: 64 });

/**
 * POST a chat-completions request and return when the response headers
 * arrive. Connection failures throw RetryableBackendError. The body stream
 * fails when no byte arrives for `idleTimeoutMs`. The payload can be JSON
 * text, so that a caller that retries serializes it only once.
 */
export function postChat(
  endpoint: Endpoint,
  payload: Obj | string,
  timeouts: RequestTimeouts,
  signal: AbortSignal,
): Promise<BackendStream> {
  const url = new URL(`${endpoint.config.baseUrl}/chat/completions`);
  const body = Buffer.from(typeof payload === 'string' ? payload : JSON.stringify(payload));
  const client = url.protocol === 'https:' ? https : http;
  const started = performance.now();
  return new Promise((resolve, reject) => {
    let connected = false;
    let settled = false;
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };
    const req = client.request(url, {
      method: 'POST',
      agent: url.protocol === 'https:' ? agentHttps : agentHttp,
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': body.length,
        Accept: 'text/event-stream',
      },
    });
    const connectTimer = setTimeout(() => {
      if (!connected) req.destroy(new RetryableBackendError(`connect timeout after ${timeouts.connectTimeoutMs} ms`));
    }, timeouts.connectTimeoutMs);
    const headersTimer = setTimeout(() => {
      req.destroy(new Error(`no response headers after ${timeouts.headersTimeoutMs} ms`));
    }, timeouts.headersTimeoutMs);
    const onAbort = () => req.destroy(new Error('request aborted'));
    signal.addEventListener('abort', onAbort, { once: true });
    const cleanup = () => {
      clearTimeout(connectTimer);
      clearTimeout(headersTimer);
    };

    req.on('socket', (socket) => {
      // A reused keep-alive socket is already connected.
      if (!(socket as { connecting?: boolean }).connecting) connected = true;
      else socket.once('connect', () => { connected = true; });
    });
    req.on('error', (error: Error & { code?: string }) => {
      signal.removeEventListener('abort', onAbort);
      if (error instanceof RetryableBackendError) return fail(error);
      // Errors before the request reached the server are safe to retry on the
      // next endpoint. The request is stateless, so a replay has no side effect.
      const retryable = !connected || ['ECONNREFUSED', 'ECONNRESET', 'EHOSTUNREACH', 'ENETUNREACH', 'EPIPE'].includes(error.code ?? '');
      fail(retryable && !signal.aborted ? new RetryableBackendError(error.message || error.code || 'connection error') : error);
    });
    req.on('response', (res) => {
      cleanup();
      const headersMs = performance.now() - started;
      const status = res.statusCode ?? 0;
      // Idle timeout for the body. It resets on every socket read. The
      // socket goes back to the keep-alive pool after the response, so the
      // listener must go when the response closes. Else each request on the
      // socket adds one more listener that holds its old response.
      const socket = res.socket;
      const onIdle = () => res.destroy(new Error(`backend idle for ${timeouts.idleTimeoutMs} ms`));
      socket?.setTimeout(timeouts.idleTimeoutMs, onIdle);
      res.on('close', () => {
        signal.removeEventListener('abort', onAbort);
        socket?.setTimeout(0);
        socket?.off('timeout', onIdle);
      });
      if (status < 200 || status >= 300) {
        const chunks: Buffer[] = [];
        res.on('data', (c: Buffer) => { if (chunks.length < 64) chunks.push(c); });
        res.on('end', () => {
          settled = true;
          resolve({ status, errorText: Buffer.concat(chunks).toString('utf8').slice(0, 4000), headersMs });
        });
        res.on('error', (error) => fail(error));
        return;
      }
      settled = true;
      resolve({ status, body: Readable.toWeb(res) as ReadableStream<Uint8Array>, headersMs });
    });
    req.end(body);
  });
}

/** Probe one endpoint. A 200 from the health URL means healthy. */
export function probe(endpoint: Endpoint, timeoutMs: number): Promise<void> {
  const url = new URL(healthUrlFor(endpoint.config));
  const client = url.protocol === 'https:' ? https : http;
  return new Promise((resolve) => {
    const req = client.get(url, { agent: url.protocol === 'https:' ? agentHttps : agentHttp }, (res) => {
      res.resume();
      if (res.statusCode === 200) endpoint.markUp();
      else endpoint.markDown(`health returned HTTP ${res.statusCode}`);
      endpoint.lastCheckAt = Date.now();
      resolve();
    });
    req.setTimeout(timeoutMs, () => req.destroy(new Error(`health timeout after ${timeoutMs} ms`)));
    req.on('error', (error: Error & { code?: string }) => {
      endpoint.markDown(error.message || error.code || 'health check failed');
      endpoint.lastCheckAt = Date.now();
      resolve();
    });
  });
}

/** Runs health checks for all endpoints on a fixed interval. */
export class HealthChecker {
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(private readonly routes: ModelRoute[], private readonly config: GatewayConfig) {}

  async checkAll(): Promise<void> {
    const all = this.routes.flatMap((r) => r.endpoints).filter((e) => e.enabled);
    await Promise.all(all.map((e) => probe(e, this.config.healthTimeoutMs)));
  }

  start(): void {
    void this.checkAll();
    if (this.config.healthIntervalMs > 0) {
      this.timer = setInterval(() => void this.checkAll(), this.config.healthIntervalMs);
      this.timer.unref();
    }
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }
}

export function closeAgents(): void {
  agentHttp.destroy();
  agentHttps.destroy();
}
