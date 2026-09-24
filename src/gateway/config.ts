// Gateway configuration. Sources, in order of precedence:
//
//   1. environment variables (PULSE_GATEWAY_*, PULSE_QWEN_*)
//   2. a JSON file named by PULSE_GATEWAY_CONFIG or `--config <path>`
//   3. the defaults below
//
// See docs/QWEN38.md and config/qwen38-gateway.json.

import { readFileSync } from 'node:fs';
import type { ProfileName } from './translate.js';
import { defaultWarmupConfig } from './warmer.js';
import type { WarmupConfig } from './warmer.js';

export interface EndpointConfig {
  /** Short name for logs and metrics, for example `spark1`. */
  name: string;
  /** OpenAI-compatible base URL that ends in /v1. */
  baseUrl: string;
  /** Health URL. The default is `<origin of baseUrl>/health`. */
  healthUrl?: string;
  /** Disabled endpoints stay in the config but get no traffic. */
  enabled?: boolean;
}

export interface ModelConfig {
  /** The model slug that Codex sends. */
  id: string;
  /** The name the backend serves. The default is `id`. */
  upstreamModel?: string;
  profile: ProfileName;
  /** Endpoints in failover order. The first healthy endpoint gets the request. */
  endpoints: EndpointConfig[];
}

export interface GatewayConfig {
  host: string;
  port: number;
  models: ModelConfig[];
  /** Emit backend reasoning text as Responses reasoning items. */
  emitReasoning: boolean;
  /** Send the text of earlier reasoning items back to the model. */
  replayReasoning: boolean;
  /** Append each request and its translated payload to this JSONL file. */
  traceFile?: string;
  /** Cap for one tool output in characters. 0 disables the cap. */
  maxToolOutputChars: number;
  maxBodyBytes: number;
  /** Time to wait for the backend response headers (includes prefill). */
  headersTimeoutMs: number;
  /** Time without any byte from the backend before the stream fails. */
  idleTimeoutMs: number;
  /**
   * Upper limit for one request from start to end. 0 sets no limit, so a long
   * generation that streams tokens never gets cut. The idle limit still stops
   * a backend that goes silent.
   */
  requestTimeoutMs: number;
  /**
   * Time in which the gateway retries a request that has no output yet: the
   * backend refuses the connection, returns 502/503/504, or drops the stream
   * before the first token. A backend restart shorter than this does not end
   * the conversation. 0 tries each endpoint once.
   */
  retryWindowMs: number;
  /**
   * Interval of the keepalive event on a stream that has no other event. Codex
   * ends a stream after stream_idle_timeout_ms without an SSE event, and it
   * does not count SSE comment lines. 0 turns the keepalive off.
   */
  keepaliveMs: number;
  /**
   * Time that a streaming request waits for the backend before the gateway
   * sends the response headers, response.created and keepalives. A backend
   * error before this time goes to the client as an HTTP status.
   */
  streamStartMs: number;
  /** Time to connect to a backend before the gateway tries the next one. */
  connectTimeoutMs: number;
  healthIntervalMs: number;
  healthTimeoutMs: number;
  /** Time that in-flight requests get to finish after SIGTERM. */
  shutdownGraceMs: number;
  /** Prefill of the recent prompt prefixes when a backend becomes healthy. */
  warmup: WarmupConfig;
}

export const DEFAULT_QWEN_MODEL = 'qwen3.8-flash-next';
export const DEFAULT_QWEN_URL = 'http://127.0.0.1:8888/v1';

export function defaultConfig(): GatewayConfig {
  return {
    host: '127.0.0.1',
    port: 8800,
    models: [{
      id: DEFAULT_QWEN_MODEL,
      profile: 'qwen38',
      endpoints: [{ name: 'spark1', baseUrl: DEFAULT_QWEN_URL }],
    }],
    emitReasoning: true,
    replayReasoning: true,
    maxToolOutputChars: 12_000,
    maxBodyBytes: 64 * 1024 * 1024,
    // A cold 500k-token prefill takes minutes. vLLM can send the headers
    // before the prefill or after it, and it sends no byte during the
    // prefill. So both limits must be longer than the longest legitimate
    // prefill.
    headersTimeoutMs: 900_000,
    idleTimeoutMs: 900_000,
    requestTimeoutMs: 0,
    retryWindowMs: 180_000,
    keepaliveMs: 10_000,
    streamStartMs: 3_000,
    connectTimeoutMs: 3_000,
    healthIntervalMs: 10_000,
    healthTimeoutMs: 3_000,
    shutdownGraceMs: 30_000,
    warmup: defaultWarmupConfig(),
  };
}

/**
 * Parse `name=url,name=url` (or plain `url,url`) into endpoints. The order of
 * the list is the failover order.
 */
export function parseEndpointList(value: string): EndpointConfig[] {
  return value.split(',').map((entry) => entry.trim()).filter(Boolean).map((entry, i) => {
    const eq = entry.indexOf('=');
    const hasName = eq > 0 && !entry.slice(0, eq).includes('/');
    return {
      name: hasName ? entry.slice(0, eq) : `backend${i + 1}`,
      baseUrl: hasName ? entry.slice(eq + 1) : entry,
    };
  });
}

export function healthUrlFor(endpoint: EndpointConfig): string {
  if (endpoint.healthUrl) return endpoint.healthUrl;
  return `${new URL(endpoint.baseUrl).origin}/health`;
}

function num(env: NodeJS.ProcessEnv, key: string): number | undefined {
  const raw = env[key];
  if (raw === undefined || raw === '') return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) throw new Error(`${key} must be a non-negative number`);
  return value;
}

export function validateConfig(config: GatewayConfig): GatewayConfig {
  if (!config.models.length) throw new Error('config has no models');
  const ids = new Set<string>();
  for (const model of config.models) {
    if (!model.id) throw new Error('every model needs an id');
    if (ids.has(model.id)) throw new Error(`duplicate model id: ${model.id}`);
    ids.add(model.id);
    if (model.profile !== 'qwen38' && model.profile !== 'llamacpp') {
      throw new Error(`model ${model.id}: unknown profile ${String(model.profile)}`);
    }
    if (!model.endpoints?.length) throw new Error(`model ${model.id}: no endpoints`);
    const names = new Set<string>();
    for (const endpoint of model.endpoints) {
      if (!endpoint.name || !endpoint.baseUrl) throw new Error(`model ${model.id}: every endpoint needs a name and a baseUrl`);
      if (names.has(endpoint.name)) throw new Error(`model ${model.id}: duplicate endpoint name ${endpoint.name}`);
      names.add(endpoint.name);
      new URL(endpoint.baseUrl); // throws on a bad URL
      endpoint.baseUrl = endpoint.baseUrl.replace(/\/+$/, '');
    }
    if (!model.endpoints.some((e) => e.enabled !== false)) {
      throw new Error(`model ${model.id}: all endpoints are disabled`);
    }
  }
  if (!Number.isInteger(config.port) || config.port < 0 || config.port > 65535) {
    throw new Error(`invalid port: ${config.port}`);
  }
  return config;
}

export function loadConfig(
  env: NodeJS.ProcessEnv = process.env,
  argv: string[] = process.argv.slice(2),
): GatewayConfig {
  const config = defaultConfig();
  const flag = argv.indexOf('--config');
  const file = flag >= 0 ? argv[flag + 1] : env.PULSE_GATEWAY_CONFIG;
  if (file) {
    const fromFile = JSON.parse(readFileSync(file, 'utf8')) as Partial<GatewayConfig>;
    const warmup = { ...config.warmup, ...fromFile.warmup };
    Object.assign(config, fromFile);
    config.warmup = warmup;
  }

  if (env.PULSE_GATEWAY_HOST) config.host = env.PULSE_GATEWAY_HOST;
  config.port = num(env, 'PULSE_GATEWAY_PORT') ?? config.port;

  // Shortcut for the common case: one Qwen3.8 model and its endpoint list.
  // The env list replaces the endpoints of the qwen38 model from the file.
  if (env.PULSE_QWEN_BACKENDS || env.PULSE_QWEN_MODEL) {
    let qwen = config.models.find((m) => m.profile === 'qwen38');
    if (!qwen) {
      qwen = { id: DEFAULT_QWEN_MODEL, profile: 'qwen38', endpoints: [{ name: 'spark1', baseUrl: DEFAULT_QWEN_URL }] };
      config.models.unshift(qwen);
    }
    if (env.PULSE_QWEN_BACKENDS) qwen.endpoints = parseEndpointList(env.PULSE_QWEN_BACKENDS);
    if (env.PULSE_QWEN_MODEL) qwen.id = env.PULSE_QWEN_MODEL;
    if (env.PULSE_QWEN_UPSTREAM_MODEL) qwen.upstreamModel = env.PULSE_QWEN_UPSTREAM_MODEL;
  }

  if (env.PULSE_GATEWAY_EMIT_REASONING) config.emitReasoning = env.PULSE_GATEWAY_EMIT_REASONING !== '0';
  if (env.PULSE_GATEWAY_REPLAY_REASONING) config.replayReasoning = env.PULSE_GATEWAY_REPLAY_REASONING !== '0';
  if (env.PULSE_GATEWAY_TRACE_FILE) config.traceFile = env.PULSE_GATEWAY_TRACE_FILE;
  config.maxToolOutputChars = num(env, 'PULSE_GATEWAY_MAX_TOOL_OUTPUT_CHARS') ?? config.maxToolOutputChars;
  config.maxBodyBytes = num(env, 'PULSE_GATEWAY_MAX_BODY_BYTES') ?? config.maxBodyBytes;
  config.headersTimeoutMs = num(env, 'PULSE_GATEWAY_HEADERS_TIMEOUT_MS') ?? config.headersTimeoutMs;
  config.idleTimeoutMs = num(env, 'PULSE_GATEWAY_IDLE_TIMEOUT_MS') ?? config.idleTimeoutMs;
  config.requestTimeoutMs = num(env, 'PULSE_GATEWAY_REQUEST_TIMEOUT_MS') ?? config.requestTimeoutMs;
  config.connectTimeoutMs = num(env, 'PULSE_GATEWAY_CONNECT_TIMEOUT_MS') ?? config.connectTimeoutMs;
  config.retryWindowMs = num(env, 'PULSE_GATEWAY_RETRY_WINDOW_MS') ?? config.retryWindowMs;
  config.keepaliveMs = num(env, 'PULSE_GATEWAY_KEEPALIVE_MS') ?? config.keepaliveMs;
  config.streamStartMs = num(env, 'PULSE_GATEWAY_STREAM_START_MS') ?? config.streamStartMs;
  config.healthIntervalMs = num(env, 'PULSE_GATEWAY_HEALTH_INTERVAL_MS') ?? config.healthIntervalMs;
  config.healthTimeoutMs = num(env, 'PULSE_GATEWAY_HEALTH_TIMEOUT_MS') ?? config.healthTimeoutMs;
  config.shutdownGraceMs = num(env, 'PULSE_GATEWAY_SHUTDOWN_GRACE_MS') ?? config.shutdownGraceMs;
  if (env.PULSE_GATEWAY_WARMUP) config.warmup.enabled = env.PULSE_GATEWAY_WARMUP !== '0';
  config.warmup.sessions = num(env, 'PULSE_GATEWAY_WARMUP_SESSIONS') ?? config.warmup.sessions;
  config.warmup.sessionMaxAgeMs = num(env, 'PULSE_GATEWAY_WARMUP_SESSION_MAX_AGE_MS') ?? config.warmup.sessionMaxAgeMs;
  if (env.PULSE_GATEWAY_WARMUP_STATE_FILE) config.warmup.stateFile = env.PULSE_GATEWAY_WARMUP_STATE_FILE;
  return validateConfig(config);
}
