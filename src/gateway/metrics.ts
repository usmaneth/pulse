// In-process counters for the /metrics endpoint. The values reset when the
// process restarts. They are JSON, not Prometheus text, because the only
// consumers today are curl, jq and the status scripts.

const WINDOW = 256;

/** Fixed-size window of recent samples, for percentiles. */
export class LatencyWindow {
  private readonly samples: number[] = [];
  private next = 0;
  count = 0;
  sum = 0;

  add(ms: number): void {
    this.count++;
    this.sum += ms;
    if (this.samples.length < WINDOW) this.samples.push(ms);
    else this.samples[this.next] = ms;
    this.next = (this.next + 1) % WINDOW;
  }

  summary(): { count: number; mean_ms: number | null; p50_ms: number | null; p95_ms: number | null; max_recent_ms: number | null } {
    if (!this.samples.length) return { count: this.count, mean_ms: null, p50_ms: null, p95_ms: null, max_recent_ms: null };
    const sorted = [...this.samples].sort((a, b) => a - b);
    const at = (q: number) => round(sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]);
    return {
      count: this.count,
      mean_ms: round(this.sum / this.count),
      p50_ms: at(0.5),
      p95_ms: at(0.95),
      max_recent_ms: round(sorted[sorted.length - 1]),
    };
  }
}

const round = (value: number) => Math.round(value * 10) / 10;

export class EndpointStats {
  requests = 0;
  errors = 0;
  /** Requests that this endpoint could not take, so the next endpoint got them. */
  failovers = 0;
  inFlight = 0;
  readonly headers = new LatencyWindow();
  readonly firstToken = new LatencyWindow();
  readonly total = new LatencyWindow();
}

export class GatewayMetrics {
  readonly startedAt = Date.now();
  requests = 0;
  inFlight = 0;
  completed = 0;
  incomplete = 0;
  failed = 0;
  cancelled = 0;
  clientErrors = 0;
  /** Backend attempts that the gateway repeated before the first output. */
  retries = 0;
  inputTokens = 0;
  cachedInputTokens = 0;
  outputTokens = 0;
  reasoningTokens = 0;
  toolCalls = 0;
}
