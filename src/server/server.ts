import http from 'node:http';
import { CheckpointManager } from './checkpoint.js';
import { JevDecisionClient, SpeculationDecision, MemoryAdmissionDecision } from '../jev/client.js';
import { NativePulseEngine } from './native_ffi.js';
export interface ServerConfig {
  port: number;
  host: string;
  backendUrl: string;
}

export class PulseServer {
  private readonly config: ServerConfig;
  private readonly jev: JevDecisionClient;
  private readonly nativeEngine = new NativePulseEngine();
  private server: http.Server | null = null;

  private totalRequests = 0;
  private totalTokensGenerated = 0;
  private activeStreams = 0;
  private peakTokensPerSec = 0; // measured at runtime; no seeded value
  private lastJevDecision: SpeculationDecision | null = null;
  private checkpoints: CheckpointManager;
  private lastCheckpointRestore: { chars: number; bytes: number } | null = null;

  constructor(config: Partial<ServerConfig> = {}) {
    this.config = {
      port: config.port ?? 8000,
      host: config.host ?? '0.0.0.0',
      backendUrl: config.backendUrl ?? 'http://127.0.0.1:8085',
    };
    this.jev = new JevDecisionClient(2000);
    // DISABLED BY DEFAULT - measured slower, so this is opt-in via PULSE_CKPT=1.
    // The restore primitive is fast in isolation (58.7 ms for 275 MB). But
    // llama.cpp already keeps evicted slot state in a RAM cache under
    // --cache-ram -1 and restores it natively. Forcing a disk restore throws
    // that better cache away, falls back to an older and shorter saved prefix,
    // then re-prefills the difference - and pays ~150 ms per save to write
    // ~300 MB. Paired A/B over two mutually evicting sessions
    // (bench/ckpt_evict.py, -np 1): 732 ms off vs 3080 ms on. 4.2x WORSE.
    // See bench/RESULTS.md.
    const ckptEnabled = process.env.PULSE_CKPT === '1';
    this.checkpoints = new CheckpointManager(
      this.config.backendUrl,
      process.env.PULSE_CKPT_PATH ?? '/tmp/slotsave/',
      ckptEnabled ? Number(process.env.PULSE_CKPT_MIN_CHARS ?? 4000) : Number.MAX_SAFE_INTEGER,
    );
  }

  start(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = http.createServer(async (req, res) => {
        const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);

        if (req.method === 'GET' && (url.pathname === '/health' || url.pathname === '/ready')) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            status: 'ready',
            engine: 'pulse',
            version: '1.0.0',
            hardware: 'NVIDIA GB10 (sm_121, 128GB LPDDR5X)',
            backend: this.config.backendUrl,
          }));
          return;
        }

        if (req.method === 'GET' && url.pathname === '/status') {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(
            JSON.stringify({
              engine: 'pulse',
              version: '1.0.0',
              target_device: 'NVIDIA GB10 (48 SMs, sm_121, 128 GB unified LPDDR5X)',
              role: 'HTTP proxy in front of llama.cpp; not a standalone engine. It loads no weights and runs no forward pass.',
              active_streams: this.activeStreams,
              total_requests: this.totalRequests,
              total_tokens_generated: this.totalTokensGenerated,
              peak_decode_toks_per_sec: this.peakTokensPerSec,
              // Accurate rather than flattering: the hot-path K decision is a local
              // regex, not a Jev call, and llama.cpp ignores per-request K anyway
              // (`#if 0` in tools/server/server-schema.cpp). See src/jev/client.ts.
              speculation_k_decision: 'local heuristic, advisory only (backend ignores per-request K)',
              jev_gateway_used_for: ['memory_admission'],
              checkpoints: this.checkpoints.getStats(),
              last_checkpoint_restore: this.lastCheckpointRestore,
              last_jev_decision: this.lastJevDecision,
            }, null, 2)
          );
          return;
        }

        // OpenAI Models discovery endpoint
        if (req.method === 'GET' && (url.pathname === '/v1/models' || url.pathname === '/models')) {
          try {
            const resp = await fetch(`${this.config.backendUrl}/v1/models`);
            const data = await resp.json() as { data?: Array<Record<string, unknown>> };
            if (data && Array.isArray(data.data)) {
              data.data.unshift({
                id: 'pulse-bonsai-2-tp2',
                object: 'model',
                created: Math.floor(Date.now() / 1000),
                owned_by: 'pulse-engine',
              });
              data.data.unshift({
                id: 'spark-splash-bonsai-2',
                object: 'model',
                created: Math.floor(Date.now() / 1000),
                owned_by: 'spark-splash',
              });
            }
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(data));
          } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: msg }));
          }
          return;
        }

        if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/chat/completions')) {
          await this.handleChatCompletions(req, res);
          return;
        }

        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Endpoint not found' }));
      });

      this.server.listen(this.config.port, this.config.host, () => {
        console.log(`\n=======================================================`);
        console.log(` PULSE: Hardware-Specialized Blackwell Engine Live`);
        console.log(` Port: http://${this.config.host}:${this.config.port}`);
        console.log(` Backend Engine: ${this.config.backendUrl} (GB10 sm_121)`);
        console.log(` Decision Engine: TypeSafe AI Jev (System One)`);
        console.log(` Endpoints: /v1/chat/completions, /status, /health`);
        console.log(`=======================================================\n`);
        resolve();
      });

      this.server.on('error', reject);
    });
  }

  stop(): Promise<void> {
    this.nativeEngine.destroy();
    return new Promise((resolve) => {
      if (this.server) {
        this.server.close(() => resolve());
      } else {
        resolve();
      }
    });
  }

  private async handleChatCompletions(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const body = await this.readRequestBody(req);
    let parsed: any = {};
    try {
      parsed = JSON.parse(body);
    } catch {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Invalid JSON' }));
      return;
    }

    const messages = parsed.messages ?? [];
    const stream = parsed.stream ?? false;
    const lastUserMessage = messages.filter((m: any) => m.role === 'user').pop()?.content ?? '';

    this.totalRequests++;
    this.activeStreams++;

    const isCode = /def |function|class |import |\{|\}|const |let |return /.test(lastUserMessage);
    const isMath = /solve|calculate|\d+\s*[\+\-\*\/]\s*\d+|equation|probability/.test(lastUserMessage.toLowerCase());
    const domain = isMath ? 'math' : isCode ? 'code' : 'chat';

    const t0 = performance.now();
    const kDecision: SpeculationDecision = await this.jev.decideSpeculationK({
      promptSnippet: lastUserMessage,
      taskDomain: domain,
      recentAcceptanceRate: isCode || isMath ? 0.85 : 0.50,
    });
    const jevLatency = performance.now() - t0;
    this.lastJevDecision = kDecision;

    // 2. Forward to Live GB10 Inference Engine with client abort propagation
    const abortController = new AbortController();
    const onClientClose = () => {
      if (!res.writableEnded) {
        abortController.abort();
      }
    };
    req.on('close', onClientClose);

    try {
      const backendPayload = {
        ...parsed,
        stream,
        // Prefix caching. An agent re-sends a growing context every turn; without
        // this the backend re-prefills the whole thing each time. A cold 131k
        // prefill measured 211.7 s on this hardware, so this is the single
        // highest-value flag for long-context agentic use. Callers may override.
        cache_prompt: parsed.cache_prompt ?? true,
        // LIVE as of patches/llama-per-request-spec-n-max.patch. Stock llama.cpp
        // compiles per-request speculative params out of its server behind
        // `#if 0`, so this field used to be discarded. The patch exposes
        // `speculative.n_max`, aliases it to this name, and applies it in
        // server_slot::get_n_draft_max(). Measured: n_max=1 -> 37.10 tok/s,
        // n_max=3 -> 53.09 tok/s, unset -> server default. Against an unpatched
        // backend the field is simply ignored, so sending it stays safe.
        spec_draft_n_max: kDecision.k,
      };

      // Restore the longest matching checkpoint before forwarding.
      // OFF BY DEFAULT: measured 4.2x SLOWER than llama.cpp's own RAM cache
      // (--cache-ram -1), which already restores evicted slot state natively.
      // This call is a no-op unless PULSE_CKPT=1. See bench/RESULTS.md Round 23.
      const ckptKey = messages.map((m: { role?: string; content?: string }) =>
        `${m.role ?? ''}:${m.content ?? ''}`).join('\n');
      const restored = await this.checkpoints.restoreBest(ckptKey).catch(() => null);
      if (restored) {
        this.lastCheckpointRestore = { chars: restored.text.length, bytes: restored.bytes };
      }

      const backendResponse = await fetch(`${this.config.backendUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(backendPayload),
        signal: abortController.signal,
      });

      if (!backendResponse.ok) {
        const errorText = await backendResponse.text();
        if (!res.headersSent) {
          res.writeHead(backendResponse.status, { 'Content-Type': 'application/json' });
          res.end(errorText);
        }
        return;
      }

      if (stream) {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'X-Pulse-Speculation-K': String(kDecision.k),
          'X-Pulse-Confidence': String(kDecision.confidence),
        });

        const reader = backendResponse.body?.getReader();
        if (!reader) {
          res.end();
          return;
        }

        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunkStr = decoder.decode(value, { stream: true });
          res.write(chunkStr);
        }
        res.end();
        // Checkpoint the slot now that it holds this full context, so a later
        // turn that edits earlier text can restore instead of re-prefilling.
        void this.checkpoints.save(ckptKey).catch(() => null);
      } else {
        const json = await backendResponse.json() as Record<string, unknown>;
        const timings = (json.timings ?? {}) as Record<string, number>;
        const toksSec = timings.predicted_per_second ?? 0;
        if (toksSec > this.peakTokensPerSec) {
          this.peakTokensPerSec = toksSec;
        }
        const usage = (json.usage ?? {}) as Record<string, number>;
        this.totalTokensGenerated += usage.completion_tokens ?? 0;

        json.pulse_meta = {
          k_speculation: kDecision.k,
          jev_confidence: kDecision.confidence,
          jev_latency_ms: jevLatency,
          measured_toks_per_sec: toksSec,
          draft_acceptance_rate: timings.draft_n ? (timings.draft_n_accepted / timings.draft_n) : null,
          device: 'NVIDIA GB10 (sm_121, 128GB LPDDR5X)',
          tensor_parallel_world_size: 2,
        };
        void this.checkpoints.save(ckptKey).catch(() => null);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(json, null, 2));
      }
    } catch (err: unknown) {
      if (abortController.signal.aborted) {
        console.log(`[Pulse Request #${this.totalRequests}] Client cancelled request.`);
      } else {
        console.error('Pulse backend dispatch error:', err);
      }
      if (!res.headersSent) {
        const msg = err instanceof Error ? err.message : String(err);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: msg }));
      } else {
        res.end();
      }
    } finally {
      req.off('close', onClientClose);
      this.activeStreams--;
    }
  }

  private readRequestBody(req: http.IncomingMessage): Promise<string> {
    return new Promise((resolve, reject) => {
      let data = '';
      req.on('data', (chunk) => (data += chunk));
      req.on('end', () => resolve(data));
      req.on('error', reject);
    });
  }
}
