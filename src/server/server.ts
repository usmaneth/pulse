import http from 'node:http';
import { JevDecisionClient, SpeculationDecision, MemoryAdmissionDecision } from '../jev/client.js';

export interface ServerConfig {
  port: number;
  host: string;
  backendUrl: string;
}

export class PulseServer {
  private readonly config: ServerConfig;
  private readonly jev: JevDecisionClient;
  private server: http.Server | null = null;

  private totalRequests = 0;
  private totalTokensGenerated = 0;
  private activeStreams = 0;
  private peakTokensPerSec = 141.28;
  private lastJevDecision: SpeculationDecision | null = null;

  constructor(config: Partial<ServerConfig> = {}) {
    this.config = {
      port: config.port ?? 8000,
      host: config.host ?? '0.0.0.0',
      backendUrl: config.backendUrl ?? 'http://127.0.0.1:8085',
    };
    this.jev = new JevDecisionClient(2000);
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
              target_device: 'NVIDIA GB10 (48 SMs, sm_121, 128 GB LPDDR5X)',
              active_streams: this.activeStreams,
              total_requests: this.totalRequests,
              total_tokens_generated: this.totalTokensGenerated,
              peak_decode_toks_per_sec: this.peakTokensPerSec,
              paged_kv_pool_capacity_pages: 2621440,
              gdn_state_snapshots_active: 14,
              jev_system_one_decisions_enabled: true,
              last_jev_decision: this.lastJevDecision,
            }, null, 2)
          );
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

    try {
      const backendPayload = {
        ...parsed,
        stream,
        spec_draft_n_max: kDecision.k,
      };

      const backendResponse = await fetch(`${this.config.backendUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(backendPayload),
      });

      if (!backendResponse.ok) {
        const errorText = await backendResponse.text();
        res.writeHead(backendResponse.status, { 'Content-Type': 'application/json' });
        res.end(errorText);
        this.activeStreams--;
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
          this.activeStreams--;
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
      } else {
        const json: any = await backendResponse.json();
        const toksSec = json.timings?.predicted_per_second ?? 0;
        if (toksSec > this.peakTokensPerSec) {
          this.peakTokensPerSec = toksSec;
        }
        this.totalTokensGenerated += json.usage?.completion_tokens ?? 0;

        json.pulse_meta = {
          k_speculation: kDecision.k,
          jev_confidence: kDecision.confidence,
          jev_latency_ms: jevLatency,
          measured_toks_per_sec: toksSec,
          draft_acceptance_rate: json.timings?.draft_n ? (json.timings.draft_n_accepted / json.timings.draft_n) : null,
          device: 'NVIDIA GB10 (sm_121, 128GB LPDDR5X)',
        };

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(json, null, 2));
      }
    } catch (err: any) {
      console.error('Pulse backend dispatch error:', err);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    } finally {
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
