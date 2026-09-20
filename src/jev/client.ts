import { experimental_evaluate } from 'ai';

export interface SpeculationDecision {
  k: number;
  confidence: number;
  reasoning: string;
}

export interface MemoryAdmissionDecision {
  action: 'admit_immediate' | 'queue_wait' | 'evict_prefix_snapshot';
  confidence: number;
  probability: number;
}

export interface KernelGeometryDecision {
  kernelPath: 'gemv_warp_specialized' | 'gemv_persistent_cta' | 'gemm_tensor_core_fp4';
  confidence: number;
}

export interface ToolRoutingDecision {
  isToolCall: boolean;
  probability: number;
}

function extractConfidence(metadata: unknown, key: string, fallback: number = 0.9): number {
  if (
    metadata &&
    typeof metadata === 'object' &&
    'typesafe' in metadata &&
    typeof metadata.typesafe === 'object' &&
    metadata.typesafe !== null &&
    'confidence' in metadata.typesafe &&
    typeof metadata.typesafe.confidence === 'object' &&
    metadata.typesafe.confidence !== null &&
    key in metadata.typesafe.confidence
  ) {
    const val = (metadata.typesafe.confidence as Record<string, unknown>)[key];
    if (typeof val === 'number') {
      return val;
    }
  }
  return fallback;
}

export class JevDecisionClient {
  private readonly model = 'typesafe-ai/jev';
  private readonly timeoutMs: number;

  constructor(timeoutMs: number = 2500) {
    this.timeoutMs = timeoutMs;
  }

  /**
   * Selects a speculative draft window K.
   *
   * IMPORTANT, so nobody is misled by the name: this method does NOT call Jev.
   * It is a local regex heuristic and runs on the request hot path, where a
   * network round trip would cost more than the decision can win. Only
   * decideMemoryAdmission and the other off-path methods call the gateway.
   *
   * The return value IS live now. llama.cpp used to compile per-request
   * speculative parameters out of its server behind `#if 0`, so this number went
   * nowhere. patches/llama-per-request-spec-n-max.patch exposes
   * `speculative.n_max` (aliased to `spec_draft_n_max`) and applies it in
   * server_slot::get_n_draft_max(). Verified end to end: n_max=1 gives 37.10
   * tok/s and n_max=3 gives 53.09, with the field unset matching the server
   * default exactly.
   *
   * Two limits still apply:
   *   1. K is capped by the drafter's block size (v1=4, v2=7).
   *   2. A request can only LOWER K below the server's --spec-draft-n-max; it
   *      cannot raise it. Run the server at the drafter's block size so
   *      per-request K has room to move down.
   */
  async decideSpeculationK(state: {
    promptSnippet: string;
    taskDomain: 'code' | 'math' | 'chat' | 'reasoning' | 'tool_call';
    recentAcceptanceRate?: number;
    consecutiveRejections?: number;
  }): Promise<SpeculationDecision> {
    const snippet = state.promptSnippet || '';

    // MEASURED POLICY (bench/kcurve.py, v1 drafter block_size=4, single stream,
    // 3 interleaved repeats, spreads 0.1-1.7% against a 3.4% noise floor).
    //
    //            K=1     K=2     K=3     K=4     K=6
    //   code    36.57   46.95   52.19   56.07   56.43
    //   schema  36.74   45.51   51.14   49.22   49.22
    //   chat    33.93   39.05   40.92   41.38   41.62
    //
    // Two things the curve says, both of which contradict the heuristic that
    // used to live here:
    //   1. Everything saturates at K=4, the drafter's block size. K=5 and above
    //      are a no-op, so the old "structured -> K=7" rule did nothing.
    //   2. Schema traffic peaks at K=3 and gets 3.9% WORSE at K=4, which is far
    //      outside its 0.1-0.6% spread. The old rule sent schema to the
    //      deepest draft; the measurement wants it at the shallowest.
    //
    // Honest size of this lever: against a single global K=4 it is worth ~3.9%,
    // and only on schema-like traffic. It is not a large win.
    if (/\bjson\b|schema|interface |dataclass|struct |enum |typedef |\btable\b/i.test(snippet)) {
      return {
        k: 3,
        confidence: 0.9,
        reasoning: 'measured (bench/kcurve.py): schema/JSON peaks at K=3; K=4 is 3.9% worse',
      };
    }
    // code, math, reasoning and chat all sit at the K=4 knee.
    return {
      k: 4,
      confidence: 0.9,
      reasoning: `measured (bench/kcurve.py): ${state.taskDomain} saturates at the K=4 block-size knee`,
    };
  }

  /**
   * Evaluates memory pressure on the 128 GB unified memory pool and chooses admission policy.
   */
  async decideMemoryAdmission(state: {
    activeSessions: number;
    usedKVMemoryGB: number;
    totalAddressableGB: number;
    incomingTokens: number;
  }): Promise<MemoryAdmissionDecision> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const evaluation = await experimental_evaluate({
        model: this.model,
        abortSignal: controller.signal,
        state: {
          active_sessions: state.activeSessions,
          used_kv_gb: state.usedKVMemoryGB,
          total_gb: state.totalAddressableGB,
          incoming_tokens: state.incomingTokens,
          memory_headroom_gb: state.totalAddressableGB - state.usedKVMemoryGB,
        },
        questions: {
          admission_policy: {
            type: 'choice',
            instructions: 'Evaluate memory headroom on 128GB LPDDR5X and select request admission action.',
            criteria: {
              admit_immediate: 'Sufficient headroom (>20GB free); admit request immediately.',
              queue_wait: 'High memory pressure (>95GB used); queue request until a slot completes.',
              evict_prefix_snapshot: 'Moderate pressure; evict oldest LRU prefix cache page and admit.',
            },
          },
        },
      });

      clearTimeout(timeoutId);
      const ans = evaluation.answers.admission_policy;
      if (ans && ans.type === 'choice') {
        const choice = ans.choice as 'admit_immediate' | 'queue_wait' | 'evict_prefix_snapshot';
        const prob = ans.probabilities?.[choice] ?? 0.95;
        const conf = extractConfidence(evaluation.providerMetadata, 'admission_policy', prob);
        return {
          action: choice,
          confidence: conf,
          probability: prob,
        };
      }
    } catch {
      clearTimeout(timeoutId);
    }

    const freeGB = state.totalAddressableGB - state.usedKVMemoryGB;
    if (freeGB > 15) {
      return { action: 'admit_immediate', confidence: 1.0, probability: 1.0 };
    }
    return { action: 'queue_wait', confidence: 0.9, probability: 0.9 };
  }

  /**
   * Fast classification of tool-call likelihood for structured agent turns.
   */
  async decideToolRouting(prompt: string): Promise<ToolRoutingDecision> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const evaluation = await experimental_evaluate({
        model: this.model,
        abortSignal: controller.signal,
        state: { prompt: prompt.slice(0, 400) },
        questions: {
          needs_tool: {
            type: 'boolean',
            instructions: 'Does this user request require calling a software tool, reading a file, searching code, or executing a command?',
          },
        },
      });

      clearTimeout(timeoutId);
      const ans = evaluation.answers.needs_tool;
      if (ans && ans.type === 'boolean') {
        return {
          isToolCall: ans.probability > 0.5,
          probability: ans.probability,
        };
      }
    } catch {
      clearTimeout(timeoutId);
    }

    const lower = prompt.toLowerCase();
    const hasToolKeywords = /git|file|read|grep|run|bash|search|test|build|edit/.test(lower);
    return { isToolCall: hasToolKeywords, probability: hasToolKeywords ? 0.8 : 0.2 };
  }
}
