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
   * Dynamically selects the optimal speculative decoding draft window K (3 to 7).
   * Math and structured code benefit from K=5 to 7, while open-ended chat degrades past K=4.
   */
  async decideSpeculationK(state: {
    promptSnippet: string;
    taskDomain: 'code' | 'math' | 'chat' | 'reasoning' | 'tool_call';
    recentAcceptanceRate?: number;
    consecutiveRejections?: number;
  }): Promise<SpeculationDecision> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const evaluation = await experimental_evaluate({
        model: this.model,
        abortSignal: controller.signal,
        state: {
          task_domain: state.taskDomain,
          snippet: state.promptSnippet.slice(0, 500),
          recent_acceptance: state.recentAcceptanceRate ?? 0.5,
          consecutive_rejections: state.consecutiveRejections ?? 0,
        },
        questions: {
          speculation_window: {
            type: 'choice',
            instructions: 'Select the optimal speculative drafting block size K to maximize net throughput on NVIDIA GB10.',
            criteria: {
              k_4: 'K=4: Safe baseline for chat or medium acceptance (~40-50%)',
              k_5: 'K=5: Optimal balanced window for code and reasoning (~50-65% acceptance)',
              k_7: 'K=7: Aggressive window for high-repetition tasks, math, or boilerplate code (>70% acceptance)',
              k_3: 'K=3: Conservative small window for difficult, noisy, or low-acceptance streams (<35% acceptance)',
            },
          },
        },
      });

      clearTimeout(timeoutId);
      const answer = evaluation.answers.speculation_window;
      if (answer && answer.type === 'choice') {
        const choice = answer.choice;
        const conf = extractConfidence(evaluation.providerMetadata, 'speculation_window');
        let k = 5;
        if (choice === 'k_4') k = 4;
        else if (choice === 'k_7') k = 7;
        else if (choice === 'k_3') k = 3;
        else k = 5;

        return {
          k,
          confidence: conf,
          reasoning: `Jev selected ${choice} (conf: ${conf.toFixed(2)}) for domain ${state.taskDomain}`,
        };
      }
    } catch {
      clearTimeout(timeoutId);
    }

    // Deterministic fallback if Jev is unreachable or times out
    const fallbackK = state.taskDomain === 'math' || state.taskDomain === 'code' ? 5 : 4;
    return {
      k: fallbackK,
      confidence: 0.8,
      reasoning: `Local heuristic fallback K=${fallbackK} for domain ${state.taskDomain}`,
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
