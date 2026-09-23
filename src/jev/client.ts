export interface SpeculationDecision {
  k: number; confidence: null; source: 'local_heuristic'; reasoning: string;
}
export interface MemoryAdmissionDecision {
  action: 'admit_immediate' | 'queue_wait' | 'evict_prefix_snapshot';
  confidence: null; probability: null; source: 'local_policy';
}
export interface KernelGeometryDecision {
  kernelPath: 'gemv_warp_specialized' | 'gemv_persistent_cta' | 'gemm_tensor_core_fp4';
  confidence: null;
}
export interface ToolRoutingDecision {
  isToolCall: boolean; probability: null; source: 'local_heuristic';
}
/** Compatibility methods use local policies. They never call an external API. */
export class JevDecisionClient {
  constructor(_timeoutMs = 2500) {}
  async decideSpeculationK(state: {
    promptSnippet: string; taskDomain: 'code' | 'math' | 'chat' | 'reasoning' | 'tool_call';
    recentAcceptanceRate?: number; consecutiveRejections?: number;
  }): Promise<SpeculationDecision> {
    const schema = /\bjson\b|schema|interface |dataclass|struct |enum |typedef |\btable\b/i.test(state.promptSnippet || '');
    return { k: schema ? 3 : 4, confidence: null, source: 'local_heuristic',
      reasoning: 'Historical v1 K-curve policy. The backend must support per-request n_max. This is not a Jev prediction.' };
  }
  async decideMemoryAdmission(state: {
    activeSessions: number; usedKVMemoryGB: number; totalAddressableGB: number; incomingTokens: number;
  }): Promise<MemoryAdmissionDecision> {
    const valid = Object.values(state).every(value => Number.isFinite(value) && value >= 0) && state.totalAddressableGB > 0 && state.usedKVMemoryGB <= state.totalAddressableGB;
    return { action: valid && state.totalAddressableGB - state.usedKVMemoryGB > 20 ? 'admit_immediate' : 'queue_wait',
      confidence: null, probability: null, source: 'local_policy' };
  }
  async decideToolRouting(prompt: string): Promise<ToolRoutingDecision> {
    return { isToolCall: /\b(git|file|read|grep|run|bash|search|test|build|edit)\b/i.test(prompt), probability: null, source: 'local_heuristic' };
  }
}
