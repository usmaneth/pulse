#pragma once

#include "types.hpp"
#include "memory_pool.hpp"
#include "hadamard.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <functional>

namespace pulse {

class SpeculativeEngine {
public:
    SpeculativeEngine(
        ModelFormat format = ModelFormat::PQ2_0_TERNARY,
        uint32_t default_k = DEFAULT_SPECULATION_K,
        size_t memory_budget = GB10_TOTAL_MEMORY_BYTES
    );
    ~SpeculativeEngine();

    bool initialize();
    SpeculativeStepResult step(uint32_t k = DEFAULT_SPECULATION_K);

    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt_tokens,
        uint32_t max_output_tokens,
        uint32_t k = DEFAULT_SPECULATION_K,
        std::function<void(int32_t token_id)> on_token = nullptr
    );

    bool capture_cuda_graph(uint32_t k);

    double get_last_step_toks_per_sec() const { return last_step_toks_per_sec_; }
    double get_cumulative_acceptance_rate() const;
    size_t get_total_accepted_tokens() const { return total_accepted_tokens_; }
    size_t get_total_drafted_tokens() const { return total_drafted_tokens_; }

private:
    ModelFormat format_;
    uint32_t k_;
    size_t memory_budget_;
    bool initialized_{false};
    bool graph_captured_{false};

    std::unique_ptr<MemoryGovernor> governor_;

    cudaStream_t compute_stream_{nullptr};
    cudaGraph_t speculative_graph_{nullptr};
    cudaGraphExec_t graph_exec_{nullptr};

    float* d_hidden_states_{nullptr};
    float* d_draft_hidden_states_{nullptr};
    float* d_verify_logits_{nullptr};
    int32_t* d_draft_tokens_{nullptr};
    int32_t* d_target_argmax_tokens_{nullptr};
    int32_t* d_accepted_count_{nullptr};
    int32_t* d_committed_tokens_{nullptr};
    void* d_gdn_recurrent_state_{nullptr};

    int32_t* h_draft_tokens_{nullptr};
    int32_t* h_target_argmax_{nullptr};
    int32_t* h_accepted_count_{nullptr};

    double last_step_toks_per_sec_{0.0};
    size_t total_drafted_tokens_{0};
    size_t total_accepted_tokens_{0};
    size_t total_steps_{0};
};

} // namespace pulse
