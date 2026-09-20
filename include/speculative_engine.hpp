#pragma once

#include "types.hpp"
#include "memory_pool.hpp"
#include "hadamard.hpp"
#include "nvfp4_kernel.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <functional>

namespace pulse {

class SpeculativeEngine {
public:
    SpeculativeEngine(
        ModelFormat format = ModelFormat::PQ2_0_TERNARY,
        ModelArchitecture arch = ModelArchitecture::DENSE_27B,
        uint32_t default_k = DEFAULT_SPECULATION_K,
        size_t memory_budget = GB10_TOTAL_MEMORY_BYTES,
        TensorParallelConfig tp_config = {}
    );
    ~SpeculativeEngine();

    bool initialize();

    // Standard synchronous speculative step with dynamic kernel dispatch
    SpeculativeStepResult step(uint32_t k = DEFAULT_SPECULATION_K, ExecutionPhase phase = ExecutionPhase::DECODE);

    // Asynchronous pipelined step
    SpeculativeStepResult step_pipelined(uint32_t k = DEFAULT_SPECULATION_K, ExecutionPhase phase = ExecutionPhase::DECODE);

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
    const char* get_active_kernel_name() const;

private:
    ModelFormat format_;
    ModelArchitecture arch_;
    uint32_t k_;
    size_t memory_budget_;
    bool initialized_{false};
    bool graph_captured_{false};
    bool ping_pong_state_{false};

    std::unique_ptr<MemoryGovernor> governor_;
    TensorParallelConfig tp_config_{};

    cudaStream_t compute_stream_{nullptr};
    cudaStream_t draft_stream_{nullptr};

    cudaEvent_t draft_ready_event_{nullptr};
    cudaEvent_t verify_ready_event_{nullptr};

    cudaGraph_t speculative_graph_{nullptr};
    cudaGraphExec_t graph_exec_{nullptr};

    // Device Buffers for Model Activations and Speculation
    float* d_hidden_states_{nullptr};
    float* d_draft_hidden_states_{nullptr};
    float* d_verify_logits_{nullptr};
    float* d_ffn_out_{nullptr};

    // Packed Weights Buffers
    uint32_t* d_ternary_packed_w_{nullptr}; // 1.76-bit Ternary with Hadamard
    uint8_t* d_nvfp4_packed_w_{nullptr};    // Blackwell NVFP4 E2M1
    float* d_nvfp4_scales_{nullptr};        // FP8 block micro-scales

    // Ping-Pong Buffers
    int32_t* d_draft_tokens_ping_{nullptr};
    int32_t* d_draft_tokens_pong_{nullptr};
    int32_t* d_target_argmax_ping_{nullptr};
    int32_t* d_target_argmax_pong_{nullptr};

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
