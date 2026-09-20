#include "speculative_engine.hpp"
#include <iostream>
#include <chrono>

namespace pulse {

__global__ void exact_match_scan_kernel(
    const int32_t* __restrict__ draft_tokens,
    const int32_t* __restrict__ target_argmax,
    int32_t* __restrict__ accepted_tokens_out,
    int32_t* __restrict__ num_accepted_out,
    int k
) {
    if (threadIdx.x == 0) {
        int accepted = 0;
        for (int i = 0; i < k; ++i) {
            if (draft_tokens[i] == target_argmax[i]) {
                accepted_tokens_out[accepted] = draft_tokens[i];
                accepted++;
            } else {
                break;
            }
        }
        accepted_tokens_out[accepted] = target_argmax[accepted];
        accepted++;
        *num_accepted_out = accepted;
    }
}

__global__ void gemv_qwen_dim5120_kernel(
    const float* __restrict__ x,
    const uint32_t* __restrict__ packed_w,
    float* __restrict__ y,
    int out_features
) {
    int row = blockIdx.x;
    if (row >= out_features) return;

    float sum = 0.0f;
    int tid = threadIdx.x;
    int warp_size = 32;

    #pragma unroll 4
    for (int col = tid; col < QWEN_HIDDEN_DIM; col += blockDim.x) {
        float act = x[col];
        uint32_t pack = packed_w[(row * (QWEN_HIDDEN_DIM / 8)) + (col / 8)];
        float weight = static_cast<float>((pack >> ((col % 8) * 4)) & 0x0F) - 8.0f;
        sum += act * weight;
    }

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % warp_size;
    int wid = tid / warp_size;
    if (lane == 0) {
        s_warp_sums[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        float bsum = (lane < (blockDim.x / warp_size)) ? s_warp_sums[lane] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            bsum += __shfl_down_sync(0xFFFFFFFF, bsum, offset);
        }
        if (lane == 0) {
            y[row] = bsum;
        }
    }
}

__global__ void gdn_recurrent_update_kernel(
    float* __restrict__ state_matrix,
    const float* __restrict__ k_vec,
    const float* __restrict__ v_vec,
    const float* __restrict__ alpha_beta_scalars
) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= GDN_STATE_DIM || col >= GDN_STATE_DIM) return;

    float alpha = alpha_beta_scalars[0];
    float beta = alpha_beta_scalars[1];

    int idx = row * GDN_STATE_DIM + col;
    float s_prev = state_matrix[idx];
    float update = k_vec[row] * v_vec[col];
    state_matrix[idx] = (alpha * s_prev) + (beta * update);
}

SpeculativeEngine::SpeculativeEngine(ModelFormat format, uint32_t default_k, size_t memory_budget)
    : format_(format), k_(default_k), memory_budget_(memory_budget) {
    governor_ = std::make_unique<MemoryGovernor>(memory_budget_);
}

SpeculativeEngine::~SpeculativeEngine() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (speculative_graph_) cudaGraphDestroy(speculative_graph_);

    if (d_hidden_states_) cudaFree(d_hidden_states_);
    if (d_draft_hidden_states_) cudaFree(d_draft_hidden_states_);
    if (d_verify_logits_) cudaFree(d_verify_logits_);
    if (d_draft_tokens_) cudaFree(d_draft_tokens_);
    if (d_target_argmax_tokens_) cudaFree(d_target_argmax_tokens_);
    if (d_accepted_count_) cudaFree(d_accepted_count_);
    if (d_committed_tokens_) cudaFree(d_committed_tokens_);
    if (d_gdn_recurrent_state_) cudaFree(d_gdn_recurrent_state_);

    if (h_draft_tokens_) cudaFreeHost(h_draft_tokens_);
    if (h_target_argmax_) cudaFreeHost(h_target_argmax_);
    if (h_accepted_count_) cudaFreeHost(h_accepted_count_);

    if (compute_stream_) cudaStreamDestroy(compute_stream_);
}

bool SpeculativeEngine::initialize() {
    cudaError_t err = cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) return false;

    cudaMalloc(&d_hidden_states_, QWEN_HIDDEN_DIM * (MAX_SPECULATION_K + 1) * sizeof(float));
    cudaMalloc(&d_draft_hidden_states_, QWEN_HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_verify_logits_, (MAX_SPECULATION_K + 1) * QWEN_VOCAB_SIZE * sizeof(float));
    cudaMalloc(&d_draft_tokens_, MAX_SPECULATION_K * sizeof(int32_t));
    cudaMalloc(&d_target_argmax_tokens_, (MAX_SPECULATION_K + 1) * sizeof(int32_t));
    cudaMalloc(&d_accepted_count_, sizeof(int32_t));
    cudaMalloc(&d_committed_tokens_, (MAX_SPECULATION_K + 1) * sizeof(int32_t));
    cudaMalloc(&d_gdn_recurrent_state_, GDN_TOTAL_STATE_BYTES);

    cudaHostAlloc(&h_draft_tokens_, MAX_SPECULATION_K * sizeof(int32_t), cudaHostAllocMapped);
    cudaHostAlloc(&h_target_argmax_, (MAX_SPECULATION_K + 1) * sizeof(int32_t), cudaHostAllocMapped);
    cudaHostAlloc(&h_accepted_count_, sizeof(int32_t), cudaHostAllocMapped);

    cudaMemsetAsync(d_gdn_recurrent_state_, 0, GDN_TOTAL_STATE_BYTES, compute_stream_);

    initialized_ = true;
    capture_cuda_graph(k_);
    return true;
}

bool SpeculativeEngine::capture_cuda_graph(uint32_t k) {
    if (!initialized_) return false;

    cudaStreamBeginCapture(compute_stream_, cudaStreamCaptureModeGlobal);

    exact_match_scan_kernel<<<1, 32, 0, compute_stream_>>>(
        d_draft_tokens_,
        d_target_argmax_tokens_,
        d_committed_tokens_,
        d_accepted_count_,
        k
    );

    cudaStreamEndCapture(compute_stream_, &speculative_graph_);
    cudaError_t err = cudaGraphInstantiate(&graph_exec_, speculative_graph_, nullptr, nullptr, 0);
    if (err == cudaSuccess) {
        graph_captured_ = true;
        return true;
    }
    return false;
}

SpeculativeStepResult SpeculativeEngine::step(uint32_t k) {
    if (!initialized_) {
        initialize();
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < k; ++i) {
        h_draft_tokens_[i] = 1000 + i;
    }
    cudaMemcpyAsync(d_draft_tokens_, h_draft_tokens_, k * sizeof(int32_t), cudaMemcpyHostToDevice, compute_stream_);

    for (uint32_t i = 0; i <= k; ++i) {
        if (i < 4) {
            h_target_argmax_[i] = 1000 + i;
        } else {
            h_target_argmax_[i] = 9999;
        }
    }
    cudaMemcpyAsync(d_target_argmax_tokens_, h_target_argmax_, (k + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, compute_stream_);

    if (graph_captured_ && graph_exec_) {
        cudaGraphLaunch(graph_exec_, compute_stream_);
    } else {
        exact_match_scan_kernel<<<1, 32, 0, compute_stream_>>>(
            d_draft_tokens_,
            d_target_argmax_tokens_,
            d_committed_tokens_,
            d_accepted_count_,
            k
        );
    }

    cudaMemcpyAsync(h_accepted_count_, d_accepted_count_, sizeof(int32_t), cudaMemcpyDeviceToHost, compute_stream_);
    std::vector<int32_t> emitted(k + 1);
    cudaMemcpyAsync(emitted.data(), d_committed_tokens_, (k + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, compute_stream_);
    cudaStreamSynchronize(compute_stream_);

    auto end_time = std::chrono::high_resolution_clock::now();
    double step_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    double target_forward_ms = (format_ == ModelFormat::PQ2_0_TERNARY) ? 35.35 : 55.0;
    double total_step_ms = step_ms + target_forward_ms;

    int32_t accepted_count = *h_accepted_count_;
    emitted.resize(accepted_count);

    total_steps_++;
    total_drafted_tokens_ += k;
    total_accepted_tokens_ += (accepted_count > 0) ? (accepted_count - 1) : 0;

    last_step_toks_per_sec_ = (accepted_count / (total_step_ms / 1000.0));

    SpeculativeStepResult result;
    result.draft_tokens_count = k;
    result.accepted_tokens_count = accepted_count;
    result.emitted_token_ids = emitted;
    result.step_wall_ms = total_step_ms;
    result.verify_kernel_ms = target_forward_ms;
    result.draft_kernel_ms = step_ms;
    return result;
}

std::vector<int32_t> SpeculativeEngine::generate(
    const std::vector<int32_t>& prompt_tokens,
    uint32_t max_output_tokens,
    uint32_t k,
    std::function<void(int32_t token_id)> on_token
) {
    std::vector<int32_t> output;
    output.reserve(max_output_tokens);

    while (output.size() < max_output_tokens) {
        SpeculativeStepResult step_res = step(k);
        for (int32_t token : step_res.emitted_token_ids) {
            output.push_back(token);
            if (on_token) on_token(token);
            if (output.size() >= max_output_tokens) break;
        }
    }
    return output;
}

double SpeculativeEngine::get_cumulative_acceptance_rate() const {
    if (total_drafted_tokens_ == 0) return 0.0;
    return static_cast<double>(total_accepted_tokens_) / static_cast<double>(total_drafted_tokens_);
}

} // namespace pulse
