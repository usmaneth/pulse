#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace pulse {

// Hardware constants for NVIDIA DGX Spark (NVIDIA GB10)
constexpr size_t GB10_TOTAL_MEMORY_BYTES = 128ULL * 1024ULL * 1024ULL * 1024ULL; // 128 GB Unified LPDDR5X
constexpr double GB10_PEAK_BANDWIDTH_GBS = 273.0; // GB/s
constexpr int GB10_SM_COUNT = 48;
constexpr int GB10_CUDA_ARCH = 121; // sm_121 / Blackwell compute capability 12.1

// Model geometry constants (Qwen3.8-27B and Bonsai 2 27B)
constexpr uint32_t QWEN_HIDDEN_DIM = 5120;
constexpr uint32_t QWEN_INTERMEDIATE_DIM = 13824;
constexpr uint32_t QWEN_NUM_HEADS = 64;
constexpr uint32_t QWEN_NUM_KV_HEADS = 8;
constexpr uint32_t QWEN_HEAD_DIM = 128;
constexpr uint32_t QWEN_NUM_LAYERS = 62;
constexpr uint32_t QWEN_VOCAB_SIZE = 248320;

// Paged Attention & KV Cache Geometry
constexpr uint32_t PAGE_SIZE_TOKENS = 16;
constexpr size_t KV_PAGE_BYTES = PAGE_SIZE_TOKENS * QWEN_NUM_KV_HEADS * QWEN_HEAD_DIM * 2; // 32 KB per page

// Gated DeltaNet (GDN) Recurrent State Geometry
constexpr uint32_t GDN_STATE_DIM = 128;
constexpr size_t GDN_LAYER_STATE_BYTES = GDN_STATE_DIM * GDN_STATE_DIM * sizeof(float); // 64 KB per recurrent layer
constexpr size_t GDN_TOTAL_STATE_BYTES = QWEN_NUM_LAYERS * GDN_LAYER_STATE_BYTES; // ~3.9 MB total snapshot

// Speculative Decoding Parameters
constexpr uint32_t MAX_SPECULATION_K = 7;
constexpr uint32_t DEFAULT_SPECULATION_K = 5;

// Memory Allocation Ceilings
constexpr size_t WEIGHTS_RESERVED_BYTES = 16ULL * 1024ULL * 1024ULL * 1024ULL; // 16 GB for weights + draft
constexpr size_t KV_POOL_RESERVED_BYTES = 80ULL * 1024ULL * 1024ULL * 1024ULL; // 80 GB for Paged KV pool
constexpr size_t STATE_CACHE_RESERVED_BYTES = 4ULL * 1024ULL * 1024ULL * 1024ULL; // 4 GB for 1,000+ GDN snapshots

enum class ModelFormat {
    PQ2_0_TERNARY,  // Bonsai 2 27B 1.76-bit with Hadamard transformation (6.7 GB)
    NVFP4,          // Blackwell SM120 native FP4 with micro-scaling (14.5 GB)
    Q4_K_M          // Standard 4-bit GGUF/GGML (15.5 GB)
};

enum class TaskDomain {
    CODE,
    MATH,
    REASONING,
    CHAT,
    TOOL_CALL
};

enum class TensorParallelMode {
    DISABLED,
    DUAL_SPARK_SHARDED,      // 2 nodes over 400G QSFP fabric (spark1 + spark2)
    MULTI_GPU_NVLINK         // Intra-node multi-GPU NVLink / P2P
};

struct TensorParallelConfig {
    bool enabled{false};
    TensorParallelMode mode{TensorParallelMode::DISABLED};
    uint32_t world_size{1};
    uint32_t rank{0};
    std::string peer_host{"10.99.0.2"};
    uint16_t peer_port{50055};

    // Sharded geometry metrics
    uint32_t sharded_hidden_dim{QWEN_HIDDEN_DIM};
    uint32_t sharded_intermediate_dim{QWEN_INTERMEDIATE_DIM};
    uint32_t sharded_num_heads{QWEN_NUM_HEADS};
    uint32_t sharded_kv_heads{QWEN_NUM_KV_HEADS};
    size_t sharded_model_bytes{6700000000ULL}; // 6.7 GB full
    double target_sweep_ms{35.5};

    void configure(uint32_t ws, uint32_t r, TensorParallelMode m = TensorParallelMode::DUAL_SPARK_SHARDED) {
        enabled = (ws > 1);
        world_size = ws;
        rank = r;
        mode = m;
        if (enabled) {
            sharded_hidden_dim = QWEN_HIDDEN_DIM / ws;
            sharded_intermediate_dim = QWEN_INTERMEDIATE_DIM / ws;
            sharded_num_heads = QWEN_NUM_HEADS / ws;
            sharded_kv_heads = std::max(1U, QWEN_NUM_KV_HEADS / ws);
            sharded_model_bytes = (6700000000ULL / ws);
            target_sweep_ms = 35.5 / ws; // 17.75 ms on 2 nodes
        }
    }
};

struct SpeculativeStepResult {
    uint32_t draft_tokens_count;
    uint32_t accepted_tokens_count;
    std::vector<int32_t> emitted_token_ids;
    double step_wall_ms;
    double verify_kernel_ms;
    double draft_kernel_ms;
};

} // namespace pulse
