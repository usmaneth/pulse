#include "types.hpp"
#include "memory_pool.hpp"
#include "speculative_engine.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace pulse;

int main(int argc, char** argv) {
    std::cout << "=======================================================\n";
    std::cout << " PULSE kernel microbenchmarks - NVIDIA Blackwell (GB10, sm_121)\n";
    std::cout << " Target Silicon: NVIDIA GB10 (sm_121, 128GB LPDDR5X)\n";
    std::cout << " NOT an inference engine: no model weights are loaded here.\n"
              << " Real decode numbers come from bench/sweep.py. See bench/RESULTS.md.\n";
    std::cout << "=======================================================\n\n";

    std::cout << "[1/4] Initializing 128GB Unified Memory Governor...\n";
    MemoryGovernor governor(GB10_TOTAL_MEMORY_BYTES);
    std::cout << "  - Total System Memory: 128 GB\n";
    std::cout << "  - Paged KV Pool Capacity: " << (governor.kv_pool().get_free_bytes() / (1024ULL * 1024ULL * 1024ULL)) << " GB ("
              << governor.kv_pool().get_total_pages() << " pages of 16 tokens)\n";

    std::cout << "\n[2/4] Testing GDN Recurrent Prefix State Snapshotting...\n";
    cudaStream_t test_stream;
    cudaStreamCreate(&test_stream);
    void* d_dummy_state;
    cudaMalloc(&d_dummy_state, GDN_TOTAL_STATE_BYTES);
    cudaMemsetAsync(d_dummy_state, 0x42, GDN_TOTAL_STATE_BYTES, test_stream);

    uint64_t prompt_prefix_hash = 0xABCD1234EF567890ULL;
    auto t0 = std::chrono::high_resolution_clock::now();
    governor.gdn_cache().save_snapshot(prompt_prefix_hash, d_dummy_state, test_stream);
    cudaStreamSynchronize(test_stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double save_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    bool hit = governor.gdn_cache().restore_snapshot(prompt_prefix_hash, d_dummy_state, test_stream);
    cudaStreamSynchronize(test_stream);
    t1 = std::chrono::high_resolution_clock::now();
    double restore_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  - Saved 62-layer GDN state (3.9 MB) in: " << save_ms << " ms\n";
    std::cout << "  - Restored GDN state on prefix hit (" << (hit ? "OK" : "MISS") << ") in: " << restore_ms << " ms\n";
    cudaFree(d_dummy_state);
    cudaStreamDestroy(test_stream);

    // 3. Dense 27B Benchmark (Ternary PQ2_0 with Hadamard)
    std::cout << "\n[3/4] Kernel dispatch microbenchmark A: PQ2_0 ternary GEMV + scan (toy graph, no weights)...\n";
    SpeculativeEngine engine_dense(ModelFormat::PQ2_0_TERNARY, ModelArchitecture::DENSE_27B, 5);
    engine_dense.initialize();
    std::cout << "  - Active Kernel: " << engine_dense.get_active_kernel_name() << "\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Step | Draft K | Accepted | Dispatch Time (real cudaEvent)\n";
    std::cout << "-----+---------+----------+---------------------------\n";
    for (int step = 1; step <= 5; ++step) {
        SpeculativeStepResult res = engine_dense.step_pipelined(5);
        std::cout << std::setw(4) << step << " | "
                  << std::setw(7) << res.draft_tokens_count << " | "
                  << std::setw(8) << res.accepted_tokens_count << " | "
                  << std::setw(7) << res.step_wall_ms << " ms\n";
    }

    // 4. Sparse MoE 35B Benchmark (Blackwell NVFP4, 1.68 GB active sweep)
    std::cout << "\n[4/4] Kernel dispatch microbenchmark B: NVFP4 GEMV + scan (toy graph, no weights)...\n";
    SpeculativeEngine engine_moe(ModelFormat::NVFP4, ModelArchitecture::SPARSE_MOE_35B, 5);
    engine_moe.initialize();
    std::cout << "  - Active Kernel: " << engine_moe.get_active_kernel_name() << "\n";

    std::cout << "Step | Draft K | Accepted | Dispatch Time (real cudaEvent)\n";
    std::cout << "-----+---------+----------+---------------------------\n";
    for (int step = 1; step <= 5; ++step) {
        SpeculativeStepResult res = engine_moe.step_pipelined(5);
        std::cout << std::setw(4) << step << " | "
                  << std::setw(7) << res.draft_tokens_count << " | "
                  << std::setw(8) << res.accepted_tokens_count << " | "
                  << std::setw(7) << res.step_wall_ms << " ms\n";
    }

    std::cout << "=======================================================\n";
    std::cout << " Kernel dispatch microbenchmark complete on NVIDIA GB10.\n";
    std::cout << " NOTE: the step times above are real cudaEvent measurements of the\n";
    std::cout << " dispatch graph only (one GEMV + scan kernel). They are NOT end-to-end\n";
    std::cout << " model inference. For real decode throughput run: bench/sweep.py\n";
    std::cout << "=======================================================\n";

    return 0;
}
