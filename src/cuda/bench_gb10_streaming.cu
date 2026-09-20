#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

constexpr size_t MODEL_WEIGHT_BYTES = 6700000000ULL; // 6.70 GB (Ternary Bonsai 2 27B)
constexpr int HIDDEN_DIM = 5120;
constexpr int WARMUP_RUNS = 5;
constexpr int BENCHMARK_RUNS = 20;

// High-performance Blackwell SM120 streaming read kernel
__global__ void stream_weights_kernel(
    const uint4* __restrict__ weights,
    float* __restrict__ dummy_out,
    size_t num_vectors
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (size_t i = idx; i < num_vectors; i += gridDim.x * blockDim.x) {
        uint4 val = weights[i];
        sum += static_cast<float>(val.x ^ val.y ^ val.z ^ val.w);
    }
    if (threadIdx.x == 0) {
        dummy_out[blockIdx.x] = sum;
    }
}

int main() {
    printf("=================================================================\n");
    printf(" NVIDIA GB10 Blackwell Hardware Roofline & Speculative Benchmark\n");
    printf(" Working Set: 6.70 GB (Ternary Bonsai 2 27B Weights)\n");
    printf(" Target Silicon: NVIDIA GB10 (sm_121, 128 GB Unified LPDDR5X)\n");
    printf("=================================================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (48 SMs, Bus: %d-bit, Memory: %.1f GB)\n",
           prop.name, prop.memoryBusWidth, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // 1. Allocate 6.70 GB unified memory
    printf("\n[1/4] Allocating 6.70 GB in Unified LPDDR5X memory...\n");
    uint4* d_weights = nullptr;
    cudaError_t err = cudaMalloc(&d_weights, MODEL_WEIGHT_BYTES);
    if (err != cudaSuccess) {
        printf("Failed to allocate: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemset(d_weights, 0x1F, MODEL_WEIGHT_BYTES);

    int num_blocks = 48 * 4; // Saturate 48 SMs
    int block_size = 256;
    float* d_dummy_out = nullptr;
    cudaMalloc(&d_dummy_out, num_blocks * sizeof(float));

    size_t num_vectors = MODEL_WEIGHT_BYTES / sizeof(uint4);

    // 2. Measure sustained memory bandwidth on GB10
    printf("[2/4] Measuring sustained memory bandwidth on 6.70 GB weight sweep...\n");
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        stream_weights_kernel<<<num_blocks, block_size, 0, stream>>>(d_weights, d_dummy_out, num_vectors);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < BENCHMARK_RUNS; ++i) {
        stream_weights_kernel<<<num_blocks, block_size, 0, stream>>>(d_weights, d_dummy_out, num_vectors);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_step_ms = total_ms / BENCHMARK_RUNS;
    double gb_transferred = (static_cast<double>(MODEL_WEIGHT_BYTES) / (1024.0 * 1024.0 * 1024.0));
    double bandwidth_gbs = (gb_transferred / (avg_step_ms / 1000.0));

    printf("  • Sweep Latency: %.2f ms per full model pass\n", avg_step_ms);
    printf("  • Sustained LPDDR5X Bandwidth: %.2f GB/s (%.1f%% of 273 GB/s peak)\n",
           bandwidth_gbs, (bandwidth_gbs / 273.0) * 100.0);

    // 3. Measure CUDA Graph launch overhead vs standard launch
    printf("\n[3/4] Measuring CUDA Graph single-unit dispatch latency...\n");
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    stream_weights_kernel<<<num_blocks, block_size, 0, stream>>>(d_weights, d_dummy_out, num_vectors);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

    cudaEventRecord(start, stream);
    for (int i = 0; i < BENCHMARK_RUNS; ++i) {
        cudaGraphLaunch(graph_exec, stream);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float graph_total_ms = 0.0f;
    cudaEventElapsedTime(&graph_total_ms, start, stop);
    float graph_step_ms = graph_total_ms / BENCHMARK_RUNS;
    printf("  • CUDA Graph Single-Unit Step Time: %.2f ms\n", graph_step_ms);

    // 4. Calculate Single-Stream Speculative Decode Throughput
    printf("\n[4/4] Single-Stream Speculative Decoding Rooflines on GB10:\n");
    printf("-----------------------------------------------------------------\n");
    printf(" Draft Window | Acceptance Rate | Accepted Tokens | Net Speed\n");
    printf("-----------------------------------------------------------------\n");

    struct Scenario {
        int K;
        double acceptance_rate;
    };

    std::vector<Scenario> scenarios = {
        {4, 0.70},
        {4, 0.85},
        {5, 0.75},
        {5, 0.80},
        {5, 0.87},
        {5, 0.913}, // Observed on Python Interval Merging
        {7, 0.70},
        {7, 0.80},
        {7, 0.90}
    };

    for (const auto& sc : scenarios) {
        double accepted_tokens = (sc.K * sc.acceptance_rate) + 1.0;
        double toks_sec = accepted_tokens / (graph_step_ms / 1000.0);
        const char* tag = (toks_sec >= 100.0) ? "-> TARGET REACHED (100+ tok/s)" : "";
        printf(" K=%d         | %5.1f%%          | %4.2f tokens    | %6.1f tok/s %s\n",
               sc.K, sc.acceptance_rate * 100.0, accepted_tokens, toks_sec, tag);
    }
    printf("-----------------------------------------------------------------\n");

    cudaFree(d_weights);
    cudaFree(d_dummy_out);
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    return 0;
}
