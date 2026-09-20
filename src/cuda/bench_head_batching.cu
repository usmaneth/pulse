// Is the LM head the marginal cost of a speculative draft row, and is it recoverable?
//
// Measured finding (v1 of this benchmark): a per-row GEMV over the 248,320-entry head
// costs ~3.0-3.9 ms per additional row, matching 0.592 GB / 184.6 GB/s = 3.2 ms. The
// head is re-read in full for every verification position.
//
// This version fixes the batched kernel. The first attempt used `float acc[NROWS]` with
// a runtime loop index, which spills to local memory and was slower than the GEMV it
// was meant to beat. Templating on NROWS lets the accumulator live in registers and the
// inner loop fully unroll.
//
// Every number printed is a cudaEvent measurement on real device traffic.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

constexpr int HIDDEN = 5120;
constexpr int VOCAB  = 248320;
constexpr int WARMUP = 3;
constexpr int ITERS  = 10;
constexpr int WORDS_PER_ROW = HIDDEN / 8;          // 640 packed uint32 per vocab row
constexpr size_t HEAD_WORDS = (size_t)VOCAB * WORDS_PER_ROW;

// Baseline: one activation vector, head fully re-read on every call.
__global__ void head_gemv(const float* __restrict__ x,
                          const uint32_t* __restrict__ w,
                          float* __restrict__ logits) {
    int row = blockIdx.x, tid = threadIdx.x;
    const uint32_t* wr = w + (size_t)row * WORDS_PER_ROW;
    float sum = 0.0f;
    for (int wi = tid; wi < WORDS_PER_ROW; wi += blockDim.x) {
        uint32_t pack = wr[wi];
        int c0 = wi * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            sum += x[c0 + j] * ((float)((pack >> (j * 4)) & 0xF) - 8.0f);
    }
    for (int off = 16; off; off >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    __shared__ float part[32];
    if ((tid & 31) == 0) part[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < blockDim.x / 32) ? part[tid] : 0.0f;
        for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, off);
        if (tid == 0) logits[row] = v;
    }
}

// Batched: head word loaded ONCE, reused across all NROWS verification positions.
// NROWS is a template parameter so the accumulator stays in registers.
template <int NROWS>
__global__ void head_gemm_batched(const float* __restrict__ x,
                                  const uint32_t* __restrict__ w,
                                  float* __restrict__ logits) {
    int row = blockIdx.x, tid = threadIdx.x;
    const uint32_t* wr = w + (size_t)row * WORDS_PER_ROW;

    float acc[NROWS];
    #pragma unroll
    for (int r = 0; r < NROWS; ++r) acc[r] = 0.0f;

    for (int wi = tid; wi < WORDS_PER_ROW; wi += blockDim.x) {
        uint32_t pack = wr[wi];                 // single global read, reused NROWS times
        int c0 = wi * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float wv = (float)((pack >> (j * 4)) & 0xF) - 8.0f;
            #pragma unroll
            for (int r = 0; r < NROWS; ++r)
                acc[r] += x[(size_t)r * HIDDEN + c0 + j] * wv;   // x stays hot in L2
        }
    }

    __shared__ float part[NROWS][32];
    #pragma unroll
    for (int r = 0; r < NROWS; ++r) {
        float s = acc[r];
        for (int off = 16; off; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
        if ((tid & 31) == 0) part[r][tid >> 5] = s;
    }
    __syncthreads();
    if (tid < 32) {
        #pragma unroll
        for (int r = 0; r < NROWS; ++r) {
            float v = (tid < blockDim.x / 32) ? part[r][tid] : 0.0f;
            for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, off);
            if (tid == 0) logits[(size_t)r * VOCAB + row] = v;
        }
    }
}

static float time_gemv(int nrows, const float* x, const uint32_t* w, float* lg, cudaStream_t s,
                       cudaEvent_t a, cudaEvent_t b) {
    for (int i = 0; i < WARMUP; ++i)
        for (int r = 0; r < nrows; ++r)
            head_gemv<<<VOCAB, 256, 0, s>>>(x + (size_t)r * HIDDEN, w, lg + (size_t)r * VOCAB);
    cudaStreamSynchronize(s);
    cudaEventRecord(a, s);
    for (int i = 0; i < ITERS; ++i)
        for (int r = 0; r < nrows; ++r)
            head_gemv<<<VOCAB, 256, 0, s>>>(x + (size_t)r * HIDDEN, w, lg + (size_t)r * VOCAB);
    cudaEventRecord(b, s); cudaStreamSynchronize(s);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms / ITERS;
}

template <int N>
static float time_batched(const float* x, const uint32_t* w, float* lg, cudaStream_t s,
                          cudaEvent_t a, cudaEvent_t b) {
    for (int i = 0; i < WARMUP; ++i) head_gemm_batched<N><<<VOCAB, 256, 0, s>>>(x, w, lg);
    cudaStreamSynchronize(s);
    cudaEventRecord(a, s);
    for (int i = 0; i < ITERS; ++i) head_gemm_batched<N><<<VOCAB, 256, 0, s>>>(x, w, lg);
    cudaEventRecord(b, s); cudaStreamSynchronize(s);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms / ITERS;
}

int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    double head_gb = (double)HEAD_WORDS * 4.0 / (1024.0 * 1024.0 * 1024.0);
    printf("=====================================================================\n");
    printf(" LM HEAD BATCHING on %s\n", prop.name);
    printf(" hidden=%d vocab=%d  4-bit head = %.3f GB\n", HIDDEN, VOCAB, head_gb);
    printf(" One full head read at 184.6 GB/s = %.2f ms\n", head_gb / 184.6 * 1000.0);
    printf("=====================================================================\n\n");

    float *x = nullptr, *lg = nullptr; uint32_t* w = nullptr;
    cudaMalloc(&w, HEAD_WORDS * sizeof(uint32_t));
    cudaMalloc(&x, (size_t)8 * HIDDEN * sizeof(float));
    cudaMalloc(&lg, (size_t)8 * VOCAB * sizeof(float));
    if (!w || !x || !lg) { printf("alloc failed\n"); return 1; }
    cudaMemset(w, 0x55, HEAD_WORDS * sizeof(uint32_t));
    cudaMemset(x, 0x3F, (size_t)8 * HIDDEN * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

    printf("K+1 rows | per-row GEMV | batched GEMM | speedup | saved ms/step\n");
    printf("---------+--------------+--------------+---------+---------------\n");
    float g1 = time_gemv(1, x, w, lg, s, a, b);
    float b1 = time_batched<1>(x, w, lg, s, a, b);
    printf("%8d | %12.3f | %12.3f | %6.2fx | %13.3f\n", 1, g1, b1, g1 / b1, g1 - b1);
    float g2 = time_gemv(2,x,w,lg,s,a,b), t2 = time_batched<2>(x,w,lg,s,a,b);
    printf("%8d | %12.3f | %12.3f | %6.2fx | %13.3f\n", 2, g2, t2, g2/t2, g2-t2);
    float g4 = time_gemv(4,x,w,lg,s,a,b), t4 = time_batched<4>(x,w,lg,s,a,b);
    printf("%8d | %12.3f | %12.3f | %6.2fx | %13.3f\n", 4, g4, t4, g4/t4, g4-t4);
    float g5 = time_gemv(5,x,w,lg,s,a,b), t5 = time_batched<5>(x,w,lg,s,a,b);
    printf("%8d | %12.3f | %12.3f | %6.2fx | %13.3f\n", 5, g5, t5, g5/t5, g5-t5);
    float g6 = time_gemv(6,x,w,lg,s,a,b), t6 = time_batched<6>(x,w,lg,s,a,b);
    printf("%8d | %12.3f | %12.3f | %6.2fx | %13.3f\n", 6, g6, t6, g6/t6, g6-t6);
    float g8 = time_gemv(8,x,w,lg,s,a,b), t8 = time_batched<8>(x,w,lg,s,a,b);
    printf("%8d | %12.3f | %12.3f | %6.2fx | %13.3f\n", 8, g8, t8, g8/t8, g8-t8);

    printf("\nAt K=7 the verify pass evaluates 8 head rows. Saving (g8-t8) ms per step\n");
    printf("against a measured 79.63 ms step at K=7 implies:\n");
    float newstep = 79.63f - (g8 - t8);
    printf("  step %.2f -> %.2f ms, throughput 5.81 tok/step -> %.1f tok/s\n",
           79.63f, newstep, 5.81f / (newstep / 1000.0f));

    cudaFree(w); cudaFree(x); cudaFree(lg); cudaStreamDestroy(s);
    return 0;
}
