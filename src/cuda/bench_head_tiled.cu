// Batched LM head, take 3: shared-memory column tiling.
//
// Established by the previous two benchmarks on GB10:
//   * a single-row head GEMV runs at 3.233 ms against a 3.21 ms theoretical read of the
//     0.592 GB 4-bit head, i.e. it is already memory-bandwidth optimal;
//   * therefore verifying K+1=8 positions with per-row GEMV costs 8 full head reads
//     (25.7 ms measured) where one read (3.2 ms) would suffice;
//   * a naive batched kernel that unrolls 8 columns x 8 rows spills registers and is
//     slower than the GEMV it replaces (67.6 ms).
//
// Strategy here: stage the NROWS activation vectors for a tile of columns into shared
// memory, then stream the head weights through exactly once, reusing each loaded weight
// across all NROWS rows out of shared memory rather than re-loading from global.
//
// Every printed number is a cudaEvent measurement.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

constexpr int HIDDEN = 5120;
constexpr int VOCAB  = 248320;
constexpr int WORDS_PER_ROW = HIDDEN / 8;
constexpr size_t HEAD_WORDS = (size_t)VOCAB * WORDS_PER_ROW;
constexpr int WARMUP = 3, ITERS = 10;
constexpr int THREADS = 256;

__global__ void head_gemv(const float* __restrict__ x, const uint32_t* __restrict__ w,
                          float* __restrict__ logits) {
    int row = blockIdx.x, tid = threadIdx.x;
    const uint32_t* wr = w + (size_t)row * WORDS_PER_ROW;
    float sum = 0.0f;
    for (int wi = tid; wi < WORDS_PER_ROW; wi += THREADS) {
        uint32_t p = wr[wi]; int c0 = wi * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) sum += x[c0 + j] * ((float)((p >> (j*4)) & 0xF) - 8.0f);
    }
    for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xFFFFFFFF, sum, o);
    __shared__ float part[32];
    if ((tid & 31) == 0) part[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < THREADS/32) ? part[tid] : 0.0f;
        for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, o);
        if (tid == 0) logits[row] = v;
    }
}

// Shared-memory tiled batched head.
// sx holds NROWS x HIDDEN activations for this block; HIDDEN=5120 floats = 20 KB per row.
// For NROWS=8 that is 160 KB, too large, so we tile the column dimension.
template <int NROWS, int TILE>
__global__ void head_tiled(const float* __restrict__ x, const uint32_t* __restrict__ w,
                           float* __restrict__ logits) {
    __shared__ float sx[NROWS][TILE];
    int row = blockIdx.x, tid = threadIdx.x;
    const uint32_t* wr = w + (size_t)row * WORDS_PER_ROW;

    float acc[NROWS];
    #pragma unroll
    for (int r = 0; r < NROWS; ++r) acc[r] = 0.0f;

    for (int c0 = 0; c0 < HIDDEN; c0 += TILE) {
        // stage activations for this column tile once per block
        for (int i = tid; i < NROWS * TILE; i += THREADS) {
            int r = i / TILE, c = i % TILE;
            sx[r][c] = x[(size_t)r * HIDDEN + c0 + c];
        }
        __syncthreads();

        int w0 = c0 / 8, w1 = (c0 + TILE) / 8;
        for (int wi = w0 + tid; wi < w1; wi += THREADS) {
            uint32_t p = wr[wi];                       // one global read, reused NROWS times
            int lc = wi * 8 - c0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float wv = (float)((p >> (j*4)) & 0xF) - 8.0f;
                #pragma unroll
                for (int r = 0; r < NROWS; ++r) acc[r] += sx[r][lc + j] * wv;
            }
        }
        __syncthreads();
    }

    __shared__ float part[NROWS][32];
    #pragma unroll
    for (int r = 0; r < NROWS; ++r) {
        float s = acc[r];
        for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, o);
        if ((tid & 31) == 0) part[r][tid >> 5] = s;
    }
    __syncthreads();
    if (tid < 32) {
        #pragma unroll
        for (int r = 0; r < NROWS; ++r) {
            float v = (tid < THREADS/32) ? part[r][tid] : 0.0f;
            for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, o);
            if (tid == 0) logits[(size_t)r * VOCAB + row] = v;
        }
    }
}

static float t_gemv(int n, const float* x, const uint32_t* w, float* l,
                    cudaStream_t s, cudaEvent_t a, cudaEvent_t b) {
    for (int i=0;i<WARMUP;i++) for (int r=0;r<n;r++)
        head_gemv<<<VOCAB,THREADS,0,s>>>(x+(size_t)r*HIDDEN, w, l+(size_t)r*VOCAB);
    cudaStreamSynchronize(s); cudaEventRecord(a,s);
    for (int i=0;i<ITERS;i++) for (int r=0;r<n;r++)
        head_gemv<<<VOCAB,THREADS,0,s>>>(x+(size_t)r*HIDDEN, w, l+(size_t)r*VOCAB);
    cudaEventRecord(b,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,a,b); return ms/ITERS;
}

template <int N, int TILE>
static float t_tiled(const float* x, const uint32_t* w, float* l,
                     cudaStream_t s, cudaEvent_t a, cudaEvent_t b) {
    for (int i=0;i<WARMUP;i++) head_tiled<N,TILE><<<VOCAB,THREADS,0,s>>>(x,w,l);
    cudaStreamSynchronize(s);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  [launch error N=%d TILE=%d: %s]\n", N, TILE, cudaGetErrorString(e)); return -1.0f; }
    cudaEventRecord(a,s);
    for (int i=0;i<ITERS;i++) head_tiled<N,TILE><<<VOCAB,THREADS,0,s>>>(x,w,l);
    cudaEventRecord(b,s); cudaStreamSynchronize(s);
    float ms=0; cudaEventElapsedTime(&ms,a,b); return ms/ITERS;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    double gb = (double)HEAD_WORDS*4.0/(1024.0*1024.0*1024.0);
    printf("=================================================================\n");
    printf(" TILED BATCHED LM HEAD on %s\n", p.name);
    printf(" head = %.3f GB, one read at 184.6 GB/s = %.2f ms\n", gb, gb/184.6*1000.0);
    printf(" shared mem per SM: %zu KB\n", p.sharedMemPerMultiprocessor/1024);
    printf("=================================================================\n\n");

    float *x,*l; uint32_t* w;
    cudaMalloc(&w, HEAD_WORDS*sizeof(uint32_t));
    cudaMalloc(&x, (size_t)8*HIDDEN*sizeof(float));
    cudaMalloc(&l, (size_t)8*VOCAB*sizeof(float));
    cudaMemset(w,0x55,HEAD_WORDS*sizeof(uint32_t));
    cudaMemset(x,0x3F,(size_t)8*HIDDEN*sizeof(float));
    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);

    printf("rows | GEMV x N | tiled(512) | tiled(256) | best speedup | saved ms\n");
    printf("-----+----------+------------+------------+--------------+---------\n");
    {   float g=t_gemv(4,x,w,l,s,a,b), a1=t_tiled<4,512>(x,w,l,s,a,b), a2=t_tiled<4,256>(x,w,l,s,a,b);
        float bst=(a1>0&&a2>0)?((a1<a2)?a1:a2):((a1>0)?a1:a2);
        printf("%4d | %8.3f | %10.3f | %10.3f | %11.2fx | %7.3f\n",4,g,a1,a2,g/bst,g-bst); }
    {   float g=t_gemv(6,x,w,l,s,a,b), a1=t_tiled<6,512>(x,w,l,s,a,b), a2=t_tiled<6,256>(x,w,l,s,a,b);
        float bst=(a1>0&&a2>0)?((a1<a2)?a1:a2):((a1>0)?a1:a2);
        printf("%4d | %8.3f | %10.3f | %10.3f | %11.2fx | %7.3f\n",6,g,a1,a2,g/bst,g-bst); }
    {   float g=t_gemv(8,x,w,l,s,a,b), a1=t_tiled<8,512>(x,w,l,s,a,b), a2=t_tiled<8,256>(x,w,l,s,a,b);
        float bst=(a1>0&&a2>0)?((a1<a2)?a1:a2):((a1>0)?a1:a2);
        printf("%4d | %8.3f | %10.3f | %10.3f | %11.2fx | %7.3f\n",8,g,a1,a2,g/bst,g-bst);
        float saved = g-bst;
        printf("\nAt K=7 (8 head rows), saving %.2f ms against the measured 79.63 ms step:\n", saved);
        printf("  step -> %.2f ms, 5.81 tok/step -> %.1f tok/s\n",
               79.63f-saved, 5.81f/((79.63f-saved)/1000.0f)); }

    cudaFree(w);cudaFree(x);cudaFree(l);cudaStreamDestroy(s);
    return 0;
}
