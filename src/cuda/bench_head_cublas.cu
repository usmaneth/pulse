// What is the real floor for a batched LM head on GB10?
//
// Established so far on this device:
//   * per-row 4-bit head GEMV is exactly bandwidth-bound: 8 rows = 26.0 ms measured
//     against 8 x 0.592 GB / 184.6 GB/s = 25.7 ms predicted;
//   * three hand-written batched kernels were all SLOWER than the GEMV, sitting
//     12x off the 3.2 ms single-read floor, i.e. bound by shared-memory bank
//     conflicts and register pressure rather than bandwidth.
//
// Before investing in a tuned mixed-input kernel, measure what a mature library GEMM
// achieves for the same logical operation. cuBLAS on an FP16 head is a pessimistic
// proxy: the FP16 head is 2.5 GB versus 0.592 GB for 4-bit, so it moves 4.2x more
// bytes. If even that beats 8 sequential 4-bit GEMVs, the batching win is real and the
// remaining work is purely kernel engineering.
//
// Every number is a cudaEvent measurement.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int HIDDEN = 5120;
constexpr int VOCAB  = 248320;
constexpr int WARMUP = 3, ITERS = 10;

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    double head_fp16_gb = (double)HIDDEN * VOCAB * 2.0 / (1024.0*1024.0*1024.0);
    double head_q4_gb   = (double)HIDDEN * VOCAB * 0.5 / (1024.0*1024.0*1024.0);
    printf("=================================================================\n");
    printf(" BATCHED HEAD FLOOR via cuBLAS on %s\n", p.name);
    printf(" hidden=%d vocab=%d\n", HIDDEN, VOCAB);
    printf(" head as FP16 = %.3f GB (one read @184.6 GB/s = %.2f ms)\n",
           head_fp16_gb, head_fp16_gb/184.6*1000.0);
    printf(" head as 4bit = %.3f GB (one read @184.6 GB/s = %.2f ms)\n",
           head_q4_gb, head_q4_gb/184.6*1000.0);
    printf(" reference: 8 sequential 4-bit GEMVs measured 26.0 ms\n");
    printf("=================================================================\n\n");

    cublasHandle_t h; 
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { printf("cublasCreate failed\n"); return 1; }
    cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);

    __half *dW=nullptr, *dX=nullptr, *dY=nullptr;
    size_t wbytes = (size_t)HIDDEN * VOCAB * sizeof(__half);
    if (cudaMalloc(&dW, wbytes) != cudaSuccess) { printf("alloc W (%.2f GB) failed\n", wbytes/1073741824.0); return 1; }
    cudaMalloc(&dX, (size_t)8 * HIDDEN * sizeof(__half));
    cudaMalloc(&dY, (size_t)8 * VOCAB * sizeof(__half));
    cudaMemset(dW, 0x11, wbytes);
    cudaMemset(dX, 0x11, (size_t)8 * HIDDEN * sizeof(__half));

    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);

    printf("rows | cuBLAS FP16 GEMM (ms) | vs 8x q4 GEMV | implied step @K=7 | tok/s\n");
    printf("-----+-----------------------+---------------+-------------------+-------\n");

    for (int n : {1, 2, 4, 6, 8}) {
        // C[VOCAB x n] = W^T[VOCAB x HIDDEN] * X[HIDDEN x n]
        for (int i = 0; i < WARMUP; ++i)
            cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB, n, HIDDEN,
                        &alpha, dW, HIDDEN, dX, HIDDEN, &beta, dY, VOCAB);
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int i = 0; i < ITERS; ++i)
            cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB, n, HIDDEN,
                        &alpha, dW, HIDDEN, dX, HIDDEN, &beta, dY, VOCAB);
        cudaEventRecord(b); cudaDeviceSynchronize();
        float ms=0; cudaEventElapsedTime(&ms,a,b); ms/=ITERS;

        if (n == 8) {
            float saved = 26.025f - ms;
            float step  = 79.63f - saved;
            printf("%4d | %21.3f | %12.2fx | %17.2f | %6.1f\n",
                   n, ms, 26.025f/ms, step, 5.81f/(step/1000.0f));
        } else {
            printf("%4d | %21.3f | %12s | %17s | %6s\n", n, ms, "-", "-", "-");
        }
    }

    printf("\nInterpretation: cuBLAS here reads a 2.5 GB FP16 head, 4.2x more bytes than\n");
    printf("the 0.592 GB 4-bit head. A mixed-input INT4 x FP16 tensor-core GEMM would\n");
    printf("move 4.2x less and should approach the 3.21 ms single-read floor.\n");

    cudaFree(dW); cudaFree(dX); cudaFree(dY); cublasDestroy(h);
    return 0;
}
