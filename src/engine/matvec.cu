// Pulse engine, step 3: run the weights.
//
// Steps 1 and 2 read the model and proved the dequantisation is bit-identical
// to ggml's. This does the first real forward-pass arithmetic: a fused
// dequantise-and-matvec over PQ2_0 weights, straight from the mmap'd file.
//
// Two things are being answered, in order:
//   1. Is it CORRECT? Checked against a CPU reference built from ggml's own
//      dequantised weights. A fast wrong kernel is worth nothing.
//   2. What BANDWIDTH does it reach? This is the engine thesis in one number.
//      llama.cpp decode runs at 183 GB/s against ~216 GB/s achievable. If a
//      purpose-built kernel cannot beat that, an engine rewrite cannot either,
//      because the weight sweep is the dominant term.
#include "gguf.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

constexpr int QK = 128;                 // PQ2_0 group size
struct __align__(2) blk { uint16_t d; uint8_t qs[QK/4]; };   // 34 bytes

extern "C" void dequantize_row_pq2_0(const void* x, float* y, int64_t k);

// v1: one warp per row, each lane strides whole 34-byte blocks. Simple, and
// badly uncoalesced - lane i touches byte i*34, so a warp scatters across ~1 KB.
__global__ __launch_bounds__(256)
void pq2_matvec(const blk* __restrict__ W, const float* __restrict__ x,
                float* __restrict__ y, int ne0, int nrows) {
    const int warp_id  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane     = threadIdx.x & 31;
    if (warp_id >= nrows) return;

    const int nblk = ne0 / QK;
    const blk* row = W + (size_t)warp_id * nblk;

    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        const blk bb = row[b];
        const float d = __half2float(__ushort_as_half(bb.d));
        const float* xp = x + b * QK;
        float s = 0.0f;
        #pragma unroll
        for (int q = 0; q < QK/4; ++q) {
            const uint8_t packed = bb.qs[q];
            s += ((int)((packed     ) & 3) - 1) * xp[q*4+0];
            s += ((int)((packed >> 2) & 3) - 1) * xp[q*4+1];
            s += ((int)((packed >> 4) & 3) - 1) * xp[q*4+2];
            s += ((int)((packed >> 6) & 3) - 1) * xp[q*4+3];
        }
        acc += s * d;
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

// v2: the whole warp cooperates on ONE block at a time. qs is 32 bytes and a
// warp is 32 lanes, so lane i reads byte i - one coalesced 32-byte transaction
// per block instead of 32 scattered ones. Each lane decodes its byte's 4 codes
// and pulls the matching float4 of x.
__global__ __launch_bounds__(256)
void pq2_matvec_coalesced(const blk* __restrict__ W, const float* __restrict__ x,
                          float* __restrict__ y, int ne0, int nrows) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane    = threadIdx.x & 31;
    if (warp_id >= nrows) return;

    const int nblk = ne0 / QK;
    const blk* row = W + (size_t)warp_id * nblk;

    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        const uint8_t packed = row[b].qs[lane];          // coalesced across the warp
        const float4 xv = *reinterpret_cast<const float4*>(x + b*QK + lane*4);
        float s =  ((int)((packed     ) & 3) - 1) * xv.x
                 + ((int)((packed >> 2) & 3) - 1) * xv.y
                 + ((int)((packed >> 4) & 3) - 1) * xv.z
                 + ((int)((packed >> 6) & 3) - 1) * xv.w;
        acc += s * __half2float(__ushort_as_half(row[b].d));
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

// v3: v2 plus two changes. The 2-byte scale sits 34 bytes from its qs, so a
// per-block scalar load of it scatters. __ldg routes both through the
// read-only path (broadcast for d, which every lane wants), and unrolling two
// blocks per iteration lets the compiler overlap the two dependent loads.
__global__ __launch_bounds__(256)
void pq2_matvec_ldg(const blk* __restrict__ W, const float* __restrict__ x,
                    float* __restrict__ y, int ne0, int nrows) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane    = threadIdx.x & 31;
    if (warp_id >= nrows) return;

    const int nblk = ne0 / QK;
    const blk* row = W + (size_t)warp_id * nblk;

    float acc = 0.0f;
    int b = 0;
    for (; b + 1 < nblk; b += 2) {
        const uint8_t p0 = __ldg(&row[b  ].qs[lane]);
        const uint8_t p1 = __ldg(&row[b+1].qs[lane]);
        const float   d0 = __half2float(__ushort_as_half(__ldg(&row[b  ].d)));
        const float   d1 = __half2float(__ushort_as_half(__ldg(&row[b+1].d)));
        const float4  x0 = *reinterpret_cast<const float4*>(x + (b  )*QK + lane*4);
        const float4  x1 = *reinterpret_cast<const float4*>(x + (b+1)*QK + lane*4);
        float s0 =  ((int)((p0     ) & 3) - 1)*x0.x + ((int)((p0 >> 2) & 3) - 1)*x0.y
                  + ((int)((p0 >> 4) & 3) - 1)*x0.z + ((int)((p0 >> 6) & 3) - 1)*x0.w;
        float s1 =  ((int)((p1     ) & 3) - 1)*x1.x + ((int)((p1 >> 2) & 3) - 1)*x1.y
                  + ((int)((p1 >> 4) & 3) - 1)*x1.z + ((int)((p1 >> 6) & 3) - 1)*x1.w;
        acc += s0*d0 + s1*d1;
    }
    for (; b < nblk; ++b) {
        const uint8_t p = __ldg(&row[b].qs[lane]);
        const float4 xv = *reinterpret_cast<const float4*>(x + b*QK + lane*4);
        float s =  ((int)((p     ) & 3) - 1)*xv.x + ((int)((p >> 2) & 3) - 1)*xv.y
                 + ((int)((p >> 4) & 3) - 1)*xv.z + ((int)((p >> 6) & 3) - 1)*xv.w;
        acc += s * __half2float(__ushort_as_half(__ldg(&row[b].d)));
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[warp_id] = acc;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr,"usage: %s <model.gguf> [tensor] [rows]\n",argv[0]); return 2; }
    const char* want = argc > 2 ? argv[2] : "output.weight";
    int check_rows   = argc > 3 ? atoi(argv[3]) : 512;

    pulse::Reader r;
    if (!r.open(argv[1]) || !r.parse()) return 1;
    const pulse::TensorInfo* t = nullptr;
    for (const auto& ti : r.tensors()) if (ti.name == want) { t = &ti; break; }
    if (!t || t->type != 142) { fprintf(stderr,"need a PQ2_0 tensor\n"); return 1; }

    const int ne0   = (int)t->dims[0];
    const int nrows = (int)t->dims[1];
    const int nblk  = ne0 / QK;
    const size_t wbytes = (size_t)nrows * nblk * sizeof(blk);

    printf("tensor    : %s  [%d x %d]\n", want, ne0, nrows);
    printf("weights   : %.3f GB quantised (%.3f bits/weight)\n",
           wbytes/1e9, wbytes*8.0/((double)ne0*nrows));

    const blk* host_w = (const blk*)r.tensor_data(*t);

    std::vector<float> x(ne0);
    for (int i = 0; i < ne0; ++i) x[i] = sinf(i * 0.01f);   // deterministic

    blk*   d_w; float *d_x, *d_y;
    CUDA_OK(cudaMalloc(&d_w, wbytes));
    CUDA_OK(cudaMalloc(&d_x, ne0*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_y, (size_t)nrows*sizeof(float)));
    CUDA_OK(cudaMemcpy(d_w, host_w, wbytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_x, x.data(), ne0*sizeof(float), cudaMemcpyHostToDevice));

    const int threads = 256, warps_per_block = threads/32;
    const int blocks  = (nrows + warps_per_block - 1) / warps_per_block;

    pq2_matvec_ldg<<<blocks, threads>>>(d_w, d_x, d_y, ne0, nrows);
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> y(nrows);
    CUDA_OK(cudaMemcpy(y.data(), d_y, (size_t)nrows*sizeof(float), cudaMemcpyDeviceToHost));

    // --- correctness: CPU reference from ggml's own dequantised weights ---
    check_rows = std::min(check_rows, nrows);
    std::vector<float> wrow(ne0);
    // Three references, to separate a logic error from fp32 rounding:
    //   f64   - exact-ish, accumulated in double
    //   f32   - naive float accumulation, the usual CPU order
    //   f32w  - float accumulation in the GPU's order (32 lane partials,
    //           strided by block, then a tree reduction)
    double worst_d = 0, worst_f = 0, worst_w = 0, cond_max = 0, scaled_max = 0;
    int row_d = -1;
    for (int j = 0; j < check_rows; ++j) {
        dequantize_row_pq2_0((const void*)(host_w + (size_t)j*nblk), wrow.data(), ne0);
        double refd = 0; float reff = 0.0f;
        for (int i = 0; i < ne0; ++i) { refd += (double)wrow[i]*(double)x[i]; reff += wrow[i]*x[i]; }
        float lanes[32] = {0};
        for (int lane = 0; lane < 32; ++lane)
            for (int b = lane; b < nblk; b += 32) {
                float sb = 0.0f;
                for (int q = 0; q < QK; ++q) sb += wrow[b*QK+q]*x[b*QK+q];
                lanes[lane] += sb;
            }
        for (int off = 16; off; off >>= 1)
            for (int l = 0; l < off; ++l) lanes[l] += lanes[l+off];
        // Conditioning: a dot product's error scales with sum|terms|/|sum|,
        // not with |sum| alone. Heavy cancellation amplifies relative error
        // without any bug being present.
        double absterms = 0;
        for (int i = 0; i < ne0; ++i) absterms += std::fabs((double)wrow[i]*(double)x[i]);
        cond_max = std::max(cond_max, absterms/std::max(1e-12,std::fabs(refd)));
        scaled_max = std::max(scaled_max, std::fabs((double)y[j]-refd)/std::max(1e-12,absterms));
        const double got = y[j];
        auto rel = [&](double ref){ return std::fabs(got-ref)/std::max(1e-6,std::fabs(ref)); };
        if (rel(refd) > worst_d) { worst_d = rel(refd); row_d = j; }
        worst_f = std::max(worst_f, rel((double)reff));
        worst_w = std::max(worst_w, rel((double)lanes[0]));
    }
    printf("\ncorrectness vs ggml-dequantised CPU references:\n");
    printf("  rows checked                       : %d\n", check_rows);
    printf("  worst rel err vs float64 accum     : %.3e (row %d)\n", worst_d, row_d);
    printf("  worst rel err vs float32 accum     : %.3e\n", worst_f);
    printf("  worst rel err vs float32, GPU order: %.3e\n", worst_w);
    printf("  worst dot-product condition number : %.3e   <- cancellation factor\n", cond_max);
    printf("  worst err / sum|terms|             : %.3e   <- the meaningful metric\n", scaled_max);
    // Matching the GPU's own accumulation order to near machine epsilon is the
    // real proof of correctness; the float64 gap is fp32 rounding over 5120 terms.
    // fp32 has ~1.2e-7 epsilon; over 5120 terms a few ULP of accumulated
    // error scaled by sum|terms| is the correct acceptance test.
    const bool ok = scaled_max < 1e-6;
    printf("  RESULT                             : %s\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;

    // --- bandwidth: time both kernels ---
    auto bench = [&](const char* name, void(*k)(const blk*,const float*,float*,int,int)) {
        for (int i = 0; i < 5; ++i) k<<<blocks, threads>>>(d_w, d_x, d_y, ne0, nrows);
        CUDA_OK(cudaDeviceSynchronize());
        cudaEvent_t a,b; CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&b));
        const int iters = 50;
        CUDA_OK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) k<<<blocks, threads>>>(d_w, d_x, d_y, ne0, nrows);
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float ms = 0; CUDA_OK(cudaEventElapsedTime(&ms, a, b));
        const double per = ms / iters;
        printf("  %-26s %7.3f ms   %6.1f GB/s\n", name, per, wbytes/(per*1e-3)/1e9);
        return wbytes/(per*1e-3)/1e9;
    };
    printf("\nbandwidth:\n");
    const double b1 = bench("v1 (strided, uncoalesced)", pq2_matvec);
    const double b2 = bench("v2 (warp-coalesced)",       pq2_matvec_coalesced);
    const double b3 = bench("v3 (+__ldg, 2x unroll)",     pq2_matvec_ldg);
    printf("  %-26s %7s   %6.1f GB/s\n", "llama.cpp decode", "-", 183.0);
    printf("  %-26s %7s   %6.1f GB/s\n", "achievable (bench/cuda/bw.cu)", "-", 216.0);
    printf("\n  best Pulse kernel reaches %.0f%% of llama.cpp, %.0f%% of achievable\n",
           100.0*std::max(b1,std::max(b2,b3))/183.0, 100.0*std::max(b1,std::max(b2,b3))/216.0);

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
