// Pulse engine: ops, and a validation harness for them.
//
// Each kernel is checked against a CPU reference built from ggml's own
// dequantised weights. An op that is not validated does not go in the layer
// loop, because a silent numerical bug 30 layers deep is unfindable later.
#include "model.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

extern "C" void dequantize_row_pq2_0(const void* x, float* y, int64_t k);

namespace pulse {

constexpr int QK = 128;
struct __align__(2) blk { uint16_t d; uint8_t qs[QK/4]; };

// ---------------------------------------------------------------- RMSNorm
// y = x / sqrt(mean(x^2) + eps) * w     (one block per row)
__global__ __launch_bounds__(256)
void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
               float* __restrict__ y, int n, float eps) {
    __shared__ float red[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float v = x[i]; ss += v*v; }
    #pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0; for (int i = 0; i < (int)(blockDim.x>>5); ++i) t += red[i];
        red[0] = rsqrtf(t/n + eps);
    }
    __syncthreads();
    const float scale = red[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i]*scale*w[i];
}

// ---------------------------------------------------- PQ2_0 mat-vec (v4)
__global__ __launch_bounds__(256)
void k_matvec_pq2(const blk* __restrict__ W, const float* __restrict__ x,
                  float* __restrict__ y, int ne0, int nrows) {
    const int wid = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= nrows) return;
    const int nblk = ne0/QK;
    const blk* row = W + (size_t)wid*nblk;
    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b) {
        const uint8_t p = __ldg(&row[b].qs[lane]);
        const float   d = __half2float(__ushort_as_half(__ldg(&row[b].d)));
        const float4 xv = *reinterpret_cast<const float4*>(x + b*QK + lane*4);
        acc += (((int)((p    )&3)-1)*xv.x + ((int)((p>>2)&3)-1)*xv.y
              + ((int)((p>>4)&3)-1)*xv.z + ((int)((p>>6)&3)-1)*xv.w) * d;
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[wid] = acc;
}

// ------------------------------------------------------------- SwiGLU
// out = (gate * sigmoid(gate)) * up      (SiLU on the gate branch)
__global__ void k_swiglu(const float* __restrict__ g, const float* __restrict__ u,
                         float* __restrict__ o, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i];
    o[i] = (v / (1.0f + __expf(-v))) * u[i];
}

} // namespace pulse

using namespace pulse;

static void cpu_matvec_pq2(const uint8_t* W, const float* x, float* y, int ne0, int nrows) {
    const int nblk = ne0/QK;
    std::vector<float> row(ne0);
    for (int j = 0; j < nrows; ++j) {
        dequantize_row_pq2_0(W + (size_t)j*nblk*34, row.data(), ne0);
        double s = 0; for (int i = 0; i < ne0; ++i) s += (double)row[i]*(double)x[i];
        y[j] = (float)s;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr,"usage: %s <model.gguf>\n", argv[0]); return 2; }

    cudaEvent_t t0,t1; CU(cudaEventCreate(&t0)); CU(cudaEventCreate(&t1));
    CU(cudaEventRecord(t0));
    Model m;
    if (!m.load(argv[1])) return 1;
    CU(cudaEventRecord(t1)); CU(cudaEventSynchronize(t1));
    float load_ms = 0; CU(cudaEventElapsedTime(&load_ms,t0,t1));
    printf("load time : %.2f s  (%.1f GB/s)\n\n", load_ms/1000.0, m.bytes_on_gpu()/(load_ms*1e-3)/1e9);

    const auto& hp = m.hp();
    const int n = hp.n_embd;

    // layer-type census, straight from the loaded tensors
    int n_full = 0; for (int il = 0; il < hp.n_layer; ++il) n_full += m.is_full_attn(il);
    printf("layer census: %d full-attention, %d gated-delta (SSM)\n\n",
           n_full, hp.n_layer - n_full);

    std::vector<float> hx(n);
    for (int i = 0; i < n; ++i) hx[i] = sinf(i*0.017f);
    float *d_x,*d_y,*d_g,*d_u,*d_o;
    CU(cudaMalloc(&d_x,n*4)); CU(cudaMalloc(&d_y,n*4));
    CU(cudaMemcpy(d_x,hx.data(),n*4,cudaMemcpyHostToDevice));

    int pass = 0, total = 0;
    auto report = [&](const char* name, double err, double tol) {
        ++total; const bool ok = err < tol; pass += ok;
        printf("  %-34s err %.3e  tol %.0e  %s\n", name, err, tol, ok?"PASS":"FAIL");
    };

    printf("op validation (vs CPU reference from ggml-dequantised weights):\n");

    // ---- RMSNorm on blk.0.attn_norm
    if (const DevTensor* w = m.layer(0,"attn_norm.weight")) {
        k_rmsnorm<<<1,256>>>(d_x,(const float*)w->ptr,d_y,n,hp.rms_eps);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(n); CU(cudaMemcpy(got.data(),d_y,n*4,cudaMemcpyDeviceToHost));
        std::vector<float> hw(n);
        CU(cudaMemcpy(hw.data(), w->ptr, n*4, cudaMemcpyDeviceToHost));
        double ss = 0; for (int i=0;i<n;++i) ss += (double)hx[i]*hx[i];
        const double sc = 1.0/std::sqrt(ss/n + hp.rms_eps);
        double e = 0, mag = 0;
        for (int i=0;i<n;++i) { const double r = hx[i]*sc*hw[i];
            e = std::max(e, std::fabs(got[i]-r)); mag = std::max(mag, std::fabs(r)); }
        report("rmsnorm(blk.0.attn_norm)", e/std::max(mag,1e-9), 1e-5);
    }

    // ---- PQ2_0 mat-vec on the FFN gate projection
    const DevTensor* gt = m.layer(0,"ffn_gate.weight");
    if (gt && gt->type == T_PQ2_0) {
        const int rows = (int)gt->ne[1];
        float* d_gate; CU(cudaMalloc(&d_gate, (size_t)rows*4));
        k_matvec_pq2<<<(rows+7)/8,256>>>((const blk*)gt->ptr,d_x,d_gate,n,rows);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(rows);
        CU(cudaMemcpy(got.data(),d_gate,(size_t)rows*4,cudaMemcpyDeviceToHost));
        const int chk = std::min(rows, 512);
        std::vector<float> ref(chk);
        cpu_matvec_pq2(m.host_data("blk.0.ffn_gate.weight"), hx.data(), ref.data(), n, chk);
        // scale error by sum|terms|: these dot products cancel heavily
        std::vector<float> row(n); double worst = 0;
        for (int j = 0; j < chk; ++j) {
            dequantize_row_pq2_0(m.host_data("blk.0.ffn_gate.weight")+(size_t)j*(n/QK)*34, row.data(), n);
            double at = 0; for (int i=0;i<n;++i) at += std::fabs((double)row[i]*hx[i]);
            worst = std::max(worst, std::fabs((double)got[j]-ref[j])/std::max(at,1e-12));
        }
        report("matvec_pq2(blk.0.ffn_gate)", worst, 1e-6);
        cudaFree(d_gate);
    }

    // ---- SwiGLU
    {
        const int nf = 4096;
        std::vector<float> a(nf), b(nf);
        for (int i=0;i<nf;++i){ a[i]=sinf(i*0.03f)*3.0f; b[i]=cosf(i*0.02f)*2.0f; }
        CU(cudaMalloc(&d_g,nf*4)); CU(cudaMalloc(&d_u,nf*4)); CU(cudaMalloc(&d_o,nf*4));
        CU(cudaMemcpy(d_g,a.data(),nf*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_u,b.data(),nf*4,cudaMemcpyHostToDevice));
        k_swiglu<<<(nf+255)/256,256>>>(d_g,d_u,d_o,nf);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(nf); CU(cudaMemcpy(got.data(),d_o,nf*4,cudaMemcpyDeviceToHost));
        double e=0, mag=0;
        for (int i=0;i<nf;++i){ const double r = (a[i]/(1.0+std::exp(-(double)a[i])))*b[i];
            e = std::max(e, std::fabs(got[i]-r)); mag = std::max(mag, std::fabs(r)); }
        report("swiglu", e/std::max(mag,1e-9), 1e-5);
        cudaFree(d_g); cudaFree(d_u); cudaFree(d_o);
    }

    printf("\n%d/%d ops validated\n", pass, total);
    cudaFree(d_x); cudaFree(d_y);
    return pass == total ? 0 : 1;
}
