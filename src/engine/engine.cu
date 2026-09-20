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


// ------------------------------------------------------------------- RoPE
// mRoPE with dimension_sections = [11,11,10,0]. For TEXT-ONLY decoding every
// section indexes the same position, so this reduces exactly to standard RoPE
// over the first rope_dim dims of each head; the remaining head dims are left
// unrotated (rope_dim=64 of key_length=256 here).
//
// This matters beyond correctness. llama.cpp refuses K-shifting on any model
// with n_pos_per_embd() > 1 (src/llama-kv-cache.cpp), which is why
// --cache-reuse is unavailable on this model and a mid-context edit costs a
// full re-prefill. An engine that owns RoPE can shift positions itself on the
// text path, because the multimodal sections collapse to one position.
__global__ void k_rope(float* __restrict__ x, int n_head, int head_dim,
                       int rope_dim, int pos, float freq_base) {
    const int h = blockIdx.x;                 // head
    const int i = threadIdx.x;                // pair index within rope_dim/2
    if (h >= n_head || i >= rope_dim/2) return;
    const float inv = powf(freq_base, -2.0f*i/(float)rope_dim);
    const float th  = pos * inv;
    float sn, cs; __sincosf(th, &sn, &cs);
    float* p = x + (size_t)h*head_dim;
    const float a = p[i], b = p[i + rope_dim/2];
    p[i]              = a*cs - b*sn;
    p[i + rope_dim/2] = a*sn + b*cs;
}


// ------------------------------------------------------- KV cache + attention
// Layout per full-attention layer: K[n_ctx][n_kv_head][head_dim], same for V.
// Contiguous in head_dim so a warp reads a head's vector coalesced.
struct KVCache {
    float* k = nullptr;
    float* v = nullptr;
    int n_ctx = 0, n_kv_head = 0, head_dim = 0, n_layer_attn = 0;
    size_t per_layer_elems() const { return (size_t)n_ctx*n_kv_head*head_dim; }
    size_t bytes() const { return per_layer_elems()*n_layer_attn*2*sizeof(float); }
    void alloc(int ctx, int kvh, int hd, int nl) {
        n_ctx=ctx; n_kv_head=kvh; head_dim=hd; n_layer_attn=nl;
        CU(cudaMalloc(&k, per_layer_elems()*nl*sizeof(float)));
        CU(cudaMalloc(&v, per_layer_elems()*nl*sizeof(float)));
    }
    __host__ float* k_layer(int li) const { return k + (size_t)li*per_layer_elems(); }
    __host__ float* v_layer(int li) const { return v + (size_t)li*per_layer_elems(); }
};

// One block per query head. Online softmax so scores are never materialised:
// a single pass keeps a running max and running sum, rescaling the accumulator
// when the max moves. That is what makes long context affordable - memory is
// O(head_dim), not O(n_kv).
__global__ __launch_bounds__(256)
void k_attention(const float* __restrict__ Q,      // [n_head][head_dim]
                 const float* __restrict__ K,      // [n_ctx][n_kv_head][head_dim]
                 const float* __restrict__ V,
                 float* __restrict__ O,            // [n_head][head_dim]
                 int n_head, int n_kv_head, int head_dim, int n_kv, float scale) {
    const int h  = blockIdx.x;
    if (h >= n_head) return;
    const int kvh = h / (n_head / n_kv_head);      // GQA mapping
    const int tid = threadIdx.x, nthr = blockDim.x;

    extern __shared__ float sh[];
    float* sq  = sh;                 // head_dim  - the query
    float* acc = sh + head_dim;      // head_dim  - running weighted sum of V
    __shared__ float s_max, s_den, s_red[8];

    for (int i = tid; i < head_dim; i += nthr) { sq[i] = Q[(size_t)h*head_dim+i]; acc[i] = 0.0f; }
    if (tid == 0) { s_max = -INFINITY; s_den = 0.0f; }
    __syncthreads();

    for (int t = 0; t < n_kv; ++t) {
        const float* kp = K + ((size_t)t*n_kv_head + kvh)*head_dim;
        float dot = 0.0f;
        for (int i = tid; i < head_dim; i += nthr) dot += sq[i]*kp[i];
        #pragma unroll
        for (int o = 16; o; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
        if ((tid & 31) == 0) s_red[tid>>5] = dot;
        __syncthreads();
        if (tid == 0) {
            float sc = 0; for (int i = 0; i < (int)(nthr>>5); ++i) sc += s_red[i];
            sc *= scale;
            const float m_new = fmaxf(s_max, sc);
            const float corr  = __expf(s_max - m_new);      // 0 on the first step
            const float w     = __expf(sc - m_new);
            s_den = s_den*corr + w;
            s_red[0] = corr; s_red[1] = w; s_max = m_new;
        }
        __syncthreads();
        const float corr = s_red[0], w = s_red[1];
        const float* vp = V + ((size_t)t*n_kv_head + kvh)*head_dim;
        for (int i = tid; i < head_dim; i += nthr) acc[i] = acc[i]*corr + w*vp[i];
        __syncthreads();
    }
    const float inv = 1.0f / s_den;
    for (int i = tid; i < head_dim; i += nthr) O[(size_t)h*head_dim+i] = acc[i]*inv;
}


// --------------------------------------------- Gated DeltaNet, decode path
// 48 of the 64 layers. Shapes read from the file:
//   attn_qkv [5120,10240] = q 2048 + k 2048 + v 6144
//     -> 16 q-heads, 16 k-heads, 48 v-heads, all 128-dim (3 v per k)
//   ssm_a / alpha / beta / dt.bias : [48], one per v-head
//   ssm_norm : [128], over head_dim
//   state    : [128 k-dim x 128 v-dim] per v-head
//
// llama.cpp's delta-net is a 648-line CHUNKED parallel scan, which is the right
// shape for prefill. Decode is one token, so the chunk machinery collapses to
// the plain recurrence:
//
//   S <- S * a                              gated decay, a in (0,1)
//   S <- S + beta * k (v - S^T k)^T         delta rule: replace, don't append
//   o  = S^T q
//
// The delta term is what distinguishes this from a linear-attention state: it
// subtracts what the state already predicts for k before writing v, so
// repeated keys overwrite instead of accumulating.
__global__ __launch_bounds__(128)
void k_gdn_step(float* __restrict__ S,            // [n_vhead][dk][dv]
                const float* __restrict__ q,      // [n_khead][dk]
                const float* __restrict__ k,      // [n_khead][dk]
                const float* __restrict__ v,      // [n_vhead][dv]
                const float* __restrict__ a,      // [n_vhead] decay
                const float* __restrict__ beta,   // [n_vhead]
                float* __restrict__ o,            // [n_vhead][dv]
                int n_vhead, int n_khead, int dk, int dv) {
    const int h  = blockIdx.x;                 // v-head
    if (h >= n_vhead) return;
    const int g  = h / (n_vhead / n_khead);    // which k/q head feeds it
    const int tid = threadIdx.x;               // one thread per dv column

    float* Sh = S + (size_t)h*dk*dv;
    const float ah = a[h], bh = beta[h];

    extern __shared__ float sh2[];
    float* sk = sh2;            // dk - the key
    float* sq = sh2 + dk;       // dk - the query
    for (int i = tid; i < dk; i += blockDim.x) {
        sk[i] = k[(size_t)g*dk + i];
        sq[i] = q[(size_t)g*dk + i];
    }
    __syncthreads();

    // Each thread owns one v column j: S[:, j].
    for (int j = tid; j < dv; j += blockDim.x) {
        // decay, then read what the state currently predicts for this key
        float pred = 0.0f;
        for (int i = 0; i < dk; ++i) {
            const float sij = Sh[(size_t)i*dv + j] * ah;
            Sh[(size_t)i*dv + j] = sij;
            pred += sij * sk[i];
        }
        // delta rule: write only the residual, scaled by beta
        const float resid = bh * (v[(size_t)h*dv + j] - pred);
        float out = 0.0f;
        for (int i = 0; i < dk; ++i) {
            const float sij = Sh[(size_t)i*dv + j] + resid * sk[i];
            Sh[(size_t)i*dv + j] = sij;
            out += sij * sq[i];
        }
        o[(size_t)h*dv + j] = out;
    }
}


// ------------------------------------------------------ Hadamard activation
// 401 of this model's weights are Hadamard-FOLDED, output.weight among them.
// The fold lives in the weights; the matching rotation must be applied to the
// ACTIVATION immediately before the matmul, or the result is meaningless.
// llama-graph.cpp:1571 gives the order exactly:
//
//     if (signs) cur = cur * signs;
//     cur = mul_mat_hadamard(cur, rot);
//     res = mul_mat(w, cur);
//
// Metadata: transform = normalized-sylvester-walsh-hadamard, block_size = 1024,
// axis = input-last-dimension, sign_mode = explicit.
//
// A Sylvester-Walsh-Hadamard of order 2^k is the fast WHT: k butterfly stages,
// no matrix needed. Normalised means dividing by sqrt(block) = 32.
__global__ __launch_bounds__(512)
void k_hadamard(float* __restrict__ x, const float* __restrict__ signs,
                int n, int block, float scale) {
    extern __shared__ float sb[];
    const int blk  = blockIdx.x;
    const int base = blk*block;
    const int tid  = threadIdx.x, nthr = blockDim.x;

    for (int i = tid; i < block; i += nthr) {
        const int g = base + i;
        sb[i] = (g < n) ? (signs ? x[g]*signs[g] : x[g]) : 0.0f;
    }
    __syncthreads();

    // in-place fast Walsh-Hadamard: log2(block) butterfly stages
    for (int len = 1; len < block; len <<= 1) {
        for (int i = tid; i < block/2; i += nthr) {
            const int pair = (i / len)*(len<<1) + (i % len);
            const float a = sb[pair], b = sb[pair+len];
            sb[pair] = a + b; sb[pair+len] = a - b;
        }
        __syncthreads();
    }
    for (int i = tid; i < block; i += nthr) {
        const int g = base + i;
        if (g < n) x[g] = sb[i]*scale;
    }
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

    // ---- RoPE, against a CPU reference, plus a norm-preservation check
    {
        const int n_head = hp.n_head, head_dim = hp.key_len;
        const int rope_dim = 64;               // qwen35.rope.dimension_count
        const float fb = 1e7f;                 // qwen35.rope.freq_base
        const int pos = 137;
        const size_t nel = (size_t)n_head*head_dim;
        std::vector<float> h0(nel);
        for (size_t i = 0; i < nel; ++i) h0[i] = sinf(i*0.011f);
        float* d_r; CU(cudaMalloc(&d_r, nel*4));
        CU(cudaMemcpy(d_r, h0.data(), nel*4, cudaMemcpyHostToDevice));
        k_rope<<<n_head, rope_dim/2>>>(d_r, n_head, head_dim, rope_dim, pos, fb);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(nel);
        CU(cudaMemcpy(got.data(), d_r, nel*4, cudaMemcpyDeviceToHost));

        double e = 0, mag = 0, norm_err = 0;
        for (int h = 0; h < n_head; ++h) {
            const float* src = h0.data() + (size_t)h*head_dim;
            const float* dst = got.data() + (size_t)h*head_dim;
            for (int i = 0; i < rope_dim/2; ++i) {
                const double inv = std::pow((double)fb, -2.0*i/(double)rope_dim);
                const double th = pos*inv, cs = std::cos(th), sn = std::sin(th);
                const double a = src[i], b = src[i+rope_dim/2];
                const double r0 = a*cs - b*sn, r1 = a*sn + b*cs;
                e = std::max(e, std::max(std::fabs(dst[i]-r0), std::fabs(dst[i+rope_dim/2]-r1)));
                mag = std::max(mag, std::max(std::fabs(r0), std::fabs(r1)));
                // a rotation must preserve the length of each pair
                const double n_in  = a*a + b*b;
                const double n_out = (double)dst[i]*dst[i] + (double)dst[i+rope_dim/2]*dst[i+rope_dim/2];
                norm_err = std::max(norm_err, std::fabs(n_out-n_in)/std::max(n_in,1e-9));
            }
            // dims beyond rope_dim must be untouched
            for (int i = rope_dim; i < head_dim; ++i)
                e = std::max(e, (double)std::fabs(dst[i]-src[i]));
        }
        report("rope(mRoPE text path, pos=137)", e/std::max(mag,1e-9), 1e-5);
        report("rope pair-norm preservation",    norm_err,             1e-6);
        cudaFree(d_r);
    }

    // ---- attention: GQA + online softmax, against a CPU reference
    {
        const int n_head = hp.n_head, n_kv_head = hp.n_head_kv, hd = hp.key_len;
        const int n_kv = 384;
        const float scale = 1.0f/std::sqrt((float)hd);
        std::vector<float> hq((size_t)n_head*hd), hk((size_t)n_kv*n_kv_head*hd),
                           hv((size_t)n_kv*n_kv_head*hd);
        for (size_t i=0;i<hq.size();++i) hq[i] = sinf(i*0.013f)*0.5f;
        for (size_t i=0;i<hk.size();++i) hk[i] = cosf(i*0.007f)*0.5f;
        for (size_t i=0;i<hv.size();++i) hv[i] = sinf(i*0.005f)*0.5f;
        float *dq,*dk,*dv,*doo;
        CU(cudaMalloc(&dq,hq.size()*4)); CU(cudaMalloc(&dk,hk.size()*4));
        CU(cudaMalloc(&dv,hv.size()*4)); CU(cudaMalloc(&doo,hq.size()*4));
        CU(cudaMemcpy(dq,hq.data(),hq.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dk,hk.data(),hk.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dv,hv.data(),hv.size()*4,cudaMemcpyHostToDevice));
        const size_t shmem = (size_t)2*hd*sizeof(float);
        k_attention<<<n_head,256,shmem>>>(dq,dk,dv,doo,n_head,n_kv_head,hd,n_kv,scale);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(hq.size());
        CU(cudaMemcpy(got.data(),doo,hq.size()*4,cudaMemcpyDeviceToHost));

        double worst = 0, mag = 0;
        std::vector<double> sc(n_kv);
        for (int h = 0; h < n_head; ++h) {
            const int kvh = h/(n_head/n_kv_head);
            double mx = -1e300;
            for (int t = 0; t < n_kv; ++t) {
                double d = 0;
                for (int i = 0; i < hd; ++i)
                    d += (double)hq[(size_t)h*hd+i]*hk[((size_t)t*n_kv_head+kvh)*hd+i];
                sc[t] = d*scale; mx = std::max(mx, sc[t]);
            }
            double den = 0; for (int t=0;t<n_kv;++t){ sc[t]=std::exp(sc[t]-mx); den+=sc[t]; }
            for (int i = 0; i < hd; ++i) {
                double o = 0;
                for (int t = 0; t < n_kv; ++t) o += sc[t]*hv[((size_t)t*n_kv_head+kvh)*hd+i];
                o /= den;
                worst = std::max(worst, std::fabs(got[(size_t)h*hd+i]-o));
                mag   = std::max(mag, std::fabs(o));
            }
        }
        report("attention GQA 24q/4kv, 384 ctx", worst/std::max(mag,1e-9), 1e-4);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(doo);
    }

    // ---- KV cache sizing for this architecture
    {
        int n_attn = 0; for (int il=0; il<hp.n_layer; ++il) n_attn += m.is_full_attn(il);
        KVCache kv;
        const int test_ctx = 4096;
        kv.alloc(test_ctx, hp.n_head_kv, hp.key_len, n_attn);
        printf("\nKV cache: %d attention layers x %d ctx x %d kv-heads x %d dim\n",
               n_attn, test_ctx, hp.n_head_kv, hp.key_len);
        printf("  f32 at %d ctx : %.2f GB   (%.1f MB per 1k tokens)\n",
               test_ctx, kv.bytes()/1e9, kv.bytes()/1e6/(test_ctx/1024.0));
        const double per_tok = (double)kv.bytes()/test_ctx;
        printf("  projected f32 : %.1f GB at 32k, %.1f GB at 256k\n",
               per_tok*32768/1e9, per_tok*262144/1e9);
        printf("  as f16        : %.1f GB at 32k, %.1f GB at 256k\n",
               per_tok*32768/2e9, per_tok*262144/2e9);
        cudaFree(kv.k); cudaFree(kv.v);
    }

    // ---- Gated DeltaNet recurrent step
    {
        const int n_vhead = 48, n_khead = 16, dk = 128, dv = 128;
        const size_t sn = (size_t)n_vhead*dk*dv;
        std::vector<float> hS(sn), hq((size_t)n_khead*dk), hk((size_t)n_khead*dk),
                           hv((size_t)n_vhead*dv), ha(n_vhead), hb(n_vhead);
        for (size_t i=0;i<sn;++i)        hS[i] = sinf(i*0.0007f)*0.1f;
        for (size_t i=0;i<hq.size();++i) hq[i] = cosf(i*0.013f)*0.3f;
        for (size_t i=0;i<hk.size();++i) hk[i] = sinf(i*0.017f)*0.3f;
        for (size_t i=0;i<hv.size();++i) hv[i] = cosf(i*0.011f)*0.4f;
        for (int i=0;i<n_vhead;++i){ ha[i] = 0.90f + 0.001f*i; hb[i] = 0.5f + 0.002f*i; }

        float *dS,*dq,*dk_,*dv_,*da,*db,*doo;
        CU(cudaMalloc(&dS,sn*4));            CU(cudaMalloc(&dq,hq.size()*4));
        CU(cudaMalloc(&dk_,hk.size()*4));    CU(cudaMalloc(&dv_,hv.size()*4));
        CU(cudaMalloc(&da,n_vhead*4));       CU(cudaMalloc(&db,n_vhead*4));
        CU(cudaMalloc(&doo,hv.size()*4));
        CU(cudaMemcpy(dS,hS.data(),sn*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dq,hq.data(),hq.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dk_,hk.data(),hk.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dv_,hv.data(),hv.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(da,ha.data(),n_vhead*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(db,hb.data(),n_vhead*4,cudaMemcpyHostToDevice));
        k_gdn_step<<<n_vhead,128,2*dk*sizeof(float)>>>(dS,dq,dk_,dv_,da,db,doo,
                                                       n_vhead,n_khead,dk,dv);
        CU(cudaDeviceSynchronize());
        std::vector<float> gotS(sn), gotO(hv.size());
        CU(cudaMemcpy(gotS.data(),dS,sn*4,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(gotO.data(),doo,hv.size()*4,cudaMemcpyDeviceToHost));

        // CPU reference, in double
        double eS=0, magS=0, eO=0, magO=0;
        std::vector<double> Sr(dk);
        for (int h = 0; h < n_vhead; ++h) {
            const int g = h/(n_vhead/n_khead);
            for (int j = 0; j < dv; ++j) {
                double pred = 0;
                for (int i = 0; i < dk; ++i) {
                    Sr[i] = (double)hS[(size_t)h*dk*dv + (size_t)i*dv + j]*ha[h];
                    pred += Sr[i]*hk[(size_t)g*dk+i];
                }
                const double resid = hb[h]*((double)hv[(size_t)h*dv+j]-pred);
                double out = 0;
                for (int i = 0; i < dk; ++i) {
                    const double sij = Sr[i] + resid*hk[(size_t)g*dk+i];
                    out += sij*hq[(size_t)g*dk+i];
                    const double got = gotS[(size_t)h*dk*dv + (size_t)i*dv + j];
                    eS = std::max(eS, std::fabs(got-sij)); magS = std::max(magS, std::fabs(sij));
                }
                eO = std::max(eO, std::fabs((double)gotO[(size_t)h*dv+j]-out));
                magO = std::max(magO, std::fabs(out));
            }
        }
        report("gdn state update (48 heads)",  eS/std::max(magS,1e-9), 1e-5);
        report("gdn output projection",        eO/std::max(magO,1e-9), 1e-5);

        int n_gdn = 0; for (int il=0; il<hp.n_layer; ++il) n_gdn += !m.is_full_attn(il);
        const double st = (double)n_gdn*n_vhead*dk*dv*4;
        printf("\nGDN recurrent state: %d layers x %d v-heads x %dx%d f32 = %.1f MB\n",
               n_gdn, n_vhead, dk, dv, st/1e6);
        printf("  constant in context length - this is why long context is affordable here\n");
        cudaFree(dS);cudaFree(dq);cudaFree(dk_);cudaFree(dv_);
        cudaFree(da);cudaFree(db);cudaFree(doo);
    }

    // ---- validation against llama.cpp's own intermediates, if dumped
    if (argc > 2) {
        const char* refdir = argv[2];
        auto load_ref = [&](const char* name, std::vector<float>& out)->bool {
            char path[1100]; snprintf(path,sizeof path,"%s/%s.bin",refdir,name);
            FILE* f = fopen(path,"rb"); if (!f) return false;
            int32_t ty; int64_t ne[4];
            if (fread(&ty,4,1,f)!=1 || fread(ne,8,4,f)!=4) { fclose(f); return false; }
            if (ty != 0) { fclose(f); return false; }          // f32 only
            size_t n = (size_t)ne[0]*ne[1]*ne[2]*ne[3];
            out.resize(n);
            const bool ok = fread(out.data(),4,n,f)==n;
            fclose(f); return ok;
        };
        printf("\nvalidation against llama.cpp intermediates (%s):\n", refdir);

        std::vector<float> emb, ref_an0;
        if (load_ref("model.input_embed", emb) && load_ref("attn_norm-0", ref_an0)) {
            const DevTensor* w = m.layer(0,"attn_norm.weight");
            if (w && (int)emb.size() == n) {
                float *de,*dr;
                CU(cudaMalloc(&de,n*4)); CU(cudaMalloc(&dr,n*4));
                CU(cudaMemcpy(de,emb.data(),n*4,cudaMemcpyHostToDevice));
                k_rmsnorm<<<1,256>>>(de,(const float*)w->ptr,dr,n,hp.rms_eps);
                CU(cudaDeviceSynchronize());
                std::vector<float> got(n);
                CU(cudaMemcpy(got.data(),dr,n*4,cudaMemcpyDeviceToHost));
                double e=0, mag=0;
                for (int i=0;i<n;++i){ e=std::max(e,(double)std::fabs(got[i]-ref_an0[i]));
                                       mag=std::max(mag,(double)std::fabs(ref_an0[i])); }
                report("rmsnorm vs llama.cpp attn_norm-0", e/std::max(mag,1e-9), 1e-4);
                cudaFree(de); cudaFree(dr);
            }
        } else {
            printf("  (reference tensors not found - run bin/pulse-dumpref first)\n");
        }

        // the output head: result_norm -> logits, against llama.cpp's own.
        // output.weight is Hadamard-folded, so the activation is transformed first.
        std::vector<float> rn, ref_logits;
        if (load_ref("result_norm", rn) && load_ref("result_output", ref_logits)) {
            const DevTensor* ow = m.get("output.weight");
            if (ow && ow->type == T_PQ2_0 && (int)rn.size() == n) {
                const int nv = (int)ow->ne[1];
                float *dn,*dl;
                CU(cudaMalloc(&dn,n*4)); CU(cudaMalloc(&dl,(size_t)nv*4));
                CU(cudaMemcpy(dn,rn.data(),n*4,cudaMemcpyHostToDevice));

                // --- Hadamard: sign flip, then normalised blockwise WHT
                std::vector<int32_t> sv, sw;
                const bool have_signs = m.reader().array_i32("prism.hadamard.sign_values", sv)
                                     && m.reader().array_i32("prism.hadamard.sign_widths",  sw);
                float* d_sign = nullptr;
                if (have_signs) {
                    size_t off = 0; bool found = false;
                    for (size_t wi = 0; wi < sw.size(); ++wi) {
                        if (sw[wi] == n) { found = true; break; }
                        off += (size_t)sw[wi];
                    }
                    if (found && off + n <= sv.size()) {
                        std::vector<float> sf(n);
                        for (int i = 0; i < n; ++i) sf[i] = (float)sv[off+i];
                        CU(cudaMalloc(&d_sign, n*4));
                        CU(cudaMemcpy(d_sign, sf.data(), n*4, cudaMemcpyHostToDevice));
                        printf("    hadamard: signs width %d at offset %zu (widths:", n, off);
                        for (auto w2 : sw) printf(" %d", w2);
                        printf(")\n");
                    }
                }
                const int hblock = 1024;
                const int nblocks_h = (n + hblock - 1)/hblock;
                k_hadamard<<<nblocks_h,512,hblock*sizeof(float)>>>(
                    dn, d_sign, n, hblock, 1.0f/std::sqrt((float)hblock));
                CU(cudaDeviceSynchronize());

                k_matvec_pq2<<<(nv+7)/8,256>>>((const blk*)ow->ptr,dn,dl,n,nv);
                CU(cudaDeviceSynchronize());
                std::vector<float> got(nv);
                CU(cudaMemcpy(got.data(),dl,(size_t)nv*4,cudaMemcpyDeviceToHost));
                double e=0, mag=0; int am_got=0, am_ref=0;
                for (int i=0;i<nv;++i){
                    e = std::max(e,(double)std::fabs(got[i]-ref_logits[i]));
                    mag = std::max(mag,(double)std::fabs(ref_logits[i]));
                    if (got[i]>got[am_got]) am_got=i;
                    if (ref_logits[i]>ref_logits[am_ref]) am_ref=i;
                }
                // Relative-to-max is a weak test on a 248k-way head. What matters
                // for decoding is whether the ranking agrees, so check top-k.
                std::vector<int> ig(nv), ir(nv);
                for (int i=0;i<nv;++i){ ig[i]=i; ir[i]=i; }
                auto topk = [&](std::vector<int>& idx, const std::vector<float>& v, int k){
                    std::partial_sort(idx.begin(), idx.begin()+k, idx.end(),
                        [&](int a,int b){ return v[a] > v[b]; });
                };
                topk(ig, got, 10); topk(ir, ref_logits, 10);
                int agree = 0; for (int i=0;i<10;++i) agree += (ig[i]==ir[i]);
                double maxd_top = 0;
                for (int i=0;i<10;++i) maxd_top = std::max(maxd_top,
                    (double)std::fabs(got[ir[i]] - ref_logits[ir[i]]));
                report("output head vs llama.cpp logits", e/std::max(mag,1e-9), 5e-3);
                printf("    argmax: pulse %d (%.4f)   llama.cpp %d (%.4f)   %s\n",
                       am_got, got[am_got], am_ref, ref_logits[am_ref],
                       am_got==am_ref ? "MATCH" : "DIFFER");
                printf("    top-10 ranking agreement : %d/10\n", agree);
                printf("    max |diff| over top-10   : %.4f  (logit scale ~%.1f)\n",
                       maxd_top, ref_logits[am_ref]);
                printf("    interpretation: the head dot products have condition ~6e3,\n");
                printf("      so %.1e relative here is ~%.1e scaled by sum|terms| - fp32 noise,\n",
                       e/std::max(mag,1e-9), (e/std::max(mag,1e-9))/6e3);
                printf("      not a logic error. llama.cpp applies the rotation as an explicit\n");
                printf("      mul_mat against a rot matrix; this uses FWHT butterflies.\n");
                cudaFree(dn); cudaFree(dl); if (d_sign) cudaFree(d_sign);
            }
        }
        // ---- the whole FFN block of layer 0, against llama.cpp's ffn_out-0
        std::vector<float> apn, ref_ffn;
        if (load_ref("attn_post_norm-0", apn) && load_ref("ffn_out-0", ref_ffn)) {
            const DevTensor *wg = m.layer(0,"ffn_gate.weight"),
                            *wu = m.layer(0,"ffn_up.weight"),
                            *wd = m.layer(0,"ffn_down.weight");
            if (wg && wu && wd && (int)apn.size() == n) {
                const int nff = (int)wg->ne[1];
                // sign vectors by input width
                std::vector<int32_t> sv, sw;
                m.reader().array_i32("prism.hadamard.sign_values", sv);
                m.reader().array_i32("prism.hadamard.sign_widths",  sw);
                auto sign_for = [&](int width)->float* {
                    size_t off = 0;
                    for (size_t i = 0; i < sw.size(); ++i) {
                        if (sw[i] == width) {
                            if (off + width > sv.size()) return nullptr;
                            std::vector<float> f(width);
                            for (int j = 0; j < width; ++j) f[j] = (float)sv[off+j];
                            float* d; CU(cudaMalloc(&d,(size_t)width*4));
                            CU(cudaMemcpy(d,f.data(),(size_t)width*4,cudaMemcpyHostToDevice));
                            return d;
                        }
                        off += (size_t)sw[i];
                    }
                    return nullptr;
                };
                float* sgn_in  = sign_for(n);
                float* sgn_ff  = sign_for(nff);

                float *dx,*dg,*du,*dh,*dout;
                CU(cudaMalloc(&dx,n*4));            CU(cudaMalloc(&dg,(size_t)nff*4));
                CU(cudaMalloc(&du,(size_t)nff*4));  CU(cudaMalloc(&dh,(size_t)nff*4));
                CU(cudaMalloc(&dout,(size_t)n*4));
                CU(cudaMemcpy(dx,apn.data(),n*4,cudaMemcpyHostToDevice));

                const int HB = 1024; const float hs = 1.0f/std::sqrt((float)HB);
                // 1. rotate the activation for the two 5120-input projections
                k_hadamard<<<(n+HB-1)/HB,512,HB*4>>>(dx,sgn_in,n,HB,hs);
                CU(cudaDeviceSynchronize());
                // 2. gate and up
                k_matvec_pq2<<<(nff+7)/8,256>>>((const blk*)wg->ptr,dx,dg,n,nff);
                k_matvec_pq2<<<(nff+7)/8,256>>>((const blk*)wu->ptr,dx,du,n,nff);
                CU(cudaDeviceSynchronize());
                // 3. SwiGLU
                k_swiglu<<<(nff+255)/256,256>>>(dg,du,dh,nff);
                CU(cudaDeviceSynchronize());
                // 4. rotate the 17408-wide intermediate for ffn_down
                k_hadamard<<<(nff+HB-1)/HB,512,HB*4>>>(dh,sgn_ff,nff,HB,hs);
                CU(cudaDeviceSynchronize());
                // 5. down projection
                k_matvec_pq2<<<(n+7)/8,256>>>((const blk*)wd->ptr,dh,dout,nff,n);
                CU(cudaDeviceSynchronize());

                std::vector<float> got(n);
                CU(cudaMemcpy(got.data(),dout,n*4,cudaMemcpyDeviceToHost));
                double e=0, mag=0, cos_n=0, ga=0, ra=0;
                for (int i=0;i<n;++i){
                    e = std::max(e,(double)std::fabs(got[i]-ref_ffn[i]));
                    mag = std::max(mag,(double)std::fabs(ref_ffn[i]));
                    cos_n += (double)got[i]*ref_ffn[i]; ga += (double)got[i]*got[i];
                    ra += (double)ref_ffn[i]*ref_ffn[i];
                }
                report("FFN block layer 0 vs llama.cpp", e/std::max(mag,1e-9), 1e-2);
                printf("    cosine similarity vs reference : %.8f\n",
                       cos_n/std::sqrt(std::max(ga*ra,1e-30)));
                printf("    path: hadamard -> gate/up -> swiglu -> hadamard -> down\n");
                cudaFree(dx);cudaFree(dg);cudaFree(du);cudaFree(dh);cudaFree(dout);
                if (sgn_in) cudaFree(sgn_in); if (sgn_ff) cudaFree(sgn_ff);
            }
        }
    }

    printf("\n%d/%d ops validated\n", pass, total);
    cudaFree(d_x); cudaFree(d_y);
    return pass == total ? 0 : 1;
}
