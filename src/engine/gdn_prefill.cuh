#pragma once
// Adapted from llama.cpp 999b0a9a6f2fb3e3b60bb3d6bbac4b6b0c0f7fd7,
// ggml/src/ggml-cuda/gated_delta_net.cu. This is a recurrent tile, not a parallel chunk solve.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace pulse_gdn_prefill {
constexpr int dimension=128, key_heads=16, value_heads=48;
constexpr size_t state_elements=size_t(value_heads)*dimension*dimension;
struct Strides { int q_head,q_token,v_head,v_token; };
inline void validate(int tokens,Strides s) {
    if(tokens<1 || tokens>64) throw std::invalid_argument("GDN tile requires 1 through 64 tokens");
    if(s.q_head<dimension || int64_t(s.q_token)<int64_t(key_heads)*s.q_head ||
       s.v_head<dimension || int64_t(s.v_token)<int64_t(value_heads)*s.v_head)
        throw std::invalid_argument("Invalid GDN input strides");
}
__device__ __forceinline__ float sum_warp(float x) {
    #pragma unroll
    for(int offset=16;offset;offset>>=1) x+=__shfl_xor_sync(0xffffffff,x,offset);
    return x;
}
__global__ __launch_bounds__(128,2) void recurrence(
    const float* __restrict__ q,const float* __restrict__ k,const float* __restrict__ v,
    const float* __restrict__ log_decay,const float* __restrict__ beta,
    const float* __restrict__ initial,float* __restrict__ output,float* __restrict__ final_state,
    int tokens,Strides strides) {
    int h=blockIdx.x, lane=threadIdx.x;
    int col=(blockIdx.y*blockDim.y+threadIdx.y)*4;
    const float* src=initial+size_t(h)*128*128;
    float* dst=final_state+size_t(h)*128*128;
    float shard[4][4];
    #pragma unroll
    for(int c=0;c<4;++c) {
        #pragma unroll
        for(int r=0;r<4;++r) shard[c][r]=src[(col+c)*128+r*32+lane];
    }
    for(int t=0;t<tokens;++t) {
        const float* qt=q+size_t(t)*strides.q_token+(h%16)*strides.q_head;
        const float* kt=k+size_t(t)*strides.q_token+(h%16)*strides.q_head;
        const float* vt=v+size_t(t)*strides.v_token+h*strides.v_head;
        float qr[4],kr[4];
        #pragma unroll
        for(int r=0;r<4;++r) { qr[r]=qt[r*32+lane]; kr[r]=kt[r*32+lane]; }
        float decay=expf(log_decay[t*48+h]), b=beta[t*48+h];
        #pragma unroll
        for(int c=0;c<4;++c) {
            float kv=0;
            #pragma unroll
            for(int r=0;r<4;++r) kv+=shard[c][r]*kr[r];
            float delta=(vt[col+c]-decay*sum_warp(kv))*b;
            float attention=0;
            #pragma unroll
            for(int r=0;r<4;++r) {
                shard[c][r]=decay*shard[c][r]+kr[r]*delta;
                attention+=shard[c][r]*qr[r];
            }
            float value=sum_warp(attention);
            if(lane==0) output[(size_t(t)*48+h)*128+col+c]=value*(1.0f/sqrtf(128.0f));
        }
    }
    #pragma unroll
    for(int c=0;c<4;++c) {
        #pragma unroll
        for(int r=0;r<4;++r) dst[(col+c)*128+r*32+lane]=shard[c][r];
    }
}
inline void launch(const float* q,const float* k,const float* v,const float* g,const float* b,
                   const float* initial,float* output,float* final_state,int tokens,Strides strides,cudaStream_t stream=nullptr) {
    validate(tokens,strides);
    if(!q || !k || !v || !g || !b || !initial || !output || !final_state || initial==final_state)
        throw std::invalid_argument("GDN tile requires separate valid input and output buffers");
    recurrence<<<dim3(48,8),dim3(32,4),0,stream>>>(q,k,v,g,b,initial,output,final_state,tokens,strides);
}
}
