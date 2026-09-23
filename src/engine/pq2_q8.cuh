#pragma once
// The quantizer and chunk dot follow llama.cpp commit
// 999b0a9a6f2fb3e3b60bb3d6bbac4b6b0c0f7fd7:
// ggml/src/ggml-cuda/quantize.cu and vecdotq.cuh.
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
#include <cuda_fp16.h>
#include <cstdint>
namespace pulse_q8 {
struct alignas(4) Activation { half2 ds; int8_t qs[32]; };
struct alignas(2) Weight { uint16_t d; uint8_t qs[32]; };
static_assert(sizeof(Activation)==36 && sizeof(Weight)==34,"Invalid quantized block size");
__global__ void quantize(const float* x,Activation* y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return; // Supported widths contain complete warps.
    float v=x[i], maximum=fabsf(v), sum=v;
    #pragma unroll
    for(int o=16;o;o>>=1) {
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,o));
        sum+=__shfl_xor_sync(0xffffffff,sum,o);
    }
    float d=maximum/127.0f;
    y[i/32].qs[i%32]=maximum==0.0f ? 0 : int8_t(roundf(v/d));
    if(i%32==0) y[i/32].ds=__floats2half2_rn(d,sum);
}
__device__ __forceinline__ float chunk_dot(const Weight* w,const Activation* a,int chunk) {
    const int16_t* codes=reinterpret_cast<const int16_t*>(w->qs)+chunk*4;
    int sum=0;
    #pragma unroll
    for(int j=0;j<4;++j) {
        int q=codes[j];
        int u=reinterpret_cast<const int*>(a->qs)[j*2];
        int v=reinterpret_cast<const int*>(a->qs)[j*2+1];
        int even=__byte_perm(0x020100FF,0x020100FF,q);
        int odd=__byte_perm(0x020100FF,0x020100FF,q>>2);
        int low=__byte_perm(even,odd,0x5140);
        int high=__byte_perm(even,odd,0x7362);
        sum=__dp4a(u,low,sum);
        sum=__dp4a(v,high,sum);
    }
    return __half2float(__ushort_as_half(w->d))*__low2float(a->ds)*sum;
}
__global__ __launch_bounds__(256) void matvec(const Weight* weights,const Activation* input,float* out,int width,int rows) {
    int row=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    if(row>=rows) return;
    const Weight* w=weights+size_t(row)*(width/128);
    float sum=0;
    for(int c=lane;c<width/32;c+=32) sum+=chunk_dot(w+c/4,input+c,c%4);
    #pragma unroll
    for(int o=16;o;o>>=1) sum+=__shfl_down_sync(0xffffffff,sum,o);
    if(lane==0) out[row]=sum;
}
}
