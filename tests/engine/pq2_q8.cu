#include <gguf.h>
#include "quantize.cuh"
#include "vecdotq.cuh"
#include "pq2_q8.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#define CHECK(call) do { auto s=(call); if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); } while(0)
__global__ void reference_chunks(const block_pq2_0* w,const block_q8_1* a,float* out,int chunks,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<chunks*rows) out[i]=vec_dot_pq2_0_q8_1(w,a+((i%chunks)/4)*4,i/4,i%4);
}
__global__ void native_chunks(const pulse_q8::Weight* w,const pulse_q8::Activation* a,float* out,int chunks,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<chunks*rows) out[i]=pulse_q8::chunk_dot(w+i/4,a+i%chunks,i%4);
}
int main() try {
    constexpr int n=17408, chunks=n/32, rows=9;
    static_assert(sizeof(block_q8_1)==sizeof(pulse_q8::Activation),"Q8 layout mismatch");
    static_assert(sizeof(block_pq2_0)==sizeof(pulse_q8::Weight),"PQ2 layout mismatch");
    float *x,*a,*b,*result; void *qa,*qb; pulse_q8::Weight* weights;
    CHECK(cudaMalloc(&x,n*4)); CHECK(cudaMalloc(&a,chunks*rows*4)); CHECK(cudaMalloc(&b,chunks*rows*4)); CHECK(cudaMalloc(&result,rows*4));
    CHECK(cudaMalloc(&qa,chunks*sizeof(block_q8_1))); CHECK(cudaMalloc(&qb,chunks*sizeof(block_q8_1)));
    CHECK(cudaMalloc(&weights,(n/128)*rows*sizeof(pulse_q8::Weight)));
    std::mt19937 rng(42); std::uniform_real_distribution<float> random(-3,3);
    std::vector<pulse_q8::Weight> hw((n/128)*rows);
    for(size_t i=0;i<hw.size();++i) {
        hw[i].d=__half_as_ushort(__float2half((i%17+1)*.03125f));
        for(int j=0;j<32;++j) hw[i].qs[j]=uint8_t(i*32+j);
    }
    CHECK(cudaMemcpy(weights,hw.data(),hw.size()*sizeof(hw[0]),cudaMemcpyHostToDevice));
    for(int mode=0;mode<5;++mode) {
        std::vector<float> h(n);
        for(int i=0;i<n;++i) {
            if(mode==0) h[i]=random(rng);
            if(mode==1) h[i]=0;
            if(mode==2) h[i]=(i%32==0)?127.0f:float((i%15)-7)+.5f;
            if(mode==3) h[i]=random(rng)*1e-8f;
            if(mode==4) h[i]=((i&1)?1.0f:-1.0f)*32768.0f;
        }
        CHECK(cudaMemcpy(x,h.data(),n*4,cudaMemcpyHostToDevice));
        quantize_row_q8_1_cuda(x,nullptr,qa,GGML_TYPE_PQ2_0,n,n,n,n,n,1,1,1,nullptr);
        pulse_q8::quantize<<<(n+255)/256,256>>>(x,static_cast<pulse_q8::Activation*>(qb),n);
        CHECK(cudaDeviceSynchronize());
        std::vector<unsigned char> ah(chunks*sizeof(block_q8_1)),bh(ah.size());
        CHECK(cudaMemcpy(ah.data(),qa,ah.size(),cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(bh.data(),qb,bh.size(),cudaMemcpyDeviceToHost));
        if(ah!=bh) throw std::runtime_error("Activation bytes differ from upstream in case "+std::to_string(mode));
        reference_chunks<<<(chunks*rows+255)/256,256>>>(reinterpret_cast<block_pq2_0*>(weights),static_cast<block_q8_1*>(qa),a,chunks,rows);
        native_chunks<<<(chunks*rows+255)/256,256>>>(weights,static_cast<pulse_q8::Activation*>(qb),b,chunks,rows);
        pulse_q8::matvec<<<(rows+7)/8,256>>>(weights,static_cast<pulse_q8::Activation*>(qb),result,n,rows);
        CHECK(cudaDeviceSynchronize());
        std::vector<float> ha(chunks*rows),hb(chunks*rows);
        CHECK(cudaMemcpy(ha.data(),a,chunks*rows*4,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hb.data(),b,chunks*rows*4,cudaMemcpyDeviceToHost));
        if(std::memcmp(ha.data(),hb.data(),chunks*rows*4)) throw std::runtime_error("PQ2 chunk dots differ from upstream");
        std::vector<float> actual(rows); CHECK(cudaMemcpy(actual.data(),result,rows*4,cudaMemcpyDeviceToHost));
        double worst_error=0;
        for(int row=0;row<rows;++row) {
            double expected=0,absolute_sum=0;
            for(int c=0;c<chunks;++c) { float v=ha[row*chunks+c]; expected+=v; absolute_sum+=std::abs(v); }
            double error=std::abs(actual[row]-expected); worst_error=std::max(worst_error,error);
            if(!std::isfinite(actual[row]) || error>1e-6*std::max(1.0,absolute_sum))
                throw std::runtime_error("Matvec reduction exceeds its fixed error bound");
        }
        printf("case %d: quantizer and chunk dots exact, nine-row worst reduction error %.9g\n",mode,worst_error);
    }
    cudaFree(x);cudaFree(a);cudaFree(b);cudaFree(result);cudaFree(qa);cudaFree(qb);cudaFree(weights);
    puts("Q8 activation and PQ2 DP4A unit gate passed");return 0;
} catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what());return 1; }
