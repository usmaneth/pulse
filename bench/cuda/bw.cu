// Memory bandwidth characterisation for GB10.
//
// Question: the model decode sustains ~184 GB/s against a 273 GB/s spec peak.
// Is that the hardware's real ceiling, or is our kernel just not extracting it?
//
// Sweeps memory-level parallelism (independent loads in flight per thread),
// occupancy, and access pattern. Reports the best achieved for each.
#include <cstdio>
#include <cuda_runtime.h>
#include <utility>
#include <cstdlib>

#define CHK(x) do { cudaError_t err_=(x); if(err_!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(err_),__LINE__);exit(1);} } while(0)

// ILP = independent loads in flight per thread iteration.
template <int ILP>
__global__ void read_kernel(const uint4* __restrict__ p, uint4* __restrict__ out, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (; i + (ILP-1)*stride < n; i += ILP*stride) {
        uint4 v[ILP];
#pragma unroll
        for (int j = 0; j < ILP; ++j) v[j] = p[i + j*stride];   // all issued before any use
#pragma unroll
        for (int j = 0; j < ILP; ++j) { acc.x ^= v[j].x; acc.y ^= v[j].y; acc.z ^= v[j].z; acc.w ^= v[j].w; }
    }
    if (acc.x == 0xFFFFFFFFu) out[blockIdx.x] = acc;   // never true; keeps the loads live
}

__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

static double time_ms(void (*launch)(const uint4*, uint4*, size_t, int, int),
                      const uint4* a, uint4* b, size_t n, int blocks, int threads, int reps) {
    cudaEvent_t evs,eve; CHK(cudaEventCreate(&evs)); CHK(cudaEventCreate(&eve));
    launch(a,b,n,blocks,threads);  CHK(cudaDeviceSynchronize());   // warm
    CHK(cudaEventRecord(evs));
    for (int r=0;r<reps;++r) launch(a,b,n,blocks,threads);
    CHK(cudaEventRecord(eve)); CHK(cudaEventSynchronize(eve));
    float ms=0; CHK(cudaEventElapsedTime(&ms,evs,eve));
    CHK(cudaEventDestroy(evs)); CHK(cudaEventDestroy(eve));
    return ms/reps;
}

template <int ILP> void launch_read(const uint4* a, uint4* b, size_t n, int bl, int th) {
    read_kernel<ILP><<<bl,th>>>(a,b,n);
}
void launch_copy(const uint4* a, uint4* b, size_t n, int bl, int th) { copy_kernel<<<bl,th>>>(a,b,n); }

int main(int argc, char** argv) {
    size_t GB = (argc>1)? atof(argv[1])*1e9 : 4e9;
    cudaDeviceProp prop; CHK(cudaGetDeviceProperties(&prop,0));
    printf("%s, %d SMs, %d-bit bus\n", prop.name, prop.multiProcessorCount, prop.memoryBusWidth);
    printf("spec peak for 256-bit LPDDR5X @8533MT/s = 273 GB/s\n\n");

    size_t n = GB/sizeof(uint4);
    uint4 *a=nullptr,*b=nullptr;
    CHK(cudaMalloc(&a, n*sizeof(uint4)));
    CHK(cudaMalloc(&b, 65536*sizeof(uint4)));
    CHK(cudaMemset(a, 1, n*sizeof(uint4)));
    double bytes = (double)n*sizeof(uint4);
    int SM = prop.multiProcessorCount;

    printf("READ-ONLY  (%.2f GB buffer)\n", bytes/1e9);
    printf("%-6s %-8s %-10s %10s %10s\n","ILP","threads","blocks","ms","GB/s");
    double best=0; const char* bestcfg="";
    static char cfgbuf[128];
    for (int th : {128,256,512,1024}) {
      for (int bpsm : {1,2,4,8,16}) {
        int bl = SM*bpsm;
        double ms1 = time_ms(launch_read<1>,a,b,n,bl,th,3);
        double ms4 = time_ms(launch_read<4>,a,b,n,bl,th,3);
        double ms8 = time_ms(launch_read<8>,a,b,n,bl,th,3);
        for (auto pr : {std::pair<int,double>{1,ms1},{4,ms4},{8,ms8}}) {
            double gbs = bytes/(pr.second/1000.0)/1e9;
            if (gbs>best){best=gbs; snprintf(cfgbuf,sizeof cfgbuf,"ILP=%d threads=%d blocks=%d(%dx SM)",pr.first,th,bl,bpsm); bestcfg=cfgbuf;}
            printf("%-6d %-8d %-10d %10.2f %10.1f\n", pr.first, th, bl, pr.second, gbs);
        }
      }
    }
    printf("\nBEST READ: %.1f GB/s  (%.0f%% of 273 spec peak)   [%s]\n", best, 100*best/273.0, bestcfg);

    double msc = time_ms(launch_copy,a,b,65536,SM*8,256,3);
    (void)msc;
    printf("\nnote: copy needs a second full-size buffer; read-only is the relevant\n"
           "      pattern for weight-streaming decode anyway.\n");
    CHK(cudaFree(a)); CHK(cudaFree(b));
    return 0;
}
