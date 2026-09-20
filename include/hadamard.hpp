#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace pulse {

constexpr int HADAMARD_BLOCK_SIZE = 1024;
constexpr float HADAMARD_SCALE = 0.03125f; // 1.0f / sqrtf(1024.0f) = 1.0f / 32.0f

#ifdef __CUDACC__
__device__ inline void fwht_1024_shared(float* s_data, int tid) {
    #pragma unroll
    for (int len = 1; len < 1024; len <<= 1) {
        int idx = 2 * tid - (tid & (len - 1));
        if (idx + len < 1024) {
            float u = s_data[idx];
            float v = s_data[idx + len];
            s_data[idx] = u + v;
            s_data[idx + len] = u - v;
        }
        __syncthreads();
    }
}

__global__ void hadamard_inverse_embedding_kernel(
    const float* __restrict__ raw_embeddings,
    float* __restrict__ primal_embeddings,
    int total_elements
) {
    __shared__ float s_block[HADAMARD_BLOCK_SIZE];
    int block_offset = blockIdx.x * HADAMARD_BLOCK_SIZE;
    int tid = threadIdx.x;

    if (block_offset + tid < total_elements) {
        s_block[tid] = raw_embeddings[block_offset + tid];
    } else {
        s_block[tid] = 0.0f;
    }
    __syncthreads();

    fwht_1024_shared(s_block, tid);

    if (block_offset + tid < total_elements) {
        primal_embeddings[block_offset + tid] = s_block[tid] * HADAMARD_SCALE;
    }
}

__global__ void hadamard_forward_head_kernel(
    const float* __restrict__ post_norm_activations,
    float* __restrict__ rotated_activations,
    int total_elements
) {
    __shared__ float s_block[HADAMARD_BLOCK_SIZE];
    int block_offset = blockIdx.x * HADAMARD_BLOCK_SIZE;
    int tid = threadIdx.x;

    if (block_offset + tid < total_elements) {
        s_block[tid] = post_norm_activations[block_offset + tid];
    } else {
        s_block[tid] = 0.0f;
    }
    __syncthreads();

    fwht_1024_shared(s_block, tid);

    if (block_offset + tid < total_elements) {
        rotated_activations[block_offset + tid] = s_block[tid] * HADAMARD_SCALE;
    }
}
#endif

} // namespace pulse
