#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "types.hpp"

namespace pulse {

// NVFP4 E2M1 Lookup Table: maps 4-bit nibbles (0x0 to 0xF) to IEEE-754 FP32 values
__constant__ float C_NVFP4_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/**
 * Fast inline NVFP4 (E2M1) dequantization with FP8 block micro-scaling.
 * Each 8-bit byte packs two 4-bit values (low nibble, high nibble).
 */
__device__ inline void dequant_nvfp4_pair(
    uint8_t packed_byte,
    float scale,
    float& val_low,
    float& val_high
) {
    uint8_t low_idx = packed_byte & 0x0F;
    uint8_t high_idx = (packed_byte >> 4) & 0x0F;
    val_low = C_NVFP4_LUT[low_idx] * scale;
    val_high = C_NVFP4_LUT[high_idx] * scale;
}

/**
 * Blackwell SM121 Fused NVFP4 Matrix-Vector Kernel
 * Dimension: K = 5120 (Qwen Hidden Dim), N = out_features
 * Weights: NVFP4 (0.5 byte/element) + FP8 block scales (1 scale per 16 elements = 0.0625 byte/element)
 * Total weight memory footprint: 0.5625 bytes / weight (vs 2.0 bytes for FP16 -> 3.55x compression!)
 */
__global__ void gemv_blackwell_nvfp4_dim5120_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ w_nvfp4,
    const float* __restrict__ block_scales, // 1 scale per 16 weights
    float* __restrict__ y,
    int out_features
) {
    int row = blockIdx.x;
    if (row >= out_features) return;

    float sum = 0.0f;
    int tid = threadIdx.x;
    int warp_size = 32;

    size_t row_weight_offset = static_cast<size_t>(row) * (QWEN_HIDDEN_DIM / 2);
    size_t row_scale_offset = static_cast<size_t>(row) * (QWEN_HIDDEN_DIM / 16);

    // Each thread processes 8 weights (4 packed bytes) per iteration
    #pragma unroll 4
    for (int col = tid * 2; col < QWEN_HIDDEN_DIM; col += blockDim.x * 2) {
        uint8_t packed = w_nvfp4[row_weight_offset + (col / 2)];
        float scale = block_scales[row_scale_offset + (col / 16)];

        float w0, w1;
        dequant_nvfp4_pair(packed, scale, w0, w1);

        sum += x[col] * w0;
        if (col + 1 < QWEN_HIDDEN_DIM) {
            sum += x[col + 1] * w1;
        }
    }

    // Warp-level reduction
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    __shared__ float s_warp_sums[32];
    int lane = tid % warp_size;
    int wid = tid / warp_size;
    if (lane == 0) {
        s_warp_sums[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        float bsum = (lane < (blockDim.x / warp_size)) ? s_warp_sums[lane] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            bsum += __shfl_down_sync(0xFFFFFFFF, bsum, offset);
        }
        if (lane == 0) {
            y[row] = bsum;
        }
    }
}

} // namespace pulse
