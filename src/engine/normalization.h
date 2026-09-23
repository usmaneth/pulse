#pragma once
#ifdef __CUDACC__
#define PULSE_HD __host__ __device__
#else
#define PULSE_HD
#endif
namespace pulse {
PULSE_HD inline float l2_squared_norm_floor(float sum, float epsilon) {
    const float minimum=epsilon*epsilon;
    return sum>minimum ? sum : minimum;
}
}
#undef PULSE_HD
