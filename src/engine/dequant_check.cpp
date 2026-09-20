// Pulse engine, step 2: read actual WEIGHTS, and prove we read them correctly.
//
// Step 1 read the model's metadata. This reads the quantised bytes of a real
// tensor, dequantises them with Pulse's own PQ2_0 implementation, and compares
// the result against ggml's `dequantize_row_pq2_0` - the same function
// llama.cpp uses - element by element.
//
// The comparison is the point. A dequant that is merely plausible is worthless;
// it has to be bit-identical to the reference or every layer built on it is
// silently wrong.
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <vector>

extern "C" void dequantize_row_pq2_0(const void* x, float* y, int64_t k);

namespace {

// PQ2_0: 128 weights per block, one fp16 scale, 2 bits per weight.
// Codes are 00=-1, 01=0, 10=+1, 11=+2, scaled by d.
constexpr int QK_PQ2_0 = 128;
struct block_pq2_0 { uint16_t d; uint8_t qs[QK_PQ2_0 / 4]; };
static_assert(sizeof(block_pq2_0) == 2 + QK_PQ2_0 / 4, "block size");

// Own fp16 -> fp32, so this does not borrow the reference's conversion either.
float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;                       // +-0
        else {                                            // subnormal
            int e = -1; uint32_t m = mant;
            do { m <<= 1; ++e; } while ((m & 0x400) == 0);
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);         // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

void pulse_dequantize_pq2_0(const block_pq2_0* x, float* y, int64_t k) {
    const int64_t nb = k / QK_PQ2_0;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK_PQ2_0; ++j) {
            const uint8_t q = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            y[i * QK_PQ2_0 + j] = ((int)q - 1) * d;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [tensor_name]\n", argv[0]); return 2; }
    pulse::Reader r;
    if (!r.open(argv[1]) || !r.parse()) return 1;

    const char* want = argc > 2 ? argv[2] : "blk.0.attn_qkv.weight";
    const pulse::TensorInfo* t = nullptr;
    for (const auto& ti : r.tensors()) if (ti.name == want) { t = &ti; break; }
    if (!t) { fprintf(stderr, "tensor '%s' not found\n", want); return 1; }
    if (t->type != 142) { fprintf(stderr, "tensor is type %u, not PQ2_0 (142)\n", t->type); return 1; }

    uint64_t n = 1; for (auto d : t->dims) n *= d;
    if (n % QK_PQ2_0) { fprintf(stderr, "element count %llu not a multiple of 128\n",
                                (unsigned long long)n); return 1; }

    const auto* blocks = (const block_pq2_0*)r.tensor_data(*t);
    const uint64_t nb  = n / QK_PQ2_0;

    printf("tensor          : %s\n", t->name.c_str());
    printf("dims            : ");
    for (size_t i = 0; i < t->dims.size(); ++i)
        printf("%llu%s", (unsigned long long)t->dims[i], i+1<t->dims.size()?" x ":"");
    printf("\nelements        : %llu\n", (unsigned long long)n);
    printf("blocks (128/ea) : %llu\n", (unsigned long long)nb);
    printf("quantised bytes : %llu (%.3f bits/weight)\n",
           (unsigned long long)(nb * sizeof(block_pq2_0)),
           nb * sizeof(block_pq2_0) * 8.0 / n);
    printf("data offset     : %llu (file data section starts at %zu)\n\n",
           (unsigned long long)t->offset, r.data_start());

    std::vector<float> mine(n), ref(n);
    pulse_dequantize_pq2_0(blocks, mine.data(), (int64_t)n);
    dequantize_row_pq2_0((const void*)blocks, ref.data(), (int64_t)n);

    uint64_t mismatch = 0; double maxdiff = 0; uint64_t first_bad = 0;
    for (uint64_t i = 0; i < n; ++i) {
        if (memcmp(&mine[i], &ref[i], sizeof(float)) != 0) {
            if (!mismatch) first_bad = i;
            ++mismatch;
            maxdiff = std::fmax(maxdiff, std::fabs((double)mine[i] - (double)ref[i]));
        }
    }
    printf("compared against ggml dequantize_row_pq2_0:\n");
    printf("  elements compared : %llu\n", (unsigned long long)n);
    printf("  bitwise mismatches: %llu\n", (unsigned long long)mismatch);
    if (mismatch) {
        printf("  first mismatch at : %llu (pulse=%g ggml=%g)\n",
               (unsigned long long)first_bad, mine[first_bad], ref[first_bad]);
        printf("  max abs diff      : %g\n", maxdiff);
        printf("  RESULT: FAIL\n");
        return 1;
    }
    // codec sanity: every value must be one of {-d, 0, d, 2d} for its block
    uint64_t off_codec = 0;
    for (uint64_t b = 0; b < nb; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        for (int j = 0; j < QK_PQ2_0; ++j) {
            const float v = mine[b*QK_PQ2_0 + j];
            if (v != -d && v != 0.0f && v != d && v != 2*d) ++off_codec;
        }
    }
    printf("  values outside {-d,0,d,2d}: %llu\n", (unsigned long long)off_codec);
    printf("  RESULT: bit-identical to the reference\n");
    return 0;
}
