#pragma once
// Pulse engine: the model, resident on the GPU.
//
// Loads every tensor of a GGUF into device memory, indexed by name, and exposes
// the architecture the layer loop needs. This is the piece that makes Pulse an
// engine rather than a proxy: it owns the weights.
#include "gguf.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <map>
#include <string>
#include <vector>
#include <cstdio>

#define CU(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

namespace pulse {

// ggml type codes this engine understands
enum : uint32_t { T_F32 = 0, T_BF16 = 30, T_PQ2_0 = 142 };

struct DevTensor {
    void*    ptr   = nullptr;     // device pointer
    uint32_t type  = 0;
    uint64_t ne[4] = {1,1,1,1};   // ggml order: ne[0] is contiguous
    size_t   bytes = 0;
    uint64_t elems() const { return ne[0]*ne[1]*ne[2]*ne[3]; }
};

struct Hparams {
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0;
    int n_ff = 0, n_ctx_train = 0, key_len = 0, val_len = 0;
    int full_attn_interval = 0;
    float rms_eps = 1e-6f;
    int   n_vocab = 0;
};

class Model {
public:
    bool load(const char* path, bool verbose = true) {
        if (!r_.open(path) || !r_.parse()) return false;

        auto geti = [&](const char* k, int dflt)->int {
            auto it = r_.kv().find(k);
            return it == r_.kv().end() ? dflt : atoi(it->second.c_str());
        };
        auto getf = [&](const char* k, float dflt)->float {
            auto it = r_.kv().find(k);
            return it == r_.kv().end() ? dflt : (float)atof(it->second.c_str());
        };
        arch_ = r_.kv().count("general.architecture") ? r_.kv().at("general.architecture") : "?";
        const std::string p = arch_ + ".";
        hp_.n_layer   = geti((p+"block_count").c_str(), 0);
        hp_.n_embd    = geti((p+"embedding_length").c_str(), 0);
        hp_.n_head    = geti((p+"attention.head_count").c_str(), 0);
        hp_.n_head_kv = geti((p+"attention.head_count_kv").c_str(), 0);
        hp_.n_ff      = geti((p+"feed_forward_length").c_str(), 0);
        hp_.n_ctx_train = geti((p+"context_length").c_str(), 0);
        hp_.key_len   = geti((p+"attention.key_length").c_str(), 0);
        hp_.val_len   = geti((p+"attention.value_length").c_str(), 0);
        hp_.full_attn_interval = geti((p+"full_attention_interval").c_str(), 0);
        hp_.rms_eps   = getf((p+"attention.layer_norm_rms_epsilon").c_str(), 1e-6f);

        size_t total = 0, skipped = 0;
        for (const auto& ti : r_.tensors()) {
            const size_t nb = tensor_bytes(ti);
            if (!nb) { ++skipped; continue; }
            DevTensor d;
            d.type = ti.type; d.bytes = nb;
            for (size_t i = 0; i < ti.dims.size() && i < 4; ++i) d.ne[i] = ti.dims[i];
            CU(cudaMalloc(&d.ptr, nb));
            CU(cudaMemcpy(d.ptr, r_.tensor_data(ti), nb, cudaMemcpyHostToDevice));
            tensors_[ti.name] = d;
            total += nb;
            if (ti.name == "output.weight") hp_.n_vocab = (int)d.ne[1];
        }
        bytes_on_gpu_ = total;
        if (verbose) {
            printf("model     : %s (%s)\n", path, arch_.c_str());
            printf("layers    : %d   embd %d   ff %d   heads %d/%d kv\n",
                   hp_.n_layer, hp_.n_embd, hp_.n_ff, hp_.n_head, hp_.n_head_kv);
            printf("vocab     : %d   train ctx %d   full-attn every %d\n",
                   hp_.n_vocab, hp_.n_ctx_train, hp_.full_attn_interval);
            printf("uploaded  : %zu tensors, %.2f GB on GPU (%zu skipped)\n\n",
                   tensors_.size(), total/1e9, skipped);
        }
        return true;
    }

    const DevTensor* get(const std::string& n) const {
        auto it = tensors_.find(n);
        return it == tensors_.end() ? nullptr : &it->second;
    }
    const DevTensor* layer(int il, const char* suffix) const {
        char buf[128]; snprintf(buf, sizeof buf, "blk.%d.%s", il, suffix);
        return get(buf);
    }
    // A layer is full-attention when it has attn_q; the others carry ssm_*.
    bool is_full_attn(int il) const { return layer(il, "attn_q.weight") != nullptr; }

    const Reader& reader() const { return r_; }
    const Hparams& hp() const { return hp_; }
    size_t bytes_on_gpu() const { return bytes_on_gpu_; }
    const std::string& arch() const { return arch_; }
    size_t n_tensors() const { return tensors_.size(); }
    // host-side view, for building CPU references
    const uint8_t* host_data(const std::string& n) const {
        for (const auto& ti : r_.tensors()) if (ti.name == n) return r_.tensor_data(ti);
        return nullptr;
    }

    static size_t tensor_bytes(const TensorInfo& ti) {
        uint64_t n = 1; for (auto d : ti.dims) n *= d;
        switch (ti.type) {
            case T_F32:   return n * 4;
            case T_BF16:  return n * 2;
            case T_PQ2_0: return (n / 128) * 34;      // fp16 scale + 32 bytes
            default:      return 0;                    // unhandled -> skip
        }
    }

private:
    Reader r_;
    std::map<std::string, DevTensor> tensors_;
    Hparams hp_;
    std::string arch_;
    size_t bytes_on_gpu_ = 0;
};

} // namespace pulse
