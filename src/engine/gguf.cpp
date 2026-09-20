// Pulse engine, step 1: read the model file.
//
// Everything in Pulse to date has been a proxy in front of llama.cpp. The one
// component named "engine" (src/cuda/speculative_engine.cu) runs CUDA kernels
// over synthetic weights and contains no file I/O at all. So nothing in this
// repository has ever opened the GGUF.
//
// This does. It is a self-contained GGUF v3 reader: header, metadata key/value
// pairs, and the tensor directory with names, shapes, quantisation types and
// data offsets. No llama.cpp, no ggml. It either reads the real file correctly
// or it fails, which makes it verifiable against `llama-server`'s own load log.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pulse {

enum : uint32_t { GGUF_MAGIC = 0x46554747 };  // "GGUF" little-endian

enum gguf_type : uint32_t {
    GT_UINT8=0, GT_INT8=1, GT_UINT16=2, GT_INT16=3, GT_UINT32=4, GT_INT32=5,
    GT_FLOAT32=6, GT_BOOL=7, GT_STRING=8, GT_ARRAY=9, GT_UINT64=10,
    GT_INT64=11, GT_FLOAT64=12,
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t offset;
};

class Reader {
public:
    bool open(const char* path) {
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) { perror("open"); return false; }
        struct stat st{};
        if (fstat(fd_, &st) != 0) { perror("fstat"); return false; }
        size_ = (size_t)st.st_size;
        base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (base_ == MAP_FAILED) { perror("mmap"); return false; }
        p_ = base_;
        return true;
    }
    ~Reader() {
        if (base_ && base_ != MAP_FAILED) munmap((void*)base_, size_);
        if (fd_ >= 0) ::close(fd_);
    }

    bool parse() {
        uint32_t magic = u32();
        if (magic != GGUF_MAGIC) {
            fprintf(stderr, "not a GGUF file (magic 0x%08x)\n", magic);
            return false;
        }
        version_   = u32();
        n_tensors_ = u64();
        n_kv_      = u64();

        for (uint64_t i = 0; i < n_kv_; ++i) {
            std::string key = str();
            uint32_t t = u32();
            std::string val = read_value(t);
            if (!val.empty()) kv_[key] = val;
        }

        tensors_.reserve(n_tensors_);
        for (uint64_t i = 0; i < n_tensors_; ++i) {
            TensorInfo ti;
            ti.name = str();
            uint32_t nd = u32();
            for (uint32_t d = 0; d < nd; ++d) ti.dims.push_back(u64());
            ti.type   = u32();
            ti.offset = u64();
            tensors_.push_back(std::move(ti));
        }
        return true;
    }

    uint32_t version() const { return version_; }
    uint64_t n_tensors() const { return n_tensors_; }
    uint64_t n_kv() const { return n_kv_; }
    size_t file_size() const { return size_; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string,std::string>& kv() const { return kv_; }

private:
    uint8_t  u8()  { uint8_t v; memcpy(&v,p_,1); p_+=1; return v; }
    uint16_t u16() { uint16_t v; memcpy(&v,p_,2); p_+=2; return v; }
    uint32_t u32() { uint32_t v; memcpy(&v,p_,4); p_+=4; return v; }
    uint64_t u64() { uint64_t v; memcpy(&v,p_,8); p_+=8; return v; }
    float    f32() { float v;    memcpy(&v,p_,4); p_+=4; return v; }
    double   f64() { double v;   memcpy(&v,p_,8); p_+=8; return v; }
    std::string str() {
        uint64_t n = u64();
        std::string s((const char*)p_, n);
        p_ += n;
        return s;
    }
    // Returns a printable rendering for scalars; consumes arrays and returns
    // a summary so the directory stays parseable.
    std::string read_value(uint32_t t) {
        char buf[64];
        switch (t) {
            case GT_UINT8:  snprintf(buf,sizeof buf,"%u",  u8());  return buf;
            case GT_INT8:   snprintf(buf,sizeof buf,"%d",  (int8_t)u8()); return buf;
            case GT_UINT16: snprintf(buf,sizeof buf,"%u",  u16()); return buf;
            case GT_INT16:  snprintf(buf,sizeof buf,"%d",  (int16_t)u16()); return buf;
            case GT_UINT32: snprintf(buf,sizeof buf,"%u",  u32()); return buf;
            case GT_INT32:  snprintf(buf,sizeof buf,"%d",  (int32_t)u32()); return buf;
            case GT_FLOAT32:snprintf(buf,sizeof buf,"%g",  (double)f32()); return buf;
            case GT_FLOAT64:snprintf(buf,sizeof buf,"%g",  f64()); return buf;
            case GT_BOOL:   return u8() ? "true" : "false";
            case GT_UINT64: snprintf(buf,sizeof buf,"%llu",(unsigned long long)u64()); return buf;
            case GT_INT64:  snprintf(buf,sizeof buf,"%lld",(long long)u64()); return buf;
            case GT_STRING: return str();
            case GT_ARRAY: {
                uint32_t et = u32();
                uint64_t n  = u64();
                std::string first;
                for (uint64_t i = 0; i < n; ++i) {
                    std::string v = read_value(et);
                    if (i == 0) first = v;
                }
                snprintf(buf,sizeof buf,"[%llu items", (unsigned long long)n);
                std::string s = buf;
                if (!first.empty() && first.size() < 24) s += ", first=" + first;
                return s + "]";
            }
            default:
                fprintf(stderr, "unknown gguf type %u\n", t);
                return "";
        }
    }

    int fd_ = -1;
    const uint8_t* base_ = nullptr;
    const uint8_t* p_ = nullptr;
    size_t size_ = 0;
    uint32_t version_ = 0;
    uint64_t n_tensors_ = 0, n_kv_ = 0;
    std::vector<TensorInfo> tensors_;
    std::map<std::string,std::string> kv_;
};

} // namespace pulse

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [--tensors N]\n", argv[0]); return 2; }
    int show = 0;
    for (int i = 2; i < argc-1; ++i) if (!strcmp(argv[i], "--tensors")) show = atoi(argv[i+1]);

    pulse::Reader r;
    if (!r.open(argv[1])) return 1;
    if (!r.parse()) return 1;

    printf("file            : %s\n", argv[1]);
    printf("size            : %.2f GB\n", r.file_size()/1073741824.0);
    printf("gguf version    : %u\n", r.version());
    printf("metadata keys   : %llu\n", (unsigned long long)r.n_kv());
    printf("tensors         : %llu\n\n", (unsigned long long)r.n_tensors());

    // Print every non-tokenizer key: the architecture prefix is not known in
    // advance, so guessing key names hides the model's actual shape.
    printf("metadata:\n");
    for (const auto& [k, v] : r.kv()) {
        if (k.rfind("tokenizer.", 0) == 0 && k != "tokenizer.ggml.model") continue;
        printf("  %-46s = %s\n", k.c_str(), v.c_str());
    }

    // quantisation histogram, which is what a loader must dispatch on
    std::map<uint32_t,size_t> by_type;
    uint64_t total_elems = 0;
    for (const auto& t : r.tensors()) {
        by_type[t.type]++;
        uint64_t n = 1; for (auto d : t.dims) n *= d;
        total_elems += n;
    }
    printf("\ntensor types (ggml_type -> count):\n");
    for (auto& [ty,c] : by_type) printf("  type %-4u : %zu tensors\n", ty, c);
    printf("\ntotal elements  : %.2f B\n", total_elems/1e9);

    if (show > 0) {
        printf("\nfirst %d tensors:\n", show);
        int i = 0;
        for (const auto& t : r.tensors()) {
            if (i++ >= show) break;
            printf("  %-40s type=%-4u off=%-12llu dims=[", t.name.c_str(), t.type,
                   (unsigned long long)t.offset);
            for (size_t d = 0; d < t.dims.size(); ++d)
                printf("%llu%s", (unsigned long long)t.dims[d], d+1<t.dims.size()?", ":"");
            printf("]\n");
        }
    }
    return 0;
}
