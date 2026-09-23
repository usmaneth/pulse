#pragma once
#include <cstdlib>
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
        int map_flags=MAP_PRIVATE;
#ifdef MAP_POPULATE
        const char* prefetch=std::getenv("PULSE_PREFETCH_MODEL");
        if(prefetch && std::strcmp(prefetch,"0")!=0) map_flags|=MAP_POPULATE;
#endif
        base_ = (const uint8_t*)mmap(nullptr, size_, PROT_READ, map_flags, fd_, 0);
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
            cur_key_ = key;
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
        // Tensor data starts after the directory, padded up to general.alignment
        // (32 by default). Offsets in the directory are relative to that point.
        uint64_t align = 32;
        auto it = kv_.find("general.alignment");
        if (it != kv_.end()) align = strtoull(it->second.c_str(), nullptr, 10);
        if (align == 0) align = 32;
        size_t pos = (size_t)(p_ - base_);
        data_start_ = (pos + align - 1) / align * align;
        return true;
    }

    const uint8_t* tensor_data(const TensorInfo& t) const {
        return base_ + data_start_ + t.offset;
    }
    size_t data_start() const { return data_start_; }

    uint32_t version() const { return version_; }
    uint64_t n_tensors() const { return n_tensors_; }
    uint64_t n_kv() const { return n_kv_; }
    size_t file_size() const { return size_; }
    const uint8_t* mapped_data() const { return base_; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string,std::string>& kv() const { return kv_; }

    // Raw access to array-valued metadata. prism.hadamard.sign_values is 28672
    // entries and must be used, not printed.
    struct ArrRef { uint32_t type; uint64_t n; const uint8_t* data; };
    const ArrRef* array(const std::string& k) const {
        auto it = arrays_.find(k); return it == arrays_.end() ? nullptr : &it->second;
    }
    // Reads an integer array of any width into int32.
    bool array_i32(const std::string& k, std::vector<int32_t>& out) const {
        const ArrRef* a = array(k); if (!a) return false;
        out.resize(a->n);
        for (uint64_t i = 0; i < a->n; ++i) {
            switch (a->type) {
                case GT_INT8:   out[i] = (int8_t)a->data[i]; break;
                case GT_UINT8:  out[i] = a->data[i]; break;
                case GT_INT16:  { int16_t v; memcpy(&v,a->data+i*2,2); out[i]=v; } break;
                case GT_UINT16: { uint16_t v; memcpy(&v,a->data+i*2,2); out[i]=v; } break;
                case GT_INT32:  { int32_t v; memcpy(&v,a->data+i*4,4); out[i]=v; } break;
                case GT_UINT32: { uint32_t v; memcpy(&v,a->data+i*4,4); out[i]=(int32_t)v; } break;
                default: return false;
            }
        }
        return true;
    }

    bool array_strings(const std::string& key, std::vector<std::string>& out) const {
        const ArrRef* a=array(key);
        if(!a || a->type!=GT_STRING) return false;
        const uint8_t* cursor=a->data;
        const uint8_t* end=base_+size_;
        out.clear();
        for(uint64_t i=0;i<a->n;++i) {
            if(cursor>end || size_t(end-cursor)<8) return false;
            uint64_t length; memcpy(&length,cursor,8); cursor+=8;
            if(length>uint64_t(end-cursor)) return false;
            out.emplace_back(reinterpret_cast<const char*>(cursor),size_t(length));
            cursor+=length;
        }
        return true;
    }

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
                cur_arr_type_ = et; cur_arr_n_ = n;
                cur_arr_begin_ = p_;
                // Keep short numeric arrays verbatim - rope.dimension_sections
                // and friends are configuration the engine must honour exactly.
                std::string joined; bool keep = (n <= 8 && et != GT_STRING);
                std::string first;
                for (uint64_t i = 0; i < n; ++i) {
                    std::string v = read_value(et);
                    if (i == 0) first = v;
                    if (keep) { if (i) joined += ", "; joined += v; }
                }
                if (cur_key_.size()) {
                    arrays_[cur_key_] = ArrRef{ cur_arr_type_, cur_arr_n_, cur_arr_begin_ };
                }
                if (keep) return "[" + joined + "]";
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
    size_t data_start_ = 0;
    std::string cur_key_;
    uint32_t cur_arr_type_ = 0; uint64_t cur_arr_n_ = 0; const uint8_t* cur_arr_begin_ = nullptr;
    std::vector<TensorInfo> tensors_;
    std::map<std::string,std::string> kv_;
    std::map<std::string,ArrRef> arrays_;
};

} // namespace pulse
