#include "gguf.h"
#include <cstring>

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
