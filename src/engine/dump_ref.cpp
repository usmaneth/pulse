// Reference dumper: capture llama.cpp's intermediate tensors for one token.
//
// Building an engine without this is guesswork. A forward pass has 64 layers
// and eight distinct ops; if the final logits disagree, the difference tells
// you nothing about WHERE. llama.cpp names its intermediates through cb() -
// "attn_norm", "attn_residual", "ffn_out", "l_out", "result_norm",
// "result_output" - and exposes them via the scheduler eval callback.
//
// This runs one token and writes every matching tensor to disk, so the engine
// can be validated one layer at a time against ground truth.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>

static std::string g_outdir = "/tmp/pulse-ref";
static std::string g_filter;
static int g_count = 0;

static bool eval_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) {
        // ask phase: say whether we want this tensor's data
        if (!t->name[0]) return false;
        if (!g_filter.empty() && std::string(t->name).find(g_filter) == std::string::npos)
            return false;
        return true;
    }
    // data phase: t now holds computed values
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);

    std::string safe(t->name);
    for (auto& c : safe) if (c=='/'||c==' ') c = '_';
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.bin", g_outdir.c_str(), safe.c_str());
    if (FILE* f = fopen(path, "wb")) {
        // header: type, 4 dims, then raw bytes
        int32_t ty = (int32_t)t->type;
        int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        fwrite(&ty, 4, 1, f); fwrite(ne, 8, 4, f);
        fwrite(buf.data(), 1, nb, f);
        fclose(f);
        ++g_count;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [token_id] [outdir] [name_filter]\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    const int   token_id   = argc > 2 ? atoi(argv[2]) : 100;
    if (argc > 3) g_outdir = argv[3];
    if (argc > 4) g_filter = argv[4];

    char mk[1100]; snprintf(mk, sizeof mk, "mkdir -p '%s'", g_outdir.c_str());
    if (system(mk) != 0) { fprintf(stderr, "cannot create %s\n", g_outdir.c_str()); return 1; }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 512;
    cp.n_batch = 512;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = nullptr;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    llama_token tok = (llama_token)token_id;
    llama_batch batch = llama_batch_get_one(&tok, 1);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    printf("dumped %d tensors for token %d to %s\n", g_count, token_id, g_outdir.c_str());
    const float* logits = llama_get_logits(ctx);
    if (logits) {
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        int best = 0; for (int i = 1; i < nv; ++i) if (logits[i] > logits[best]) best = i;
        printf("reference argmax token %d, logit %.6f  (vocab %d)\n", best, logits[best], nv);
        char path[1100]; snprintf(path, sizeof path, "%s/logits.bin", g_outdir.c_str());
        if (FILE* f = fopen(path, "wb")) { fwrite(logits, 4, nv, f); fclose(f); }
    }
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
