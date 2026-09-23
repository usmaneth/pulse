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
#include "ggml-backend.h"
#include "reference_layout.h"
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <chrono>
#include "sequence_contract.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>

static std::string g_outdir = "/tmp/pulse-ref";
static std::string g_filter;
static int g_count = 0;
static bool g_failed = false;
static bool g_sequential = false;

static bool eval_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) {
        // ask phase: say whether we want this tensor's data
        if (!t->name[0]) return false;
        if(g_sequential && g_filter.empty()) {
            const std::string name(t->name);
            if(name.rfind("l_out-",0)!=0 && name.rfind("new_state-",0)!=0 &&
               name!="model.input_embed" && name!="result_norm" && name!="result_output") return false;
        }
        if (!g_filter.empty() && std::string(t->name).find(g_filter) == std::string::npos)
            return false;
        return true;
    }
    // data phase: t now holds computed values
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);

    // Scalar tensors use contiguous logical order. Quantized tensors must be contiguous.
    if (ggml_blck_size(t->type) == 1) {
        buf = pulse::pack_reference(buf.data(), buf.size(), t->ne, t->nb,
                                    ggml_type_size(t->type));
    } else if (!ggml_is_contiguous(t)) {
        fprintf(stderr, "cannot dump noncontiguous quantized tensor %s\n", t->name);
        g_failed = true;
        return false;
    }
    std::string safe(t->name);
    for (auto& c : safe) if (c=='/'||c==' ') c = '_';
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.bin", g_outdir.c_str(), safe.c_str());
    if (FILE* f = fopen(path, "wb")) {
        // Header: type, four dimensions, then contiguous logical values.
        int32_t ty = (int32_t)t->type;
        int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        bool ok = fwrite(&ty, 4, 1, f)==1 && fwrite(ne, 8, 4, f)==4;
        ok = ok && fwrite(buf.data(), 1, buf.size(), f)==buf.size();
        if (fclose(f)!=0 || !ok) g_failed=true;
        ++g_count;
    } else {
        g_failed = true;
        fprintf(stderr, "cannot create reference file %s\n", path);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [token_id] [outdir] [name_filter] [n_tokens]\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    const bool profile = argc > 2 && std::string(argv[2]) == "--profile";
    const bool production=profile && argc>5 && std::string(argv[5])=="--production";
    const bool sequential = argc > 2 && (std::string(argv[2]) == "--sequence" || profile);
    size_t prefill_count=0;
    g_sequential=sequential;
    int greedy_steps=0;
    if (sequential && argc < 5) {
        fprintf(stderr, "usage: %s MODEL --sequence TOKEN_IDS OUTDIR [FILTER]\n", argv[0]);
        return 2;
    }
    const int token_id = argc > 2 && !sequential ? atoi(argv[2]) : 100;
    std::vector<llama_token> sequence;
    if (sequential) {
        try {
            std::stringstream input(argv[3]); std::string field;
            while (std::getline(input, field, ',')) {
                size_t end=0; int t=std::stoi(field,&end);
                if (end!=field.size() || t<0) throw std::runtime_error("invalid token");
                sequence.push_back(t);
            }
            if(sequence.empty()) throw std::runtime_error("empty sequence");
        } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 2; }
        if(profile) {
            try {
                size_t end=0; int count=std::stoi(argv[4],&end);
                if(end!=strlen(argv[4]) || count<1 || size_t(count)>sequence.size()) return 2;
                prefill_count=size_t(count);
            } catch(...) { return 2; }
        } else { g_outdir=argv[4]; if(argc>5) g_filter=argv[5]; }
        if(argc>6) {
            try {
                size_t end=0; greedy_steps=std::stoi(argv[6],&end);
                if(end!=strlen(argv[6]) || greedy_steps<0 || greedy_steps>4096) return 2;
            } catch(...) { return 2; }
        }
    } else {
        if (argc > 3) g_outdir = argv[3];
        if (argc > 4) g_filter = argv[4];
    }

    if(!profile && sequential && std::filesystem::exists(g_outdir) && !std::filesystem::is_empty(g_outdir)) {
        fprintf(stderr,"reference output directory must be empty\n"); return 2;
    }
    if(!profile) std::filesystem::create_directories(g_outdir);

    const auto load_start=std::chrono::steady_clock::now();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = sequential ? std::max<size_t>(512,sequence.size()+greedy_steps) : 512;
    cp.n_batch = 512;
    cp.cb_eval = profile ? nullptr : eval_cb;
    if(profile) {
        cp.n_batch=1; cp.n_ubatch=1;
        cp.type_k=production ? GGML_TYPE_F16 : GGML_TYPE_F32;
        cp.type_v=production ? GGML_TYPE_F16 : GGML_TYPE_F32;
        cp.flash_attn_type=production ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    cp.cb_eval_user_data = nullptr;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    if(profile) {
        const auto load_end=std::chrono::steady_clock::now();
        const int vocabulary=llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::vector<float> values(vocabulary);
        std::vector<int> predictions;
        const auto start=std::chrono::steady_clock::now();
        auto prefill_end=start;
        for(size_t i=0;i<sequence.size();++i) {
            if(sequence[i]<0 || sequence[i]>=vocabulary) return 2;
            llama_batch batch=llama_batch_get_one(&sequence[i],1);
            if(llama_decode(ctx,batch)!=0) return 1;
            llama_synchronize(ctx);
            const float* logits=llama_get_logits(ctx);
            std::copy(logits,logits+vocabulary,values.begin());
            // Match the native diagnostic's finite-logit check on every step.
            for(float value:values) if(!std::isfinite(value)) return 1;
            if(i+1==prefill_count) prefill_end=std::chrono::steady_clock::now();
            if(i>=prefill_count) predictions.push_back(pulse::greedy_token(values));
        }
        llama_synchronize(ctx);
        const auto finish=std::chrono::steady_clock::now();
        const double prefill_seconds=std::chrono::duration<double>(prefill_end-start).count();
        const double decode_seconds=std::chrono::duration<double>(finish-prefill_end).count();
        const size_t steps=sequence.size()-prefill_count;
        printf("profile {\"load_and_context_seconds\":%.9f,\"prefill_input_tokens\":%zu,"
               "\"prefill_seconds\":%.9f,\"decode_steps\":%zu,\"decode_seconds\":%.9f,"
               "\"decode_steps_per_second\":%.6f,\"kv_type\":\"%s\",\"batch_size\":1,"
               "\"flash_attention\":%s,\"context_capacity\":%u}\n",
               std::chrono::duration<double>(load_end-load_start).count(),prefill_count,prefill_seconds,
               steps,decode_seconds,steps/decode_seconds,production?"f16":"f32",production?"true":"false",llama_n_ctx(ctx));
        for(int token:predictions) printf("forced_prediction %d\n",token);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    }

    if (sequential) {
        const std::string root=g_outdir;
        const int vocabulary=llama_vocab_n_tokens(llama_model_get_vocab(model));
        const size_t total_steps=sequence.size()+size_t(greedy_steps);
        for(size_t i=0;i<sequence.size();++i) {
            if(sequence[i]>=vocabulary) { fprintf(stderr,"token exceeds vocabulary\n"); return 2; }
            g_outdir=root+"/token-"+std::to_string(i);
            std::filesystem::create_directories(g_outdir);
            llama_batch step=llama_batch_get_one(&sequence[i],1);
            if(llama_decode(ctx,step)!=0 || g_failed) { fprintf(stderr,"reference decode failed\n"); return 1; }
            std::string path=g_outdir+"/logits.f32";
            FILE* f=fopen(path.c_str(),"wb");
            if(!f) return 1;
            bool ok=fwrite(llama_get_logits(ctx),4,vocabulary,f)==size_t(vocabulary);
            if(fclose(f)!=0 || !ok) return 1;
            path=g_outdir+"/token.txt";
            f=fopen(path.c_str(),"w"); if(!f) return 1;
            fprintf(f,"%d\n",sequence[i]); fclose(f);
            if(i+1==sequence.size() && sequence.size()<total_steps) {
                const float* logits=llama_get_logits(ctx);
                sequence.push_back(int(std::max_element(logits,logits+vocabulary)-logits));
            }
        }
        FILE* inputs=fopen((root+"/inputs.txt").c_str(),"w");
        if(!inputs) return 1;
        for(size_t i=0;i<sequence.size();++i) fprintf(inputs,"%s%d",i?",":"",sequence[i]);
        fprintf(inputs,"\n"); fclose(inputs);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        printf("dumped %zu sequential tokens to %s\n",sequence.size(),root.c_str());
        return 0;
    }

    // A multi-token prompt exercises RoPE at nonzero positions, real attention
    // over >1 key, and the GDN recurrence carrying state between tokens - all of
    // which cancel at a single token.
    const int n_tok = (argc > 5) ? atoi(argv[5]) : 1;
    std::vector<llama_token> toks;
    for (int i = 0; i < n_tok; ++i) toks.push_back((llama_token)(token_id + i));
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t)toks.size());
    if (llama_decode(ctx, batch) != 0 || g_failed) { fprintf(stderr, "decode failed\n"); return 1; }
    printf("decoded %d tokens starting at %d\n", n_tok, token_id);

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
