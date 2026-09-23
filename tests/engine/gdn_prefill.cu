#include "gdn_prefill.cuh"
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <ggml-alloc.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
using namespace pulse_gdn_prefill;
static void compare(const std::vector<float>& expected,const std::vector<float>& actual,const char* label) {
    double largest=0,difference=0,norm=0;
    for(size_t i=0;i<expected.size();++i) {
        if(!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw std::runtime_error("Nonfinite GDN result");
        double error=double(actual[i])-expected[i]; largest=std::max(largest,std::abs(error));
        difference+=error*error; norm+=double(expected[i])*expected[i];
    }
    double relative=std::sqrt(difference/std::max(norm,1e-24));
    printf("%s max_abs=%.9g relative_l2=%.9g\n",label,largest,relative);
    if(largest>2e-5 || relative>2e-6) throw std::runtime_error("GDN reference tolerance exceeded");
}
static void exact(const std::vector<float>& a,const std::vector<float>& b,const char* label) {
    if(a.size()!=b.size() || std::memcmp(a.data(),b.data(),a.size()*4)) throw std::runtime_error(label);
}
static void fixture(ggml_backend_t backend,int tokens,bool padded,bool nonzero) {
    ggml_context* ctx=ggml_init({16*1024*1024,nullptr,true});
    if(!ctx) throw std::runtime_error("Cannot create ggml fixture context");
    Strides st{128,(padded?32:16)*128,(padded?256:128),(padded?96:48)*128};
    auto* qb=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,st.q_token/128,tokens);
    auto* kb=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,st.q_token/128,tokens);
    auto* vb=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,st.v_token/128,tokens);
    auto view=[&](ggml_tensor* base,int heads,int hs,int ts) { return ggml_view_4d(ctx,base,128,heads,tokens,1,hs*4,ts*4,size_t(ts)*tokens*4,0); };
    auto* q=view(qb,16,st.q_head,st.q_token); auto* k=view(kb,16,st.q_head,st.q_token);
    auto* v=view(vb,48,st.v_head,st.v_token);
    auto* g=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,48,tokens,1);
    auto* b=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,1,48,tokens,1);
    auto* state=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,128,128,48,1);
    auto* result=ggml_gated_delta_net(ctx,q,k,v,g,b,state,1);
    auto* graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,result);
    ggml_backend_buffer_t allocation=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!allocation) throw std::runtime_error("Cannot allocate ggml fixture buffers");
    std::mt19937 rng(123+tokens+int(nonzero)*1000); std::uniform_real_distribution<float> random(-1,1);
    std::vector<float> hq(size_t(st.q_token)*tokens,99),hk(hq.size(),99),hv(size_t(st.v_token)*tokens,99);
    for(int t=0;t<tokens;++t) {
        for(int h=0;h<16;++h) {
            double nq=0,nk=0; size_t offset=size_t(t)*st.q_token+h*st.q_head;
            for(int d=0;d<128;++d) { float a=random(rng),c=random(rng);hq[offset+d]=a;hk[offset+d]=c;nq+=a*a;nk+=c*c; }
            for(int d=0;d<128;++d) { hq[offset+d]/=std::sqrt(nq);hk[offset+d]/=std::sqrt(nk); }
        }
        for(int h=0;h<48;++h) for(int d=0;d<128;++d) hv[size_t(t)*st.v_token+h*st.v_head+d]=random(rng)+h*.001f;
    }
    std::vector<float> hg(tokens*48),hb(hg.size()),hs(state_elements);
    for(size_t i=0;i<hg.size();++i) { hg[i]=-(.001f+float(i%31)*.0125f);hb[i]=.05f+.9f*(i%17)/16; }
    for(float& x:hs) x=nonzero ? random(rng)*.01f : 0;
    auto upload=[](ggml_tensor* x,const std::vector<float>& values) { ggml_backend_tensor_set(x,values.data(),0,values.size()*4); };
    upload(qb,hq);upload(kb,hk);upload(vb,hv);upload(g,hg);upload(b,hb);upload(state,hs);
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("Upstream GDN graph failed");
    ggml_backend_synchronize(backend);
    size_t output_elements=size_t(tokens)*48*128, total=output_elements+state_elements;
    std::vector<float> reference(total),actual(total),replay(total),split(total);
    ggml_backend_tensor_get(result,reference.data(),0,total*4);
    float *native,*temporary; CHECK(cudaMalloc(&native,total*4));CHECK(cudaMalloc(&temporary,state_elements*4));
    auto native_run=[&](const float* initial,float* output,float* final_state,int count,int start) {
        launch(static_cast<float*>(q->data)+size_t(start)*st.q_token,static_cast<float*>(k->data)+size_t(start)*st.q_token,
               static_cast<float*>(v->data)+size_t(start)*st.v_token,static_cast<float*>(g->data)+start*48,
               static_cast<float*>(b->data)+start*48,initial,output,final_state,count,st);
        CHECK(cudaDeviceSynchronize());
    };
    native_run(static_cast<float*>(state->data),native,native+output_elements,tokens,0);
    CHECK(cudaMemcpy(actual.data(),native,total*4,cudaMemcpyDeviceToHost));
    printf("fixture tokens=%d padded=%d nonzero=%d\n",tokens,padded,nonzero);
    compare(std::vector<float>(reference.begin(),reference.begin()+output_elements),std::vector<float>(actual.begin(),actual.begin()+output_elements),"outputs");
    compare(std::vector<float>(reference.begin()+output_elements,reference.end()),std::vector<float>(actual.begin()+output_elements,actual.end()),"state");
    // A separate initial state must not change a later replay of the first sequence.
    CHECK(cudaMemset(temporary,0,state_elements*4));
    native_run(temporary,native,native+output_elements,tokens,0);
    native_run(static_cast<float*>(state->data),native,native+output_elements,tokens,0);
    CHECK(cudaMemcpy(replay.data(),native,total*4,cudaMemcpyDeviceToHost));exact(actual,replay,"Reset or independent-state replay mismatch");
    if(tokens>1) {
        int first=tokens==64 ? 17 : 1;
        native_run(static_cast<float*>(state->data),native,temporary,first,0);
        native_run(temporary,native+size_t(first)*48*128,native+output_elements,tokens-first,first);
        CHECK(cudaMemcpy(split.data(),native,total*4,cudaMemcpyDeviceToHost));exact(actual,split,"Split GDN tile mismatch");
    }
    CHECK(cudaFree(native));CHECK(cudaFree(temporary));ggml_backend_buffer_free(allocation);ggml_free(ctx);
}
int main(int argc,char** argv) try {
    for(int n: {0,65}) { bool caught=false;try {validate(n,{128,2048,128,6144});}catch(const std::invalid_argument&) {caught=true;}if(!caught) throw std::runtime_error("Invalid token count accepted"); }
    bool caught=false;try {validate(1,{127,2048,128,6144});}catch(const std::invalid_argument&) {caught=true;}if(!caught) throw std::runtime_error("Invalid stride accepted");
    const int largest=std::numeric_limits<int>::max();
    for(Strides strides: {Strides{largest,largest,128,6144},Strides{128,2048,largest,largest}}) {
        bool rejected=false;
        try {validate(1,strides);} catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected) throw std::runtime_error("Overflowing stride extent accepted");
    }
    validate(64,{128,4096,256,12288});
    if(argc==2 && std::strcmp(argv[1],"--cpu-only")==0) {
        puts("GDN CPU stride and token contracts passed; no GPU backend initialized");return 0;
    }
    ggml_backend_t backend=ggml_backend_cuda_init(0);if(!backend) throw std::runtime_error("Cannot initialize CUDA backend");
    for(int tokens: {1,2,17,64}) for(bool padded: {false,true}) for(bool nonzero: {false,true}) fixture(backend,tokens,padded,nonzero);
    ggml_backend_free(backend);puts("All 16 GDN prefill fixtures passed");return 0;
} catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what());return 1; }
