// Evaluate the actual ggml CPU RoPE operator without a model or GPU.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if(argc!=2 || (std::strcmp(argv[1],"original") && std::strcmp(argv[1],"yarn"))) return 2;
    const bool yarn=std::strcmp(argv[1],"yarn")==0;
    constexpr int dim=128, count=6;
    const int32_t positions[count]={0,511,4095,8191,32767,65535};
    ggml_init_params params={16*1024*1024,nullptr,false};
    ggml_context* ctx=ggml_init(params);
    if(!ctx) return 3;
    ggml_tensor* x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,dim,1,count);
    ggml_tensor* pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,count);
    std::memcpy(pos->data,positions,sizeof(positions));
    // Values vary by dimension. Every position receives the same vector.
    for(int row=0;row<count;++row)
        for(int i=0;i<dim;++i) static_cast<float*>(x->data)[row*dim+i]=float((i*37)%101-50)/32.0f;
    // llama-context.cpp cancels the kernel's extra magnitude factor once.
    // For ordinary YaRN, the operator receives attn_factor=1 and applies mscale.
    ggml_tensor* y=ggml_rope_ext(ctx,x,pos,nullptr,dim,GGML_ROPE_TYPE_NEOX,
            yarn?8192:262144,10000000.0f,yarn?1.0f/32:1.0f,
            yarn?1.0f:0.0f,1.0f,32.0f,1.0f);
    ggml_cgraph* graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,y);
    if(ggml_graph_compute_with_ctx(ctx,graph,2)!=GGML_STATUS_SUCCESS) return 4;
    bool ok=std::fwrite(y->data,sizeof(float),dim*count,stdout)==dim*count;
    ggml_free(ctx);
    return ok?0:5;
}
