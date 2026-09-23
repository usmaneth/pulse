#include "pq2_q8.cuh"
#pragma once
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <set>
#include <chrono>
#include <memory>
#include "sequence_contract.h"

namespace pulse {
inline void check_runtime_cuda(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
#define PULSE_RUNTIME_CUDA(expression) ::pulse::check_runtime_cuda((expression),#expression)

// Each channel owns its three previous convolution inputs.
__global__ void k_conv_step(const float* input, const float* weights,
                            float* history, float* output, int channels) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float* h = history + size_t(c)*3;
    const float* w = weights + size_t(c)*4;
    float x = h[0]*w[0] + h[1]*w[1] + h[2]*w[2] + input[c]*w[3];
    h[0]=h[1]; h[1]=h[2]; h[2]=input[c];
    output[c]=x/(1.0f+expf(-x));
}

__global__ void k_split_query(const float* qg, float* q, int count, int hd) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) q[i]=qg[(i/hd)*2*hd+i%hd];
}

__global__ void k_gate_attention(const float* qg, float* value, int count, int hd) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) value[i]/=1.0f+expf(-qg[(i/hd)*2*hd+hd+i%hd]);
}

__global__ void k_embedding_pq2(const blk* weights, float* out, int n, const int* control=nullptr) {
    if(control) weights+=size_t(control[0])*(n/128);
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const blk& b=weights[i/128];
    int j=i%128;
    out[i]=float(int((b.qs[j/4] >> (2*(j%4)))&3)-1)*__half2float(__ushort_as_half(b.d));
}

__global__ void k_store_kv(const float* key,const float* value,float* keys,float* values,
                           const int* control,int width) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<width) {
        const size_t offset=size_t(control[1])*width+i;
        keys[offset]=key[i]; values[offset]=value[i];
    }
}

struct ExecutionResources {
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    int* host_control=nullptr;
    ExecutionResources() {
        PULSE_RUNTIME_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cudaError_t status=cudaMallocHost(&host_control,2*sizeof(int));
        if(status!=cudaSuccess) {
            cudaStreamDestroy(stream); stream=nullptr;
            check_runtime_cuda(status,"cudaMallocHost control allocation");
        }
    }
    ExecutionResources(const ExecutionResources&)=delete;
    ExecutionResources& operator=(const ExecutionResources&)=delete;
    ~ExecutionResources() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        if(host_control) cudaFreeHost(host_control);
        if(stream) cudaStreamDestroy(stream);
    }
};

struct DeviceBuffers {
    DeviceBuffers()=default;
    DeviceBuffers(const DeviceBuffers&)=delete;
    DeviceBuffers& operator=(const DeviceBuffers&)=delete;
    std::vector<float*> pointers;
    float* make(size_t n) {
        float* p=nullptr; PULSE_RUNTIME_CUDA(cudaMalloc(&p,n*sizeof(float))); pointers.push_back(p); return p;
    }
    ~DeviceBuffers() { for(auto p:pointers) cudaFree(p); }
};

// This runtime supports the existing text-only Bonsai PQ2_0 model.
// Each instance owns one sequence. It does not share state with other instances.
class SequentialDecoder {
    Model& m;
    DeviceBuffers memory;
    ExecutionResources execution;
    int* device_control=nullptr;
    std::vector<float*> layer_snapshots;
    bool graph_enabled_=false;
    bool q8_enabled_=false;
    pulse_q8::Activation* quantized_input=nullptr;
    bool valid_state_=true;
    SequenceCursor cursor;
    bool batch_head_norm_=true;
    float *x,*res,*norm,*rot,*qkv,*z,*conv,*alpha,*beta,*inner,*perm,*gate,*up,*ff;
    float *qfull,*query,*key,*value,*logits,*s5,*s6,*s17;
    struct LayerState { float* recurrent=nullptr; float* history=nullptr; float* k=nullptr; float* v=nullptr; };
    std::vector<LayerState> states;

    const DevTensor* require(const std::string& name, uint32_t type, int width, int rows=1) {
        const auto* t=m.get(name);
        if(!t || t->type!=type || t->ne[0]!=uint64_t(width) || t->ne[1]!=uint64_t(rows)
              || t->ne[2]!=1 || t->ne[3]!=1)
            throw std::runtime_error("Unsupported tensor: "+name);
        return t;
    }
    float* signs(int width) {
        std::vector<int32_t> values,widths;
        m.reader().array_i32("prism.hadamard.sign_values",values);
        m.reader().array_i32("prism.hadamard.sign_widths",widths);
        size_t offset=0;
        for(int w:widths) {
            if(w<=0 || offset+size_t(w)>values.size()) break;
            if(w==width) {
                std::vector<float> host(width);
                for(int i=0;i<width;++i) {
                    if(values[offset+i]!=1 && values[offset+i]!=-1)
                        throw std::runtime_error("Invalid Hadamard sign");
                    host[i]=float(values[offset+i]);
                }
                float* p=memory.make(width); PULSE_RUNTIME_CUDA(cudaMemcpy(p,host.data(),width*4,cudaMemcpyHostToDevice)); return p;
            }
            offset+=size_t(w);
        }
        throw std::runtime_error("Missing Hadamard signs");
    }
    void rotate(float* p,float* signs,int n) {
        k_hadamard<<<n/1024,512,4096,execution.stream>>>(p,signs,n,1024,1.0f/32);
    }
    void prepare_mat(const float* input,int width) {
        if(q8_enabled_) pulse_q8::quantize<<<(width+255)/256,256,0,execution.stream>>>(input,quantized_input,width);
    }
    void mat(const DevTensor* w,const float* input,float* out) {
        if(q8_enabled_) {
            pulse_q8::matvec<<<(int(w->ne[1])+7)/8,256,0,execution.stream>>>(
                static_cast<const pulse_q8::Weight*>(w->ptr),quantized_input,out,int(w->ne[0]),int(w->ne[1]));
            return;
        }
        k_matvec_pq2<<<(int(w->ne[1])+7)/8,256,0,execution.stream>>>(
            static_cast<const blk*>(w->ptr),input,out,int(w->ne[0]),int(w->ne[1]));
    }
    static void save(const std::string& path,const float* device,size_t n) {
        std::vector<float> host(n); PULSE_RUNTIME_CUDA(cudaMemcpy(host.data(),device,n*4,cudaMemcpyDeviceToHost));
        FILE* f=fopen(path.c_str(),"wb");
        if(!f) throw std::runtime_error("Cannot create "+path);
        bool ok=fwrite(host.data(),4,n,f)==n;
        if(fclose(f)!=0 || !ok) throw std::runtime_error("Cannot write "+path);
    }
    void upload_control(int token,int position) {
        execution.host_control[0]=token; execution.host_control[1]=position;
        PULSE_RUNTIME_CUDA(cudaMemcpyAsync(device_control,execution.host_control,2*sizeof(int),
                           cudaMemcpyHostToDevice,execution.stream));
    }
    void enqueue_step() {
        const blk* row=static_cast<const blk*>(m.get("token_embd.weight")->ptr);
        k_embedding_pq2<<<20,256,0,execution.stream>>>(row,x,5120,device_control);
        rotate(x,nullptr,5120); k_apply_signs<<<20,256,0,execution.stream>>>(x,s5,5120);
        for(int il=0;il<m.hp().n_layer;++il) {
            auto& state=states[il];
            auto weight=[&](const char* name) { return m.layer(il,name); };
            PULSE_RUNTIME_CUDA(cudaMemcpyAsync(res,x,5120*4,cudaMemcpyDeviceToDevice,execution.stream));
            k_rmsnorm<<<1,256,0,execution.stream>>>(x,(const float*)weight("attn_norm.weight")->ptr,norm,5120,m.hp().rms_eps);
            PULSE_RUNTIME_CUDA(cudaMemcpyAsync(rot,norm,5120*4,cudaMemcpyDeviceToDevice,execution.stream)); rotate(rot,s5,5120);
            prepare_mat(rot,5120);
            if(state.k) {
                mat(weight("attn_q.weight"),rot,qfull); mat(weight("attn_k.weight"),rot,key);
                mat(weight("attn_v.weight"),rot,value);
                k_split_query<<<24,256,0,execution.stream>>>(qfull,query,6144,256);
                if(batch_head_norm_) {
                    k_rmsnorm<<<24,256,0,execution.stream>>>(query,(const float*)weight("attn_q_norm.weight")->ptr,query,256,m.hp().rms_eps);
                    k_rmsnorm<<<4,256,0,execution.stream>>>(key,(const float*)weight("attn_k_norm.weight")->ptr,key,256,m.hp().rms_eps);
                } else {
                    for(int h=0;h<24;++h)
                        k_rmsnorm<<<1,256,0,execution.stream>>>(query+h*256,(const float*)weight("attn_q_norm.weight")->ptr,query+h*256,256,m.hp().rms_eps);
                    for(int h=0;h<4;++h)
                        k_rmsnorm<<<1,256,0,execution.stream>>>(key+h*256,(const float*)weight("attn_k_norm.weight")->ptr,key+h*256,256,m.hp().rms_eps);
                }
                k_rope<<<24,32,0,execution.stream>>>(query,24,256,64,0,1e7f,device_control);
                k_rope<<<4,32,0,execution.stream>>>(key,4,256,64,0,1e7f,device_control);
                k_store_kv<<<4,256,0,execution.stream>>>(key,value,state.k,state.v,device_control,1024);
                k_attention<<<24,256,512*4,execution.stream>>>(query,state.k,state.v,inner,24,4,256,0,1.0f/16,device_control);
                k_gate_attention<<<24,256,0,execution.stream>>>(qfull,inner,6144,256);
                rotate(inner,s6,6144); prepare_mat(inner,6144); mat(weight("attn_output.weight"),inner,rot);
            } else {
                mat(weight("attn_qkv.weight"),rot,qkv); mat(weight("attn_gate.weight"),rot,z);
                k_matvec_bf16<<<48,256,0,execution.stream>>>((const uint16_t*)weight("ssm_alpha.weight")->ptr,norm,alpha,5120,48);
                k_matvec_bf16<<<48,256,0,execution.stream>>>((const uint16_t*)weight("ssm_beta.weight")->ptr,norm,beta,5120,48);
                k_conv_step<<<40,256,0,execution.stream>>>(qkv,(const float*)weight("ssm_conv1d.weight")->ptr,state.history,conv,10240);
                k_l2_head<<<16,128,0,execution.stream>>>(conv,128); k_l2_head<<<16,128,0,execution.stream>>>(conv+2048,128);
                k_gdn_gates<<<1,64,0,execution.stream>>>(alpha,beta,(const float*)weight("ssm_a")->ptr,
                    (const float*)weight("ssm_dt.bias")->ptr,alpha,beta,48);
                k_gdn_step<<<48,128,256*4,execution.stream>>>(state.recurrent,conv,conv+2048,conv+4096,alpha,beta,inner,48,16,128,128);
                k_gdn_norm_gate<<<48,128,0,execution.stream>>>(inner,(const float*)weight("ssm_norm.weight")->ptr,z,128,m.hp().rms_eps);
                k_perm_tiled_to_grouped<<<24,256,0,execution.stream>>>(inner,perm,128,16,3);
                rotate(perm,s6,6144); prepare_mat(perm,6144); mat(weight("ssm_out.weight"),perm,rot);
            }
            k_add<<<20,256,0,execution.stream>>>(rot,res,5120);
            PULSE_RUNTIME_CUDA(cudaMemcpyAsync(res,rot,5120*4,cudaMemcpyDeviceToDevice,execution.stream));
            k_rmsnorm<<<1,256,0,execution.stream>>>(rot,(const float*)weight("post_attention_norm.weight")->ptr,norm,5120,m.hp().rms_eps);
            rotate(norm,s5,5120); prepare_mat(norm,5120); mat(weight("ffn_gate.weight"),norm,gate); mat(weight("ffn_up.weight"),norm,up);
            k_swiglu<<<68,256,0,execution.stream>>>(gate,up,ff,17408); rotate(ff,s17,17408); prepare_mat(ff,17408);
            mat(weight("ffn_down.weight"),ff,x); k_add<<<20,256,0,execution.stream>>>(x,res,5120);
            if(!layer_snapshots.empty())
                PULSE_RUNTIME_CUDA(cudaMemcpyAsync(layer_snapshots[il],x,5120*4,cudaMemcpyDeviceToDevice,execution.stream));
        }
        k_rmsnorm<<<1,256,0,execution.stream>>>(x,(const float*)m.get("output_norm.weight")->ptr,norm,5120,m.hp().rms_eps);
        rotate(norm,s5,5120); prepare_mat(norm,5120); mat(m.get("output.weight"),norm,logits);
    }
    void capture_graph() {
        // Load every kernel before capture, then restore empty sequence state.
        upload_control(0,0); enqueue_step(); synchronize(); reset();
        cudaError_t status=cudaStreamBeginCapture(execution.stream,cudaStreamCaptureModeThreadLocal);
        if(status!=cudaSuccess) throw std::runtime_error(std::string("Cannot start CUDA graph capture: ")+cudaGetErrorString(status));
        try {
            enqueue_step();
            status=cudaStreamEndCapture(execution.stream,&execution.graph);
            if(status!=cudaSuccess) throw std::runtime_error(std::string("Cannot finish CUDA graph capture: ")+cudaGetErrorString(status));
            status=cudaGraphInstantiate(&execution.executable,execution.graph,nullptr,nullptr,0);
            if(status!=cudaSuccess) throw std::runtime_error(std::string("Cannot instantiate CUDA graph: ")+cudaGetErrorString(status));
        } catch(...) {
            cudaStreamCaptureStatus capture;
            if(cudaStreamIsCapturing(execution.stream,&capture)==cudaSuccess && capture!=cudaStreamCaptureStatusNone) {
                cudaGraph_t abandoned=nullptr;
                cudaStreamEndCapture(execution.stream,&abandoned);
                if(abandoned) cudaGraphDestroy(abandoned);
            }
            throw;
        }
    }
public:
    SequentialDecoder(Model& model,int context,bool retain_layers=false,int graph_mode=-1):m(model),cursor(context) {
        const char* q8_setting=std::getenv("PULSE_Q8_ACTIVATIONS");
        q8_enabled_=q8_setting && std::strcmp(q8_setting,"0")!=0;
        printf("PQ2 activation arithmetic: %s\n",q8_enabled_?"Q8_1 DP4A experimental":"FP32 baseline");
        const char* graph_setting=std::getenv("PULSE_CUDA_GRAPH");
        graph_enabled_=graph_mode<0 ? graph_setting && std::strcmp(graph_setting,"0")!=0 : graph_mode!=0;
        printf("complete CUDA graph: %s\n",graph_enabled_?"enabled":"disabled");
        const char* norm_setting=std::getenv("PULSE_BATCH_HEAD_NORM");
        batch_head_norm_=!norm_setting || std::strcmp(norm_setting,"0")!=0;
        printf("attention head norm: %s launches\n",batch_head_norm_?"batched":"individual");
        const auto& h=m.hp();
        if(m.arch()!="qwen35" || h.n_embd!=5120 || h.n_layer!=64 || h.n_ff!=17408 ||
           h.n_head!=24 || h.n_head_kv!=4 || h.key_len!=256 || h.val_len!=256 ||
           context<1 || context>h.n_ctx_train)
            throw std::runtime_error("Unsupported Bonsai architecture or context capacity");
        const auto& metadata=m.reader().kv();
        auto check=[&](const char* k,const char* expected) {
            auto it=metadata.find(k);
            if(it==metadata.end() || it->second!=expected)
                throw std::runtime_error(std::string("Unsupported metadata: ")+k);
        };
        check("prism.hadamard.block_size","1024");
        check("prism.hadamard.transform","normalized-sylvester-walsh-hadamard");
        check("prism.hadamard.axis","input-last-dimension");
        check("prism.hadamard.sign_mode","explicit");
        check("prism.hadamard.gdn_v_grouped","true");
        check("qwen35.rope.dimension_sections","[11, 11, 10, 0]");
        check("qwen35.ssm.conv_kernel","4");
        check("qwen35.ssm.group_count","16");
        check("qwen35.ssm.inner_size","6144");
        check("qwen35.ssm.state_size","128");
        check("qwen35.ssm.time_step_rank","48");
        if(h.rms_eps!=1e-6f) throw std::runtime_error("Unsupported normalization epsilon");
        for(const char* optional:{"qwen35.rope.scaling.type", "qwen35.attention.scale"})
            if(metadata.count(optional)) throw std::runtime_error(std::string("Unsupported metadata: ")+optional);
        check("qwen35.rope.dimension_count","64");
        // Check numerical metadata independently of its decimal representation.
        auto rb=metadata.find("qwen35.rope.freq_base");
        if(rb==metadata.end() || std::stod(rb->second)!=10000000.0)
            throw std::runtime_error("Unsupported RoPE frequency base");
        std::vector<std::string> folded,inverse;
        if(!m.reader().array_strings("prism.hadamard.weight_names",folded) ||
           !m.reader().array_strings("prism.hadamard.inverse_weight_names",inverse) ||
           inverse!=std::vector<std::string>{"token_embd.weight"})
            throw std::runtime_error("Unsupported Hadamard tensor metadata");
        std::set<std::string> actual(folded.begin(),folded.end()),expected;
        for(const auto& tensor:m.reader().tensors())
            if(tensor.type==T_PQ2_0 && tensor.name!="token_embd.weight") expected.insert(tensor.name);
        if(actual!=expected || folded.size()!=actual.size())
            throw std::runtime_error("Unsupported Hadamard tensor set");
        require("token_embd.weight",T_PQ2_0,5120,h.n_vocab);
        require("output.weight",T_PQ2_0,5120,h.n_vocab);
        require("output_norm.weight",T_F32,5120);
        states.resize(h.n_layer);
        for(int il=0;il<h.n_layer;++il) {
            if(m.is_full_attn(il)!=(il%4==3)) throw std::runtime_error("Unsupported layer order");
            std::string prefix="blk."+std::to_string(il)+".";
            require(prefix+"attn_norm.weight",T_F32,5120);
            require(prefix+"post_attention_norm.weight",T_F32,5120);
            require(prefix+"ffn_gate.weight",T_PQ2_0,5120,17408);
            require(prefix+"ffn_up.weight",T_PQ2_0,5120,17408);
            require(prefix+"ffn_down.weight",T_PQ2_0,17408,5120);
            auto& state=states[il];
            if(m.is_full_attn(il)) {
                require(prefix+"attn_q.weight",T_PQ2_0,5120,12288);
                require(prefix+"attn_k.weight",T_PQ2_0,5120,1024);
                require(prefix+"attn_v.weight",T_PQ2_0,5120,1024);
                require(prefix+"attn_output.weight",T_PQ2_0,6144,5120);
                require(prefix+"attn_q_norm.weight",T_F32,256);
                require(prefix+"attn_k_norm.weight",T_F32,256);
                state.k=memory.make(size_t(context)*1024); state.v=memory.make(size_t(context)*1024);
            } else {
                require(prefix+"attn_qkv.weight",T_PQ2_0,5120,10240);
                require(prefix+"attn_gate.weight",T_PQ2_0,5120,6144);
                require(prefix+"ssm_out.weight",T_PQ2_0,6144,5120);
                require(prefix+"ssm_alpha.weight",T_BF16,5120,48);
                require(prefix+"ssm_beta.weight",T_BF16,5120,48);
                require(prefix+"ssm_conv1d.weight",T_F32,4,10240);
                require(prefix+"ssm_a",T_F32,48); require(prefix+"ssm_dt.bias",T_F32,48);
                require(prefix+"ssm_norm.weight",T_F32,128);
                state.recurrent=memory.make(48*128*128); state.history=memory.make(10240*3);
            }
        }
        s5=signs(5120); s6=signs(6144); s17=signs(17408);
        x=memory.make(5120); res=memory.make(5120); norm=memory.make(5120); rot=memory.make(5120);
        qkv=memory.make(10240); z=memory.make(6144); conv=memory.make(10240);
        alpha=memory.make(48); beta=memory.make(48); inner=memory.make(6144); perm=memory.make(6144);
        gate=memory.make(17408); up=memory.make(17408); ff=memory.make(17408);
        qfull=memory.make(12288); query=memory.make(6144); key=memory.make(1024); value=memory.make(1024);
        logits=memory.make(h.n_vocab);
        if(q8_enabled_) quantized_input=reinterpret_cast<pulse_q8::Activation*>(memory.make((17408/32)*sizeof(pulse_q8::Activation)/sizeof(float)));
        device_control=reinterpret_cast<int*>(memory.make(2));
        if(retain_layers) for(int il=0;il<h.n_layer;++il) layer_snapshots.push_back(memory.make(5120));
        reset();
        if(graph_enabled_) capture_graph();
    }
    void synchronize() { PULSE_RUNTIME_CUDA(cudaStreamSynchronize(execution.stream)); }
    void reset() {
        PULSE_RUNTIME_CUDA(cudaStreamSynchronize(execution.stream));
        for(auto& s:states) {
            if(s.recurrent) PULSE_RUNTIME_CUDA(cudaMemsetAsync(s.recurrent,0,48*128*128*4,execution.stream));
            if(s.history) PULSE_RUNTIME_CUDA(cudaMemsetAsync(s.history,0,10240*3*4,execution.stream));
        }
        // Position bounds prevent reads of stale KV entries after a reset.
        synchronize();
        cursor.reset();
        valid_state_=true;
    }
    int position() const { return cursor.position(); }
    void compare_state(SequentialDecoder& other) {
        if(position()!=other.position() || !valid_state_ || !other.valid_state_)
            throw std::runtime_error("Cannot compare different or invalid sequence positions");
        synchronize(); other.synchronize();
        std::vector<float> first,second;
        auto compare=[&](const float* a,const float* b,size_t count,const std::string& label) {
            first.resize(count); second.resize(count);
            PULSE_RUNTIME_CUDA(cudaMemcpy(first.data(),a,count*4,cudaMemcpyDeviceToHost));
            PULSE_RUNTIME_CUDA(cudaMemcpy(second.data(),b,count*4,cudaMemcpyDeviceToHost));
            if(std::memcmp(first.data(),second.data(),count*4)!=0)
                throw std::runtime_error("Graph state mismatch: "+label+" at position "+std::to_string(position()));
        };
        if(layer_snapshots.size()!=other.layer_snapshots.size()) throw std::runtime_error("Snapshot configurations differ");
        for(int il=0;il<m.hp().n_layer;++il) {
            const auto& a=states[il]; const auto& b=other.states[il];
            const std::string suffix=" layer "+std::to_string(il);
            if(!layer_snapshots.empty()) compare(layer_snapshots[il],other.layer_snapshots[il],5120,"output"+suffix);
            if(a.recurrent) {
                compare(a.recurrent,b.recurrent,48*128*128,"recurrent"+suffix);
                compare(a.history,b.history,10240*3,"convolution"+suffix);
            } else {
                compare(a.k,b.k,size_t(position())*1024,"key"+suffix);
                compare(a.v,b.v,size_t(position())*1024,"value"+suffix);
            }
        }
    }

    std::vector<float> step(int token,const std::string& dump="") {
        if(!valid_state_) throw std::runtime_error("Reset the decoder after a failed step");
        cursor.check(token,m.hp().n_vocab);
        const int position_=cursor.position();
        if(!dump.empty() && layer_snapshots.empty())
            throw std::runtime_error("Decoder lacks diagnostic snapshots");
        valid_state_=false;
        upload_control(token,position_);
        if(graph_enabled_) PULSE_RUNTIME_CUDA(cudaGraphLaunch(execution.executable,execution.stream));
        else enqueue_step();
        PULSE_RUNTIME_CUDA(cudaGetLastError()); PULSE_RUNTIME_CUDA(cudaStreamSynchronize(execution.stream));
        std::vector<float> host(m.hp().n_vocab);
        PULSE_RUNTIME_CUDA(cudaMemcpy(host.data(),logits,host.size()*4,cudaMemcpyDeviceToHost));
        for(float v:host) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite output logit");
        if(!dump.empty()) {
            for(int il=0;il<m.hp().n_layer;++il) {
                const auto& state=states[il];
                const std::string suffix=std::to_string(il)+".f32";
                save(dump+"/layer-"+suffix,layer_snapshots[il],5120);
                if(state.recurrent) {
                    save(dump+"/state-"+suffix,state.recurrent,48*128*128);
                    save(dump+"/conv-"+suffix,state.history,10240*3);
                } else {
                    save(dump+"/key-"+suffix,state.k,size_t(position_+1)*1024);
                    save(dump+"/value-"+suffix,state.v,size_t(position_+1)*1024);
                }
            }
            save(dump+"/logits.f32",logits,host.size());
            FILE* f=fopen((dump+"/token.txt").c_str(),"w");
            if(!f) throw std::runtime_error("Cannot create token record");
            fprintf(f,"%d\n",token); fclose(f);
        }
        cursor.advance();
        valid_state_=true;
        return host;
    }
};

inline int sequential_cli(Model& model,int argc,char** argv) {
    try {
        if(argc<5) throw std::runtime_error("Usage: pulse-engine MODEL --decode TOKEN_IDS NEW_TOKENS [DUMP_DIRECTORY]");
        std::vector<int> tokens; std::stringstream input(argv[3]); std::string field;
        while(std::getline(input,field,',')) { size_t end=0; int t=std::stoi(field,&end);
            if(end!=field.size()) throw std::runtime_error("Invalid token list"); tokens.push_back(t); }
        size_t end=0; int generate=std::stoi(argv[4],&end);
        if(end!=std::strlen(argv[4]) || generate<0 || tokens.empty()) throw std::runtime_error("Invalid token count");
        if(tokens.size()+size_t(generate)>size_t(model.hp().n_ctx_train)) throw std::runtime_error("Context capacity exceeded");
        bool verify_reset=false,verify_interleave=false,verify_graph=false,profile=false,warmup=false;
        size_t prefill_count=tokens.size();
        std::string dump_root;
        for(int i=5;i<argc;++i) {
            const std::string option=argv[i];
            if(option=="--reset-check") verify_reset=true;
            else if(option=="--interleave-check") { verify_interleave=true; verify_reset=true; }
            else if(option=="--graph-check") verify_graph=true;
            else if(option=="--profile") profile=true;
            else if(option=="--warmup") warmup=true;
            else if(option=="--prefill-count" && i+1<argc) {
                size_t consumed=0; const std::string value=argv[++i];
                int count=std::stoi(value,&consumed);
                if(consumed!=value.size() || count<1 || size_t(count)>tokens.size())
                    throw std::runtime_error("Invalid prefill count");
                prefill_count=size_t(count);
            } else if(option.rfind("--",0)==0 || !dump_root.empty())
                throw std::runtime_error("Unknown or repeated decoder option");
            else dump_root=option;
        }
        if(warmup && !profile) throw std::runtime_error("Warmup requires profile mode");
        if(profile && (!dump_root.empty() || verify_reset || verify_graph))
            throw std::runtime_error("Profile mode requires no dumps or reset replay");
        if(!dump_root.empty() && std::filesystem::exists(dump_root) && !std::filesystem::is_empty(dump_root))
            throw std::runtime_error("Native output directory must be empty");
        SequentialDecoder decoder(model,int(tokens.size())+generate,!dump_root.empty() || verify_graph,verify_graph?0:-1);
        std::unique_ptr<SequentialDecoder> graph_peer;
        std::vector<size_t> compared_state_steps;
        if(verify_graph) graph_peer=std::make_unique<SequentialDecoder>(model,int(tokens.size())+generate,true,1);
        printf("Experimental sequential decoder: numerical parity is not established.\n");
        std::vector<float> logits;
        std::vector<int> replay_tokens,executed_tokens;
        std::vector<std::vector<float>> replay_logits;
        auto step=[&](int t) {
            std::string dir;
            if(!dump_root.empty()) { dir=dump_root+"/token-"+std::to_string(decoder.position()); std::filesystem::create_directories(dir); }
            auto result=decoder.step(t,dir);
            executed_tokens.push_back(t);
            if(graph_peer) {
                if(graph_peer->step(t)!=result) throw std::runtime_error("Graph logits differ at position "+std::to_string(decoder.position()));
                const size_t index=executed_tokens.size()-1;
                const size_t total=tokens.size()+size_t(std::max(generate-1,0));
                if(index<3 || (index & (index-1))==0 || ((index+1) & index)==0 || index+1==total) {
                    decoder.compare_state(*graph_peer); compared_state_steps.push_back(index);
                }
            }
            if(verify_reset) { replay_tokens.push_back(t); replay_logits.push_back(result); }
            return result;
        };
        if(warmup) {
            for(int token:tokens) decoder.step(token);
            decoder.reset();
        }
        PULSE_RUNTIME_CUDA(cudaDeviceSynchronize());
        const auto prefill_start=std::chrono::steady_clock::now();
        for(size_t i=0;i<prefill_count;++i) logits=step(tokens[i]);
        PULSE_RUNTIME_CUDA(cudaDeviceSynchronize());
        const auto prefill_end=std::chrono::steady_clock::now();
        std::vector<int> generated,forced_predictions;
        for(size_t i=prefill_count;i<tokens.size();++i) {
            logits=step(tokens[i]); forced_predictions.push_back(greedy_token(logits));
        }
        for(int i=0;i<generate;++i) {
            int t=greedy_token(logits); generated.push_back(t);
            if(i+1<generate) logits=step(t);
        }
        PULSE_RUNTIME_CUDA(cudaDeviceSynchronize());
        const auto decode_end=std::chrono::steady_clock::now();
        for(int t:generated) printf("generated_token %d\n",t);
        for(int t:forced_predictions) printf("forced_prediction %d\n",t);
        if(profile) {
            const double prefill_seconds=std::chrono::duration<double>(prefill_end-prefill_start).count();
            const double decode_seconds=std::chrono::duration<double>(decode_end-prefill_end).count();
            const size_t steps=tokens.size()-prefill_count+size_t(std::max(generate-1,0));
            printf("profile {\"prefill_input_tokens\":%zu,\"prefill_seconds\":%.9f,"
                   "\"decode_steps\":%zu,\"decode_seconds\":%.9f,\"decode_steps_per_second\":%.6f,"
                   "\"emitted_tokens\":%zu,\"warmup_sequence\":%s,\"includes_logit_transfer_and_greedy\":true}\n",
                   prefill_count,prefill_seconds,steps,decode_seconds,steps/decode_seconds,generated.size(),warmup?"true":"false");
        }
        if(!dump_root.empty()) {
            FILE* f=fopen((dump_root+"/inputs.txt").c_str(),"w");
            if(!f) throw std::runtime_error("Cannot create sequence manifest");
            for(size_t i=0;i<executed_tokens.size();++i) fprintf(f,"%s%d",i?",":"",executed_tokens[i]);
            fprintf(f,"\n"); fclose(f);
        }
        if(verify_graph) {
            printf("graph_online_passed logits=%zu state_positions=",executed_tokens.size());
            for(size_t i=0;i<compared_state_steps.size();++i) printf("%s%zu",i?",":"",compared_state_steps[i]);
            printf("\n");
        }
        if(verify_reset) {
            decoder.reset();
            for(size_t i=0;i<replay_tokens.size();++i)
                if(decoder.step(replay_tokens[i])!=replay_logits[i])
                    throw std::runtime_error("Reset replay logits differ");
            printf("reset_replay_passed %zu\n",replay_tokens.size());
        }
        if(verify_interleave) {
            const int capacity=int(executed_tokens.size());
            SequentialDecoder peer(model,capacity);
            std::vector<int> alternate(executed_tokens.rbegin(),executed_tokens.rend());
            std::vector<std::vector<float>> alternate_logits;
            for(int token:alternate) alternate_logits.push_back(peer.step(token));
            peer.reset(); decoder.reset();
            for(size_t i=0;i<executed_tokens.size();++i) {
                // Alternate the call order to detect shared controls or streams.
                if(i%2==0) {
                    if(decoder.step(executed_tokens[i])!=replay_logits[i] || peer.step(alternate[i])!=alternate_logits[i])
                        throw std::runtime_error("Interleaved sequence logits differ");
                } else {
                    if(peer.step(alternate[i])!=alternate_logits[i] || decoder.step(executed_tokens[i])!=replay_logits[i])
                        throw std::runtime_error("Interleaved sequence logits differ");
                }
            }
            SequentialDecoder boundary(model,2);
            auto first=boundary.step(executed_tokens.front());
            auto second=boundary.step(executed_tokens.back());
            bool rejected=false;
            try { boundary.step(executed_tokens.front()); } catch(const std::runtime_error&) { rejected=true; }
            if(!rejected || boundary.position()!=2) throw std::runtime_error("Context boundary check failed");
            boundary.reset();
            rejected=false;
            try { boundary.step(-1); } catch(const std::runtime_error&) { rejected=true; }
            if(!rejected || boundary.position()!=0) throw std::runtime_error("Invalid token changed sequence position");
            if(boundary.step(executed_tokens.front())!=first || boundary.step(executed_tokens.back())!=second)
                throw std::runtime_error("Boundary reset logits differ");
            printf("interleave_and_boundary_passed %zu\n",executed_tokens.size());
        }
        return 0;
    } catch(const std::exception& error) { fprintf(stderr,"%s\n",error.what()); return 1; }
}
}

#undef PULSE_RUNTIME_CUDA
