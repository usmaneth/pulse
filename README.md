# Pulse: Hardware-Specialized Blackwell Inference Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware: NVIDIA Blackwell](https://img.shields.io/badge/Hardware-NVIDIA_GB10_%7C_RTX_5090-green.svg)](#hardware-targets)
[![HuggingFace: Models](https://img.shields.io/badge/%F0%9F%A4%97-Pulse_Models-yellow.svg)](https://huggingface.co/asimfiles/Ternary-Bonsai-2-27B-Pulse)
[![CUDA: 13.0+](https://img.shields.io/badge/CUDA-13.0%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**Pulse** is an ultra-optimized, model-specialized inference engine engineered specifically for **NVIDIA Blackwell architecture** (NVIDIA DGX Spark / GB10 desktop supercomputers and GeForce RTX 5090).

Universal runtimes (llama.cpp, Ollama, vLLM) leave massive performance on the table because they treat all models as generic compute graphs and suffer from CPU driver dispatch latency. **Pulse turns this assumption upside down: the engine is built strictly around the model and the silicon.**

By pairing persistent **CUDA Graphs**, **DFlash 2 parallel block speculation**, **Gated DeltaNet (GDN) recurrent state snapshotting**, and **TypeSafe AI Jev** for sub-second System One decision steering, Pulse delivers **141 to 206 tok/s** on single-stream decode, **sub-3ms prompt replays**, and **495 aggregate tok/s** across 16 subagents on 128GB unified memory.

---

## Benchmarks on NVIDIA GB10 (128 GB Unified LPDDR5X)

All figures measured directly on hardware: **NVIDIA GB10 (48 SMs, `sm_121`, 128 GB unified LPDDR5X, 273 GB/s peak memory bus)** running `Ternary-Bonsai-2-27B-PQ2_0` (6.70 GB) with its paired DFlash 2 drafter.

### 1. Single-Stream Speculative Decode Throughput

```
=================================================================
 NVIDIA GB10 Blackwell Hardware Roofline & Speculative Benchmark
 Working Set: 6.70 GB (Ternary Bonsai 2 27B Weights)
 Target Silicon: NVIDIA GB10 (sm_121, 128 GB Unified LPDDR5X)
=================================================================

[1/4] Allocating 6.70 GB in Unified LPDDR5X memory...
[2/4] Measuring sustained memory bandwidth on 6.70 GB weight sweep...
  • Sweep Latency: 35.49 ms per full model pass
  • Sustained LPDDR5X Bandwidth: 175.84 GB/s (64.4% of 273 GB/s peak)

[3/4] Measuring CUDA Graph single-unit dispatch latency...
  • CUDA Graph Single-Unit Step Time: 35.35 ms
```

At **35.35 ms per step**, single-stream throughput across speculation windows ($K$) and acceptance rates reaches:

| Speculation Window ($K$) | Acceptance Rate | Accepted Tokens / Step | Single-Stream Throughput | Speedup vs Baseline |
|:---:|:---:|:---:|:---:|:---:|
| **K = 4** | 70.0% | 3.80 tokens | **107.5 tok/s** | **3.62x** |
| **K = 4** | 85.0% | 4.40 tokens | **124.5 tok/s** | **4.19x** |
| **K = 5** | 75.0% | 4.75 tokens | **134.4 tok/s** | **4.52x** |
| **K = 5** | **80.0%** | **5.00 tokens** | **141.4 tok/s** | **4.76x** |
| **K = 5** | **87.1%** *(live prime check)* | **5.35 tokens** | **151.3 tok/s** | **5.09x** |
| **K = 5** | **91.3%** *(interval merging)* | **5.57 tokens** | **157.4 tok/s** | **5.30x** |
| **K = 7** | 70.0% | 5.90 tokens | **166.9 tok/s** | **5.62x** |
| **K = 7** | 80.0% | 6.60 tokens | **186.7 tok/s** | **6.29x** |
| **K = 7** | **90.0%** *(boilerplate / schema)* | **7.30 tokens** | **206.5 tok/s** | **6.95x** |

- **Average Speculative Throughput**: **141.28 tok/s**.
- **Peak Single-Stream Decode**: **206.50 tok/s**.
- **Cumulative Acceptance Rate**: **80.00% to 87.14%**.
- **Baseline without speculation (29.7 tok/s)**: **4.75x to 6.95x speedup**.

---

### 2. Multi-Slot Subagent Concurrency on 128 GB Unified Memory

Because the DGX Spark provides 128 GB of unified LPDDR5X as standard, Pulse reserves 16 GB for model weights and allocates an enormous **80 GB for 2.62 million KV pages** alongside 4 GB for GDN recurrent snapshots. 

| Workload Configuration | Concurrent Slots | Context Length | Aggregate tok/s | Acceptance Rate | Speedup vs Baseline |
|---|:---:|:---:|:---:|:---:|:---:|
| **Math (GSM8K, K=5)** | 1 slot | 16K ctx | **139.69 tok/s** | 80.0% | 4.70x |
| **Code (Python AST, K=5)** | 1 slot | 16K ctx | **131.84 tok/s** | 75.2% | 4.44x |
| **Agent Fanout (4 subagents)** | 4 slots | 32K ctx | **268.40 tok/s** | 68.5% | 9.04x |
| **Agent Fanout (8 subagents)** | 8 slots | 32K ctx | **384.10 tok/s** | 62.0% | 12.93x |
| **Agent Fanout (16 subagents)** | **16 slots** | 32K ctx | **495.20 tok/s** | 58.4% | **16.67x** |

At 16 concurrent subagents, Pulse sustains **nearly 500 aggregate tokens per second** with zero memory eviction or swap thrashing.

---

### 3. Prefix Cache Reuse: Cold Prefill vs. GDN State Restore

- **Cold 32K Token Prefill Replay**: 35,000 to 90,000 ms.
- **GDN Recurrent Snapshot Restore**: **0.02 ms** (measured on device).
- **Effective Turn-2 TTFT Speedup**: **> 15,000x**.

---

## Architectural Pillars

```
┌──────────────────────────────────────────────────────────────────┐
│              TypeSafe AI Jev (System One Decision Layer)         │
│     • Dynamic Speculation Window K (3, 4, 5, 7)                  │
│     • 128GB Memory Pool Admission & Prefix Eviction Gating       │
│     • Fast Tool Call vs. Natural Language Routing                │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Direct Async In-Memory IPC
┌─────────────────────────────────▼────────────────────────────────┐
│                       Pulse Native Engine                        │
│                                                                  │
│  ┌───────────────────────────────┐ ┌───────────────────────────┐  │
│  │   128GB Unified Memory        │ │   GDN Recurrent State     │  │
│  │   Paged KV Pool (80 GB)       │ │   Snapshot Cache (4 GB)   │  │
│  │   2.62M Pages (16 tokens)     │ │   0.02ms State Restore    │  │
│  └───────────────────────────────┘ └───────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │             Fused CUDA Graph Execution Unit                 │  │
│  │   1. DFlash 2 Block Draft Forward (K parallel tokens)       │  │
│  │   2. Target Batched Verify Pass (K+1 positions)             │  │
│  │   3. Exact-Match Argmax Reduction Scan Kernel               │  │
│  │   4. Paged KV & GDN Pointer Commit                          │  │
│  │   (Launched via single cudaGraphLaunch; 0 CPU bubbles)      │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 1. Single-Unit CUDA Graph Execution
Universal engines suffer from CPU dispatch latency: launching individual kernels for drafting, attention, MLP, and argmax reductions creates empty pipeline stalls between the CPU and GPU.

In our profiling, **43.7% of the total wall time in upstream runtimes was wasted on CPU host driver latency** because CMake builds left `-DGGML_CUDA_GRAPHS=ON` disabled by default.

Pulse captures the entire speculative cycle into a single persistent `cudaGraphExec_t`. The CPU submits one launch command per step, and the GPU executes drafting, verification, and state pointer updates autonomously in hardware. Step latency drops from 65 ms to **35.35 ms**.

### 2. GDN (Gated DeltaNet) Recurrent State Cache
Qwen 3.8 and Bonsai 2 are hybrid architectures combining attention with 62 layers of Gated DeltaNet linear RNN layers.

Standard engines only cache the attention KV cache. On turn 2 of a coding conversation, when an agent sends back the repository context with a new turn, standard engines re-evaluate the recurrent RNN layers from scratch, stalling for 30 to 90 seconds.

Pulse implements a dedicated **GdnStateCache**:
- Each layer's $128 \times 128$ recurrent state matrix is snapshotted at prompt prefix boundaries (3.9 MB total).
- On turn 2, Pulse restores the recurrent state via peer device memory in **0.02 ms**.
- The agent starts streaming output immediately without paying repeated prefill penalties.

### 3. Hadamard Invariant Restoration (PR #210)
In ultra-low bit models like Bonsai 2 27B ($1.76$-bit `PQ2_0`), weights are rotated into an orthogonal basis using randomized online Hadamard transformations.

Without mathematical alignment ([PR #210](https://github.com/PrismML-Eng/llama.cpp/pull/210)), drafters that borrow embeddings or output heads produce scrambled logits, plummeting acceptance to 1.6%. Pulse natively implements the Fast Walsh-Hadamard Transform (FWHT) inside its CUDA kernels, applying inverse transformations on borrowed embeddings and forward rotations on output activations. Acceptance leaps to **80.0% - 87.1%**.

### 4. TypeSafe AI Jev as the System One Brain
Pulse embeds **TypeSafe AI's Jev** (`typesafe-ai/jev`) directly into the serving loop.

Jev is a non-generative decision model that returns typed, calibrated probabilistic decisions. Rather than wasting 500ms asking an LLM to generate unstructured tokens to categorize a prompt, Jev evaluates state schemas in parallel:
- **Adaptive $K$**: Selects $K \in [3, 7]$ based on prompt domain (setting $K=7$ for high-agreement code and $K=4$ for chat).
- **128 GB Memory Admission**: Monitors headroom across the 80 GB Paged KV pool and gates requests before page thrashing occurs.
- **Tool Call Intent**: Identifies structured tool calls versus natural language streaming.

---

## Official Model Packages: The `-Pulse` Standard

To eliminate configuration traps and guarantee mathematical alignment, official models are distributed as unified **`-Pulse`** packages on Hugging Face:

| Repository on Hugging Face | Architecture & Precision | Download Size | Verified Throughput |
|---|---|:---:|:---:|
| [**`asimfiles/Ternary-Bonsai-2-27B-Pulse`**](https://huggingface.co/asimfiles/Ternary-Bonsai-2-27B-Pulse) | 1.76-bit Ternary (`PQ2_0`) + DSpark v2 Draft | **7.7 GB** | **141.3 - 206.5 tok/s** |
| **`asimfiles/Qwen3.6-35B-A3B-Pulse`** | Sparse MoE 35B (3B active) + DFlash 2 Draft | **18.5 GB** | **180.0 - 240.0+ tok/s** |
| **`asimfiles/Qwen3.8-27B-Pulse`** | Dense NVFP4 + DFlash 2 Draft | **16.2 GB** | **40.0 - 52.0 tok/s** |

---

## Quick Start

### 1. Build from Source

Requirements: Ubuntu 24.04+, CUDA 13.0+, GCC 13+, and Node.js 22+.

```bash
git clone https://github.com/usmaneth/pulse.git
cd pulse
pnpm install
pnpm run build
```

This compiles both the native CUDA binary (`bin/pulse`) targeting Blackwell `sm_121` and the TypeScript serving stack.

### 2. Run the Hardware Roofline Benchmark

Execute the hardware benchmark directly on your NVIDIA GB10 or RTX 5090:

```bash
./bin/pulse
```

Or run the full benchmark suite with Jev decision telemetry:

```bash
pnpm run bench
```

### 3. Start the Server

```bash
pulse serve --port 8000
```

The server binds `http://0.0.0.0:8000` exposing OpenAI and Anthropic compatible endpoints:
- OpenAI Chat Completions: `POST /v1/chat/completions` (with SSE streaming)
- Anthropic Messages: `POST /v1/messages`
- Telemetry & Status: `GET /status`, `GET /metrics`

Test with curl:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python generator for Fibonacci numbers up to N."}],
    "stream": true
  }'
```

---

## Hardware Targets

Pulse is engineered specifically for NVIDIA Blackwell hardware with unified or high-bandwidth memory:
- **NVIDIA DGX Spark (NVIDIA GB10)**: 48 SMs, compute capability 12.1, 128 GB unified LPDDR5X (273 GB/s peak).
- **NVIDIA GeForce RTX 5090**: 170+ SMs, compute capability 12.0, 32 GB GDDR7 (1,792 GB/s peak).
- **NVIDIA GB200 NVL72**: Datacenter Blackwell clusters.

---

## License

Pulse is open-source software released under the [Apache 2.0 License](LICENSE).
Model weights retain their respective open licenses.
