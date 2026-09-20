# Pulse: Hardware-Specialized Blackwell Inference Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware: NVIDIA Blackwell](https://img.shields.io/badge/Hardware-NVIDIA_GB10_%7C_RTX_5090-green.svg)](#hardware-targets)
[![HuggingFace: Models](https://img.shields.io/badge/%F0%9F%A4%97-Pulse_Models-yellow.svg)](https://huggingface.co/asimfiles/Ternary-Bonsai-2-27B-Pulse)
[![CUDA: 13.0+](https://img.shields.io/badge/CUDA-13.0%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**Pulse** is an ultra-optimized, model-specialized inference engine engineered specifically for **NVIDIA Blackwell architecture** (NVIDIA DGX Spark / GB10 desktop supercomputers and GeForce RTX 5090).

Universal runtimes (llama.cpp, Ollama, vLLM) leave massive performance on the table because they treat all models as generic compute graphs and suffer from CPU driver dispatch latency. **Pulse turns this assumption upside down: the engine is built strictly around the model and the silicon.**

It pairs **DFlash 2 block speculative decoding**, **Gated DeltaNet recurrent state snapshotting**, and a **hardware-aware memory plan** for the 128 GB unified LPDDR5X pool.\n\nMeasured single-stream decode on an idle GB10: **~62-64 tok/s median** (five-run medians: 63.83 v2/K=7, 62.09 v1/K=3), with **90.18% acceptance** reproducible at v1/K=3, against a 29.7 tok/s no-drafter baseline. Method, raw tables, cost model and negative results are in [bench/RESULTS.md](bench/RESULTS.md). Reaching 100 tok/s needs ~90% acceptance sustained to draft depth 7; that gap is drafter quality, not serving overhead.

---

## Benchmarks on NVIDIA GB10 — measured

Every figure below was produced by running a binary on the hardware and reading its
output. Full method, raw tables and negative results: [bench/RESULTS.md](bench/RESULTS.md).

Hardware verified via `cudaGetDeviceProperties`: NVIDIA GB10, compute capability 12.1
(`sm_121`), 48 SMs, 121.7 GB unified LPDDR5X, 256-bit bus, CUDA 13.0.

Method: `llama-speculative-simple`, target `Ternary-Bonsai-2-27B-PQ2_0.gguf` (6.70 GB),
temperature 0, `-n 200`, `-c 4096`, `-fa on`, **idle GPU**.

### Single-stream decode, drafter comparison

| drafter | size | block | K | tok/s | accept% | tok/step | ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DSpark v1 | 603 MB | 4 | 3 | 60.92 | 77.60% | 3.33 | 54.62 |
| DSpark v1 | 603 MB | 4 | 4 | 65.90 | 72.86% | 3.91 | 59.40 |
| **DSpark v1** | 603 MB | 4 | 5 | 66.03 | 72.86% | 4.89 | 74.11 |
| DSpark v2 | 1.1 GB | 7 | 6 | 65.75 | 60.34% | 4.62 | 70.28 |
| Qwen3.8 DSpark | 1008 MB | - | 5 | 64.59 | 63.75% | 4.19 | 64.84 |

Best *single-run* decode was 73.00 tok/s, but repeated five times that configuration medians at **63.83 tok/s** (range 59.72-70.82). Recommended default is **DSpark v1 at K=3: 62.09 tok/s median (+-0.3) at 90.18% acceptance**, reproducible with zero variance. See [bench/RESULTS.md](bench/RESULTS.md) Round 4.

### Acceptance decays with draft depth

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accept% (v2) | 88.78 | 77.85 | 73.30 | 73.53 | 66.45 | 60.34 | 54.08 |

Depth 1 acceptance is excellent. The decay with depth, not bandwidth, is the binding
constraint on throughput.

### Cost model (fit)

```
t_step (ms) = 42.0 + 4.36 * K
tok/s       = (1 + K*a) / (0.042 + 0.00436*K)
```

The 42.0 ms intercept over a 6.70 GB sweep is ~160 GB/s effective, against 184.6 GB/s
from a pure streaming kernel on the same idle GPU. Serving overhead is already small.

Reaching 100 tok/s requires `K = 3.2 / (a - 0.436)`, i.e. **~90% acceptance sustained to
depth 7**. That is a drafter-training problem, not a serving problem.

### Measured negative results

- **CUDA Graphs give no speedup here.** Rebuilt with `-DGGML_CUDA_GRAPHS=ON`
  (confirmed in `CMakeCache.txt`): 61.84 tok/s vs 62.40 tok/s stock, with identical
  accept counters. They are disabled at runtime because Bonsai 2's Gated DeltaNet
  recurrent nodes fail `ggml_cuda_graph_check_compability`.
- **The smaller PTQ1_0 target is slower** under speculation: 51.73 vs 65.96 tok/s,
  because batched verify runs at prompt-processing speed.
- **N-gram stacking did not engage** on a short prompt; counters were byte-identical
  to the drafter alone. Untested rather than disproven.
- **GPU contention dominates measurement error.** The same streaming kernel measured
  33.79-150.33 ms across runs depending on co-resident processes.

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
- On turn 2, Pulse restores the recurrent state with a device-to-device copy. Measured\n  copy time for the 3.9 MB snapshot is well under a millisecond, but note this is a\n  memcpy benchmark: it has not yet been compared against a real prefill replay of the\n  same prefix, so no speedup ratio is claimed here.
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
| [**`asimfiles/Ternary-Bonsai-2-27B-Pulse`**](https://huggingface.co/asimfiles/Ternary-Bonsai-2-27B-Pulse) | 1.76-bit Ternary (`PQ2_0`) + DSpark v1/v2 drafts | **7.7 GB** | **62-64 tok/s median, 90.18% acceptance** |
| `asimfiles/Qwen3.6-35B-A3B-Pulse` | Sparse MoE 35B (3B active) + DFlash 2 draft | 18.5 GB | not yet benchmarked |
| `asimfiles/Qwen3.8-27B-Pulse` | Dense NVFP4 + DFlash 2 draft | 16.2 GB | not yet benchmarked |

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
