---
language:
- en
license: apache-2.0
tags:
- pulse
- speculative-decoding
- dflash2
- dspark
- blackwell
- gb10
- rtx-5090
- ternary
- qwen35
model_name: Ternary-Bonsai-2-27B-Pulse
base_model: PrismML/Ternary-Bonsai-2-27B
pipeline_tag: text-generation
inference: false
---

# Ternary-Bonsai-2-27B-Pulse

**Hardware-Specialized Inference Package for NVIDIA Blackwell (DGX Spark / GB10 / RTX 5090)**

This repository provides the validated **Pulse** package for **Bonsai 2 27B**:
- **Target Model**: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (1.76-bit, 6.70 GB).
- **Speculative Drafter**: `Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf` (DFlash 2 / DSpark v2, 1.03 GB, block size 7).
- **Mathematical Invariant**: Incorporates the Hadamard coordinate transform fix ([PR #210](https://github.com/PrismML-Eng/llama.cpp/pull/210)), ensuring 80-90% speculative acceptance instead of 1.6%.

## Performance on NVIDIA GB10 (128 GB Unified LPDDR5X)

Measured with **Pulse** native single-unit CUDA Graphs:
- **Single-Stream Decode**: **141.3 tok/s** (peaking at **206.5 tok/s**).
- **Speculative Acceptance Rate**: **80.0% to 87.1%**.
- **GDN Recurrent Prefix State Restore**: **0.02 ms** (>15,000x faster than cold 32K prefill replay).
- **16 Subagent Concurrency**: **495.2 aggregate tok/s** on 128 GB unified memory.

## Quick Start with Pulse

```bash
# 1. Install Pulse CLI
npm install -g @usmaneth/pulse

# 2. Serve the model with zero configuration
pulse serve --model usmaneth/Ternary-Bonsai-2-27B-Pulse
```

OpenAI and Anthropic compatible endpoints are exposed on `http://127.0.0.1:8000`.
