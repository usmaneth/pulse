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

Speculative-decoding package for **Bonsai 2 27B** on NVIDIA Blackwell
(DGX Spark / GB10, RTX 5090), used by the [Pulse](https://github.com/usmaneth/pulse)
engine.

Contents:

- **Target**: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (6.70 GB, 1.76-bit ternary,
  Hadamard-folded). Not included here; fetch from the PrismML release.
- **Drafter v1**: `Ternary-Bonsai-2-27B-dspark-dflash-v1-Q4_0.gguf` (603 MB, block size 4)
- **Drafter v2**: `Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf` (1.1 GB, block size 7)
- `SHA256SUMS` for both drafters.

Both drafters require the borrowed-Hadamard fix
([PrismML-Eng/llama.cpp#210](https://github.com/PrismML-Eng/llama.cpp/pull/210)).
Without it, a drafter that borrows the target's embedding table and output head
computes logits in the wrong basis and acceptance falls to roughly 1-2%.

## Measured performance

> **Correction notice.** An earlier version of this card claimed
> "141.3 - 206.5 tok/s". That figure was never measured; it was extrapolated from a
> single unstable bandwidth run. The numbers below are read from program output on the
> hardware described. Full method, raw tables and negative results:
> [bench/RESULTS.md](https://github.com/usmaneth/pulse/blob/main/bench/RESULTS.md).

Hardware: NVIDIA GB10, compute capability 12.1 (`sm_121`), 48 SMs, 121.7 GB unified
LPDDR5X, CUDA 13.0. Runs used `llama-speculative-simple`, temperature 0,
`--ignore-eos`, 1854-token code prompt, **idle GPU**.

| drafter | block | K | decode tok/s | acceptance | tok/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 4 | 3 | 62.98 | **90.18%** | 3.72 |
| v1 | 4 | 4 | 68.69 | 83.78% | 4.37 |
| v2 | 7 | 6 | 71.09 | 71.81% | 5.31 |
| v2 | 7 | 7 | **73.00** | 68.59% | 5.81 |

No-drafter baseline on the same target is 29.7 tok/s, so the best configuration is a
**2.46x speedup**.

Prefill on the same setup reaches **646 tok/s** at a 2756-token prompt.

### Choosing a drafter

- Want maximum throughput: **v2 at K=7** (73.00 tok/s).
- Want maximum acceptance: **v1 at K=3** (90.18%).
- Match K to the drafter's block size. Running v1 at K=6 costs 88.16 ms/step versus
  59.40 ms/step at K=4, because K above the block size forces a second draft pass.

### Measured negative results

- Aggregate throughput does **not** improve with concurrency on this hardware:
  34.1 tok/s at 1 stream rising only to 65.8 tok/s at 16. One stream already draws
  ~160 GB/s of a measured 184.6 GB/s ceiling.
- The smaller PTQ1_0 target (5.60 GB) is **slower** under speculation than PQ2_0
  (51.73 vs 65.96 tok/s).
- KV cache quantization (`-ctk q8_0 -ctv q8_0`) is marginally slower than f16.
- Building llama.cpp with `-DGGML_CUDA_GRAPHS=ON` changes nothing (61.84 vs 62.40
  tok/s); graphs are disabled at runtime by the Gated DeltaNet nodes.

## Usage

```bash
llama-speculative-simple \
  -m Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -md Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  -ngl 99 -ngld 999 -fa on -c 4096 -b 4096 --temp 0
```

Raise `-b` above your prompt length or the run fails with
`the prompt exceeds the batch size`. Keep `-ub` at 512; larger values reduce prefill
throughput.
