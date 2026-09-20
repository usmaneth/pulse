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
|---|---|---|---|---|---|
| v1 | 4 | 3 | 58.08 | 78.29% | 3.32 |
| **v1** | 4 | **4** | **63.77** | 72.28% | 3.86 |
| v2 | 7 | 7 | 63.83 median (59.72-70.82) | 68.59% | 5.81 |

Measured over 3-5 repeats on an idle GB10 with a realistic code context, temperature 0,
`--ignore-eos`. Deterministic at temperature 0: spread is 0.37 tok/s for v1.

No-drafter baseline on the same target is **27.54 tok/s**, so the best configuration is
a **2.3x** speedup. Prefill reaches **~995 tok/s** with a properly configured server.

> **Correction.** An earlier version of this card reported 73.00 tok/s as the headline.
> That was a single best run, not a median; five repeats give 63.83 for that config. It
> also reported 646 tok/s prefill, measured with a smaller batch setting than the tuned
> configuration. Both are corrected above.

Higher acceptance is reachable at lower draft depth: **90.18%** at K=3 on some prompts,
reproducible to the digit across five runs. Acceptance is strongly prompt-dependent
(measured range 21% to 96% across workloads), so treat it as a property of the workload
rather than of the drafter.

### Which to use

- Want maximum throughput: **v1 at K=4** (63.77 tok/s). Use the full block size; K past
  the drafter's block size is a no-op.
- Want maximum acceptance: **v1 at K=3** (78-90% depending on prompt).
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
