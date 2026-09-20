# Should Pulse become a real inference engine?

Inco's Splash is a genuine from-scratch engine. Its `runtime/` is 135 files -
`Engine.cpp`, `Scheduler.cpp`, `StateCache.cpp`, `model/DFlashDraft.cpp`,
`model/QwenState.cpp`, `ops/{Linear,Normalization,Sampling,DraftAttention}`. It
loads its own weights, runs its own forward pass, owns the KV cache and the
scheduler. Decode compiles to four fixed batch widths - **M8/M16/M24/M32, one
kernel per width** - and requests are grouped into `BatchCohort`s by sampling
mode.

Pulse today loads no weights and runs no forward pass. So the honest answer to
"is Pulse an engine like Splash" is no.

This document is the measured case for whether it should become one.

## What the hardware allows

Everything below was measured on this GB10. Nothing is extrapolated.

| quantity | value | how |
|---|---|---|
| sustained memory bandwidth | 184.6 GB/s | `bench_gb10_streaming`, idle GPU |
| target model (`PQ2_0`) | 6.70 GB | file size |
| predicted batch-1 step | 36.3 ms | 6.70 / 184.6 |
| **measured** full decode step at 1 tok/step | **36.65 ms** | n-gram design curve |
| measured no-drafter decode | 27.54 tok/s | `llama-batched-bench` B=1 |

The prediction and the measurement agree to 1%. **llama.cpp's batch-1 decode is
already at the memory-bandwidth roofline.** There is no runtime overhead to
reclaim in the target forward pass.

## Where the remaining time actually goes

From the design curve (model-free n-gram drafter, 100% acceptance, so it isolates
hardware verify cost from drafter quality):

| | ms/step | tok/s |
|---|---|---|
| curve at 3.86 tok/step, **free** drafter | 44.9 | 86 |
| measured, v1 drafter at K=4 | 60.3 | 64 |
| **drafter cost** | **15.4** | -26% |

### Measured decomposition

Two drafters at **matched K=3**, same prompt, warm interleaved protocol
(2 warmup discarded, 5 repeats, spreads 2.9% and 3.0%):

| drafter | size | tok/s | acceptance | ms/step |
|---|---|---|---|---|
| v1 | 0.632 GB | 53.62 | 78.29% | 61.95 |
| v2 | 1.105 GB | 53.19 | 83.23% | 65.46 |

Slope: `3.51 ms / 0.473 GB` = **7.42 ms/GB, i.e. ~135 GB/s** on the drafter's
weights - 73% of the 184.6 GB/s this machine sustains.

An earlier revision of this document claimed ~5.2 ms/GB (~191 GB/s, "essentially
full bandwidth"). That was computed from numbers taken in different machine
states and is withdrawn. The figure above is a same-prompt, matched-K,
warm-protocol differential and is the trustworthy one.

No-drafter baseline on the **same context length** (1359 tokens, B=1,
`llama-batched-bench`): **24.67 tok/s = 40.5 ms/step**. Note this is 40.5 ms
rather than the 36.6 ms measured at 256-token context - KV cache reads add
roughly 4 ms at this context.

**What is solid:** the size slope (7.42 ms/GB) and the per-verify-row cost, both
same-binary same-prompt differentials.

**What is weaker:** splitting the drafter cost into a fixed orchestration term
plus a size term requires a no-drafter baseline from a *different binary*
(`llama-batched-bench` vs `llama-speculative-simple`), which carries its own
overhead. The fixed term is therefore indicative, not precise.

## So what would an engine actually buy?

| change | measured payoff | note |
|---|---|---|
| fuse draft+verify into one graph | 60.3 -> ~54.1 ms, **~+11%** | this is the real one |
| `mmvq` kernels for 9-16 columns | ~13%, but only at 9-15 rows | our drafters give 5 and 8 rows, so it does not apply |
| CUDA Graphs over the whole step | unmeasured | llama.cpp cannot: Gated DeltaNet nodes fail `ggml_cuda_graph_check_compability`. A static-shape engine could |

**Roughly 1.1-1.2x on single-stream, for months of work.** That is the honest
number, and it is not 10x. It is also not nothing.

## What an engine would NOT fix

- the 36.6 ms weight sweep - physics
- the drafter's own ~9.2 ms weight read - physics
- acceptance rate - that is a model problem, not a runtime problem

## The actual highest-leverage work

The design curve says what throughput requires:

| accepted tok/step | 1 | 4 | 5 | 6.9 | 12.8 | 17 |
|---|---|---|---|---|---|---|
| tok/s (free drafter) | 27.3 | 85.4 | **101.3** | 126.2 | 174.1 | 219.4 |

With v1's real 15.4 ms drafter cost, 5 tok/step gives ~77 tok/s, not 101. To
reach **100 tok/s single-stream you need roughly 7 accepted tokens per step at
v1's cost** - which is a drafter problem, not a runtime problem. v2 currently
delivers 5.74 tok/step at 69% acceptance.

**Ranked by measured payoff per unit of work:**

1. **A better drafter** - higher acceptance at depth. Directly moves tok/step,
   which is the term the curve is most sensitive to. `dflash-training/` already
   has the pipeline, and `convert_safetensors_to_dspark.py` already emits
   block 7.
2. **Drafter fusion** (~11%) - the one genuine engine win. Could be prototyped
   inside llama.cpp by running the drafter in the target's context rather than a
   separate one, without writing an engine.
3. **A full engine rewrite** - captures 2, plus CUDA Graphs, plus arbitrary
   kernel shapes. Months. Competes with a mature CUDA backend that is already at
   the roofline on the dominant term.

## Recommendation

Do not rewrite. The measurements say llama.cpp is at the bandwidth roofline for
the target forward pass, and the only engine-class win that survives measurement
is ~11% from drafter fusion - which can be prototyped inside llama.cpp first.

Spend the effort on the drafter instead. That is where the curve is steep.
