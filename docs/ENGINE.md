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

Decomposing that 15.4 ms across two drafters at matched K, solving
`overhead = F + size x B`:

- **B ~ 5.2 ms/GB, i.e. ~191 GB/s.** The drafter's weights are read at
  essentially full bandwidth. This is physics and is **not recoverable**.
- **F ~ 6.2 ms/step of fixed orchestration.** llama.cpp runs the drafter as a
  separate `llama_context` with its own graph, feature staging and KV add/remove
  every step. **This is what engine fusion eliminates.**

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
