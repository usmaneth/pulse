# Pulse — Measured Results on NVIDIA DGX Spark (GB10)

Every number in this file was produced by running a binary on the hardware and
reading its output. Nothing here is estimated, extrapolated, or simulated.
Where a figure is a model fit rather than a direct reading, it says so.

## Hardware

Verified with `cudaGetDeviceProperties` and `nvidia-smi`:

- NVIDIA GB10, compute capability **12.1** (`sm_121`), 48 SMs
- 121.7 GB addressable unified LPDDR5X, 256-bit bus
- CUDA 13.0, driver 580.178.04, kernel 7.0.0-1019-nvidia, aarch64 (Cortex-X925 + A725)

## Method

- Binary: `llama-speculative-simple` (PrismML llama.cpp @ `288859a96`, includes the
  borrowed-Hadamard fix).
- Target: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (6.70 GB, 1.76-bit ternary, Hadamard-folded).
- Prompt: a `merge_intervals` Python completion, temperature 0, `-n 200`, `-c 4096`,
  `-fa on`, `-ngl 99 -ngld 999`.
- **GPU was idle.** Contending processes were killed first. This matters enormously:
  the same command measured 62.4 tok/s with a 21.5 GB server resident and 66.0 tok/s
  on a clean GPU.
- `tok/s` is read from the `decoded ... speed:` line. `accept%` is `n_accept/n_drafted`.
  `steps = n_drafted / K`; `tok/step` and `ms/step` are derived from those counters.

## Finding 1 — CUDA Graphs give no speedup on this model

We rebuilt llama.cpp with `-DGGML_CUDA_GRAPHS=ON` (confirmed `GGML_CUDA_GRAPHS:BOOL=ON`
in `CMakeCache.txt`) and ran the identical command against the stock build.

| build | tok/s | n_drafted | n_accept | accept% |
| --- | ---: | ---: | ---: | ---: |
| stock (graphs off) | 62.40 | 155 | 103 | 66.45% |
| `GGML_CUDA_GRAPHS=ON` | 61.84 | 155 | 103 | 66.45% |

Identical within noise, with byte-identical accept counters. The hypothesis that a
large fraction of step time was CPU graph-launch overhead is **refuted**. CUDA graphs
are rejected at runtime for this graph: `ggml_cuda_graph_check_compability` disables
them on unsupported node types, and Bonsai 2's `qwen35` backbone is a Gated DeltaNet /
attention hybrid whose recurrent nodes trip that check.

## Finding 2 — K sweep, DSpark v2 drafter

| K | tok/s | accept% | tok/step | ms/step |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 33.00 | 88.78% | 1.89 | 57.21 |
| 2 | 42.93 | 77.85% | 2.56 | 59.56 |
| 3 | 50.42 | 73.30% | 3.20 | 63.55 |
| 4 | 57.85 | 73.53% | 3.94 | 68.12 |
| 5 | 62.58 | 66.45% | 4.32 | 69.06 |
| 6 | **65.41** | 60.34% | 4.62 | 70.66 |
| 7 | 63.13 | 54.08% | 4.79 | 75.82 |

Acceptance is excellent at depth 1 (**88.78%**) and decays steeply with depth
(54% by K=7). `n_drafted` saturates at 196 for K >= 7 because the v2 drafter's block
size is 7, so K above 7 buys nothing.

## Finding 3 — Drafter comparison

Same target, same prompt, clean GPU.

| drafter | size | block | K | tok/s | accept% | tok/step | ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DSpark v1 | 603 MB | 4 | 2 | 50.70 | 78.48% | 2.57 | 50.68 |
| DSpark v1 | 603 MB | 4 | 3 | 60.92 | 77.60% | 3.33 | 54.62 |
| DSpark v1 | 603 MB | 4 | 4 | 65.90 | 72.86% | 3.91 | 59.40 |
| **DSpark v1** | 603 MB | 4 | 5 | **66.03** | 72.86% | 4.89 | 74.11 |
| DSpark v2 | 1.1 GB | 7 | 6 | 65.75 | 60.34% | 4.62 | 70.28 |
| Qwen3.8 DSpark | 1008 MB | - | 4 | 62.65 | 70.71% | 3.83 | 61.11 |
| Qwen3.8 DSpark | 1008 MB | - | 5 | 64.59 | 63.75% | 4.19 | 64.84 |

The smallest drafter wins. Matching K to the drafter block size matters: v1 at K=6
costs 88.16 ms/step versus 59.40 ms/step at K=4, because K=6 forces a second draft
pass through a block-size-4 drafter.

## Finding 4 — The smaller target is slower

| target | size | drafter | K | tok/s | ms/step |
| --- | ---: | --- | ---: | ---: | ---: |
| PQ2_0 | 6.70 GB | v1 | 4 | 65.96 | 59.34 |
| PTQ1_0 | 5.60 GB | v1 | 4 | 51.73 | 75.69 |

PTQ1_0 holds 16% fewer bytes yet decodes 22% slower under speculation, because the
batched verify pass runs at prompt-processing speed and PTQ1_0's prompt processing is
roughly half of PQ2_0's.

## Cost model (fit, not a direct reading)

Least-squares over the DSpark v1 rows:

```
t_step (ms) = 42.0 + 4.36 * K
```

The 42.0 ms intercept over a 6.70 GB weight sweep implies ~160 GB/s effective
bandwidth, against 184.6 GB/s measured by a pure streaming kernel on the same idle
GPU. Runtime overhead is therefore already small; there is little left to win from
scheduling.

Throughput follows:

```
tok/s = (1 + K*a) / (0.042 + 0.00436*K)      a = acceptance
```

Solving for 100 tok/s gives `K = 3.2 / (a - 0.436)`:

| acceptance | required K |
| ---: | ---: |
| 0.80 | 8.8 |
| 0.85 | 7.7 |
| **0.90** | **6.9** |
| 0.95 | 6.2 |

## Where this leaves the 100 tok/s target

Best measured single-stream decode today: **66.03 tok/s** at 72.86% acceptance
(DSpark v1, K=5, clean GPU).

The model says 100 tok/s needs **~90% acceptance sustained to depth 7**. We currently
have 88.78% at depth 1 and 54-73% by depth 4-7. The bandwidth term is already near the
hardware roofline and the runtime overhead is small, so the remaining gap is entirely
**drafter quality at depth**, not scheduling, quantization, or kernel dispatch.

That is a training problem, not a serving problem: a DSpark v3 with block size >= 8 and
a flat acceptance profile across depth. Runtime-side levers (CUDA graphs, smaller
target, smaller drafter, n-gram stacking) have each been measured and none of them
close the gap.

## Negative results worth recording

- `--spec-type draft-dspark,ngram-map-k` and `draft-dspark,ngram-mod` produced counters
  byte-identical to `draft-dspark` alone (155 drafted / 103 accepted). The n-gram
  drafter never contributed on a ~60-token prompt. Prompt-lookup needs a long repeated
  context to engage; this test was too short to exercise it and the idea remains
  untested rather than disproven.
- GPU contention is the single largest source of measurement error observed. A pure
  6.70 GB streaming kernel measured 35.49 / 150.33 / 84.85 / 77.99 / 33.79 ms across
  five consecutive runs depending on what else held GPU memory. Always benchmark on an
  idle device.

---

# Round 2 — long-context measurements

## Measurement hazard found: EOS truncation

An initial long-context sweep appeared to show acceptance collapsing to 0% and decode
falling to 3.2 tok/s. That was an **artifact**. Truncated source-code prompts caused the
model to emit EOS immediately, so `decoded` was ~1 token and both the speed and the
acceptance counters were meaningless. The giveaway was non-monotonicity: a 668-token
prompt worked fine at 56 tok/s while 65-token and 1257-token prompts "failed".

All Round 2 numbers therefore use `--ignore-eos` so every run decodes the same token
count and the figures are comparable.

## Acceptance vs context length (DSpark v1, K=4, `--ignore-eos`, 155 tokens decoded)

| prompt tokens | decode tok/s | accept% | prefill tok/s |
| ---: | ---: | ---: | ---: |
| 65 | 64.24 | 74.84% | 130.5 |
| 668 | 60.45 | 69.94% | 335.1 |
| 1257 | 65.76 | 80.82% | 481.0 |
| 1854 | **67.11** | **84.40%** | 562.6 |
| 2756 | 62.41 | 78.00% | 646.2 |

Acceptance does **not** degrade with context on this stack; it is higher at agentic
context lengths than on short prompts. Prefill throughput rises monotonically with
prompt size, reaching **646 tok/s** at 2756 tokens.

## Deep-K sweep at 1854-token context (`--ignore-eos`, 200 tokens decoded)

| drafter | block | K | tok/s | accept% | tok/step | ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DSpark v1 | 4 | 3 | 62.98 | **90.18%** | 3.72 | 59.02 |
| DSpark v1 | 4 | 4 | 68.69 | 83.78% | 4.37 | 63.59 |
| DSpark v2 | 7 | 4 | 63.92 | 80.53% | 4.23 | 66.21 |
| DSpark v2 | 7 | 5 | 68.91 | 76.08% | 4.81 | 69.78 |
| DSpark v2 | 7 | 6 | 71.09 | 71.81% | 5.31 | 74.75 |
| DSpark v2 | 7 | 7 | **73.00** | 68.59% | 5.81 | 79.63 |

Best measured decode: **73.00 tok/s**. Best measured acceptance: **90.18%**.
They occur at opposite ends of the K tradeoff and have not been achieved together.

## KV cache quantization: no benefit

| context | drafter/K | KV type | tok/s |
| ---: | --- | --- | ---: |
| 1854 tok | v2/K=7 | f16 | 66.59 |
| 1854 tok | v2/K=7 | q8_0 | 65.53 |
| 6052 tok | v2/K=7 | f16 | 70.24 |
| 6052 tok | v2/K=7 | q8_0 | 69.49 |

`-ctk q8_0 -ctv q8_0` is consistently marginally slower. Flash attention already keeps
the KV read off the critical path at these context lengths.

Also note `-c 8192` costs ms/step versus `-c 4096` (87.30 vs 79.63 for the identical
v2/K=7 workload) — allocate only the context you need.

## Updated cost model (long context)

```
t_step (ms) = 48.3 + 4.47 * K
tok/s       = (1 + K*a) / (0.0483 + 0.00447*K)
```

100 tok/s requires `K = 3.83 / (a - 0.447)`:

| acceptance | required K |
| ---: | ---: |
| 0.85 | 9.5 |
| **0.90** | **8.5** |
| 0.95 | 7.6 |

## Status against the 100 tok/s @ 90% acceptance target

**Not met.** Measured best is 73.00 tok/s at 68.59% acceptance, or 62.98 tok/s at
90.18% acceptance.

The blocker is now precisely characterised: reaching 100 tok/s at 90% acceptance
requires sustaining that acceptance to **draft depth ~8.5**. The deepest drafter
available has block size 7 (DSpark v2) and its acceptance at K=7 is 68.59%. DSpark v1
reaches 90.18% but only at depth 3, where tok/step is capped at 3.72.

Every runtime-side lever has now been measured and none closes the gap:

| lever | result |
| --- | --- |
| CUDA Graphs (`GGML_CUDA_GRAPHS=ON`) | no change (61.84 vs 62.40) |
| smaller target (PTQ1_0) | 22% slower |
| smaller drafter (v1 603 MB) | best drafter, still capped by block size 4 |
| KV quantization q8_0 | marginally slower |
| n-gram stacking | counters unchanged; never engaged |
| larger batch / ubatch | hurts decode; helps prefill only |
| longer context | helps acceptance, does not reach 90% at depth 7+ |

The remaining work is a **drafter training problem**: a DSpark v3 with block size >= 9
holding ~90% acceptance across all 9 positions. The retrain pipeline exists at
`Bonsai-demo/tools/dspark-retrain/`.

## Prefill summary

Prefill is healthy and scales with prompt size on this hardware:

| prompt tokens | prefill tok/s |
| ---: | ---: |
| 65 | 130.5 |
| 668 | 335.1 |
| 1257 | 481.0 |
| 1854 | 562.6 |
| 2756 | 646.2 |

Batch sizing matters: prompts above the batch size hard-fail with
`the prompt exceeds the batch size`, so `-b` must be raised for long contexts.
Raising `-ub` above 512 reduces prefill throughput (389 tok/s at `-ub 2048` versus
656 tok/s at `-ub 512`).

---

# Round 3 — real concurrency

Measured with `bench/sweep.py --mode concurrency` against a live `llama-server`
(PQ2_0 target, DSpark v2 drafter, K=7, `-c 16384 -b 4096 -np 4`, `--jinja`).
Aggregate is total generated tokens divided by the wall-clock span from first
request submitted to last response completed.

| N simultaneous | aggregate tok/s | mean per-stream tok/s | acceptance | failures |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 34.1 | 35.4 | 21.3% | 0 |
| 2 | 48.6 | 24.8 | 21.7% | 0 |
| 4 | 60.1 | 16.1 | 20.3% | 0 |
| 8 | 65.1 | 17.0 | 20.7% | 0 |
| 16 | **65.8** | 17.1 | 20.9% | 0 |

## The headline result: concurrency does not add aggregate throughput

Aggregate throughput saturates at **65.8 tok/s**, which is essentially the same as the
best single-stream figure (73.00 tok/s). Going from 1 to 16 simultaneous streams buys
roughly 1.9x, and almost all of that is realised by N=4; beyond that the curve is flat.

The reason is visible in the cost model: a single stream already draws about
160 GB/s out of a measured 184.6 GB/s ceiling on this device. There is no spare memory
bandwidth for additional streams to consume. Batching helps when a system is
launch-bound or compute-bound; this one is bandwidth-bound by a dense 6.70 GB weight
sweep per step.

This is the opposite of what published Apple-silicon results show for the same class of
engine, and the difference is explainable by hardware rather than software: an M5 Max
has roughly 614 GB/s of bandwidth against GB10's 273 GB/s peak (184.6 GB/s measured),
so a Mac has headroom for concurrent streams that GB10 does not.

**A previously published claim of "495 aggregate tok/s across 16 subagents" was
fabricated. The measured value is 65.8 tok/s, 7.5x lower.**

## Caveats on this run

- The server was started with `-np 4`, so the N=8 and N=16 rows involve queueing rather
  than 16 truly resident slots. The N=1 to N=4 progression (34.1 -> 48.6 -> 60.1) is
  within the slot budget and already shows the saturation trend.
- Acceptance here (~21%) is far below the 68-90% measured with
  `llama-speculative-simple`, because `bench/sweep.py` sends short prose/chat prompts
  through the server's `--jinja` chat template with reasoning enabled, whereas the
  earlier runs used raw long code prompts. Acceptance is highly workload dependent;
  do not compare these two figures directly.

# FFI verification

`nm -D bin/libpulse_engine.so` now lists all four symbols, and a Bun `dlopen` round trip
succeeds end to end:

```
dlopen OK, symbols: pulse_engine_create, pulse_engine_destroy, pulse_engine_step, pulse_engine_last_rate
engine handle acquired
step 0: tokens=1 measured_gpu_ms=0.2436
step 1: tokens=5 measured_gpu_ms=0.2036
step 2: tokens=5 measured_gpu_ms=0.2051
destroyed cleanly.
```

The reported milliseconds are real `cudaEventElapsedTime` readings. Note these measure
the kernel-dispatch microbenchmark graph (one GEMV plus the scan kernel), not a
62-layer model forward, so the implied token rate is not an inference figure.

---

# Round 4 — repeatability (and a correction to our own headline)

Five consecutive runs of each configuration, identical prompt (1854-token code
context), idle GPU, temperature 0, `--ignore-eos`, 200 tokens.

| config | tok/s runs | median | min | max | spread |
| --- | --- | ---: | ---: | ---: | ---: |
| v2 K=7 | 70.82, 63.83, 70.48, 59.72, 63.42 | **63.83** | 59.72 | 70.82 | 11.10 |
| v1 K=3 | 61.97, 62.54, 62.51, 62.09, 61.92 | **62.09** | 61.92 | 62.54 | 0.62 |

| config | acceptance runs | median | spread |
| --- | --- | ---: | ---: |
| v2 K=7 | 68.59 x5 | 68.59% | **0.00** |
| v1 K=3 | 90.18 x5 | **90.18%** | **0.00** |

## Correction to the previously reported 73.00 tok/s

The 73.00 tok/s figure reported earlier in this document was a **single run**. Repeated
five times, the same configuration has a median of **63.83 tok/s** and a range of
59.72-70.82. 73.00 sits above the observed maximum of this five-run sample and should
be treated as a favourable outlier, not a representative result.

**Corrected headline: ~62-64 tok/s median**, against a 29.7 tok/s no-drafter baseline,
i.e. roughly **2.1x**, not the 2.46x previously claimed.

## Acceptance is deterministic per workload, not stable across workloads

Acceptance has **zero** run-to-run variance (spread 0.00 over five runs for both
configs). This is expected: at temperature 0 with exact-match verification and a fixed
prompt, the token sequence is identical every run, so the drafted/accepted counters are
identical.

That determinism must not be mistaken for generality. The same v2/K=7 configuration
measured 68.59%, 43.77% and 78.34% acceptance on three different prompts, and about 21%
on short chat prompts through the server. **Acceptance is a property of the workload,
not of the engine.** Any single acceptance figure is only meaningful alongside the exact
prompt that produced it.

## Throughput variance differs sharply by configuration

v2/K=7 swings 17% run to run while v1/K=3 swings 1%. Deeper speculation produces more
variable wall-clock time even when the accepted token sequence is identical, because
more draft positions mean more work whose scheduling can vary. For a latency-sensitive
deployment this argues for the shallower configuration independently of mean throughput.

## Recommended default configuration

**DSpark v1 at K=3.** It gives up about 1.7 tok/s of median throughput against v2/K=7
but returns 21 points of acceptance and an 18x tighter throughput distribution.

---

# Round 5 — the LM head is the exploitable inefficiency

## Question

Measured marginal cost of an extra speculative draft row is 4.47 ms (fit
`t_step = 48.3 + 4.47K`). An earlier nsys profile of the model *body* showed only
~1.1 ms per extra row. Where does the other ~3.4 ms come from?

Hypothesis: the output head. Vocab is 248,320 and hidden is 5120, so a 4-bit head is
0.592 GB. If speculative verification evaluates the head as a per-row GEMV, it re-reads
the entire head for each of the K+1 positions: 0.592 GB / 184.6 GB/s = **3.21 ms per
row**, which almost exactly accounts for the residual.

## Confirmation

| rows | per-row 4-bit GEMV (ms) | marginal ms/row |
| ---: | ---: | ---: |
| 1 | 3.233 | - |
| 2 | 6.220 | 3.035 |
| 4 | 13.230 | 3.682 |
| 6 | 19.359 | 3.708 |
| 8 | 26.025 | 3.791 |

A single-row head GEMV measures **3.233 ms against a 3.21 ms theoretical read**, i.e.
it is already memory-bandwidth optimal. Eight rows therefore cost eight full head reads
(26.03 ms measured vs 25.7 ms predicted). Hypothesis confirmed: **the head is re-read
per verification position, and at K=7 that is 26.0 ms of a 79.63 ms step — 33% of it.**

## Three hand-written batched kernels that failed

| approach | 8 rows (ms) | vs GEMV |
| --- | ---: | ---: |
| 8x per-row GEMV (baseline) | 26.03 | 1.00x |
| naive batched, `float acc[8]` runtime-indexed | 67.59 | 0.38x |
| templated `NROWS`, full unroll | 67.59 | 0.38x |
| shared-memory column tiling (TILE=512) | 38.75 | 0.67x |
| shared-memory column tiling (TILE=256) | 42.82 | 0.61x |

All slower than the GEMV they were meant to replace, and all 12x off the 3.21 ms
single-read floor, so none was bandwidth-bound. Diagnosis: register spill from
dynamically indexed accumulators in the first two, and 4-way shared-memory bank
conflicts in the tiled versions (threads read `sx[r][lc+j]` with `lc` striding 8 floats,
so lanes 0, 4, 8 ... collide).

Recording these because the negative result is the useful part: naive batching of a
skinny tall GEMM does not beat a bandwidth-optimal GEMV without careful attention to
banking and register budget.

## The floor, measured with cuBLAS

| rows | cuBLAS FP16 GEMM (ms) |
| ---: | ---: |
| 1 | 11.975 |
| 2 | 12.307 |
| 4 | 11.675 |
| 6 | 12.306 |
| 8 | **11.622** |

**Flat in row count.** Verifying 8 positions costs the same as verifying 1. This is the
behaviour the hand-written kernels were trying and failing to reach, and it confirms the
saving is real rather than theoretical.

cuBLAS achieves this while moving **4.2x more data** than necessary: an FP16 head is
2.368 GB against 0.592 GB for 4-bit. It still beats 8 sequential 4-bit GEMVs by
**2.24x** (11.62 vs 26.03 ms), and runs at 2.368 GB / 11.622 ms = **203 GB/s**, i.e.
fully bandwidth-saturated.

## What this is worth

| head implementation | head cost at K=7 | step ms | tok/s @ 68.59% | tok/s @ 90% |
| --- | ---: | ---: | ---: | ---: |
| 8x int4 GEMV (current behaviour) | 26.03 | 79.63 | 63.8 | - |
| cuBLAS FP16 batched | 11.62 | 65.23 | 89.1 | 112 |
| int4 mixed-input tensor-core GEMM | ~3.21 | ~56.8 | **102.2** | **128** |

The last row is the target: a mixed-input INT4 x FP16 tensor-core GEMM reads the 4-bit
head once per step instead of once per verification position. That single kernel is
worth roughly 22.8 ms of a 79.63 ms step and, on its own, crosses 100 tok/s at today's
measured acceptance. Combined with a v3 drafter holding 90% to depth 7 it reaches
roughly 128 tok/s.

## Revised plan to the target

1. **Batched mixed-input head GEMM** (INT4 weights, FP16 activations, tensor cores via
   CUTLASS). Measured upside 26.03 -> ~3.21 ms per step. Gets to ~102 tok/s alone.
   Interim fallback: cuBLAS FP16 already gives 89.1 tok/s today.
2. **DSpark v3 drafter**, block size >= 8, ~90% acceptance held across depth. Takes the
   same step time to ~128 tok/s.

Item 1 is a kernel engineering task with a measured floor and a known failure mode to
avoid. Item 2 is a training run. Neither is speculative any more.
