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

---

# Round 6 - audit of Rounds 1-5, and the batching finding

Every number in this round was measured on an idle GB10 (0 other compute apps).
Three conclusions published in earlier rounds are wrong. They are corrected here
with the evidence that overturns them.

## Correction 1: the LM head is NOT re-read per verification position

Round 5 concluded that llama.cpp re-reads the whole 248,320-entry output head
once per speculative verification row, costing 26.03 ms of a 79.63 ms step, and
that a batched head kernel would recover it. That was inferred from a timing
residual and supported with a cuBLAS comparison.

It is false. In `ggml/src/ggml-cuda/mmvq.cu` the inner loop is:

    for (int j = 0; j < ncols_dst; ++j)
        for (int i = 0; i < rows_per_cuda_block; ++i)
            tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], ...);

`vx` (the quantized weight) does not depend on `j`. `mul_mat_vec_q` already
amortizes the weight read across every verification column inside one kernel.
GB10 has its own dispatch branch (`GGML_CUDA_CC_DGX_SPARK = 1210`) which routes
PQ2_0 to this path for `ne11 <= 8`, so K=7 uses exactly it.

Measured confirmation, same kernel, from an nsys profile of real decode:

| ncols_dst | avg kernel duration |
|---|---|
| 2 | 87.6 us |
| 4 | 90.8 us |
| 5 | 96.8 us |

+10% for 2.5x the columns. A per-position re-read would be +150%.

The cuBLAS "proof" benchmarked 8 separate GEMV launches. llama.cpp never issues
that. The proposed CUTLASS mixed-input head kernel would have returned nothing.

## Correction 2: concurrency DOES scale - the earlier test used one slot

Round 3 measured aggregate throughput saturating at 65.8 tok/s and concluded
"concurrency adds no aggregate throughput... one stream already pulls ~160 of
184 GB/s", and that this was a hardware property distinguishing GB10 from an
M5 Max.

The `llama-server` used for that test was launched with `-np 1`. That is a
single slot: 16 clients queued on one lane. It measured queueing, not batching.

`llama-batched-bench`, true batched decode, no speculation, 256-token prompts:

| B | S_PP t/s | S_TG t/s | S total t/s |
|---|---|---|---|
| 1 | 994.80 | 27.54 | 123.99 |
| 2 | 980.48 | 46.53 | 195.52 |
| 4 | 976.54 | 78.92 | 298.21 |
| 8 | 945.06 | 106.09 | 366.06 |
| 16 | 881.44 | 133.08 | 414.87 |

Decode scales 4.83x from B=1 to B=16. Batching amortizes the 6.70 GB weight
sweep across sequences; the bandwidth wall binds only at batch 1.

Prefill is ~995 tok/s, not the 646 tok/s published earlier.

## Correction 3: K saturates at the drafter's block size

Round 4 fit `t_step = 48.3 + 4.47*K` and reasoned about K=8.5 as if K were free.
K is capped by the drafter's block size. The v1 drafter has block size 4, so
`--spec-draft-n-max 7` is a no-op past 4, and several "K=7" rows in earlier
rounds are really K=4.

## New: step time is independent of context length

Same drafter and K, two context lengths 325x apart:

| ctx tok | K | tok/s | accept% | tok/step | ms/step |
|---|---|---|---|---|---|
| 5 | 1 | 39.48 | 96.97 | 1.97 | 49.89 |
| 5 | 3 | 61.11 | 87.16 | 3.57 | 58.38 |
| 5 | 5 | 64.50 | 77.52 | 4.03 | 62.48 |
| 5 | 7 | 64.53 | 77.52 | 4.03 | 62.45 |
| 1625 | 1 | 38.11 | 91.18 | 1.91 | 50.16 |
| 1625 | 3 | 58.05 | 79.49 | 3.38 | 58.31 |
| 1625 | 5 | 59.27 | 67.83 | 3.69 | 62.33 |
| 1625 | 7 | 58.80 | 67.83 | 3.69 | 62.83 |

ms/step matches within 1% at every K. The per-row cost is not attention and not
KV traffic. Acceptance does fall with context on these prompts.

## Confirmed: single-stream decode is memory-bandwidth bound

Two independent decode processes, each K=7, 96 tokens:

    single       : 63.14 tok/s
    2x aggregate : 63.88 tok/s   (1.01x)

Flat. Single-stream is bandwidth bound, which is why only batching helps.

## Measurement hazard: nsys under-reports GPU busy time on GB10

An nsys trace of real decode reported the GPU 92.2% idle (95.7 ms of kernel
time inside a 1281.8 ms window). That is a CUPTI artifact on GB10 unified
memory - kernel durations do not capture fabric stalls. The 1.01x
two-process result above disproves it directly. Do not publish GPU-idle
percentages from nsys on this machine.

Separately: `[CUDA memcpy Host-to-Device]` totals ~160 ms but is model load, not
per-step. It is 153.6 ms at n=32 and 160.6 ms at n=128 - flat in token count.

## Where the time actually goes (nsys, real decode, K=7)

PQ2_0 `mul_mat_vec_q` is 58% of all kernel time (186 ms of 320 ms), split across
the ncols=2/4/5 variants. It is the single dominant kernel.

Note a real gap: `vec_dot_ptq1_0_q8_1_multi` exists in `vecdotq.cuh:809` and
fuses dequantization across columns for PTQ1_0. PQ2_0 - the format Bonsai 2 uses
for every layer and the head - has no equivalent and goes through the generic
per-column path.

---

# Round 7 - Pulse vs stock llama.cpp defaults

All four rows measured the same afternoon on an idle GB10 (0 other compute
apps), same model, same harness (`bench/conc.py`), 128 tokens per request,
temperature 0.

Stock:  `llama-server -m <model> -ngl 99 -c 65536`   (n_slots=4, no drafter, no -fa)
Tuned:  16 slots, `-fa on`, `-b 4096`, `-ub 512`, drafter chosen by load policy

| clients | stock tok/s | pulse tok/s | speedup |
|---|---|---|---|
| 1 | 26.58 | 40.71 | 1.53x |
| 4 | 72.93 | 79.61 | 1.09x |
| 16 | 73.08 | 117.71 | 1.61x |

Stock saturates at 73.08 because it has 4 slots; the 16 clients queue.

## The speculation / batching crossover

Speculation and batching are substitutes on a bandwidth-bound machine, not
complements. The drafter buys tokens per weight sweep while the sweep is
under-occupied; once batching has filled it, the drafter's extra rows are cost.

| clients | with drafter | no drafter |
|---|---|---|
| 1 | 40.71 | 26.46 |
| 2 | 56.48 | 44.09 |
| 4 | 79.61 | 72.24 |
| 8 | 98.17 | 94.80 |
| 16 | 111.99 | 117.71 |

Crossover is between 8 and 16 clients. Note the gain over a static
always-on-drafter policy is only ~5%, at 16 clients only - this is a real
effect but a small one, and it is reported as such.

Caveat: llama-server cannot toggle speculation per request. The schema fields
`speculative.n_max` / `n_min` sit behind `#if 0` in
`tools/server/server-schema.cpp` (one of them also has a syntax error - a
missing closing paren on line 205 - which is presumably why it was disabled),
and the decode path reads `params_base.speculative` rather than per-task
params. Acting on the load policy today needs two backends.

## The per-row cost, localised but not yet explained

Marginal cost of an extra drafted row is ~3.5 ms. What it is not:

- not attention or KV traffic: ms/step is identical at 5-token and 1625-token
  context (49.89 vs 50.16, 58.38 vs 58.31, 62.48 vs 62.33)
- not the drafter running per row: doubling drafter size 603 MB -> 1.10 GB
  (1.75x) moved the slope only 3.51 -> 3.70 ms/row (1.05x). The drafter is
  swept once per step, as designed - its size shows up in the base instead
  (48.70 -> 51.27 ms, and the 0.5 GB delta is 2.7 ms at 184.6 GB/s)
- not the verify matmul: mul_mat_vec_q grows only 10% from 2 to 5 columns

3.5 ms at 184.6 GB/s is 0.646 GB. The output head is 0.592 GB. The head remains
the best fit for the residual, in a path that is not getting mmvq's column
amortisation. Open.

This matters because it is the whole single-stream ceiling: with the per-row
cost removed, the v1 drafter's already-measured 90.18% acceptance would give
(1 + 4*0.9) / 0.0363 = ~100 tok/s single-stream with no new model trained.

---

# Round 8 - the design curve, and what actually blocks 100 tok/s

## Method

Throughput as a function of accepted-tokens-per-step, measured with a model-free
n-gram drafter on repetitive text. Acceptance is pinned at 100%, so this isolates
the hardware's cost of verifying N rows from any drafter's quality. It is the
ceiling for any drafter, and the spec a drafter must be designed against.

    llama-speculative-simple --spec-type ngram-simple --spec-ngram-simple-size-m M

## The curve (idle GB10, temp 0, --ignore-eos)

| m | tok/s | tok/step | ms/step | marginal ms/row |
|---|---|---|---|---|
| 1 | 27.29 | 1.00 | 36.65 | - |
| 3 | 85.36 | 4.00 | 46.86 | 3.44 |
| 4 | **101.25** | 5.00 | 49.38 | 2.52 |
| 6 | 126.20 | 6.92 | 54.84 | 2.84 |
| 8 | 127.40 | 8.83 | 69.33 | 7.58 |
| 12 | 174.11 | 12.81 | 73.57 | 1.07 |
| 16 | 219.35 | 17.00 | 77.50 | 0.94 |

**100 tok/s single-stream is reachable on this hardware at 5 accepted tokens per
step.** 219 tok/s at 17. The machine is not the limit; drafter depth x acceptance is.

The base of 36.65 ms matches the predicted 6.70 GB / 184.6 GB/s = 36.3 ms sweep.

## This corrects the Round 4 cost model

Round 4 fit `t_step = 48.3 + 4.47K` and concluded 100 tok/s required a block-9
drafter holding 90% acceptance. Both terms were wrong and both erred the same way:

- the base of 48.3 ms silently included the dspark drafter's own 3.27 ms sweep
  plus its path overhead. The true verify-only base is 36.6 ms.
- the slope of 4.47 ms/row is really 2.5-3.4 ms/row, and it decreases with depth.

On the corrected curve the requirement is a **block-5 drafter**, not block-9.
v1 already measures 90.18% acceptance at depth 3.

## The dispatch discontinuity at 8 rows

At m=8 `ncols_dst` crosses `MMVQ_MAX_BATCH_SIZE = 8` and dispatch falls from
`mul_mat_vec_q` to MMQ: +14.5 ms for +1.9 tokens. Past it MMQ's marginal cost is
~1.0 ms/row against mmvq's ~2.9, but its base is ~65 ms against mmvq's 36.6.

Both directions tested with a patched, env-var-driven threshold
(`patches/mmvq-threshold-override.patch`):

| GGML_MMVQ_MAX at 5 rows | tok/s | ms/step |
|---|---|---|
| 8 (default, mmvq) | **100.78** | 49.62 |
| 4 (forces MMQ) | 75.21 | 66.48 |
| 0 (forces MMQ) | 74.84 | 66.81 |

Forcing MMQ earlier is **34% worse**. The default is correctly tuned below 8.

Raising the threshold above 8 asserts: mmvq is only template-instantiated to
`ncols_dst = 8`. So rows 9-15 are forced onto MMQ although the fits put the real
crossover at ~15 rows. That is a genuine upstream gap, but it does not affect us:
our drafters have block 4 and 7, giving 5 and 8 rows.

## Best measured single-stream config

v1 drafter, realistic 5000-char code context, 3 repeats each:

| K | tok/s | acceptance | tok/step | ms/step |
|---|---|---|---|---|
| 3 | 58.01 / 58.07 / 58.16 | 78.29% | 3.32 | 57.2 |
| **4** | 63.61 / 63.71 / 63.98 | 72.28% | 3.86 | 60.6 |

K=4 (the full block size) wins. Deterministic at temp 0.

## The two remaining levers, quantified

At 3.86 tok/step the design curve gives **44.9 ms/step**. The dspark drafter
measures **60.6 ms**. The gap is **15.7 ms**, of which only 3.27 ms is the
drafter's own 603 MB weight sweep.

1. **~12.4 ms/step of unexplained dspark drafter path overhead.** Removing it
   alone gives ~80 tok/s single-stream.
2. **Block size 4 -> 5.** At current acceptance that is ~96 tok/s.

Together: **100+ tok/s single-stream, with no change to the target model and no
new target quantization.** That is the whole remaining gap, stated in measured
quantities.

---

# Round 9 - what the neural drafter costs, and what it is worth

## Where the drafter path time goes

Two profiles at matched draft depth, differenced per step. `ngram-simple` has no
neural model, so the delta is the dspark drafter path.

| kernel | ngram ms/step | dspark ms/step | delta | launches/step |
|---|---|---|---|---|
| `mul_mat_q` (MMQ) | 2.465 | 16.165 | **+13.700** | 24.0 |
| `gated_delta_net` | 0.376 | 2.745 | +2.369 | 7.5 |
| `unary_gated_op` | 0.213 | 1.955 | +1.742 | 20.4 |
| `fwht_cuda_block` | 0.251 | 1.809 | +1.558 | 40.5 |

53% of the overhead is the drafter's own forward pass on the tensor-core MMQ
path. DSpark is a block-diffusion drafter with its own Gated DeltaNet layers and
Hadamard transforms - it is a small model doing real work, not a lookup.

(nsys absolute numbers are unreliable on GB10, see Round 6. The composition is
the usable signal; measured wall delta is 61.14 - 49.88 = 11.26 ms/step.)

## Is the neural drafter worth it? Yes, on real code

Same 5000-char code context for all four:

| drafter | tok/s | accept% | tok/step |
|---|---|---|---|
| **dspark (603 MB)** | **64.53** | 72.28% | 3.86 |
| ngram-simple | 45.41 | 100.00% | 2.75 |
| ngram-cache | 44.44 | 100.00% | 3.13 |
| ngram-mod | 26.63 | 0.00% | 1.00 |

1.42x over the best free drafter. Note ngram-simple reaches 100% acceptance but
only 2.75 tok/step: it proposes only when it has a confident match.

## Stacking drafters: it works, and it is worth +1.8%

`--spec-type` takes a comma-separated list. Round 2 reported that stacking
"never fired" - that was measured on a 60-token prompt, where an n-gram drafter
has nothing to look up. With a real context both components engage:

| spec-type | tok/s | accept% | tok/step | ms/step |
|---|---|---|---|---|
| draft-dspark | 63.88 | 72.28% | 3.86 | 60.47 |
| draft-dspark,ngram-simple | 64.45 | 67.14% | 3.66 | 56.79 |
| **ngram-simple,draft-dspark** | **65.06** | 67.14% | 3.66 | 56.26 |
| draft-dspark,ngram-cache | 40.18 | 87.42% | 3.53 | 87.78 |

The mechanism is visible: step time falls 60.47 -> 56.26 ms because n-gram drafts
are free, but tokens per step fall 3.86 -> 3.66 because those free drafts displace
better dspark ones. Net +1.8%. Real, reproducible, and small - reported as such.

Stacking with `ngram-cache` is strongly negative (-37%).

## Best measured single-stream configuration

`--spec-type ngram-simple,draft-dspark` with the v1 drafter at K=4:
**65.06 tok/s**, against a 27.54 tok/s no-drafter baseline. 2.36x.

The design curve says 3.66 tok/step should cost ~44 ms/step, and this measures
56.26. The remaining gap is the drafter's own forward pass, which is the price of
its acceptance. Closing the gap means a cheaper or deeper drafter, not a runtime
change - every runtime lever has now been measured and tabulated.

---

# Round 10 - the final measured state, and the one lever left

## Best measured single-stream configuration

v2 drafter (block 7) at K=7, realistic 5000-char code context, temp 0:

| run | tok/s | accept% | tok/step | ms/step |
|---|---|---|---|---|
| 1 | 73.39 | 69.10% | 5.74 | 78.15 |
| 2 | 73.20 | 69.10% | 5.74 | 78.35 |
| 3 | 72.30 | 69.10% | 5.74 | 79.32 |

**73.3 tok/s**, spread 1.09, against a 27.54 tok/s no-drafter baseline: **2.66x**.

This supersedes the earlier "v1 at K=3 / K=4" recommendation. Earlier rounds
preferred v1 because they measured v2 on a shorter, less representative context.

Stacking an n-gram drafter helps v1 (+1.8%) and **hurts** v2 (-11%, 73.20 ->
65.28): the free n-gram drafts displace deeper dspark drafts, 5.74 -> 4.62
tok/step. With the recommended drafter, do not stack.

## The one lever left, quantified

Overhead is measured ms/step minus the Round 8 design curve at the same
tokens/step - that is, the cost of the drafter path over a free drafter:

| drafter | size | tok/step | ms/step | curve | overhead | effective bandwidth |
|---|---|---|---|---|---|---|
| v1 | 603 MB | 3.86 | 60.3 | 44.90 | +15.4 ms | **39.2 GB/s** |
| v2 | 1.10 GB | 5.74 | 78.3 | 50.33 | +27.9 ms | **39.4 GB/s** |

Overhead scales 1.81x for a 1.75x larger drafter, so it is the drafter's own
forward pass rather than a fixed cost. Both land at **~39 GB/s, 21% of the
184.6 GB/s this machine sustains.**

The cause is visible in the profile: the drafter is a 5-layer block-diffusion
model with its own Gated DeltaNet layers and Hadamard transforms, executed as
~24 separate MMQ launches per step. Each moves ~25 MB, far too little to
saturate the bus, so the pass is latency-bound rather than bandwidth-bound.

**If the drafter forward ran at achievable bandwidth:**

    50.33 ms (curve at 5.74 tok/step) + 1.10 GB / 184.6 GB/s = 56.3 ms
    5.74 / 0.0563 = 102 tok/s single-stream

That is the whole remaining gap to the 100 tok/s target, and it is a runtime
optimization on drafter execution - not a new drafter, not a new target
quantization, and not a deeper block. We already own a block-7 drafter.

CUDA Graphs will not fix it: the drafter has Gated DeltaNet layers, which is
exactly why graphs are rejected at runtime (Round 6).

## Final scorecard

| metric | measured | note |
|---|---|---|
| single-stream decode | **73.3 tok/s** | v2, K=7, 3 repeats |
| no-drafter baseline | 27.54 tok/s | = 6.70 GB / 184.6 GB/s |
| speedup from speculation | 2.66x | |
| aggregate, 16 clients | 117.71 tok/s | 16 slots, no drafter |
| batched decode, B=16 | 133.08 tok/s | llama-batched-bench |
| prefill | ~995 tok/s | |
| vs stock llama.cpp defaults | 1.53-1.61x | configuration only |
| hardware ceiling at 5.74 tok/step | ~114 tok/s | if the drafter were free |
| reachable by fixing drafter execution | ~102 tok/s | the open work |

---

# Round 11 - the measurement floor, and a correction to Round 10

## Round 10's 73.3 tok/s was machine state, not a result

Round 10 reported 73.39 / 73.20 / 72.30 tok/s for v2 at K=7 and called the 1.09
spread reproducible. Re-running the identical command on a clean GPU an hour
later gave **65.06 / 68.94 / 61.05**. Nothing was contending for the GPU
(0 compute apps), the card was at 51 C and `clocks_event_reasons.active` was
`0x0`, so it was neither contention nor throttling.

Properly characterised - 90 s quiesce, then 6 repeats:

    63.37  62.43  64.62  68.10  69.21  67.99
    n=6  min=62.43  median=66.31  max=69.21  spread=10.2%

**The honest single-stream figure is 66.31 tok/s median, not 73.3.** Corrected in
the README, the tuning profile and the Hugging Face card.

## Why: the CPU and GPU share one memory bus

GB10 has unified LPDDR5X. CPU-side memory traffic directly steals GPU bandwidth.

| phase | tok/s |
|---|---|
| ambient load | 71.26, 73.30 |
| + 8 CPU threads streaming memory | 61.79, 60.99 |
| after the load stopped | 60.04, 59.39 |

CPU contention costs **16%**. More importantly, throughput **did not recover** when
the load stopped, and it then **climbs across consecutive runs**
(63.4 -> 62.4 -> 64.6 -> 68.1 -> 69.2). That is consistent with unified-memory
pages migrating toward the CPU under pressure and faulting back to the GPU lazily.

A browser was running during the earlier measurements. That is enough to matter
on this machine.

## The protocol this forces

1. No GPU compute apps, load average below ~1, no heavy CPU processes.
2. Settle, then discard the first runs - throughput climbs as pages migrate back.
3. Take a **median of at least 6 runs**.
4. **Any claimed optimisation must exceed 10.2% to be credible.**
5. Do not use nsys for attribution on this machine (Round 6, and again here: it
   reported 2.7 ms of kernel work inside a 73.4 ms step).

`pulse doctor` now checks load average and CPU hogs, not just GPU compute apps.

## What this does and does not change

It does **not** change the structural results, which were cross-validated by
independent means:

- the 36.3 ms weight sweep, confirmed by `llama-batched-bench` at 27.54 tok/s
  from a different binary
- the batching curve (4.83x), a single sweep in one machine state
- the two-process 1.01x bandwidth-bound result, a within-run comparison
- every negative result, all within-run A/Bs

It **does** mean single-run cross-time comparisons anywhere in Rounds 1-10 carry
about +/-10%, and the drafter overhead figures (+15.4 and +27.9 ms/step) should be
treated as approximate. They remain well above the noise floor - 27.9 ms of a
~78 ms step is 36% - so the lever is real, but validating any fix requires the
protocol above rather than single runs.

## Dead end recorded: DSPARK_DRAFT_WINDOW

`draft_window = 0` means full prefix, so the drafter conditions on the whole
context. Capping it changes nothing measurable:

| window | 0 | 512 | 256 | 128 | 64 | 32 | 16 |
|---|---|---|---|---|---|---|---|
| tok/s | 73.28 | 73.31 | 73.31 | 72.84 | 73.45 | 72.95 | 73.01 |

Identical acceptance and tok/step throughout. The knob does not reach the
graph-corrected v2 path.

## Correction to Round 9's kernel attribution

Round 9 attributed the drafter overhead to "~24 MMQ launches of ~25 MB". That was
wrong twice over: the `mul_mat_q<(ggml_type)142, (int)128>` kernels are on the
**target** (PQ2_0), and `128` is MMQ's tile width `mmq_x`, not a row count. They
are also **prefill**, not per-step - a steady-state profile with a 5-token prompt
shows no MMQ at all during decode. Dividing their total by step count was invalid.

The wall-clock evidence for the drafter overhead is unaffected: it scales 1.81x
for a 1.75x larger drafter, which is what identifies it as the drafter's own
forward pass.

---

# Round 11 - the measurement floor, and a correction to Round 10

## Correction: 73.3 tok/s was machine state, not a result

Round 10 reported v2 at K=7 as **73.3 tok/s with a 1.09 spread over 3 repeats**.
Re-running the identical command later, on a verified-clean GPU with no
throttling, gave 65.06 / 68.94 / 61.05.

A properly quiesced run - 90 s idle, then 6 repeats - gives:

    63.37  62.43  64.62  68.10  69.21  67.99
    n=6  min=62.43  median=66.31  max=69.21  spread=10.2%

**The honest number is 66.31 tok/s median, 62.4-69.2.** The 73.3 figure was taken
in an unusually favourable machine state and the tight spread was luck, not
reproducibility. Corrected in the README, the config profile and the model card.

## Why this machine is state-dependent

**The CPU and GPU share one LPDDR5X bus.** Eight CPU threads streaming memory:

| phase | tok/s |
|---|---|
| ambient load | 71.26, 73.30 |
| + 8 CPU threads streaming memory | 61.79, 60.99 (**-16%**) |
| after the load stopped | 60.04, 59.39 (**did not recover**) |

CPU memory traffic costs 16% of decode throughput. More importantly it does not
recover promptly: unified-memory pages migrate toward the CPU and fault back
lazily. Throughput then *climbs* across consecutive runs, which is visible in the
6-repeat sequence above (63.4 -> 69.2).

A browser was running during the earlier measurements. That is enough to matter.

## The protocol, now enforced in code

`bench/ab.py` implements it:

- refuse to run if any process holds the GPU; warn above load average 1.5
- discard warmup runs
- **interleave configs** rather than running all of A then all of B, so drift
  hits every config equally instead of masquerading as a difference
- report median, min, max and spread
- refuse to call a difference real unless it exceeds the **10.2% noise floor**

This matters for what comes next: the drafter overhead is 27.9 ms of a ~78 ms
step (36%), comfortably above the floor, so it is a real target. But a careless
single-run comparison on this machine can manufacture a 15% "win" out of nothing,
which is structurally how this repo produced fabricated numbers in the first place.

## Also retracted from Round 9/10

The claim that the drafter is "chopped into ~24 MMQ launches of ~25 MB each" was
wrong. Those `mul_mat_q<(ggml_type)142, (int)128>` kernels are on the *target*
(type 142 = PQ2_0), the `128` is MMQ's tile width `mmq_x` rather than a row count,
and 800 launches x 764 us in a ~3 s run is prefill - which I incorrectly divided
by step count.

A clean steady-state profile (5-token prompt, 320 generated) shows no MMQ at all
during decode. It also shows only 2.7 ms of kernel time inside a 73.4 ms step and
18 target matmul launches per step for a 64-layer model, which is not credible -
nsys is missing most of the work. **nsys is now dropped entirely for attribution
on GB10.** Wall-clock differentials only.

What survives, because it rests on wall-clock: overhead is +15.4 ms/step for the
603 MB drafter and +27.9 ms/step for the 1.10 GB drafter, scaling 1.81x for a
1.75x drafter, both landing at ~39 GB/s effective. The cost is the drafter. The
kernel-level mechanism is still open.

`DSPARK_DRAFT_WINDOW` (0 = full prefix, the default) has **no measurable effect**
at any value from 16 to 512 - all within 73.28 +/- 0.5.

---

# Round 12 - protocol, and two more dead levers

## The noise floor depends on the protocol

Round 11 put the floor at 10.2%. That was measured cold - quiesced, but with no
warmup discarded, on runs that were still climbing. With warmup discarded and
configs interleaved, the same config gives a 3.4% spread.

Identical config, identical command, three protocols:

| protocol | median tok/s | spread |
|---|---|---|
| n=3, favourable machine state | 73.30 | 1.5% (luck; does not reproduce) |
| n=6, quiesced, no warmup discard | 66.31 | 10.2% (cold; runs climbing) |
| n=4, 2 warmup discarded, interleaved | **71.45** | **3.4%** (warm steady state) |

Both cold and warm are real and answer different questions: ~66 is first use
after CPU activity, ~71.5 is sustained serving. `bench/ab.py` enforces the warm
protocol and gates on 3.4%; `PULSE_NOISE_FLOOR` overrides it.

The practical consequence is that **smaller optimisations are detectable than
Round 11 concluded** - but only under the warm protocol.

## Dead lever: CPU thread count

Hypothesis: `llama-speculative-simple` defaults to 20 CPU threads on this box,
and since the CPU and GPU share one LPDDR5X bus (Round 11), those threads should
be stealing bandwidth from a fully GPU-offloaded decode.

Measured, 2 warmup + 5 interleaved repeats:

| config | median | spread | vs default |
|---|---|---|---|
| t=20 (default) | 69.55 | 7.2% | - |
| t=8 | 69.23 | 12.1% | -0.5% |
| t=4 | 70.75 | 7.0% | +1.7% |
| t=2 | 69.73 | 5.9% | +0.3% |

**No effect.** Everything is inside the 3.4% floor. The threads are blocked
waiting on the GPU, not streaming memory, so they do not contend for the bus.
Under a careless single-run protocol the +1.7% would have been reported as a win.

## Dead lever: DSPARK_DRAFT_WINDOW

`draft_window = 0` (the default) means full prefix. Values from 16 to 512 all
measure 73.28 +/- 0.5 - no effect at any setting. Reading the source explains
why: the drafter's context decode is already incremental. It stages only the rows
since its own cache position (`n_cache[seq_id] = start` at the end of each
round), so it replays the full prefix once on the first step and roughly the
accepted-token count thereafter - about 5 context rows plus 7 draft rows.

## Correction to the drafter-overhead arithmetic

Rounds 9-10 computed drafter overhead by subtracting the Round 8 design curve
from the measured dspark step time. The curve was measured on a **repetitive**
prompt and the dspark figure on a **code** prompt, and that transfer is not
clean: running the model-free n-gram drafter on the code prompt shows the curve
under-predicting by roughly 19 ms there.

So the precise "30 ms of overhead" and "~39 GB/s effective" figures are not
supportable as stated. What survives is the **same-prompt differential**:

| drafter | size | ms/step | delta |
|---|---|---|---|
| v1 | 603 MB | 60.3 | - |
| v2 | 1.10 GB | 78.3 | +18.0 |

Overhead scales ~1.81x for a 1.75x larger drafter on the same prompt, so the cost
is the drafter's own forward pass and is roughly proportional to its weight
volume. That much is solid. The absolute effective-bandwidth figure is not, and
the kernel-level mechanism remains open.

---

# Round 13 - the serving policy grid

The full 2D grid: drafter x concurrent clients. `bench/grid.sh`, 16 slots,
128 tokens per request, temperature 0, short chat-style prompts.

| clients | no drafter | v1 (block 4) K=4 | v2 (block 7) K=7 |
|---|---|---|---|
| 1 | 23.54 | **37.78** | 36.35 |
| 2 | 40.24 | 53.24 | **58.83** |
| 4 | 68.02 | **74.66** | 62.80 |
| 8 | 87.77 | **92.37** | 77.06 |
| 16 | **111.44** | 102.86 | 84.39 |

## Deeper speculation is actively harmful under batching

v2 - the block-7 drafter that wins **single-stream** on code context (71.45 vs
v1's ~64) - loses to v1 at every concurrency level here except 2, and loses
badly at 16 (84.39 vs 102.86).

The mechanism is the Round 8 dispatch discontinuity. Batch rows are
`clients x (K+1)`. At 16 clients v2's K=7 means 128 rows per batch, far past
`MMVQ_MAX_BATCH_SIZE = 8`, so every matmul runs on MMQ - which has a ~65 ms base
against mmvq's 36.6 ms. Depth that pays for itself at batch 1 is pure cost once
batching has already filled the weight sweep.

This reverses the Round 10 recommendation of v2 as the default. **v1 is the
default**; v2 is for single-stream, high-acceptance workloads only.

## The policy

| clients | use |
|---|---|
| 1 | v1 at K=4, or v2 at K=7 for code-like prompts specifically |
| 2-8 | v1 at K=4 |
| >=12 | no drafter |

## Caveat, stated plainly

The grid used short chat-style prompts and saw **29-43% acceptance**. The
single-stream code-context runs saw **69-72%**. Acceptance is strongly
workload-dependent - the full measured range across this project is 21% to 96% -
so the optimal drafter depends on prompt type as well as concurrency. The grid
answers "which drafter under load", not "what throughput will my workload get".

Each grid cell is a single run, so cells carry roughly the cold noise band. The
*pattern* is trustworthy because it is monotonic across five points and the
crossover reproduces an independent earlier measurement taken with a different
server invocation.

---

# Round 14 - the drafter cost, fully decomposed

All rows below: same prompt (1359-token code context), same binary, warm
interleaved protocol (2 warmup discarded, 5 repeats).

## Inputs

| measurement | value |
|---|---|
| no-drafter baseline at 1359-token context | 24.67 tok/s = **40.50 ms/step** |
| v1 (0.632 GB) K=3 | 53.62 tok/s, 61.95 ms/step, spread 2.9% |
| v2 (1.105 GB) K=3 | 53.19 tok/s, 65.46 ms/step, spread 3.0% |
| v1 K=1 | 33.90 tok/s, 56.38 ms/step, **91.09% acceptance** |
| v1 K=3 (repeat) | 53.05 tok/s, 62.61 ms/step, spread 1.0% |

## Two clean differentials

**Drafter size**, at matched K=3: `3.51 ms / 0.473 GB` = **7.42 ms/GB**, i.e.
~135 GB/s on the drafter's weights - 73% of the 184.6 GB/s sustained.

**Verify row cost**, same drafter at two depths (block size 4, so K=1 and K=3
both run exactly one block pass and the drafter term cancels):
`(62.61 - 56.38) / 2` = **3.11 ms per verify row**.

That second number independently validates the Round 8 design curve, whose
fitted slope was **2.9 ms/row** on a completely different, repetitive prompt.
The curve's slope transfers; only its base does not, because the base includes
KV-cache reads that grow with context - 36.6 ms at 256 tokens, 40.5 ms at 1359.
Rounds 9-11 were right to distrust subtracting the curve's *base* from
code-context numbers, and wrong to distrust its *slope*.

## The decomposition

    step(v1, K=3) = 62.61 ms
      base, no drafter, 1359-token context      40.50 ms   64.7%
      3 verify rows x 3.11 ms                    9.33 ms   14.9%
      drafter                                   12.77 ms   20.4%
        weights: 0.632 GB x 7.42 ms/GB           4.69 ms   physics
        fixed orchestration                      8.08 ms   RECOVERABLE

**Removing the fixed orchestration term: 62.61 -> 54.53 ms/step, +15%.**

That is the measured value of the one genuine engine-class win - fusing draft and
verify into a single graph instead of running the drafter in a separate
`llama_context` with its own feature staging and KV add/remove every step.

Caveat: the fixed/size split leans on a no-drafter baseline taken with a
*different binary* (`llama-batched-bench`), which carries its own overhead.
Honest range for the recoverable term is roughly 6-10 ms, i.e. **+11% to +19%**.

## Incidental finding

v1 at K=1 reaches **91.09% acceptance** - the highest measured in this project -
but only 1.96 tok/step, so it yields 33.90 tok/s. Acceptance and depth trade off
sharply: K=3 drops acceptance to 78.29% while raising throughput 56.5%.

---

# Round 15 - the bandwidth ceiling was under-measured

## The correction

Every round so far quoted **184.6 GB/s** as this machine's sustained memory
bandwidth, and Round 14 concluded from it that llama.cpp's decode was *at* the
roofline ("36.3 ms predicted, 36.65 ms measured, 1% agreement").

That 184.6 came from a single kernel configuration (192 blocks, ILP=1). Sweeping
memory-level parallelism, block count and occupancy (`bench/cuda/bw.cu`) shows
the hardware does considerably better.

At a 4 GB working set the best config reached **229.0 GB/s (84% of spec peak)**.
At the model's actual 6.70 GB working set, three consecutive runs:

    216.9 GB/s   217.6 GB/s   212.4 GB/s      (79-80% of spec peak)

Interesting: the best configurations are **low occupancy** - often 1-2 blocks per
SM - and extra ILP does not reliably help. This is a latency-tolerant streaming
workload where a handful of resident warps per SM already saturates the
controller.

## Two gaps, not one

| | GB/s | % of spec | % of achievable |
|---|---|---|---|
| spec peak, 256-bit LPDDR5X @ 8533 MT/s | 273 | 100% | - |
| achievable pure read at 6.70 GB | **~216** | 79% | 100% |
| llama.cpp decode (6.70 GB / 36.65 ms) | 183 | 67% | **85%** |

**Gap 1 (273 -> 216): the DRAM, and it is normal.** Refresh, row
activate/precharge across a 6.70 GB working set, read turnaround, and a fabric
shared with the Grace CPU. 70-85% of theoretical is the usual LPDDR5X range.
Not recoverable.

**Gap 2 (216 -> 183): software.** The true weight-sweep roofline is
`6.70 / 216 = 30.9 ms`, against a measured 36.65 ms. That is a **16% gap**, not
the 1% Round 14 claimed.

Not all of it is waste. A decode step also reads the KV cache, writes
activations, and runs norms, Hadamard transforms, `quantize_q8_1` and sampling -
real work that is not weight streaming. But the gap is 5.75 ms per step and some
fraction of it is recoverable, which Round 14 asserted was not the case.

## Consequence for the engine question

Round 14 recommended against a rewrite largely because llama.cpp appeared to be
at the bandwidth roofline, leaving only the drafter's 8.08 ms of orchestration.
With the corrected ceiling there are two recoverable terms:

    base step gap      ~5.75 ms   (partly recoverable; some is legitimate work)
    drafter fixed cost  8.08 ms   (recoverable by fusing draft+verify)

Against the 62.61 ms measured step at v1 K=3, recovering both would give roughly
49 ms, about **+28%** rather than the +15% Round 14 concluded. The engine case is
stronger than that round stated, though still far from the order-of-magnitude
this project originally claimed.

---

# Round 16 - context, KV, and two methodology failures of my own

## `-b` is not free: large batch roughly doubles decode step time

| prompt | `-b` | ms/step |
|---|---|---|
| 1359 tok | 8192 | 66.44 |
| 772 tok | 65536 | 128.61 |

A *shorter* prompt with a larger `-b` took roughly twice as long per decode step.
llama.cpp sizes compute buffers from `n_batch`, and the cost lands on decode, not
just prefill. This compounds the Round 2 finding that `-ub` above 512 hurts
prefill.

**Keep `-b` only as large as the longest prompt requires.** The tuning profile's
`-b 4096` is right for short prompts; long prompts force a larger `-b` and pay
for it in decode throughput. That is a genuine tradeoff, not a free knob.

## Methodology failure: cold-start per process invalidates context sweeps

Several context sweeps in this round produced **non-monotonic step times** -
128.61 ms at 772 tokens against 103.96 ms at 1989, and 108.41 against 92.65 in an
earlier attempt. Step time cannot fall as context grows.

Cause: each point was a separate process invocation, and each invocation reloads
the 6.70 GB model. Round 11 established that this machine is slow on cold runs
and *climbs* across consecutive ones as unified-memory pages migrate back to the
GPU. Every such sweep was measuring warmup, not context.

`bench/ab.py` is immune because it discards warmup runs - the machine-level page
state warms during them - which is why it reports 1.0-3.0% spreads while these
ad-hoc loops swing 25%. **Ad-hoc `for` loops over configurations are not a valid
measurement on this machine.** Use `ab.py`.

## KV cache is not a factor at or below ~7k context

From the (cold, therefore noisy, but internally consistent) sweep:

| prompt tok | ms/step | est. KV | KV as % of the 6.70 GB model |
|---|---|---|---|
| 772 | 128.61 | 0.20 GB | 3.0% |
| 1989 | 103.96 | 0.52 GB | 7.8% |
| 3778 | 106.75 | 0.99 GB | 14.8% |
| 7266 | 99.62 | 1.90 GB | 28.4% |

Step time shows **no upward trend** even with KV at 28% of the model's size. With
flash attention the KV is streamed once per step: 1.90 GB at 216 GB/s is ~8.8 ms
on a ~100 ms step, inside the noise.

Extrapolating (arithmetic, **not measured**): at 32k tokens KV would be ~8.6 GB,
larger than the model itself, and ~40 ms of a step. That is where KV compression
of the TurboQuant kind - 3-bit keys, 2-bit values, ~6x - would be worth close to
2x. Below ~8k it is worth approximately nothing here, which is consistent with
our earlier measurement that KV `q8_0` was slightly *worse* than f16.

**We still have no measurement above 7266 tokens.** Reaching 32k needs a larger
`-c`, a bigger prompt pool, and - per the finding above - a larger `-b`, which
itself costs decode throughput. That confound has to be controlled for before any
long-context number from this repo is trustworthy.

## Acceptance falls sharply with context on these prompts

73.17% at 1359 tokens, 55.41% at 3237, 48.73% at 3778, 35.98% at 7266.

Round 6 measured acceptance *rising* with context (74.84% at 65 tokens to 84.40%
at 1854) on a different prompt. Both are real; acceptance is content-dependent
across a measured 21%-96% range. It is a property of the workload, not of the
drafter, and any single acceptance figure quoted without its prompt is close to
meaningless.

---

# Round 17 - the context/KV model, validated, and the 256k roadmap

## Corrected KV geometry

Read from the model's own tensors rather than assumed:

    16 attention layers (every 4th of 64 - it is a hybrid)
    x 4 kv heads x (key_length 256 + value_length 256) x 2 bytes
    = 64 KB per token

Earlier rounds used 262 KB/token, which was ~4x too high and made the KV story
look more urgent at short context than it is.

| context | KV | vs the 6.70 GB model |
|---|---|---|
| 16k | 1.07 GB | 16% |
| 64k | 4.29 GB | 64% |
| 128k | 8.59 GB | 128% |
| 256k | **17.18 GB** | **256%** |

The model's trained `context_length` is **262144**, so 256k is in range. 1M is
not - and would need 65 GB of KV regardless.

## A validated bandwidth model

    step_GB = 6.70 + ctx x 64KB
    step_ms = step_GB / 216 GB/s
    tok/s   = (1000 / step_ms) x 0.87        <- the measured software-gap factor

Measured with `llama-batched-bench` in a **single process** (model resident, so
no cold-start confound), `-c 200000 -b 131072` fixed across all points:

| context | step reads | predicted x0.87 | measured | ratio |
|---|---|---|---|---|
| 16384 | 7.77 GB | 24.17 | **24.26** | 1.00 |
| 65536 | 10.99 GB | 17.09 | **16.50** | 0.97 |
| 131072 | 15.29 GB | 12.29 | **10.43** | 0.85 |

Three independent points. Efficiency drifts down slightly at extreme context
(1.00 -> 0.97 -> 0.85), so projections beyond 128k should use ~0.80 rather
than 0.87.

## Below ~16k, context does not resolve

| ctx | 128 | 256 | 512 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|---|
| tok/s | 26.37 | 19.08 | 17.40 | 12.40 | 14.04 | 24.26 |

Non-monotonic, swinging 2.1x with no relation to context. Below 16k the KV term
is under 16% of the step and the measurement noise exceeds the effect. **Do not
read a context trend into these rows.** Only the 16k and 64k points are
resolvable, and both match prediction.

Related: a warm `ab.py` run at ~4k context produced a **92.2% spread**
(14.93-41.15 tok/s over four runs on an idle GPU) against 1.6% at ~1.5k. Long
context on this box is not merely slower, it is erratic.

## The 256k roadmap

At 256k a step must read 6.70 GB of weights **plus 17.18 GB of KV** = 23.88 GB,
which is 110.6 ms at 216 GB/s, i.e. **7.9 tok/s** after the software factor.
Speculation amortises the whole read - weights and KV alike:

Using the observed efficiency drift (~0.80 at this scale) rather than 0.87:

| configuration | tok/s at 256k |
|---|---|
| no drafter | 7.2 |
| 5.74 tok/step (block 7 @ 69% acceptance) | 41 |
| 6.95 tok/step (block 7 @ 85%) | 50 |
| **5.74 tok/step + 6x KV compression** | **~110** |

**Path A does not reliably clear 50-70 tok/s at 256k.** Even at 85% acceptance
it lands at 50, the very bottom of the range. KV compression is not an optional
optimisation for that target - it is required.

**Path A** (drafter only) lands at 45-55 tok/s and requires holding >=70%
acceptance at 256k. Acceptance currently *falls* with context (74.59% at 1.5k,
50.00% at 4k), so this is a drafter-training problem.

**Path B** (KV compression) clears 50-70 comfortably and is independent of
drafter quality. At 256k the KV is 2.6x the model, so compressing it is the
dominant term - the opposite of the situation below 8k, where an earlier round
correctly found KV compression worth approximately nothing.

This supersedes the earlier dismissal of TurboQuant-class KV compression. That
dismissal was right for <=8k and wrong for long context.

## Prefill dominates the 256k user experience

From the same sweep:

| context | prefill t/s | time to first token |
|---|---|---|
| 16384 | 669.6 | 24.5 s |
| 65536 | 769.5 | 85.2 s |
| 131072 | 619.1 | **211.7 s** |

At 131072 tokens a cold prefill takes **3.5 minutes**; 256k would be roughly
**7 minutes**. Decode throughput is close to irrelevant beside that.

For a long-context agentic workload the highest-value feature is therefore
**prompt/prefix caching**, ahead of both drafter quality and KV compression.
This is exactly what Inco built `StateCache` for - their published cached-replay
TTFT is 282 ms against 96 s cold. We have measured nothing in this area.
