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
