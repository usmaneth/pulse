# Historical README snapshot

This file preserves the README before the September 2026 integration review.
Its numbers and claims are historical evidence, not current qualification results.
The full model file size is not the weight traffic per decode step.
Prompt changes and tool schema changes can prevent prefix reuse.
Read the current README and linked validation documents before use.

---

# Pulse

**Measurement-driven speculative decoding for NVIDIA DGX Spark (GB10).**

Pulse is not an inference engine and does not replace llama.cpp. It is two things:

1. A **benchmark suite** for speculative decoding on GB10, built after a previous
   iteration of this repo published numbers that turned out to be invented. Every
   figure below was measured on a quiesced GPU under a stated protocol
   (see *Measurement hygiene*) and is reproducible via `bench/reproduce.sh`.
2. A **tuning profile plus launcher** that starts llama.cpp with the configuration
   measured to be best on this hardware, instead of making you rediscover it.

The negative results in [`bench/RESULTS.md`](bench/RESULTS.md) are the more useful
half. Several widely-assumed optimisations measurably do nothing here.

---

## The single biggest win: prefix caching (131.7x)

llama.cpp's server has supported `cache_prompt` all along. It was **off** — and
every benchmark in this repo before now measured the cold path exclusively.

Measured at 16k context (`bench/cache.py`):

| scenario | prefill | decode |
|---|---|---|
| cold (`cache_prompt=false`) | **22,321.6 ms** | 7.89 t/s |
| exact replay, cached | 169.4 ms | 17.59 t/s |
| **agentic turn (prefix + new message)** | **321.0 ms** | 22.70 t/s |

**131.7x on exact replay, 69.5x on a realistic agentic turn** — and decode
roughly doubles too, because the slot stops re-prefilling. `pulse serve` now
enables it by default.

This is what makes long context usable. A cold 131k prefill genuinely takes
211.7 s (256k ≈ 7 min), but an agent pays that **once** and every later turn is
a few hundred ms to first token.

And that covers nearly everything: across **17,989 records in 23 real Claude Code
sessions**, only **0.59%** carry a marker that mutates earlier context
(`truncated`, `elided`, `compact_boundary`). Agent traffic is ~99.4% append, so
the prefix cache hits almost every turn. This is also why the slot-checkpointing
layer below is off — it was built for the 0.6%, and it loses to `--cache-ram -1`
even there.

## Measured: Pulse vs stock llama.cpp defaults

Same model, same harness (`bench/conc.py`), 128 tokens/request, temperature 0,
idle GB10 (0 other compute apps).

| concurrent clients | stock defaults | pulse | speedup |
|---|---|---|---|
| 1 | 26.58 tok/s | **40.71 tok/s** | **1.53x** |
| 4 | 72.93 tok/s | **79.61 tok/s** | 1.09x |
| 16 | 73.08 tok/s | **117.71 tok/s** | **1.61x** |

Stock is `llama-server -m <model> -ngl 99 -c 65536` (defaults to 4 slots, no
drafter, no flash attention). It saturates at 73 tok/s because 16 clients queue
on 4 slots.

That is the whole honest claim: **~1.5-1.6x from configuration**. It is not 10x,
and nothing here will make a bandwidth-bound machine go 10x.

## The physics, so the numbers make sense

At batch 1 this is a memory-bandwidth problem and nothing else. Bonsai 2 27B at
`PQ2_0` is **6.70 GB**, so decode is set by how fast this machine sweeps those
weights once per token.

| | GB/s | % of spec peak |
|---|---|---|
| spec peak, 256-bit LPDDR5X @ 8533 MT/s | 273 | 100% |
| **achievable** pure read at a 6.70 GB working set | **216** | 79% |
| llama.cpp decode (6.70 GB / 36.65 ms measured) | 183 | 67% |

Two separate gaps, and only one of them is a bug:

- **273 -> 216 is the DRAM, and it is not recoverable.** Refresh, row
  activate/precharge across a 6.70 GB working set, read turnaround, and a fabric
  shared with the Grace CPU. 70-85% of theoretical is the normal LPDDR5X range.
- **216 -> 183 is software, and it is 16%.** The true weight-sweep roofline is
  `6.70 / 216 = 30.9 ms` against a measured 36.65 ms. Some of that 5.75 ms is real
  non-weight work - KV reads, norms, Hadamard transforms, `quantize_q8_1`,
  sampling - but not all of it is.

**Correction.** Every earlier round of this repo quoted **184.6 GB/s** as the
hardware ceiling and concluded llama.cpp decode was *at* the roofline ("1%
agreement"). That 184.6 came from a single kernel configuration. Sweeping
occupancy and memory-level parallelism (`bench/cuda/bw.cu`) reaches **229.0 GB/s**
at a 4 GB working set and 212-218 GB/s at 6.70 GB. The real gap is 16%, not 1%.
Curiously, the best configurations are **low occupancy**, often 1-2 blocks per SM.

Two constraints that bound everything downstream:

- No-drafter decode measures **27.54 tok/s** against 27.3 predicted.
- Two *independent* decode processes give **1.01x** aggregate. There is no
  headroom at batch 1; only batching or speculation can help.

Batching amortises the sweep across sequences, so it scales:

| batch | decode tok/s | prefill tok/s |
|---|---|---|
| 1 | 27.54 | 994.80 |
| 2 | 46.53 | 980.48 |
| 4 | 78.92 | 976.54 |
| 8 | 106.09 | 945.06 |
| 16 | **133.08** | 881.44 |

## The biggest single-stream lever: speculation is a net loss at long context

Draft acceptance does not hold up as context grows. It collapses
(`bench/acceptlong.py`, v1 drafter at K=4, spreads 0.3-0.7%):

| ctx tokens | acceptance | decode tok/s |
|---|---|---|
| 2,134 | **26.25%** | 37.41 |
| 8,666 | 18.18% | 29.48 |
| 16,165 | 6.25% | 20.41 |
| 34,196 | **0.00%** | 14.41 |

At 34k the drafter proposes 144 tokens per run and **none** are accepted. It
still streams its 0.59 GB every step and still pays ~3.11 ms per verify row, so
speculation stops paying for itself:

| ctx tokens | spec K=4 | no spec | winner |
|---|---|---|---|
| 2,134 | **38.18** | 27.25 | spec, +40.1% |
| 8,666 | **29.86** | 25.52 | spec, +17.0% |
| 12,279 | 24.65 | 24.19 | wash (inside noise) |
| 14,036 | 21.24 | **24.04** | no spec, +13.2% |
| 34,196 | 14.59 | **20.03** | **no spec, +37.3%** |

**Turning speculation off above ~14k is worth 1.37x at 34k** — more than the
entire configuration story, at the context length where agents actually run.

It is a taper rather than a cliff. Sweeping both axes (`bench/ctxk.py`):

| ctx | K=0 | K=1 | K=2 | K=3 | K=4 | best |
|---|---|---|---|---|---|---|
| 8,666 | 25.12 | 26.84 | 30.25 | **30.67** | 29.49 | K=3 |
| 12,279 | 23.89 | 23.09 | **25.57** | 25.10 | 24.06 | K=2 |
| 16,165 | **23.13** | 19.87 | 21.08 | 20.46 | 19.69 | K=0 |
| 35,541 | **19.53** | 15.26 | 14.85 | 14.50 | 14.11 | K=0 |

The optimum falls 4 → 3 → 2 → 0. A first revision of this gate cut straight to
K=0 at 12,288 tokens and gave up 7.1% at 12.3k, where K=2 still beats no
speculation.

The serving policy below only ever considered *concurrency*. Context length is a
second, independent axis and nobody had looked at it. `pulse serve` now gates on
it (`PULSE_SPEC_CTX_CUTOFF`, default 12288 tokens). This uses the per-request
`n_max` patch above: `n_max=0` disables drafting for a single request.

## Speculation and batching are substitutes, not complements

The drafter buys extra tokens per weight sweep while the sweep is under-occupied.
Once batching has filled it, the drafter's extra rows are pure cost.

| clients | with drafter | no drafter |
|---|---|---|
| 1 | **40.71** | 26.46 |
| 4 | **79.61** | 72.24 |
| 8 | **98.17** | 94.80 |
| 16 | 111.99 | **117.71** |

Crossover is between 8 and 16 clients. `pulse serve` picks the side of that
curve for you. Honest caveat: versus a static always-on drafter the gain is only
~5%, at 16 clients only.

## The serving policy (measured)

Full grid: drafter x concurrent clients, `bench/grid.sh`, 16 slots, 128 tok/request.

| clients | no drafter | **v1** (block 4) K=4 | **v2** (block 7) K=7 |
|---|---|---|---|
| 1 | 23.54 | **37.78** | 36.35 |
| 2 | 40.24 | 53.24 | **58.83** |
| 4 | 68.02 | **74.66** | 62.80 |
| 8 | 87.77 | **92.37** | 77.06 |
| 16 | **111.44** | 102.86 | 84.39 |

**Deeper speculation is actively harmful under batching.** v2 wins *single-stream*
on code context (71.5 vs ~64) but loses to v1 at every concurrency level except 2.
Batch rows are `clients × (K+1)`, so at 16 clients v2's K=7 means **128 rows per
batch** — past `MMVQ_MAX_BATCH_SIZE = 8`, putting every matmul on MMQ, which has a
~65 ms base against `mul_mat_vec_q`'s 36.6 ms. Depth that pays for itself at batch 1
is pure cost once batching has already filled the weight sweep.

| clients | use |
|---|---|
| 1 | v1 at K=4 — or v2 at K=7 for code-like prompts specifically |
| 2–8 | v1 at K=4 |
| ≥12 | no drafter |

`pulse serve` applies this automatically from `--slots`.

**Caveat.** The grid ran short chat-style prompts and saw 29–43% acceptance; the
single-stream code runs saw 69–72%. Acceptance is strongly workload-dependent —
measured range across this project is **21% to 96%** — so the right drafter depends
on prompt type as well as concurrency. The grid answers "which drafter under load",
not "what throughput will my workload get".

## Drafters

| drafter | size | block | best K | single-stream tok/s | acceptance |
|---|---|---|---|---|---|
| **v1** (default) | 603 MB | 4 | 4 | ~64 | 72.3% |
| v2 | 1.10 GB | 7 | 7 | **71.5** warm median | 69.1% |

**K is capped by the drafter's block size.** With v1, `--spec-draft-n-max 5` and `7`
are byte-identical to `4`. Requesting more is a no-op, and earlier rounds of this
repo reported "K=7" numbers that were really K=4.

## Per-request draft depth (patch to llama.cpp)

Pulse sent `spec_draft_n_max` on every request and llama.cpp threw it away.
Per-request speculative parameters are compiled out of its server behind `#if 0`
(`tools/server/server-schema.cpp`), and `server_slot::get_n_draft_max()` computed
only a context-fit bound, never reading the task's value.

`patches/llama-per-request-spec-n-max.patch` fixes both in 24 lines. It exposes
`speculative.n_max` (aliased to `spec_draft_n_max`) and applies it per slot.
Backward compatible: an omitted field keeps the server-wide value, and a request
can only *lower* K, never raise it, so `--spec-draft-n-max` stays the ceiling.

That made the K curve measurable per workload for the first time
(`bench/kcurve.py`, v1 drafter, block 4, 3 interleaved repeats):

| | K=1 | K=2 | K=3 | K=4 | K=6 |
|---|---|---|---|---|---|
| code | 36.57 | 46.95 | 52.19 | 56.07 | **56.43** |
| schema | 36.74 | 45.51 | **51.14** | 49.22 | 49.22 |
| chat | 33.93 | 39.05 | 40.92 | 41.38 | **41.62** |

Everything saturates at K=4, the drafter's block size — an independent
re-confirmation of the cap. But **schema peaks at K=3 and loses 3.9% at K=4**,
against a 0.1-0.7% spread.

The heuristic that shipped before this sent schema-like prompts to **K=7**, citing
"80-87% acceptance, 140-157 tok/s". Those numbers were never measured and are
gone. The measurement wants schema at the shallowest depth of the three, not the
deepest. The policy is now the measured one.

**Honest size:** against a single global K=4 this is worth **3.9%, on schema
traffic only**. It is in here because the feature was advertised and did not
work, not because the number is large.

## Measured negative results

| lever | result |
|---|---|
| `GGML_CUDA_GRAPHS=ON` | 61.84 vs 62.40 tok/s — **no effect**. Disabled at runtime: Gated DeltaNet nodes fail `ggml_cuda_graph_check_compability` |
| batched LM-head kernel | **no gain available**. `mul_mat_vec_q` already amortises the weight read across `ncols_dst`; +10% kernel time for 2.5x the columns |
| smaller `PTQ1_0` target | **22% slower** (51.73 vs 65.96) despite being smaller |
| KV cache `q8_0` | 65.53 vs 66.59 f16 — slightly worse |
| n-gram stacking | counters byte-identical to the drafter alone; never engaged |
| `-ub` above 512 | prefill 389 vs 656 tok/s — worse |
| **slot checkpointing** (this repo built it) | **4.2x slower.** 3080 ms vs 732 ms returning to an evicted session. llama.cpp's `--cache-ram -1` already restores evicted slot state from RAM; a forced disk restore discards it. Off by default |

## Measurement hygiene

Two traps that produced false results here, both now guarded by `pulse doctor`:

- **GPU contention.** An identical benchmark swung 33.79 ms → 150.33 ms (4.4x)
  with stray processes on the GPU. `pulse bench` refuses to run on a busy GPU.
- **CPU steals GPU bandwidth.** GB10 shares one LPDDR5X bus. 8 CPU threads streaming
  memory cost **16%** of decode throughput (73.3 → 61.0 tok/s) — and throughput did
  **not** recover when the load stopped. Unified-memory pages migrate and fault back
  lazily, so throughput also *climbs* across consecutive runs.
- **The noise floor depends on the protocol.** Cold, without a warmup discard, it
  is **10.2%**; warm and interleaved (`bench/ab.py`) it is **3.4%**. Take the
  median of >=6 runs after a quiesce, and do not call a delta real unless it beats
  the floor for the protocol you actually ran.
- **nsys under-reports GPU busy time on GB10.** A trace reported the GPU 92.2%
  idle during decode. CUPTI does not capture fabric stalls on unified memory.
  The two-process 1.01x result disproves it directly. Do not publish GPU-idle
  percentages from nsys on this machine.

## Measurement protocol

This machine is state-dependent, so the protocol changes the answer. The identical
config and command, three ways:

| protocol | median | spread |
|---|---|---|
| n=3, favourable machine state | 73.3 tok/s | 1.5% (luck, doesn't reproduce) |
| n=6, quiesced, **no warmup discard** | 66.3 tok/s | 10.2% (cold; runs climb) |
| n=4, **2 warmup + interleaved** | **71.5 tok/s** | **3.4%** (warm steady state) |

Cold and warm are both real: ~66 is first use after CPU activity, ~71.5 is
sustained serving. `bench/ab.py` enforces the warm protocol — it refuses a busy
GPU, discards warmup, **interleaves configs** so drift can't masquerade as a
difference, and won't call a delta real below the noise floor.

## Quick start

```sh
pulse doctor                  # is the GPU clean enough to measure on?
pulse serve --slots 16        # llama.cpp with the measured-optimal config
pulse bench --levels 1,4,16   # reproduce the concurrency table
pulse profile                 # print the tuning profile with provenance
```

## What would actually move the number

**The drafter's forward pass is the cost, and it scales with its weight volume.**
Measured on the *same* prompt at matched K=3, so nothing has to be subtracted
across workloads:

| drafter | size | ms/step | delta |
|---|---|---|---|
| v1 | 603 MB | 60.3 | - |
| v2 | 1.10 GB | 78.3 | +18.0 |

The clean differential is `3.51 ms / 0.473 GB` = **7.42 ms/GB**, i.e. roughly
**135 GB/s** on the drafter's weights - 73% of what llama.cpp achieves on the
target's. The drafter is somewhat less efficient, not catastrophically so.

A second differential, same drafter at two depths (block size 4, so K=1 and K=3
both run exactly one block pass and the drafter term cancels):
`(62.61 - 56.38) / 2` = **3.11 ms per verify row**. That independently confirms
the design curve's fitted slope of 2.9 ms/row, measured on an unrelated prompt.

**Retracted.** Earlier rounds claimed the drafter ran at **~39 GB/s** ("21% of
bandwidth"), that it was "chopped into ~24 separate MMQ launches per step", and
that a pure runtime fix would therefore reach 102 tok/s with no retrain. All of
that is withdrawn. The 39 GB/s came from subtracting a design curve measured on a
*repetitive* prompt from figures measured on a *code* prompt, where the curve
under-predicts by ~19 ms - the slope transfers, the base does not. The "24 MMQ
launches" were `ggml_type 142`, which is the **target**, not the drafter; `128`
was MMQ's tile width `mmq_x`, not a row count; and they were prefill launches
divided by the wrong step count. The kernel-level mechanism is still open.

Remaining levers, in measured order:

| lever | status |
|---|---|
| prefix caching | **shipped, 131.7x** on append-only turns |
| the 16% software gap on the weight sweep | open, bounded at 1.19x |
| block >=7 drafter holding >=70% acceptance at long context | needs a retrain |

## History

This repo previously published 141-206 tok/s single-stream, 495 tok/s across 16
subagents, and 525 tok/s for an MoE model that was never loaded. Those numbers
were not measured. They came from a benchmark harness with hardcoded latency
constants and from tables typed by hand. They have been removed from the code,
the README, the model card and the blog post, and the corrections are recorded
commit by commit. `bench/RESULTS.md` documents what replaced them and why.

## License

Apache-2.0.
