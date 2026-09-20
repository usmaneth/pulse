# Pulse

**Measurement-driven speculative decoding for NVIDIA DGX Spark (GB10).**

Pulse is not an inference engine and does not replace llama.cpp. It is two things:

1. A **benchmark suite** for speculative decoding on GB10, built after a previous
   iteration of this repo published numbers that turned out to be invented. Every
   figure below was measured on an idle GPU and is reproducible from this repo.
2. A **tuning profile plus launcher** that starts llama.cpp with the configuration
   measured to be best on this hardware, instead of making you rediscover it.

The negative results in [`bench/RESULTS.md`](bench/RESULTS.md) are the more useful
half. Several widely-assumed optimisations measurably do nothing here.

---

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

At batch 1 this is a memory-bandwidth problem and nothing else:

- Bonsai 2 27B at `PQ2_0` is **6.70 GB**. Measured sustained bandwidth is
  **184.6 GB/s**. One weight sweep is therefore **36.3 ms**.
- That predicts 27.5 tok/s with no drafter. Measured: **27.54 tok/s**.
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

## Drafters

| drafter | size | block | best K | tok/s | acceptance | tok/step |
|---|---|---|---|---|---|---|
| v1 | 603 MB | 4 | 4 | 64.1 | 72.28% | 3.86 |
| **v2** | 1.10 GB | 7 | **7** | **73.3** | 69.10% | 5.74 |

v2 at K=7 is the default: **73.3 tok/s** over 3 repeats (spread 1.09) on a realistic
code context, against a 27.54 tok/s no-drafter baseline — **2.66x**.

Stacking a free n-gram drafter ahead of the neural one helps v1 (+1.8%) and *hurts*
v2 (−11%): the free drafts displace deeper dspark drafts, 5.74 → 4.62 tok/step. With
the recommended drafter, don't stack.

**K is capped by the drafter's block size.** With v1, `--spec-draft-n-max 5` and
`7` are byte-identical to `4`. Requesting more is a no-op, and earlier rounds of
this repo reported "K=7" numbers that were really K=4.

## Measured negative results

| lever | result |
|---|---|
| `GGML_CUDA_GRAPHS=ON` | 61.84 vs 62.40 tok/s — **no effect**. Disabled at runtime: Gated DeltaNet nodes fail `ggml_cuda_graph_check_compability` |
| batched LM-head kernel | **no gain available**. `mul_mat_vec_q` already amortises the weight read across `ncols_dst`; +10% kernel time for 2.5x the columns |
| smaller `PTQ1_0` target | **22% slower** (51.73 vs 65.96) despite being smaller |
| KV cache `q8_0` | 65.53 vs 66.59 f16 — slightly worse |
| n-gram stacking | counters byte-identical to the drafter alone; never engaged |
| `-ub` above 512 | prefill 389 vs 656 tok/s — worse |

## Measurement hygiene

Two traps that produced false results here, both now guarded by `pulse doctor`:

- **GPU contention.** An identical benchmark swung 33.79 ms → 150.33 ms (4.4x)
  with stray processes on the GPU. `pulse bench` refuses to run on a busy GPU.
- **nsys under-reports GPU busy time on GB10.** A trace reported the GPU 92.2%
  idle during decode. CUPTI does not capture fabric stalls on unified memory.
  The two-process 1.01x result disproves it directly. Do not publish GPU-idle
  percentages from nsys on this machine.

## Quick start

```sh
pulse doctor                  # is the GPU clean enough to measure on?
pulse serve --slots 16        # llama.cpp with the measured-optimal config
pulse bench --levels 1,4,16   # reproduce the concurrency table
pulse profile                 # print the tuning profile with provenance
```

## What would actually move the number

**The drafter's forward pass runs at ~39 GB/s — 21% of the 184.6 GB/s this machine
sustains.** Measured by differencing each drafter's step time against the design
curve at matched tokens/step:

| drafter | size | overhead | effective bandwidth |
|---|---|---|---|
| v1 | 603 MB | +15.4 ms/step | 39.2 GB/s |
| v2 | 1.10 GB | +27.9 ms/step | 39.4 GB/s |

Overhead scales 1.81x for a 1.75x drafter, so it's the forward pass itself. The
cause is visible in the profile: a 5-layer block-diffusion model executed as ~24
separate MMQ launches per step, each moving ~25 MB — far too little to saturate
the bus. It's latency-bound, not bandwidth-bound.

At achievable bandwidth, v2 would be `50.33 + 1.10/184.6 = 56.3 ms/step` →
**102 tok/s single-stream**. That closes the entire gap to the target, and it is a
runtime optimization on drafter execution: we already own a block-7 drafter, so no
retrain is required. CUDA Graphs won't fix it — the drafter has Gated DeltaNet
layers, which is exactly why graphs get rejected at runtime.

## History

This repo previously published 141-206 tok/s single-stream, 495 tok/s across 16
subagents, and 525 tok/s for an MoE model that was never loaded. Those numbers
were not measured. They came from a benchmark harness with hardcoded latency
constants and from tables typed by hand. They have been removed from the code,
the README, the model card and the blog post, and the corrections are recorded
commit by commit. `bench/RESULTS.md` documents what replaced them and why.

## License

Apache-2.0.
