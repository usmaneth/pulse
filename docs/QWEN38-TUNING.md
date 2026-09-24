# Qwen3.8-Flash-Next tuning on a DGX Spark

This guide records the tools and results of the Qwen3.8-Flash-Next speed work
on one DGX Spark (GB10, 128 GB unified memory). The base is the MiaAI-Lab
single-Spark recipe (vLLM, NVFP4 checkpoint, MTP K=3). A change becomes part of
the serving profile only when it wins a measured A/B.

## Tools

`bench/qwen38/`:

| File | Purpose |
|---|---|
| `codexbench.py` | Decode tok/s, TTFT and speculative tokens per step on Codex-shaped prompts. `--xl` adds a ~110k-token context case. |
| `acceptbench.py` | Tokens per step and per-position acceptance on held-out prompts. It does not depend on the GPU state, so it isolates draft quality. |
| `qualitygate.py` | HumanEval pass@1 at temperature 0. |
| `driftgate.py` | Top-1 agreement and approximate KL of prompt logprobs between two configurations. Use it for changes to the target model's math. |
| `gpuprobe.py`, `probe.sh` | GEMV bandwidth probe. The GB10 has a slow state (about 70-160 GB/s against 210-230 GB/s). Discard benchmark runs that start or end in it. |
| `copyrate.py` | Fraction of real agent output that is copied from its context (sizes suffix decoding). |
| `verifycost.py` | Forward time against the number of new tokens on a cached prefix (prices wide verify steps). |
| `profstep.py` | Summary of a vLLM torch-profiler trace: GPU busy time, idle gaps and top kernels. |
| `headbench.py`, `hcbench.py` | Microbenchmarks for FP8 against BF16 projections at decode shapes. |
| `build_corpus.py`, `coverage.py` | Draft-vocabulary corpus and coverage. |
| `relaunch.sh` | Stops the server, waits for host memory, starts it again. |

`distill/qwen38/` holds the MTP head self-distillation pipeline:

1. `build_prompts.py` assembles prompts: Magicoder OSS-Instruct, self-oss-instruct, SWE-bench issues, UltraChat, the user's own session prompts, and a Codex-shaped slice with the real Codex system prompt and tools.
2. `gen_client.py` has the target model answer every prompt and stores the exact token ids.
3. `capture_client.py` replays each row as a prefill-only request with a unique `cache_salt`. The `capture` overlay (`overlays/qwen38/`) writes the MTP input hidden states: the pre-final-mixer multi-stream state, `[T, 4 x 2560]` BF16.
4. `train_mtp.py` trains the MTP block. The block is one Qwen4Exp decoder layer from transformers 5.16 behind the vLLM input fusion. The NVFP4 routed experts stay frozen at the served values, and the BF16 tensors train (90.6M parameters). `--eval-only` with the stock head is the parity check.
5. `export_mtp.py` writes the trained tensors into a copy of the MTP shard. The serving profile bind-mounts it over the original.

Data files are not in the repository. They hold private code or prompts, or they are large: long-context test texts, held-out prompts, the Codex template, generated rows, captures and trained heads. The scripts document how to rebuild each one.

## Results (single stream, one Spark)

Measured with probe-gated runs. Tokens per step comes from the held-out acceptance bench.

| Change | Result | Status |
|---|---|---|
| Kernel memory compaction off (`vm.compaction_proactiveness=0`, THP defrag `never`) | Removes GPU slow-state periods. Data generation went from 30 to 127 tok/s on the affected node. | shipped |
| FP8 rowwise MTP draft head | +5-6% code decode | shipped |
| vllm#53388 block-drop backport + 8192-token prefill chunks | TTFT on cached Codex turns 1.8 s to 0.8-0.9 s; +15% prefill | shipped |
| Self-distilled MTP head, round 2 | Tokens per step 2.96 to 3.09 (+4.3%); own prompts +6.9% | shipped |
| MTP index share for draft steps | Neutral at 30k context | shipped |
| MTP K=4 (stock head) | Slower: the extra draft pass costs more than it returns | rejected |
| Reduced draft vocabulary from own traffic | Acceptance unchanged | rejected |
| Fused QSA draft metadata | Neutral after probe normalization | rejected |
| FP8 target lm_head (`_scaled_mm`) | No speed gain after normalization; 92.9% top-1 agreement | rejected |
| FP8 hyper-connection projections | 4-5x slower at these shapes | rejected |

The vLLM patches for the shipped and rejected speculation-layer changes are in `overlays/qwen38/`. `overlays/qwen38/MANIFEST.md` gives the status, the evidence and the profile arguments of each one.

Other measured facts:

- About 29% of real Codex output tokens (40% of text, 10% of tool calls) are copied from context in spans of 8 or more tokens.
- A forward pass of 16 extra tokens on a cached prefix costs about 9-17 ms.
- Temperature-0 output is not bit-deterministic on this stack. Gate quality with task success and logprob drift, not exact text.
