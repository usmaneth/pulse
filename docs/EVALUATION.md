# Matched Spark evaluation

This suite is a protocol and small code-task qualification suite. It does not establish a general model quality rank.
No GPU measurements accompany this change.

## Run contract

Run one model configuration at a time. The root orchestrator owns the GPU lease.
The runner uses one shared advisory lock and checks the compute-process allowlist on each node.
The lease expires at a fixed Unix time. It must name the exact model nodes.
The resource check reads unified memory availability and free disk. Host memory and GPU memory share the same pool.
The lock cannot stop unrelated clients. The operator must isolate the server from unrelated traffic.

1. Copy a model example from `bench/agent-eval/`.
2. Set the actual endpoint, nodes, model hash, engine commit, and template hash.
3. Set the per-request context limit from the running server.
4. Set the resource thresholds from the deployment plan.
5. Set the token-counter and cache-reset commands.
6. Copy `lease.example.json` and enter the current server process IDs and expiry.
7. Run the plan command.
8. Run the live command only after the root grants the GPU slot.

```sh
python3 bench/agent-eval/runner.py /path/to/config.json
python3 bench/agent-eval/runner.py /path/to/config.json --run --lease /path/to/lease.json --output /path/to/new-results
```

The examples are incomplete deployment manifests. The runner rejects missing provenance.
The llama.cpp hooks require `/apply-template`, `/tokenize`, and the idle slot erase endpoint.
The other engines need a cache-reset hook that returns `{"cleared": true}` after the reset completes.
The local Transformers counter never downloads files and disables remote tokenizer code.
A custom server template needs a matching counter. Tokenizer names alone do not prove a template match.

## Workloads

Each model runs 8K, 32K, and 64K input contexts, with concurrency 1, 2, and 4, and three repeats.
128K is optional. Allow output and tool-turn space above the input target.
For llama.cpp, total context capacity must cover all slots. A shared 64K allocation is not four 64K slots.

The context contains deterministic, varied Python archive records. Every model uses the same record generator.
The counter adjusts record count to the token target. Thus equal-token rows are not byte-identical model inputs.
Set `fixture_mode: fixed_bytes` in a separate run for a byte-identical quality comparison.
The same context target selects the same record count in that mode. Actual token counts can differ.
Keep `fixture_seed` equal across model configurations. The runner saves every prepared fixture.
Stable request identifiers derive from the seed, context target, concurrency, repeat, and stream index.
The runner records request hashes and measured server prompt tokens.
A counter mismatch invalidates the task measurement and prevents success.

The task requires an exact `read_record` call and arguments. The runner returns a private numeric record.
The next request includes the actual assistant call and tool response. The model must compute the correct total.
The client then compacts the record and asks for a changed quantity. The model must preserve the fact and recalculate.
This is deterministic client compaction. It does not test the Codex compaction implementation.

A code task then requests `invoice_total`. Eight withheld cases check defaults, row-level rounding, empty input, discounts, and invalid values.
The validator rejects imports and unsafe syntax. It executes permitted code in a separate process with time and memory limits.
The prompt declares the permitted syntax. Unsupported syntax reports `evaluated: false`, not an incorrect task outcome.
This restricted task does not accept every valid Python implementation. Extend the syntax contract only with an isolation review.
The small fixture set is a regression gate. A real repository repair holdout remains necessary for broad model quality claims.

## Metrics

- `ttft_any_s`: request start to the first reasoning, content, or tool delta.
- `ttft_visible_s`: request start to the first content or tool delta.
- `chunk_gap_p50_s`, `chunk_gap_p95_s`: observed SSE chunk gaps, not token latency.
- `elapsed_s`: complete request wall time, including prefill and transfer.
- `end_to_end_tokens_s`: server completion tokens divided by request wall time.
- `server_decode_tokens_s`: the explicit server decode metric, or null.
- `aggregate_tokens_s`: all server completion tokens divided by the complete task-wave wall time.
- `draft_acceptance`: `timings.draft_n_accepted / timings.draft_n`, or null.
- `success`: correct tool action, both arithmetic outcomes, code holdout, and verified input token count.

The runner never substitutes wall throughput for decode throughput.
It preserves the complete usage and timing objects. Missing speculation counters stay null.
The runner does not infer token timestamps from SSE chunks. A speculative step can emit several tokens in one chunk.

Cold runs require a cache-reset hook. Primed-cache runs first prime all initial requests after the previous complete task.
The report stores priming requests separately and excludes them from the measured task-wave wall time.
A prime does not prove cache residency: scheduling and limited slots can still evict a prefix.
Only reported cached-token evidence can set `cache_verified`.
When cached-token evidence is absent, report primed-request latency and mark cache status unknown.
Append latency belongs to `appended_tool_turn`; it is separate from the primed-cache phase and compaction.
Cold here means empty model prefix cache, not a cold OS page cache or unloaded model.

Chat template, protocol, reasoning mode, quantization, draft model, and engine revision are separate experiment dimensions.
The runner currently uses Chat Completions. Test the Responses gateway and actual Codex loop separately.
Do not mix those measurements into a Chat Completions engine comparison.

## Exactness audit

```sh
python3 bench/agent-eval/audit_saved.py /path/to/gb10-clean-1
cd bench/agent-eval
python3 -m unittest -v
```

The saved audit reproduces 37/65 matches for each draft. Draft outputs match each other on 59/65 pairs.
Both drafts first diverge from baseline at the same position on all 28 failed comparisons.
The saved chat-template hashes do not differ between configurations.
The evidence does not identify the numerical cause.

The Hadamard fix `288859a96` repairs borrowed drafter tensors. Its stated symptom is low draft acceptance.
It does not establish target verification exactness. The saved runs already show much higher acceptance than that symptom.
A binary version string does not prove a clean build or identify uncommitted source changes.

1. Pin the binary hash, model hash, template, sampler, and prompt tokens.
2. Repeat the earliest divergence, `reasoning-06`, with draft depth zero and four on the same binary.
3. Compare logits at the first differing position with the same forced token prefix.
4. Compare one-token target decode with the multi-token verify batch.
5. Record the top-two logit margin, recurrent state, and KV state.
6. Repeat with both draft revisions and with cache reuse disabled.
7. Test output limits and EOS separately from content differences.
8. Repeat the complete 65-output set after the controlled cases pass.

A changed greedy token near a tie can follow floating-point batch differences.
A state or verify error can also cause divergence. Neither explanation is confirmed by these saved files.
