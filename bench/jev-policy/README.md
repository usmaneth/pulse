# Bounded Jev policy experiment

Jev proposes policies for the next request. Local code controls admission and application.
The controller does not replace target-token verification, CUDA correctness checks, or memory limits.
It does not modify the backend, allocator, or active request.

## Proposal contract

The pinned model is `jev-1.13.0` at `https://api.typesafe.ai/v1/systemone`.
One call requests three atomic choices: speculative depth, kernel candidate, and cache policy.
The code retains each raw probability distribution and confidence value.
The code calculates telemetry categories before the API call.
Only the public workload description, context category, telemetry categories, and legal candidate list enter the request.
Expected outputs, hidden tests, raw source files, and API credentials do not enter that projection.

The speculative candidates are K0, K2, K4, and K7. Both the server cap and draft block cap must equal seven.
The current kernel manifest permits only `reference`. The graph candidate requires a separate local correctness gate.
The cache manifest permits only LRU in live mode. Semantic eviction remains a separate simulation.
Every choice supports explicit abstention.
The initial confidence threshold is 0.8. This threshold is experimental and is not calibrated on runtime performance.

Local validation rejects unknown candidates, malformed distributions, nonfinite probabilities, capacity overflow, and disabled target verification.
A changed snapshot or proposal age above 30 seconds forces the default.
The default is K4, reference kernels, and LRU.
A local memory admission failure rejects the request, including the default.
`NextRequestProposal` starts the API call in a separate worker thread.
Its `poll` method does not wait for an unfinished proposal at the request boundary.
Apply the result to the next request with `speculative.n_max`. The backend cannot change K during a request.

## CPU and API checks

```sh
python3 -m unittest discover -s bench/jev-policy -p 'test_*.py'
python3 bench/jev-policy/policy.py --fixture bench/jev-policy/selection-fixture.json --output /tmp/jev-proposal.json --live
```

The API command reads `~/.config/pulse/typesafe.env` as data.
It does not print the key or execute the environment file.
The fixture contains synthetic capacity values. It does not authorize real GPU admission.
The first live proposal took 291.96 milliseconds and abstained on speculation and cache.
Local validation retained K4, reference kernels, and LRU.
This API check establishes connectivity and policy handling. It establishes no speed improvement.

## Next GPU experiment

Obtain a new exclusive GPU grant before the experiment.
Use the same corrected v2 draft, target model, server build, sampler, and context capacity.
Set the server speculative maximum to seven and verify the loaded draft block size is seven.
Send per-request `speculative.n_max` values 0, 2, 4, and 7.
Verify the response settings and counters for every candidate.
Record K0 under the loaded-draft server as its own control.
Keep a separate no-draft process baseline because draft residency or feature extraction can add overhead.

Use `holdout-manifest.json` for the frozen task definitions.
Run each candidate at 3072, 8192, and 32768 input tokens.
The initial gate has one repeat per task and candidate: 24 task runs.
A task can require more than one model request.
Use a 2048-token output limit per request initially. Mark a limit stop as incomplete.
Continue the tool loop with actual assistant output and the locally executed tool result.
Run hidden behavioral tests for the code task.
Report unsupported harness syntax separately from a failed behavioral test.

Generate Jev proposals from selection data before each holdout run.
Freeze and hash each proposal before exposing any holdout outcome.
Do not send hidden tests, expected answers, candidate outputs, or holdout timing back to the selector.
Compare the frozen selected candidate against fixed K4 on the same holdout task.
Use all-candidate measurements only for the final counterfactual report, not to revise that task's selection.
Repeat three times after the gate passes. Rotate candidate order deterministically.

Report completed-task success, total task latency, output tokens per wall second, prefill/decode time, acceptance, and K0 exactness separately.
Include API latency, proposal coverage, abstention rate, deadline misses, and fallback frequency.
Show paired ratios and the full timing range. No 10x claim follows from API latency or acceptance alone.

## Cache replay

`cache_replay.py` is a simulation. It does not evict real KV state.
Each decision sees only the current request and resident page metadata.
It cannot see future accesses.
The replay enforces byte capacity, pinned pages, eligible victims, snapshot identity, and a confidence threshold.
Missing or invalid proposals use LRU.
The CPU test verifies that an invalid proposal cannot evict a pinned page.
A future comparison must count misses, bytes fetched, replay cost, API cost, and accepted semantic decisions.
No semantic hit-rate improvement is measured yet.

## Future task-level cascade

After the K experiment, evaluate a separate task-level policy.
Bonsai proposes a bounded patch. A local test runner measures its behavior.
Jev assesses semantic fit from the task, bounded patch summary, and actual test result.
Jev can recommend acceptance, abstention, or escalation to a larger Qwen model.
A confidence value cannot turn a failed test into a pass or replace numerical target verification.
Use hidden task tests to measure false acceptance after the decision.
Report completed-task latency, success, escalation rate, false-accept rate, and total inference cost.
The larger-model comparison must supply measured holdout results before this policy can rank models.
This cascade is a hypothesis, not an implemented speed feature.

## Experiment proposals versus production advice

Use `--mode advice` for the confidence-gated proposal described above.
Use `--mode experiment` to rank trial order among all legal bounded candidates.
Experiment mode does not apply the 0.8 confidence gate.
It retains raw confidence and probabilities and schedules every legal candidate, including low-probability candidates.
It returns no selected production policy and no request settings.
A graph trial can test a new legal implementation before numerical validation permits promotion.
Every trial still requires local capacity, current snapshot identity, and numerical target verification.
Promotion separately requires numerical correctness, task tests, local admission, and measured paired benefit.
Low confidence broadens research; it does not authorize production changes.

The saved API call accepted zero of its three advice choices.
Its 0.22 speculation confidence provides no evidence that semantics alone identify the hardware optimum.
The CPU replay of that distribution ranks K0, K4, K2, then K7.
That ranking only defines trial order. It does not claim those candidates perform in that order.
The replay uses a synthetic fresh timestamp and cannot authorize a live request.

## Runnable task gate

The default dry run contains four tasks: two frozen tasks at 3K with K4 and K7.

```sh
python3 bench/jev-policy/task_sweep.py --output /tmp/task-gate --proposal /path/to/jev-task-frozen-proposal.json
```

The proposal only orders trials. The root froze this live API proposal before any task holdout output existed.
It ranks K0, K4, K2, and K7 despite explicit raw abstention at confidence 0.23.
It does not authorize production policy promotion.
After the root grants the GPU slot, add `--run --exclusive-grant GRANT_ID` with a new output directory.
The runner starts its own backend on port 18125 with context 65536 and maximum K7.
It uses log verbosity five to verify the actual per-request draft cap.
This diagnostic log cost is part of this protocol. Do not compare its timings directly with quieter runs.
The runner confirms the corrected draft's YaRN metadata and block size before requests.
It stops only its own child and uses the shared evaluation lock.

The runner imports existing agent-eval streaming metrics, code validation, answer validation, token counting, and cache reset hooks.
It keeps each successful request record if a later phase fails.
It preserves assistant reasoning and the actual tool result for the next turn.
It checks each request's token reserve before inference.
Each request has a 2048-token generation budget, including reasoning.
The code task has one request. The two-turn tool task permits at most 4096 generation tokens total.
Only complete code with passing behavioral tests or a complete correct tool loop counts as task success.

After the four-task gate passes, use the full 24-task schedule:

```sh
python3 bench/jev-policy/task_sweep.py --output /tmp/task-full --proposal PROPOSAL.json --contexts 3072 8192 32768 --ks 0 2 4 7
```

Add `--no-draft-control` for six additional no-draft task controls.
Loaded-draft K0 and a true no-draft process remain separate groups.
Add `--repeats 3` only after the initial schedule passes its measurement checks.
Generate the outcome-first report with:

```sh
python3 bench/jev-policy/summarize_tasks.py OUTPUT_DIRECTORY
```

The report compares the frozen top-ranked candidate with fixed K4.
It reports a latency ratio only when both tasks pass and both measurements are valid.
It compares complete text/reasoning/tool transcripts with K0 when available.
The streaming protocol does not expose token IDs, so this report does not claim token-ID equality.

## Separate practical-budget profile

The default-budget gate passed three of four tasks. K7 exhausted 2048 tokens in reasoning on the invoice task.
Preserve that failure. Do not combine it with the next profile.
The next predeclared profile uses `--reasoning-budget 512`, with the unchanged 2048-token total limit per request.
Use all four K candidates at 3K plus `--no-draft-control` for ten task runs.
The runner records profile identity and can recover exact emitted token IDs from its debug logs.
Verify the token-log count against the reported completion-token count before using that evidence.
See `jev-task-gate-report.md` in the orchestration directory for the pending command and failed-gate evidence.
