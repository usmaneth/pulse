# Local Bonsai in Codex

Pulse provides a stateless Responses adapter for the installed Codex CLI.
The launcher selects the local provider for one process.
It does not edit `~/.codex/config.toml` or replace the active provider.

## Start the private service

1. Install the JavaScript dependencies with `pnpm install --frozen-lockfile`.
2. Compile the server with `pnpm exec tsc`.
3. Start the server with the command below.
4. Keep the server terminal open.
5. Start Codex from another terminal with `scripts/codex-bonsai`.
6. Press Ctrl-C in the server terminal to stop its child processes.

```sh
scripts/codex-bonsai-install-draft
scripts/codex-bonsai-server
```

The installer copies the validated corrected v2 draft into the stable model directory.
It verifies SHA256 before and after the copy and preserves existing model files.
The default path is `~/Bonsai-demo/models/bonsai2-gguf/27B/bonsai2-v2-Q4_K_M-yarn32.gguf`.
The default backend is `~/llama.cpp-upstream/build-cuda/bin/llama-server`.
The target is `~/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`.
Use `--backend-bin`, `--model`, or `--draft` for explicit local controls.
A missing corrected artifact causes an error, not a silent fallback.

Use a Prism-compatible binary for Bonsai 2.
Stock llama.cpp cannot apply the required activation transform.
The default corrected v2 draft uses fixed K4 without the historical v1 taper.
Use `--legacy-v1` for the original v1 draft.
Use `--legacy-v1 --verified-spec-patch` only after the backend's per-request draft control passes a live check.
Use `--no-draft` for a baseline without speculative decode.
The three draft selections are mutually exclusive.
The launcher rejects the legacy taper with the corrected or custom draft.
The launcher refuses occupied ports and stops only its own child processes.

The default service uses loopback ports 18085 and 18086, one slot, 65536 context tokens, and F16 KV storage.
It uses a 4096-token batch and a 512-token microbatch.
The archived prompt cache has an 8192 MiB limit; change it with `--cache-ram`.
This limit does not cap live recurrent-state checkpoints, which have a separate count limit.
The default template and checkpoint behavior remain unchanged.
The isolated intermediate-checkpoint binary remains experimental.
Jev remains opt-in through `--jev`; no causal speed gain is established.
The service does not start the native Pulse prototype or its checkpoint service.

### Reproduce the corrected artifact

Use this procedure if the original validation artifact is unavailable.
The source v2 GGUF must have SHA256 `eb80f88fc94f267b6b612cba4deac31ecb0dfff0f587f2786fc3de855ceaf84a`.
The destination and manifest paths must not exist.
Run from the Pulse repository:

```sh
python3 bench/drafter-parity/metadata_variant.py \
  --gguf-py /home/usman/Bonsai-demo/llama.cpp/gguf-py \
  --source /home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf \
  --destination /home/usman/Bonsai-demo/models/bonsai2-gguf/27B/bonsai2-v2-Q4_K_M-yarn32.gguf \
  --manifest /home/usman/Bonsai-demo/models/bonsai2-gguf/27B/bonsai2-v2-Q4_K_M-yarn32.manifest.json
sha256sum /home/usman/Bonsai-demo/models/bonsai2-gguf/27B/bonsai2-v2-Q4_K_M-yarn32.gguf
```

Require output SHA256 `a39f89a344155e90e76b2fb0f8a79ef0c3bd42634c5e229a5fc9ab281f94f31d` before launch.
The converter changes four draft metadata fields and verifies identical tensor payloads.
It does not change the target or the original draft file.
The absolute paths describe the validated Spark installation; adjust their root on another machine.

For an existing private corrected-v2 backend, start only the adapter:

```sh
HOST=127.0.0.1 PORT=18086 \
PULSE_BACKEND_URL=http://127.0.0.1:18085 \
PULSE_RESPONSES_SPEC_POLICY=0 \
node dist/server/index.js
```

Enable `PULSE_RESPONSES_SPEC_POLICY=1` only for the verified legacy v1 control.
`GET /ready` checks backend health and returns 503 when the backend is unavailable.
The adapter does not obtain model weights or start a cloud fallback.

## Select the local profile

```sh
scripts/codex-bonsai
scripts/codex-bonsai exec 'Inspect the project and run its tests.'
```

The launcher uses its own local model catalog.
The catalog supplies a short coding instruction and enables text input, shell tools, and patch tools.
Apps, web search, and subagents are off for this invocation to reduce the prompt size.
The launcher preserves configured skills, project instructions, permissions, and MCP servers.
Unused skills and MCP tools still increase prompt processing time.
The launcher does not change approval or sandbox settings.
It rejects provider, model, catalog, profile, remote, and OSS overrides that could replace the local route.
Use `BONSAI_CODEX_URL` for a different local adapter.
Non-routing `-c` overrides work with `exec` and `exec resume`.
The launcher places all overrides in the same CLI parser scope.

Set `BONSAI_CODEX_URL` for a different loopback adapter port.
The launcher reads the actual slot context and model identity from `/ready`.
If you set `BONSAI_CODEX_CONTEXT`, it must match that slot context.
The default context is 65536, with local compaction at 75 percent.
Set `CODEX_BIN` to select another Codex executable.
For an explicit latency/quality tradeoff, use `BONSAI_CODEX_REASONING_BUDGET=512 scripts/codex-bonsai`.
The adapter sends this value as the backend `reasoning_budget_tokens` field.
The default leaves the backend reasoning budget unchanged.
A value of zero ends reasoning immediately; minus one selects the backend default.
Validate task quality before selecting a smaller budget.
For a remote Spark, use an SSH tunnel and a loopback URL.

## Protocol contract

The adapter supports text messages, instructions, function tools, custom tools, namespaces, and complete tool-result history.
It preserves call IDs and merges adjacent assistant tool calls.
It places system and developer instructions in one initial system message, as the Bonsai template requires.
Their relative order remains unchanged.
This transformation also places later developer instructions in that initial message.

Custom tools use a JSON function with one string field named `input` on the backend.
The adapter converts that string to a Responses custom tool call.
It includes the custom grammar in the backend tool description as guidance.
It does not enforce that grammar on the backend.
Tool output waits for complete arguments, so a partial name cannot select the wrong tool type.
Text and reasoning use incremental SSE events.
The adapter emits response creation before prompt processing, then sends a heartbeat every ten seconds.
These events do not count as first-token latency.
The adapter exposes local backend reasoning text through the Responses summary fields.
This text is the backend reasoning text, not a separate summary from an OpenAI model.
The adapter restores that text as `reasoning_content` on the next turn.
The managed backend enables `--reasoning-format deepseek --reasoning-preserve`.
Preserved reasoning increases context use but allows the template to retain the prior reasoning prefix.
Measure cached tokens to confirm reuse; `cache_prompt=true` alone does not prove it.

The adapter requires the full input history and `store=false`.
Stored response IDs, hosted tools, images, files, structured text formats, and server compaction are unsupported.
Unsupported requests return an explicit error.
The adapter drops incomplete tool calls and returns `response.incomplete` on the backend token limit.
A truncated connection returns `response.failed`.
Client disconnects cancel the backend request.
The adapter honors response backpressure.
The default total request deadline is 600 seconds; set `PULSE_BACKEND_TIMEOUT_MS` to change it.

The Responses path does not call Jev or an external gateway.
The existing Chat Completions route retains its current policy.
Responses requests count toward active requests, but `/status` token totals currently cover only Chat Completions.

## Validation

```sh
pnpm exec tsc
node --test dist/server/responses.test.js
python3 scripts/codex-launcher-test.py
```

The CPU tests cover protocol translation, split SSE frames, tool continuation, truncation, and cancellation.
A live Codex 0.154.0 text request returned `BONSAI_OK` on the private Spark backend.
That cold request required about 9.6 seconds for a 7409-token prompt and 0.7 seconds for decode.
This is an integration check, not a model quality score or a throughput comparison.

## References

- [OpenAI Codex provider configuration](https://learn.chatgpt.com/docs/config-file/config-advanced)
- [OpenAI Responses stream events](https://developers.openai.com/api/reference/resources/responses/streaming-events)

A live interval-repair fixture also passed through Codex shell, edit, and test tools.
Bonsai repaired the function and all eight tests passed.
Its first patch call failed the patch format check; it recovered with a shell edit.
The five requests reused 0, 0, 11135, 11590, and 11919 prompt tokens.
Their first backend token latencies were 10.54, 12.19, 0.354, 0.570, and 0.475 seconds.
These measurements include one cold run and do not establish a general latency distribution.

Set `PULSE_RESPONSES_METRICS_FILE` on the server for JSONL latency, usage, and backend timing records.
The records contain no prompts.
Set `PULSE_RESPONSES_TRACE_FILE` only for prompt-cache diagnosis.
That optional trace contains the complete backend request, including user and tool text.
New trace files use owner-only permissions.

## Actual CLI protocol check

```sh
python3 scripts/codex-cli-smoke.py --output /tmp/bonsai-cli-check
```

This CPU test uses the real Codex executable and an isolated temporary configuration.
Its fallback provider points to a local tripwire endpoint.
The test checks local routing, shell continuation, automatic compaction, session resume, and routing-override rejection.
Codex 0.154.0 performs local compaction with a normal Responses request and a checkpoint-summary prompt.
It does not require `/responses/compact` for this custom provider.
This protocol check does not test Bonsai summary quality.

## Compaction quality limitation

A forced live compaction test preserved the codeword and tool result in its summary.
The summary falsely claimed that the final answer was already sent.
The continuation repeated the completed shell command despite an explicit one-call limit.
The operator cancelled the run after the duplicate action.
This test used a 1000-token threshold to force compaction; the default threshold remains 75 percent of the slot context.
The failure remains in `scripts/codex-regressions/compaction-repeats-completed-tool.json`.

An unqualified candidate prompt records completed tools separately from the final answer:

```sh
scripts/codex-bonsai exec \
  -c experimental_compact_prompt_file='"/absolute/path/to/pulse/scripts/codex-bonsai-compact-prompt.txt"' \
  'Your task'
```

This prompt is opt-in and has not passed a live comparison.
Do not treat protocol support as proof of reliable model behavior after compaction.
The [OpenAI example configuration](https://learn.chatgpt.com/docs/config-file/config-sample) documents the prompt override.

## Prompt cache qualification

The live first follow-up added MCP tools after asynchronous discovery.
That change reduced the common prompt prefix to 689 tokens and forced prompt processing again.
Later stable tool lists preserved the full prior prompt prefix and reused cached tokens.
The default profile retains user MCP configuration and skills.
Do not compare the discovery request with a warm request as if both used the same prompt.

The 1000-token stress threshold was below the persistent prompt size and caused repeated compactions.
It does not establish failure at the normal threshold.
`scripts/codex-compaction-fixture.py` prepares a separate one-compaction case with a 12000-token threshold and an archive tool result.
That case still needs live qualification.

Set `BONSAI_CODEX_MCP_STARTUP_GRACE_MS=0` to wait for each MCP server's startup timeout before the first tool catalog.
The default preserves Codex's normal grace period.
This option keeps all user MCP servers and skills; it does not remove tools.
It may delay startup when a server is slow or unavailable.
Its effect on the live discovery cache miss remains unmeasured.

### Jev request decisions

Add `--jev` to the managed server command to enable bounded native TypeSafe advice.
The adapter sends a bounded latest user message, tool names, and structured tool outcome facts.
A high-confidence simple step can select a 512-token reasoning budget.
An explicit user budget takes priority. Errors preserve the normal budget.
This mode does not change the prompt or tool catalog.
See [Jev integration](JEV.md) for data scope, credentials, tests, and current limitations.
A simple answer and a coding fixture passed with this mode.
The coding run retained its normal budget and paid 1.220 seconds of API overhead.
No causal end-to-end speed improvement is established.

### Corrected v2 integration checks

A live patch fixture passed with the corrected v2 YaRN drafter and fixed K4.
The test used a 64k backend context, stock template, and explicit 512-token reasoning budget.
The custom patch succeeded on its first attempt, and both unit tests passed.
The managed launcher now selects this corrected draft with fixed K4.

MCP startup grace `0` did not preserve the initial tool catalog in the tested Codex version.
The catalog still increased from eight tools to twelve on the first follow-up.
Both grace settings caused a full first-follow-up prefill.
Keep the normal grace setting unless another measurement supports a change.

The realistic 12k compaction fixture passed with one file read and one compaction.
The model returned the correct final token without another tool call.
The initial prompt used 8384 tokens; the summary continuation used 9411 tokens.
Both stayed below the threshold, while the compaction request used 14918 tokens.
The earlier threshold-1000 failure remains a separate pathological configuration.

An experimental system-first template increased the common prefix from 689 to 4153 tokens.
The backend still reused zero tokens on the first follow-up with checkpoint spacing 2048.
That setting controls checkpoint spacing, not periodic checkpoint creation within a system message.
The template candidate remains an artifact, not a recommended profile.
`--chat-template-file` permits an explicit private backend template for controlled tests.

A later isolated backend patch permitted ordinary intermediate checkpoints.
Its exact two-request replay matched outputs and restored 4096 tokens.
Both real Codex arms passed the patch tests; the enabled arm reduced first-follow-up first-token time from 11.134 to 6.733 seconds.
These are bounded single-pair results, not general long-context qualification.
Live checkpoints have a count limit, not the archived cache's byte limit.
The isolated binary and template pair remain experimental and are not launcher defaults.

## Durable task continuity

Enable a stable task checkpoint for one launch:

```sh
BONSAI_CODEX_TASK=repair-cache \
BONSAI_CODEX_OBJECTIVE='Repair the cache bug and verify the regression test.' \
  scripts/codex-bonsai
```

Resume the same task with the same ID:

```sh
BONSAI_CODEX_TASK=repair-cache scripts/codex-bonsai resume --last
```

The local backend must be ready before either command.
Task IDs accept letters, digits, underscores, and hyphens.
Use a distinct ID for each independent task. The launcher selects the current directory or the explicit `--cd` directory.
Set `BONSAI_CODEX_PROJECT` to select an existing project directory explicitly. An explicit `--cd` takes precedence.
Create managed worktrees before launch; continuity rejects `--worktree` to avoid an incorrect project checkpoint.

The helper stores checkpoints in `~/.local/state/pulse/continuity/<project-hash>/<task-id>/`.
`BONSAI_CODEX_OBJECTIVE` initializes a new checkpoint. It does not replace an existing objective.
Without this value, the agent must replace the placeholder objective before substantive work.
Set `BONSAI_CODEX_VAULT` to select a vault other than `~/vault`.

The launcher adds three lifecycle hooks for this invocation:

- `SessionStart` restores bounded checkpoint context and relevant vault references at startup, resume, and immediately after compaction.
- `PreCompact` preserves the current structured revision in `precompact.json`.
- `UserPromptSubmit` adds bounded topic advice and a checkpoint reference for a new user request.

Existing user and project hooks remain active. Existing developer instructions, permission settings, and the compact prompt remain unchanged.
The launcher retains the existing local Bonsai provider and automatic compaction threshold.
It rejects conflicting caller hook overrides. It does not add a hook-trust bypass flag.

Codex 0.154 requires persistent trust for these exact hook definitions.
The launcher reads hook metadata through the local app-server API before the first model request.
It registers only the three verified definition hashes under `hooks.state` in the current `CODEX_HOME/config.toml`.
It preserves the original config bytes and refuses a different existing hash for the same key.
A registration message identifies the file. Use `/hooks` to inspect these entries.
To remove this registration, remove only the three `Pulse task continuity` state tables. Retain unrelated hook state.
No provider or permission settings are written to that file.

Hooks preserve and retrieve state; they do not infer a reliable task summary from a transcript.
The agent must update the structured checkpoint after meaningful progress and before planned compaction.
Updates require the current revision and include the objective, constraints, decisions, completed work, next steps, blockers, and evidence.
The helper rejects stale revisions. It stores no automatic transcript or raw tool-output dump in the vault.

This explicit task profile enables Jev continuity advice by default.
Set `PULSE_CONTINUITY_JEV=0` to disable that advice.
Only a nonempty new user prompt or an explicit context query invokes the bounded advisory call.
Startup and compaction rehydration do not require Jev. Advice does not change the compaction threshold or override the saved task.
The helper documentation describes the bounded metadata sent to Jev and its failure behavior.

CPU qualification uses the real Codex 0.154 CLI with a local mock Responses endpoint.
It verifies startup, resume, one automatic compaction, existing same-event hook preservation, and immediate checkpoint rehydration.
A completed shell action executes once. The test does not establish days-long reliability or local-model summary quality.
Run `python3 scripts/codex-bonsai-continuity-test.py` to repeat this test without a GPU or external model call.

The event behavior follows the [official hooks reference](https://learn.chatgpt.com/docs/hooks).
The available settings follow the [official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
