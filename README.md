# Pulse

Pulse is a set of local LLM serving and measurement tools for Codex CLI on NVIDIA DGX Spark (GB10).

## Overview

Pulse connects Codex CLI to a local model through the OpenAI Responses API.
The main path uses vLLM with Qwen3.8-Flash-Next on one DGX Spark node:

- The **Pulse gateway** translates Responses API requests to Chat Completions requests for vLLM.
- The **runtime manager** (`pulse model`) starts, stops and verifies measured vLLM configurations on a node.
- The **overlay generators** make the speculation-layer patches that the vLLM image mounts.
- The **benchmarks** and the **MTP head self-distillation** scripts measure and tune the configuration.

Pulse does not run the model on the production path. vLLM and llama.cpp run the model.
The repository also has an experimental path for Ternary Bonsai 2 27B on llama.cpp.
It also has a research CUDA decoder for that model. The decoder is a correctness diagnostic, and it is slower than llama.cpp.

## Components

Status labels:

- **Stable**: in daily use on the validated node, with automated tests.
- **Experimental**: works on the validated installation. Tests or live runs are limited.
- **Research**: diagnostic or measurement code. It is not a service.
- **Historical**: records of earlier work, kept for reference.
- **Legacy**: earlier code that the current paths do not use.

| Component | Path | Status | Purpose |
|---|---|---|---|
| Pulse gateway | `src/gateway`, `config/`, `deploy/systemd` | Stable (one node) | Responses API for Codex CLI 0.156 over a vLLM backend that serves Qwen3.8-Flash-Next. |
| Gateway failover to a second node | `src/gateway/backends.ts` | Experimental | Ordered endpoint list. Unit tests and a fake-backend drill only. The shipped config disables the second endpoint. |
| Gateway `llamacpp` translation profile | `src/gateway/translate.ts` | Experimental | Chat Completions requests for llama.cpp. Unit tests only. No shipped config and no live run. |
| Runtime manager (`pulse model`) | `src/runtime`, `runtime/`, `bin/pulse-cli` | Experimental | Renders a runtime profile into the recipe `.env`, then stops, starts and verifies vLLM on a node. |
| Runtime profiles `best`, `datagen`, `capture` | `runtime/qwen38/profiles` | Stable | Complete recipe key sets for the measured configurations. |
| Runtime profile `tp2` | `runtime/qwen38/profiles/tp2.env` | Experimental | Two-node tensor parallel draft. Render only. Not measured. |
| vLLM overlay generators | `overlays/qwen38` | Experimental | Generators for speculation-layer patches to the pinned vLLM image. |
| Qwen3.8 benchmarks | `bench/qwen38` | Research | Decode, acceptance, quality and drift A/B tools. Their data files are not in the repository. |
| MTP head self-distillation | `distill/qwen38` | Research | Scripts that build prompts, generate data, capture hidden states, train and export an MTP draft head. |
| Bonsai adapter and launchers | `src/server`, `scripts/codex-bonsai*` | Experimental | Responses and Chat Completions proxy in front of a Prism-compatible llama.cpp `llama-server`. |
| Task continuity hooks | `scripts/codex-bonsai-continuity.py` | Experimental | Stores task state outside the model context, so a task can continue after compaction. CPU tests only. |
| Jev reasoning budget | `src/jev` | Experimental, opt-in | Optional external decision API that sets a reasoning budget in the Bonsai adapter. No measured speed gain. |
| Agent evaluation suite | `bench/agent-eval` | Experimental | Matched evaluation protocol over Chat Completions. No GPU results are recorded. |
| Native CUDA decoder | `src/engine`, `tests/engine` | Research | Sequential decoder for Ternary Bonsai 2 27B PQ2_0, with a llama.cpp reference dumper and comparators. |
| llama.cpp benchmark log and patches | `bench/RESULTS.md`, `bench/*.py`, `patches/` | Historical | Rounds 1 to 43 of Bonsai 2 27B measurements on llama.cpp. |
| Kernel scaffold | `src/cuda`, `include/`, `src/main.cpp`, `src/server/mesh.ts` | Legacy | Synthetic-weight kernels that `make all` builds, and an unused router prototype. Their model constants do not match the real model. |

The repository has no continuous integration. Run the tests in [Development](#development) before a change.

## Supported hardware and models

**Hardware.** All live tests and measurements used NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory).
The CUDA targets compile for sm_121 only. Other GPUs, for example RTX 5090 (sm_120), are not supported or tested.
The gateway and the runtime manager are TypeScript programs. They need no GPU.

**Models.**

| Model | Weights | Backend | Pulse components | Status |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | NVFP4 checkpoint, MTP draft head | vLLM, image `vllm/vllm-openai:qwen38-flash-next` | gateway, `pulse model`, overlays | Stable (gateway) |
| Ternary Bonsai 2 27B | GGUF PQ2_0, DSpark draft GGUF | Prism-compatible llama.cpp `llama-server` | Bonsai adapter, launchers | Experimental |
| Ternary Bonsai 2 27B | GGUF PQ2_0 | Pulse native decoder | `src/engine` | Research |

The [upstream single-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) starts the vLLM server (`start.sh`, `stop.sh`, `.env`).
This repository does not contain the recipe. Pulse writes only the recipe `.env`, a backup of it, and a log file.

**Client.** The measured gateway runs used Codex CLI 0.156.1.
The Bonsai path was validated with Codex CLI 0.154.0. Its CPU tests also pass with 0.156.1.

## Architecture

```text
 Codex CLI
    |  OpenAI Responses API (HTTP, SSE)
    v
 Pulse gateway          src/gateway, 127.0.0.1:8800
    |  OpenAI Chat Completions, endpoints in failover order
    v
 vLLM                   Qwen3.8-Flash-Next NVFP4, MTP speculative decoding, port 8888
    ^                   started by the upstream single-Spark recipe
    |  recipe .env, stop, start, readiness and log checks
 pulse model            src/runtime, runtime/qwen38/profiles, runtime/nodes.json
                        optional overlay mounts from overlays/qwen38
```

The experimental Bonsai path has the same client and a different backend:

```text
 Codex CLI  ->  Bonsai adapter (src/server, 127.0.0.1:18086)  ->  llama-server (Prism-compatible llama.cpp, 127.0.0.1:18085)
```

### Gateway behavior

- **Stateless.** Codex sends the full history on each turn (`store = false`). The gateway refuses `previous_response_id`, `conversation`, `background` and `store: true` with HTTP 400.
- **Failover.** The gateway sends a request to the first healthy endpoint in config order. It moves to the next endpoint after a connect failure, a 3 s connect timeout, or HTTP 502, 503 or 504. It does not move after a backend accepts the request.
- **Retry.** The gateway retries a request that has no output for up to 180 s. After that window, it returns HTTP 503 with `Retry-After`, or a `response.failed` event. It does not retry after the first output event.
- **Keepalive.** During a long prefill, the gateway sends a `response.in_progress` event every 10 s. Codex does not count SSE comment lines for its idle timeout.
- **Prefix warmup.** When an endpoint becomes healthy again, the gateway sends recent prompt prefixes as `max_tokens` 1 requests. This refills the vLLM prefix cache. A warm request goes only when no real request is in flight.
- **Tool calls.** The `qwen38` translation profile sends a forced `tool_choice` as a JSON schema, because vLLM accepted `tool_choice` but did not apply it. The gateway adds absent `*** Begin Patch` and `*** End Patch` lines to an `apply_patch` input.
- **Status.** `GET /health` gives the state of each endpoint. `GET /metrics` gives JSON counters that stay in memory and reset on restart.

The gateway does not supervise vLLM, and `pulse model` does not supervise it after the start.
A request fails when the backend stops after the first output, or when a backend restart takes more than 180 s.

## Quick start: Codex CLI with Qwen3.8-Flash-Next

Requirements:

- A DGX Spark node with a vLLM server for Qwen3.8-Flash-Next at `http://127.0.0.1:8888/v1`. The upstream recipe starts this server.
- Node.js and pnpm. The tests ran with Node.js 24.21.0.
- Codex CLI. The measured runs used 0.156.1.

1. Build and test the gateway. These steps need no GPU and no model server.

   ```sh
   pnpm install --frozen-lockfile
   pnpm exec tsc
   npm run test:gateway
   ```

   Use `pnpm exec tsc` for the TypeScript components. `npm run build` also runs `make`, which needs `nvcc` and builds the legacy kernel scaffold.

2. Start the gateway. It listens on `127.0.0.1:8800`.

   ```sh
   npm run gateway
   ```

   To use a different backend, set `PULSE_QWEN_BACKENDS` to a list of `name=url` pairs in failover order:

   ```sh
   PULSE_QWEN_BACKENDS="node1=http://127.0.0.1:8888/v1" npm run gateway
   ```

3. Check the gateway. The health status is `ok` (HTTP 200) when each model has a healthy endpoint. Else it is `degraded` (HTTP 503).

   ```sh
   curl -s http://127.0.0.1:8800/health | jq .status
   curl -s http://127.0.0.1:8800/v1/models | jq
   ```

4. Add the provider to `~/.codex/config.toml`:

   ```toml
   [model_providers.pulse]
   name = "Pulse (Qwen3.8 on DGX Spark)"
   base_url = "http://127.0.0.1:8800/v1"
   wire_api = "responses"
   stream_idle_timeout_ms = 900000
   request_max_retries = 1
   stream_max_retries = 3
   ```

5. Start Codex with the provider:

   ```sh
   codex -c model_provider=pulse -m qwen3.8-flash-next -c model_auto_compact_token_limit=400000
   ```

Codex reads the context window and the tool types of the model from a model catalog.
[docs/QWEN38.md](docs/QWEN38.md) gives the catalog entry (524,288-token context window, `freeform` `apply_patch`).
Pass the catalog file with `-c model_catalog_json=<file>`.

### Run the gateway as a service

```sh
mkdir -p ~/.config/systemd/user ~/.config/pulse
cp deploy/systemd/pulse-qwen38.service ~/.config/systemd/user/
cp deploy/systemd/pulse-qwen38.env.example ~/.config/pulse/qwen38.env
systemctl --user daemon-reload
systemctl --user enable --now pulse-qwen38.service
```

Edit the unit before you enable it. It expects the repository at `~/pulse`, and `ExecStart` names one nvm Node.js path.
The unit reads `config/qwen38-gateway.json`. The values in `~/.config/pulse/qwen38.env` override it.

Security:

- Keep the default loopback address, or put a proxy with authentication in front of the gateway.
- `PULSE_GATEWAY_API_KEY` protects `/v1/*` only. `/health` and `/metrics` have no authentication.
- `PULSE_GATEWAY_TRACE_FILE` and `PULSE_GATEWAY_WARMUP_STATE_FILE` hold full prompts. Use them for diagnosis only.

[docs/QWEN38.md](docs/QWEN38.md) describes the protocol rules, the configuration variables, failover, warmup, metrics and limits.
Some steps in that document are specific to one installation. The provider block above is the portable setup.

## Runtime manager: `pulse model`

`pulse model` brings a runtime profile up or down on a node through the recipe.
It is experimental. The merged code has 40 offline tests with a fake recipe. One development build ran `up` on a test node.

```sh
pnpm exec tsc
node bin/pulse-cli model profiles
node bin/pulse-cli model up best --node <node> --dry-run
node bin/pulse-cli model status --node <node>
```

`--dry-run` prints the rendered `.env` and the script of each step. It runs no command on a node and writes no file.
`up` does these steps:

1. It runs read-only preflight checks.
2. It stops the old server and waits for free memory (100 GiB `MemAvailable` by default).
3. It writes the recipe `.env` and keeps a backup of the old file.
4. It starts the recipe and waits until `/v1/models` returns 200.
5. It checks the model name, the context length, the KV cache line and the proof lines in the log.
6. It writes the node state and a gateway backend file.

Site configuration:

- `runtime/nodes.json` describes the nodes of one site: hosts, recipe directories, cache paths and gateway URLs. Copy it, edit it for your nodes, and pass it with `--nodes <file>`.
- The profiles copy the keys of a recipe branch with changes that are not in the upstream recipe. On a stock recipe checkout, review each profile key before use.
- The `best` profile mounts a self-distilled MTP head shard. This repository does not publish that shard.
- `up` does not take a GPU lease. `tp2` is render only.

See [docs/RUNTIME.md](docs/RUNTIME.md) for the profile format, the safety rules and the known limits.

## Experimental: Bonsai 2 through llama.cpp

This path is validated on one installation only. A fresh clone cannot run it, because it needs:

- a Prism-compatible llama.cpp build (stock llama.cpp cannot apply the activation transform of Bonsai 2),
- local Ternary Bonsai 2 27B GGUF files,
- a corrected v2 draft GGUF that is not published.

```sh
pnpm install --frozen-lockfile
pnpm exec tsc
scripts/codex-bonsai-install-draft
scripts/codex-bonsai-server
```

Then, in the project that you want to edit, run `<pulse>/scripts/codex-bonsai`. `<pulse>` is the path of this repository.
The server starts `llama-server` on `127.0.0.1:18085` (one slot, 65536-token context) and the adapter on `127.0.0.1:18086`.
The launcher selects the local provider for one Codex process. It does not edit `~/.codex/config.toml`.
If you start the adapter directly (`node dist/server/index.js`), it listens on `0.0.0.0:8000`. Set `HOST=127.0.0.1` to keep it on loopback.
See [docs/CODEX.md](docs/CODEX.md) and [docs/LOCAL-CONTINUITY.md](docs/LOCAL-CONTINUITY.md).

## Research: native CUDA decoder

`src/engine` holds a sequential CUDA decoder for one model: Ternary Bonsai 2 27B in GGUF PQ2_0 format.
The model has 48 Gated DeltaNet layers and 16 full-attention layers.
The decoder checks the model math layer by layer and step by step against llama.cpp. It is not a service.

Limits:

- The decoder rejects any other model shape or metadata.
- Input is raw token IDs. Selection is greedy. Each decoder holds one sequence.
- It has no tokenizer, EOS policy, sampler, server, scheduler, batch support or speculative decoding.
- Prefill runs one token per step. The multi-token GDN kernel (1 to 64 tokens) is a standalone fixture. It is not connected to the decoder.
- The decoder is slower than llama.cpp in every measured case. Round 42 of [bench/RESULTS.md](bench/RESULTS.md) records that no measured regime gives the native path higher throughput than llama.cpp.
- The opt-in CUDA graph (`PULSE_CUDA_GRAPH=1`) and Q8_1 activations (`PULSE_Q8_ACTIVATIONS=1`) make the native path faster. They do not close the gap to llama.cpp.
- Correctness is tested on short forced traces only. The comparator limits (0.05 relative error, 0.999 cosine, identical greedy tokens) are diagnostic limits, not bitwise equivalence. Long-context correctness and task quality are not established.

The build needs a CUDA build of a Prism-compatible llama.cpp tree that defines `GGML_TYPE_PQ2_0`.
Always set `LLAMA_DIR`, and write the output to `build/native`. The binaries in `bin/` are out of date.

```sh
make BIN_DIR=build/native build/native/pulse-engine build/native/pulse-dumpref LLAMA_DIR=<llama.cpp tree> -j1
```

Run the GPU checks only when no other process uses the GPU.
[docs/NATIVE-CORRECTNESS.md](docs/NATIVE-CORRECTNESS.md) gives the reference comparison procedure.

## Measured results

All results come from one DGX Spark (GB10). Each result states its conditions.
The recipe fixed a PLE row defect on 2026-09-23. Qwen3.8 results from before that fix do not apply to the current model.
This section contains no pre-fix results.

### Qwen3.8-Flash-Next decode: MTP K=6 against K=4

Conditions: test node, 2026-09-24, `best` runtime profile with `MAX_NUM_SEQS=4`, one stream, self-distilled r2 MTP head.
Decode speed comes from `bench/qwen38/codexbench.py` with its defaults: temperature 0, thinking off, 600 output tokens, mean of 3 repetitions.
Tokens per step comes from `bench/qwen38/acceptbench.py` on 40 held-out prompts. Each K has two interleaved arms.

| Metric | K=4 (two arms) | K=6 (two arms) |
|---|---|---|
| `code_write` decode, tok/s | 44.0, 64.6 | 69.0, 70.4 |
| `code_edit` decode, tok/s | 63.3, 61.2 | 72.2, 44.1 |
| Mixed prompts, tokens per step | 3.29, 3.40 | 3.75, 3.89 |

Two of the eight decode runs (44.0 and 44.1 tok/s) are outliers. Tokens per step is the more stable signal.
On the production node (warm, same benchmark), K=6 gave 64.4 tok/s (`code_write`) and 64.0 tok/s (`code_edit`).
A K=4 baseline of 2026-09-23 on that node gave 60.0 and 55.0 tok/s.
Record: the header of [runtime/qwen38/profiles/best.env](runtime/qwen38/profiles/best.env).

### Qwen3.8-Flash-Next quality

HumanEval pass@1 was 160 of 164 at temperature 0 (`bench/qwen38/qualitygate.py`, 2026-09-23).
Conditions: MTP K=3, r2 MTP head, recipe with the PLE fix. No HumanEval result at K=6 is recorded.
Output at temperature 0 is not bit-deterministic on this stack.

### Gateway: time to first token

Conditions: production node, 2026-09-23, K=4 runtime profile, one request at a time, Codex CLI 0.156.1 request shape with about 11k prompt tokens.

| Case | Time to first token |
|---|---|
| First turn of a new session, empty prefix cache | 5.7 s |
| First turn of a new session after the prefix warmup | 0.9 s (10,080 of 11,213 prompt tokens cached) |
| First turn after a backend restart, warmup on | 1.0 s client, 0.87 s server |
| First turn after a backend restart, warmup off | 5.7 s |

Record: the "Cache warmup" section of [docs/QWEN38.md](docs/QWEN38.md), and the gateway integration report (unpublished).

### Gateway: Codex CLI integration checks

These results check the transport. They are not a model quality score.

| Check | Conditions | Result |
|---|---|---|
| `codex exec` task suite | 2026-09-24, Codex CLI 0.156.1, K=4 runtime profile, 7 tasks (create, three `apply_patch` fixes, resume, output schema, long output) at efforts low and medium, 1 repetition | 14 of 14 passed, 0 transport errors, 87% of input tokens cached, median wall time 12.2 s |
| tool-eval-bench v2.1.0 | 84 scenarios, 1 trial, Codex Responses request shape, effort medium, temperature 0, seed 42 | Score 86.0 (65 pass, 14 partial, 5 fail), 0 infrastructure errors |
| Gateway process kill | `kill -9` of the gateway during a multi-turn `codex exec` on the production node | systemd restarted the gateway. Codex reconnected once and completed all 7 steps. |
| Backend outage | 120 s outage of a fake backend during `codex exec` | The gateway retried 43 times. `codex exec` exited 0 with all 4 work steps correct. |

The task suite and the tool bench ran on a gateway release before the `apply_patch` repair and the forced `tool_choice` change.
In the task suite, one `apply_patch` call without `*** Begin Patch` failed the Codex check, and the model sent it again.
The published Qwen3.8-Flash-Next tool-eval-bench result (85.4) used 8 trials over direct Chat Completions. The two scores are not directly comparable.
A different local transport scored 89.0 in the same session. These results do not show that the gateway improves tool calls.

### Bonsai adapter

Conditions: Codex CLI 0.154.0, `llama-server` with one slot, the interval-repair fixture in [docs/CODEX.md](docs/CODEX.md).
Five requests reused 0, 0, 11135, 11590 and 11919 prompt tokens.
Their first-token latencies were 10.54, 12.19, 0.354, 0.570 and 0.475 s.
The run includes one cold start and does not give a general latency distribution.
No throughput record exists for the launcher default (corrected v2 draft at fixed K4).

No result on any path establishes 100 tok/s on agent workloads, 90% draft acceptance, or higher throughput than an established engine.

## Repository layout

```text
src/gateway/          Pulse gateway: Responses API to Chat Completions (TypeScript)
src/runtime/          pulse model: profile render, node plan, status, gateway fragment
src/server/           Bonsai adapter over llama.cpp (Responses and Chat Completions)
src/jev/              optional Jev reasoning-budget client for the Bonsai adapter
src/engine/           research CUDA decoder for Ternary Bonsai 2 27B PQ2_0
src/cuda/, include/   legacy kernel scaffold (make all)
runtime/              node configuration and Qwen3.8 runtime profiles
overlays/qwen38/      vLLM speculation-layer overlay generators and tests
distill/qwen38/       MTP head self-distillation scripts
bench/qwen38/         Qwen3.8 A/B benchmarks
bench/agent-eval/     matched agent evaluation suite
bench/                historical llama.cpp benchmark scripts and log (RESULTS.md)
config/               gateway and Bonsai configuration files
deploy/systemd/       systemd user unit for the gateway
scripts/              Codex launchers, continuity hooks and CLI tests for the Bonsai path
tests/engine/         native decoder contract tests and comparators
patches/              llama.cpp patches for the Bonsai path
packages/             staged model card for two llama.cpp Bonsai drafters (GGUF files not in git)
bin/pulse-cli         command-line entry point (pulse model, Bonsai launcher commands)
docs/                 design, protocol and measurement documents
```

Some tracked files contain paths and addresses of one site, for example `runtime/nodes.json`. Treat them as site configuration.

## Development

Build the TypeScript components (about 1 s, no GPU):

```sh
pnpm install --frozen-lockfile
pnpm exec tsc
```

CPU tests. None of them needs a GPU or a model server:

```sh
node --test dist/gateway/*.test.js                    # gateway: 103 tests
node --test dist/runtime/*.test.js                    # runtime manager: 40 tests
node --test dist/server/responses.test.js             # Bonsai adapter: 9 tests
node --test dist/jev/test.js dist/jev/continuity-test.js   # Jev: 13 tests
python3 scripts/codex-launcher-test.py                # Bonsai launcher: 6 tests
make native-cpu-test BIN_DIR=build/cpu                # native decoder contracts and comparators (g++, numpy)
(cd bench/agent-eval && python3 -m unittest)          # evaluation suite: 12 tests
(cd bench/jev-policy && python3 -m unittest)          # Jev policy: 15 tests
```

`npm run test:gateway` and `npm run test:runtime` compile and run the first two suites.

Tests with more requirements:

```sh
make overlays-test                                    # needs docker and the pinned vLLM image; tests skip without them
python3 scripts/codex-bonsai-continuity-test.py       # needs the Codex CLI on PATH; uses a CPU mock backend
python3 scripts/codex-cli-smoke.py --output <dir>     # needs the Codex CLI on PATH
```

The test counts are from `main` on 2026-09-24 with Node.js 24.21.0 and Codex CLI 0.156.1.

For a benchmark result, record the model, the backend, the chat template, the sampler, the context, the draft configuration and the date.
Draft acceptance measures accepted draft tokens. It does not measure task accuracy.
A cached-latency result needs the reported cache reuse. A request flag alone does not prove a cache hit.

## Documentation

| Document | Content |
|---|---|
| [docs/QWEN38.md](docs/QWEN38.md) | Gateway: request translation, configuration, failover, retry, warmup, metrics, limits |
| [docs/RUNTIME.md](docs/RUNTIME.md) | `pulse model`: commands, profile format, `up` steps, safety rules, limits |
| [overlays/qwen38/MANIFEST.md](overlays/qwen38/MANIFEST.md) | vLLM overlays: build procedure, status of each overlay, ownership. Its results are from before the PLE fix. |
| [docs/QWEN38-TUNING.md](docs/QWEN38-TUNING.md) | Qwen3.8 benchmark and distillation tools. Its results table is from before the PLE fix. |
| [docs/CODEX.md](docs/CODEX.md) | Bonsai adapter and launchers: requirements, protocol scope, live checks |
| [docs/LOCAL-CONTINUITY.md](docs/LOCAL-CONTINUITY.md) | Task continuity hooks for the Bonsai path |
| [docs/JEV.md](docs/JEV.md) | Jev reasoning-budget decisions and their measured limits |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Matched evaluation suite and metric definitions |
| [docs/NATIVE-CORRECTNESS.md](docs/NATIVE-CORRECTNESS.md) | Native decoder: build, reference comparison, forced-trace profile |
| [docs/NATIVE-GDN-PREFILL.md](docs/NATIVE-GDN-PREFILL.md) | Multi-token GDN kernel fixture |
| [docs/ARCHITECTURE-SPEC.md](docs/ARCHITECTURE-SPEC.md) | Measured Bonsai 2 27B architecture notes |
| [docs/ENGINE.md](docs/ENGINE.md) | Historical: decision record from before the native decoder. Some statements are out of date. |
| [bench/RESULTS.md](bench/RESULTS.md) | Historical: llama.cpp benchmark log, Rounds 1 to 43, with corrections |
| [docs/HISTORICAL-README.md](docs/HISTORICAL-README.md) | Historical: an earlier README, kept for review |

## License

Apache License 2.0. See [LICENSE](LICENSE).
Parts of `src/engine/pq2_q8.cuh` and `src/engine/gdn_prefill.cuh` adapt llama.cpp code and keep its MIT license text.
