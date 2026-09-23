# Pulse

Pulse develops local inference and Codex integration for NVIDIA DGX Spark.
The current usable service runs Bonsai through a patched llama.cpp backend.
The native CUDA engine is a separate experimental decoder.

## Current capabilities

- A local Responses adapter supports Codex text, shell tools, patch tools, and tool-result history.
- A launcher selects Bonsai for one Codex process without changing the global provider.
- A managed launcher starts a private backend and adapter on loopback ports.
- The native decoder owns separate recurrent, convolution, and attention state for each layer.
- An optional native TypeSafe Jev decision selects a bounded reasoning budget at the request boundary.
- A sequential evaluation suite records task outcomes, latency, cache reuse, and speculative acceptance.

Pulse does not currently lead llama.cpp on measured complete native decode.
A short matched trace measured about 15.5 native steps/s and 27.0 llama.cpp steps/s.
All 49 greedy predictions matched in that trace. This does not establish long-context correctness or general task quality.
An optional full-step CUDA graph passes exact native parity over the tested traces.
Its paired decode improvement was about 4.3 percent; it did not close the gap to llama.cpp.
A separate opt-in Q8 activation path measured 18.66 versus 14.21 native decode steps/s in three matched graph-mode pairs.
All reference and graph gates passed. This arithmetic change remains experimental and below the earlier llama.cpp result.
The native path accepts token IDs and lacks a tokenizer, EOS policy, server, scheduler, and speculative decoder.

## Local Codex setup

Read [the Codex integration guide](docs/CODEX.md) for backend requirements and configuration.
On the validated Spark installation:

```sh
pnpm install --frozen-lockfile
pnpm exec tsc
scripts/codex-bonsai-install-draft
scripts/codex-bonsai-server
```

Start a second terminal in the project that you want to edit:

```sh
/home/usman/pulse/scripts/codex-bonsai
```

The private service uses ports 18085 and 18086.
The launcher preserves project instructions, permissions, skills, and configured MCP servers.
The default uses the validated corrected v2 draft at fixed K4.
Use `--legacy-v1` for the original draft control.
The guide describes the optional historical depth policy and its backend requirements.
The guide provides a baseline without speculative decode.

## Validation and measurement

- [Codex integration](docs/CODEX.md): protocol scope, live tool checks, and latency records.
- [Native correctness](docs/NATIVE-CORRECTNESS.md): build commands, state checks, and complete-path profiles.
- [Jev decisions](docs/JEV.md): private credentials, typed advice, request control, and measured limits.
- [Matched evaluation](docs/EVALUATION.md): sequential model tests, resource checks, and metric definitions.
- [Architecture specification](docs/ARCHITECTURE-SPEC.md): measured Bonsai architecture conventions.
- [Historical README](docs/HISTORICAL-README.md): previous experiments and claims, retained for review.

```sh
pnpm exec tsc
node --test dist/server/responses.test.js
python3 scripts/codex-launcher-test.py
make native-cpu-test
```

Run GPU tests sequentially with an explicit resource lease.
Record the model, backend, template, sampler, context, and draft configuration with each result.
Acceptance measures accepted draft tokens. It does not measure task accuracy.
Cached latency requires reported cache reuse. A request flag alone does not prove a cache hit.
The model file size does not equal memory traffic per token; embedding lookup reads one row.

No 100 tok/s agent-workload result, 90 percent acceptance result, or engine leadership claim is established.
