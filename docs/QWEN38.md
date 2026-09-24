# Qwen3.8-Flash-Next behind the Pulse gateway

The Pulse gateway serves the OpenAI Responses API for Codex CLI. It translates
each request to Chat Completions and sends it to a vLLM server that runs
Qwen3.8-Flash-Next on a DGX Spark. It replaces the `qwen38` route of
`~/.codex/model-router.py` for this model.

```
codex  --Responses-->  pulse gateway :8800  --Chat Completions-->  vLLM spark1 :8888
                                            \--(failover)-------->  vLLM spark2 :8888
```

The gateway is stateless. Codex sends the full history on every turn
(`store = false`), and the gateway keeps no conversation state.

## Files

| path | purpose |
|---|---|
| `src/gateway/translate.ts` | Responses to Chat Completions request translation, stream translation |
| `src/gateway/backends.ts` | endpoints, health checks, failover order, backend HTTP client |
| `src/gateway/server.ts` | HTTP server: `/v1/responses`, `/v1/models`, `/health`, `/metrics` |
| `src/gateway/config.ts` | configuration from file and environment |
| `src/gateway/index.ts` | entry point, graceful shutdown |
| `config/qwen38-gateway.json` | production configuration |
| `deploy/systemd/pulse-qwen38.service` | systemd user unit |
| `deploy/systemd/pulse-qwen38.env.example` | environment overrides for the unit |

The gateway does not change `src/server`. The Bonsai proxy on port 8000 works
as before.

## Build, test and run

```
npm run build          # tsc and the CUDA objects
npm run test:gateway   # unit and HTTP tests; no GPU and no model server
npm run gateway        # listens on 127.0.0.1:8800
```

Run on another port for a test next to the production unit:

```
PULSE_GATEWAY_PORT=18800 node dist/gateway/index.js
```

## Switch Codex to the gateway

Do these steps in order. The router on port 8790 can stay up until the last step.

1. Build the repo in `~/pulse` with `npm run build`.
2. Install and start the unit:

   ```
   mkdir -p ~/.config/systemd/user ~/.config/pulse
   cp deploy/systemd/pulse-qwen38.service ~/.config/systemd/user/
   cp deploy/systemd/pulse-qwen38.env.example ~/.config/pulse/qwen38.env
   systemctl --user daemon-reload
   systemctl --user enable --now pulse-qwen38.service
   curl -s http://127.0.0.1:8800/health | jq .status
   ```

3. Add the provider to `~/.codex/config.toml`:

   ```toml
   [model_providers.pulse]
   name = "Pulse (Qwen3.8 on DGX Spark)"
   base_url = "http://127.0.0.1:8800/v1"
   wire_api = "responses"
   # A cold prefill of a long context sends no event for minutes.
   stream_idle_timeout_ms = 900000
   # The gateway already fails over between Sparks.
   request_max_retries = 1
   stream_max_retries = 1
   ```

   If `PULSE_GATEWAY_API_KEY` is set, also add `env_key = "PULSE_GATEWAY_API_KEY"`
   and export the same value in the shell.

4. Change the aliases in `~/.zshrc`. Only `model_provider` changes. The other
   flags stay the same:

   ```sh
   alias codex-qwen='/home/usman/.bun/bin/codex -c model_provider=pulse -m qwen3.8-flash-next -c model_auto_compact_token_limit=400000 -c model_catalog_json=/home/usman/.codex/model_catalog.all.json'
   alias codex-qwen-swarm='/home/usman/.bun/bin/codex -c model_provider=pulse -m qwen3.8-flash-next -c model_auto_compact_token_limit=150000 -c model_catalog_json=/home/usman/.codex/model_catalog.all.json'
   ```

To go back, set `model_provider=all-models` again. The router route for
`qwen3.8-flash-next` does not change.

### Model catalog entry

Codex reads the model metadata from `model_catalog_json`. The entry for
`qwen3.8-flash-next` in `~/.codex/model_catalog.all.json` already has the
correct values. These are the fields that matter for the gateway:

```json
{
  "slug": "qwen3.8-flash-next",
  "display_name": "Qwen3.8 Flash Next (local, spark1)",
  "default_reasoning_level": "medium",
  "supported_reasoning_levels": [
    { "effort": "low", "description": "low reasoning" },
    { "effort": "medium", "description": "medium reasoning" },
    { "effort": "high", "description": "high reasoning" }
  ],
  "shell_type": "shell_command",
  "apply_patch_tool_type": "freeform",
  "supports_reasoning_summaries": true,
  "supports_parallel_tool_calls": true,
  "context_window": 524288,
  "max_context_window": 524288,
  "effective_context_window_percent": 90,
  "input_modalities": ["text"]
}
```

`model_auto_compact_token_limit=400000` starts compaction before the 90%
effective window (471,859 tokens). The swarm alias uses 150000 to keep many
parallel sessions inside the KV cache.

## Request translation

The rules follow `~/.codex/model-router.py` (`responses_to_chat`,
`responses_to_qwen_chat`, `flatten_tools`, `chat_to_response`,
`_stream_bonsai`), which served this model before.

### Reasoning effort

| Codex effort | chat request |
|---|---|
| `none`, `minimal`, `low` | `chat_template_kwargs: {enable_thinking: false}`, no `reasoning_effort` |
| `medium`, `high`, `xhigh` | `chat_template_kwargs: {enable_thinking: true}`, `reasoning_effort: <effort>` |
| not set | nothing; the template default applies (thinking on, effort medium) |

The chat template on the server maps the effort aliases to its own levels.

### Tools

| Codex tool | backend tool |
|---|---|
| `function` | the same function |
| `custom` (freeform, for example `apply_patch`) | a function with one string argument `input`; a `grammar` format goes into the description |
| `namespace` (MCP servers) | one function per nested tool, named `<namespace>__<name>` |
| `tool_search` | dropped; Codex accepts a `tool_search` call only in its own item shape |
| `web_search`, `local_shell`, other hosted tools | dropped |

Codex sends the Lark grammar of `apply_patch` in the tool `format`. A
function tool cannot carry a grammar, so the gateway appends it to the
description (`The input string must follow this lark grammar: ...`).

`tool_choice` maps as follows: `auto`, `none` and `required` pass through. A
named `function` or `custom` tool becomes `{"type": "function", "function":
{"name": ...}}`, with `<namespace>__<name>` for a namespaced tool. A name that
is not in `tools` gives HTTP 400. `parallel_tool_calls` passes through. Both
go to vLLM only when the request has tools, because vLLM rejects
`tool_choice` without tools.

vLLM runs with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`. The
parser returns tool calls as OpenAI `tool_calls` deltas. The gateway joins the
deltas and then:

- restores `<namespace>__<name>` to `namespace` and `name`,
- unwraps `{"input": "..."}` of a custom tool to a `custom_tool_call` with the
  raw `input` text (if the model skipped the JSON wrapper, the raw arguments
  become the input),
- drops a tool call that the token limit cut off, because Codex would run it.

### History

- `instructions` and the leading `developer` or `system` messages merge into
  one system message at the start. A later `developer` message (Codex sends
  one when a setting changes during a session) stays in its position. The
  template renders it in place. If it moved to the start, the prompt prefix
  would change and the vLLM prefix cache would miss for the full history.
- The items of one model turn (reasoning, text, tool calls) merge into one
  assistant message, in that order. That is the order in which the model
  wrote them. Reasoning after text or tool calls starts a new turn, and so
  does text after tool calls.
- The summary text of a `reasoning` item goes back to the model as
  `reasoning_content`. The Qwen3.8 template renders each assistant turn as
  `<think>reasoning</think>content` (`preserve_thinking` is on by default), so
  the history is the text that the model wrote. Without the replay, each
  earlier turn has an empty think block. Set `PULSE_GATEWAY_REPLAY_REASONING=0`
  to turn the replay off.
- Tool calls without an output, and outputs without a call, are dropped. An
  interrupted turn leaves such items.
- A tool output longer than 12,000 characters keeps its first and last 6,000
  characters (`maxToolOutputChars`, 0 disables the cap).
- Image parts and reasoning items without text (for example encrypted
  reasoning from another provider) are dropped.
- The `<recommended_plugins>` block is dropped. Codex sends it in one user
  message together with the AGENTS.md instructions and the
  `<environment_context>` (cwd, shell, date, workspace roots). Only the block
  goes. The rest of the message stays, because without the cwd the model
  searches the whole disk for the files of the task.
- A function call whose arguments are not a JSON object (for example
  `{"cmd": "ls` from a stream that stopped early) goes back with `{}` as its
  arguments. vLLM parses the arguments of each earlier call to render the
  template and refuses the request when they are not an object, so one such
  call would fail every later request of the session. Codex already gave the
  model the parse error as the output of the call. Valid arguments keep their
  exact text.

### Unsupported input

The gateway returns HTTP 400 with a clear message, and does not call the
backend, for:

- `previous_response_id`, `conversation` or `background` (the gateway keeps
  no state),
- `store: true`,
- `text.format` other than `text`, `json_schema` or `json_object`, or a
  `json_schema` format without a `schema` object,
- an input item type other than `message`, `reasoning`, `function_call`,
  `function_call_output`, `custom_tool_call` and `custom_tool_call_output`,
- a message role other than `system`, `developer`, `user` and `assistant`.

Tool types that the backend cannot run are dropped, not refused (see Tools).

### Structured output

`codex exec --output-schema <file>` sends `text.format` of type
`json_schema`. The gateway adds the schema to the end of the system message
as an instruction: the final answer must be one JSON object that matches the
schema. `json_object` gets the same instruction without a schema. The gateway
does not send `response_format` to vLLM. That constraint applies to every
answer, and Codex sends the format on each request of the task, so the model
could not call tools. The gateway does not check the answer against the
schema.

### Stream events

The backend request always streams, with `stream_options.include_usage`. The
gateway sends the Responses events in this order:

```
response.created
response.in_progress
  reasoning item:  output_item.added, reasoning_summary_part.added,
                   reasoning_summary_text.delta..., reasoning_summary_text.done,
                   reasoning_summary_part.done, output_item.done
  message item:    output_item.added, content_part.added, output_text.delta...,
                   output_text.done, content_part.done, output_item.done
  function calls:  output_item.added, function_call_arguments.delta...
                   (for each call, when it arrives)
  -- the backend sends the finish reason --
  function calls:  function_call_arguments.done, output_item.done
  custom calls:    output_item.added, custom_tool_call_input.delta,
                   custom_tool_call_input.done, output_item.done
response.completed | response.incomplete | response.failed
```

Text, reasoning and function-call arguments go out as they arrive. The
message item closes when the first tool call starts. The `done` events of the
function calls wait for the finish reason, because Codex runs a call when its
`output_item.done` arrives. Custom tool calls go out complete after the
finish reason, because their arguments are JSON that the gateway unwraps as a
whole.

Every event has a `sequence_number`. `finish_reason: length` gives
`response.incomplete` with `incomplete_details.reason = max_output_tokens`, and
no tool call of that turn gets a `done` event or stays in the final output.
Codex retries an incomplete turn, so a call that ran would run again. A stream
that ends without a finish reason gives `response.failed`, never
`response.completed`. A stream that has a finish reason but no text, no
reasoning and no tool call also gives `response.failed`. vLLM can generate
tokens and stream none of them, and an empty completed response would end the
Codex task with no answer.

The reasoning text comes from the vLLM `qwen3` reasoning parser
(`reasoning_content` or `reasoning`). Set `PULSE_GATEWAY_EMIT_REASONING=0` to
hide it.

## Endpoints and failover

Each model has an ordered list of endpoints. For each request the gateway tries
the healthy endpoints in list order, then the unhealthy ones as a last resort.
It moves to the next endpoint when:

- the connection fails or does not open within `connectTimeoutMs` (3 s), or
- the backend returns HTTP 502, 503 or 504.

It does not move on after the backend accepts the request. A backend HTTP 4xx
(for example a prompt longer than the context) goes to Codex unchanged. A
failure during the stream gives `response.failed`.

A health check sends `GET <origin>/health` to every enabled endpoint every
`healthIntervalMs` (10 s). A request failure also marks the endpoint unhealthy
at once.

Spark2 is in `config/qwen38-gateway.json` with `"enabled": false`, because its
vLLM server binds to loopback only today. To use it, start that server with
`BIND=10.99.0.2` (or `0.0.0.0` on the private link), then set
`"enabled": true` or add it to `PULSE_QWEN_BACKENDS`:

```
PULSE_QWEN_BACKENDS=spark1=http://127.0.0.1:8888/v1,spark2=http://10.99.0.2:8888/v1
```

## Health and metrics

`GET /health` returns 200 when every model has at least one enabled endpoint
that is not marked unhealthy, and 503 otherwise or during shutdown:

```
curl -s http://127.0.0.1:8800/health | jq
```

`GET /metrics` returns JSON counters since the process start:

- `requests`: total, in_flight, completed, incomplete, failed, cancelled,
  client_errors
- `tokens`: input, cached_input, output, reasoning (from backend usage)
- `tool_calls`
- `backends[]`: per endpoint health, requests, errors, failovers, in_flight,
  and latency windows (count, mean, p50, p95, recent maximum) for the
  response headers, the first token and the full request

```
curl -s http://127.0.0.1:8800/metrics | jq '.backends[] | {name, healthy, requests, p50: .latency.first_token.p50_ms}'
```

## Configuration

The unit reads `config/qwen38-gateway.json`. Environment variables override
it. The file `~/.config/pulse/qwen38.env` is the place for local overrides.

| variable | default | meaning |
|---|---|---|
| `PULSE_GATEWAY_CONFIG` | none | JSON config file (`--config` also works) |
| `PULSE_GATEWAY_HOST` | `127.0.0.1` | listen address |
| `PULSE_GATEWAY_PORT` | `8800` | listen port |
| `PULSE_QWEN_BACKENDS` | `spark1=http://127.0.0.1:8888/v1` | endpoints in failover order |
| `PULSE_QWEN_MODEL` | `qwen3.8-flash-next` | model id that Codex sends |
| `PULSE_QWEN_UPSTREAM_MODEL` | same as the id | vLLM `--served-model-name` |
| `PULSE_GATEWAY_API_KEY` | none | bearer key for `/v1/*` |
| `PULSE_GATEWAY_MODEL_CATALOG` | none | Codex catalog file; `/v1/models` returns its matching entries |
| `PULSE_GATEWAY_EMIT_REASONING` | `1` | `0` hides reasoning items |
| `PULSE_GATEWAY_REPLAY_REASONING` | `1` | `0` does not send earlier reasoning back to the model |
| `PULSE_GATEWAY_TRACE_FILE` | none | append each request and its chat payload, or the reason the gateway refused it, to this JSONL file (the gateway sets mode 0600, also on an existing file; for debugging, it holds full prompts) |
| `PULSE_GATEWAY_MAX_TOOL_OUTPUT_CHARS` | `12000` | tool output cap, 0 disables it |
| `PULSE_GATEWAY_MAX_BODY_BYTES` | 64 MiB | request body limit |
| `PULSE_GATEWAY_CONNECT_TIMEOUT_MS` | `3000` | connect limit per endpoint |
| `PULSE_GATEWAY_HEADERS_TIMEOUT_MS` | `900000` | wait for backend response headers |
| `PULSE_GATEWAY_IDLE_TIMEOUT_MS` | `900000` | longest gap between backend bytes |
| `PULSE_GATEWAY_REQUEST_TIMEOUT_MS` | `3600000` | limit for one request |
| `PULSE_GATEWAY_HEALTH_INTERVAL_MS` | `10000` | health check period, 0 checks once at start |
| `PULSE_GATEWAY_HEALTH_TIMEOUT_MS` | `3000` | health check limit |
| `PULSE_GATEWAY_SHUTDOWN_GRACE_MS` | `30000` | drain time after SIGTERM |
| `PULSE_LOG_LEVEL` | `info` | `debug`, `info`, `warn` or `error` |

The idle and headers limits are long on purpose. vLLM sends no byte while it
runs the prefill, and a cold prefill near the 524k window takes minutes.

## Logs

Each line on stdout or stderr is one JSON object. Each Responses request
writes one `response` line with the request id, backend, effort, status,
time to first token, elapsed time, usage and output item types:

```
journalctl --user -u pulse-qwen38 -o cat | jq 'select(.msg == "response") | {backend, status, first_token_ms, elapsed_ms}'
```

When the request fails, the `error` field of the `response` line holds the
reason, also for a backend 4xx that goes to Codex unchanged. A request that
the gateway refuses before the backend call (bad JSON, unknown model,
unsupported input, wrong API key, shutdown) writes one `request rejected`
line with the request id, HTTP status and message instead:

```
journalctl --user -u pulse-qwen38 -o cat | jq 'select(.msg == "request rejected") | {request_id, status, error}'
```

A `tool call arguments are not a JSON object` warning shows a function call
from the backend that Codex cannot parse.

## Shutdown

On SIGTERM or SIGINT the gateway stops accepting connections, reports
`draining` on `/health`, and lets in-flight requests finish for
`shutdownGraceMs`. After that it cancels the remaining streams with
`response.failed`. A second signal exits at once. The unit gives 45 s before
systemd sends SIGKILL.

## Known limits

- `previous_response_id` is not supported. Codex does not use it with
  `store = false`.
- Images are dropped. The model is text only.
- Hosted tools (`web_search`, `local_shell`) are dropped, as in the router.
- Metrics are in memory and reset on restart. The `response` log line of each
  request is the durable record: journald keeps it, and `jq` can add up tokens,
  statuses and latency for any period. A metrics file would repeat that
  record, and the latency windows cannot merge across restarts.
- Structured output is an instruction only (see Structured output). The
  answer can differ from the schema.
