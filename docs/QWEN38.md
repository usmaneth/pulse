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
   # The gateway sends a keepalive event every 10 s during a prefill, so
   # the idle limit only has to cover a stalled backend.
   stream_idle_timeout_ms = 900000
   # The gateway already fails over between Sparks and retries a request
   # that has no output yet for 180 s. Codex retries a stream that fails
   # after the output started.
   request_max_retries = 1
   stream_max_retries = 3
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

`tool_choice` maps as follows: `auto` and `none` pass through. A named
`function` or `custom` tool becomes `{"type": "function", "function":
{"name": ...}}`, with `<namespace>__<name>` for a namespaced tool. A name that
is not in `tools` gives HTTP 400. `parallel_tool_calls` passes through. Both
go to vLLM only when the request has tools, because vLLM rejects
`tool_choice` without tools.

A forced `tool_choice` (`required` or a named tool) needs a different path on
spark1. vLLM accepts it, but does not apply it: the answer is plain text, and
the finish reason is `tool_calls` or `stop`. The structured output of
`response_format` works. So for the `qwen38` profile the gateway sends no
`tool_choice`. It sends a JSON schema of the allowed calls in
`response_format` instead: an array of `{"name": ..., "parameters": {...}}`
objects, with one item at most for a named tool or `parallel_tool_calls:
false`. The tools stay in the prompt, and the schema applies after the
reasoning block. A system message at the end of the prompt tells the model
that the reply must be a tool call, so that its reasoning selects the tool.
At the end, the message does not change the cached prefix. The gateway turns the JSON answer into call items. An answer
that is not a valid call gives `response.failed`, and the client can retry.
`PULSE_GATEWAY_FORCED_TOOL_CHOICE=native` sends the `tool_choice` to the
backend unchanged. The `llamacpp` profile uses `native` by default. Codex
sends `auto`, so this path is for other Responses clients.

vLLM runs with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`. The
parser returns tool calls as OpenAI `tool_calls` deltas. The gateway joins the
deltas and then:

- restores `<namespace>__<name>` to `namespace` and `name`,
- unwraps `{"input": "..."}` of a custom tool to a `custom_tool_call` with the
  raw `input` text (if the model skipped the JSON wrapper, the raw arguments
  become the input),
- drops a tool call that the token limit cut off, because Codex would run it,
- adds the `*** Begin Patch` and `*** End Patch` lines to an `apply_patch`
  input that starts at its first hunk header (`*** Add File: a.py`) and does
  not have them. Codex refuses such a patch, and the model needs one more turn
  to send it again. The gateway changes nothing else in the input, and it
  writes a `repaired tool call` log line for each repair.

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

## Backend restarts and long prefills

A request that has no output yet survives a short backend restart:

- When no endpoint can take the request (connection refused, connect
  timeout, HTTP 502, 503 or 504), the gateway waits and tries again. The wait
  starts at 250 ms and doubles up to 3 s.
- When the backend stream breaks before the first output event (for example
  vLLM stops during the prefill), the gateway sends the request again. The
  request is stateless, and the prefix cache makes the new prefill short.
- The retries stop after `retryWindowMs` (180 s) from the request start. Then
  the client gets HTTP 503 with `Retry-After`, or `response.failed` with the
  code `backend_unavailable` when the stream already started.

After the first output event the gateway does not retry, because Codex already
shows that output. The stream ends with `response.failed`, and Codex retries
the turn (`stream_max_retries`).

A streaming request starts its SSE stream (headers, `response.created`) when
the backend sends its headers, or after `streamStartMs` (3 s) when the backend
has not answered yet. After the start, a backend 4xx goes to Codex as
`response.failed`. A context-length message gets the code
`context_length_exceeded`, which Codex handles as a full context window. Other
4xx messages get `invalid_prompt`, which Codex does not retry.

vLLM sends no byte during a prefill. Codex ends a stream after
`stream_idle_timeout_ms` (default 300 s) without an SSE event, and it does not
count SSE comment lines: the Codex SSE parser (`eventsource-stream`) drops
comments before the idle timer sees them. So the gateway sends a
`response.in_progress` event after `keepaliveMs` (10 s) without another event.
Codex ignores the event type (`codex-rs/codex-api/src/sse/responses.rs`), but
the event resets its idle timer.

The gateway has no total time limit for a request by default
(`requestTimeoutMs` is 0). A long generation that streams tokens does not get
cut. The backend idle limit (`idleTimeoutMs`) still stops a backend that goes
silent.

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
that is not marked unhealthy, and 503 with `Retry-After` otherwise or during
shutdown. The body has the status (`ok`, `degraded` or `draining`), the
uptime, the in-flight count, the retry count, the retry window, and for each
endpoint: enabled, healthy, in_flight, consecutive_failures, last_check_at,
last_ok_at and last_error:

```
curl -s http://127.0.0.1:8800/health | jq
```

`GET /metrics` returns JSON counters since the process start:

- `requests`: total, in_flight, completed, incomplete, failed, cancelled,
  client_errors, retries (backend attempts repeated before the first output)
- `tokens`: input, cached_input, output, reasoning (from backend usage)
- `tool_calls`, `tool_call_repairs` (patch envelope repairs),
  `forced_tool_choice` (requests that went to the backend with a JSON schema
  of the calls)
- `warmup`: triggers, requests, completed, aborted, errors, skipped, prompt
  and cached tokens of the warm requests, and the last warm result
- `backends[]`: per endpoint health, requests, errors, failovers, in_flight,
  and latency windows (count, mean, p50, p95, recent maximum) for the
  response headers, the first token and the full request

```
curl -s http://127.0.0.1:8800/metrics | jq '.backends[] | {name, healthy, requests, p50: .latency.first_token.p50_ms}'
```

## Cache warmup

A restarted vLLM server has an empty prefix cache. The first turn of each new
Codex session then pays the prefill of the system prompt, the tools and the
environment message: about 11k tokens, 5.7 s to the first token on spark1.
The next turn of a session that was active before the restart pays the
prefill of its full history.

The gateway keeps the most recent prompt prefix of each template variant (the
effort level changes the head of the Qwen3.8 prompt) and the last payload of
the most recent session. When an endpoint becomes healthy after a failure, the
gateway sends these to the endpoint as prefill-only requests (`max_tokens` 1):

1. the prefix: the system message, the environment message and the tools
2. the last payload of the most recent session, when it is less than 30
   minutes old

The gateway sends each item two times. With the Mamba "align" prefix cache,
the state at some block boundaries is not reusable after the first prefill.
The second request computes it again, and its log line shows the cached token
count. On spark1 the first turn of a new session after a warmup took 0.9 s to
the first token (10080 of 11213 prompt tokens cached), against 5.7 s cold.

A warm request goes only when the gateway has no real request in flight, so
the requests that wait in the retry window go first. A real request aborts the
warm request in flight, and vLLM keeps the blocks that the warm request
completed. The gateway does not warm a prefix or a session on an endpoint when
a real request with it reached that endpoint after the endpoint became healthy.

The prefixes are in memory. Set `PULSE_GATEWAY_WARMUP_STATE_FILE` to keep them
in a file (mode 0600), so that the gateway can also warm the backend after its
own restart. The file holds the system prompt, the tools and the environment
message. The gateway writes a new file and renames it, and it writes a due
change also at shutdown.

```
curl -s http://127.0.0.1:8800/metrics | jq .warmup
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
| `PULSE_GATEWAY_FORCED_TOOL_CHOICE` | profile default | `grammar` forces `required` and named tool choices with a JSON schema, `native` sends them to the backend unchanged |
| `PULSE_GATEWAY_TRACE_FILE` | none | append each request and its chat payload, or the reason the gateway refused it, to this JSONL file (the gateway sets mode 0600, also on an existing file; for debugging, it holds full prompts) |
| `PULSE_GATEWAY_MAX_TOOL_OUTPUT_CHARS` | `12000` | tool output cap, 0 disables it |
| `PULSE_GATEWAY_MAX_BODY_BYTES` | 64 MiB | request body limit |
| `PULSE_GATEWAY_CONNECT_TIMEOUT_MS` | `3000` | connect limit per endpoint |
| `PULSE_GATEWAY_HEADERS_TIMEOUT_MS` | `900000` | wait for backend response headers |
| `PULSE_GATEWAY_IDLE_TIMEOUT_MS` | `900000` | longest gap between backend bytes |
| `PULSE_GATEWAY_REQUEST_TIMEOUT_MS` | `0` | limit for one request, 0 sets no limit |
| `PULSE_GATEWAY_RETRY_WINDOW_MS` | `180000` | retry time for a request without output, 0 tries each endpoint once |
| `PULSE_GATEWAY_KEEPALIVE_MS` | `10000` | `response.in_progress` keepalive interval, 0 turns it off |
| `PULSE_GATEWAY_STREAM_START_MS` | `3000` | wait for the backend before the SSE stream starts |
| `PULSE_GATEWAY_HEALTH_INTERVAL_MS` | `10000` | health check period, 0 checks once at start |
| `PULSE_GATEWAY_HEALTH_TIMEOUT_MS` | `3000` | health check limit |
| `PULSE_GATEWAY_SHUTDOWN_GRACE_MS` | `30000` | drain time after SIGTERM |
| `PULSE_GATEWAY_WARMUP` | `1` | `0` turns the cache warmup off |
| `PULSE_GATEWAY_WARMUP_SESSIONS` | `1` | recent sessions to warm after a restart, 0 warms the prefixes only |
| `PULSE_GATEWAY_WARMUP_SESSION_MAX_AGE_MS` | `1800000` | do not warm a session older than this |
| `PULSE_GATEWAY_WARMUP_STATE_FILE` | none | keep the warm prefixes in this file (mode 0600; it holds the system prompt) |
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

On SIGTERM or SIGINT the gateway stops accepting connections and lets
in-flight requests finish for `shutdownGraceMs`. A request that arrives on an
open connection during the drain gets HTTP 503 with `Retry-After: 2` and
`Connection: close`, and Codex retries it. When the grace time ends, the
gateway cancels the remaining streams. Each cancelled stream gets up to 1 s
to send its `response.failed` event before the sockets close, so Codex
retries the turn. A second signal exits at once. The unit gives 45 s before
systemd sends SIGKILL, and it restarts the gateway after any exit.

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
