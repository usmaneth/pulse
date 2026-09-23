# Jev decision integration

Pulse calls the native TypeSafe API with model `jev-1.13.0`.
It does not use the Vercel gateway for these decisions.
The model supplies advice. Pulse does not execute tools or ask the user from this advice alone.

## Responses request mode

Start the managed stack with `scripts/codex-bonsai-server --jev`.
This option enables one bounded decision request before each Responses backend request.
The base adapter keeps this mode off unless `PULSE_JEV_MODE=request` is set.

The decision receives at most 4096 characters from the latest user message and 64 tool names.
Known credential patterns are removed from that text.
This filter is not a complete secret detector. Treat the bounded text as an external disclosure.
The last tool outcome includes an exit code, output length, and whether the outcome is known.
The decision receives no tool output text, repository scan, vault content, or automatic file attachments.

A high-confidence simple step selects a 512-token backend reasoning budget.
All other results preserve the normal backend budget.
An explicit `X-Pulse-Reasoning-Budget` value takes priority and skips the decision call.
The decision changes only the backend budget field. It does not change messages or tool definitions.
This preserves the rendered prompt prefix when the caller's messages and tools remain stable.

The request timeout is 750 milliseconds. Pulse does not retry the request.
Timeouts, absent credentials, malformed answers, and uncertain answers preserve normal backend behavior.
The default confidence threshold is 0.8; `PULSE_JEV_CONFIDENCE` can raise it to at most 1.
The selected option also needs probability 0.9 and clarification probability at most 0.2.
These thresholds are conservative policy choices, not calibrated correctness guarantees.
No decision cache exists. Each request uses its current bounded state.

The response metadata and optional Responses metrics contain the decision source, model, latency, confidence, status, and selected budget.
They also contain the clarification probability as advice.
They do not contain the supplied text or the key.
The separate opt-in prompt trace retains its existing behavior and can contain request content.

## Explicit advisory command

Build the TypeScript files with `npx tsc`.
Pass one JSON object to `scripts/pulse-decide` through standard input.

```sh
printf '%s\n' '{"task":"Find the service port","evidence":["The repository is available"],"available_inspections":["Read the service configuration"],"unresolved":["Configured port"]}' | scripts/pulse-decide
```

The command sends only the supplied state and fixed questions to TypeSafe.
All four state fields are required. Unknown fields are rejected.
The result includes the source, pinned model, latency, typed answers, and an advisory action.
The actions are `inspect`, `ready`, `clarify`, and `abstain`.
`ready` describes sufficient context. It does not grant permission for a side effect.
`clarify` describes a missing fact. It does not generate or send a question.
An abstention exits with code 2. A valid advisory action exits with code 0.

## Credentials

Set `TYPESAFE_API_KEY`, or use `~/.config/pulse/typesafe.env`.
`TYPESAFE_API_KEY_FILE` can select another file.
The file must belong to the current user and permit no group or other access.
The loader rejects symbolic links and files larger than 16 KiB.
The file contains one `TYPESAFE_API_KEY=value` assignment.
The loader parses that assignment without shell execution.
Do not place the key in the repository, command arguments, logs, or test fixtures.

## Validation and local policies

The client validates the pinned response model and every expected answer.
Choice answers must contain every allowed option, finite probabilities, a normalized distribution, and a valid selected maximum.
Score answers must contain matching levels, finite probabilities, a normalized distribution, and a consistent weighted score.
Noul values and confidence values must remain within zero and one.
HTTP errors expose only their status code. Upstream error bodies are not copied into results.

The legacy speculation, memory-admission, and tool-routing methods use local code only.
Their source fields identify the local policy, and their unavailable confidence values are null.
The memory method is a compatibility headroom rule, not a complete memory allocator.
The proxy does not invoke that memory method.
Numeric cache, kernel, and speculative-depth control stays in local code.

## Current evidence

Twelve CPU tests cover native schemas, failure behavior, cancellation, policy gates, and budget precedence.
The nine Responses protocol tests also pass.
An eight-case synthetic live check selected the expected Choice action in all eight cases.
The conservative Noul gate accepted three actions and abstained on five.
No accepted action was wrong in this small set.
Do not interpret abstention as a request for user clarification.

The request-boundary check selected the short budget for the simple arithmetic case only.
It retained the normal budget for the other seven cases.
Those API calls took approximately 102 to 274 milliseconds on this run.
Actual Codex tests passed one simple task and one coding fixture across off, on, and fixed-budget arms.
The coding off and on arms each needed one failed-patch recovery.
Jev kept the normal budget on all coding turns and added 1.220 seconds of API time.
The simple task selected 512 tokens, but its output did not reach that cap.
No measured token saving or causal end-to-end speed improvement follows from these single runs.
A large speed improvement is a hypothesis, not a measured result.

API contract: [TypeSafe API reference](https://docs.typesafe.ai/api).
Confidence semantics: [TypeSafe confidence](https://docs.typesafe.ai/confidence).

## Predeclared choice-only candidate

`PULSE_JEV_POLICY=choice-only-v2` selects an experimental candidate when request mode is enabled.
The default remains `conservative-v1`.
Both policies use the same questions, confidence threshold, probability threshold, and 512-token budget.
The candidate removes only the clarification-probability gate.
Clarification remains advisory metadata.
An invalid policy name preserves the normal budget without an API request.

The frozen fixture is `src/jev/fixtures/reasoning-policy-v2.json`.
Its twelve cases and correctness criteria precede live outputs.
The earlier eight cases are development evidence, not this holdout.
Run the API-only fixture with an unused result path:

```sh
node dist/jev/holdout.js --live src/jev/fixtures/reasoning-policy-v2.json /tmp/jev-policy-new-results.json
```

The runner makes one API call per state and applies both policies to the same answers.
It records fixture and question hashes and refuses to overwrite an existing result file.
This isolates the gate change from model variation across separate requests.
The runner does not execute tools or measure downstream task correctness.

The unchanged Choice rubric does not classify a short clarification as simple.
Two fixture cases record this coverage limitation explicitly.
Changing that rubric requires another policy comparison; it would not isolate the gate change.
No candidate promotion follows from this classification fixture alone.
Promotion requires no incorrect short budgets, no task regression, and lower paired end-to-end latency including API overhead.
