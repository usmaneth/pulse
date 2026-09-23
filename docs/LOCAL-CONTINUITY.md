# Local task continuity

The continuity helper stores task state outside the model context.
The harness still controls context limits, conversation compaction, and tool execution.
The model updates the checkpoint after meaningful progress. Hooks do not infer task state from raw transcripts.

## State contract

Each checkpoint contains these fields:

| Field | Contents |
|---|---|
| `objective` | The current user objective |
| `constraints` | Limits that remain active |
| `decisions` | Selected approaches and their reasons |
| `completed` | Completed work with verified outcomes |
| `next_steps` | The next concrete actions |
| `blockers` | Failures and unresolved dependencies |
| `evidence` | File paths, test results, and source references |

The objective is text. All other fields are lists of text entries.
Do not store credentials or full tool output in the checkpoint.
Use separate task IDs for independent work in the same project.

## Helper commands

Initialize a task without replacing an existing checkpoint:

```sh
python3 scripts/local_context.py init --project "$PWD" --task retry-fix \
  --objective 'Repair retry behavior without a public API change.'
```

Read the checkpoint and relevant vault excerpts:

```sh
python3 scripts/local_context.py context --project "$PWD" --task retry-fix
```

Prepare a JSON file with all seven fields.
Use the current revision from the context output:

```sh
python3 scripts/local_context.py checkpoint --project "$PWD" --task retry-fix \
  --expected-revision 1 --input /path/to/checkpoint-input.json
```

Search additional vault topics:

```sh
python3 scripts/local_context.py search --query 'retry idempotency' --vault-root "$HOME/vault"
```

The default state root is `~/.local/state/pulse/continuity`.
`PULSE_CONTINUITY_ROOT` or `--state-root` selects another root.
Keep the same project, task ID, and state root across resume operations.

## Checkpoint integrity

The helper identifies the project by its resolved absolute path.
Writes use an exclusive lock, an expected revision, atomic replacement, and disk synchronization.
A stale revision fails without a state change.
The previous revision remains in `previous.json`.
The pre-compaction hook copies the current checkpoint to `precompact.json`.
These files have owner-only permissions. State paths cannot contain symlinks.

The helper does not delete old tasks automatically.
Keep a checkpoint until its task is complete and its evidence remains available elsewhere.
No component resumes side effects automatically after a crash.
The agent must inspect the saved state and tool artifacts before it repeats an action.

## Vault retrieval

The initial retriever uses local lexical matches in filenames and Markdown contents.
It searches `_index`, `projects`, `knowledge`, and `daily`.
It excludes raw agent transcripts, hidden paths, archives, and symlinks.
It returns at most four excerpts with source paths and line numbers.
Each search has directory-entry, depth, file-size, and total-byte limits.
These limits can omit relevant notes in a large vault.
An empty result does not prove that no relevant note exists.

Read the linked source before a decision depends on an excerpt.
Retrieved notes cannot override the current user request.
The helper does not copy the complete vault into the model prompt.

## Jev advice

Set `PULSE_CONTINUITY_JEV=1` to enable the optional advisory call.
The explicit local launch profiles can set this value; `0` disables it.
Build the TypeScript client with `pnpm exec tsc` before use.

One call asks whether to checkpoint, retrieve evidence, or treat the topic as changed.
The helper calls Jev only when a nonempty user prompt or explicit query is available.
Startup rehydration alone does not require an API call.
The request contains the bounded objective, optional query, revision, and item counts.
It excludes vault excerpts, complete checkpoints, and raw transcripts.
Known credential patterns are removed before the request. This filter is not a general data classification system.
The existing private TypeSafe credential loader supplies the key.

The helper caches advice for five minutes by request content and client build identity.
An unchanged request does not repeat the call during that interval.
The client has a 750 ms request timeout. The helper limits the subprocess to two seconds.
Missing credentials, errors, and timeouts select a local fallback.
The receipt records API latency, total helper latency, and cache reuse.

The model can use the advice to choose its next context action.
The advice cannot remove a constraint, change a revision, or disable the harness context limit.
The helper has no context-pressure telemetry. It reports that limitation beside the advice.
There is no measured Jev quality or latency benefit for this workflow yet.

## Validation limits

CPU tests cover revision conflicts, repeated updates, restart reads, path isolation, bounded retrieval, and advisory cache reuse.
Harness tests must verify actual hook execution and checkpoint reuse after compaction.
These tests do not establish multi-day reliability or model recall quality.
TurboQuant changes numerical KV storage. It does not replace this semantic state or its source evidence.

## References

- [Codex lifecycle hooks](https://learn.chatgpt.com/docs/hooks)
- [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
- [QMD local retrieval](https://github.com/tobi/qmd): a possible indexed retrieval replacement for the bounded lexical scan.
- [TurboQuant paper](https://arxiv.org/abs/2504.19874): numerical vector compression, not semantic task memory.
