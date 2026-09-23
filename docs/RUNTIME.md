# Runtime manager: `pulse model`

`pulse model` starts and stops a measured vLLM profile on a DGX Spark node. One
command writes the recipe `.env`, stops the old server, waits for memory,
starts the recipe, waits until `/v1/models` returns 200, and checks the result.
It also writes a backend file for the Pulse gateway.

The recipe is the single-Spark Qwen3.8-Flash-Next recipe (`start.sh`,
`stop.sh`, `.env`) in `/models/usman/qwen38-flash` on each node. Pulse does not
change the recipe. It writes only the recipe `.env`, a backup of it, and a log
file in `logs/`.

## Build

The command is TypeScript. Compile it before use:

```bash
npx tsc
node bin/pulse-cli model --help
```

Do not use `npm run build` for this. That script also runs `make` for the CUDA code.

## Commands

| Command | What it does |
|---|---|
| `pulse model profiles` | Lists the profiles and the nodes. |
| `pulse model up <profile> --node <node>` | Brings the profile up on the node. |
| `pulse model up ... --dry-run` | Prints the rendered `.env` and the script of each step. It runs nothing. |
| `pulse model up ... --dry-run --preflight` | Also runs the read-only checks on the node. |
| `pulse model down --node <node>` | Stops the server on the node. |
| `pulse model status [--node <node>]` | Shows the container, the profile, drift, health, the KV cache size and memory. |
| `pulse model smoke --node <node>` | Sends one short chat completion on the node. A protected node needs `--yes`. |
| `pulse model fragment` | Writes the gateway backend file again from the saved state. |

Options for `up`:

- `--overlay DIR` adds the pairs in `DIR/docker-args.txt` to `EXTRA_DOCKER_ARGS`.
- `--overlay-wins` lets an overlay mount replace a profile mount on the same path.
- `--yes` allows a change on a protected node. spark1 is protected, because it serves live Codex traffic.
- `--experimental` allows a profile with `pulse-status: experimental`.
- `--restart` restarts the server also when the node already runs the same `.env`.
- `--timeout-s N` sets the readiness timeout. The default is `READY_TIMEOUT_S` (1800) plus 300.
- `--min-mem-gib N` sets the memory gate for this run.
- `--verbose` shows the raw `@pulse` records of the node scripts.

Options for `status`: `--json`, `--no-http` (no request to the server), and
`--gpu-probe`. The GPU probe starts a GPU container on the node, so it runs
only when you ask for it.

A protected node gets these limits, also when `status` has no `--node`:

- `status` sends no HTTP request to the server without `--yes`. It still
  shows the container, the `.env`, the KV cache line and memory.
- `status --gpu-probe` refuses without `--yes`. Name the node with `--node`
  to probe only another node.
- `smoke` refuses without `--yes`.

## Files

### runtime/nodes.json

One entry for each node:

| Field | Meaning |
|---|---|
| `host` | `local` (this host) or `ssh`. |
| `ssh` | The ssh target, for example `spark2`. |
| `protected` | `true` makes `up`, `down`, `smoke` and `status --gpu-probe` refuse without `--yes`, and `status` sends no HTTP request without `--yes`. |
| `env` | The node keys of the `.env`: `HF_HOME`, `BIND`, `PORT`, `REQUIRE_IDLE_GPU`. A profile cannot set them. |
| `vars` | Values for `{{name}}` in profiles, for example `tritonCache` and `mtpShardR2`. |
| `gatewayUrl` | The URL that the gateway uses for this node. |
| `gpuProbe` | The command for `status --gpu-probe`. |
| `minMemAvailableGiB` | Optional. It replaces the top-level memory gate for this node. |
| `recipes.<name>` | `dir`, `start`, `stop`, `container`, optional `dockerArgs` and `vars`. |

The top level also sets `minMemAvailableGiB` (100), `memTimeoutS` (600) and
`readyGraceS` (300). The variable `{{recipeDir}}` is always the `dir` of the recipe.

### Profiles: runtime/qwen38/profiles/*.env

A profile is a complete key set for the recipe `.env`. It is not a list of
changes. The reason: `start.sh` has built-in defaults that are different from
`.env.sample`, so a key that is not in the file silently changes the server.

The format is a small, safe part of bash:

```bash
# pulse-recipe: qwen38-flash
# pulse-status: stable
# pulse-description: one line for `pulse model profiles`
# pulse-mkdir: {{tritonCache}}
# pulse-proof: MTP draft head: FP8 rowwise copy engaged
MAX_NUM_SEQS=4
CHAT_TEMPLATE="files/chat-template/chat_template.jinja"
EXTRA_DOCKER_ARGS="
  -e VLLM_USE_V2_MODEL_RUNNER=1
  -v {{tritonCache}}:/triton-cache
"
```

Rules:

- A value in double quotes must not contain `$`, a backtick, a backslash or a double quote.
- A value without quotes can contain only letters, digits and `_ . / : = , @ + % -`.
- Only `EXTRA_DOCKER_ARGS` and `EXTRA_VLLM_ARGS` can use more than one line.
- `EXTRA_DOCKER_ARGS` can contain only `-e NAME=VALUE` and `-v SRC:DST[:ro|rw]` pairs.
  The reason: `start.sh` puts this value without quotes into a generated bash script.
- A key can occur only one time. `HF_TOKEN`, `API_KEY` and `TP1_CONTAINER_NAME` are not allowed.
- `--api-key` is not allowed in `EXTRA_VLLM_ARGS`. Put the key in the node `.env` as `API_KEY`.

Directives:

| Directive | Meaning |
|---|---|
| `pulse-recipe` | Required. The recipe in `recipes` of the node. |
| `pulse-status` | `stable` or `experimental`. |
| `pulse-runnable` | `false` makes the profile render-only. |
| `pulse-mkdir` | A directory that `up` creates on the node before the start. |
| `pulse-proof` | A text that must occur in the container log. A missing text gives a warning. |
| `pulse-client` | A note for the client, which `up` prints. |

The shipped profiles:

| Profile | Use |
|---|---|
| `best` | The best single-Spark profile: FP8 draft head, block-drop backport, MTP index share, the r2 MTP shard, 524288 context. |
| `datagen` | Data generation: 8 sequences, 262144 context, the stock MTP head. |
| `capture` | MTP hidden-state capture. Each request needs a unique `cache_salt` and `X-Request-Id: cap-<row>`. |
| `tp2` | Draft only. Two nodes with tensor parallel size 2. It is render-only (see below). |

### The rendered .env

Each value is written as `KEY="value"`. A header comes first:

```text
# pulse-profile: best
# pulse-node: spark2
# pulse-recipe: qwen38-flash
# pulse-overlay: none
# pulse-rendered: 2026-09-23T10:15:00Z
# pulse-sha256: <sha256 of the assignment lines>
```

The sha256 covers only the assignment lines. `status` uses it to find a file
that somebody edited after the render. With an overlay, the `pulse-overlay`
line holds the overlay directory and `overlay-sha256`, the sha256 of
`docker-args.txt` and `manifest.json`.

The identity of a `.env` is the body sha256, the profile and the
`pulse-overlay` line. `up` writes no new file when all three are the same.

If the old `.env` has `HF_TOKEN=` or `API_KEY=` lines, the node script copies
them to the end of the new file. Pulse never prints these values.

`start.sh` gives `API_KEY` to vLLM as `--api-key`. Then vLLM refuses requests
to `/v1` without the key. So each `curl` of `up`, `status` and `smoke` reads
`API_KEY` from the node `.env` and sends it as a bearer token. The key goes to
`curl` on stdin, so it is not on a command line.

## What `up` does

1. Preflight (read-only). The node script checks the recipe, the image, every
   mount source, the files in the HF cache under each mount target,
   `CHAT_TEMPLATE`, `MTP_DRAFT_VOCAB`, and that no other vLLM container runs. It
   reads the current `.env` in an empty environment and prints the difference
   between the current key map and the render. It also counts the open client
   connections. A failure stops `up` before any change.
2. It creates the `pulse-mkdir` directories.
3. It runs `stop.sh` and makes sure that the container is gone. `stop.sh`
   reads the `.env`, so the old `.env` stays in place until this step is done.
4. It waits until `MemAvailable` is at the gate (100 GiB). `stop.sh` returns
   before the memory is free.
5. It writes the `.env`. First it copies the old file to
   `.env.pulse-bak-<UTC time>`. It does not write the file when the identity
   (the sha256, the profile and the overlay) is the same.
6. It starts `start.sh` in a new session with `setsid`, so a lost ssh
   connection cannot stop it. `env -i` removes the caller environment, so the
   `.env` alone sets the recipe keys. The log goes to `logs/pulse-up-<UTC time>.log`.
7. It waits until `/v1/models` returns 200 on the node.
8. It checks the served model name, `max_model_len`, the KV cache line and the
   `pulse-proof` texts.
9. It writes the node state and the gateway backend file.

If a node already runs a `.env` with the same identity, `up` does only steps
1, 7, 8 and 9. This is true only when the container started after the last
change of the `.env` and after the last change (ctime) of each file that it
mounts. A bind mount of a file keeps the old inode, so a rebuilt overlay file
or a new MTP shard needs a restart, and `up` does it. Use `--restart` to
restart the server in all cases.

If a step fails after the stop step started, `up` marks the node `failed` in
its state, and the gateway backend file disables its endpoint. `up` prints the
backup path if it wrote one, and the command that goes back to the previous
profile. If `down` fails in its stop step, it also marks the node `failed`.

## Safety

- `--dry-run` runs no command on any node and writes no file. It prints the
  rendered `.env` and the full script of each step.
- `--dry-run --preflight` runs only step 1, which only reads.
- Every node script goes to `bash -s` on stdin. The login shell of the node
  does not parse it. On spark2 the login shell is zsh.
- `up` and `down` refuse a protected node without `--yes`. `smoke` and
  `status --gpu-probe` also refuse it, and `status` sends it no HTTP request.
- `up` refuses a render-only profile, and an experimental profile without `--experimental`.
- Pulse never calls `start.sh --no-launch`. That mode starts helper containers
  and writes `.last_launch.sh`.
- Pulse does not copy files between nodes. A missing mount source stops the preflight.

## The gateway backend file

Pulse keeps its state in `$PULSE_STATE_DIR`. The default is
`$XDG_STATE_HOME/pulse/runtime` or `~/.local/state/pulse/runtime`:

- `nodes/<node>.json`: the last `up` or `down` on the node.
- `gateway-backends.json`: a partial gateway config with one `qwen38` model
  and one endpoint for each node.

An endpoint is enabled when the gateway can reach the node and the node state
is not `down`, `failed` or `starting`. The gateway rejects a model with no
enabled endpoint, so the first node stays enabled in that case, with a warning.

The gateway reads its config only when it starts. To use the file:

1. Set `PULSE_GATEWAY_CONFIG` to the file, or start the gateway with `--config <file>`.
2. Make sure that `PULSE_QWEN_BACKENDS` is not set. That variable replaces the
   endpoints of the file. If you run the `pulse-qwen38` user service from
   `deploy/systemd`, remove the variable from `~/.config/pulse/qwen38.env`.
3. Restart the gateway. For the user service: `systemctl --user restart pulse-qwen38`.

Pulse does not do these steps. The file holds only `models`, so the other
gateway settings get their default values.

### spark2 and the gateway

The server on spark2 listens on `127.0.0.1` (`BIND` in `runtime/nodes.json`).
The gateway on spark1 cannot reach it, so the spark2 endpoint stays disabled.
Do not change `BIND` without a decision:

- `BIND=10.99.0.2` breaks the readiness check of `start.sh`, which uses
  `localhost`. `start.sh` then removes the container after the timeout.
- `BIND=0.0.0.0` makes the server open on every interface of spark2, also the
  WLAN and tailscale. The server has no key, and the gateway cannot send one.

Possible solutions: an ssh tunnel from spark1, a change to `start.sh` so that
it checks `$BIND`, or `0.0.0.0` with a firewall.

## Overlays

`overlays/qwen38/build.sh --out DIR --set NAMES` writes `DIR/docker-args.txt`
and `DIR/manifest.json`. `docker-args.txt` holds `-e` and `-v` pairs with white
space between the words. The builder writes all pairs on one line. Pulse also
reads pairs on more than one line, and lines that start with `#` are comments.
The paths must be absolute paths on the node. Pulse writes `manifest.json`
into the node state without a change.

A new build with a different `docker-args.txt` or `manifest.json` changes the
`overlay-sha256` of the `.env`, so `up` writes a new `.env` and restarts the
server. The builder replaces a changed file with a new inode. A running
container does not see the new file, and the mount check of `up` restarts the
server for this case too.

An overlay mount on the same path as a profile mount is an error. Use
`--overlay-wins` to replace the profile mount; the plan then shows each
replacement. A mount on a path that `start.sh` mounts itself (for example
`mtp.py`) is always an error.

## The tp2 profile

`start.sh` of the single-Spark recipe always uses tensor parallel size 1.
Tensor parallel size 2 needs the two-node recipe in `/models/usman/qwen38-dual`.
spark1 is its head node, and its `stop.sh` stops the servers on both nodes. So
`tp2` is render-only: `up tp2 --dry-run` shows the `.env`, and a real `up` refuses.

## Known limits

- The memory gate is 100 GiB, and it depends on `vm.watermark_scale_factor`.
  A higher factor makes `MemAvailable` smaller for the same free memory. With
  factor 300 (a test on 2026-09-23), `MemAvailable` with no server was 96.6 GiB
  on spark1 and 99.4 GiB on spark2, so the memory wait (step 4) would time
  out. The same change also stopped the spark2 server: `MemAvailable` fell
  below the 6 GiB floor of the recipe watchdog. `status` shows the factor. If you keep a high factor,
  set a new gate with `minMemAvailableGiB` in `runtime/nodes.json` or with `--min-mem-gib`.
- `nvidia-smi` on spark2 did not always list the vLLM container. `status` shows
  the list for information only. The container state and memory are the real signals.
- Docker rotates the container log at 50 MB, three files. When the KV cache
  line is gone from the log, `status` shows the value that `up` saved.
