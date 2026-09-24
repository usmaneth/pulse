# Qwen3.8-Flash-Next vLLM overlays

An overlay is a change to the vLLM package inside the serving image
`vllm/vllm-openai:qwen38-flash-next`. Pulse owns the generators for the
speculation-layer overlays. The MiaAI-Lab recipe owns the other patches (PLE,
modelopt, QSA FP8 KV, reduced draft vocabulary, PLE offload).

The generators are pinned to the image with RepoDigest
`vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`.
Use the RepoDigest, not the image Id: the Id is different on each node for the
same image.

## Build

```bash
overlays/qwen38/build.sh --out <dir> --set <comma list of overlay names>
overlays/qwen38/build.sh --out overlays/qwen38/out --set best   # block-drop,mtp-fp8-head
```

Names: `block-drop`, `capture`, `mtp-fp8-head`, `lm-head-fp8`. Aliases: `best`
(the serving set) and `all`. An unknown name is an error.

The builder does these steps:

1. It extracts the pristine vLLM files from the image with `docker create` and
   `docker cp`. It does not use the GPU.
2. It runs each generator in `generators/` on the pristine files. For
   `mtp-fp8-head`, the input is the output of the recipe
   `patch_mtp_draft_vocab.py` (see below). Every generator checks that each of
   its anchors occurs exactly once, and stops if not.
3. It compiles every output with `compile()`. `ast.parse` is not
   enough: it accepts a `from __future__` import after another statement,
   and the import of that module then fails in the container.
4. It writes `<dir>/<overlay>/*.py`, `<dir>/docker-args.txt` and
   `<dir>/manifest.json`.

A rebuild with the same inputs writes no file. A changed file is replaced with
`os.replace`, so a container that runs keeps the inode that it mounted.

A rebuild into the same `<dir>` with a smaller set (for example `all`, then
`best`) does not remove the directories of the overlays that are not in the
set. A container that started with an earlier `docker-args.txt` names those
files, and a restart of that container needs them. The builder gives a
warning for each such directory. `docker-args.txt` and `manifest.json` name
only the overlays of the current set. Remove an old directory when no
container uses it.

`docker-args.txt` is one line: the `-e` and `-v` arguments that a recipe
profile adds to `EXTRA_DOCKER_ARGS`. The recipe pastes `EXTRA_DOCKER_ARGS`
into its launch script with no quotes. So the builder accepts only these
characters in each argument: `A-Z a-z 0-9 _ . / : @ + = , -`. The builder
also refuses a mount target that repeats or that the recipe `start.sh` mounts
itself.

`manifest.json` has the keys `image`, `image_digest`, `vllm_pkg`, `sources`
(sha256 of each pristine file) and `overlays`. Each overlay entry has `name`,
`status`, `files`, `dirs`, `env`, `generator` and `requires_profile`. The
`mtp-fp8-head` entry also has `applied_by` and `reference`. The file has no
timestamps, so a rebuild gives the same bytes.

- `files`: a list of `{src, mount_target, mode}`. Each `src` is a generated
  regular file in `<dir>`, and `mode` is `ro`.
- `dirs`: a list of `{src, mount_target, mode}` for host directories that the
  overlay mounts. Only `capture` has one (`--capture-dir` on `/cap`, `rw`).
  The builder does not make or check these directories.

`docker-args.txt` holds the mounts of both lists. It is the source of truth
for a profile. A tool that makes docker arguments from `manifest.json` must
read `files` and `dirs`. If it reads only `files`, it loses the `/cap` mount
of `capture`. The `best` set has no `dirs` entry.

Options:

| Option | Default | Purpose |
|---|---|---|
| `--image` | `vllm/vllm-openai:qwen38-flash-next` | Image to extract from. |
| `--recipe` | `$QWEN38_RECIPE_DIR`, else `/mnt/models/qwen38-flash` | Recipe checkout. `mtp-fp8-head` needs it for its reference file. |
| `--capture-dir` | `$QWEN38_CAPTURE_DIR`, else `/mnt/models/distill/cap` | Host directory that `capture` mounts on `/cap`. |
| `--src` | none | An extracted vLLM package to use instead of the image. |
| `--no-recipe-check` | off | Do not compare with the recipe copy of `patch_mtp_fp8_head.py`. |

`out/` in this directory is ignored by git. Do not commit generated vLLM
files, extracted sources or the reference `mtp_patched.py`. The builder
refuses an `--out` in this repository that git does not ignore.

## Overlays

| Name | Status | Changes | Env | Needs in the profile |
|---|---|---|---|---|
| `block-drop` | shipped | `config/speculative.py`, `v1/core/kv_cache_utils.py`, `v1/core/sched/scheduler.py` | none | `MTP_DISABLE_BLOCK_DROP=1` |
| `mtp-fp8-head` | shipped | `nvidia/mtp.py`, through the recipe hook (no mount) | `VLLM_MTP_DRAFT_HEAD_FP8=1` | `MTP_DRAFT_VOCAB` set |
| `capture` | experimental | `v1/worker/gpu/spec_decode/autoregressive/speculator.py`, plus `/cap` | `VLLM_MTP_CAPTURE_DIR=/cap` | MTP on, V2 model runner |
| `lm-head-fp8` | rejected | `models/qwen3_8_flash_next/nvidia/model.py` | `VLLM_LM_HEAD_FP8=1` | none |

### block-drop (shipped)

Backport of vllm#53388 (`disable_eagle_block_drop`). With an EAGLE-style
drafter (MTP included), vLLM drops the trailing matched prefix-cache block and
computes it again on every request. On Codex traffic that is about one
1664-token block per turn. The overlay adds the config key and keeps the block
when the key is true. The target verifies every draft token, so the change
cannot change output correctness.

Evidence (spark1, one stream, probe-gated codexbench):

- Measured together with `MAX_NUM_BATCHED_TOKENS=8192` (tag
  `ttft-blockdrop-8k`): TTFT on cached Codex turns 1.8 s to 0.8-0.9 s,
  prefill +15%. The `code_edit` case measured 878 ms against 1776 ms for the
  stock recipe (tag `base-mia-K3`). No run measured block-drop without the
  8192-token chunks, so the two changes share this result.
- HumanEval: 79/80.

Turn it on: mount the three files and set `MTP_DISABLE_BLOCK_DROP=1` in the
profile. The scheduler then logs `EAGLE trailing prefix-cache block dropping
is disabled (vllm#53388 backport).` at startup.

Rule: always remove the mounts and the knob together. `start.sh` merges
the knob into the speculative config as `"disable_eagle_block_drop": true`. A
comment in `start.sh` says that a vLLM without the key ignores it. The vLLM
code does not agree: `SpeculativeConfig` uses `extra="forbid"`
(`config/utils.py`), and `engine/arg_utils.py` builds it with
`SpeculativeConfig(**...)`. So the knob without the mounts should fail config
validation. This follows from the code and was not run.

### mtp-fp8-head (shipped)

An FP8 (e4m3, per-row scale) copy of the reduced MTP draft head. The draft
head is a batch-1 GEMV over the reduced vocabulary at each draft step. Only
the argmax is used and the target verifies every draft token, so the output
does not change.

Evidence:

- Draft-head call: 0.83 ms (FP8 rowwise `torch._scaled_mm`) against 1.96 ms
  (BF16 linear) at 47184 x 2560 on GB10.
- Code decode: +5-6% (spark1, one stream).
- HumanEval: 79/80.

Turn it on: set `-e VLLM_MTP_DRAFT_HEAD_FP8=1` and a non-empty
`MTP_DRAFT_VOCAB`. The FP8 copy attaches only after the reduced head exists.
The speculator reads the reduced head only when `start.sh` sets
`use_local_argmax_reduction`. `start.sh` sets it only when `MTP_DRAFT_VOCAB`
is set. The server logs `MTP draft head: FP8 rowwise copy engaged` at load.

How Pulse owns it. The patch applies to `nvidia/mtp.py` after the recipe's
`files/patch_mtp_draft_vocab.py` (MiaAI Lab, AGPL-3.0). At each launch
`start.sh` runs that script, then `files/patch_mtp_fp8_head.py`, and mounts the
result `files/mtp_patched.py` on `nvidia/mtp.py`. An overlay cannot mount the
same target a second time: docker refuses to start with `Duplicate mount
point`. So:

- `generators/patch_mtp_fp8_head.py` is the canonical generator. It gives
  the same output bytes as the recipe file, and it can replace the recipe
  file with no other change. It differs from the recipe file in three ways.
  It takes an optional `TARGET` argument (without it, it patches
  `mtp_patched.py` next to itself, as before). Its anchors are short
  fragments, not full lines of the MiaAI Lab code. It stops when an anchor
  does not occur exactly once.
- The builder runs the full launch chain on the pristine `mtp.py`. First it
  runs the recipe's `patch_mtp_draft_vocab.py` from the recipe checkout. That
  script is never copied into Pulse. Then it runs the Pulse generator. It
  writes the result to `<dir>/mtp-fp8-head/mtp_patched.py` as a reference and
  records its sha256.
- The builder also runs the recipe copy of the generator on the same input.
  The build fails if the two outputs are not byte-equal.
- The overlay emits only `-e VLLM_MTP_DRAFT_HEAD_FP8=1`. Its `files` list is
  empty. Do not mount the reference file.

On the pinned image the reference sha256 is
`ddcb79c3b701fc235488ce5afced291730ac5baabe9d97668fe9e75e8d2c1892`. A server
that runs serves the same bytes when `sha256sum` of `nvidia/mtp.py` inside the
container gives this value.

### capture (experimental)

Adds a hook to the autoregressive speculator `propose()`. When
`VLLM_MTP_CAPTURE_DIR` is set, each prefill chunk of a request whose id holds
`cap-<row>` writes `<dir>/<row>_<start>.pt`. The file holds the
pre-final-mixer multi-stream hidden state that the MTP head reads at draft
step 0. `distill/qwen38/capture_client.py` sends these requests.

Use it only on the offline capture server. Never mount it on a serving
profile. It needs the V2 model runner (`-e VLLM_USE_V2_MODEL_RUNNER=1`) and
`MTP_NUM_SPECULATIVE_TOKENS` greater than 0. The builder mounts the
`--capture-dir` host directory read-write on `/cap`.

Evidence: the round-2 self-distilled MTP head was trained from these
captures. It moved held-out acceptance from 2.960 to 3.088 tokens per step.

### lm-head-fp8 (rejected)

An FP8 (e4m3, per-row scale) copy of the target lm_head (248320 x 2560) for
`compute_logits`, active when `VLLM_LM_HEAD_FP8=1`. It changes the target's
own logits.

Evidence: no speed gain after probe normalization, and 92.9% top-1 agreement
with the BF16 head. The builder builds it when asked, with a warning, for a
repeat test on a future image. Do not use it on a serving profile.

## Items that the profile owns

The overlays replace only a part of `EXTRA_DOCKER_ARGS`. A profile that uses
the builder keeps every key of the recipe `profiles/spark1-best.env` with its
value. In `EXTRA_DOCKER_ARGS`, the builder output replaces only two kinds of
tokens. These are the `-e` arguments with an overlay env key, and the `-v`
mounts on a file in the vLLM package. All other tokens stay.

A node can change only the keys that give a host path or a host check, for
example `HF_HOME` or `REQUIRE_IDLE_GPU`. It must not change a speed, memory
or model knob. Such a change makes a different profile, and its numbers do
not compare with the best profile.

These items are not overlays, and they are easy to lose in a new profile:

- `-e VLLM_USE_V2_MODEL_RUNNER=1`.
- The persistent Triton cache: `-e TRITON_CACHE_DIR=/triton-cache` and
  `-v <host dir>:/triton-cache`. Create the host directory as the user before
  the first start, or docker creates it as root.
- The PLE cache mount: `-v <PLE cache dir>:<PLE cache dir>`. The host path
  and the container path are the same, because the host PLE cache
  (`~/.cache/vllm/ple_cache`) is a symlink to that directory.
- The round-2 MTP shard, mounted read-only over
  `model-00034-of-00034.safetensors` in the HF snapshot.
- `MTP_INDEX_SHARE=1`, `MTP_DISABLE_BLOCK_DROP=1`,
  `MAX_NUM_BATCHED_TOKENS=8192`, `YARN=1` and the KV pin
  (`--kv-cache-memory-bytes 10737418240`). `start.sh` does not keep
  `MTP_INDEX_SHARE` from the environment, so the `.env` value wins.
- The host memory budget: `HOST_RESERVE_GIB=32` and `HOST_SLACK_GIB=4`.
- The fixed chat template:
  `CHAT_TEMPLATE=files/chat-template/chat_template.jinja`.
- The watchdog relief: `MEMWATCH_RELIEF=drop_caches`. Without it, the MemFree
  floor can stop the server while clean page cache is still resident.

A test checks each key and value in this section against the value that
`source` of the profile gives.

Example `EXTRA_DOCKER_ARGS` for the best set (host paths are examples):

```bash
EXTRA_DOCKER_ARGS="-e VLLM_USE_V2_MODEL_RUNNER=1 -e TRITON_CACHE_DIR=/triton-cache -v <Triton cache dir>:/triton-cache -v <PLE cache dir>:<PLE cache dir> <contents of docker-args.txt> -v <r2 shard>:/root/.cache/huggingface/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/<revision>/model-00034-of-00034.safetensors:ro"
```

## Rejected records

These changes are not overlays. The builder does not build them. Each one is
kept in this directory so that the work is not lost and a repeat test can use
it.

### qsa-fused-draft (rejected)

`rejected/qsa_cache.diff` is a unified diff against the pristine
`models/qwen3_8_flash_next/common/qsa_cache.py` of the pinned image. It holds
the hand edit that the recipe kept as `files/ours/qsa_cache.py` (sha256
`0b8079aeab062f3bc71e66571e710b6f757677bf9c7a889368dd4dbc6e29fc42`). That
file was not in any git history.

The edit lets the multi-step MTP draft refresh the QSA metadata in place.
When `VLLM_QSA_FUSED_DRAFT=1` is set, the metadata builder sets
`supports_draft_decode_metadata_update`. The builder then runs the same
metadata kernel again on its persistent buffers between draft steps.

Evidence: A/B 2 (tag `ab2-fusedqsa`) was neutral after probe normalization.
So it is rejected.

A repeat test on this image:

```bash
cid=$(docker create vllm/vllm-openai:qwen38-flash-next /bin/true)
docker cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/common/qsa_cache.py" qsa_cache.py
docker rm "$cid"
patch qsa_cache.py overlays/qwen38/rejected/qsa_cache.diff
```

Then mount `qsa_cache.py` read-only on that path and add
`-e VLLM_QSA_FUSED_DRAFT=1`. A test applies the diff to the pinned image file
and checks the sha256 above.

## Recipe hook follow-up

Today the code that runs at launch is still the recipe copy of
`patch_mtp_fp8_head.py`. The builder check keeps the two copies equal. To make
the Pulse generator the one that runs, change the FP8 line in the recipe
`start.sh` (after `patch_mtp_draft_vocab.py`) to:

```bash
python3 "${MTP_FP8_HEAD_GENERATOR:-$SCRIPT_DIR/files/patch_mtp_fp8_head.py}" "$PATCHED_MTP" || err "patch_mtp_fp8_head.py failed"
```

The recipe copy ignores the argument, so the line is safe before and after
the switch. Then set `MTP_FP8_HEAD_GENERATOR` to
`<pulse>/overlays/qwen38/generators/patch_mtp_fp8_head.py`. Add
`MTP_FP8_HEAD_GENERATOR` to `_ENV_SNAPSHOT_VARS` if an environment value must
win over `.env`.

## Tests

```bash
make overlays-test
python3 -m unittest discover -s overlays/qwen38/tests -v   # the same tests
```

The tests use docker without the GPU. They skip when docker or the pinned
image is not present. The `--out` guard test skips when git is not present.
They check that:

- every generator applies to the pristine image files (`mtp-fp8-head`: to
  the output of the recipe `patch_mtp_draft_vocab.py`);
- every generator fails on an absent anchor and on a repeated anchor, for
  each of its anchors;
- the SRC/OUT generators refuse a source that they already patched, and the
  FP8 generator makes no change on a second run;
- every output compiles with `compile()`, and the build refuses an output
  that `ast.parse` accepts but `compile()` refuses;
- `capture` and `lm-head-fp8` put `import os` after a docstring and a
  `from __future__` import, if the source has them;
- a rebuild writes nothing, and a changed output is replaced with a new inode;
- a rebuild with a smaller set keeps the old overlay directories, warns for
  each one, and writes only the current set to `docker-args.txt`;
- the builder refuses an `--out` in this repository that git does not ignore;
- the output sha256 values equal the pinned values and the files that the
  recipe serves today;
- the `best` arguments equal the overlay mounts in the spark1 best profile,
  with no host paths;
- no mount target repeats or collides with a recipe mount;
- `docker-args.txt` is one shell-safe line, and `manifest.json` matches the
  schema above (each `files[].src` is a regular file, and only `capture` has
  `dirs`);
- `$QWEN38_CAPTURE_DIR` sets the capture directory, and `--capture-dir` wins
  over it;
- `rejected/qsa_cache.diff` applies to the pinned image file with no fuzz and
  gives the recorded sha256;
- each key and value in "Items that the profile owns" equals the profile
  value.
