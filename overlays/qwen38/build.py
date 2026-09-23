#!/usr/bin/env python3
"""Build the Qwen3.8-Flash-Next vLLM overlays.

An overlay is a change to the vLLM package in the serving image. The builder
extracts the pristine vLLM files from the image (docker create + docker cp, no
GPU), runs the generators in generators/, checks every output with ast.parse
and writes the results to OUT:

    OUT/<overlay>/<file>.py   generated files that a profile mounts
    OUT/docker-args.txt       one shell-safe line for EXTRA_DOCKER_ARGS
    OUT/manifest.json         what was built, from which sources

    build.sh --out OUT --set block-drop,mtp-fp8-head
    build.sh --out OUT --set best

A rebuild with the same inputs writes no file. A changed file is replaced
with os.replace, so a running container keeps the inode that it mounted.
See MANIFEST.md for each overlay, its status and its evidence.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATORS = os.path.join(HERE, "generators")

IMAGE_DEFAULT = "vllm/vllm-openai:qwen38-flash-next"
VLLM_PKG = "/usr/local/lib/python3.12/dist-packages/vllm"
RECIPE_DEFAULT = os.environ.get("QWEN38_RECIPE_DIR", "/models/usman/qwen38-flash")
CAPTURE_DIR_DEFAULT = "/models/usman/distill/cap"

MTP_REL = "models/qwen3_8_flash_next/nvidia/mtp.py"

# Registry in canonical order. docker-args.txt and manifest.json follow it.
# "outputs" lists (generated file name, path under VLLM_PKG that it replaces).
OVERLAYS = [
    {
        "name": "block-drop",
        "status": "shipped",
        "generator": "patch_block_drop.py",
        "sources": ["config/speculative.py", "v1/core/kv_cache_utils.py",
                    "v1/core/sched/scheduler.py"],
        "outputs": [("speculative.py", "config/speculative.py"),
                    ("kv_cache_utils.py", "v1/core/kv_cache_utils.py"),
                    ("scheduler.py", "v1/core/sched/scheduler.py")],
        "env": {},
        "requires_profile": {"MTP_DISABLE_BLOCK_DROP": "1"},
    },
    {
        "name": "capture",
        "status": "experimental",
        "generator": "patch_capture.py",
        "sources": ["v1/worker/gpu/spec_decode/autoregressive/speculator.py"],
        "outputs": [("ar_speculator_capture.py",
                     "v1/worker/gpu/spec_decode/autoregressive/speculator.py")],
        "env": {"VLLM_MTP_CAPTURE_DIR": "/cap"},
        "requires_profile": {"MTP_NUM_SPECULATIVE_TOKENS": "greater than 0",
                             "VLLM_USE_V2_MODEL_RUNNER": "1"},
        "warning": "capture is for the offline capture server only. "
                   "Do not use it on a serving profile.",
    },
    {
        "name": "mtp-fp8-head",
        "status": "shipped",
        "generator": "patch_mtp_fp8_head.py",
        "sources": [MTP_REL],
        "outputs": [],
        "env": {"VLLM_MTP_DRAFT_HEAD_FP8": "1"},
        "requires_profile": {"MTP_DRAFT_VOCAB": "non-empty"},
        "applied_by": "recipe start.sh: files/patch_mtp_fp8_head.py runs after "
                      "files/patch_mtp_draft_vocab.py on files/mtp_patched.py, "
                      "which the recipe mounts on " + VLLM_PKG + "/" + MTP_REL
                      + ". Do not mount the reference file: a second mount on "
                      "that target makes docker refuse to start.",
    },
    {
        "name": "lm-head-fp8",
        "status": "rejected",
        "generator": "patch_lm_head_fp8.py",
        "sources": ["models/qwen3_8_flash_next/nvidia/model.py"],
        "outputs": [("nvidia_model_fp8head.py",
                     "models/qwen3_8_flash_next/nvidia/model.py")],
        "env": {"VLLM_LM_HEAD_FP8": "1"},
        "requires_profile": {},
        "warning": "lm-head-fp8 is rejected (no speed gain, 92.9% top-1 "
                   "agreement). Build it only for a repeat test.",
    },
]
BY_NAME = {o["name"]: o for o in OVERLAYS}
ALIASES = {
    "best": ["block-drop", "mtp-fp8-head"],
    "all": [o["name"] for o in OVERLAYS],
}

# Container paths that the recipe start.sh mounts itself. An overlay that
# mounts on one of them makes docker fail with "Duplicate mount point".
RECIPE_TARGETS = {VLLM_PKG + "/" + rel for rel in (
    "models/qwen3_8_flash_next/nvidia/ple_layer.py",
    "model_executor/layers/quantization/modelopt.py",
    "models/qwen3_8_flash_next/nvidia/ops/qsa.py",
    "models/qwen3_8_flash_next/nvidia/qsa.py",
    MTP_REL,
    "model_executor/layers/ple_offload_layer.py",
    "v1/ple_offload/connector.py",
    "v1/ple_offload/worker.py",
    "v1/ple_offload/protocol.py",
)} | {"/root/draft_vocab.txt", "/root/chat_template.jinja",
      "/root/.cache/huggingface", "/root/.cache/vllm"}

# docker-args.txt is pasted unquoted into the recipe launch script, so every
# token must be free of shell metacharacters and whitespace.
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_./:@+=,-]+")


class BuildError(Exception):
    pass


def warn(msg):
    print(f"build.py: warning: {msg}", file=sys.stderr)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_set(text):
    """Expand a comma list of names and aliases into canonical order."""
    wanted = set()
    for item in (x.strip() for x in text.split(",")):
        if not item:
            continue
        if item in ALIASES:
            wanted.update(ALIASES[item])
        elif item in BY_NAME:
            wanted.add(item)
        else:
            known = ", ".join(list(BY_NAME) + list(ALIASES))
            raise BuildError(f"unknown overlay {item!r}. Known: {known}")
    if not wanted:
        raise BuildError("--set is empty")
    return [o["name"] for o in OVERLAYS if o["name"] in wanted]


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise BuildError(f"command failed ({p.returncode}): {' '.join(cmd)}\n"
                         f"{p.stdout}{p.stderr}")
    return p.stdout


def image_digest(image):
    """Return (digest, kind). RepoDigests is the same on every node; Id is not."""
    try:
        p = subprocess.run(
            ["docker", "image", "inspect", image, "--format",
             "{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}|{{.Id}}"],
            capture_output=True, text=True)
    except FileNotFoundError:
        return None, None
    if p.returncode != 0:
        return None, None
    repo, _, image_id = p.stdout.strip().partition("|")
    if "@" in repo:
        return repo.split("@", 1)[1], "RepoDigests"
    return image_id or None, "Id"


def extract(image, rels, dest):
    """Copy VLLM_PKG/<rel> for each rel out of the image into dest/<rel>."""
    cid = run(["docker", "create", image, "/bin/true"]).strip()
    try:
        for rel in rels:
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            run(["docker", "cp", f"{cid}:{VLLM_PKG}/{rel}", target])
    finally:
        subprocess.run(["docker", "rm", cid], capture_output=True)


def generate_mtp_reference(src, recipe, work, stage_dir, recipe_check):
    """Run the launch-time MTP chain and return the manifest reference.

    The chain is the recipe's patch_mtp_draft_vocab.py (not part of this
    repository), then generators/patch_mtp_fp8_head.py. The recipe's own copy
    of the FP8 generator must give the same bytes.
    """
    vocab_script = os.path.join(recipe, "files", "patch_mtp_draft_vocab.py")
    recipe_fp8 = os.path.join(recipe, "files", "patch_mtp_fp8_head.py")
    if not os.path.isfile(vocab_script):
        warn(f"{vocab_script} not found: mtp-fp8-head emits only its env, "
             "with no reference file")
        return None
    chain = os.path.join(work, "mtp-chain")
    os.makedirs(chain)
    shutil.copy(vocab_script, chain)
    shutil.copy(os.path.join(src, MTP_REL),
                os.path.join(chain, "mtp_patched.py.orig"))
    run([sys.executable, os.path.join(chain, "patch_mtp_draft_vocab.py")], cwd=chain)
    vocab_out = os.path.join(chain, "mtp_patched.py")

    os.makedirs(stage_dir, exist_ok=True)
    final = os.path.join(stage_dir, "mtp_patched.py")
    shutil.copy(vocab_out, final)
    out = run([sys.executable, os.path.join(GENERATORS, "patch_mtp_fp8_head.py"), final])
    if "patch_mtp_fp8_head: applied" not in out:
        raise BuildError(f"patch_mtp_fp8_head.py did not apply: {out.strip()}")

    ref = {
        "path": "mtp-fp8-head/mtp_patched.py",
        "sha256": sha256_file(final),
        "container_path": f"{VLLM_PKG}/{MTP_REL}",
        "mount": False,
        "chain": ["recipe files/patch_mtp_draft_vocab.py",
                  "overlays/qwen38/generators/patch_mtp_fp8_head.py"],
        "draft_vocab_script_sha256": sha256_file(vocab_script),
    }
    if not recipe_check:
        ref["recipe_generator_check"] = "skipped"
    elif not os.path.isfile(recipe_fp8):
        warn(f"{recipe_fp8} not found: recipe generator check skipped")
        ref["recipe_generator_check"] = "absent"
    else:
        other = os.path.join(work, "mtp-recipe")
        os.makedirs(other)
        shutil.copy(recipe_fp8, other)
        shutil.copy(vocab_out, os.path.join(other, "mtp_patched.py"))
        run([sys.executable, os.path.join(other, "patch_mtp_fp8_head.py")], cwd=other)
        got = sha256_file(os.path.join(other, "mtp_patched.py"))
        if got != ref["sha256"]:
            raise BuildError(
                "the recipe files/patch_mtp_fp8_head.py gives a different "
                f"mtp_patched.py ({got}) than generators/patch_mtp_fp8_head.py "
                f"({ref['sha256']}). Bring the two generators back in line, or "
                "use --no-recipe-check.")
        ref["recipe_generator_check"] = "equal"
        ref["recipe_generator_sha256"] = sha256_file(recipe_fp8)
    return ref


def check_mounts(mounts):
    seen = set()
    for _src, target, _mode in mounts:
        if target in RECIPE_TARGETS:
            raise BuildError(f"mount target {target} is owned by the recipe start.sh")
        if target in seen:
            raise BuildError(f"mount target {target} occurs more than once")
        seen.add(target)


def docker_args_tokens(env_items, mounts):
    tokens = []
    for key, value in env_items:
        tokens += ["-e", f"{key}={value}"]
    for src, target, mode in mounts:
        tokens += ["-v", f"{src}:{target}" + (":ro" if mode == "ro" else "")]
    for tok in tokens:
        if not SAFE_TOKEN.fullmatch(tok):
            raise BuildError(f"docker argument {tok!r} is not shell-safe")
    return tokens


def install(data, dest):
    """Write bytes to dest only when they differ. Returns True on a write."""
    if os.path.isfile(dest):
        with open(dest, "rb") as f:
            if f.read() == data:
                return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return True


def build(args):
    names = resolve_set(args.set)
    out = os.path.abspath(args.out)
    capture_dir = os.path.abspath(args.capture_dir)
    recipe = os.path.abspath(args.recipe)
    selected = [BY_NAME[n] for n in names]
    for o in selected:
        if "warning" in o:
            warn(o["warning"])

    rels = []
    for o in selected:
        rels += [r for r in o["sources"] if r not in rels]

    digest, digest_kind = image_digest(args.image)
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="qwen38-overlays-")
    try:
        if args.src:
            src = os.path.abspath(args.src)
        else:
            if digest is None:
                raise BuildError(f"image {args.image} not found. Pull it, or pass --src")
            src = os.path.join(work, "src")
            extract(args.image, rels, src)
        sources = {}
        for rel in rels:
            path = os.path.join(src, rel)
            if not os.path.isfile(path):
                raise BuildError(f"source file missing: {path}")
            sources[rel] = sha256_file(path)

        stage = os.path.join(work, "stage")
        manifest_overlays = []
        env_items, mounts = [], []
        for o in selected:
            stage_dir = os.path.join(stage, o["name"])
            entry = {"name": o["name"], "status": o["status"], "env": dict(o["env"]),
                     "files": [], "generator": "generators/" + o["generator"],
                     "requires_profile": dict(o["requires_profile"])}
            if o["name"] == "mtp-fp8-head":
                entry["applied_by"] = o["applied_by"]
                ref = generate_mtp_reference(src, recipe, work, stage_dir,
                                             not args.no_recipe_check)
                entry["reference"] = ref
                if ref is not None:
                    ref["path"] = os.path.join(out, ref["path"])
            else:
                run([sys.executable, os.path.join(GENERATORS, o["generator"]),
                     src, stage_dir])
                for fname, rel in o["outputs"]:
                    if not os.path.isfile(os.path.join(stage_dir, fname)):
                        raise BuildError(f"{o['generator']} did not write {fname}")
                    host = os.path.join(out, o["name"], fname)
                    target = f"{VLLM_PKG}/{rel}"
                    entry["files"].append({"src": host, "mount_target": target,
                                           "mode": "ro"})
                    mounts.append((host, target, "ro"))
            if o["name"] == "capture":
                entry["files"].append({"src": capture_dir, "mount_target": "/cap",
                                       "mode": "rw"})
                mounts.append((capture_dir, "/cap", "rw"))
            env_items += sorted(o["env"].items())
            manifest_overlays.append(entry)

        # Every generated file must be valid Python.
        staged = []
        for root, _dirs, files in os.walk(stage):
            for fname in files:
                path = os.path.join(root, fname)
                with open(path, "rb") as f:
                    data = f.read()
                try:
                    ast.parse(data, filename=path)
                except SyntaxError as e:
                    raise BuildError(f"generated file does not parse: {path}: {e}")
                staged.append((os.path.relpath(path, stage), data))

        check_mounts(mounts)
        tokens = docker_args_tokens(env_items, mounts)
        line = " ".join(tokens)

        manifest = {
            "image": args.image,
            "image_digest": digest,
            "image_digest_source": digest_kind,
            "vllm_pkg": VLLM_PKG,
            "sources": sources,
            "overlays": manifest_overlays,
            "docker_args": line,
        }
        if args.src:
            manifest["source_dir"] = src

        written = []
        for rel, data in sorted(staged):
            if install(data, os.path.join(out, rel)):
                written.append(rel)
        if install((line + "\n").encode(), os.path.join(out, "docker-args.txt")):
            written.append("docker-args.txt")
        text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        if install(text.encode(), os.path.join(out, "manifest.json")):
            written.append("manifest.json")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"overlays: {', '.join(names)}")
    print(f"out: {out}")
    print(f"written: {', '.join(written) if written else 'nothing (no change)'}")
    print(f"docker-args: {line}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the Qwen3.8-Flash-Next vLLM overlays.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--set", required=True,
                    help="comma list of overlays: " + ", ".join(BY_NAME)
                    + " (aliases: best, all)")
    ap.add_argument("--image", default=IMAGE_DEFAULT)
    ap.add_argument("--recipe", default=RECIPE_DEFAULT,
                    help="recipe checkout (default $QWEN38_RECIPE_DIR or %(default)s)")
    ap.add_argument("--capture-dir", default=CAPTURE_DIR_DEFAULT,
                    help="host directory for capture files (default %(default)s)")
    ap.add_argument("--src", help="use an extracted vLLM package instead of the image")
    ap.add_argument("--no-recipe-check", action="store_true",
                    help="do not compare with the recipe copy of patch_mtp_fp8_head.py")
    args = ap.parse_args(argv)
    try:
        return build(args)
    except BuildError as e:
        print(f"build.py: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
