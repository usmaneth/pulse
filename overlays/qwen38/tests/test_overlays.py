"""CPU-only tests for the Qwen3.8-Flash-Next vLLM overlays.

    python3 -m unittest discover -s overlays/qwen38/tests -v

The tests need docker and the pinned image (no GPU). Without them they are
skipped. Tests that compare with the recipe checkout or the serving profile
are skipped when those files are not present. Set QWEN38_RECIPE_DIR and
QWEN38_BEST_PROFILE to point at other copies.
"""
import ast
import collections
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
BUILD_SH = os.path.join(PKG, "build.sh")
GEN = os.path.join(PKG, "generators")

_spec = importlib.util.spec_from_file_location("qwen38_build", os.path.join(PKG, "build.py"))
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

IMAGE = build.IMAGE_DEFAULT
PINNED_DIGEST = "sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
RECIPE = build.RECIPE_DEFAULT
PROFILE = os.environ.get("QWEN38_BEST_PROFILE",
                         os.path.join(RECIPE, "profiles", "spark1-best.env"))

# sha256 of each output on the pinned image. They equal the files that the
# spark1 serving profile mounts today.
PINNED_OUTPUTS = {
    "block-drop/speculative.py":
        "50bbf69d017caf8c13cffd4abb554e087abd86aa88b041754dbea09ab9cbb8f0",
    "block-drop/kv_cache_utils.py":
        "5f52fe0d0ed7e23b949bf9c8a525d121a3e86995c83585201dc15316d01a23a9",
    "block-drop/scheduler.py":
        "be32019fc9e9ecf33b0765c97b1bb09585bff67afdb012d1d74e4da1a4d39894",
    "capture/ar_speculator_capture.py":
        "37e951aaef7de6c60cf75546107b3cb1a58743d207e89a200dfc0b5d2d96c4af",
    "lm-head-fp8/nvidia_model_fp8head.py":
        "77dffeadddc87bd4fb07f1fb8dd9e84ab0e80fa811505c259dd102e1de05c0f7",
}
PINNED_MTP_REFERENCE = "ddcb79c3b701fc235488ce5afced291730ac5baabe9d97668fe9e75e8d2c1892"
# Served copies in the recipe checkout, for a byte compare.
RECIPE_COPIES = {
    "block-drop/speculative.py": "files/ours/speculative.py",
    "block-drop/kv_cache_utils.py": "files/ours/kv_cache_utils.py",
    "block-drop/scheduler.py": "files/ours/scheduler.py",
    "capture/ar_speculator_capture.py": "files/ours/ar_speculator_capture.py",
    "lm-head-fp8/nvidia_model_fp8head.py": "files/ours/nvidia_model_fp8head.py",
    "mtp-fp8-head/mtp_patched.py": "files/mtp_patched.py",
}
# The overlay-owned part of the best profile, with no host paths.
BEST_EXPECTED = sorted([
    ("e", "VLLM_MTP_DRAFT_HEAD_FP8=1", "", ""),
    ("v", "speculative.py", build.VLLM_PKG + "/config/speculative.py", "ro"),
    ("v", "kv_cache_utils.py", build.VLLM_PKG + "/v1/core/kv_cache_utils.py", "ro"),
    ("v", "scheduler.py", build.VLLM_PKG + "/v1/core/sched/scheduler.py", "ro"),
])
SRC_TO_OUT = {
    # generator: (marker, [(source rel, output name)])
    "patch_block_drop.py": ("eagle_block_drop", [
        ("config/speculative.py", "speculative.py"),
        ("v1/core/kv_cache_utils.py", "kv_cache_utils.py"),
        ("v1/core/sched/scheduler.py", "scheduler.py")]),
    "patch_capture.py": ("_capture_hidden", [
        ("v1/worker/gpu/spec_decode/autoregressive/speculator.py",
         "ar_speculator_capture.py")]),
    "patch_lm_head_fp8.py": ("_Fp8LMHeadApply", [
        ("models/qwen3_8_flash_next/nvidia/model.py", "nvidia_model_fp8head.py")]),
}
ALL_SOURCES = sorted({rel for o in build.OVERLAYS for rel in o["sources"]})
# Every anchor of the SRC/OUT generators, by source file. Each occurs once in
# the pristine image file.
SRC_OUT_ANCHORS = {
    "patch_block_drop.py": [
        ("config/speculative.py", "    use_local_argmax_reduction: bool = False\n"),
        ("config/speculative.py",
         '        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")\n'),
        ("v1/core/kv_cache_utils.py",
         "    if spec_config is None or not spec_config.use_eagle():\n"),
        ("v1/core/sched/scheduler.py", "        self.use_eagle = False\n"),
        ("v1/core/sched/scheduler.py",
         "            self.use_eagle = speculative_config.use_eagle()\n"),
        ("v1/core/sched/scheduler.py", "            use_eagle=self.use_eagle,\n"),
        ("v1/core/sched/scheduler.py",
         "        if self.use_eagle:\n"
         "            last_cache_position = max(last_cache_position - block_size, 0)\n"),
    ],
    "patch_capture.py": [
        ("v1/worker/gpu/spec_decode/autoregressive/speculator.py",
         "        self.hidden_states[:num_tokens_padded].copy_(hidden_states)\n"),
    ],
    "patch_lm_head_fp8.py": [
        ("models/qwen3_8_flash_next/nvidia/model.py",
         "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
         "        return self.logits_processor(self.lm_head, hidden_states)\n"),
    ],
}
# The rejected fused-QSA record: a diff against this pristine image file.
QSA_REL = "models/qwen3_8_flash_next/common/qsa_cache.py"
QSA_DIFF = os.path.join(PKG, "rejected", "qsa_cache.diff")
QSA_PRISTINE = "e3460b06cd7ed309e47ad5dfd3d4250890539b912385503133bd98a003f73ba8"
QSA_EDITED = "0b8079aeab062f3bc71e66571e710b6f757677bf9c7a889368dd4dbc6e29fc42"
MANIFEST_MD = os.path.join(PKG, "MANIFEST.md")
# Anchors of patch_mtp_fp8_head.py in the output of patch_mtp_draft_vocab.py.
FP8_ANCHORS = [
    "\n    full_gib = ",
    "\n        logits = torch.nn.functional.linear(",
    "\n\ndef _remap_ignored_layers(",
]


def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def sha256(path):
    return hashlib.sha256(read(path, "rb")).hexdigest()


def image_present():
    if shutil.which("docker") is None:
        return False
    p = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return p.returncode == 0


def parse_args_line(line):
    """Return a multiset of (kind, value-or-basename, target, opts)."""
    toks = shlex.split(line)
    items = []
    for flag, val in zip(toks[::2], toks[1::2]):
        if flag == "-e":
            items.append(("e", val, "", ""))
        elif flag == "-v":
            parts = val.split(":")
            src, target = parts[0], parts[1]
            opts = parts[2] if len(parts) > 2 else ""
            items.append(("v", os.path.basename(src), target, opts))
        else:
            raise AssertionError(f"unexpected token {flag!r}")
    return items


def sourced_values(path, keys):
    """Return {key: value} after bash sources path, as start.sh does."""
    script = 'source "$1" >/dev/null || exit 1; shift; for k; do printf "%s\\0" "${!k-<unset>}"; done'
    p = subprocess.run(["bash", "--noprofile", "--norc", "-c", script, "bash", path, *keys],
                       capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
    if p.returncode != 0:
        raise AssertionError(f"cannot source {path}: {p.stderr}")
    return dict(zip(keys, p.stdout.split("\0")))


def gen(script, *argv):
    return subprocess.run([sys.executable, os.path.join(GEN, script), *argv],
                          capture_output=True, text=True)


def run_build(*argv):
    return subprocess.run([BUILD_SH, *argv], capture_output=True, text=True)


class RegistryTest(unittest.TestCase):
    """Checks that need no docker."""

    def test_aliases_resolve_in_canonical_order(self):
        self.assertEqual(build.resolve_set("best"), ["block-drop", "mtp-fp8-head"])
        self.assertEqual(build.resolve_set("mtp-fp8-head,block-drop"),
                         ["block-drop", "mtp-fp8-head"])
        self.assertEqual(build.resolve_set("all"),
                         ["block-drop", "capture", "mtp-fp8-head", "lm-head-fp8"])

    def test_unknown_name_fails(self):
        with self.assertRaises(build.BuildError):
            build.resolve_set("block-drop,nope")
        with tempfile.TemporaryDirectory() as t:
            p = run_build("--out", t, "--set", "nope")
            self.assertEqual(p.returncode, 1)
            self.assertIn("unknown overlay 'nope'", p.stderr)

    def test_statuses(self):
        self.assertEqual({o["name"]: o["status"] for o in build.OVERLAYS}, {
            "block-drop": "shipped", "capture": "experimental",
            "mtp-fp8-head": "shipped", "lm-head-fp8": "rejected"})

    def test_mount_checks(self):
        t = build.VLLM_PKG + "/config/speculative.py"
        with self.assertRaises(build.BuildError):
            build.check_mounts([("/a", t, "ro"), ("/b", t, "ro")])
        with self.assertRaises(build.BuildError):
            build.check_mounts([("/a", build.VLLM_PKG + "/" + build.MTP_REL, "ro")])
        with self.assertRaises(build.BuildError):
            build.docker_args_tokens([], [("/a b", t, "ro")])
        with self.assertRaises(build.BuildError):
            build.docker_args_tokens([("K", "$(x)")], [])
        with self.assertRaises(build.BuildError):
            build.docker_args_tokens([("K", "v\n")], [])

    def test_out_dir_in_repository_must_be_ignored(self):
        try:
            p = subprocess.run(["git", "-C", PKG, "rev-parse", "--show-toplevel"],
                               capture_output=True, text=True)
        except FileNotFoundError:
            self.skipTest("git not present")
        if p.returncode != 0:
            self.skipTest("the package is not in a git work tree")
        top = p.stdout.strip()
        with self.assertRaises(build.BuildError):
            build.check_out_dir(os.path.join(top, "out"))
        with self.assertRaises(build.BuildError):
            build.check_out_dir(top)
        build.check_out_dir(os.path.join(PKG, "out"))
        build.check_out_dir(os.path.join(PKG, "out", "best"))
        with tempfile.TemporaryDirectory() as t:
            build.check_out_dir(t)
        # The CLI refuses before it extracts or writes anything.
        bad = os.path.join(top, f"qwen38-overlays-guard-test-{os.getpid()}")
        self.assertFalse(os.path.exists(bad))
        self.addCleanup(shutil.rmtree, bad, True)
        p = run_build("--out", bad, "--set", "best")
        self.assertEqual(p.returncode, 1)
        self.assertIn("git does not ignore it", p.stderr)
        self.assertFalse(os.path.exists(bad))

    def test_python_check_uses_compile(self):
        # ast.parse accepts a "from __future__" import after another
        # statement. compile() refuses it, and so must the build.
        late_future = b"import os\nfrom __future__ import annotations\n"
        ast.parse(late_future)
        with self.assertRaises(build.BuildError):
            build.check_python(late_future, "late_future.py")
        with self.assertRaises(build.BuildError):
            build.check_python(b"x = 1\0\n", "null_byte.py")
        build.check_python(b"from __future__ import annotations\nimport os\n", "ok.py")

    def test_leftover_dirs(self):
        with tempfile.TemporaryDirectory() as t:
            for name in ("block-drop", "capture", "lm-head-fp8"):
                os.makedirs(os.path.join(t, name))
            self.assertEqual(build.leftover_dirs(t, ["block-drop", "mtp-fp8-head"]),
                             ["capture", "lm-head-fp8"])
            self.assertEqual(build.leftover_dirs(t, build.resolve_set("all")), [])

    def test_profile_owned_items_match_profile(self):
        # Each KEY=VALUE item that MANIFEST.md lists as owned by the profile
        # must be the value that the profile gives.
        text = read(MANIFEST_MD)
        section = text.split("## Items that the profile owns", 1)[1].split("\n## ", 1)[0]
        items = dict(re.findall(r"`([A-Z][A-Z0-9_]*)=([^`\s]*)`", section))
        for key in ("MEMWATCH_RELIEF", "HOST_RESERVE_GIB", "HOST_SLACK_GIB",
                    "CHAT_TEMPLATE", "MTP_INDEX_SHARE", "MAX_NUM_BATCHED_TOKENS"):
            self.assertIn(key, items)
        if not os.path.isfile(PROFILE):
            self.skipTest(f"profile {PROFILE} not present")
        got = sourced_values(PROFILE, sorted(items))
        for key, value in items.items():
            self.assertEqual(got[key], value, f"MANIFEST.md says {key}={value}")


@unittest.skipUnless(image_present(), f"docker or image {IMAGE} not present")
class RejectedRecordTest(unittest.TestCase):

    def test_qsa_diff_rebuilds_the_rejected_file(self):
        if shutil.which("patch") is None:
            self.skipTest("patch not present")
        with tempfile.TemporaryDirectory() as t:
            build.extract(IMAGE, [QSA_REL], t)
            path = os.path.join(t, QSA_REL)
            self.assertEqual(sha256(path), QSA_PRISTINE)
            p = subprocess.run(["patch", "--fuzz=0", "--no-backup-if-mismatch",
                                path, QSA_DIFF], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertEqual(sha256(path), QSA_EDITED)
            ast.parse(read(path, "rb"), filename=path)
            self.assertIn("VLLM_QSA_FUSED_DRAFT", read(path))
            served = os.path.join(RECIPE, "files", "ours", "qsa_cache.py")
            if os.path.isfile(served):
                self.assertEqual(sha256(served), QSA_EDITED)

    def test_qsa_target_is_not_an_overlay_or_recipe_mount(self):
        target = build.VLLM_PKG + "/" + QSA_REL
        self.assertNotIn(target, build.RECIPE_TARGETS)
        for o in build.OVERLAYS:
            self.assertNotIn(QSA_REL, o["sources"])


@unittest.skipUnless(image_present(), f"docker or image {IMAGE} not present")
class BuildTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="qwen38-overlays-test-")
        cls.src = os.path.join(cls.tmp, "src")
        build.extract(IMAGE, ALL_SOURCES, cls.src)
        cls.out_all = os.path.join(cls.tmp, "all")
        cls.out_best = os.path.join(cls.tmp, "best")
        for out, name in ((cls.out_all, "all"), (cls.out_best, "best")):
            p = run_build("--out", out, "--set", name)
            if p.returncode != 0:
                raise RuntimeError(f"build --set {name} failed:\n{p.stdout}{p.stderr}")
        cls.has_recipe = os.path.isfile(
            os.path.join(RECIPE, "files", "patch_mtp_draft_vocab.py"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def files(self, out):
        got = {}
        for root, _dirs, names in os.walk(out):
            for n in names:
                p = os.path.join(root, n)
                got[os.path.relpath(p, out)] = p
        return got

    # (a) every generator applies on the pristine sources (all anchor counts 1)
    def test_generators_apply_on_pristine_sources(self):
        with tempfile.TemporaryDirectory() as t:
            for script, (_mark, pairs) in SRC_TO_OUT.items():
                p = gen(script, self.src, t)
                self.assertEqual(p.returncode, 0, f"{script}: {p.stdout}{p.stderr}")
                for _rel, name in pairs:
                    self.assertTrue(os.path.isfile(os.path.join(t, name)))

    def mtp_chain(self, t):
        """Run the recipe patch_mtp_draft_vocab.py in t. Return its output path."""
        if not self.has_recipe:
            self.skipTest("recipe patch_mtp_draft_vocab.py not present")
        shutil.copy(os.path.join(RECIPE, "files", "patch_mtp_draft_vocab.py"), t)
        shutil.copy(os.path.join(self.src, build.MTP_REL),
                    os.path.join(t, "mtp_patched.py.orig"))
        subprocess.run([sys.executable, os.path.join(t, "patch_mtp_draft_vocab.py")],
                       cwd=t, check=True, capture_output=True)
        return os.path.join(t, "mtp_patched.py")

    def test_fp8_head_generator_applies_then_is_idempotent(self):
        with tempfile.TemporaryDirectory() as t:
            target = self.mtp_chain(t)
            p = gen("patch_mtp_fp8_head.py", target)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("patch_mtp_fp8_head: applied", p.stdout)
            first = sha256(target)
            self.assertEqual(first, PINNED_MTP_REFERENCE)
            p = gen("patch_mtp_fp8_head.py", target)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("patch_mtp_fp8_head: already applied", p.stdout)
            self.assertEqual(sha256(target), first)

    def test_anchor_count_must_be_one(self):
        # A missing anchor (count 0) and a repeated anchor (count 2) both fail.
        for script, anchors in SRC_OUT_ANCHORS.items():
            for rel, anchor in anchors:
                text = read(os.path.join(self.src, rel))
                self.assertEqual(text.count(anchor), 1, f"{script}: {anchor!r}")
                for count, changed in ((0, text.replace(anchor, "", 1)),
                                       (2, text.replace(anchor, anchor + anchor, 1))):
                    with self.subTest(script=script, anchor=anchor[:50], count=count), \
                            tempfile.TemporaryDirectory() as t:
                        tree = os.path.join(t, "src")
                        shutil.copytree(self.src, tree)
                        write(os.path.join(tree, rel), changed)
                        p = gen(script, tree, os.path.join(t, "out"))
                        self.assertNotEqual(p.returncode, 0)
                        self.assertIn(f"anchor count {count}", p.stderr)

    def test_fp8_head_anchor_count_must_be_one(self):
        with tempfile.TemporaryDirectory() as t:
            target = self.mtp_chain(t)
            text = read(target)
            for anchor in FP8_ANCHORS:
                self.assertEqual(text.count(anchor), 1, repr(anchor))
                for count, changed in ((0, text.replace(anchor, "\n#" + anchor[1:], 1)),
                                       (2, text.replace(anchor, anchor + anchor, 1))):
                    with self.subTest(anchor=anchor, count=count):
                        write(target, changed)
                        p = gen("patch_mtp_fp8_head.py", target)
                        self.assertNotEqual(p.returncode, 0)
                        self.assertIn(f"anchor count {count}", p.stderr)
                        self.assertEqual(read(target), changed, "the file was changed")

    # (b) every output compiles
    def test_outputs_compile(self):
        n = 0
        for rel, path in self.files(self.out_all).items():
            if rel.endswith(".py"):
                compile(read(path, "rb"), path, "exec", dont_inherit=True)
                n += 1
        self.assertEqual(n, 6 if self.has_recipe else 5)

    def test_import_os_goes_after_future_imports(self):
        # A future image can start with a docstring and a __future__ import,
        # and have no "import os". The output must still compile.
        head = ('"""Module docstring."""\n'
                "from __future__ import (\n    annotations,\n)\n")
        for script, (_mark, pairs) in SRC_TO_OUT.items():
            if script == "patch_block_drop.py":
                continue  # it adds no import
            (rel, name), = pairs
            with self.subTest(script=script), tempfile.TemporaryDirectory() as t:
                tree = os.path.join(t, "src")
                shutil.copytree(self.src, tree)
                text = read(os.path.join(tree, rel))
                self.assertNotIn("\nimport os\n", text)
                write(os.path.join(tree, rel), head + text)
                p = gen(script, tree, os.path.join(t, "out"))
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                path = os.path.join(t, "out", name)
                data = read(path, "rb")
                compile(data, path, "exec", dont_inherit=True)
                body = ast.parse(data).body
                self.assertIsInstance(body[0], ast.Expr)
                self.assertEqual(body[1].module, "__future__")
                self.assertIsInstance(body[2], ast.Import)
                self.assertEqual([a.name for a in body[2].names], ["os"])

    def test_smaller_set_keeps_and_reports_leftover_dirs(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "out")
            p = run_build("--out", out, "--set", "all", "--src", self.src)
            self.assertEqual(p.returncode, 0, p.stderr)
            p = run_build("--out", out, "--set", "best", "--src", self.src)
            self.assertEqual(p.returncode, 0, p.stderr)
            for name in ("capture", "lm-head-fp8"):
                self.assertIn(f"{os.path.join(out, name)} is from an earlier build", p.stderr)
                self.assertTrue(os.path.isdir(os.path.join(out, name)))
            self.assertNotIn(os.path.join(out, "block-drop") + " is from", p.stderr)
            line = read(os.path.join(out, "docker-args.txt"))
            self.assertEqual(sorted(parse_args_line(line)), BEST_EXPECTED)
            m = json.loads(read(os.path.join(out, "manifest.json")))
            self.assertEqual([o["name"] for o in m["overlays"]], ["block-drop", "mtp-fp8-head"])

    # (c) idempotency
    def test_rebuild_writes_nothing(self):
        before = {rel: (os.stat(p).st_ino, os.stat(p).st_mtime_ns, sha256(p))
                  for rel, p in self.files(self.out_all).items()}
        p = run_build("--out", self.out_all, "--set", "all")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("written: nothing (no change)", p.stdout)
        after = {rel: (os.stat(p).st_ino, os.stat(p).st_mtime_ns, sha256(p))
                 for rel, p in self.files(self.out_all).items()}
        self.assertEqual(before, after)

    def test_changed_output_is_replaced_not_rewritten(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "out")
            p = run_build("--out", out, "--set", "block-drop")
            self.assertEqual(p.returncode, 0, p.stderr)
            dest = os.path.join(out, "block-drop", "scheduler.py")
            with open(dest, "a") as f:
                f.write("# local edit\n")
            ino = os.stat(dest).st_ino
            p = run_build("--out", out, "--set", "block-drop")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("block-drop/scheduler.py", p.stdout)
            self.assertNotEqual(os.stat(dest).st_ino, ino)
            self.assertEqual(sha256(dest), PINNED_OUTPUTS["block-drop/scheduler.py"])

    def test_generators_refuse_their_own_output(self):
        for script, (mark, pairs) in SRC_TO_OUT.items():
            for rel, name in pairs:
                with self.subTest(script=script, source=rel), \
                        tempfile.TemporaryDirectory() as t:
                    first = os.path.join(t, "first")
                    self.assertEqual(gen(script, self.src, first).returncode, 0)
                    tree = os.path.join(t, "tree")
                    shutil.copytree(self.src, tree)
                    shutil.copy(os.path.join(first, name), os.path.join(tree, rel))
                    p = gen(script, tree, os.path.join(t, "second"))
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn("source is already patched", p.stderr)
                    self.assertIn(mark, p.stderr)

    # (d) outputs equal the pinned and served files
    def test_outputs_match_pinned_sha256(self):
        got = self.files(self.out_all)
        for rel, want in PINNED_OUTPUTS.items():
            self.assertEqual(sha256(got[rel]), want, rel)
        if self.has_recipe:
            self.assertEqual(sha256(got["mtp-fp8-head/mtp_patched.py"]),
                             PINNED_MTP_REFERENCE)

    def test_outputs_match_recipe_served_files(self):
        if not os.path.isdir(os.path.join(RECIPE, "files", "ours")):
            self.skipTest("recipe checkout not present")
        got = self.files(self.out_all)
        for rel, served in RECIPE_COPIES.items():
            with self.subTest(file=rel):
                with open(got[rel], "rb") as a, open(os.path.join(RECIPE, served), "rb") as b:
                    self.assertEqual(a.read(), b.read(), f"{rel} != {served}")

    # (e) best set against the serving profile
    def test_best_args_static(self):
        line = read(os.path.join(self.out_best, "docker-args.txt"))
        self.assertEqual(sorted(parse_args_line(line)), BEST_EXPECTED)
        for tok in shlex.split(line):
            if tok.startswith("/"):
                self.assertTrue(tok.startswith(self.out_best + "/"), tok)

    def test_best_args_match_profile(self):
        if not os.path.isfile(PROFILE):
            self.skipTest(f"profile {PROFILE} not present")
        lines = read(PROFILE).splitlines()
        extra = [l for l in lines if l.startswith("EXTRA_DOCKER_ARGS=")][-1]
        value = shlex.split(extra.split("=", 1)[1], comments=True)[0]
        env_keys = {k for o in build.OVERLAYS for k in o["env"]}
        profile_items = [
            it for it in parse_args_line(value)
            if (it[0] == "e" and it[1].split("=", 1)[0] in env_keys)
            or (it[0] == "v" and it[2].startswith(build.VLLM_PKG + "/"))
        ]
        ours = parse_args_line(read(os.path.join(self.out_best, "docker-args.txt")))
        self.assertEqual(collections.Counter(profile_items), collections.Counter(ours))
        knobs = [l.split("#", 1)[0].strip() for l in lines]
        self.assertIn("MTP_DISABLE_BLOCK_DROP=1", knobs)
        vocab = [k for k in knobs if k.startswith("MTP_DRAFT_VOCAB=")]
        self.assertTrue(vocab and vocab[-1] != "MTP_DRAFT_VOCAB=", "MTP_DRAFT_VOCAB is empty")

    # (f) no collision with the recipe mounts
    def test_no_recipe_owned_targets(self):
        for out in (self.out_all, self.out_best):
            line = read(os.path.join(out, "docker-args.txt"))
            targets = [it[2] for it in parse_args_line(line) if it[0] == "v"]
            self.assertEqual(len(targets), len(set(targets)))
            self.assertFalse(set(targets) & build.RECIPE_TARGETS)

    # (g) docker-args.txt format
    def test_docker_args_line_format(self):
        for out in (self.out_all, self.out_best):
            raw = read(os.path.join(out, "docker-args.txt"))
            self.assertTrue(raw.endswith("\n"))
            self.assertEqual(raw.count("\n"), 1)
            toks = shlex.split(raw)
            self.assertEqual(" ".join(shlex.quote(t) for t in toks), raw.strip())
            for t in toks:
                self.assertRegex(t, r"^[A-Za-z0-9_./:@+=,-]+$")
            manifest = json.loads(read(os.path.join(out, "manifest.json")))
            self.assertEqual(manifest["docker_args"], raw.strip())

    # (h) manifest schema
    def test_manifest_schema(self):
        m = json.loads(read(os.path.join(self.out_all, "manifest.json")))
        self.assertEqual(m["image"], IMAGE)
        self.assertEqual(m["image_digest"], PINNED_DIGEST)
        self.assertEqual(m["vllm_pkg"], build.VLLM_PKG)
        self.assertEqual(sorted(m["sources"]), ALL_SOURCES)
        for rel, digest in m["sources"].items():
            self.assertEqual(digest, sha256(os.path.join(self.src, rel)), rel)
        names = [o["name"] for o in m["overlays"]]
        self.assertEqual(names, ["block-drop", "capture", "mtp-fp8-head", "lm-head-fp8"])
        status = {o["name"]: o["status"] for o in build.OVERLAYS}
        for o in m["overlays"]:
            self.assertEqual(o["status"], status[o["name"]])
            self.assertIsInstance(o["env"], dict)
            self.assertIsInstance(o["files"], list)
            for f in o["files"]:
                self.assertTrue(os.path.isabs(f["mount_target"]))
                self.assertTrue(os.path.isfile(f["src"]), f["src"])
                self.assertEqual(f["mode"], "ro")
            want_dirs = []
            if o["name"] == "capture":
                want_dirs = [{"src": build.CAPTURE_DIR_DEFAULT, "mount_target": "/cap",
                              "mode": "rw"}]
            self.assertEqual(o["dirs"], want_dirs)
        fp8 = m["overlays"][2]
        self.assertEqual(fp8["files"], [])
        self.assertEqual(fp8["env"], {"VLLM_MTP_DRAFT_HEAD_FP8": "1"})
        if self.has_recipe:
            self.assertEqual(fp8["reference"]["sha256"], PINNED_MTP_REFERENCE)
            self.assertFalse(fp8["reference"]["mount"])
            self.assertEqual(fp8["reference"]["recipe_generator_check"], "equal")

    def test_manifest_has_no_volatile_fields(self):
        text = read(os.path.join(self.out_all, "manifest.json"))
        self.assertIsNone(re.search(r"20\d\d-\d\d-\d\dT", text))
        self.assertNotIn("qwen38-overlays-", text.replace(self.tmp, ""))

    def test_build_from_extracted_source(self):
        with tempfile.TemporaryDirectory() as t:
            p = run_build("--out", t, "--set", "all", "--src", self.src)
            self.assertEqual(p.returncode, 0, p.stderr)
            for rel, want in PINNED_OUTPUTS.items():
                self.assertEqual(sha256(os.path.join(t, rel)), want, rel)

    def test_capture_dir_env_override(self):
        with tempfile.TemporaryDirectory() as t:
            cap = os.path.join(t, "cap-dir")
            out = os.path.join(t, "out")
            env = dict(os.environ, QWEN38_CAPTURE_DIR=cap)
            p = subprocess.run([BUILD_SH, "--out", out, "--set", "capture", "--src", self.src],
                               capture_output=True, text=True, env=env)
            self.assertEqual(p.returncode, 0, p.stderr)
            m = json.loads(read(os.path.join(out, "manifest.json")))
            self.assertEqual(m["overlays"][0]["dirs"],
                             [{"src": cap, "mount_target": "/cap", "mode": "rw"}])
            toks = shlex.split(read(os.path.join(out, "docker-args.txt")))
            self.assertIn(("-v", f"{cap}:/cap"), list(zip(toks[::2], toks[1::2])))
            # The option still wins over the environment.
            other = os.path.join(t, "other")
            p = subprocess.run([BUILD_SH, "--out", out, "--set", "capture", "--src", self.src,
                                "--capture-dir", other], capture_output=True, text=True, env=env)
            self.assertEqual(p.returncode, 0, p.stderr)
            m = json.loads(read(os.path.join(out, "manifest.json")))
            self.assertEqual(m["overlays"][0]["dirs"][0]["src"], other)

    def test_recipe_generator_drift_fails_the_build(self):
        if not self.has_recipe:
            self.skipTest("recipe checkout not present")
        with tempfile.TemporaryDirectory() as t:
            fake = os.path.join(t, "recipe")
            os.makedirs(os.path.join(fake, "files"))
            shutil.copy(os.path.join(RECIPE, "files", "patch_mtp_draft_vocab.py"),
                        os.path.join(fake, "files"))
            text = read(os.path.join(GEN, "patch_mtp_fp8_head.py"))
            text = text.replace("clamp_min(1e-12) / 448.0", "clamp_min(1e-12) / 440.0", 1)
            write(os.path.join(fake, "files", "patch_mtp_fp8_head.py"), text)
            out = os.path.join(t, "out")
            p = run_build("--out", out, "--set", "mtp-fp8-head", "--recipe", fake)
            self.assertEqual(p.returncode, 1)
            self.assertIn("gives a different", p.stderr)
            p = run_build("--out", out, "--set", "mtp-fp8-head", "--recipe", fake,
                          "--no-recipe-check")
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_no_recipe_emits_env_only(self):
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, "out")
            p = run_build("--out", out, "--set", "mtp-fp8-head",
                          "--recipe", os.path.join(t, "none"))
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("emits only its env", p.stderr)
            self.assertEqual(read(os.path.join(out, "docker-args.txt")),
                             "-e VLLM_MTP_DRAFT_HEAD_FP8=1\n")
            m = json.loads(read(os.path.join(out, "manifest.json")))
            self.assertIsNone(m["overlays"][0]["reference"])


if __name__ == "__main__":
    unittest.main()
