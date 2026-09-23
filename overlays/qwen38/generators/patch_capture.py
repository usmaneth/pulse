#!/usr/bin/env python3
"""Capture the target hidden states that feed the MTP drafter.

Generates ar_speculator_capture.py: the image's autoregressive speculator plus a
hook in propose(). When VLLM_MTP_CAPTURE_DIR is set, every prefill chunk of a
request whose id contains "cap-<row>" writes one file:

    <dir>/<row>_<start>.pt = {"start": start, "hidden": bf16 [n, hc_count*hidden]}

`start` is the absolute token position of the chunk's first row, so the rows
line up with the request's token ids. This is the pre-final-mixer multi-stream
state the MTP head consumes at draft step 0. Only for the offline capture
server; a serving profile never mounts this file.

    python3 patch_capture.py SRC OUT

SRC is an extracted vLLM package. The anchor must occur exactly once. A source
that already holds the hook is refused, so the generator never patches its own
output a second time.
"""
import ast, os, sys


def add_import_os(s):
    """Add "import os" at the start, but after each "from __future__" import.

    A "from __future__" import must come before all other statements.
    """
    if "\nimport os\n" in s:
        return s
    end = 0
    for node in ast.parse(s).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            end = node.end_lineno
    pos = 0
    for _ in range(end):
        pos = s.index("\n", pos) + 1
    return s[:pos] + "import os\n" + s[pos:]


if len(sys.argv) != 3:
    sys.exit("usage: patch_capture.py SRC OUT")
SRC, OUT = sys.argv[1], sys.argv[2]
REL = "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
MARK = "_capture_hidden"

HELPER = '''

_CAPTURE_DIR = os.environ.get("VLLM_MTP_CAPTURE_DIR", "")


def _capture_hidden(input_batch, hidden_states) -> None:
    """Write prefill hidden-state rows for requests tagged cap-<row>."""
    qsl = input_batch.query_start_loc_np
    for i, req_id in enumerate(input_batch.req_ids[: input_batch.num_reqs]):
        if "cap-" not in req_id or not input_batch.is_prefilling_np[i]:
            continue
        row = req_id.split("cap-", 1)[1].split("-", 1)[0]
        start = int(input_batch.num_computed_tokens_np[i])
        rows = hidden_states[int(qsl[i]) : int(qsl[i + 1])]
        torch.save(
            {"start": start, "hidden": rows.detach().to("cpu", torch.bfloat16)},
            os.path.join(_CAPTURE_DIR, f"{row}_{start}.pt"),
        )
'''

ANCHOR = "        self.hidden_states[:num_tokens_padded].copy_(hidden_states)\n"
HOOK = ANCHOR + (
    "        if _CAPTURE_DIR and not dummy_run and not is_profile:\n"
    "            _capture_hidden(input_batch, hidden_states)\n"
)

s = open(os.path.join(SRC, REL), encoding="utf-8").read()
if MARK in s:
    sys.exit(f"{REL}: source is already patched (found {MARK!r})")
n = s.count(ANCHOR)
if n != 1:
    sys.exit(f"{REL}: anchor count {n} in propose(), expected 1")
s = s.replace(ANCHOR, HOOK, 1)
first_def = s.index("\nclass ")
s = s[:first_def] + HELPER + s[first_def:]
s = add_import_os(s)
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "ar_speculator_capture.py"), "w", encoding="utf-8").write(s)
print("wrote ar_speculator_capture.py")
