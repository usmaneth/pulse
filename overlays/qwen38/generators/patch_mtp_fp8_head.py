#!/usr/bin/env python3
"""Add an FP8 copy of the reduced MTP draft head to files/mtp_patched.py.

Runs after patch_mtp_draft_vocab.py. Inert unless VLLM_MTP_DRAFT_HEAD_FP8=1 in
the container environment. Idempotent: a second run changes nothing.

The draft head is a batch-1 GEMV over the reduced vocabulary each draft step.
On GB10 the rowwise FP8 torch._scaled_mm measured 0.83 ms against 1.96 ms for
the BF16 linear at 47184 x 2560. Only the argmax is used and the target verifies
every draft token, so output stays exact.

    python3 patch_mtp_fp8_head.py [TARGET]

TARGET is the mtp_patched.py that patch_mtp_draft_vocab.py wrote. The file is
patched in place. Without TARGET the script patches mtp_patched.py next to
itself, which is the recipe layout, so this file can replace the recipe copy
of the generator with no other change.

This generator gives the same output bytes as the recipe copy
files/patch_mtp_fp8_head.py. It differs from that copy in three ways: the
TARGET argument, short anchors, and a check that each anchor occurs exactly
once. The first two anchors are short fragments of the code that the MiaAI Lab
script patch_mtp_draft_vocab.py (AGPL-3.0) writes. This repository does not
hold that script. The third anchor is from the vLLM file mtp.py.
"""
import os, sys

PATH = (sys.argv[1] if len(sys.argv) > 1 else
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtp_patched.py"))
MARK = "_attach_fp8_draft_head"

HELPER = '''

def _attach_fp8_draft_head(model: nn.Module) -> None:
    """FP8 (e4m3, per-row scale) copy of the reduced draft head.

    Rows are padded to a multiple of 16 for torch._scaled_mm; get_top_tokens
    drops the pad columns before the argmax.
    """
    w = model._draft_lm_head_weight
    rows = w.shape[0]
    pad = (-rows) % 16
    wf = w.float()
    if pad:
        wf = torch.cat([wf, wf.new_zeros(pad, wf.shape[1])], dim=0)
    scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    model.register_buffer("_draft_head_fp8", (wf / scale).to(torch.float8_e4m3fn), persistent=False)
    model.register_buffer("_draft_head_fp8_scale", scale.t().contiguous(), persistent=False)
    model._draft_head_rows = rows
    logger.info("MTP draft head: FP8 rowwise copy engaged (%d rows, %d pad).", rows, pad)
'''

# The attach call goes at the start of the line that computes the full head
# size, after the reduced head exists.
ATTACH_AT = "\n    full_gib = "
ATTACH = '''    if os.environ.get("VLLM_MTP_DRAFT_HEAD_FP8", "0") == "1":
        _attach_fp8_draft_head(model)
'''

# The FP8 branch goes at the start of the BF16 linear in get_top_tokens.
TOP_AT = "\n        logits = torch.nn.functional.linear("
TOP = '''        w8 = getattr(self, "_draft_head_fp8", None)
        if w8 is not None:
            x = hidden_states.float()
            xs = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
            logits = torch._scaled_mm(
                (x / xs).to(torch.float8_e4m3fn), w8.t(),
                scale_a=xs, scale_b=self._draft_head_fp8_scale,
                out_dtype=torch.bfloat16,
            )[:, : self._draft_head_rows]
            return self._draft_id_to_target_id[logits.argmax(dim=-1)].to(torch.long)
'''

# The helper goes before this vLLM function.
HELPER_AT = "\n\ndef _remap_ignored_layers("

with open(PATH, encoding="utf-8") as f:
    s = f.read()
if MARK in s:
    print("patch_mtp_fp8_head: already applied")
    sys.exit(0)
for anchor in (ATTACH_AT, TOP_AT, HELPER_AT):
    n = s.count(anchor)
    if n != 1:
        sys.exit(f"patch_mtp_fp8_head: anchor count {n}, expected 1: {anchor!r}")
s = s.replace(ATTACH_AT, "\n" + ATTACH + ATTACH_AT[1:], 1)
s = s.replace(TOP_AT, "\n" + TOP + TOP_AT[1:], 1)
s = s.replace(HELPER_AT, HELPER + HELPER_AT, 1)
with open(PATH, "w", encoding="utf-8") as f:
    f.write(s)
print("patch_mtp_fp8_head: applied")
