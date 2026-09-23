#!/usr/bin/env python3
"""FP8 rowwise copy of the target lm_head (248320 x 2560) for decode.

STATUS: REJECTED. No speed gain after probe normalization, and 92.9% top-1
agreement against the BF16 head. The generator stays here as a record and for
a repeat test on a future image. Do not mount its output on a serving profile.

The target reads its whole BF16 lm_head (1.27 GB) every engine step to score the
verify tokens. This generates nvidia_model_fp8head.py: the image's
models/qwen3_8_flash_next/nvidia/model.py with compute_logits routed through an
FP8 (e4m3, per-row scale) torch._scaled_mm when VLLM_LM_HEAD_FP8=1. The wrapper
replaces only lm_head.quant_method.apply, so the logits processor still applies
scaling, soft cap and vocab trimming as before. The FP8 copy is built on the
first eager call (profile run), never inside CUDA graph capture.

Unlike the draft head, this changes the target's own logits: gate it with
driftgate.py (top-1 agreement, KL) and the HumanEval gate before shipping.

    python3 patch_lm_head_fp8.py SRC OUT

SRC is an extracted vLLM package. The anchor must occur exactly once. A source
that already holds the wrapper is refused.
"""
import os, sys

if len(sys.argv) != 3:
    sys.exit("usage: patch_lm_head_fp8.py SRC OUT")
SRC, OUT = sys.argv[1], sys.argv[2]
REL = "models/qwen3_8_flash_next/nvidia/model.py"
MARK = "_Fp8LMHeadApply"

HELPER = '''

class _Fp8LMHeadApply:
    """quant_method stand-in: logits = x @ W^T with W in FP8 e4m3, per-row scale."""

    def __init__(self, inner):
        self.inner = inner
        self.w8 = None

    def _build(self, weight):
        rows = weight.shape[0]
        pad = (-rows) % 16
        wf = weight.float()
        if pad:
            wf = torch.cat([wf, wf.new_zeros(pad, wf.shape[1])], dim=0)
        scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
        self.w8 = (wf / scale).to(torch.float8_e4m3fn)
        self.w8_scale = scale.t().contiguous()
        self.rows = rows

    def apply(self, layer, x, bias=None):
        if self.w8 is None:
            if torch.cuda.is_current_stream_capturing():
                return self.inner.apply(layer, x, bias)
            self._build(layer.weight)
        xf = x.float()
        xs = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
        out = torch._scaled_mm((xf / xs).to(torch.float8_e4m3fn), self.w8.t(),
                               scale_a=xs, scale_b=self.w8_scale, out_dtype=x.dtype)
        out = out[..., : self.rows]
        return out if bias is None else out + bias

    def __getattr__(self, name):
        return getattr(self.inner, name)
'''

OLD = """    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
"""
NEW = """    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if os.environ.get("VLLM_LM_HEAD_FP8", "0") == "1" and not isinstance(
            self.lm_head.quant_method, _Fp8LMHeadApply
        ):
            self.lm_head.quant_method = _Fp8LMHeadApply(self.lm_head.quant_method)
        return self.logits_processor(self.lm_head, hidden_states)
"""

s = open(os.path.join(SRC, REL), encoding="utf-8").read()
if MARK in s:
    sys.exit(f"{REL}: source is already patched (found {MARK!r})")
n = s.count(OLD)
if n != 1:
    sys.exit(f"{REL}: compute_logits anchor count {n}, expected 1")
s = s.replace(OLD, NEW, 1)
first_class = s.index("\nclass ")
s = s[:first_class] + HELPER + s[first_class:]
if "\nimport os\n" not in s:
    s = "import os\n" + s
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "nvidia_model_fp8head.py"), "w", encoding="utf-8").write(s)
print("wrote nvidia_model_fp8head.py")
