#!/usr/bin/env python3
"""Write trained MTP BF16 tensors into a copy of Mia's MTP shard.

Only tensors present in the trained state dict change; the NVFP4 routed experts
and every non-MTP tensor are copied through byte for byte. The output shard is
bind-mounted over model-00034-of-00034.safetensors for an A/B, so reverting is
removing one mount.

    python3 export_mtp.py --trained mtp_trained.pt --out /models/usman/distill/shard34_tuned.safetensors
"""
import argparse, glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SNAP = glob.glob("/models/usman/hf/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/")[0]


def trained_to_ckpt_name(n):
    n = n.replace("layer.", "layers.0.", 1) if n.startswith("layer.") else n
    n = n.replace("mixer.", "hyper_connection_mixer.", 1) if n.startswith("mixer.") else n
    return "mtp." + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trained = {trained_to_ckpt_name(k): v for k, v in torch.load(a.trained).items()}
    src = SNAP + "model-00034-of-00034.safetensors"
    out, changed = {}, 0
    with safe_open(src, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            if k in trained:
                new = trained[k].to(t.dtype)
                if new.shape != t.shape:
                    raise SystemExit(f"shape mismatch {k}: {tuple(new.shape)} vs {tuple(t.shape)}")
                changed += int(not torch.equal(new, t))
                t = new
            out[k] = t.contiguous()
    unknown = sorted(set(trained) - set(out))
    if unknown:
        raise SystemExit(f"trained tensors with no checkpoint slot: {unknown[:8]}")
    save_file(out, a.out, metadata=meta)
    print(f"wrote {a.out}: {changed} tensors changed of {len(trained)} trained")


if __name__ == "__main__":
    main()
