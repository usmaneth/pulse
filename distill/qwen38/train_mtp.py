#!/usr/bin/env python3
"""Self-distill the Qwen3.8-Flash-Next MTP head on captured target hidden states.

The MTP block is one Qwen4Exp attention decoder layer (QSA + 512-expert MoE +
hyper-connections) behind an input fusion, then the final hyper-connection
mixer and the shared LM head:

    e  = fc_embedding(pre_fc_norm_embedding(embed(x[t+k])))        [T, H]
    h  = fc_hidden(pre_fc_norm_hidden(multi[t]).view(T, hc, H))     [T, hc, H]
    m  = layer((e[:, None] + h).flatten(-2))                        [T, hc*H]
    y  = lm_head(mixer(m))                                          predicts x[t+k+1]

Draft step 1 takes `multi` = the target hidden state; step k>1 takes the m of
step k-1 (FastMTP recursion). The routed experts stay frozen at the NVFP4
values vLLM serves (dequantized here); every BF16 tensor of the block trains.
The export overwrites only those BF16 tensors in a copy of the MTP shard.

Modes:
    --eval-only   offline top-1 accuracy per draft step, no training. With the
                  stock head this must reproduce the live per-position
                  acceptance (parity check for this port).
"""
import argparse, glob, json, math, os, random, struct

import torch
import torch.nn as nn
import torch.nn.functional as F

SNAP = glob.glob("/models/usman/hf/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/")
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def read_tensors(path, names):
    """Read selected tensors from a safetensors file without loading the rest."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt") as f:
        for n in names:
            out[n] = f.get_tensor(n)
    return out


def dequant_nvfp4(w_u8, scale_f8, scale2):
    """ModelOpt NVFP4: 2 e2m1 values per byte (low nibble first), fp8 scale per 16."""
    lo = (w_u8 & 0x0F).long()
    hi = (w_u8 >> 4).long()
    vals = torch.stack([FP4.to(w_u8.device)[lo], FP4.to(w_u8.device)[hi]], dim=-1).flatten(-2)
    scale = scale_f8.float().repeat_interleave(16, dim=-1) * scale2.float()
    return (vals * scale).to(torch.bfloat16)


def build(cfg_path, shard, device):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as M
    raw = json.load(open(cfg_path))
    tc = Qwen4ExpTextConfig(**raw["text_config"])
    tc._attn_implementation = "sdpa"
    attn_idx = next(i for i, t in enumerate(tc.layer_types) if t != "linear_attention"
                    and (i + 1) not in (tc.ple_layer_ids or []))
    H, hc = tc.hidden_size, tc.hc_count

    class MTPHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_fc_norm_embedding = M.Qwen4ExpTextRMSNorm(H, eps=tc.rms_norm_eps)
            self.pre_fc_norm_hidden = M.Qwen4ExpTextRMSNorm(H * hc, eps=tc.rms_norm_eps)
            self.fc_embedding = nn.Linear(H, H, bias=False)
            self.fc_hidden = nn.Linear(H, H, bias=False)
            self.layer = M.Qwen4ExpTextDecoderLayer(tc, attn_idx)
            self.mixer = M.Qwen4ExpTextGatedResidual(tc, use_combine=False)
            self.rotary = M.Qwen4ExpTextRotaryEmbedding(tc)

        def forward(self, multi, emb, pos):
            T = multi.shape[0]
            e = self.fc_embedding(self.pre_fc_norm_embedding(emb))
            h = self.fc_hidden(self.pre_fc_norm_hidden(multi).view(T, hc, H))
            x = (e.unsqueeze(-2) + h).flatten(-2).unsqueeze(0)
            pe = self.rotary(x, pos.unsqueeze(0))
            # HF QSA attention needs a real 4D mask (it ANDs its sparse block
            # selection into it); None would not be causal. Bool mask for SDPA.
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None]
            m = self.layer(x, position_embeddings=pe, attention_mask=causal).squeeze(0)
            return m, self.mixer(m)

    head = MTPHead()
    # Load BF16 tensors of the block; dequantize the NVFP4 routed experts.
    from safetensors import safe_open
    sd = {}
    with safe_open(shard, framework="pt") as f:
        keys = [k for k in f.keys() if k.startswith("mtp.")]
        for k in keys:
            if ".experts." in k:
                continue
            sd[k] = f.get_tensor(k)
        experts = {}
        for k in keys:
            if ".experts." in k and k.endswith(".weight"):
                base = k[: -len(".weight")]
                experts[base] = dequant_nvfp4(f.get_tensor(k), f.get_tensor(base + ".weight_scale"),
                                              f.get_tensor(base + ".weight_scale_2"))
    remap = {}
    for k, v in sd.items():
        n = k[len("mtp."):]
        n = n.replace("layers.0.", "layer.", 1).replace("hyper_connection_mixer.", "mixer.", 1)
        remap[n] = v
    missing, unexpected = head.load_state_dict(remap, strict=False)
    head = head.to(device, torch.bfloat16)
    # HF keeps routed experts as stacked tensors; fill them from the dequantized weights.
    exp = head.layer.mlp.experts
    ne = tc.num_experts
    with torch.no_grad():
        gate = torch.stack([experts[f"mtp.layers.0.mlp.experts.{i}.gate_proj"] for i in range(ne)])
        up = torch.stack([experts[f"mtp.layers.0.mlp.experts.{i}.up_proj"] for i in range(ne)])
        down = torch.stack([experts[f"mtp.layers.0.mlp.experts.{i}.down_proj"] for i in range(ne)])
        filled = False
        for name, p in exp.named_parameters():
            if "gate_up" in name:
                p.copy_(torch.cat([gate, up], dim=1).to(p.device, p.dtype).view_as(p)); filled = True
            elif "down" in name:
                p.copy_(down.to(p.device, p.dtype).view_as(p))
        if not filled:
            raise RuntimeError(f"unknown expert layout: {[n for n, _ in exp.named_parameters()]}")
    for p in exp.parameters():
        p.requires_grad_(False)
    missing = [m for m in missing if ".experts." not in m and "rotary" not in m]
    return head, tc, missing, unexpected


def load_examples(cap_dir, max_rows):
    rows = []
    for p in sorted(glob.glob(os.path.join(cap_dir, "*.ids.pt"))):
        rid = os.path.basename(p).split(".")[0]
        parts = sorted(glob.glob(os.path.join(cap_dir, f"{rid}_*.pt")),
                       key=lambda q: int(os.path.basename(q).split("_")[1].split(".")[0]))
        if parts:
            rows.append((p, parts))
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def example(ids_path, parts, max_len, ctx=1536):
    """Window = the last `ctx` prompt tokens + the completion, capped at max_len.

    Codex-shaped rows carry ~9.6k-token prompts; the completion is where drafting
    happens, so the window keeps all of it and only enough prompt for context.
    """
    meta = torch.load(ids_path)
    hid = torch.cat([torch.load(q)["hidden"] for q in parts], dim=0)
    n = min(len(meta["ids"]), hid.shape[0])
    start = max(0, meta["prompt_len"] - ctx)
    end = min(n, start + max_len)
    return (meta["ids"][start:end].long(), hid[start:end],
            meta["prompt_len"] - start)


def run_steps(head, embed, lm_head, ids, hid, steps, device, prompt_len, train):
    """Returns per-step (loss, correct, count) over completion positions."""
    T = ids.shape[0]
    pos = torch.arange(T, device=device)
    multi = hid.to(device, torch.bfloat16)
    stats = []
    total = 0.0
    for k in range(1, steps + 1):
        # position t sees x[t+k] and predicts x[t+k+1]
        valid = T - k - 1
        if valid <= 0:
            break
        emb = embed(ids[k:k + valid].to(device))
        m, y = head(multi[:valid], emb, pos[:valid])
        logits = lm_head(y).float()
        target = ids[k + 1:k + 1 + valid].to(device)
        mask = torch.zeros(valid, dtype=torch.bool, device=device)
        mask[max(prompt_len - 1, 0):] = True  # score the completion, where drafting happens
        loss = F.cross_entropy(logits[mask], target[mask]) if mask.any() else logits.sum() * 0
        correct = (logits.argmax(-1)[mask] == target[mask]).sum().item()
        stats.append((loss.item(), correct, int(mask.sum().item())))
        total = total + loss * (0.8 ** (k - 1))  # FastMTP-style decay over draft steps
        multi = m  # recursion: step k+1 consumes this step's multi-stream state
    if train:
        total.backward()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default="/models/usman/distill/mtp_trained.pt")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    device = "cuda"
    snap = SNAP[0]
    head, tc, missing, unexpected = build(snap + "config.json", snap + "model-00034-of-00034.safetensors", device)
    print("missing(non-expert):", missing[:10], "unexpected:", unexpected[:10], flush=True)
    idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
    ek = next(k for k in idx if k.endswith("embed_tokens.weight") and "visual" not in k)
    lk = next((k for k in idx if k.endswith("lm_head.weight")), ek)
    t = read_tensors(snap + idx[ek], [ek]); embed_w = t[ek].to(device, torch.bfloat16)
    t = read_tensors(snap + idx[lk], [lk]); lm_w = t[lk].to(device, torch.bfloat16)
    embed = lambda x: F.embedding(x, embed_w)
    lm_head = lambda y: F.linear(y, lm_w)

    rows = load_examples(a.cap, a.max_rows)
    random.seed(0); random.shuffle(rows)
    n_eval = max(1, int(len(rows) * a.eval_frac))
    evals, trains = rows[:n_eval], rows[n_eval:]
    print(f"{len(trains)} train / {len(evals)} eval rows", flush=True)

    def evaluate(tag):
        head.eval(); agg = {}
        with torch.no_grad():
            for ids_p, parts in evals:
                ids, hid, pl = example(ids_p, parts, a.max_len)
                for k, (l, c, n) in enumerate(run_steps(head, embed, lm_head, ids, hid, a.steps, device, pl, False), 1):
                    s = agg.setdefault(k, [0.0, 0, 0]); s[0] += l * n; s[1] += c; s[2] += n
        res = {k: (round(v[0] / max(v[2], 1), 4), round(v[1] / max(v[2], 1), 4)) for k, v in agg.items()}
        # conditional acceptance proxy: top-1 accuracy per step (loss, acc)
        print(f"[{tag}] per-step (loss, top1):", res, flush=True)
        return res

    evaluate("stock")
    if a.eval_only:
        return
    params = [p for p in head.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    step = 0
    for ep in range(a.epochs):
        random.shuffle(trains)
        head.train()
        for ids_p, parts in trains:
            ids, hid, pl = example(ids_p, parts, a.max_len)
            opt.zero_grad(set_to_none=True)
            run_steps(head, embed, lm_head, ids, hid, a.steps, device, pl, True)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); step += 1
            if step % 10 == 0:
                import time as _t
                now = _t.time(); print(f"step {step}  {(now - getattr(main, '_t0', now)) / 10:.2f} s/step", flush=True); main._t0 = now
            if a.max_steps and step >= a.max_steps:
                break
            if step % 200 == 0:
                evaluate(f"ep{ep} step{step}")
                torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()
                            if ".experts." not in k and "rotary" not in k}, a.out)
    evaluate("final")
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()
                if ".experts." not in k and "rotary" not in k}, a.out)


if __name__ == "__main__":
    main()
