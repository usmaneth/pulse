#!/usr/bin/env python3
"""How much of real agent output is copied from its own context.

For each assistant output in the Codex session logs (message text, tool-call
arguments, patch input), the context is everything earlier in that session:
user turns, tool outputs and earlier assistant outputs. A token counts as
"copyable" when it lies inside a span of >= N tokens that also appears in the
context. The script also simulates a simple suffix drafter: at each position it
looks up the longest recent suffix (up to 24 tokens) in the context and, if
found with length >= 4, drafts up to 16 following tokens; it reports the mean
tokens gained per verify step on those positions.

    .venv/bin/python copyrate.py --n-sessions 60
"""
import argparse, glob, json, os
from tokenizers import Tokenizer

TOK = Tokenizer.from_file(glob.glob(
    "/models/usman/hf/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/*/tokenizer.json")[0])


def events(path):
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        p = d.get("payload") or {}
        t = p.get("type")
        if t == "message":
            txt = "".join(c.get("text", "") for c in p.get("content", []) if isinstance(c, dict))
            yield ("out" if p.get("role") == "assistant" else "ctx"), txt
        elif t in ("function_call", "custom_tool_call"):
            yield "out", p.get("input") or p.get("arguments") or ""
        elif t in ("function_call_output", "custom_tool_call_output"):
            o = p.get("output")
            if isinstance(o, list):
                o = "".join(x.get("text", "") for x in o if isinstance(x, dict))
            yield "ctx", o if isinstance(o, str) else json.dumps(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=60)
    ap.add_argument("--span", type=int, default=8)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.expanduser("~/.codex/sessions/**/*.jsonl"), recursive=True),
                   key=os.path.getmtime)[-a.n_sessions:]
    N = a.span
    tot_out = copy_tok = 0
    sim_steps = sim_tokens = 0
    per_kind = {}
    for f in files:
        ctx_ids = []
        grams = {}
        def add(ids):
            base = len(ctx_ids)
            ctx_ids.extend(ids)
            for i in range(max(0, base - N + 1), len(ctx_ids) - N + 1):
                grams.setdefault(tuple(ctx_ids[i:i + N]), i)
        for kind, text in events(f):
            if not text:
                continue
            ids = TOK.encode(text, add_special_tokens=False).ids
            if kind == "out" and ctx_ids:
                covered = [False] * len(ids)
                for i in range(len(ids) - N + 1):
                    if tuple(ids[i:i + N]) in grams:
                        for j in range(i, i + N):
                            covered[j] = True
                c = sum(covered)
                tot_out += len(ids); copy_tok += c
                k = "json/tool" if text.lstrip().startswith(("{", "*** Begin Patch", "[")) else "text"
                pk = per_kind.setdefault(k, [0, 0]); pk[0] += len(ids); pk[1] += c
                # suffix-drafter simulation over this output
                i = 0
                while i < len(ids):
                    hit = None
                    for L in (8, 6, 4):
                        if i >= L:
                            key = tuple(ids[i - L:i])
                            # find a context occurrence of the last L tokens (use N-gram table when L==N)
                            if L == N and key in grams:
                                hit = grams[key] + L
                                break
                    if hit is None:
                        sim_steps += 1; sim_tokens += 1; i += 1
                        continue
                    acc = 0
                    while acc < 16 and i + acc < len(ids) and hit + acc < len(ctx_ids) and ctx_ids[hit + acc] == ids[i + acc]:
                        acc += 1
                    sim_steps += 1; sim_tokens += acc + 1; i += acc + 1
            add(ids)
    print(json.dumps({
        "sessions": len(files), "output_tokens": tot_out,
        f"copyable_frac_span{N}": round(copy_tok / max(tot_out, 1), 4),
        "by_kind": {k: {"tokens": v[0], "copyable_frac": round(v[1] / max(v[0], 1), 4)} for k, v in per_kind.items()},
        "suffix_only_tokens_per_step": round(sim_tokens / max(sim_steps, 1), 3),
    }, indent=1))


if __name__ == "__main__":
    main()
