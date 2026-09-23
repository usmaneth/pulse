#!/usr/bin/env python3
"""Precise draft-acceptance bench on held-out prompts.

Tokens per engine step (1 + accepted / drafts) does not depend on the GPU
bandwidth state, so it isolates draft quality. The bench runs N held-out
prompts from the distillation set (rows after the training snapshot) at
temperature 0 and reports tokens/step and per-position acceptance, overall and
per source.

    python3 acceptbench.py --tag mtp-r1 --n 40
"""
import argparse, json, os, re, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(r'^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$')


def metrics(port):
    out = {}
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            m = LINE.match(line)
            if not m or "spec_decode" not in m.group(1):
                continue
            key = m.group(1)
            pos = re.search(r'position="(\d+)"', m.group(2) or "")
            if pos:
                key += f"[{pos.group(1)}]"
            out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def run(port, model, row, template):
    if row["shape"] == "codex":
        msgs = [{"role": "system", "content": template["system"]}, {"role": "user", "content": row["prompt"]}]
        tools = template["tools"]
    else:
        msgs = [{"role": "user", "content": row["prompt"]}]
        tools = None
    body = {"model": model, "messages": msgs, "max_tokens": 400, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": bool(row.get("thinking"))}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["usage"]["completion_tokens"]


def summarize(d):
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    pos = [round(d[k] / drafts, 3) for k in sorted(d) if "accepted_tokens_per_pos" in k and "[" in k and drafts]
    return {"tok_per_step": round(1 + acc / drafts, 3) if drafts else None, "per_pos": pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--heldout", default=f"{HERE}/heldout_prompts.jsonl")
    a = ap.parse_args()
    template = json.load(open("/models/usman/distill/codex_template.json"))
    rows = [json.loads(l) for l in open(a.heldout)][: a.n]
    by_src, total0 = {}, metrics(a.port)
    toks = 0
    for row in rows:
        m0 = metrics(a.port)
        toks += run(a.port, a.model, row, template)
        m1 = metrics(a.port)
        agg = by_src.setdefault(row["src"], {})
        for k in m1:
            agg[k] = agg.get(k, 0) + m1[k] - m0.get(k, 0)
    total1 = metrics(a.port)
    overall = summarize({k: total1[k] - total0.get(k, 0) for k in total1})
    res = {"tag": a.tag, "n": len(rows), "tokens": toks, **overall,
           "by_src": {s: summarize(d) for s, d in by_src.items()}}
    print(json.dumps(res))
    with open(f"{HERE}/accept.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
