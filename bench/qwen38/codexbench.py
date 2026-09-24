#!/usr/bin/env python3
"""Codex-shaped decode benchmark for a local vLLM server.

The script sends coding prompts with a large fixed prefix (the size of a Codex
system prompt plus tool list), streams each reply, and reads the vLLM
speculative-decode counters from /metrics before and after each run.

It reports, per prompt set:
  decode tok/s   = completion tokens / (last byte - first token), per stream
  TTFT           = request start -> first streamed token
  tokens/step    = 1 + accepted / drafts
  per-position   = accepted at draft position i / drafts

    python3 codexbench.py --tag K3 --reps 3
    python3 codexbench.py --tag K3 --streams 1 2 --out results.jsonl
"""
import argparse, json, re, threading, time, urllib.request

PREFIX = (
    "You are Codex, a coding agent. You run in a terminal with shell and "
    "apply_patch tools. Keep answers short. Prefer small diffs. "
) + "\n".join(
    f"Tool {i}: name=tool_{i}, input=JSON object with fields path (string), "
    f"start_line (integer), end_line (integer), reason (string)."
    for i in range(160)
)

PROMPTS = {
    "code_write": "Write a Python function `merge_intervals(intervals)` that merges "
                  "overlapping [start, end] pairs. Include type hints, a docstring and "
                  "five pytest tests. Output only the code.",
    "code_edit": "Here is a TypeScript function:\n\n```ts\nexport function sum(xs: number[]) {\n"
                 "  let t = 0\n  for (let i = 0; i <= xs.length; i++) t += xs[i]\n  return t\n}\n```\n\n"
                 "Find the bug, then output a unified diff that fixes it and adds an "
                 "empty-array guard. Output only the diff.",
    "tool_json": "Return a JSON array of 12 tool calls that read the files src/a.ts through "
                 "src/l.ts in 40-line windows. Use the tool_3 schema. Output only JSON.",
    "long_ctx": "__LONG__",
    "explain": "Explain in about 250 words how a write-ahead log gives crash safety "
               "in a database. Plain prose, no lists.",
}

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


LONG_CTX = open(__file__.rsplit("/", 1)[0] + "/long_context.txt").read()
LONG_XL = open(__file__.rsplit("/", 1)[0] + "/long_context_xl.txt").read()
LONG_Q = ("\n\nAbove is part of a TypeScript CLI codebase. Write a new function "
          "`summarizeFiles(paths: string[])` that fits its style, reusing its existing helpers. "
          "Output only the code.")


def one(port, model, prompt, max_tokens, res):
    if prompt == "__LONG__":
        prompt = LONG_CTX + LONG_Q
    elif prompt == "__LONG_XL__":
        prompt = LONG_XL + LONG_Q
    body = {
        "model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
        "messages": [{"role": "system", "content": PREFIX},
                     {"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    t0 = time.time(); first = None; toks = 0; text = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                toks = d["usage"]["completion_tokens"]
            ch = d.get("choices") or []
            piece = (ch[0].get("delta") or {}).get("content") if ch else None
            if piece:
                text.append(piece)
                if first is None:
                    first = time.time()
    end = time.time()
    first = first or end
    res.append({"ttft": first - t0, "tok": toks, "text": "".join(text),
                "tps": (toks - 1) / (end - first) if end > first else 0.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--streams", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--out", default="/mnt/models/qwen38-tune/results.jsonl")
    ap.add_argument("--only", nargs="*", default=None, help="prompt names to run")
    ap.add_argument("--xl", action="store_true", help="add the ~110k-token context case")
    a = ap.parse_args()
    one(a.port, a.model, "Say ok.", 8, [])  # warm the prefix cache
    prompts = dict(PROMPTS)
    if a.xl:
        prompts["long_xl"] = "__LONG_XL__"
    if a.only:
        prompts = {k: v for k, v in prompts.items() if k in a.only}
    for s in a.streams:
        for name, prompt in prompts.items():
            runs = []
            m0 = metrics(a.port)
            for _ in range(a.reps):
                res = []
                th = [threading.Thread(target=one, args=(a.port, a.model, prompt, a.max_tokens, res))
                      for _ in range(s)]
                [t.start() for t in th]; [t.join() for t in th]
                runs.extend(res)
            m1 = metrics(a.port)
            d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
            drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
            acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            pos = [round(d[k] / drafts, 3) for k in sorted(d)
                   if "accepted_tokens_per_pos" in k and "[" in k and drafts]
            row = {
                "tag": a.tag, "streams": s, "prompt": name,
                "tps": round(sum(r["tps"] for r in runs) / len(runs), 1),
                "agg_tps": round(sum(r["tps"] for r in runs) / a.reps, 1),
                "ttft_ms": round(1000 * sum(r["ttft"] for r in runs) / len(runs)),
                "tok_per_step": round(1 + acc / drafts, 2) if drafts else None,
                "per_pos": pos,
            }
            if s == 1:
                with open(a.out.replace(".jsonl", f".outputs.{a.tag}.jsonl"), "a") as f:
                    f.write(json.dumps({"prompt": name, "text": runs[0]["text"]}) + "\n")
            print(json.dumps(row), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
