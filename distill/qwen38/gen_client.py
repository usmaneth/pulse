#!/usr/bin/env python3
"""Generate self-distillation data: the target model answers every prompt.

Runs next to a local vLLM server. Writes one JSONL row per prompt with the
exact prompt and completion token ids (vLLM `return_token_ids`), so training
sees the target's own tokens and not a re-tokenization of its text.
The script resumes: ids already present in the output file are skipped.

    python3 gen_client.py --prompts prompts.jsonl --out gen.jsonl --concurrency 8
"""
import argparse, json, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

lock = threading.Lock()
stats = {"done": 0, "tokens": 0, "errors": 0, "t0": time.time()}


def request(url, model, row, template, max_tokens):
    if row["shape"] == "codex":
        messages = [{"role": "system", "content": template["system"]},
                    {"role": "user", "content": row["prompt"]}]
        tools = template["tools"]
    else:
        messages = [{"role": "user", "content": row["prompt"]}]
        tools = None
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": bool(row["thinking"])}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return {
        "id": row["id"], "src": row["src"], "shape": row["shape"], "thinking": row["thinking"],
        "prompt_token_ids": d.get("prompt_token_ids") or ch.get("prompt_token_ids"),
        "token_ids": ch.get("token_ids"),
        "finish_reason": ch.get("finish_reason"),
        "content": ch["message"].get("content"),
        "reasoning": ch["message"].get("reasoning") or ch["message"].get("reasoning_content"),
        "tool_calls": ch["message"].get("tool_calls"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=os.path.join(os.path.dirname(__file__), "codex_template.json"))
    ap.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    template = json.load(open(a.template))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    rows = [json.loads(l) for l in open(a.prompts)]
    rows = [r for r in rows if r["id"] not in done]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(done)} done, {len(rows)} to go", flush=True)
    out = open(a.out, "a")

    def work(row):
        try:
            res = request(a.url, a.model, row, template, a.max_tokens)
        except Exception as e:
            with lock:
                stats["errors"] += 1
            print(f"error id={row['id']}: {type(e).__name__} {str(e)[:120]}", flush=True)
            return
        with lock:
            out.write(json.dumps(res) + "\n"); out.flush()
            stats["done"] += 1
            stats["tokens"] += len(res["token_ids"] or [])
            if stats["done"] % 50 == 0:
                dt = time.time() - stats["t0"]
                print(f"{stats['done']} rows  {stats['tokens']:,} tokens  "
                      f"{stats['tokens'] / dt:.0f} tok/s  errors={stats['errors']}", flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(work, rows))
    print("finished", stats, flush=True)


if __name__ == "__main__":
    main()
