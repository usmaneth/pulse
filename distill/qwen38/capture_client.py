#!/usr/bin/env python3
"""Replay generated rows through a capture server to dump MTP input hidden states.

Each row's exact token ids (prompt + completion) go to /v1/completions as a
prefill-only request (max_tokens=1) tagged X-Request-Id: cap-<row>. The capture
overlay (overlays/qwen38, overlay "capture") writes <row>_<start>.pt per prefill
chunk. The script then writes <row>.ids.pt with the token ids, so a training
example is (ids, concatenated hidden rows).

    python3 capture_client.py --gen gen.jsonl --dir /mnt/models/distill/cap --max-rows 2000
"""
import argparse, json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch


def send(url, model, row, ids):
    # A unique cache_salt per row: no request reuses another's prefix cache, so
    # the hook sees every position recomputed (prefix caching cannot be turned
    # off on this model; its mamba block size requires it).
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
            "cache_salt": f"capture-{row}"}
    req = urllib.request.Request(url, json.dumps(body).encode(), {
        "Content-Type": "application/json", "X-Request-Id": f"cap-{row}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8888/v1/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    rows = []
    for line in open(a.gen):
        d = json.loads(line)
        if not d.get("token_ids") or not d.get("prompt_token_ids"):
            continue
        if os.path.exists(os.path.join(a.dir, f"{d['id']}.ids.pt")):
            continue
        rows.append(d)
        if a.max_rows and len(rows) >= a.max_rows:
            break
    print(f"{len(rows)} rows to capture", flush=True)

    def work(d):
        ids = d["prompt_token_ids"] + d["token_ids"]
        try:
            send(a.url, a.model, d["id"], ids)
        except Exception as e:
            print(f"error row={d['id']}: {type(e).__name__} {str(e)[:100]}", flush=True)
            return
        torch.save({"ids": torch.tensor(ids, dtype=torch.int32),
                    "prompt_len": len(d["prompt_token_ids"]),
                    "src": d["src"], "shape": d["shape"]},
                   os.path.join(a.dir, f"{d['id']}.ids.pt"))

    with ThreadPoolExecutor(a.concurrency) as ex:
        for i, _ in enumerate(ex.map(work, rows)):
            if i % 100 == 0:
                print(f"{i} captured", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
