#!/usr/bin/env python3
"""Concurrency benchmark against a llama.cpp server. Stdlib only.

Measures aggregate and per-stream decode throughput at several client counts.
All values come from the server's own timings block. Nothing is extrapolated.
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
LEVELS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1","2","4","8","16"])]
NPRED = int(sys.argv[3]) if len(sys.argv) > 3 else 128

PROMPTS = [
    "Write a Python function that merges overlapping intervals. Include type hints.",
    "Write a Python binary search over a sorted list returning the index or -1.",
    "Write a Python dataclass NetworkPacket with src_ip, dest_ip, payload, checksum.",
    "Write a Python LRU cache class with get and put in O(1). Include type hints.",
]

def one(i):
    body = json.dumps({
        "prompt": PROMPTS[i % len(PROMPTS)],
        "n_predict": NPRED, "temperature": 0, "ignore_eos": True, "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(URL + "/completion", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    wall = time.perf_counter() - t0
    t = d.get("timings", {})
    return {
        "wall": wall,
        "pred_n": t.get("predicted_n", 0),
        "pred_ms": t.get("predicted_ms", 0.0),
        "tps_server": t.get("predicted_per_second", 0.0),
        "draft_n": d.get("draft_n", t.get("draft_n", 0)),
        "draft_accept": d.get("draft_n_accepted", t.get("draft_n_accepted", 0)),
    }

print(f"{'N':>3} | {'aggregate t/s':>13} | {'per-stream t/s':>14} | {'accept%':>7} | {'wall s':>7}")
print("-" * 60)
base = None
for n in LEVELS:
    one(0)  # warm
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.perf_counter() - t0
    toks = sum(r["pred_n"] for r in res)
    agg = toks / wall
    per = sum(r["tps_server"] for r in res) / len(res)
    dn = sum(r["draft_n"] for r in res); da = sum(r["draft_accept"] for r in res)
    acc = (100.0 * da / dn) if dn else float("nan")
    if base is None: base = agg
    print(f"{n:>3} | {agg:>13.2f} | {per:>14.2f} | {acc:>6.2f}% | {wall:>7.2f}   ({agg/base:.2f}x)")
