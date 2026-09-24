#!/usr/bin/env python3
"""Decode throughput vs context length against a PERSISTENT llama.cpp server.

Why a server: a sweep that spawns one process per context length measures
cold-start, not context (Round 16). Keeping the model resident removes that.

Prompts are built from real source text and truncated by the server's own
tokenizer count, so the x-axis is true token counts, not character estimates.
"""
import json, statistics as st, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
TARGETS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                            else "128,256,512,1024,4096,16384,65536".split(","))]
NPRED = int(sys.argv[3]) if len(sys.argv) > 3 else 128
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3

def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"])

# build a large natural pool
import os
import glob
pool = ""
for p in sorted(glob.glob(os.path.expanduser('~/llama.cpp-upstream/src/*.cpp')) +
                glob.glob(os.path.expanduser('~/llama.cpp-upstream/common/*.cpp')) +
                glob.glob(os.path.expanduser('~/llama.cpp-upstream/ggml/src/ggml-cuda/*.cu'))):
    try: pool += open(p, errors='replace').read() + "\n"
    except Exception: pass

def prompt_of(target):
    """Binary-search the character count that yields ~target tokens."(server tokenizer)"""
    lo, hi = 1, min(len(pool), target * 12)
    best = pool[:hi]
    for _ in range(12):
        mid = (lo + hi) // 2
        n = ntok(pool[:mid])
        if n < target: lo = mid + 1
        else: hi = mid; best = pool[:mid]
        if abs(n - target) <= max(4, target // 200): return pool[:mid], n
    return best, ntok(best)

print(f"{'ctx tok':>8} | {'decode t/s':>10} | {'prefill t/s':>11} | {'accept%':>7} | {'ms/tok':>7}")
print("-" * 60)
for tgt in TARGETS:
    try:
        text, n = prompt_of(tgt)
    except Exception as e:
        print(f"{tgt:>8} | prompt build failed: {e}"); continue
    runs = []
    for r in range(REPS + 1):          # first is warmup, discarded
        try:
            d = post("/completion", {"prompt": text, "n_predict": NPRED, "temperature": 0,
                                     "ignore_eos": True, "cache_prompt": False})
        except Exception as e:
            print(f"{n:>8} | FAILED: {str(e)[:40]}"); runs = []; break
        if r == 0: continue
        t = d.get("timings", {})
        runs.append((t.get("predicted_per_second", 0.0), t.get("prompt_per_second", 0.0),
                     d.get("draft_n", 0), d.get("draft_n_accepted", 0)))
    if not runs: continue
    dec = st.median([x[0] for x in runs]); pre = st.median([x[1] for x in runs])
    dn = sum(x[2] for x in runs); da = sum(x[3] for x in runs)
    acc = (100.0*da/dn) if dn else float('nan')
    print(f"{n:>8} | {dec:>10.2f} | {pre:>11.1f} | {acc:>6.2f}% | {1000/dec if dec else 0:>7.2f}")
