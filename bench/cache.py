#!/usr/bin/env python3
"""Prefix-cache benefit: the agentic turn pattern.

An agent re-sends a growing context every turn. Turn 1 pays a cold prefill;
turns 2..N should reuse the cached prefix and pay only for the new tokens.

llama.cpp's server supports this via `cache_prompt`. Every benchmark in this
repo before now ran with cache_prompt=false, which measures the cold path only.

Measures time-to-first-token for: cold, exact replay, and a realistic
turn (same prefix + a short new user message).
"""
import glob, json, statistics as st, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
NPRED = int(sys.argv[3]) if len(sys.argv) > 3 else 32

def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.perf_counter() - t0

def ntok(t):
    return len(post("/tokenize", {"content": t})[0]["tokens"])

pool = ""
for p in sorted(glob.glob('/home/usman/llama.cpp-upstream/src/*.cpp') +
                glob.glob('/home/usman/llama.cpp-upstream/common/*.cpp') +
                glob.glob('/home/usman/llama.cpp-upstream/ggml/src/ggml-cuda/*.cu')):
    try: pool += open(p, errors='replace').read() + "\n"
    except Exception: pass

# size the prompt by real token count
lo, hi = 1, min(len(pool), TARGET*12)
for _ in range(14):
    mid = (lo+hi)//2
    n = ntok(pool[:mid])
    if n < TARGET: lo = mid+1
    else: hi = mid
base = pool[:hi]
nbase = ntok(base)
print(f"prefix: {nbase} tokens\n")

def run(prompt, cache, label):
    d, wall = post("/completion", {"prompt": prompt, "n_predict": NPRED, "temperature": 0,
                                   "ignore_eos": True, "cache_prompt": cache})
    t = d.get("timings", {})
    pms = t.get("prompt_ms", 0.0)
    print(f"  {label:<34} prompt_ms={pms:9.1f}  wall={wall*1000:9.1f} ms  "
          f"decode={t.get('predicted_per_second',0):6.2f} t/s")
    return pms

print(f"{'scenario':<36} {'prefill time':>14}")
print("-"*66)
cold  = run(base, False, "1. cold (cache_prompt=false)")
warm1 = run(base, True,  "2. first with cache on")
warm2 = run(base, True,  "3. exact replay, cached")
turn  = run(base + "\n\n// Now refactor the function above.\n", True,
            "4. agentic turn (prefix + new msg)")
print("-"*66)
if cold and warm2:
    print(f"\n  exact replay speedup : {cold/max(warm2,0.001):8.1f}x   ({cold:.0f} ms -> {warm2:.0f} ms)")
if cold and turn:
    print(f"  agentic turn speedup : {cold/max(turn,0.001):8.1f}x   ({cold:.0f} ms -> {turn:.0f} ms)")
