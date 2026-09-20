#!/usr/bin/env python3
"""Do the concurrency and context axes interact?

The serving policy has two independent rules, measured separately:
  - concurrency: drafter below 12 clients, none above (Round 20)
  - context:     K tapers 4 -> 3 -> 2 -> 0 as context grows (Round 28)

Neither was measured in the presence of the other. If they interact, the shipped
policy is wrong somewhere. This measures aggregate throughput (total tokens /
wall clock) across both axes, with speculation on and off.

Each client gets a DIFFERENT prompt offset so they do not share a slot cache,
which is what real multi-session serving looks like.
"""
import glob, json, statistics, sys, threading, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
REPEATS = 3

pool = ""
for p in sorted(glob.glob('/home/usman/llama.cpp-upstream/src/*.cpp')):
    try: pool += open(p, errors='replace').read() + "\n"
    except Exception: pass

def post(b, t=1800):
    r = urllib.request.Request(URL + "/completion", json.dumps(b).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as x:
        return json.loads(x.read())

def burst(ctx_tokens, clients, k):
    chars = int(ctx_tokens * 3.6)
    out = [None] * clients
    def one(i):
        # distinct prefix per client so slots do not share a cache
        prompt = pool[i * 5000: i * 5000 + chars] + f"\n\n// client {i}: summarise.\n"
        b = {"prompt": prompt, "n_predict": 96, "temperature": 0,
             "cache_prompt": True, "speculative.n_max": k}
        try: out[i] = post(b)["timings"]
        except Exception as e: out[i] = {"err": str(e)}
    ths = [threading.Thread(target=one, args=(i,)) for i in range(clients)]
    t0 = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - t0
    tok = sum(t.get("predicted_n", 0) for t in out if t and "err" not in t)
    return tok / wall if wall else 0.0

print(f"  {'ctx':>7} {'clients':>8} {'spec K=4':>10} {'no spec':>9} {'winner':>9} {'delta':>8}")
for ctx in (2048, 8192, 16384, 34000):
    for clients in (1, 4):
        burst(ctx, clients, 4)                      # warm
        res = {}
        for k in (4, 0):
            v = [burst(ctx, clients, k) for _ in range(REPEATS)]
            res[k] = statistics.median(v)
        win = "spec" if res[4] > res[0] else "no spec"
        d = (max(res.values()) / min(res.values()) - 1) * 100 if min(res.values()) else 0
        print(f"  {ctx:>7} {clients:>8} {res[4]:10.2f} {res[0]:9.2f} {win:>9} {d:7.1f}%", flush=True)
