#!/usr/bin/env python3
"""The full context x K surface.

Round 27 shipped a binary gate: K=4 below a cutoff, K=0 above it. That assumes
the best depth jumps straight from 4 to 0. It may instead taper, in which case
the medium band is being served at the wrong depth. This sweeps both axes.

K values are interleaved within each repeat so machine drift cannot masquerade
as a K effect.
"""
import glob, json, statistics, sys, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
KS = [0, 1, 2, 3, 4]
CTXS = [8192, 12288, 16384, 34000]
REPEATS = 3

pool = ""
for p in sorted(glob.glob('/home/usman/llama.cpp-upstream/src/*.cpp')):
    try: pool += open(p, errors='replace').read() + "\n"
    except Exception: pass

def post(b, t=900):
    r = urllib.request.Request(URL + "/completion", json.dumps(b).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as x:
        return json.loads(x.read())

hdr = "  " + f"{'ctx':>8}" + "".join(f"{'K='+str(k):>9}" for k in KS) + f"{'best':>7}{'gain vs K=0':>13}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
for tgt in CTXS:
    prompt = pool[:int(tgt * 3.6)] + "\n\n// Summarise the code above in one sentence.\n"
    base = {"prompt": prompt, "n_predict": 128, "temperature": 0, "cache_prompt": True}
    post({**base, "speculative.n_max": 4})   # warm cache + machine
    res = {k: [] for k in KS}
    n_tok = 0
    for _ in range(REPEATS):
        for k in KS:
            d = post({**base, "speculative.n_max": k}); t = d["timings"]
            res[k].append(t["predicted_per_second"])
            n_tok = t.get("prompt_n", 0) + t.get("cache_n", 0)
    med = {k: statistics.median(v) for k, v in res.items()}
    best = max(med, key=med.get)
    noisy = any((max(res[k]) - min(res[k])) / med[k] * 100 > 3.4 for k in KS)
    gain = (med[best] / med[0] - 1) * 100
    row = "  " + f"{n_tok:>8}" + "".join(f"{med[k]:9.2f}" for k in KS)
    print(row + f"{('K='+str(best)):>7}{gain:12.1f}%" + ("  NOISY" if noisy else ""))
