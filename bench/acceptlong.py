#!/usr/bin/env python3
"""Does draft acceptance hold up at long context?

Round 25 concluded that deeper drafting does not pay, from sweeps at short
context. That conclusion named one escape hatch: a drafter with materially
better acceptance AT LONG CONTEXT would still be worth training. This measures
whether acceptance actually decays with context, which is the input to that
decision and was never measured.

Reports acceptance (draft_n_accepted / draft_n) and decode tok/s per context
length, warm, with the prompt cached so prefill is not what is being timed.
"""
import glob, json, statistics, sys, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
REPEATS = 3

pool = ""
for p in sorted(glob.glob('/home/usman/llama.cpp-upstream/src/*.cpp')):
    try: pool += open(p, errors='replace').read() + "\n"
    except Exception: pass

def post(body, timeout=900):
    req = urllib.request.Request(URL + "/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

# ~3.6 chars/token for source text; targets are approximate token counts
TARGETS = [512, 2048, 8192, 16384, 32768]

print(f"  {'ctx tokens':>12} {'accept %':>10} {'tok/s':>8} {'spread':>8} {'draft_n':>9}")
for tgt in TARGETS:
    chars = int(tgt * 3.6)
    if chars > len(pool):
        print(f"  {tgt:>12}  (not enough source text)"); continue
    prompt = pool[:chars] + "\n\n// Summarise the code above in one sentence.\n"
    body = {"prompt": prompt, "n_predict": 128, "temperature": 0, "cache_prompt": True}
    post(body)  # warm the cache and the machine; discarded
    acc, tps, dn, pn = [], [], [], None
    for _ in range(REPEATS):
        d = post(body); t = d.get("timings", {})
        pn = t.get("prompt_n", 0) + t.get("cache_n", 0)
        if t.get("draft_n"):
            acc.append(100.0 * t["draft_n_accepted"] / t["draft_n"]); dn.append(t["draft_n"])
        tps.append(t.get("predicted_per_second", 0.0))
    m = statistics.median(tps); sp = (max(tps) - min(tps)) / m * 100 if m else 0
    a = f"{statistics.median(acc):9.2f}" if acc else "      n/a"
    d_ = f"{statistics.median(dn):9.0f}" if dn else "      n/a"
    flag = "" if sp < 3.4 else "  NOISY"
    print(f"  {pn:>12} {a} {m:8.2f} {sp:7.1f}% {d_}{flag}")
