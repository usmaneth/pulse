#!/usr/bin/env python3
"""Does --cache-reuse help when the agent EDITS context mid-stream?

Exact-prefix reuse (Round 18) only helps when the context grows append-only.
Real agents edit: they replace a file's contents, drop a tool result, rewrite a
plan. Any change before the tail invalidates an exact-prefix cache.

llama.cpp's `--cache-reuse N` reuses via KV shifting across such a divergence.
It defaults to 0 (disabled).

Scenarios, all after the same prefix is already cached:
  A. append-only   (exact prefix + new suffix)   <- best case, Round 18
  B. mid-edit      (a chunk in the middle changed)
  C. prefix-drop   (an early chunk removed)
"""
import glob, json, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 8192

def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def ntok(t): return len(post("/tokenize", {"content": t})["tokens"])

pool = ""
for p in sorted(glob.glob('/home/usman/llama.cpp-upstream/src/*.cpp') +
                glob.glob('/home/usman/llama.cpp-upstream/common/*.cpp')):
    try: pool += open(p, errors='replace').read() + "\n"
    except Exception: pass

lo, hi = 1, min(len(pool), TARGET*12)
for _ in range(14):
    mid = (lo+hi)//2
    if ntok(pool[:mid]) < TARGET: lo = mid+1
    else: hi = mid
base = pool[:hi]
n = ntok(base)
print(f"prefix: {n} tokens\n")

def prefill_ms(prompt, label):
    d = post("/completion", {"prompt": prompt, "n_predict": 8, "temperature": 0,
                             "ignore_eos": True, "cache_prompt": True})
    ms = d.get("timings", {}).get("prompt_ms", 0.0)
    print(f"  {label:<40} prompt_ms = {ms:9.1f}")
    return ms

third = len(base)//3
cold   = prefill_ms(base, "0. prime the cache (cold)")
a      = prefill_ms(base + "\n// append: refactor the above\n", "A. append-only")
edited = base[:third] + "\n// EDITED BLOCK REPLACED HERE\n" + base[third+2000:]
b      = prefill_ms(edited + "\n// now continue\n", "B. mid-context edit")
dropped= base[:third] + base[third+4000:]
c      = prefill_ms(dropped + "\n// now continue\n", "C. early chunk dropped")
print()
for lbl, v in [("A append-only", a), ("B mid-edit", b), ("C prefix-drop", c)]:
    print(f"  {lbl:<18} {cold/max(v,0.001):7.1f}x vs cold   ({v:.0f} ms)")
