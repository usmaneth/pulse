#!/usr/bin/env python3
"""Target-only decode throughput, for tuning the GB10 mmvq nwarps for PQ2_0.

Speculation is disabled (speculative.n_max=0) so this isolates the target
model's matvec path, which is what calc_nwarps() governs. Short context so KV
reads do not dominate.
"""
import json, statistics, sys, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
P = "Write a Python function that reverses a string and explain how it works.\n\n"
def run():
    b = {"prompt": P, "n_predict": 128, "temperature": 0,
         "cache_prompt": False, "speculative.n_max": 0}
    r = urllib.request.Request(URL + "/completion", json.dumps(b).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as x:
        return json.loads(x.read())["timings"]["predicted_per_second"]
for _ in range(2): run()
v = [run() for _ in range(6)]
m = statistics.median(v)
print(f"  target-only decode: median {m:6.2f} tok/s   "
      f"spread {(max(v)-min(v))/m*100:4.1f}%   runs {[round(x,1) for x in v]}")
