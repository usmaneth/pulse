#!/usr/bin/env python3
"""Per-request draft depth K, swept per workload class.

This became measurable only after per-request `speculative.n_max` was enabled in
llama.cpp (see patches/). Before that a server had one global K for all traffic.

Interleaves K values within each repeat so machine drift cannot masquerade as a
K effect.
"""
import json, statistics, sys, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8085"
KS = [1, 2, 3, 4, 5, 6, 7, 8]
REPEATS = 3

WORKLOADS = {
    "code": ("Write a Python class named InventoryManager with methods add_item, "
             "remove_item, and get_stock. Use a dataclass for Item.\n\n"),
    "schema": ("Emit a JSON schema for a user record with fields id, email, "
               "created_at, is_active, and roles.\n\n"),
    "chat": ("Explain why the sky appears blue to someone who has no physics "
             "background. Keep it conversational.\n\n"),
}

def post(body, timeout=300):
    req = urllib.request.Request(URL + "/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def run(prompt, k):
    d = post({"prompt": prompt, "n_predict": 128, "temperature": 0,
              "cache_prompt": False, "speculative.n_max": k})
    return d.get("timings", {}).get("predicted_per_second", 0.0)

for name, prompt in WORKLOADS.items():
    run(prompt, 4)  # warmup, discarded
    res = {k: [] for k in KS}
    for _ in range(REPEATS):
        for k in KS:
            res[k].append(run(prompt, k))
    med = {k: statistics.median(v) for k, v in res.items()}
    best = max(med, key=med.get)
    print(f"\n  {name}")
    for k in KS:
        spread = (max(res[k]) - min(res[k])) / statistics.median(res[k]) * 100
        mark = "  <-- best" if k == best else ""
        print(f"    K={k}  {med[k]:6.2f} tok/s   spread {spread:4.1f}%{mark}")
    print(f"    best K = {best} at {med[best]:.2f} tok/s "
          f"({med[best]/med[1]:.2f}x over K=1)")
