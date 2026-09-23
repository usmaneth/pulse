#!/usr/bin/env python3
"""Break down a vLLM torch-profiler trace of single-stream decode.

Reads the newest *.pt.trace.json(.gz) under prof/, keeps GPU kernel events and
reports: wall time covered, GPU busy time, GPU idle gaps (count and total by
size bucket), and the top kernels by total time. Idle time between kernels is
the "overhead" the decode loop pays on top of streaming weights.
"""
import glob, gzip, json, os, sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "/models/usman/qwen38-tune/prof"
files = sorted(glob.glob(os.path.join(d, "**", "*.json*"), recursive=True), key=os.path.getmtime)
path = files[-1]
op = gzip.open if path.endswith(".gz") else open
tr = json.load(op(path, "rt"))
ev = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
ev.sort(key=lambda e: e["ts"])
if not ev:
    sys.exit(f"no GPU events in {path}")
start, end = ev[0]["ts"], max(e["ts"] + e["dur"] for e in ev)
busy = 0.0
gaps = defaultdict(lambda: [0, 0.0])
cur_end = ev[0]["ts"]
for e in ev:
    if e["ts"] > cur_end:
        g = e["ts"] - cur_end
        b = "<5us" if g < 5 else "5-50us" if g < 50 else "50-500us" if g < 500 else ">=500us"
        gaps[b][0] += 1; gaps[b][1] += g
    busy += max(0, e["ts"] + e["dur"] - max(e["ts"], cur_end)) if e["ts"] + e["dur"] > cur_end else 0
    cur_end = max(cur_end, e["ts"] + e["dur"])
tot = defaultdict(float); cnt = defaultdict(int)
for e in ev:
    tot[e["name"][:90]] += e["dur"]; cnt[e["name"][:90]] += 1
wall = end - start
print(f"trace: {os.path.basename(path)}")
print(f"wall {wall/1e3:.1f} ms  gpu busy {busy/1e3:.1f} ms ({100*busy/wall:.1f}%)  kernels {len(ev)}")
for b in ("<5us", "5-50us", "50-500us", ">=500us"):
    n, t = gaps[b]; print(f"  idle gaps {b:>9}: {n:6d}  total {t/1e3:8.1f} ms")
print("top kernels by total time:")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:20]:
    print(f"  {v/1e3:8.1f} ms  x{cnt[k]:5d}  {k}")
