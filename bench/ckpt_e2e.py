#!/usr/bin/env python3
"""End-to-end: does the Pulse checkpoint layer beat a full re-prefill on a mid-edit?

Simulates a 3-turn agent conversation, then a 4th turn that EDITS turn 2's
content. Without checkpoints that is a full re-prefill; with them it should
restore the turn-1 boundary and prefill only from there.
"""
import os
import glob, json, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

def post(path, body, timeout=900):
    req = urllib.request.Request(URL+path, json.dumps(body).encode(), {"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), (time.perf_counter()-t0)*1000

pool=""
for p in sorted(glob.glob(os.path.expanduser('~/llama.cpp-upstream/src/*.cpp'))):
    try: pool+=open(p,errors='replace').read()+"\n"
    except Exception: pass
CHUNK = 6000
c1, c2, c3 = pool[:CHUNK], pool[CHUNK:2*CHUNK], pool[2*CHUNK:3*CHUNK]

def chat(msgs, label):
    d, wall = post("/v1/chat/completions",
                   {"messages": msgs, "max_tokens": 16, "temperature": 0, "stream": False})
    meta = d.get("pulse_meta", {})
    print(f"  {label:<44} wall = {wall:8.1f} ms")
    return wall

m1 = [{"role":"user","content":f"Here is file A:\n{c1}\nSummarise it."}]
w1 = chat(m1, "turn 1 (cold)")
m2 = m1 + [{"role":"assistant","content":"ok"},
           {"role":"user","content":f"Here is file B:\n{c2}\nSummarise it."}]
w2 = chat(m2, "turn 2 (append)")
m3 = m2 + [{"role":"assistant","content":"ok"},
           {"role":"user","content":f"Here is file C:\n{c3}\nSummarise it."}]
w3 = chat(m3, "turn 3 (append)")

# turn 4 EDITS turn 2's content -> everything from turn 2 onward diverges
m4 = list(m3)
m4[2] = {"role":"user","content":f"Here is file B (REVISED):\n{c2[::-1][:CHUNK]}\nSummarise it."}
m4 = m4 + [{"role":"assistant","content":"ok"},{"role":"user","content":"Now compare them."}]
w4 = chat(m4, "turn 4 (EDITS turn 2 -> divergence)")

print()
try:
    st = json.loads(urllib.request.urlopen(URL+"/status", timeout=10).read())
    c = st.get("checkpoints", {})
    print(f"  checkpoints: entries={c.get('entries')} saves={c.get('saves')} "
          f"restores={c.get('restores')} hits={c.get('hits')} misses={c.get('misses')}")
    print(f"  last restore: {c.get('lastRestoreMs')} ms   last save: {c.get('lastSaveMs')} ms")
    print(f"  last_checkpoint_restore: {st.get('last_checkpoint_restore')}")
except Exception as e:
    print("  /status unavailable:", e)
