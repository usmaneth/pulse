#!/usr/bin/env python3
"""Does the checkpoint layer pay for itself when sessions EVICT each other?

The server runs with -np 1, so there is exactly one slot. Two interleaved agent
sessions therefore evict each other on every turn. That is the case the
checkpoint layer exists for - not the mid-edit case, which transcript analysis
shows is rare (0.59% of records carry a context-mutating marker).

Sequence per repeat:  A1 (cold) -> B1 (evicts A) -> A2 (return) -> B2 (return)
A2 and B2 are the measurement: returning to a session whose slot is gone.
"""
import glob, json, statistics, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), (time.perf_counter() - t0) * 1000

def status():
    try:
        with urllib.request.urlopen(URL + "/status", timeout=10) as r:
            return json.loads(r.read()).get("checkpoints", {})
    except Exception:
        return {}

pool = ""
for p in sorted(glob.glob('/home/usman/llama.cpp-upstream/src/*.cpp')):
    try:
        pool += open(p, errors='replace').read() + "\n"
    except Exception:
        pass
if len(pool) < 40000:
    sys.exit("not enough source text to build prefixes")

N = 7000
A_CTX = pool[:N]
B_CTX = pool[20000:20000 + N]

def turn(ctx, tag, nth):
    msgs = [{"role": "user", "content": f"File {tag}:\n{ctx}\nSummarise it."}]
    if nth > 1:
        msgs += [{"role": "assistant", "content": "ok"},
                 {"role": "user", "content": f"Question {nth} about file {tag}?"}]
    _, wall = post("/v1/chat/completions",
                   {"messages": msgs, "max_tokens": 8, "temperature": 0, "stream": False})
    return wall

ret_a, ret_b, cold = [], [], []
for rep in range(REPEATS):
    cold.append(turn(A_CTX, "A", 1))          # A cold
    turn(B_CTX, "B", 1)                       # B cold, evicts A
    ret_a.append(turn(A_CTX, "A", 2))         # return to evicted A
    ret_b.append(turn(B_CTX, "B", 2))         # return to evicted B
    print(f"  repeat {rep+1}/{REPEATS}: coldA={cold[-1]:.0f} retA={ret_a[-1]:.0f} retB={ret_b[-1]:.0f} ms",
          flush=True)

med = lambda x: statistics.median(x)
print()
print(f"  cold first contact       median {med(cold):9.1f} ms")
print(f"  return to evicted A      median {med(ret_a):9.1f} ms")
print(f"  return to evicted B      median {med(ret_b):9.1f} ms")
print(f"  return (A+B combined)    median {med(ret_a+ret_b):9.1f} ms")
print()
print(f"  checkpoint stats: {json.dumps(status())}")
