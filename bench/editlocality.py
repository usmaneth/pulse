#!/usr/bin/env python3
"""INVALID METHOD - KEPT AS A RECORD. Do not use. See bench/RESULTS.md Round 23.

This script reconstructs the context at each turn from stored transcripts and
compares each turn against the previous one with a longest-common-prefix test.
The method cannot answer the question it was written for. Transcript files are
append-only by construction, so a reconstruction from them always reports 100%
append, whatever the agent actually sent.

bench/ckpt_evict.py replaced it. That script counts the events which actually
mutate earlier context, and then measures the serving case directly.
"""

import glob, json, os, statistics as st, sys
from collections import Counter

def msg_text(m):
    """Flatten one transcript record into the text that would enter the prompt."""
    if not isinstance(m, dict): return None
    msg = m.get("message", m)
    if not isinstance(msg, dict): return None
    role = msg.get("role")
    if role not in ("user", "assistant", "system"): return None
    c = msg.get("content")
    if isinstance(c, str): body = c
    elif isinstance(c, list):
        parts = []
        for p in c:
            if not isinstance(p, dict): continue
            t = p.get("type")
            if t == "text": parts.append(p.get("text", ""))
            elif t == "tool_use": parts.append(json.dumps(p.get("input", {}))[:4000])
            elif t == "tool_result":
                parts.append(json.dumps(p.get("content"))[:4000])
        body = "\n".join(parts)
    else: return None
    return f"{role}:{body}"

def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i

def analyse(path):
    msgs = []
    try:
        for line in open(path, errors="replace"):
            line = line.strip()
            if not line: continue
            try: d = json.loads(line)
            except Exception: continue
            t = msg_text(d)
            if t is not None: msgs.append(t)
    except Exception: return []
    # context at turn i = all messages up to i
    ctxs, acc = [], []
    for m in msgs:
        acc.append(m)
        ctxs.append("\n".join(acc))
    out = []
    for i in range(1, len(ctxs)):
        prev, cur = ctxs[i-1], ctxs[i]
        common = lcp(prev, cur)
        frac = common / len(prev) if prev else 1.0
        out.append(frac)
    return out

# Claude Code names a project's log directory after the cwd path, with each
# `/` turned into `-` (so /home/alice/proj becomes -home-alice-proj).
_HOME = os.path.expanduser('~')
_PROJECT_SLUG = _HOME.replace('/', '-')
files = sorted(glob.glob(f'{_HOME}/.claude/projects/{_PROJECT_SLUG}/*.jsonl'))
files += sorted(glob.glob(f'{_HOME}/.omp/profiles/mafia/agent/sessions/-/*.jsonl'))
allf, per_session = [], []
for f in files:
    fr = analyse(f)
    if len(fr) < 5: continue
    allf += fr
    per_session.append((f.split('/')[-1][:24], len(fr), sum(1 for x in fr if x < 0.999)/len(fr)))

if not allf:
    print("no usable transcripts"); sys.exit(0)

pure = sum(1 for x in allf if x >= 0.999)
div  = len(allf) - pure
print(f"sessions analysed : {len(per_session)}")
print(f"turn transitions  : {len(allf)}")
print(f"  pure append (LCP = 100%) : {pure:6d}  ({100*pure/len(allf):5.1f}%)")
print(f"  diverged                 : {div:6d}  ({100*div/len(allf):5.1f}%)")
if div:
    dv = sorted(x for x in allf if x < 0.999)
    print(f"\n  when it diverges, how much of the previous context survives:")
    print(f"    median  {100*st.median(dv):5.1f}%")
    print(f"    p25     {100*dv[len(dv)//4]:5.1f}%")
    print(f"    p75     {100*dv[3*len(dv)//4]:5.1f}%")
    print(f"    worst   {100*dv[0]:5.1f}%")
    buckets = Counter()
    for x in dv:
        b = "0-25%" if x < .25 else "25-50%" if x < .5 else "50-75%" if x < .75 else "75-99%"
        buckets[b] += 1
    print(f"\n  divergence depth distribution (survived prefix):")
    for b in ("0-25%", "25-50%", "50-75%", "75-99%"):
        n = buckets.get(b, 0)
        print(f"    {b:>8}: {n:5d}  ({100*n/len(dv):5.1f}% of divergences)")
