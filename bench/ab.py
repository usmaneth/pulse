#!/usr/bin/env python3
"""Rigorous A/B harness for speculative decoding on GB10.

This machine's throughput is state-dependent: measured 10.2% run-to-run spread,
16% loss under CPU memory traffic, and hysteresis (throughput climbs across
consecutive runs as unified-memory pages migrate back to the GPU).

So: quiesce, warm up, interleave configs (never run A's repeats then B's -
drift would masquerade as a difference), take medians, and refuse to call a
result real unless it clears the measured noise floor.

Usage:
  ab.py --config "label=ENV1=v ENV2=v -- --flag val" --config "..." [-n 6]
"""
import argparse, json, os, re, shlex, statistics as st, subprocess, sys, time

BIN   = os.environ.get("SPEC_BIN", "/home/usman/Bonsai-demo/bin/cuda/llama-speculative-simple")
ROOT  = os.environ.get("SPEC_ROOT", "/home/usman/Bonsai-demo")
MODEL = os.environ.get("SPEC_MODEL", "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf")
DRAFT = os.environ.get("SPEC_DRAFT", "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf")
NOISE_FLOOR_PCT = 10.2

BASE = ["-m", MODEL, "-md", DRAFT, "--spec-type", "draft-dspark", "--spec-draft-n-max", "7",
        "-ngl", "99", "-ngld", "999", "-fa", "on", "-c", "8192", "-b", "4096", "-ub", "512",
        "--temp", "0", "--ignore-eos"]

def gpu_clean():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return not r.stdout.strip()

def load_avg():
    try:    return float(open("/proc/loadavg").read().split()[0])
    except: return 0.0

def one(env_extra, argv_extra, prompt_file, npred):
    env = dict(os.environ); env.update(env_extra)
    cmd = [BIN] + BASE + ["-n", str(npred), "-f", prompt_file] + argv_extra
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    o = r.stdout + r.stderr
    m = re.search(r"decoded\s+(\d+) tokens in\s+([\d.]+) seconds, speed:\s+([\d.]+)", o)
    if not m:
        err = re.findall(r"(?i)error[^\n]{0,70}", o)
        raise RuntimeError(err[0] if err else "no decode line")
    dec, secs, tps = int(m.group(1)), float(m.group(2)), float(m.group(3))
    dr = int(re.search(r"n_drafted = (\d+)", o).group(1))
    ac = int(re.search(r"n_accept\s+= (\d+)", o).group(1))
    steps = max(dec - ac, 1)
    return {"tps": tps, "acc": 100.0*ac/dr if dr else 0.0,
            "tok_step": dec/steps, "ms_step": 1000.0*secs/steps, "wall": time.perf_counter()-t0}

def parse_config(spec):
    label, rest = spec.split("=", 1)
    env_part, _, argv_part = rest.partition("--")
    env = {}
    for tok in shlex.split(env_part):
        if "=" in tok:
            k, v = tok.split("=", 1); env[k] = v
    return label, env, shlex.split(argv_part)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("-n", "--repeats", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--npred", type=int, default=192)
    ap.add_argument("--prompt", default="/tmp/code_ctx.txt")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if not gpu_clean() and not a.force:
        sys.exit("GPU is busy. Quiesce or pass --force.")
    if load_avg() > 1.5 and not a.force:
        print(f"WARNING: load average {load_avg():.2f} - CPU traffic steals GPU bandwidth here.")

    cfgs = [parse_config(c) for c in a.config]
    print(f"{len(cfgs)} configs, {a.warmup} warmup + {a.repeats} interleaved repeats, "
          f"noise floor {NOISE_FLOOR_PCT}%\n")

    for label, env, argv in cfgs:
        for _ in range(a.warmup):
            try: one(env, argv, a.prompt, a.npred)
            except Exception as e: sys.exit(f"{label} warmup failed: {e}")
    print("warmup done\n")

    res = {label: [] for label, _, _ in cfgs}
    for r in range(a.repeats):                      # interleave so drift hits every config equally
        for label, env, argv in cfgs:
            try: res[label].append(one(env, argv, a.prompt, a.npred))
            except Exception as e: print(f"  {label} run {r+1} FAILED: {e}")
        print(f"  round {r+1}/{a.repeats} done", flush=True)

    print(f"\n{'config':<34} {'median':>8} {'min':>8} {'max':>8} {'spread':>8} {'accept%':>8} {'ms/step':>8}")
    print("-" * 92)
    meds = {}
    for label, _, _ in cfgs:
        v = res[label]
        if not v: print(f"{label:<34} {'ALL FAILED':>8}"); continue
        tps = sorted(x["tps"] for x in v); med = st.median(tps)
        meds[label] = med
        print(f"{label:<34} {med:>8.2f} {tps[0]:>8.2f} {tps[-1]:>8.2f} "
              f"{100*(tps[-1]-tps[0])/med:>7.1f}% {st.median([x['acc'] for x in v]):>7.2f}% "
              f"{st.median([x['ms_step'] for x in v]):>8.2f}")
    if len(meds) > 1:
        base = list(meds.items())[0]
        print(f"\nvs baseline '{base[0]}' ({base[1]:.2f} tok/s):")
        for label, med in list(meds.items())[1:]:
            d = 100*(med-base[1])/base[1]
            verdict = "REAL" if abs(d) > NOISE_FLOOR_PCT else f"within noise (<{NOISE_FLOOR_PCT}%)"
            print(f"  {label:<32} {d:>+6.1f}%   {verdict}")
    json.dump({k: v for k, v in res.items()}, open("/tmp/ab_result.json", "w"), indent=1)

main()
