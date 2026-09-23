#!/usr/bin/env python3
"""Numeric drift gate for changes that touch the target model's own math.

Scores fixed texts with prompt logprobs (top-5 per position) and saves them per
tag. `--compare A B` reports top-1 agreement and an approximate KL(A||B) over
the shared top-5 support. Draft-only changes cannot move these numbers; a
change to the target (for example an FP8 output head) can.

    python3 driftgate.py --tag best --save
    python3 driftgate.py --compare best fp8head
"""
import argparse, json, math, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def texts():
    out = [open(f"{HERE}/long_context.txt").read()[i:i + 6000] for i in range(0, 60000, 12000)]
    for r in json.load(open(f"{HERE}/humaneval.json"))[:10]:
        out.append(r["prompt"] + r["canonical_solution"])
    return out


def score(port, model, text):
    body = {"model": model, "prompt": text, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": 5}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["choices"][0].get("prompt_logprobs") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.save:
        res = [score(a.port, a.model, t) for t in texts()]
        json.dump(res, open(f"{HERE}/drift.{a.tag}.json", "w"))
        print("saved", sum(len(r) for r in res), "positions")
        return
    A = json.load(open(f"{HERE}/drift.{a.compare[0]}.json"))
    B = json.load(open(f"{HERE}/drift.{a.compare[1]}.json"))
    agree = n = 0
    kl = 0.0
    for ra, rb in zip(A, B):
        for pa, pb in zip(ra, rb):
            if not pa or not pb:
                continue
            ta = max(pa.items(), key=lambda kv: kv[1]["logprob"])[0]
            tb = max(pb.items(), key=lambda kv: kv[1]["logprob"])[0]
            agree += ta == tb; n += 1
            for tok, v in pa.items():
                if tok in pb:
                    p = math.exp(v["logprob"])
                    kl += p * (v["logprob"] - pb[tok]["logprob"])
    print(json.dumps({"compare": a.compare, "positions": n,
                      "top1_agreement": round(agree / max(n, 1), 5),
                      "approx_kl_per_pos": round(kl / max(n, 1), 6)}))


if __name__ == "__main__":
    main()
