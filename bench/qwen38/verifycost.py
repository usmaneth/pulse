#!/usr/bin/env python3
"""Price a wide verify step: forward time vs number of new tokens on a cached prefix.

A suffix drafter proposes long drafts (8-16+ tokens); on this 512-expert MoE the
target's verify cost grows with the number of tokens because each token wakes
its own experts. This sends a fixed ~2k-token prefix (warmed in the prefix
cache), then prefix + N fresh tokens with max_tokens=1, and reports the median
TTFT per N minus the N=1 time. TTFT here is one forward of N tokens plus
scheduling, so the delta approximates the marginal verify cost per token.
Two texts are used: fresh prose (random routing) and a span copied from the
prefix (the suffix-drafter case, where routing may overlap).
"""
import json, statistics, time, urllib.request

PORT = 8888
BASE = open("/models/usman/qwen38-tune/long_context.txt").read()[:9000]


def ttft(prompt):
    body = {"model": "qwen3.8-flash-next", "prompt": prompt, "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    return time.time() - t


def main():
    words_fresh = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
                   "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
    copied = BASE[2000:4000]
    ttft(BASE)  # warm the prefix cache
    res = {}
    for kind in ("fresh", "copied"):
        base_t = None
        for n in (1, 4, 8, 16, 32):
            if kind == "fresh":
                tail = " " + " ".join(words_fresh[i % len(words_fresh)] + str(i) for i in range(n // 2 + 1))
            else:
                tail = copied[: n * 4]
            ts = [ttft(BASE + tail) for _ in range(5)]
            med = statistics.median(ts[1:])
            base_t = base_t or med
            res[f"{kind}_{n}"] = {"ttft_ms": round(med * 1000, 1), "delta_ms": round((med - base_t) * 1000, 1)}
            print(kind, n, res[f"{kind}_{n}"], flush=True)
    json.dump(res, open("/models/usman/qwen38-tune/verifycost.json", "w"), indent=1)


if __name__ == "__main__":
    main()
