#!/usr/bin/env python3
import subprocess
import re
import sys

PROMPTS = [
    ("Binary Search", "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Perform binary search on a sorted list and return target index or -1.\"\"\"\n    left = 0\n    right = len(arr) - 1\n", 120),
    ("Merge Intervals", "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    \"\"\"Merge all overlapping intervals and return non-overlapping intervals.\"\"\"\n    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [intervals[0]]\n", 120),
    ("SQL DDL Schema", "CREATE TABLE telemetry_metrics (\n    metric_id BIGSERIAL PRIMARY KEY,\n    device_uuid UUID NOT NULL,\n    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\n    cpu_utilization_percent REAL CHECK (cpu_utilization_percent >= 0.0 AND cpu_utilization_percent <= 100.0),\n    memory_allocated_bytes BIGINT NOT NULL,\n", 120),
    ("Prime Fact Proof", "Task: Find the prime factorization of 3,240 and prove that the number of positive divisors is exactly 40.\nStep 1: Factor 3,240 by repeated trial division by smallest primes.\n3240 = 2 * 1620\n", 130),
    ("FlashAttention Prose", "Explain the fundamental mechanism of FlashAttention (tiling, online softmax re-computation, and SRAM memory hierarchy):\n1. Tiling in SRAM:\n", 120),
]

BIN_SPEC = "/home/usman/Bonsai-demo/bin/cuda/llama-speculative-simple"
MODEL = "/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
DRAFTER = "/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf"

def run_prompt(prompt, n_pred, use_drafter):
    cmd = [
        BIN_SPEC,
        "-m", MODEL,
        "-ngl", "99",
        "-fa", "on",
        "-c", "4096",
        "-n", str(n_pred),
        "--temp", "0",
        "-p", prompt
    ]
    if use_drafter:
        cmd.extend([
            "-md", DRAFTER,
            "--spec-type", "draft-dspark",
            "--spec-draft-n-max", "5",
            "-ngld", "999"
        ])
    else:
        cmd.extend(["--spec-type", "none"])

    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + "\n" + res.stderr

    acc_match = re.search(r"accept\s*=\s*([\d\.]+)%", out)
    speed_match = re.search(r"decoded\s+(\d+)\s+tokens in\s+([\d\.]+)\s+seconds,\s+speed:\s+([\d\.]+)\s+t/s", out)

    speed = float(speed_match.group(3)) if speed_match else 0.0
    acc = float(acc_match.group(1)) if acc_match else 0.0
    tokens = int(speed_match.group(1)) if speed_match else 0
    return speed, acc, tokens

print("=" * 86)
print(" REAL-PROMPT GROUNDED BENCHMARK: BASELINE VS SPECULATIVE (EXACT SAME PROMPTS)")
print("=" * 86)
print(f"| {'Workload Prompt':<22} | {'Tokens':<6} | {'Base (No Drafter)':<17} | {'DSpark v2 (K=5)':<15} | {'Accept %':<8} | {'Real Speedup':<12} |")
print("|" + "-" * 24 + "|" + "-" * 8 + "|" + "-" * 19 + "|" + "-" * 17 + "|" + "-" * 10 + "|" + "-" * 14 + "|")

for name, prompt, n_pred in PROMPTS:
    base_spd, _, base_toks = run_prompt(prompt, n_pred, use_drafter=False)
    spec_spd, acc, spec_toks = run_prompt(prompt, n_pred, use_drafter=True)
    speedup = (spec_spd / base_spd) if base_spd > 0 else 1.0
    print(f"| {name:<22} | {spec_toks:<6} | {base_spd:5.1f} tok/s       | {spec_spd:5.1f} tok/s       | {acc:5.1f}%   | {speedup:5.2f}x       |", flush=True)

print("=" * 86)
