import subprocess
import re
import json
import sys

PROMPTS = {
    "python_interval_merging": """def merge_intervals(intervals):
    \"\"\"Given an array of intervals where intervals[i] = [start_i, end_i], merge all overlapping intervals.\"\"\"
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]""",

    "binary_search": """def binary_search(arr, target):
    \"\"\"Perform binary search on a sorted array and return index or -1.\"\"\"
    left = 0
    right = len(arr) - 1""",

    "bat_and_ball": """Q: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Let's solve this step by step:
Let x be the cost of the ball in dollars.
Then the bat costs x + 1.00 dollars.""",

    "arithmetic_chain": """Problem: Compute the sum of the first 20 even integers step by step:
Step 1: The first 20 even positive integers are 2, 4, 6, ..., 40.
Step 2: This forms an arithmetic sequence with first term a = 2, common difference d = 2, and n = 20 terms."""
}

BIN = "/home/usman/Bonsai-demo/bin/cuda/llama-speculative-simple"
MODEL = "/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
DRAFTER_V2 = "/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf"

def run_test(prompt_key, prompt_text, k_val, n_predict=150):
    cmd = [
        BIN,
        "-m", MODEL,
        "-md", DRAFTER_V2,
        "--spec-type", "draft-dspark",
        "--spec-draft-n-max", str(k_val),
        "-ngl", "99",
        "-ngld", "999",
        "-fa", "on",
        "-c", "4096",
        "-n", str(n_predict),
        "--temp", "0",
        "-p", prompt_text
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + "\n" + res.stderr
    
    # Parse metrics
    # accept = 75.714%
    acc_match = re.search(r"accept\s*=\s*([\d\.]+)%", out)
    n_drafted_match = re.search(r"n_drafted\s*=\s*(\d+)", out)
    n_accept_match = re.search(r"n_accept\s*=\s*(\d+)", out)
    n_predict_match = re.search(r"n_predict\s*=\s*(\d+)", out)
    speed_match = re.search(r"decoded\s+(\d+)\s+tokens in\s+([\d\.]+)\s+seconds,\s+speed:\s+([\d\.]+)\s+t/s", out)
    
    acc_pct = float(acc_match.group(1)) if acc_match else 0.0
    n_drafted = int(n_drafted_match.group(1)) if n_drafted_match else 0
    n_accept = int(n_accept_match.group(1)) if n_accept_match else 0
    n_pred = int(n_predict_match.group(1)) if n_predict_match else 0
    speed_toks = float(speed_match.group(3)) if speed_match else 0.0
    wall_sec = float(speed_match.group(2)) if speed_match else 0.0
    
    tokens_per_step = (n_accept / (n_drafted / k_val)) + 1.0 if (n_drafted and k_val) else 1.0

    return {
        "prompt": prompt_key,
        "K": k_val,
        "n_predict": n_pred,
        "n_drafted": n_drafted,
        "n_accept": n_accept,
        "acceptance_rate": f"{acc_pct:.2f}%",
        "tokens_per_step": f"{tokens_per_step:.2f}",
        "raw_speed_toks_sec": speed_toks,
        "wall_time_sec": wall_sec
    }

print("Running DSpark v2 Speculative Benchmarks on NVIDIA GB10...")
print("=" * 80)

results = []
for p_key, p_text in PROMPTS.items():
    for k in [4, 5, 6, 7]:
        res = run_test(p_key, p_text, k)
        print(f"Prompt: {p_key:<24} | K={k} | Accept: {res['acceptance_rate']:<7} ({res['n_accept']}/{res['n_drafted']}) | Tok/Step: {res['tokens_per_step']:<5} | Speed: {res['raw_speed_toks_sec']} t/s")
        results.append(res)

with open("/home/usman/spark-splash/dspark_v2_bench_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nBenchmark results saved to dspark_v2_bench_results.json")
