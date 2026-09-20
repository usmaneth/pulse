"""
Real benchmarking harness for a local llama.cpp OpenAI-compatible HTTP server.

Every reported number is measured from a live HTTP response and nothing is estimated or simulated.
All metrics trace directly to parsed response fields: prompt_n, prompt_ms, prompt_per_second,
predicted_n, predicted_ms, predicted_per_second, draft_n, draft_n_accepted.
Derived values (acceptance, steps, tokens_per_step, step_ms) are computed from these
measured fields using the --k parameter only for step derivation.
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_K = 5
DEFAULT_URL = "http://127.0.0.1:8085"
CHAT_ENDPOINT = "/v1/chat/completions"


# Hardcoded prompts for decode mode: two per category (code, math, prose)
DECODE_PROMPTS = {
    "code": [
        "Implement binary search over a sorted array of integers in Python. Return the index of the target if found, otherwise return -1. Include type hints and handle edge cases.",
        "Write a function that merges a list of overlapping intervals. Each interval is a tuple (start, end). Return a list of merged non-overlapping intervals sorted by start time.",
    ],
    "math": [
        "A train leaves Station A at 60 mph heading toward Station B, 300 miles away. Another train leaves Station B at 80 mph heading toward Station A at the same time. A fly starts at Station A flying at 100 mph, touching each train and instantly reversing direction. How far does the fly travel before the trains collide?",
        "Calculate the sum of all prime numbers below 10000 that are also palindromes when written in base 10. Show your reasoning step by step.",
    ],
    "prose": [
        "Explain the concept of entropy in thermodynamics and information theory. How do the two definitions relate to each other? Give concrete examples.",
        "Describe how a modern operating system scheduler works. Cover preemption, priority inversion, CPU affinity, and the tradeoffs between throughput and latency.",
    ],
}


# Realistic source code block for prefill mode (roughly 250 tokens at 4 chars/token)
PREFILL_CODE_BLOCK = '''def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge a list of overlapping intervals.
    
    Args:
        intervals: List of (start, end) tuples where start <= end
        
    Returns:
        List of merged non-overlapping intervals sorted by start time
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        
        if current_start <= last_end:
            # Overlapping intervals, merge them
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # Non-overlapping, add to result
            merged.append((current_start, current_end))
    
    return merged


def binary_search(arr: List[int], target: int) -> int:
    """
    Binary search over a sorted array.
    
    Args:
        arr: Sorted list of integers
        target: Value to search for
        
    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


class IntervalTree:
    """Interval tree for efficient overlap queries."""
    
    def __init__(self, intervals: List[Tuple[int, int]]):
        self.root = self._build(intervals)
    
    def _build(self, intervals: List[Tuple[int, int]]) -> Optional[Node]:
        if not intervals:
            return None
        mid = len(intervals) // 2
        node = Node(intervals[mid])
        node.left = self._build(intervals[:mid])
        node.right = self._build(intervals[mid + 1:])
        return node
    
    def query(self, start: int, end: int) -> List[Tuple[int, int]]:
        """Find all intervals overlapping with [start, end]."""
        result = []
        self._query_recursive(self.root, start, end, result)
        return result
    
    def _query_recursive(self, node: Optional[Node], start: int, end: int, result: List[Tuple[int, int]]):
        if not node:
            return
        if node.interval[1] < start:
            self._query_recursive(node.right, start, end, result)
        elif node.interval[0] > end:
            self._query_recursive(node.left, start, end, result)
        else:
            result.append(node.interval)
            self._query_recursive(node.left, start, end, result)
            self._query_recursive(node.right, start, end, result)


class Node:
    def __init__(self, interval: Tuple[int, int]):
        self.interval = interval
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
'''


def build_request(prompt: str, max_tokens: int, temperature: float = 0.0, stream: bool = False) -> Dict[str, Any]:
    """Build a chat completion request payload."""
    return {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }


def post_request(url: str, payload: Dict[str, Any], timeout: float = 300.0) -> Tuple[Dict[str, Any], float]:
    """
    Send POST request to the server and return (response_json, wall_clock_ms).
    Raises on connection error or non-2xx response.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            wall_ms = (time.perf_counter() - start) * 1000.0
            body = resp.read().decode("utf-8")
            return json.loads(body), wall_ms
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}") from e


def extract_timings(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the timings object from the response. Raises if missing."""
    if "timings" not in response:
        raise ValueError("Response missing 'timings' object")
    timings = response["timings"]
    required = ["prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second", "draft_n", "draft_n_accepted"]
    for field in required:
        if field not in timings:
            raise ValueError(f"Timings missing required field: {field}")
    return timings


def derive_metrics(timings: Dict[str, Any], k: int) -> Dict[str, Any]:
    """Compute derived metrics from raw timings."""
    draft_n = timings["draft_n"]
    draft_n_accepted = timings["draft_n_accepted"]
    predicted_n = timings["predicted_n"]
    predicted_ms = timings["predicted_ms"]
    
    acceptance = draft_n_accepted / draft_n if draft_n > 0 else None
    steps = draft_n / k if draft_n > 0 else None
    tokens_per_step = predicted_n / steps if steps and steps > 0 else None
    step_ms = predicted_ms / steps if steps and steps > 0 else None
    
    return {
        "acceptance": acceptance,
        "steps": steps,
        "tokens_per_step": tokens_per_step,
        "step_ms": step_ms,
    }


def run_decode_mode(url: str, k: int) -> Dict[str, Any]:
    """Run decode benchmark: six prompts, two per category."""
    print("\n=== DECODE MODE ===")
    results = {
        "mode": "decode",
        "prompts": [],
        "aggregates": {},
    }
    
    category_stats = {"code": [], "math": [], "prose": []}
    
    for category, prompts in DECODE_PROMPTS.items():
        print(f"\nCategory: {category}")
        for i, prompt in enumerate(prompts):
            print(f"  Prompt {i+1}/2...", end=" ", flush=True)
            payload = build_request(prompt, max_tokens=200, temperature=0.0, stream=False)
            try:
                response, wall_ms = post_request(url + CHAT_ENDPOINT, payload)
                timings = extract_timings(response)
                derived = derive_metrics(timings, k)
                
                record = {
                    "category": category,
                    "prompt_index": i,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "wall_ms": wall_ms,
                    "timings": timings,
                    "derived": derived,
                }
                results["prompts"].append(record)
                category_stats[category].append({
                    "predicted_per_second": timings["predicted_per_second"],
                    "acceptance": derived["acceptance"],
                })
                print(f"OK (pred={timings['predicted_per_second']:.1f} tok/s, acc={derived['acceptance']:.3f})")
            except Exception as e:
                print(f"FAILED: {e}")
                record = {
                    "category": category,
                    "prompt_index": i,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "error": str(e),
                }
                results["prompts"].append(record)
    
    # Compute aggregates per category
    for category, stats in category_stats.items():
        if not stats:
            results["aggregates"][category] = {"predicted_per_second_median": None, "predicted_per_second_mean": None, "acceptance_median": None, "acceptance_mean": None, "count": 0}
            continue
        pred_rates = [s["predicted_per_second"] for s in stats]
        acceptances = [s["acceptance"] for s in stats if s["acceptance"] is not None]
        results["aggregates"][category] = {
            "predicted_per_second_median": statistics.median(pred_rates) if pred_rates else None,
            "predicted_per_second_mean": statistics.mean(pred_rates) if pred_rates else None,
            "acceptance_median": statistics.median(acceptances) if acceptances else None,
            "acceptance_mean": statistics.mean(acceptances) if acceptances else None,
            "count": len(stats),
        }
        print(f"  {category}: pred_median={results['aggregates'][category]['predicted_per_second_median']:.1f}, pred_mean={results['aggregates'][category]['predicted_per_second_mean']:.1f}, acc_median={results['aggregates'][category]['acceptance_median']:.3f}, acc_mean={results['aggregates'][category]['acceptance_mean']:.3f}")
    
    return results


def run_prefill_mode(url: str) -> Dict[str, Any]:
    """Run prefill benchmark: synthetic prompts of ~1000, 4000, 8000, 16000 tokens."""
    print("\n=== PREFILL MODE ===")
    results = {
        "mode": "prefill",
        "sizes": [],
    }
    
    # Target token counts (approximate, 4 chars per token)
    target_tokens = [1000, 4000, 8000, 16000]
    block = PREFILL_CODE_BLOCK
    block_chars = len(block)
    block_tokens_est = block_chars / 4.0
    
    for target in target_tokens:
        repeats = max(1, int(target / block_tokens_est))
        prompt = block * repeats
        actual_chars = len(prompt)
        actual_tokens_est = actual_chars / 4.0
        
        print(f"  Target ~{target} tokens (repeats={repeats}, est={actual_tokens_est:.0f})...", end=" ", flush=True)
        payload = build_request(prompt, max_tokens=1, temperature=0.0, stream=False)
        try:
            response, wall_ms = post_request(url + CHAT_ENDPOINT, payload)
            timings = extract_timings(response)
            
            record = {
                "target_tokens": target,
                "repeats": repeats,
                "prompt_chars": actual_chars,
                "estimated_prompt_tokens": actual_tokens_est,
                "wall_ms": wall_ms,
                "timings": timings,
            }
            results["sizes"].append(record)
            print(f"OK (prompt_n={timings['prompt_n']}, prompt_per_sec={timings['prompt_per_second']:.1f}, wall_ms={wall_ms:.1f})")
        except Exception as e:
            print(f"FAILED: {e}")
            record = {
                "target_tokens": target,
                "repeats": repeats,
                "prompt_chars": actual_chars,
                "estimated_prompt_tokens": actual_tokens_est,
                "error": str(e),
            }
            results["sizes"].append(record)
    
    # Print table
    print("\n  Prompt Size vs Input Tokens/Second:")
    print(f"  {'Target Tokens':>14} | {'Measured prompt_n':>18} | {'prompt_per_second':>18} | {'Wall ms':>10}")
    print(f"  {'-'*14}-+-{'-'*18}-+-{'-'*18}-+-{'-'*10}")
    for r in results["sizes"]:
        if "timings" in r:
            t = r["timings"]
            print(f"  {r['target_tokens']:>14} | {t['prompt_n']:>18} | {t['prompt_per_second']:>18.1f} | {r['wall_ms']:>10.1f}")
        else:
            print(f"  {r['target_tokens']:>14} | {'FAILED':>18} | {'FAILED':>18} | {'FAILED':>10}")
    
    return results


def run_concurrency_mode(url: str, k: int) -> Dict[str, Any]:
    """Run concurrency benchmark: N simultaneous requests for N in 1,2,4,8,16."""
    print("\n=== CONCURRENCY MODE ===")
    results = {
        "mode": "concurrency",
        "levels": [],
    }
    
    concurrency_levels = [1, 2, 4, 8, 16]
    prompt = "Explain how a hash table works with separate chaining. Include time complexity for insert, lookup, and delete operations."
    
    for n in concurrency_levels:
        print(f"  Concurrency N={n}...", end=" ", flush=True)
        payload = build_request(prompt, max_tokens=200, temperature=0.0, stream=False)
        
        start_wall = time.perf_counter()
        failures = 0
        per_stream_rates = []
        acceptances = []
        total_predicted_n = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(post_request, url + CHAT_ENDPOINT, payload) for _ in range(n)]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    response, _ = fut.result()
                    timings = extract_timings(response)
                    derived = derive_metrics(timings, k)
                    per_stream_rates.append(timings["predicted_per_second"])
                    total_predicted_n += timings["predicted_n"]
                    if derived["acceptance"] is not None:
                        acceptances.append(derived["acceptance"])
                except Exception as e:
                    failures += 1
                    print(f"\n    Request failed: {e}", file=sys.stderr)
        
        end_wall = time.perf_counter()
        wall_span_ms = (end_wall - start_wall) * 1000.0
        
        aggregate_throughput = total_predicted_n / (wall_span_ms / 1000.0) if wall_span_ms > 0 else 0.0
        mean_per_stream = statistics.mean(per_stream_rates) if per_stream_rates else 0.0
        mean_acceptance = statistics.mean(acceptances) if acceptances else None
        
        record = {
            "concurrency": n,
            "wall_span_ms": wall_span_ms,
            "total_predicted_n": total_predicted_n,
            "aggregate_throughput_tok_per_sec": aggregate_throughput,
            "mean_per_stream_predicted_per_second": mean_per_stream,
            "mean_acceptance": mean_acceptance,
            "failures": failures,
            "successful_requests": n - failures,
        }
        results["levels"].append(record)
        
        acc_str = f"{mean_acceptance:.3f}" if mean_acceptance else "N/A"
        print(f"OK (aggregate={aggregate_throughput:.1f} tok/s, mean_stream={mean_per_stream:.1f} tok/s, acc={acc_str}, failures={failures})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp OpenAI-compatible server")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Server base URL (default: {DEFAULT_URL})")
    parser.add_argument("--mode", choices=["decode", "prefill", "concurrency", "all"], default="all", help="Benchmark mode")
    parser.add_argument("--out", default="results.json", help="Output JSON file")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Speculative draft window size for derivations (default: {DEFAULT_K})")
    args = parser.parse_args()
    
    # Test connection first
    print(f"Connecting to {args.url}...")
    try:
        test_payload = build_request("test", max_tokens=1, temperature=0.0, stream=False)
        post_request(args.url + CHAT_ENDPOINT, test_payload, timeout=10.0)
        print("Connection OK")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {args.url}")
        print(f"       Start the llama.cpp server first, e.g.:")
        print(f"       llama-server -m model.gguf --host 127.0.0.1 --port 8085")
        print(f"       Details: {e}")
        sys.exit(1)
    
    all_results = {
        "url": args.url,
        "k": args.k,
        "timestamp": time.time(),
        "results": [],
    }
    
    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["decode", "prefill", "concurrency"]
    else:
        modes_to_run = [args.mode]
    
    for mode in modes_to_run:
        if mode == "decode":
            all_results["results"].append(run_decode_mode(args.url, args.k))
        elif mode == "prefill":
            all_results["results"].append(run_prefill_mode(args.url))
        elif mode == "concurrency":
            all_results["results"].append(run_concurrency_mode(args.url, args.k))
    
    # Write output
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()