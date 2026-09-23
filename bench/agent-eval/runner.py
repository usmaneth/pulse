#!/usr/bin/env python3
"""Run one model at a time. Require an exclusive lease for live requests."""
import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request
from code_fixture import PROMPT as CODE_PROMPT, validate as validate_code

TOOLS = [{"type": "function", "function": {"name": "read_record", "description": "Read the private inventory record.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"], "additionalProperties": False}}}]
SYSTEM = "Use read_record to fetch the requested record. Compute quantity * unit_price_cents. Reply with only the integer total. Ignore unrelated archive records."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def command(argv, payload):
    proc = subprocess.run(argv, input=json.dumps(payload), text=True, capture_output=True, timeout=120, check=True)
    return json.loads(proc.stdout)


@contextlib.contextmanager
def exclusive(path):
    with open(path, "a+") as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def preflight(config, lease):
    if lease.get("expires_unix", 0) <= time.time():
        raise ValueError("The GPU lease expired.")
    if set(config["nodes"]) != set(lease["nodes"]):
        raise ValueError("The GPU lease must name the exact model nodes.")
    # The operator hook checks remote processes, free disk, and unified memory.
    state = command(config["resource_check"], {"nodes": config["nodes"], "lease": lease})
    if state.get("exclusive") is not True:
        raise ValueError("The resource check did not confirm exclusive GPU access.")
    for node in config["nodes"]:
        observed = state["nodes"][node]
        if observed["mem_available_gib"] < config["min_mem_available_gib"]:
            raise ValueError("Insufficient unified memory headroom on " + node)
        if observed["disk_free_gib"] < config["min_disk_free_gib"]:
            raise ValueError("Insufficient free disk on " + node)
    return state


def events(lines):
    """Decode SSE data across multiple lines. Reject malformed JSON."""
    data = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                value = "\n".join(data)
                if value == "[DONE]":
                    yield {"_done": True}
                    return
                yield json.loads(value)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        raise ValueError("The SSE response ended inside an event.")


def stream(config, messages, tools):
    body = {"model": config["model"], "messages": messages, "temperature": 0,
            "seed": 42, "max_tokens": config.get("max_tokens", 512), "stream": True,
            "stream_options": {"include_usage": True}, **config.get("request_options", {})}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    headers = {"Content-Type": "application/json"}
    if config.get("api_key_env"):
        headers["Authorization"] = "Bearer " + os.environ[config["api_key_env"]]
    request = urllib.request.Request(config["url"].rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers)
    start = time.monotonic()
    stamps, visible_stamps, text, calls, usage, timings = [], [], "", {}, {}, {}
    finish, done, reasoning = None, False, ""
    with urllib.request.urlopen(request, timeout=config.get("timeout_s", 900)) as response:
        for event in events(response):
            if event.get("error"):
                raise ValueError("The server returned an error: " + str(event["error"]))
            done |= event.get("_done", False)
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                    stamps.append(time.monotonic() - start)
                if delta.get("content") or delta.get("tool_calls"):
                    visible_stamps.append(time.monotonic() - start)
                text += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                for call in delta.get("tool_calls", []):
                    target = calls.setdefault(call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    target["id"] += call.get("id") or ""
                    for key in ("name", "arguments"):
                        target["function"][key] += call.get("function", {}).get(key) or ""
                finish = choice.get("finish_reason") or finish
    elapsed = time.monotonic() - start
    if not done or not finish:
        raise ValueError("The streaming response did not finish.")
    gaps = [b-a for a, b in zip(stamps, stamps[1:])]
    n = usage.get("completion_tokens")
    accepted, drafted = timings.get("draft_n_accepted"), timings.get("draft_n")
    result = {"request_sha256": digest(body), "ttft_any_s": stamps[0] if stamps else None,
              "ttft_visible_s": visible_stamps[0] if visible_stamps else None,
              "elapsed_s": elapsed, "chunk_gap_p50_s": statistics.median(gaps) if gaps else None,
              "chunk_gap_p95_s": sorted(gaps)[max(0, int(len(gaps)*.95)-1)] if gaps else None,
              "chunk_times_s": stamps, "usage": usage, "server_timings": timings,
              "end_to_end_tokens_s": n/elapsed if n else None,
              "server_decode_tokens_s": timings.get("predicted_per_second"),
              "draft_acceptance": accepted/drafted if drafted and accepted is not None else None,
              "cached_prompt_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
              "finish_reason": finish, "content": text, "reasoning": reasoning,
              "tool_calls": list(calls.values())}
    return result


def count_tokens(config, messages, tools):
    value = command(config["token_counter"], {"model": config["model"], "messages": messages, "tools": tools})
    count = value["tokens"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("The token counter must return a positive integer.")
    return count


def archive(count):
    """Create varied source records from a fixed public seed."""
    return "\n".join(f"def archived_cost_{i}(quantity): return quantity * {(i*17)%991+1} + {(i*31)%173}  # archived SKU {i:06d}" for i in range(count))


def fixture(config, size, nonce):
    # The counter must apply the model template and include tool definitions.
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ""}]
    pad = max(1, size // 35)
    previous = None
    for _ in range(12):
        base[1]["content"] = "Request " + nonce + "\nArchive: " + archive(pad) + "\nFetch record sku-73 and compute its total."
        actual = count_tokens(config, base, TOOLS)
        if config.get("fixture_mode", "equal_tokens") == "fixed_bytes" or size <= actual <= size + 64:
            return base, actual
        slope = (actual-previous[1])/(pad-previous[0]) if previous and pad != previous[0] else 35
        slope = max(slope, .1)
        previous = (pad, actual)
        pad = max(1, pad + round((size + 32 - actual) / slope))
    raise ValueError("The fixture could not reach the context token range.")


def validate_call(result):
    calls = result["tool_calls"]
    if result["finish_reason"] != "tool_calls" or len(calls) != 1:
        return False
    call = calls[0]
    try:
        return bool(call["id"]) and call["function"]["name"] == "read_record" and json.loads(call["function"]["arguments"]) == {"key": "sku-73"}
    except (KeyError, ValueError, TypeError):
        return False


def answer_correct(result, expected):
    return result["finish_reason"] == "stop" and not result["tool_calls"] and result["content"].strip() == str(expected)


def task(config, messages, context_tokens):
    results = []
    try:
        return _task(config, messages, context_tokens, results)
    except Exception as exc:
        return {"success": False, "measurement_valid": False, "error": str(exc),
                "context_tokens": context_tokens, "requests": results}


def _task(config, messages, context_tokens, results):
    first = stream(config, messages, TOOLS)
    first["phase"] = "initial"
    results.append(first)
    if not validate_call(first):
        return {"success": False, "reason": "incorrect_tool_call", "requests": results}
    # The record exists only in the tool response. JSON syntax alone cannot pass.
    quantity = 7
    unit_price = 319
    tool_message = {"role": "tool", "tool_call_id": first["tool_calls"][0]["id"], "content": json.dumps({"key": "sku-73", "quantity": quantity, "unit_price_cents": unit_price})}
    assistant = {"role": "assistant", "content": first["content"], "tool_calls": first["tool_calls"]}
    if first.get("reasoning"):
        assistant["reasoning_content"] = first["reasoning"]
    appended = messages + [assistant, tool_message]
    second = stream(config, appended, TOOLS)
    second["phase"] = "appended_tool_turn"
    results.append(second)
    # Client compaction is deterministic. It is a separate workload, not a cache hit.
    compacted = [{"role": "system", "content": "The prior tool returned sku-73: quantity 7, unit_price_cents 319. Reply with only the integer total."}, {"role": "user", "content": "Preserve that record. Add two units to the quantity and compute the new total."}]
    third = stream(config, compacted, [])
    third["phase"] = "client_compaction"
    results.append(third)
    code = stream(config, [{"role": "system", "content": "Return valid Python function code."}, {"role": "user", "content": messages[-1]["content"] + "\nNew task:\n" + CODE_PROMPT}], [])
    code["phase"] = "code_holdout"
    code["validation"] = validate_code(code["content"]) if code["finish_reason"] == "stop" else {"passed": False, "reason": "incomplete"}
    results.append(code)
    actual = first["usage"].get("prompt_tokens")
    verified = actual is not None and abs(actual-context_tokens) <= 64
    return {"success": verified and answer_correct(second, quantity*unit_price) and answer_correct(third, (quantity+2)*unit_price) and code["validation"]["passed"] is True,
            "code_evaluated": code["validation"].get("evaluated", True),
            "measurement_valid": verified,
            "context_tokens": context_tokens, "server_prompt_tokens": actual,
            "context_verified": verified, "requests": results}


def run(config, lease, output):
    for field in ("model_sha256", "engine_commit", "template_sha256"):
        if not config.get(field) or config[field] == "REQUIRED":
            raise ValueError("Set the provenance field: " + field)
    with exclusive(lease["lock_path"]):
        initial_state = preflight(config, lease)
        output.mkdir(parents=True, exist_ok=False)
        (output / "provenance.json").write_text(json.dumps({"config": config, "lease": lease, "resources": initial_state}, indent=2))
        with (output / "results.jsonl").open("w") as report:
            for size in config.get("contexts", [8192, 32768, 65536]):
                for concurrency in config.get("concurrency", [1, 2, 4]):
                    for repeat in range(config.get("repeats", 3)):
                        preflight(config, lease)
                        if size + config.get("max_tokens", 512) + 512 > config["max_context_per_request"]:
                            raise ValueError("The configured context cannot hold the tool turn.")
                        fixtures = [fixture(config, size, digest([config.get("fixture_seed", 42), size, concurrency, repeat, i])[:32]) for i in range(concurrency)]
                        if any(count + config.get("max_tokens", 512) + 512 > config["max_context_per_request"] for _, count in fixtures):
                            raise ValueError("The actual fixture tokens exceed the context reserve.")
                        manifest_path = output / f"fixtures-{size}-{concurrency}-{repeat}.json"
                        manifest_path.write_text(json.dumps({"mode": config.get("fixture_mode", "equal_tokens"), "fixtures": fixtures}, indent=2))
                        for cache_mode in ("cold", "primed_cache"):
                            priming = []
                            if cache_mode == "cold":
                                # The hook must clear all slots. A random suffix alone is not cold.
                                cleared = command(config["reset_cache"], {"model": config["model"]})
                                if cleared.get("cleared") is not True:
                                    raise ValueError("The cache reset did not succeed.")
                            else:
                                # Prime only the initial requests, after all prior task phases.
                                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                                    priming = list(pool.map(lambda fixture: stream(config, fixture[0], TOOLS), fixtures))
                            start = time.monotonic()
                            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                                futures = [pool.submit(task, config, *item) for item in fixtures]
                                results = []
                                for future in futures:
                                    try:
                                        results.append(future.result())
                                    except Exception as exc:
                                        results.append({"success": False, "error": str(exc), "requests": []})
                            wall = time.monotonic()-start
                            tokens = [r["usage"].get("completion_tokens") for t in results for r in t["requests"]]
                            row = {"context_target": size, "fixture_mode": config.get("fixture_mode", "equal_tokens"), "concurrency": concurrency, "repeat": repeat,
                                   "cache_mode": cache_mode, "cache_verified": cache_mode == "primed_cache" and all((t["requests"][0].get("cached_prompt_tokens") or 0) >= t.get("context_tokens", size)-128 for t in results if t["requests"]) and all(t["requests"] for t in results), "wall_s": wall, "priming_requests": priming, "tasks": results,
                                   "aggregate_tokens_s": sum(tokens)/wall if tokens and all(t is not None for t in tokens) else None,
                                   "successes": sum(t["success"] for t in results)}
                            report.write(json.dumps(row)+"\n")
                            report.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.run:
        print(json.dumps({"model": config["model"], "contexts": config.get("contexts", [8192, 32768, 65536]), "concurrency": config.get("concurrency", [1, 2, 4]), "live": False}, indent=2))
        return
    if not args.lease or not args.output:
        parser.error("--run requires --lease and --output")
    run(config, json.loads(args.lease.read_text()), args.output)


if __name__ == "__main__":
    main()
