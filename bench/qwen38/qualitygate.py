#!/usr/bin/env python3
"""Functional quality gate for a serving config: HumanEval pass@1 at temperature 0.

The serving stack is not bit-deterministic at temperature 0 (two runs of one
config diverge), so exact output diffs cannot gate a change. This script
measures task success instead: each HumanEval completion runs against its unit
tests in a subprocess with a timeout.

    python3 qualitygate.py --tag base --n 80 --concurrency 4
"""
import argparse, json, os, re, subprocess, tempfile, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def complete(port, model, prompt):
    body = {"model": model, "temperature": 0, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content":
                          "Complete this Python function. Output the full function in one "
                          "```python code block, no explanation.\n\n```python\n" + prompt + "```"}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def extract(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return max(m, key=len) if m else text


def run_tests(prompt, code, test, entry):
    program = prompt + "\n" + code + "\n\n" + test + f"\n\ncheck({entry})\n"
    if f"def {entry}" in code:
        program = code + "\n\n" + test + f"\n\ncheck({entry})\n"
        head = prompt.split(f"def {entry}")[0]
        program = head + "\n" + program
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(["python3", path], capture_output=True, timeout=20)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--n", type=int, default=164)
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    probs = json.load(open(f"{HERE}/humaneval.json"))[: a.n]
    lock = threading.Lock()
    results = []

    def work(p):
        try:
            code = extract(complete(a.port, a.model, p["prompt"]))
            ok = run_tests(p["prompt"], code, p["test"], p["entry_point"])
        except Exception:
            ok = False
        with lock:
            results.append((p["task_id"], ok))

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(work, probs))
    passed = sum(ok for _, ok in results)
    row = {"tag": a.tag, "humaneval_n": len(results), "passed": passed,
           "pass_at_1": round(passed / len(results), 4)}
    print(json.dumps(row))
    with open(f"{HERE}/quality.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
