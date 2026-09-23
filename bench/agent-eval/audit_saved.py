#!/usr/bin/env python3
"""Recompute saved exactness without a model or GPU."""
import argparse
import json
from pathlib import Path


def first_diff(left, right):
    return next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                None if len(left) == len(right) else min(len(left), len(right)))


def audit(path):
    rows = json.loads((path / "outputs.json").read_text())
    outputs = {(r["config"], r["prompt_id"], r["pass"]): r for r in rows if r["slots"] == 1}
    comparisons = []
    for (name, prompt, repeat), baseline in outputs.items():
        if name != "baseline":
            continue
        drafts = [outputs[(n, prompt, repeat)] for n in ("dspark-v1", "dspark-v2")]
        values = [r["tokens"] if r["tokens"] is not None else r["content"] for r in (baseline, *drafts)]
        comparisons.append({"prompt": prompt, "pass": repeat,
                            "unit": "token" if baseline["tokens"] is not None else "character",
                            "v1_diff": first_diff(values[0], values[1]),
                            "v2_diff": first_diff(values[0], values[2]),
                            "drafts_equal": values[1] == values[2]})
    report = json.loads((path / "results.json").read_text())
    data = {(r["config"], r["prompt_id"], r["pass"]): r for r in report["results"] if r["slots"] == 1}
    template_mismatches = []
    for key, value in data.items():
        if key[0] == "baseline":
            hashes = [data[(n, *key[1:])].get("templated_sha256") for n in ("baseline", "dspark-v1", "dspark-v2")]
            if any(hashes) and len(set(hashes)) != 1:
                template_mismatches.append(key[1])
    return {"compared": len(comparisons), "v1_identical": sum(r["v1_diff"] is None for r in comparisons),
            "v2_identical": sum(r["v2_diff"] is None for r in comparisons),
            "drafts_identical": sum(r["drafts_equal"] for r in comparisons),
            "first_difference_agrees": all(r["v1_diff"] == r["v2_diff"] for r in comparisons),
            "template_mismatches": template_mismatches, "comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    print(json.dumps(audit(parser.parse_args().results), indent=2))
