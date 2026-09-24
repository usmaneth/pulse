#!/usr/bin/env python3
"""Assemble the self-distillation prompt set for the MTP head.

Each output row is one chat request for the target model. Two shapes:
  codex - the real Codex system prompt and tool list, so replies contain
          tool calls, patches and short agent prose.
  plain - a bare user turn, so replies contain free-form code and prose.

Sources: Usman's own session prompts (in-distribution), SWE-bench issues,
Magicoder OSS-Instruct and self-oss-instruct (GitHub-seeded tasks), UltraChat.
"""
import glob, json, os, random

random.seed(0)
HOME = os.path.expanduser("~")
D = "/mnt/models/distill"


def own_prompts():
    out = []
    for p in glob.glob(f"{HOME}/.codex/sessions/**/*.jsonl", recursive=True):
        for line in open(p, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            pl = d.get("payload") or {}
            if pl.get("type") == "message" and pl.get("role") == "user":
                t = "".join(c.get("text", "") for c in pl.get("content", []) if isinstance(c, dict))
                out.append(t)
    for p in glob.glob(f"{HOME}/.claude/projects/**/*.jsonl", recursive=True):
        for line in open(p, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "user" and isinstance((d.get("message") or {}).get("content"), str):
                out.append(d["message"]["content"])
    keep = []
    for t in out:
        t = t.strip()
        if not (15 < len(t) < 6000):
            continue
        if t.startswith("<") or "system-reminder" in t or "tool_result" in t or "<environment_context>" in t:
            continue
        keep.append(t)
    return list(dict.fromkeys(keep))


def main():
    open_sets = json.load(open(f"{D}/open_prompts.json"))
    own = own_prompts()
    magic = open_sets["ise-uiuc/Magicoder-OSS-Instruct-75K"]
    soi = open_sets["bigcode/self-oss-instruct-sc2-exec-filter-50k"]
    swe = open_sets["princeton-nlp/SWE-bench"]
    chat = open_sets["HuggingFaceH4/ultrachat_200k"]
    rows = []
    for t in own:
        rows.append({"src": "own", "shape": "codex", "prompt": t})
    for t in swe:
        rows.append({"src": "swe", "shape": "codex", "prompt": t})
    for t in magic[:4000]:
        rows.append({"src": "magic", "shape": "codex", "prompt": t})
    for t in magic[4000:]:
        rows.append({"src": "magic", "shape": "plain", "prompt": t})
    for t in soi:
        rows.append({"src": "soi", "shape": "plain", "prompt": t})
    for t in chat:
        rows.append({"src": "chat", "shape": "plain", "prompt": t})
    random.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = i
        r["thinking"] = (i % 2 == 0)
    with open(f"{D}/prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(rows), Counter(r["src"] for r in rows), Counter(r["shape"] for r in rows))


if __name__ == "__main__":
    main()
