#!/usr/bin/env python3
"""Build a draft-vocabulary corpus from agent output that Usman actually receives.

Sources:
  - Codex session logs: assistant text, apply_patch input, shell commands.
  - Claude Code transcripts: assistant text, Edit/Write/MultiEdit content, Bash commands.
  - Tracked source files of the local product repositories.

The script writes train.jsonl and heldout.jsonl ({"text": ...} per line).
Every 10th session file goes to heldout, so coverage is measured on unseen sessions.
"""
import glob, json, os, subprocess

HOME = os.path.expanduser("~")
OUT = os.path.dirname(os.path.abspath(__file__))
# Repositories under $HOME to include, comma separated, for example "pulse,other".
REPOS = [r for r in os.environ.get("CORPUS_REPOS", "pulse").split(",") if r]
CODE_EXT = (".ts", ".tsx", ".js", ".jsx", ".py", ".rs", ".go", ".sh", ".md", ".json",
            ".toml", ".yaml", ".yml", ".css", ".scss", ".sql", ".swift", ".kt", ".c", ".h", ".cpp")
MAX_FILE = 200_000


def codex_texts(path):
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        p = d.get("payload") or d
        t = p.get("type")
        if t == "message" and p.get("role") == "assistant":
            for c in p.get("content", []):
                if c.get("text"):
                    yield c["text"]
        elif t in ("function_call", "custom_tool_call"):
            yield p.get("input") or p.get("arguments") or ""


def claude_texts(path):
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") != "assistant":
            continue
        for c in (d.get("message") or {}).get("content", []) or []:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text":
                yield c.get("text", "")
            elif c.get("type") == "tool_use":
                inp = c.get("input") or {}
                for k in ("new_string", "content", "command"):
                    if isinstance(inp.get(k), str):
                        yield inp[k]
                for e in inp.get("edits", []) or []:
                    if isinstance(e, dict) and isinstance(e.get("new_string"), str):
                        yield e["new_string"]


def repo_texts(repo):
    root = os.path.join(HOME, repo)
    try:
        files = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True,
                               text=True, timeout=60).stdout.split()
    except Exception:
        return
    for f in files:
        if not f.endswith(CODE_EXT) or "lock" in f or f.endswith(".min.js"):
            continue
        p = os.path.join(root, f)
        try:
            if os.path.getsize(p) <= MAX_FILE:
                yield open(p, errors="replace").read()
        except OSError:
            continue


def main():
    sessions = sorted(glob.glob(f"{HOME}/.codex/sessions/**/*.jsonl", recursive=True))
    transcripts = sorted(glob.glob(f"{HOME}/.claude/projects/**/*.jsonl", recursive=True))
    stats = {"train": 0, "heldout": 0}
    with open(f"{OUT}/train.jsonl", "w") as tr, open(f"{OUT}/heldout.jsonl", "w") as ho:
        for i, path in enumerate(sessions + transcripts):
            fn = codex_texts if path in sessions else claude_texts
            dst, key = (ho, "heldout") if i % 10 == 0 else (tr, "train")
            for t in fn(path):
                if t and t.strip():
                    dst.write(json.dumps({"text": t}) + "\n")
                    stats[key] += len(t)
        for repo in REPOS:
            for t in repo_texts(repo):
                tr.write(json.dumps({"text": t}) + "\n")
                stats["train"] += len(t)
    print({k: f"{v / 1e6:.1f} MB" for k, v in stats.items()})


if __name__ == "__main__":
    main()
