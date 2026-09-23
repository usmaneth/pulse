#!/usr/bin/env python3
"""Count a local tokenizer's rendered chat input without GPU inference."""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", help="A local tokenizer directory with its pinned chat template.")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    body = json.load(sys.stdin)
    kwargs = {"tools": body["tools"]} if body.get("tools") else {}
    tokens = tokenizer.apply_chat_template(body["messages"], tokenize=True, add_generation_prompt=True, **kwargs)
    print(json.dumps({"tokens": len(tokens)}))


if __name__ == "__main__":
    main()
