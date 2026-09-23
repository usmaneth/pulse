#!/usr/bin/env python3
"""Provide llama.cpp template, token, and cache operations."""
import argparse
import json
import os
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["count", "reset"])
    parser.add_argument("--url", required=True, help="The server root URL without /v1.")
    parser.add_argument("--api-key-env")
    args = parser.parse_args()
    body = json.load(sys.stdin)
    headers = {"Content-Type": "application/json"}
    if args.api_key_env:
        headers["Authorization"] = "Bearer " + os.environ[args.api_key_env]

    def request(path, payload=None):
        req = urllib.request.Request(args.url.rstrip("/")+path, None if payload is None else json.dumps(payload).encode(), headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)

    if args.operation == "count":
        prompt = request("/apply-template", {"messages": body["messages"], "tools": body.get("tools", []), "add_generation_prompt": True})["prompt"]
        result = request("/tokenize", {"content": prompt, "add_special": False})
        print(json.dumps({"tokens": len(result["tokens"])}))
    else:
        slots = request("/slots")
        if not slots or any(slot.get("is_processing") for slot in slots):
            raise ValueError("The cache reset requires idle slots.")
        for slot in slots:
            request("/slots/" + str(slot["id"]) + "?action=erase", {})
        print(json.dumps({"cleared": True, "slots": len(slots)}))


if __name__ == "__main__":
    main()
