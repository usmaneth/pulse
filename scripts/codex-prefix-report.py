#!/usr/bin/env python3
"""Compare rendered request prefixes without an inference request."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('trace', type=Path)
parser.add_argument('--backend', default='http://127.0.0.1:18085')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

def post(route, payload):
    request = urllib.request.Request(args.backend.rstrip('/') + route, json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line]
report = []
previous = None
for row in rows:
    payload = row['payload']
    prompt = post('/apply-template', payload)['prompt']
    tokens = post('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
    item = {'response_id': row['response_id'], 'prompt_tokens': len(tokens), 'prompt_sha256': digest(prompt),
            'tools_sha256': digest(payload.get('tools', [])), 'system_sha256': digest(payload['messages'][0]),
            'messages': len(payload['messages'])}
    if previous:
        common = next((i for i, (left, right) in enumerate(zip(previous['tokens'], tokens)) if left != right),
                      min(len(previous['tokens']), len(tokens)))
        item['previous_request_common_tokens'] = common
        item['previous_prompt_tokens'] = len(previous['tokens'])
        item['previous_prompt_is_prefix'] = common == len(previous['tokens'])
        item['tools_unchanged'] = item['tools_sha256'] == previous['item']['tools_sha256']
        item['system_unchanged'] = item['system_sha256'] == previous['item']['system_sha256']
    report.append(item)
    previous = {'tokens': tokens, 'item': item}
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
