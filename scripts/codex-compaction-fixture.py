#!/usr/bin/env python3
"""Prepare a one-compaction fixture without a model request."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
rows = [f'ARCHIVED {i:04d} quantity={i % 17 + 1} price_cents={i * 31 % 10007} obsolete=yes' for i in range(400)]
rows.append('FINAL_TOKEN: CEDAR-842')
archive = '\n'.join(rows) + '\n'
(args.output / 'records.txt').write_text(archive)
prompt = ('Use exec_command exactly once to run cat records.txt with max_output_tokens=12000. '
          'The archive ends with FINAL_TOKEN. After the tool result, reply with only that final token. '
          'Do not run another tool or edit any file. Do not repeat the read after a context checkpoint. '
          'Only the final token matters; the archived records are irrelevant to the answer.')
(args.output / 'prompt.txt').write_text(prompt + '\n')
manifest = {'id': 'single-compaction-archive', 'auto_compact_token_limit': 12000,
            'archive_sha256': hashlib.sha256(archive.encode()).hexdigest(), 'archive_rows': len(rows),
            'expected': {'shell_calls': 1, 'compactions': 1, 'final_answer': 'CEDAR-842', 'file_changes': 0},
            'qualification': 'Verify the initial prompt is below 12000 tokens and the archive raises the history above that limit. Verify the summary returns below the limit.'}
(args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
