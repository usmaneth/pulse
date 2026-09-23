#!/usr/bin/env python3
"""Summarize paired draft measurements without a quality claim."""
import argparse
import json
from pathlib import Path
import statistics


def distribution(values):
    return {'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.directory/'results.jsonl').read_text().splitlines()]
    result = {'scope': 'Fixed 256-token output prefixes. Task quality is not scored.', 'requests': len(rows), 'contexts': {}}
    for context in sorted({row['context'] for row in rows}):
        group = [r for r in rows if r['context'] == context]
        variants = {}
        for variant in ['k0', 'original', 'yarn']:
            selected = [r for r in group if r['variant'] == variant]
            if not selected:
                continue
            variants[variant] = {'repeats': len(selected), 'wall_s': distribution([r['elapsed_s'] for r in selected]),
                'decode_tps': distribution([r['response']['timings']['predicted_per_second'] for r in selected]),
                'prefill_s': distribution([r['response']['timings']['prompt_ms']/1000 for r in selected]),
                'decode_s': distribution([r['response']['timings']['predicted_ms']/1000 for r in selected]),
                'exact_matches_k0': sum(r['output_matches_k0'] for r in selected),
                'output_limit_stops': sum(r['stopped_at_output_limit'] for r in selected)}
            if variant != 'k0':
                variants[variant]['acceptance'] = distribution([r['acceptance'] for r in selected])
        pairs = []
        for repeat in sorted({r['repeat'] for r in group}):
            matched = {r['variant']: r for r in group if r['repeat'] == repeat}
            if len(matched) != 3:
                continue
            k0, original, yarn = (matched[k] for k in ['k0', 'original', 'yarn'])
            pairs.append({'repeat': repeat, 'original_over_yarn_wall': original['elapsed_s']/yarn['elapsed_s'],
                          'k0_over_yarn_wall': k0['elapsed_s']/yarn['elapsed_s'],
                          'yarn_over_original_decode_tps': yarn['response']['timings']['predicted_per_second']/original['response']['timings']['predicted_per_second'],
                          'acceptance_delta': yarn['acceptance']-original['acceptance']})
        result['contexts'][str(context)] = {'variants': variants, 'paired_ratios': pairs}
    (args.directory/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
