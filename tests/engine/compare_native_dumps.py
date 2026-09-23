"""Require byte-identical tensors from two native execution modes."""
import argparse
import json
from pathlib import Path


def equal_files(first, second):
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open('rb') as a, second.open('rb') as b:
        while True:
            left, right = a.read(1024*1024), b.read(1024*1024)
            if left != right:
                return False
            if not left:
                return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline-subset', action='store_true', help='Compare only the tensors present in an older baseline')
    args = parser.parse_args()
    manifests = [(root/'inputs.txt').read_text().strip() for root in (args.baseline, args.candidate)]
    if manifests[0] != manifests[1] or not manifests[0]:
        raise ValueError('Input manifests differ or are empty')
    tokens = [int(token) for token in manifests[0].split(',')]
    expected_steps = {f'token-{i}' for i in range(len(tokens))}
    for root in (args.baseline, args.candidate):
        if {p.name for p in root.glob('token-*')} != expected_steps:
            raise ValueError('Token directories differ from the manifest')
        for i, token in enumerate(tokens):
            if int((root/f'token-{i}'/'token.txt').read_text()) != token:
                raise ValueError('Token record differs from the manifest')
    names = [{p.relative_to(root) for p in root.glob('token-*/*.f32')} for root in (args.baseline, args.candidate)]
    if not names[0] or (not args.baseline_subset and names[0] != names[1]) or not names[0].issubset(names[1]):
        raise ValueError('Tensor file sets differ')
    if not args.baseline_subset:
        expected_tensors = {'logits.f32'}
        expected_tensors |= {f'layer-{i}.f32' for i in range(64)}
        for i in range(64):
            expected_tensors |= ({f'key-{i}.f32', f'value-{i}.f32'} if i % 4 == 3 else
                                 {f'state-{i}.f32', f'conv-{i}.f32'})
        for step in expected_steps:
            if {p.name for p in names[0] if p.parent.name == step} != expected_tensors:
                raise ValueError('Incomplete native state snapshot')
    mismatches = [str(name) for name in sorted(names[0]) if not equal_files(args.baseline/name, args.candidate/name)]
    result = {'passed': not mismatches, 'tokens': len(tokens), 'compared_files': len(names[0]),
              'baseline_subset': args.baseline_subset, 'mismatches': mismatches}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return int(bool(mismatches))


if __name__ == '__main__':
    raise SystemExit(main())
