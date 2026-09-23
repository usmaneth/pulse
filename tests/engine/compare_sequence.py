"""Compare sequential native diagnostics with contiguous llama.cpp references."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np


def read_reference(path):
    payload = path.read_bytes()
    dtype, *dims = struct.unpack('<i4q', payload[:36])
    if dtype != 0 or any(d <= 0 for d in dims):
        raise ValueError(f'Unsupported reference tensor: {path}')
    expected = int(np.prod(dims)) * 4
    if len(payload) != 36 + expected:
        raise ValueError(f'Noncontiguous legacy reference: {path}; regenerate the reference')
    return np.frombuffer(payload, dtype='<f4', offset=36)


def read_raw(path):
    if path.stat().st_size % 4:
        raise ValueError(f'Invalid float payload length: {path}')
    return np.fromfile(path, dtype='<f4')


def compare(reference, native):
    if reference.shape != native.shape or not np.isfinite(reference).all() or not np.isfinite(native).all():
        raise ValueError('Shape mismatch or nonfinite values')
    a, b = reference.astype(np.float64), native.astype(np.float64)
    denominator = max(float(np.max(np.abs(a))), 1e-30)
    norm = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
    return {'max_relative_error': float(np.max(np.abs(a-b))) / denominator,
            'cosine': float(a @ b) / norm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('native', type=Path)
    parser.add_argument('--max-relative-error', type=float, default=0.05)
    parser.add_argument('--min-cosine', type=float, default=0.999)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = []
    def manifest(root):
        return [int(t) for t in (root/'inputs.txt').read_text().strip().split(',')]
    expected_tokens = manifest(args.reference)
    if expected_tokens != manifest(args.native):
        raise ValueError('Sequence manifests differ')
    directories = sorted(args.reference.glob('token-*'), key=lambda p: int(p.name.split('-')[1]))
    if not directories:
        raise ValueError('No reference tokens')
    if [p.name for p in directories] != [f'token-{i}' for i in range(len(expected_tokens))]:
        raise ValueError('Reference token directories differ from the manifest')
    if {p.name for p in directories} != {p.name for p in args.native.glob('token-*')}:
        raise ValueError('Token directory sets differ')
    for index, source in enumerate(directories):
        target = args.native / source.name
        if int((source/'token.txt').read_text()) != expected_tokens[index] or int((target/'token.txt').read_text()) != expected_tokens[index]:
            raise ValueError('Input token IDs differ from the manifest')
        tensors = [('logits', source/'logits.f32', target/'logits.f32')]
        tensors += [(f'layer-{layer}', source/f'l_out-{layer}.bin', target/f'layer-{layer}.f32')
                    for layer in range(64)]
        tensors += [(f'state-{layer}', source/f'new_state-{layer}.bin', target/f'state-{layer}.f32')
                    for layer in range(64) if layer % 4 != 3]
        for name, ref_path, native_path in tensors:
            reference = (read_raw(ref_path) if name == 'logits' else read_reference(ref_path))
            native = read_raw(native_path)
            row = {'step': source.name, 'tensor': name, **compare(reference, native)}
            row['passed'] = row['max_relative_error'] <= args.max_relative_error and row['cosine'] >= args.min_cosine
            if name == 'logits':
                row['reference_argmax'] = int(reference.argmax())
                row['native_argmax'] = int(native.argmax())
                row['passed'] &= row['reference_argmax'] == row['native_argmax']
                for label, values in [('reference', reference), ('native', native)]:
                    top = np.argsort(values)[-2:][::-1]
                    row[label + '_top2_tokens'] = top.tolist()
                    row[label + '_top2_logits'] = values[top].tolist()
                    row[label + '_top2_margin'] = float(values[top[0]] - values[top[1]])
            rows.append(row)
    result = {'passed': all(row['passed'] for row in rows), 'tokens': len(directories),
              'thresholds': {'max_relative_error': args.max_relative_error, 'min_cosine': args.min_cosine},
              'comparisons': rows}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    failures = [row for row in rows if not row['passed']]
    print(json.dumps({'passed': result['passed'], 'tokens': len(directories),
                      'comparisons': len(rows), 'failures': failures[:12]}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
