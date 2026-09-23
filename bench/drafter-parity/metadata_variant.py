#!/usr/bin/env python3
"""Create a draft-only YaRN metadata variant. Keep all tensor bytes unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_manifest(reader):
    return [{'name': t.name, 'shape': t.shape.tolist(), 'type': int(t.tensor_type),
             'bytes': t.n_bytes, 'sha256': hashlib.sha256(memoryview(t.data).cast('B')).hexdigest()}
            for t in reader.tensors]


def convert(source, destination, gguf):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('The destination already exists.')
    reader = gguf.GGUFReader(source)
    architecture = reader.fields['general.architecture'].contents()
    if architecture != 'dflash':
        raise ValueError('The source must use the dflash architecture.')
    if reader.fields['dflash.rope.freq_base'].contents() != 10000000:
        raise ValueError('The source RoPE frequency base differs from the trainer.')
    values = {
        'dflash.rope.scaling.type': ('yarn', gguf.GGUFValueType.STRING),
        'dflash.rope.scaling.factor': (32.0, gguf.GGUFValueType.FLOAT32),
        'dflash.rope.scaling.original_context_length': (8192, gguf.GGUFValueType.UINT32),
        # ggml adds the YaRN magnitude. A second magnitude here would double it.
        'dflash.rope.scaling.attn_factor': (1.0, gguf.GGUFValueType.FLOAT32),
    }
    before = tensor_manifest(reader)
    writer = gguf.GGUFWriter(destination, architecture, endianess=reader.endianess)
    for key, field in reader.fields.items():
        if key.startswith('GGUF.') or key == 'general.architecture' or key in values:
            continue
        subtype = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), field.types[0], sub_type=subtype)
    for key, (value, value_type) in values.items():
        writer.add_key_value(key, value, value_type)
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    result = gguf.GGUFReader(destination)
    after = tensor_manifest(result)
    if before != after:
        raise RuntimeError('The tensor payload verification failed.')
    metadata_before = {k: (v.types, v.contents()) for k, v in reader.fields.items() if not k.startswith('GGUF.')}
    metadata_after = {k: (v.types, v.contents()) for k, v in result.fields.items() if not k.startswith('GGUF.')}
    differences = {k for k in metadata_before.keys() | metadata_after.keys()
                   if metadata_before.get(k) != metadata_after.get(k)}
    if not differences <= values.keys():
        raise RuntimeError(f'Unexpected metadata changes: {differences - values.keys()}')
    for key, (value, value_type) in values.items():
        if metadata_after[key] != ([value_type], value):
            raise RuntimeError(f'The metadata verification failed: {key}')
    return {'source': str(source.resolve()), 'destination': str(destination.resolve()),
            'source_sha256': file_hash(source), 'destination_sha256': file_hash(destination),
            'tensor_payloads_identical': True, 'tensors': before,
            'metadata_changes': {k: v[0] for k, v in values.items()},
            'converter_sha256': file_hash(__file__),
            'scope': 'Draft metadata only. No target changes. No acceptance claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf-py', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    if args.manifest.exists():
        raise ValueError('The manifest already exists.')
    sys.path.insert(0, str(args.gguf_py))
    import gguf
    result = convert(args.source, args.destination, gguf)
    args.manifest.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}, indent=2))


if __name__ == '__main__':
    main()
