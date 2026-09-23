#!/usr/bin/env python3
"""Compare the actual trainer with the actual ggml CPU RoPE operator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

POSITIONS = [0, 511, 4095, 8191, 32767, 65535]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer', required=True, type=Path)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('dspark_training_reference', args.trainer)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module.ModelConfig()
    assert cfg.head_dim == 128 and cfg.rope_yarn and cfg.yarn_factor == 32
    rotary = module.YarnRotaryEmbedding(cfg)
    positions = torch.tensor([POSITIONS], dtype=torch.int64)
    cos, sin = rotary(positions)
    vector = torch.tensor([((i*37)%101-50)/32 for i in range(128)], dtype=torch.float32)
    queries = vector.repeat(len(POSITIONS), 1).reshape(1, 1, len(POSITIONS), 128)
    reference = module._apply_rope(queries, cos, sin).numpy().reshape(len(POSITIONS), 128)
    rows = []
    for variant in ['original', 'yarn']:
        raw = subprocess.run([str(args.probe), variant], capture_output=True, check=True).stdout
        measured = np.frombuffer(raw, dtype='<f4').reshape(len(POSITIONS), 128)
        for index, position in enumerate(POSITIONS):
            diff = measured[index].astype(np.float64)-reference[index]
            rows.append({'variant': variant, 'position': position,
                         'max_abs_error': float(np.max(np.abs(diff))),
                         'rms_error': float(np.sqrt(np.mean(diff**2))),
                         'max_reference_magnitude': float(np.max(np.abs(reference[index])))})
    result = {'positions': POSITIONS, 'trainer_sha256': hashlib.sha256(args.trainer.read_bytes()).hexdigest(),
              'probe_sha256': hashlib.sha256(args.probe.read_bytes()).hexdigest(),
              'attention_mscale': rotary.attention_scaling,
              'head_dim': cfg.head_dim, 'factor': cfg.yarn_factor, 'original_context': cfg.yarn_orig_max_pos,
              'comparisons': rows, 'scope': 'CPU operator and float32 trainer; not full draft or CUDA acceptance'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
