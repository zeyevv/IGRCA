"""Generate the two NITS anchor datasets using the original simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from datasets.nonlinear import Nonlinear

ROOT = Path(__file__).resolve().parent


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = f'{value.dtype}|{",".join(map(str, value.shape))}|'.encode('ascii')
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def generate(q, config):
    options = dict(config['dataset'])
    options.update(seed=config['data_seed'], num_envs=q + 1, data_dir='unused')
    source = Nonlinear(options)
    source.generate_example()
    data = source.data_dict
    normal = np.asarray(data['x_n_list'], dtype=np.float32)
    return {
        'train': normal[:8],
        'validation': normal[8:10],
        'test_normal': normal[10:],
        'test_abnormal': np.asarray(data['x_ab_list'][10:], dtype=np.float32),
        'labels': np.asarray(data['label_list'][10:], dtype=np.int8),
        'graph': np.asarray(data['causal_struct'], dtype=np.int8),
    }


def check_data(data, expected):
    for key, digest in expected.items():
        if array_hash(data[key]) != digest:
            raise RuntimeError(f'Dataset mismatch for {key}; use the recorded software versions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q', nargs='+', type=int, choices=[1, 3], default=[1, 3])
    parser.add_argument('--output', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    output = args.output.resolve()
    for q in args.q:
        if (output / f'q{q}.npz').exists():
            raise FileExistsError(f'Refusing to overwrite {output / f"q{q}.npz"}')
    output.mkdir(parents=True, exist_ok=True)
    for q in args.q:
        data = generate(q, config)
        check_data(data, config['conditions'][f'q{q}']['data_hashes'])
        np.savez_compressed(output / f'q{q}.npz', **data)
        print(f'q={q}: generated data match the archived arrays.')


if __name__ == '__main__':
    main()
