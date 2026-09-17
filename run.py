"""Train IGRCA or evaluate the bundled selected-seed checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['evaluate', 'train'])
    parser.add_argument('--q', nargs='+', type=int, choices=[1, 3], default=[1, 3])
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--task', choices=['rca', 'gc', 'both'], default='both')
    parser.add_argument('--check-reference', action='store_true')
    parser.add_argument('--epochs', type=int, help='Optional training-only budget override.')
    parser.add_argument('--stage1-epochs', type=int, help='Optional training-only Stage 1 override.')
    args = parser.parse_args()
    if args.action == 'evaluate' and (args.epochs is not None or args.stage1_epochs is not None):
        parser.error('Training budgets cannot be changed for checkpoint evaluation.')
    if args.check_reference and (args.action != 'evaluate' or args.task == 'gc'):
        parser.error('--check-reference requires evaluation including RCA.')
    checkpoint_root = args.checkpoints.resolve()
    if args.check_reference and checkpoint_root != (ROOT / 'checkpoints').resolve():
        parser.error('Archived references apply only to the bundled checkpoints.')

    import numpy as np
    import torch
    from generate import check_data
    from models.igrca import IGRCA

    config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu if needed.')
    output.mkdir(parents=True)
    report = {'action': args.action, 'data_seed': config['data_seed'],
              'optimization_seed': config['optimization_seed'],
              'seed_selection': config['seed_selection'], 'results': {}}
    for q in args.q:
        name = f'q{q}'
        with np.load(args.data.resolve() / f'{name}.npz', allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        condition = config['conditions'][name]
        check_data(data, condition['data_hashes'])
        seed = config['optimization_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        settings = dict(config['model'])
        settings.update(device=device, data_name=name)
        if args.epochs is not None:
            settings['epochs'] = args.epochs
        if args.stage1_epochs is not None:
            settings['stage1_epochs'] = args.stage1_epochs
        if not 0 < settings['stage1_epochs'] < settings['epochs']:
            raise ValueError('Require 0 < stage1_epochs < epochs.')
        destination = output / name
        destination.mkdir()
        print(f'{args.action}: IGRCA / {name}', flush=True)
        with tempfile.TemporaryDirectory(prefix='igrca_core_') as temporary:
            old_cwd = Path.cwd()
            try:
                os.chdir(temporary)
                model = IGRCA(**settings).to(device)
                model.save_dir = str(Path(temporary) / 'saved_models')
                model.model_name = 'model'
                if args.action == 'train':
                    observations = np.concatenate([data['train'], data['validation']], axis=0)
                    with (destination / 'training.log').open('w', encoding='utf-8') as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            model._training(observations)
                    shutil.copy2(Path(model.save_dir) / 'model.pt', destination / 'model.pt')
                    np.savez_compressed(destination / 'residual_stats.npz', **{
                        key: np.load(Path(model.save_dir) / f'model_{key}.npy', allow_pickle=False)
                        for key in ['us_mean_encoder', 'us_std_encoder']
                    })
                    report['results'][name] = {'training_finished': True,
                        'epochs_budget': settings['epochs'], 'stage1_epochs': settings['stage1_epochs']}
                else:
                    compact = checkpoint_root / name
                    if checkpoint_root == (ROOT / 'checkpoints').resolve():
                        for filename, expected in condition['checkpoint_hashes'].items():
                            if file_hash(compact / filename) != expected:
                                raise RuntimeError(f'Checkpoint hash mismatch: {name}/{filename}')
                    shutil.copy2(compact / 'model.pt', Path(model.save_dir) / 'model.pt')
                    with np.load(compact / 'residual_stats.npz', allow_pickle=False) as stats:
                        for key in stats.files:
                            np.save(Path(model.save_dir) / f'model_{key}.npy', stats[key])
                    metrics = {}
                    original_log = model._log_and_print

                    def capture(message, *values):
                        match = re.match(r'(Root cause analysis|Causal discovery) ([^:]+):', message)
                        if match and values:
                            section = 'rca' if match.group(1) == 'Root cause analysis' else 'gc'
                            metric = match.group(2)
                            metrics.setdefault(section, {})[metric] = float(np.asarray(values[0]).item())
                            if len(values) > 1:
                                metrics[section][metric + '_sequence_std'] = float(np.asarray(values[1]).item())
                        original_log(message, *values)

                    model._log_and_print = capture
                    with (destination / 'evaluation.log').open('w', encoding='utf-8') as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            if args.task in ['gc', 'both']:
                                model._testing_causal_discover(data['test_normal'], data['graph'])
                            if args.task in ['rca', 'both']:
                                model._testing_root_cause(data['test_abnormal'], data['labels'])
                    if args.task in ['rca', 'both']:
                        if set(metrics.get('rca', {})) != set(condition['reference_rca']):
                            raise RuntimeError('Incomplete RCA metric output.')
                        if args.check_reference:
                            error = max(abs(metrics['rca'][k] - v)
                                        for k, v in condition['reference_rca'].items())
                            metrics['max_reference_error'] = error
                            if error > 1e-7:
                                raise AssertionError(f'Archived RCA mismatch: {error}')
                        print(json.dumps(metrics['rca']), flush=True)
                    report['results'][name] = metrics
            finally:
                os.chdir(old_cwd)
                if 'model' in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        (output / 'results.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Finished: {output / "results.json"}')


if __name__ == '__main__':
    main()
