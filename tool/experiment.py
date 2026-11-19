"""Prediction provenance and epoch-boundary recovery; no model mathematics."""
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with open(path, 'x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def save_exclusive(path, state):
    """Publish a complete file atomically, refusing to replace an existing snapshot."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (state[0], torch.tensor(state[1].astype(np.int64)), *state[2:]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    name, keys, pos, gaussian, cached = state['numpy']
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from the saved run')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def run_identity(arguments, datasets):
    """Freeze scientific settings, execution code and actual ordered data bytes."""
    operational = {'save_path', 'resume', 'stop_after_epoch', 'ckt_path', 'save_fig', 'print_freq'}
    config = {k: v for k, v in arguments.items() if k not in operational}
    paths = [ROOT / 'main.py']
    for folder in ['model', 'dataset', 'tool', 'config']:
        paths.extend(sorted((ROOT / folder).rglob('*.py')))
    code = {str(p.relative_to(ROOT)): digest_file(p) for p in paths}
    data = {}
    for split, dataset in datasets.items():
        rows = []
        for row in dataset.data_all:
            record = dict(row)
            record['image_sha256'] = digest_file(Path(dataset.root) / row['img_path'])
            if row['anomaly']:
                record['mask_sha256'] = digest_file(Path(dataset.root) / row['mask_path'])
            rows.append(record)
        data[split] = rows
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'numpy': np.__version__, 'cuda': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(),
                   'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                   'cudnn_deterministic': torch.backends.cudnn.deterministic,
                   'cudnn_benchmark': torch.backends.cudnn.benchmark,
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    return {'config': config, 'code': code, 'data': data, 'environment': environment}


def save_training(path, trainer, epoch, config, identity, scheduler=None):
    save_exclusive(path, {'format_version': 1, 'model': trainer.state_dict(),
                         'optimizer': trainer.optimizer.state_dict(),
                         'scheduler': scheduler.state_dict() if scheduler else None,
                         'completed_epochs': epoch, 'global_step': trainer.global_step,
                         'rng': rng_state(), 'configuration': dict(config),
                         'seed': config['seed'], 'identity': identity,
                         'dataloader': {'policy': 'epoch_boundary_global_torch_rng',
                                        'shuffle': True, 'persistent_workers': False}})


def load_training(path, trainer, identity, scheduler=None):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format_version') != 1:
        raise ValueError('Resume requires a complete version-1 training checkpoint')
    if state['identity'] != identity:
        differences = [k for k in identity if state['identity'].get(k) != identity[k]]
        raise ValueError('Resume identity mismatch: ' + ', '.join(differences))
    if (state['scheduler'] is None) != (scheduler is None):
        raise ValueError('Scheduler presence differs')
    trainer.load_state_dict(state['model'], strict=True)
    trainer.optimizer.load_state_dict(state['optimizer'])
    if scheduler:
        scheduler.load_state_dict(state['scheduler'])
    trainer.global_step = state['global_step']
    restore_rng(state['rng'])
    return state['completed_epochs']


class PredictionExport:
    def __init__(self, path, dataset, seed, checkpoint, data):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.dataset = dataset
        self.seed = seed
        self.checkpoint = {'path': str(Path(checkpoint).resolve()), 'sha256': digest_file(checkpoint)}
        self.root = Path(data.root).resolve()
        self.lookup = {str((self.root / r['img_path']).resolve()): r for r in data.data_all}
        if len(self.lookup) != len(data.data_all):
            raise ValueError('Duplicate image identities in test set')
        self.rows = []
        self.seen = set()

    def add(self, image_path, category, label, score, raw_map, evaluation_map, evaluation_mask):
        key = str(Path(image_path).resolve())
        if key in self.seen:
            raise ValueError('Repeated prediction: ' + key)
        row = self.lookup[key]
        if row['cls_name'] != category or int(row['anomaly']) != int(label):
            raise ValueError('Prediction/label identity mismatch: ' + key)
        if not np.isfinite(score) or not np.isfinite(raw_map).all():
            raise ValueError('Nonfinite prediction: ' + key)
        self.seen.add(key)
        filename = f'{len(self.rows):06d}.npz'
        with (self.path / filename).open('xb') as stream:
            np.savez_compressed(stream, anomaly_map=raw_map, evaluation_map=evaluation_map,
                                evaluation_mask=evaluation_mask)
        self.rows.append({'dataset': self.dataset, 'category': category,
                          'image_id': row['img_path'], 'relative_path': row['img_path'],
                          'label': int(label), 'score': float(score), 'seed': self.seed,
                          'checkpoint_id': self.checkpoint['sha256'],
                          'mask_relative_path': row['mask_path'] if label else None,
                          'normal_mask': 'all_zero' if not label else None,
                          'prediction_file': filename, 'specie_name': row['specie_name'],
                          'cluster_id': category + '/' + row['img_path']})

    def finish(self, metrics, cal_pro, by_specie):
        if self.seen != set(self.lookup):
            raise ValueError('Test predictions are incomplete')
        with (self.path / 'predictions.jsonl').open('x') as stream:
            for row in self.rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        write_json(self.path / 'manifest.json', {
            'format_version': 1, 'complete': True, 'dataset_root': str(self.root),
            'dataset': self.dataset, 'seed': self.seed, 'checkpoint': self.checkpoint,
            'count': len(self.rows), 'metrics': metrics, 'cal_pro': cal_pro, 'by_specie': by_specie,
            'score': 'max(raw anomaly_map), before evaluator resize, without per-image normalization',
            'cluster_unit': 'image; no physical part/batch grouping is available'})
