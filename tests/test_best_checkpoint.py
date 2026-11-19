"""CPU checks of the requested legacy criterion, replacement, ties and load compatibility."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_best import save_best
from scripts.train_best import train_category
import main as entry
from model.VarAD.trainer import AnomalyMambaTrainer


def metric(image, pixel):
    return {'Average': {'auroc_im': image, 'auroc_px': pixel}}


with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'model_best.pth'
    model = torch.nn.Linear(1, 1)
    best = save_best(path, model, 5, metric(80, 80), {'seed': 111})
    with torch.no_grad():
        model.weight.fill_(2)
    best = save_best(path, model, 10, metric(np.float64(70), np.float64(95)), {'seed': 111}, best)
    assert best['epoch'] == 10 and best['score'] == 82.5
    for epoch, metrics in [(15, metric(65, 100)), (20, metric(70, 80))]:
        with torch.no_grad():
            model.weight.fill_(epoch)
        assert save_best(path, model, epoch, metrics, {'seed': 111}, best) is best
    state = torch.load(path, weights_only=True)
    assert state['selection']['epoch'] == 10 and state['model']['weight'].item() == 2
    assert state['selection']['uses_test_labels'] and 'optimizer' not in state
    assert list(Path(folder).iterdir()) == [path]
    fresh = torch.nn.Linear(1, 1)
    fresh.device = 'cpu'
    AnomalyMambaTrainer.load(fresh, path)
    assert fresh.weight.item() == 2 and fresh.checkpoint_seed == 111
    zero = Path(folder) / 'zero_best.pth'
    assert save_best(zero, model, 1, metric(0, 0), {'seed': 111})['score'] == 0
    try:
        save_best(path, model, 25, metric(float('nan'), 80), {'seed': 111}, best)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite score accepted')


class TinyTrainer(torch.nn.Module):
    data_transform = None
    target_transform = None

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.epoch = 0
        self.evaluations = []

    def train_epoch(self, loader):
        self.epoch += 1
        with torch.no_grad():
            self.weight.fill_(self.epoch)
        return 1 / self.epoch

    def evaluation(self, loader, categories, save_fig, **kwargs):
        assert save_fig is False and kwargs['exporter'] is None
        self.evaluations.append(self.epoch)
        values = {5: (80, 80), 10: (70, 95), 12: (60, 90)}[self.epoch]
        result = metric(*values)
        result['bottle'] = result['Average'].copy()
        return result


with tempfile.TemporaryDirectory() as folder:
    prefix = str(Path(folder) / 'bottle')
    args = SimpleNamespace(category='all', dataset='mvtec', seed=111, gpu=0, epoch=12,
                           batch_size=8, valid_freq=5)
    trainer = TinyTrainer()
    writer = SimpleNamespace(add_scalar=lambda *a: None, close=lambda: None)
    paths = ('name', folder, folder + '/metrics.csv', folder + '/train.log', prefix, writer)
    with patch.object(entry, 'setup_seed'), patch.object(entry, 'build_trainer', return_value=trainer), \
         patch.object(entry.AnomalyMambaTrainer, 'setup_paths', return_value=paths), \
         patch.object(entry, 'get_dataset', return_value=(['bottle'], [0], folder)), \
         patch('scripts.train_best.run_identity', return_value={'code': {}}), \
         patch('scripts.train_best.DataLoader', return_value=[0]), \
         patch.object(entry, 'log_metrics'), patch('torch.cuda.empty_cache'):
        result = train_category(args, 'bottle')
    assert trainer.evaluations == [5, 10, 12]
    assert result['Average']['auroc_im'] == 70 and result['Average']['auroc_px'] == 95
    assert args.category == 'all'
    checkpoints = list(Path(folder).glob('*.pth'))
    assert len(checkpoints) == 1 and checkpoints[0].name == 'bottle_best.pth'
    state = torch.load(checkpoints[0], weights_only=True)
    assert state['model']['weight'].item() == 10 and state['selection']['epoch'] == 10
print('PASS: highest mean test AUROC, strict improvement, one best file, zero score, inference loading')
print('PASS: evaluation frequency/final epoch, best metrics retained, no per-epoch/final checkpoints or exports')
