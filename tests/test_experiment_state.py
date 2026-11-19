"""Cheap boundary checks: atomic save, config rejection and complete RNG/optimizer restore."""
import random
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tool.experiment import save_exclusive, save_training, load_training
from tool.utils import setup_seed


class Trainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2,2)
        self.register_buffer('BOS', torch.randn(2))
        self.optimizer = torch.optim.AdamW(self.parameters())
        self.global_step = 1


with tempfile.TemporaryDirectory() as temporary:
    p = Path(temporary) / 'state.pth'
    setup_seed(111)
    trainer = Trainer()
    trainer.layer(torch.ones(1,2)).sum().backward()
    trainer.optimizer.step()
    save_training(p, trainer, 1, {'seed':111}, {'config':{'seed':111}})
    expected = (random.random(), float(np.random.rand()), torch.rand(4))
    fresh = Trainer()
    completed = load_training(p, fresh, {'config':{'seed':111}})
    actual = (random.random(), float(np.random.rand()), torch.rand(4))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    assert completed == 1 and fresh.global_step == 1
    assert all(torch.equal(v, fresh.state_dict()[k]) for k,v in trainer.state_dict().items())
    assert fresh.optimizer.state_dict()['state']
    for function in [lambda: save_exclusive(p, {}),
                     lambda: load_training(p, fresh, {'config':{'seed':222}})]:
        try:
            function()
        except (FileExistsError, ValueError):
            pass
        else:
            raise AssertionError('Boundary must reject overwrite / changed identity')
print('PASS: atomic exclusive state, optimizer/BOS/RNG restore, identity mismatch rejection')
