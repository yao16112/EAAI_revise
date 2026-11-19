"""Bounded RNG and eight-worker DataLoader replay check; no model training."""
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tool import setup_seed


class Probe(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return index, random.random(), float(np.random.rand()), float(torch.rand(()))


def sample(seed):
    setup_seed(seed)
    direct = [random.random(), float(np.random.rand()), float(torch.rand(())), float(torch.rand((), device='cuda'))]
    batches = [torch.stack(batch, dim=1) for batch in DataLoader(Probe(), batch_size=2, shuffle=True, num_workers=8)]
    return direct, torch.cat(batches)


if __name__ == '__main__':
    results = {}
    previous = []
    for seed in [111,222,333,444,555]:
        a, x = sample(seed)
        b, y = sample(seed)
        assert a == b and torch.equal(x,y)
        assert a not in previous
        previous.append(a)
        results[seed] = {'python_numpy_torch_cpu_cuda_replay': True, 'eight_worker_shuffle_rng_replay': True}
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    print(json.dumps({'status':'PASS','seeds':results,'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled()},indent=2))
