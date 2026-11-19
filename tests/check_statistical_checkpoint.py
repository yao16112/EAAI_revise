"""Bounded real-model checkpoint check; each seed runs in a fresh process, no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['XFORMERS_DISABLED'] = '1'


def worker(output, seed):
    import numpy as np
    import torch
    from main import build_trainer
    from tool import setup_seed
    from config import DATASET_ROOTS
    setup_seed(seed)
    args = json.loads((ROOT / 'configs/observed.json').read_text())
    trainer = build_trainer(SimpleNamespace(**{**args, 'seed': seed}, ckt_path=''), 'cuda:0')
    trainable = {n: p for n, p in trainer.named_parameters() if p.requires_grad}
    digest = hashlib.sha256()
    for name, p in trainable.items():
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    bos = {n: b for n, b in trainer.named_buffers() if n.endswith('.BOS')}
    assert len(bos) == 3
    assert all(not b.requires_grad for b in bos.values())
    assert all(n in trainer.state_dict() for n in bos)
    source = Path(DATASET_ROOTS['mvtec']) / 'bottle/test/broken_large/000.png'
    from PIL import Image
    image = trainer.data_transform(Image.open(source).convert('RGB')).unsqueeze(0).cuda()
    trainer.model.eval()
    checkpoint = output / 'checkpoint_l1_only.pth'
    result = {'seed': seed, 'pid': os.getpid(), 'trainable_sha256': digest.hexdigest(),
              'trainable_tensors': len(trainable), 'bos_shapes': {n: list(b.shape) for n,b in bos.items()},
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': sys.executable,
              'gpu': torch.cuda.get_device_name(), 'image': str(source)}
    if seed == 111:
        trainer.save(str(checkpoint))
    else:
        # Reproduce the old omission on the same checkpoint, then restore all state.
        state = torch.load(checkpoint, map_location='cuda:0', weights_only=True)
        legacy = {n: v for n,v in state.items() if not n.endswith('.BOS')}
        try:
            trainer.load_state_dict(legacy, strict=True)
        except RuntimeError as exc:
            assert 'BOS' in str(exc)
        else:
            raise AssertionError('Missing BOS must be rejected')
        trainer.load_state_dict(legacy, strict=False)
        with torch.no_grad():
            old_map = trainer.model.cal_am(trainer.model(image))[0]
        reference = np.load(output / 'reference.npz')
        result['legacy_max_map_difference'] = float(np.max(np.abs(old_map-reference['map'])))
        assert not np.array_equal(old_map, reference['map'])
        trainer.load(str(checkpoint))
    with torch.no_grad():
        outputs = trainer.model(image)
        arrays = {f'output_{i}_{j}': t.cpu().numpy() for i,group in enumerate(outputs) for j,t in enumerate(group)}
        arrays['map'] = trainer.model.cal_am(outputs)[0]
    assert all(np.isfinite(a).all() for a in arrays.values())
    if seed == 111:
        np.savez(output / 'reference.npz', **arrays)
        # Same-process save/reload after corrupting BOS explicitly.
        for b in bos.values():
            b.zero_()
        trainer.load(str(checkpoint))
        with torch.no_grad():
            reloaded = trainer.model(image)
            for i,group in enumerate(reloaded):
                for j,t in enumerate(group):
                    assert np.array_equal(t.cpu().numpy(), arrays[f'output_{i}_{j}'])
            assert np.array_equal(trainer.model.cal_am(reloaded)[0], arrays['map'])
        result['same_process_reload_equal'] = True
    else:
        reference = np.load(output / 'reference.npz')
        result['maximum_differences'] = {k: float(np.max(np.abs(a-reference[k]))) for k,a in arrays.items()}
        assert all(np.array_equal(a,reference[k]) for k,a in arrays.items())
        result['cross_process_elementwise_equal'] = True
    result['image_score'] = float(arrays['map'].max())
    result['status'] = 'PASS'
    (output / f'seed_{seed}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-seed', type=int)
    cli = parser.parse_args()
    if cli.worker_seed is not None:
        worker(cli.output, cli.worker_seed)
    else:
        cli.output.mkdir(parents=True, exist_ok=False)
        for seed in [111, 222, 333, 444, 555]:
            subprocess.run([sys.executable, __file__, '--output', str(cli.output), '--worker-seed', str(seed)], check=True)
        reports = [json.loads((cli.output/f'seed_{s}.json').read_text()) for s in [111,222,333,444,555]]
        assert len({r['trainable_sha256'] for r in reports}) == 5
        (cli.output/'summary.json').write_text(json.dumps({'status':'PASS', 'evidence':'L1_NO_TRAINING', 'seeds':reports},indent=2))
