"""Bounded execution evidence only; never a formal training or metric claim."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/opt/data/private/VarAD')
sys.path = [str(ROOT)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(OLD)]

def protect_source(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(OLD):
            raise PermissionError('Smoke execution must not read old repository: ' + str(p))

sys.addaudithook(protect_source)

def run(output):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from config import PATHS
    from dataset import get_dataset
    from main import build_trainer
    from tool import setup_seed

    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'evidence_level': 'L1_SMOKE_ONLY', 'formal_training_started': False,
              'source_access_blocked': str(OLD), 'stages': [], 'paths': PATHS}
    started = time.time()
    def stage(name, **details):
        result['stages'].append({'stage': name, **details})
        (output / 'smoke_result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result['stages'][-1]), flush=True)
    try:
        args_dict = json.loads((ROOT / 'configs/observed.json').read_text())
        args = SimpleNamespace(**args_dict, ckt_path='')
        assert torch.cuda.is_available(), 'A CUDA device is required for this unchanged VMamba path'
        device = 'cuda:0'
        torch.cuda.reset_peak_memory_stats()
        result['configuration'] = args_dict
        result['environment'] = {'python': sys.version, 'executable': sys.executable,
                                 'torch': torch.__version__, 'cuda': torch.version.cuda,
                                 'cudnn': torch.backends.cudnn.version(),
                                 'gpu': torch.cuda.get_device_name(0)}
        result['environment']['packages'] = {}
        for package in ['torchvision','numpy','scipy','timm','einops','fvcore','imgaug','scikit-learn','Pillow']:
            result['environment']['packages'][package] = importlib.metadata.version(package)
        stage('imports')
        setup_seed(args.seed)
        trainer = build_trainer(args, device)
        stage('model_initialization', feature_blocks=[4,8,12], input_hw=[args.image_size]*2)
        trainer.model.train()
        dummy = torch.randn(args.batch_size,3,args.image_size,args.image_size,device=device)
        outputs = trainer.model(dummy)
        assert len(outputs) == 3 and len(outputs[0]) == 3
        stage('dummy_forward', feature_shapes=[list(v.shape) for v in outputs[0]])
        loss = trainer.model.cal_loss(outputs)
        assert torch.isfinite(loss).item()
        stage('loss', value=float(loss.detach()))
        trainer.optimizer.zero_grad()
        loss.backward()
        gradients = [p.grad for p in trainer.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all().item() for g in gradients)
        assert all(p.grad is None for p in trainer.model.tokenizer.frozen_tokenizer.parameters())
        stage('backward', tensors_with_grad=len(gradients))
        probe = trainer.model.tokenizer.proj[0][0].weight
        before = probe.detach().clone()
        trainer.optimizer.step()
        assert not torch.equal(probe, before)
        stage('single_training_iteration', optimizer='AdamW', batch_size=args.batch_size)
        del outputs, loss, dummy, before, gradients, probe
        _, dataset, _ = get_dataset('mvtec', trainer.data_transform, trainer.target_transform,
                                     False, 'FS', 0, synthesis_anomalies=False,
                                     class_name_list=[['bottle']])
        normal = next(i for i,r in enumerate(dataset.data_all) if r['anomaly'] == 0)
        anomaly = next(i for i,r in enumerate(dataset.data_all) if r['anomaly'] == 1)
        sample = dataset[anomaly]
        image = sample['img'].unsqueeze(0).to(device)
        trainer.model.eval()
        with torch.no_grad():
            reference = trainer.model.cal_am(trainer.model(image))[0]
        assert reference.shape == (args.image_size,args.image_size) and np.isfinite(reference).all()
        checkpoint = output / 'smoke_only.pth'
        trainer.save(str(checkpoint))
        bos = {n: m.BOS.detach().cpu().clone() for n,m in trainer.model.named_modules() if hasattr(m,'BOS')}
        result['checkpoint_limitation'] = 'BOS is persisted; legacy checkpoints without BOS are rejected; no full training resume.'
        del trainer, image
        torch.cuda.empty_cache()
        setup_seed(args.seed + 1)
        fresh = build_trainer(args, device)
        state = torch.load(checkpoint,map_location=device,weights_only=True)
        fresh.load(str(checkpoint))
        fresh.load_state_dict(state,strict=True)
        assert all(torch.equal(bos[n], m.BOS.detach().cpu()) for n,m in fresh.model.named_modules() if n in bos)
        fresh.model.eval()
        with torch.no_grad():
            actual = fresh.model.cal_am(fresh.model(sample['img'].unsqueeze(0).to(device)))[0]
        np.testing.assert_allclose(actual,reference,rtol=1e-5,atol=1e-7)
        stage('checkpoint_save_load', strict_registered_state=True, seed_replayed=False,
              maximum_map_difference=float(np.max(np.abs(actual-reference))))
        np.save(output/'real_anomaly_map.npy',actual)
        (output/'real_image.json').write_text(json.dumps({'image':sample['img_path'],
            'label':int(sample['anomaly']), 'image_score':float(np.max(actual)),
            'map':'real_anomaly_map.npy','map_hw':list(actual.shape)},indent=2))
        stage('real_image_inference', shape=list(actual.shape), finite=bool(np.isfinite(actual).all()))
        tiny = DataLoader(Subset(dataset,[normal,anomaly]),batch_size=1,shuffle=False,num_workers=0)
        tiny_metrics = fresh.evaluation(tiny,['bottle'],False,cal_pro=False)
        assert all(np.isfinite(v) for v in tiny_metrics['bottle'].values())
        (output/'tiny_evaluation_smoke_only.json').write_text(json.dumps({'evidence_level':'SMOKE_ONLY_NOT_BENCHMARK',
            'sample_count':2,'maxF1':'Oracle maxF1','metrics':tiny_metrics},indent=2))
        stage('evaluation_entry_path', image_count=2, oracle_maxF1=True)
        project_modules = {}
        for name,module in sys.modules.items():
            location = getattr(module,'__file__',None)
            if location:
                p=Path(location).resolve()
                assert not p.is_relative_to(OLD), (name,str(p))
                if p.is_relative_to(ROOT): project_modules[name]=str(p.relative_to(ROOT))
        result['local_imports']=project_modules
        result['status']='PASS'
        result['elapsed_seconds']=time.time()-started
        result['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        stage('runtime_independence', old_repository_modules=0)
        return 0
    except Exception:
        result['status']='FAIL'
        result['error']=traceback.format_exc()
        result['elapsed_seconds']=time.time()-started
        (output/'smoke_result.json').write_text(json.dumps(result,indent=2))
        raise

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    cli=parser.parse_args()
    raise SystemExit(run(cli.output.resolve()))
