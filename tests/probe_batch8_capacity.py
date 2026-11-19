"""One FP32 batch-8 optimizer step, only on an idle GPU; never reduce batch size."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(args):
    if args.image_size % 14:
        raise ValueError('Input size must be an explicit multiple of 14; no implicit alignment')
    args.output.mkdir(parents=True,exist_ok=False)
    query=subprocess.run(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.used,memory.free,memory.total',
                          '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    used,free,total=[int(v.strip()) for v in query.stdout.strip().split(',')]
    result={'batch_size':8,'learning_rate':0.0005,'image_size':args.image_size,'gpu':args.gpu,
            'gpu_before_mib':{'used':used,'free':free,'total':total},'training_steps':0,
            'precision':'FP32','batch_size_fallback':False,'python':sys.executable}
    if used>256:
        result.update(status='NOT_RUN_GPU_BUSY',reason='Protect existing GPU jobs; no CUDA context was initialized',
                      capacity_verified=False,oom_observed=False)
        code=4
    else:
        import torch
        from main import build_trainer
        from tool import setup_seed
        setup_seed(111)
        config=json.loads((ROOT/'configs/observed.json').read_text())
        config.update(batch_size=8,learning_rate=0.0005,image_size=args.image_size)
        started=time.time()
        try:
            torch.cuda.set_device(args.gpu)
            torch.cuda.reset_peak_memory_stats()
            trainer=build_trainer(SimpleNamespace(**config,ckt_path=''),f'cuda:{args.gpu}')
            trainer.model.train()
            image=torch.randn(8,3,args.image_size,args.image_size,device=f'cuda:{args.gpu}')
            loss=trainer.model.cal_loss(trainer.model(image))
            assert torch.isfinite(loss).item()
            trainer.optimizer.zero_grad()
            loss.backward()
            assert all(torch.isfinite(p.grad).all().item() for p in trainer.parameters() if p.grad is not None)
            trainer.optimizer.step()
            torch.cuda.synchronize()
            result.update(status='PASS_ONE_STEP_ONLY',capacity_verified=True,oom_observed=False,
                          training_steps=1,loss=float(loss.detach()))
            code=0
        except Exception as exc:
            result.update(status='OOM' if isinstance(exc,torch.OutOfMemoryError) else 'ERROR',
                          capacity_verified=False,oom_observed=isinstance(exc,torch.OutOfMemoryError),
                          error_type=type(exc).__name__,error=str(exc))
            code=1
        result.update(elapsed_seconds=time.time()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    with (args.output/'capacity.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--image-size',type=int,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
