"""Fresh VisA Seed222 preflight and first-step witness; reuse existing best-only entry."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path('/opt/data/private/VarAD_EAAI');sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
plan=json.loads((OUT/'launch_plan.json').read_text())
started=time.time()


def main():
    from tool.experiment import digest_file,write_json
    assert plan['arguments']['seed']==222 and plan['arguments']['dataset']=='visa'
    assert not plan['arguments']['resume'] and not plan['arguments']['ckt_path']
    assert not (Path(plan['previous_run'])/'artifacts/VarAD/models').exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==plan['gpu_uuid']
    for name,sha in plan['code_sha256'].items():
        if digest_file(ROOT/name)!=sha:raise ValueError('Frozen core changed: '+name)
    if digest_file(plan['inventory'])!=plan['inventory_sha256']:raise ValueError('Inventory changed')
    if digest_file(Path(plan['dataset_root'])/'meta.json')!=plan['meta_sha256']:raise ValueError('VisA meta changed')
    count=0
    for line in Path(plan['inventory']).read_text().splitlines():
        row=json.loads(line)
        if row['dataset']!='visa':continue
        if digest_file(Path(plan['dataset_root'])/row['path'])!=row['sha256']:raise ValueError('VisA file changed: '+row['path'])
        count+=1
        if count%1000==0:print(f'PRECHECK {count} VisA files verified',flush=True)
    write_json(OUT/'precheck.json',{'status':'PASS','files':count})
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    from scripts.train_reference_best import run
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.9,0);torch.cuda.reset_peak_memory_stats()
    observed=False
    def first_step(optimizer,args,kwargs):
        nonlocal observed
        if observed:return
        observed=True
        steps=[int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
        assert min(steps)==max(steps)==1
        torch.cuda.synchronize()
        record={'status':'FIRST_STEP_VERIFIED','dataset':'visa','category':'candle','seed':222,
          'physical_gpu':0,'batch_size':8,'image_size':1022,'lr':optimizer.param_groups[0]['lr'],
          'source_checkpoint':None,'optimizer_step':1,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'elapsed_seconds':time.time()-started}
        write_json(OUT/'first_step.json',record);print('FIRST_STEP '+json.dumps(record),flush=True)
    hook=register_optimizer_step_post_hook(first_step)
    try:return run(OUT/'launch_plan.json')
    finally:hook.remove()


if __name__=='__main__':
    code=1;error=None
    try:code=main()
    except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)};traceback.print_exc()
    finally:
        with (OUT/'launcher_exit.json').open('x') as stream:json.dump({'exit_code':code,'error':error,'elapsed_seconds':time.time()-started},stream,indent=2)
    raise SystemExit(code)
