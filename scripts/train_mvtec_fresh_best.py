"""Fresh MVTec seed, reusing the confirmed reference-best training/category path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    result=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def validate_plan(plan):
    args=plan['arguments']
    if args['dataset']!='mvtec' or args['resume'] or args['ckt_path']:
        raise ValueError('This entry only starts fresh MVTec training, without checkpoint reuse')
    if (Path(plan['previous_run'])/'artifacts/VarAD/models').exists():
        raise ValueError('Fresh seed must not inherit an existing run')
    if args['seed']==111:raise ValueError('User requested a different seed from the prior Seed111')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=plan['gpu_uuid']:
        raise ValueError('CUDA mapping does not match the allocated GPU')


def run(plan_path):
    plan=json.loads(Path(plan_path).read_text());validate_plan(plan)
    output=Path(plan['output']);output.mkdir(exist_ok=False)
    from tool.experiment import write_json
    write_json(output/'plan.json',plan)
    write_json(output/'process.json',{'pid':os.getpid(),'physical_gpu':2,'CUDA_VISIBLE_DEVICES':plan['gpu_uuid'],
        'cwd':str(ROOT),'python':sys.executable,'orchestrator_sha256':digest(__file__)})
    started=time.time();code=0;error=None;completed={}
    try:
        for name,expected in plan['code_sha256'].items():
            if digest(ROOT/name)!=expected:raise ValueError('Frozen core code changed: '+name)
        if digest(plan['inventory'])!=plan['inventory_sha256']:raise ValueError('Frozen inventory changed')
        count=0
        for line in Path(plan['inventory']).read_text().splitlines():
            row=json.loads(line)
            if row['dataset']!='mvtec':continue
            if digest(Path(plan['dataset_root'])/row['path'])!=row['sha256']:
                raise ValueError('Frozen MVTec file changed: '+row['path'])
            count+=1
            if count%1000==0:print(f'PRECHECK {count} MVTec files verified',flush=True)
        write_json(output/'precheck.json',{'status':'PASS','files':count})
        import torch
        from torch.optim.optimizer import register_optimizer_step_post_hook
        from main import MVTEC_CATEGORIES
        from scripts.train_reference_best import run_category
        assert torch.cuda.device_count()==1
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.90,0)
        torch.cuda.reset_peak_memory_stats()
        observed=False
        def first_step(optimizer,args,kwargs):
            nonlocal observed
            if observed:return
            observed=True
            steps=[int(state['step'].item()) for state in optimizer.state.values() if 'step' in state]
            assert min(steps)==max(steps)==1,'Fresh seed unexpectedly resumed an optimizer'
            torch.cuda.synchronize()
            record={'status':'FIRST_STEP_VERIFIED','dataset':'mvtec','category':'bottle',
              'seed':plan['arguments']['seed'],'physical_gpu':2,'batch_size':plan['arguments']['batch_size'],
              'image_size':plan['arguments']['image_size'],'lr':optimizer.param_groups[0]['lr'],
              'source_checkpoint':None,'optimizer_step':1,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'elapsed_seconds':time.time()-started}
            write_json(output/'first_step.json',record);print('FIRST_STEP '+json.dumps(record),flush=True)
        hook=register_optimizer_step_post_hook(first_step)
        for category in MVTEC_CATEGORIES:completed[category]=run_category(plan,category)
        hook.remove()
        import statistics
        macro={k:statistics.mean(m[k] for m in completed.values()) for k in next(iter(completed.values()))}
        write_json(output/'RESULTS.json',{'dataset':'mvtec','seed':plan['arguments']['seed'],
          'per_category':completed,'macro_average':macro,'selection':'test_selected_best',
          'paper_protocol_verified':False,'comparable_to_old_seed111_final_epoch_protocol':False})
    except BaseException as exc:
        code=1;error={'type':type(exc).__name__,'message':str(exc)};traceback.print_exc()
    finally:write_json(output/'exit_status.json',{'exit_code':code,'error':error,'elapsed_seconds':time.time()-started,
                                              'completed_categories':list(completed)})
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    raise SystemExit(run(parser.parse_args().plan))
