"""This authorized run: validate inputs, protect shared GPU memory, record actual progress."""
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT=Path('/opt/data/private/VarAD_EAAI')
OUTPUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
started=time.time()


def write(name,data):
    with (OUTPUT/name).open('x') as stream:json.dump(data,stream,indent=2)


def digest(path):
    import hashlib
    result=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


code=0
error=None
try:
    protocol=json.loads((OUTPUT/'protocol_snapshot.json').read_text())
    scope=protocol['execution_scope']
    assert protocol['execution_authorized'] and scope['dataset']=='visa' and scope['seed']==111
    assert not scope['batch_fallback_allowed']
    print('PRECHECK: frozen code and VisA image/mask bytes',flush=True)
    for name,expected in protocol['code_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen code changed: '+name)
    inventory=ROOT/'experiments/statistical_reliability/official_protocol_v2_confirmed_hparams/dataset_files.jsonl'
    if digest(inventory)!=protocol['dataset_inventory_sha256']:raise ValueError('Frozen inventory changed')
    if digest(Path(protocol['dataset_roots']['visa'])/'meta.json')!=protocol['visa_meta_sha256']:
        raise ValueError('VisA meta changed')
    count=0
    for line in inventory.read_text().splitlines():
        row=json.loads(line)
        if row['dataset']!='visa':continue
        if digest(Path(protocol['dataset_roots']['mvtec'])/row['path'])!=row['sha256']:
            raise ValueError('Frozen data changed: '+row['path'])
        count+=1
        if count%1000==0:print(f'PRECHECK: {count} VisA files verified',flush=True)
    write('precheck.json',{'status':'PASS','verified_visa_files':count,
                           'code_files':len(protocol['code_sha256']),'elapsed_seconds':time.time()-started})
    import torch
    gpu=protocol['training']['gpu']
    torch.cuda.set_device(gpu)
    free,total=torch.cuda.mem_get_info(gpu)
    fraction=scope['allocation_cap_fraction']
    if free < total*fraction + 4*1024**3:
        raise RuntimeError('Insufficient free GPU memory for shared-GPU cap; no other job was stopped')
    torch.cuda.set_per_process_memory_fraction(fraction,gpu)
    torch.cuda.reset_peak_memory_stats(gpu)
    write('resource_policy.json',{'gpu':gpu,'free_before_bytes':free,'total_bytes':total,
        'process_allocation_cap_fraction':fraction,'process_allocation_cap_bytes':int(total*fraction),
        'purpose':'Exclusive GPU1; fail instead of lowering batch'})
    from torch.optim.optimizer import register_optimizer_step_post_hook
    first_step_done=False
    def record_first_step(optimizer,args,kwargs):
        global first_step_done
        if first_step_done:return
        first_step_done=True
        torch.cuda.synchronize(gpu)
        record={'status':'FIRST_OPTIMIZER_STEP_COMPLETED','dataset':'visa','category':'candle',
            'seed':111,'batch_size':8,'image_size':1022,'token_grid':[73,73],
            'learning_rate':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.time()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(gpu),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(gpu),
            'precision':'FP32','batch_fallback':False}
        write('first_optimizer_step.json',record)
        print('FIRST_OPTIMIZER_STEP: '+json.dumps(record),flush=True)
    hook=register_optimizer_step_post_hook(record_first_step)
    config={**protocol['training'],'dataset':'visa','seed':111,
            'category':['all'],'save_path':str(OUTPUT/'artifacts')}
    argv=[str(ROOT/'main.py')]
    for name,value in config.items():
        argv.append('--'+name)
        if isinstance(value,list):argv.extend(str(v) for v in value)
        else:argv.append(str(value).lower() if isinstance(value,bool) else str(value))
    write('training_command.json',{'argv':[sys.executable,'-u',*argv],'executed_via':'runpy in capped process',
                                 'cwd':str(ROOT),'pid':os.getpid(),'python':sys.executable})
    print('TRAINING: '+json.dumps(argv),flush=True)
    import tool.experiment as experiment
    # Output-only adaptation; evaluator receives exporter=None and retains the same metric arrays.
    experiment.PredictionExport=lambda *args,**kwargs: None
    assert experiment.PredictionExport(None) is None
    write('output_policy.json',{'save_fig':False,'per_image_predictions':False,'anomaly_maps':False,
                              'metrics':True,'resume_checkpoints':True})
    sys.argv=argv
    runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
    hook.remove()
except BaseException as exc:
    code=int(exc.code or 0) if isinstance(exc,SystemExit) and isinstance(exc.code,(int,type(None))) else 1
    if code:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        traceback.print_exc()
finally:
    result={'exit_code':code,'elapsed_seconds':time.time()-started,'error':error,
            'paper_protocol_verified':False,'batch_fallback':False}
    if 'torch' in globals() and torch.cuda.is_initialized():
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write('exit_status.json',result)
raise SystemExit(code)
