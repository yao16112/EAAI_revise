"""Complete the remaining Seed111 categories on three isolated GPUs; preserve old results."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

OUT=Path(__file__).resolve().parent
PLAN=json.loads((OUT/'allocation.json').read_text())
ROOT=Path(PLAN['project'])
started=time.time()


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2)


def digest(path):
    import hashlib
    result=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def train_one(category):
    import torch
    assert torch.cuda.device_count()==1,'Each category must see exactly one GPU'
    group=next(g for g in PLAN['groups'] if category in g['categories'])
    assert os.environ['CUDA_VISIBLE_DEVICES']==group['uuid']
    work=OUT/category
    work.mkdir(exist_ok=False)
    config={**PLAN['training'],'dataset':'mvtec','category':[category],
            'seed':111,'gpu':0,'save_path':str(work/'artifacts')}
    if category=='tile':config['resume']=PLAN['tile_resume']
    argv=[str(ROOT/'main.py')]
    for key,value in config.items():
        argv.append('--'+key)
        if isinstance(value,list):argv.extend(map(str,value))
        else:argv.append(str(value).lower() if isinstance(value,bool) else str(value))
    write(work/'command.json',{'argv':[sys.executable,'-u',*argv],'executed_via':'runpy',
          'cwd':str(ROOT),'pid':os.getpid(),'physical_gpu':group['gpu'],
          'CUDA_VISIBLE_DEVICES':group['uuid'],'logical_gpu':0})
    from torch.optim.optimizer import register_optimizer_step_post_hook
    observed=False
    def step_observer(optimizer,args,kwargs):
        nonlocal observed
        if observed:return
        observed=True
        steps=[int(value['step'].item()) for value in optimizer.state.values() if 'step' in value]
        torch.cuda.synchronize()
        expected=436 if category=='tile' else 1
        assert min(steps)==max(steps)==expected,(category,steps)
        record={'category':category,'seed':111,'batch_size':8,'image_size':1022,
                'lr':optimizer.param_groups[0]['lr'],'physical_gpu':group['gpu'],
                'optimizer_step':expected,'resume_completed_epochs':15 if category=='tile' else 0,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.time()-started}
        write(work/'first_step.json',record)
        print('FIRST_STEP '+json.dumps(record),flush=True)
    handle=register_optimizer_step_post_hook(step_observer)
    code=0;error=None
    try:
        sys.argv=argv
        runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
    except BaseException as exc:
        code=int(exc.code or 0) if isinstance(exc,SystemExit) and isinstance(exc.code,(int,type(None))) else 1
        if code:
            error={'type':type(exc).__name__,'message':str(exc)}
            traceback.print_exc()
    finally:
        handle.remove()
        write(work/'exit_status.json',{'exit_code':code,'error':error,'elapsed_seconds':time.time()-started})
    return code


def worker(gpu):
    group=next(g for g in PLAN['groups'] if g['gpu']==gpu)
    assert os.environ['CUDA_VISIBLE_DEVICES']==group['uuid']
    for category in group['categories']:
        log=OUT/(category+'.log')
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--category',category]
        with log.open('x') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            write(OUT/(category+'_process.json'),{'pid':process.pid,'physical_gpu':gpu,
                       'CUDA_VISIBLE_DEVICES':group['uuid'],'command':command,'log':str(log)})
            print(f'GPU{gpu} category={category} pid={process.pid}',flush=True)
            code=process.wait()
        if code:return code
    return 0


def consolidate():
    import statistics
    index=json.loads((OUT/'seed111_collection.json').read_text())
    per_category={};count=0
    for category,path in index['predictions'].items():
        path=Path(path)
        manifest=json.loads((path/'manifest.json').read_text())
        assert manifest['complete'] and manifest['seed']==111
        assert len(list(path.glob('*.npz')))==manifest['count']
        per_category[category]=manifest['metrics'][category]
        count+=manifest['count']
    assert len(per_category)==15 and count==1725
    macro={k:statistics.mean(m[k] for m in per_category.values()) for k in next(iter(per_category.values()))}
    result={'status':'COMPLETE_SEED111_ONLY','seed':111,'categories':15,'test_images':count,
            'per_category':per_category,'macro_average':macro,
            'aggregation':'equal category mean','paper_protocol_verified':False,
            'collection':index['predictions'],'five_seed_statistics':False}
    write(OUT/'SEED111_COMPLETE.json',result)
    keys=['auroc_im','f1_im','ap_im','auroc_px','f1_px','ap_px']
    lines=['# MVTec AD Seed111 results','', '1022 / Batch8 / LR5e-4 / fixed final epoch60. PAPER REPRODUCTION UNVERIFIED.',
           '', '| Category | I-AUROC | I-maxF1 | I-AP | P-AUROC | P-maxF1 | P-AP |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for c,m in per_category.items():lines.append('| '+c+' | '+' | '.join(f'{m[k]:.2f}' for k in keys)+' |')
    lines.append('| Macro mean | '+' | '.join(f'{macro[k]:.2f}' for k in keys)+' |')
    lines+=['','Complete15-category single-seed result. No mean±SD across seeds or bootstrap performed.']
    with (OUT/'RESULTS.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')


def dispatch():
    for name,expected in PLAN['code_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen code changed: '+name)
    assert digest(PLAN['inventory'])==PLAN['inventory_sha256']
    remaining={c for g in PLAN['groups'] for c in g['categories']}
    checked=0
    for line in Path(PLAN['inventory']).read_text().splitlines():
        row=json.loads(line)
        if row['dataset']!='mvtec' or row['category'] not in remaining:continue
        if digest(Path(PLAN['dataset_root'])/row['path'])!=row['sha256']:
            raise ValueError('Frozen data changed: '+row['path'])
        checked+=1
    write(OUT/'precheck.json',{'status':'PASS','verified_files':checked,'code_files':len(PLAN['code_sha256'])})
    processes=[]
    for group in PLAN['groups']:
        environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=group['uuid']
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--group',str(group['gpu'])]
        log=(OUT/f'gpu{group["gpu"]}.log').open('x')
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        write(OUT/f'gpu{group["gpu"]}_process.json',{'pid':process.pid,'physical_gpu':group['gpu'],
                 'CUDA_VISIBLE_DEVICES':group['uuid'],'command':command,'categories':group['categories']})
        processes.append((group['gpu'],process,log))
        print(f'LAUNCHED GPU{group["gpu"]} group pid={process.pid} {group["categories"]}',flush=True)
    codes={}
    for gpu,process,log in processes:
        code=process.wait();log.close();codes[gpu]=code
        write(OUT/f'gpu{gpu}_exit.json',{'exit_code':code})
    if any(codes.values()):return 1
    consolidate()
    return 0


if __name__=='__main__':
    if len(sys.argv)>1:
        raise SystemExit(worker(int(sys.argv[2])) if sys.argv[1]=='--group' else train_one(sys.argv[2]))
    code=1;error=None
    try:code=dispatch()
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)};traceback.print_exc()
    finally:write(OUT/'exit_status.json',{'exit_code':code,'error':error,'elapsed_seconds':time.time()-started})
    raise SystemExit(code)
