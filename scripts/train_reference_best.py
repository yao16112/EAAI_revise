"""User-authorized reference policy: test every 5 epochs; keep one model-only best.

The old entry/core files stay frozen so existing full snapshots can be migrated
without weakening their model/data/config identity checks.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def atomic_write(path, writer):
    path=Path(path)
    fd,temp=tempfile.mkstemp(prefix='.best-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            writer(stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


class BestCheckpoint:
    def __init__(self, trainer, path, metadata):
        self.trainer=trainer;self.path=Path(path);self.metadata=metadata
        if self.path.exists():raise FileExistsError(self.path)
        self.score=-math.inf;self.metrics=None;self.epoch=None

    def consider(self, metrics, epoch):
        average=metrics['Average']
        score=(average['auroc_im']+average['auroc_px'])/2
        if not math.isfinite(score):raise ValueError('Nonfinite selection score')
        if score<=self.score:return False
        atomic_write(self.path,lambda stream:torch.save(self.trainer.state_dict(),stream))
        record={**self.metadata,'best_epoch':epoch,'selection_score':score,'metrics':metrics,
                'selection':'test_mean_image_pixel_auroc','checkpoint':str(self.path),
                'contents':'model state_dict including BOS; no optimizer or RNG'}
        atomic_write(self.path.with_suffix('.json'),lambda stream:stream.write(json.dumps(record,indent=2,allow_nan=False).encode()))
        self.score=score;self.metrics=metrics;self.epoch=epoch
        return True


def should_evaluate(epoch,total,frequency=5):
    return epoch%frequency==0 or epoch==total


def run_category(plan,category):
    from main import build_trainer
    from dataset import get_dataset
    from tool import setup_seed
    from tool.experiment import run_identity,load_training,write_json
    from torch.utils.data import DataLoader
    output=Path(plan['output'])/category
    output.mkdir(exist_ok=False)
    previous=Path(plan['previous_run'])/'artifacts/VarAD/models'
    configs=list(previous.glob('*-'+category+'_config.json'))
    cfg=json.loads(configs[0].read_text())['arguments'] if configs else dict(plan['arguments'])
    cfg.update(category=category,save_path=str(output),resume='',ckt_path='')
    args=SimpleNamespace(**cfg)
    setup_seed(args.seed)
    trainer=build_trainer(args,'cuda:0')
    classes,train,_=get_dataset(args.dataset,trainer.data_transform,trainer.target_transform,True,'FS',1e6,
                                synthesis_anomalies=False,class_name_list=[[category]])
    classes,test,_=get_dataset(args.dataset,trainer.data_transform,trainer.target_transform,False,'FS',0,
                               synthesis_anomalies=False,class_name_list=[[category]])
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,pin_memory=True,num_workers=8)
    test_loader=DataLoader(test,batch_size=1,shuffle=False,num_workers=0)
    identity=run_identity(cfg,{'train':train,'test':test})
    files=sorted(previous.glob('*-'+category+'_epoch_*.training.pth'))
    latest=files[-1] if files else None
    start=load_training(latest,trainer,identity) if latest else 0
    write_json(output/'configuration.json',{'arguments':cfg,'source_checkpoint':str(latest) if latest else None,
      'completed_epochs_at_migration':start,'global_step_at_migration':trainer.global_step,
      'policy':'test_selected_best_only','validation_frequency':args.valid_freq,
      'image_export':False,'heatmap_export':False,'paper_protocol_verified':False,
      'identity':identity,'orchestrator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    selection=BestCheckpoint(trainer,output/(category+'_best.pth'),{'dataset':args.dataset,'category':category,'seed':args.seed,
          'reference':'/opt/data/private/VarAD/main.py','selection_is_test_label_dependent':True})
    print(f'CATEGORY {category}: restored epoch={start}, step={trainer.global_step}; total={args.epoch}',flush=True)
    def evaluate(epoch,origin):
        metrics=trainer.evaluation(test_loader,classes,False,cal_pro=args.cal_pro)
        improved=selection.consider(metrics,epoch)
        with (output/'evaluation_history.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch':epoch,'origin':origin,'score':(metrics['Average']['auroc_im']+metrics['Average']['auroc_px'])/2,
                                    'best_updated':improved,'metrics':metrics},allow_nan=False)+'\n')
        print(f'EVAL {category} epoch={epoch} origin={origin} score={(metrics["Average"]["auroc_im"]+metrics["Average"]["auroc_px"])/2:.6f} best_epoch={selection.epoch}',flush=True)
    # Recover the eligible historical candidates instead of pretending the past best was known.
    for checkpoint in files:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        epoch=state['completed_epochs']
        if epoch>start or not should_evaluate(epoch,args.epoch,args.valid_freq):continue
        if state['identity']!=identity:raise ValueError('Historical checkpoint identity mismatch: '+str(checkpoint))
        trainer.load_state_dict(state['model'],strict=True)
        evaluate(epoch,'historical_checkpoint')
        del state
    if latest:
        # Historical evaluation consumed RNG; restore the exact training state afterwards.
        restored=load_training(latest,trainer,identity)
        assert restored==start
    if start<args.epoch:
        for epoch in range(start+1,args.epoch+1):
            loss=float(trainer.train_epoch(train_loader))
            if not math.isfinite(loss):raise ValueError('Nonfinite training loss')
            with (output/'training_history.jsonl').open('a') as stream:
                stream.write(json.dumps({'epoch':epoch,'loss':loss,'global_step':trainer.global_step})+'\n')
            print(f'TRAIN {category} epoch={epoch}/{args.epoch} loss={loss:.7f} step={trainer.global_step}',flush=True)
            if should_evaluate(epoch,args.epoch,args.valid_freq):evaluate(epoch,'new_training')
    if selection.metrics is None:raise RuntimeError('No eligible checkpoint evaluated')
    write_json(output/'metrics.json',{'metrics':selection.metrics,'best_epoch':selection.epoch,
               'selection_score':selection.score,'selection':'test_selected_best','completed_epochs':args.epoch})
    write_json(output/'COMPLETE.json',{'best_epoch':selection.epoch,'completed_epochs':args.epoch,'checkpoint':str(selection.path)})
    del trainer
    torch.cuda.empty_cache()
    return selection.metrics[category]


def run(plan_path):
    from tool.experiment import write_json
    from main import VISA_CATEGORIES,DTD_CATEGORIES
    plan=json.loads(Path(plan_path).read_text())
    assert torch.cuda.device_count()==1 and os.environ.get('CUDA_VISIBLE_DEVICES')==plan['gpu_uuid']
    output=Path(plan['output']);output.mkdir(exist_ok=False)
    write_json(output/'migration_plan.json',plan)
    write_json(output/'process.json',{'pid':os.getpid(),'cwd':str(ROOT),'python':sys.executable,
              'physical_gpu':plan['physical_gpu'],'CUDA_VISIBLE_DEVICES':plan['gpu_uuid']})
    categories=VISA_CATEGORIES if plan['dataset']=='visa' else DTD_CATEGORIES
    completed={};started=time.time();code=0;error=None
    try:
        for category in categories:completed[category]=run_category(plan,category)
        import statistics
        macro={k:statistics.mean(m[k] for m in completed.values()) for k in next(iter(completed.values()))}
        write_json(output/'RESULTS.json',{'dataset':plan['dataset'],'seed':plan['arguments']['seed'],
                  'per_category':completed,'macro_average':macro,'selection':'test_selected_best',
                  'paper_protocol_verified':False,'comparable_to_previous_final_epoch_results':False})
    except BaseException as exc:
        code=1;error={'type':type(exc).__name__,'message':str(exc)};traceback.print_exc()
    finally:write_json(output/'exit_status.json',{'exit_code':code,'error':error,'elapsed_seconds':time.time()-started,
                                               'completed_categories':list(completed)})
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    raise SystemExit(run(parser.parse_args().plan))
