"""Real MARS: fresh-process epoch resume on four images; full bottle test export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['XFORMERS_DISABLED'] = '1'


def worker(output, mode):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from dataset import get_dataset
    from main import build_trainer
    from tool import setup_seed
    from tool.experiment import run_identity, save_training, load_training, PredictionExport, write_json
    args = json.loads((ROOT/'configs/observed.json').read_text())
    args.update(dataset='mvtec',category='bottle',ckt_path='')
    setup_seed(111)
    trainer = build_trainer(SimpleNamespace(**args), 'cuda:0')
    _, train, _ = get_dataset('mvtec',trainer.data_transform,trainer.target_transform,True,'FS',1e6,
                             synthesis_anomalies=False,class_name_list=[['bottle']])
    # In-memory bounded subset only; the external split and files remain untouched.
    train.data_all = train.data_all[:4]
    train.length = 4
    _, test, _ = get_dataset('mvtec',trainer.data_transform,trainer.target_transform,False,'FS',0,
                            synthesis_anomalies=False,class_name_list=[['bottle']])
    loader = DataLoader(train,batch_size=2,shuffle=True,num_workers=8,pin_memory=True)
    identity = run_identity(args, {'train':train,'test':test})
    start = load_training(output/'interrupted.training.pth',trainer,identity) if mode=='resumed' else 0
    losses = []
    end = 1 if mode=='interrupted' else 2
    for epoch in range(start,end):
        loss = float(trainer.train_epoch(loader))
        assert np.isfinite(loss)
        losses.append(loss)
    checkpoint = output/f'{mode}.training.pth'
    save_training(checkpoint,trainer,end,args,identity)
    write_json(output/f'{mode}.json',{'mode':mode,'start_epoch':start,'completed_epochs':end,
                                     'global_step':trainer.global_step,'losses':losses,'pid':os.getpid()})
    if mode == 'resumed':
        evaluation = DataLoader(test,batch_size=1,shuffle=False,num_workers=0)
        before = trainer.evaluation(evaluation,['bottle'],False,cal_pro=False)
        exporter = PredictionExport(output/'predictions','mvtec',111,checkpoint,test)
        after = trainer.evaluation(evaluation,['bottle'],False,cal_pro=False,exporter=exporter)
        assert before == after, (before,after)
        write_json(output/'export_metric_equality.json',{'status':'PASS','before':before,'after':after,
                   'count':len(test),'all_evaluator_metrics_identical':True,'evidence':'L1_SMALL_TRAIN_FULL_CATEGORY_EVAL'})


def compare(output):
    import torch
    from tool.experiment import write_json
    a=torch.load(output/'continuous.training.pth',map_location='cpu',weights_only=True)
    b=torch.load(output/'resumed.training.pth',map_location='cpu',weights_only=True)
    differences={}
    def visit(x,y,path):
        if isinstance(x,torch.Tensor):
            assert x.shape==y.shape and x.dtype==y.dtype,path
            if not torch.equal(x,y):
                differences[path]=float((x.double()-y.double()).abs().max())
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),path
            for key in x:visit(x[key],y[key],path+'/'+str(key))
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y),path
            for i,(u,v) in enumerate(zip(x,y)):visit(u,v,path+'/'+str(i))
        else:
            assert x==y,(path,x,y)
    for key in ['model','optimizer','rng','global_step','completed_epochs','scheduler','identity']:
        visit(a[key],b[key],key)
    write_json(output/'resume_verification.json',{'status':'BITWISE_EQUAL' if not differences else 'STATE_RESTORE_VERIFIED_WITH_NUMERICAL_DIFFERENCE',
               'bitwise_equal':not differences,'differences':differences,'global_step':b['global_step'],
               'completed_epochs':b['completed_epochs'],'evidence':'L1_4_TRAIN_IMAGES_2_EPOCHS',
               'comparison':'all model/BOS, optimizer moments, counters and all RNG states'})
    assert not any(k.startswith('rng/') for k in differences), 'RNG restoration differs'
    assert b['global_step'] == 4 and b['completed_epochs'] == 2


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',choices=['continuous','interrupted','resumed'])
    cli=parser.parse_args()
    if cli.worker:
        worker(cli.output,cli.worker)
    else:
        cli.output.mkdir(parents=True,exist_ok=False)
        for mode in ['continuous','interrupted','resumed']:
            subprocess.run([sys.executable,__file__,'--output',str(cli.output),'--worker',mode],check=True)
        compare(cli.output)
        subprocess.run([sys.executable,str(ROOT/'scripts/recompute_predictions.py'),str(cli.output/'predictions'),
                        '--output',str(cli.output/'recomputed_metrics.json')],check=True)
