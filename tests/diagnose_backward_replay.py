"""Replay identical noisy forward/backward twice, with no optimizer update."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import build_trainer
from tool import setup_seed
from tool.experiment import rng_state,restore_rng,write_json
from dataset import get_dataset

setup_seed(111)
args=json.loads((ROOT/'configs/observed.json').read_text())
trainer=build_trainer(SimpleNamespace(**args,ckt_path=''),'cuda:0')
_,data,_=get_dataset('mvtec',trainer.data_transform,trainer.target_transform,True,'FS',1e6,
                     synthesis_anomalies=False,class_name_list=[['bottle']])
x=torch.stack([data[i]['img'] for i in [0,1]]).cuda()
trainer.model.train()
state=rng_state()
records=[]
for repeat in range(2):
    restore_rng(state)
    trainer.optimizer.zero_grad()
    outputs=trainer.model(x)
    loss=trainer.model.cal_loss(outputs)
    loss.backward()
    records.append({'loss':float(loss),'outputs':[t.detach().cpu() for group in outputs[:2] for t in group],
                    'gradients':{n:p.grad.detach().cpu().clone() for n,p in trainer.named_parameters() if p.grad is not None}})
forward_equal=all(torch.equal(a,b) for a,b in zip(records[0]['outputs'],records[1]['outputs']))
differences={n:float((a-records[1]['gradients'][n]).abs().max()) for n,a in records[0]['gradients'].items() if not torch.equal(a,records[1]['gradients'][n])}
result={'identical_forward_outputs':forward_equal,'losses':[r['loss'] for r in records],
        'gradient_max_differences':differences,'optimizer_updates':0,
        'interpretation':'Backward is nondeterministic despite identical inputs/weights/RNG and forward' if forward_equal and differences else 'Further diagnosis needed',
        'backend':'existing selective_scan CUDA extension and installed Torch; precise kernel attribution unproven'}
write_json(ROOT/'experiments/statistical_reliability/engineering_v2/backward_replay.json',result)
print(json.dumps(result,indent=2))
assert forward_equal
