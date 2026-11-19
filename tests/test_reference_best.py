"""Reference selection, strict improvement, one-file replacement and BOS serialization."""
from pathlib import Path
import json
import sys
import tempfile
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_reference_best import BestCheckpoint,should_evaluate

model=torch.nn.Linear(2,2);model.register_buffer('BOS',torch.ones(2))
with tempfile.TemporaryDirectory() as temporary:
 path=Path(temporary)/'bottle_best.pth'
 selector=BestCheckpoint(model,path,{'seed':111})
 def metrics(i,p):return {'Average':{'auroc_im':i,'auroc_px':p},'bottle':{'auroc_im':i,'auroc_px':p}}
 assert selector.consider(metrics(0,0),5)
 assert selector.consider(metrics(80,100),10)
 before=path.read_bytes()
 assert not selector.consider(metrics(100,70),15)
 assert not selector.consider(metrics(100,80),20)
 assert path.read_bytes()==before
 with torch.no_grad():model.weight.add_(1)
 assert selector.consider(metrics(95,95),25)
 assert selector.epoch==25 and selector.score==95
 assert len(list(Path(temporary).glob('*.pth')))==1
 state=torch.load(path,weights_only=True)
 assert 'BOS' in state and 'optimizer' not in state
 assert torch.equal(state['weight'],model.weight)
 assert json.loads(path.with_suffix('.json').read_text())['best_epoch']==25
 assert [e for e in range(1,13) if should_evaluate(e,12,5)]==[5,10,12]
 print('PASS: AUROC-mean selection, strict improvement/ties, best-only atomic replacement, BOS, model-only contents, frequency/final epoch')
