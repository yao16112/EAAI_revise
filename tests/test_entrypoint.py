"""Test the migrated epoch/checkpoint policy with no model training."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import main

events=[]
class Model:
    global_step=0
    data_transform=None
    target_transform=None
    def train_epoch(self,loader):
        events.append('train')
        return 0.1
    def save(self,path):
        assert path.endswith('_final.pth')
        events.append('save')
    def evaluation(self,*args,**kwargs):
        events.append('test')
        metrics={k:0.0 for k in ['auroc_im','f1_im','ap_im','auroc_px','f1_px','ap_px']}
        return {'bottle':metrics,'Average':metrics.copy()}

with tempfile.TemporaryDirectory() as tmp:
    args=json.loads((ROOT/'configs/observed.json').read_text())
    args.update(epoch=3,category='bottle',dataset='mvtec',save_path=tmp,gpu=0,
                save_fig=False,do_train=True,cal_pro=False,compute_fps=False,
                print_freq=1,ckt_path='')
    args=SimpleNamespace(**args)
    writer=SimpleNamespace(add_scalar=lambda *a:None,close=lambda:None)
    paths=('name',tmp,str(Path(tmp)/'metrics.csv'),str(Path(tmp)/'log.txt'),str(Path(tmp)/'model'),writer)
    with patch.object(main,'build_trainer',return_value=Model()), \
         patch.object(main.AnomalyMambaTrainer,'setup_paths',return_value=paths), \
         patch.object(main,'get_dataset',return_value=(['bottle'],[],'')), \
         patch.object(main.torch.utils.data,'DataLoader',return_value=[]), \
         patch.object(main.torch.cuda,'is_available',return_value=False), \
         patch.object(main.torch.cuda,'empty_cache'), \
         patch.object(main,'log_metrics'),patch.object(main,'save_metric'), \
         patch.object(main,'run_identity',return_value={}), \
         patch.object(main,'save_training',side_effect=lambda *a: events.append('snapshot')), \
         patch.object(main,'PredictionExport'):

        metrics=main.train_single_category(args,'bottle')
    assert events==['train','snapshot','train','snapshot','train','snapshot','save','test'],events
    assert metrics['Average']['auroc_px']==0.0
    config=json.loads((Path(tmp)/'model_config.json').read_text())
    assert config['checkpoint_selection']=='fixed_final_epoch'
print('PASS: fixed final checkpoint saved before a single test evaluation, including zero-score case')
