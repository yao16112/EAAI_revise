"""Synthetic verification of macro aggregation, sample SD, and missing-seed rejection."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import VISA_CATEGORIES
from scripts.summarize_seeds import summarize
from tool.experiment import PredictionExport
from types import SimpleNamespace

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary)
    index={'dataset':'visa','predictions':{}}
    for i,seed in enumerate([111,222,333,444,555],1):
        index['predictions'][str(seed)]={}
        for category in VISA_CATEGORIES:
            path=root/str(seed)/category
            path.mkdir(parents=True)
            (path/'manifest.json').write_text(json.dumps({'complete':True,'seed':seed,'dataset':'visa','count':1,
                    'checkpoint':{'sha256':f'fake{seed}'},'metrics':{category:{'auroc_im':i}}}))
            (path/'predictions.jsonl').write_text(json.dumps({'image_id':'test.png','seed':seed,'category':category,'checkpoint_id':f'fake{seed}'})+'\n')
            index['predictions'][str(seed)][category]=str(path)
    result=summarize(index)
    assert result['summary']['auroc_im']['mean']==3
    np.testing.assert_allclose(result['summary']['auroc_im']['sample_sd'],np.sqrt(2.5))
    del index['predictions']['555']
    try:summarize(index)
    except AssertionError:pass
    else:raise AssertionError('Missing seed was accepted')
    checkpoint=root/'model.pth'
    checkpoint.write_bytes(b'fake test identity')
    data=SimpleNamespace(root=str(root),data_all=[{'img_path':'a.png','mask_path':'','cls_name':'bottle','anomaly':0,'specie_name':'good'},
                                                 {'img_path':'b.png','mask_path':'','cls_name':'bottle','anomaly':0,'specie_name':'good'}])
    exporter=PredictionExport(root/'export','mvtec',111,checkpoint,data)
    a=np.zeros((2,2),dtype=np.float32)
    exporter.add(root/'a.png','bottle',0,0,a,a,a)
    for action in [lambda:exporter.add(root/'a.png','bottle',0,0,a,a,a),lambda:exporter.finish({},False,False),
                   lambda:exporter.add(root/'b.png','bottle',1,0,a,a,a)]:
        try:action()
        except ValueError:pass
        else:raise AssertionError('Export accepted duplicate/incomplete/misaligned predictions')
print('PASS: equal-category macro mean, ddof=1 SD, missing seeds rejected, prediction identity boundaries')
