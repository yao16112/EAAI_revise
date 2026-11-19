"""Fixed-five-seed category-macro mean ± sample SD; no bootstrap or seed selection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import MVTEC_CATEGORIES,VISA_CATEGORIES
from tool.experiment import write_json


def summarize(index):
    seeds=[111,222,333,444,555]
    assert index['dataset'] in ('mvtec','visa')
    categories=MVTEC_CATEGORIES if index['dataset']=='mvtec' else VISA_CATEGORIES
    assert set(index['predictions'])=={str(s) for s in seeds}, 'All five seeds are required'
    values={}
    metric_keys=None
    reference_images={}
    checkpoint_ids={c:set() for c in categories}
    for seed in seeds:
        paths=index['predictions'][str(seed)]
        assert set(paths)==set(categories),'Missing/extra categories'
        per_category=[]
        for category in categories:
            path=Path(paths[category])
            manifest=json.loads((path/'manifest.json').read_text())
            assert manifest['complete'] and manifest['seed']==seed and manifest['dataset']==index['dataset']
            rows=[json.loads(line) for line in (path/'predictions.jsonl').read_text().splitlines()]
            assert len(rows)==manifest['count']>0
            assert len({r['image_id'] for r in rows})==len(rows)
            assert all(r['seed']==seed and r['category']==category and r['checkpoint_id']==manifest['checkpoint']['sha256'] for r in rows)
            image_identity=sorted((r['image_id'],r.get('label'),r.get('mask_relative_path')) for r in rows)
            if category in reference_images:
                assert reference_images[category]==image_identity,'Test identities/labels differ across seeds'
            reference_images[category]=image_identity
            checkpoint_id=manifest['checkpoint']['sha256']
            assert checkpoint_id not in checkpoint_ids[category],'A checkpoint was reused across seeds'
            checkpoint_ids[category].add(checkpoint_id)
            metrics=manifest['metrics'][category]
            if metric_keys is None:metric_keys=set(metrics)
            assert set(metrics)==metric_keys
            assert all(np.isfinite(v) for v in metrics.values())
            per_category.append(metrics)
        values[str(seed)]={k:float(np.mean([m[k] for m in per_category])) for k in sorted(metric_keys)}
    return {'dataset':index['dataset'],'seeds':seeds,'category_aggregation':'equal macro mean',
            'per_seed':values,'summary':{k:{'mean':float(np.mean([values[str(s)][k] for s in seeds])),
            'sample_sd':float(np.std([values[str(s)][k] for s in seeds],ddof=1))} for k in sorted(metric_keys)},
            'bootstrap_performed':False,'note':'Engineering aggregation does not verify paper protocol.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    write_json(args.output,summarize(json.loads(args.index.read_text())))
