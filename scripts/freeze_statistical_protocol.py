"""Read-only freeze of current MVTec/VisA splits, data bytes and observed settings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import DATASET_ROOTS
from dataset.base import BaseSolver
from dataset.visa import VisaSolver
from main import MVTEC_CATEGORIES,VISA_CATEGORIES
from tool.experiment import digest_file,write_json

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    observed=json.loads((ROOT/'configs/observed.json').read_text())
    observed.pop('seed')
    observed.update(cal_pro=False,save_fig=False,gpu=0)
    summary={}
    with (args.output/'dataset_files.jsonl').open('x') as inventory:
        for dataset,categories in [('mvtec',MVTEC_CATEGORIES),('visa',VISA_CATEGORIES)]:
            root=Path(DATASET_ROOTS[dataset])
            meta=(VisaSolver if dataset=='visa' else BaseSolver)(str(root),categories).run()
            train_hashes=set();test_hashes=set();counts=Counter();listed=set()
            for split in ['train','test']:
                for category in categories:
                    assert meta[split][category],(dataset,split,category)
                    for row in meta[split][category]:
                        assert not (split=='train' and row['anomaly'])
                        counts[split+('_A' if row['anomaly'] else '_N')]+=1
                        digest=digest_file(root/row['img_path'])
                        (train_hashes if split=='train' else test_hashes).add(digest)
                        for path,kind in [(row['img_path'],'image')]+([(row['mask_path'],'mask')] if row['anomaly'] else []):
                            identity=(split,path)
                            assert identity not in listed,(dataset,identity)
                            listed.add(identity)
                            inventory.write(json.dumps({'dataset':dataset,'split':split,'category':category,
                                'kind':kind,'path':path,'label':row['anomaly'],
                                'sha256':digest if kind=='image' else digest_file(root/path)})+'\n')
            assert not train_hashes & test_hashes
            if dataset=='visa':
                import csv
                with (root/'split_csv/1cls.csv').open() as stream:rows=list(csv.DictReader(stream))
                expected={(r['split'],r['object'],r['image'],int(r['label']!='normal'),r['mask'] or '') for r in rows}
                actual={(s,c,r['img_path'],r['anomaly'],r['mask_path'] or '') for s in meta for c in categories for r in meta[s][c]}
                assert actual==expected
            summary[dataset]={'root':str(root),'categories':categories,'counts':dict(counts),'cross_split_identical_bytes':0}
    files=[ROOT/'main.py',ROOT/'configs/observed.json',ROOT/'configs/paths.json']
    for folder in ['model','dataset','tool','config']:files.extend(sorted((ROOT/folder).rglob('*.py')))
    protocol={'status':'DRAFT_OBSERVED_IMPLEMENTATION','paper_protocol_verified':False,'execution_authorized':False,
              'seeds':[111,222,333,444,555],'checkpoint_selection':'fixed_final_epoch_60_before_test',
              'training':observed,'dataset_roots':{d:DATASET_ROOTS[d] for d in summary},
              'code_sha256':{str(p.relative_to(ROOT)):digest_file(p) for p in files},
              'dataset_inventory_sha256':digest_file(args.output/'dataset_files.jsonl')}
    write_json(args.output/'protocol.json',protocol)
    write_json(args.output/'dataset_summary.json',summary)
    commands=['#!/usr/bin/env bash','set -euo pipefail','cd /opt/data/private/VarAD_EAAI',
              'source /opt/conda/etc/profile.d/conda.sh','conda activate vmamba_new',
              'export PYTHONDONTWRITEBYTECODE=1','export PYTHONUNBUFFERED=1',
              '# Draft commands only. The launcher refuses the unresolved paper protocol.']
    for dataset,categories in [('mvtec',MVTEC_CATEGORIES),('visa',VISA_CATEGORIES)]:
        index={'dataset':dataset,'predictions':{}}
        for seed in protocol['seeds']:
            output=f'experiments/statistical_reliability/formal_{dataset}_seed{seed}_attempt01'
            commands.append(f'python scripts/run_statistical_seed.py --protocol {args.output}/protocol.json --dataset {dataset} --seed {seed} --output {output}')
            index['predictions'][str(seed)]={c:f'{output}/artifacts/predictions/{c}' for c in categories}
        write_json(args.output/f'{dataset}_collection.template.json',index)
    (args.output/'run_five_seeds.sh').write_text('\n'.join(commands)+'\n')
    print(json.dumps(summary,indent=2))
