"""Read-only file inventory against the actual dataset solvers; no split changes."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DATASET_ROOTS
from dataset.base import BaseSolver
from dataset.visa import VisaSolver
from main import MVTEC_CATEGORIES, VISA_CATEGORIES, DTD_CATEGORIES, BTAD_CATEGORIES

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
out = parser.parse_args().output
out.mkdir(parents=True, exist_ok=False)
summary = {}
with (out/'dataset_inventory.jsonl').open('x') as inventory:
    for ds, categories in [('mvtec', MVTEC_CATEGORIES), ('visa', VISA_CATEGORIES), ('dtd', DTD_CATEGORIES), ('btad', BTAD_CATEGORIES)]:
        root = Path(DATASET_ROOTS[ds])
        solver = (VisaSolver if ds == 'visa' else BaseSolver)(str(root), categories)
        meta = solver.run()
        report = {'root': str(root), 'resolved': str(root.resolve()), 'categories': {}, 'errors': [], 'cross_split_identical_bytes': []}
        hashes = defaultdict(list)
        paths = {'train': set(), 'test': set()}
        used_masks = set()
        totals = Counter()
        for split in ['train', 'test']:
            for category in categories:
                counts = Counter()
                for row in meta[split][category]:
                    record = {'dataset':ds, 'split':split, **row}
                    p = root / row['img_path']
                    paths[split].add(str(p.resolve()))
                    try:
                        with Image.open(p) as image:
                            size = image.size
                            image.verify()
                        digest = hashlib.sha256(p.read_bytes()).hexdigest()
                        record.update(size=list(size), sha256=digest)
                        hashes[digest].append((split, row['img_path']))
                        counts['anomalous' if row['anomaly'] else 'normal'] += 1
                        if split == 'train' and row['anomaly']:
                            report['errors'].append(['anomalous_training_image', row['img_path']])
                        if row['anomaly']:
                            mask_path = root / row['mask_path']
                            with Image.open(mask_path) as mask:
                                a = np.asarray(mask.convert('L'))
                                assert mask.size == size, (mask.size, size)
                                assert a.max() > 0, 'empty anomalous mask'
                            used_masks.add(str(mask_path.resolve()))
                            counts['valid_masks'] += 1
                    except Exception as exc:
                        report['errors'].append([row['img_path'], str(exc)])
                    inventory.write(json.dumps(record)+'\n')
                report['categories'].setdefault(category,{})[split] = dict(counts)
                for k,v in counts.items():
                    totals[f'{split}_{k}'] += v
        report['totals'] = dict(totals)
        report['cross_split_path_overlap'] = sorted(paths['train'] & paths['test'])
        report['cross_split_identical_bytes'] = [v for v in hashes.values() if len({s for s,p in v}) > 1]
        report['unique_anomaly_masks'] = len(used_masks)
        report['mask_reuse_count'] = totals['test_valid_masks'] - len(used_masks)
        report['all_categories_nonempty'] = all(report['categories'][c]['train'].get('normal',0)>0 and report['categories'][c]['test'].get('normal',0)>0 and report['categories'][c]['test'].get('anomalous',0)>0 for c in categories)
        if ds == 'visa':
            with (root/'split_csv/1cls.csv').open() as f:
                rows = list(csv.DictReader(f))
            actual = {(s,c,r['img_path'],r['anomaly'],r['mask_path'] or '') for s in meta for c in categories for r in meta[s][c]}
            official = {(r['split'],r['object'],r['image'],int(r['label']!='normal'),r['mask'] or '') for r in rows}
            report['matches_local_1cls_csv'] = actual == official
            report['csv_difference_count'] = len(actual ^ official)
        summary[ds] = report
        print(ds, json.dumps({k:v for k,v in report.items() if k!='categories'}),flush=True)
(out/'dataset_audit.json').write_text(json.dumps(summary,indent=2))
