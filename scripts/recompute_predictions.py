"""Recompute every evaluator metric from lossless exported predictions, without inference."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tool.metrics import calculate_metric, calculate_average_metric
from tool.experiment import write_json


def recompute(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['complete']
    rows = [json.loads(line) for line in (path / 'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == manifest['count'] == len({r['image_id'] for r in rows})
    results = {k: [] for k in ['cls_names', 'specie_name', 'imgs_gts', 'anomaly_scores', 'imgs_masks', 'anomaly_maps']}
    for row in rows:
        assert row['dataset'] == manifest['dataset'] and row['seed'] == manifest['seed']
        assert row['checkpoint_id'] == manifest['checkpoint']['sha256']
        with np.load(path / row['prediction_file'], allow_pickle=False) as data:
            raw, evaluation, mask = data['anomaly_map'], data['evaluation_map'], data['evaluation_mask']
            assert float(raw.max()) == row['score']
            expected = cv2.resize(raw, (512,512)) if raw.shape[0] > 512 else raw
            np.testing.assert_array_equal(expected, evaluation)
            assert mask.shape == evaluation.shape and set(np.unique(mask)) <= {0,1}
            results['anomaly_maps'].append(evaluation.copy())
            results['imgs_masks'].append(mask.copy())
        for key, value in [('cls_names', row['category']), ('specie_name', row['specie_name']),
                           ('imgs_gts', row['label']), ('anomaly_scores', row['score'])]:
            results[key].append(value)
    assert not manifest['by_specie'], 'Only the four public dataset category protocol is supported'
    metrics = {c: calculate_metric(results, c, manifest['cal_pro']) for c in sorted(set(results['cls_names']))}
    metrics['Average'] = calculate_average_metric(metrics)
    differences = {f'{c}/{k}': abs(v - manifest['metrics'][c][k]) for c, values in metrics.items() for k,v in values.items()}
    assert all(v == 0 for v in differences.values()), differences
    return {'status': 'PASS', 'count': len(rows), 'maximum_metric_difference': max(differences.values()),
            'metrics': metrics, 'seed': manifest['seed'], 'checkpoint': manifest['checkpoint']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = recompute(args.predictions)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
