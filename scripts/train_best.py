"""Legacy test-AUROC best-only policy for NEW runs; running entry points stay unchanged."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
import main as entry
from tool.experiment import digest_file, run_identity, save_exclusive, write_json


def save_best(path, trainer, epoch, metrics, config, previous=None):
    """Replace this new run's one model checkpoint only on a strict score improvement."""
    score = (float(metrics['Average']['auroc_im']) + float(metrics['Average']['auroc_px'])) / 2
    if not math.isfinite(score):
        raise ValueError('Nonfinite checkpoint selection score')
    if previous is not None and score <= previous['score']:
        return previous
    # sklearn metrics include NumPy scalars; keep metadata safe for weights_only loading.
    record = {'epoch': epoch, 'score': score, 'metrics': json.loads(json.dumps(metrics, allow_nan=False)),
              'criterion': 'test_mean_image_pixel_AUROC', 'uses_test_labels': True}
    # Version2 is model-only; the existing inference loader accepts it, full-resume rejects it.
    payload = {'format_version': 2, 'model': trainer.state_dict(), 'seed': config['seed'],
               'configuration': dict(config), 'selection': record}
    path = Path(path)
    if previous is None:
        save_exclusive(path, payload)
    else:
        pending = path.with_name(path.name + '.pending')
        save_exclusive(pending, payload)
        os.replace(pending, path)
    return record


def train_category(args, category):
    original_category = args.category
    args.category = category
    entry.setup_seed(args.seed)
    _, _, _, log_path, prefix, writer = entry.AnomalyMambaTrainer.setup_paths(args)
    logger = entry.Logger(log_path)
    try:
        trainer = entry.build_trainer(args, f'cuda:{args.gpu}')
        datasets = {}
        for split, training in [('train', True), ('test', False)]:
            _, data, _ = entry.get_dataset(args.dataset, trainer.data_transform, trainer.target_transform,
                                          training, 'FS', 1e6 if training else 0,
                                          synthesis_anomalies=False, class_name_list=[[category]])
            if not len(data):
                raise ValueError(f'Empty {split} dataset: {category}')
            datasets[split] = data
        identity = run_identity(vars(args), datasets)
        identity['code']['scripts/train_best.py'] = digest_file(__file__)
        write_json(prefix + '_config.json', {'arguments': vars(args), 'paths': entry.PATHS,
                                            'checkpoint_selection': 'test_mean_image_pixel_AUROC_best',
                                            'uses_test_labels': True, 'full_resume_supported': False})
        write_json(prefix + '_identity.json', identity)
        train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=8)
        test_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=0)
        best = None
        for epoch in range(1, args.epoch + 1):
            loss = trainer.train_epoch(train_loader)
            logger.info(f'epoch [{epoch}/{args.epoch}], loss:{loss:.6f}')
            writer.add_scalar('loss', loss, epoch)
            if epoch % args.valid_freq == 0 or epoch == args.epoch:
                metrics = trainer.evaluation(test_loader, [category], False, cal_pro=False, exporter=None)
                selected = save_best(prefix + '_best.pth', trainer, epoch, metrics, vars(args), best)
                if selected is not best:
                    logger.info(f'Best updated: epoch={epoch}, mean AUROC={selected["score"]:.6f}')
                best = selected
                entry.log_metrics(metrics, logger, writer, epoch)
        write_json(prefix + '_metrics.json', best['metrics'])
        write_json(prefix + '_selection.json', best)
        return best['metrics']
    finally:
        writer.close()
        args.category = original_category
        torch.cuda.empty_cache()


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['mvtec', 'visa', 'btad', 'dtd'], required=True)
    parser.add_argument('--category', nargs='+', default=['all'])
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--image_size', type=int, default=1022)
    parser.add_argument('--valid_freq', type=int, help='Default5 for MVTec/VisA;1 for BTAD/DTD')
    options = parser.parse_args()
    if options.valid_freq is None:
        options.valid_freq = 5 if options.dataset in ('mvtec', 'visa') else 1
    if (options.epoch < 1 or options.batch_size < 1 or options.learning_rate <= 0
            or options.valid_freq < 1 or options.image_size < 14 or options.image_size % 14):
        parser.error('Positive settings and an image size divisible by14 are required')
    categories = {'mvtec': entry.MVTEC_CATEGORIES, 'visa': entry.VISA_CATEGORIES,
                  'btad': entry.BTAD_CATEGORIES, 'dtd': entry.DTD_CATEGORIES}[options.dataset]
    if options.category != ['all'] and (len(set(options.category)) != len(options.category)
                                       or any(c not in categories for c in options.category)):
        parser.error('Invalid or duplicated categories')
    if not torch.cuda.is_available() or not 0 <= options.gpu < torch.cuda.device_count():
        parser.error('Requested CUDA device is unavailable; no batch/device fallback')
    config = json.loads((ROOT / 'configs/observed.json').read_text())
    config.update(vars(options), ckt_path='', resume='', stop_after_epoch=0, print_freq=1,
                  save_fig=False, cal_pro=False, compute_fps=False, do_train=True,
                  checkpoint_selection='test_mean_image_pixel_AUROC_best')
    config['category'] = 'all' if options.category == ['all'] else options.category
    Path(options.save_path).mkdir(parents=True, exist_ok=False)
    print('USER-REQUESTED LEGACY POLICY: test labels select the epoch. PAPER REPRODUCTION UNVERIFIED.', flush=True)
    entry.train_single_category = train_category
    entry.train(SimpleNamespace(**config))


if __name__ == '__main__':
    cli()
