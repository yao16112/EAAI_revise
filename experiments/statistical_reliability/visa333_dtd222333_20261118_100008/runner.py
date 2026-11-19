"""Fresh VisA/DTD launch checks; reuse the existing reference-best training entry."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/opt/data/private/VarAD_EAAI')
sys.path.insert(0, str(ROOT))


def main(plan_path):
    from tool.experiment import digest_file, write_json
    plan = json.loads(plan_path.read_text())
    out = plan_path.parent
    args = plan['arguments']
    assert (args['dataset'], args['seed']) in [('visa', 333), ('dtd', 222), ('dtd', 333)]
    assert not args['resume'] and not args['ckt_path'] and not args['save_fig']
    assert not Path(plan['previous_run']).exists()
    assert not Path(plan['output']).exists()
    assert (args['image_size'], args['batch_size'], args['learning_rate'], args['epoch']) == (1022, 8, 5e-4, 60)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == plan['gpu_uuid']
    write_json(out / 'process.json', {'pid': os.getpid(), 'cwd': str(ROOT), 'python': sys.executable,
        'physical_gpu': plan['physical_gpu'], 'gpu_uuid': plan['gpu_uuid'], 'seed': args['seed']})
    for name, sha in plan['code_sha256'].items():
        if digest_file(ROOT / name) != sha:
            raise ValueError('Frozen code changed: ' + name)
    if digest_file(plan['inventory']) != plan['inventory_sha256']:
        raise ValueError('Inventory changed')
    if args['dataset'] == 'visa' and digest_file(Path(plan['dataset_root']) / 'meta.json') != plan['meta_sha256']:
        raise ValueError('VisA metadata changed')
    count = 0
    for line in Path(plan['inventory']).read_text().splitlines():
        row = json.loads(line)
        if row['dataset'] != args['dataset']:
            continue
        if digest_file(Path(plan['dataset_root']) / row['path']) != row['sha256']:
            raise ValueError('Dataset file changed: ' + row['path'])
        count += 1
        if count % 1000 == 0:
            print(f'PRECHECK {count} files verified', flush=True)
    assert count == plan['expected_inventory_files']
    write_json(out / 'precheck.json', {'status': 'PASS', 'files': count})
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    from scripts.train_reference_best import run
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.9, 0)
    torch.cuda.reset_peak_memory_stats()
    observed = False
    def first_step(optimizer, positional, keywords):
        nonlocal observed
        if observed:
            return
        observed = True
        steps = [int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
        assert steps and min(steps) == max(steps) == 1
        torch.cuda.synchronize()
        record = {'status': 'FIRST_STEP_VERIFIED', 'dataset': args['dataset'], 'seed': args['seed'],
            'physical_gpu': plan['physical_gpu'], 'batch_size': args['batch_size'], 'image_size': args['image_size'],
            'lr': optimizer.param_groups[0]['lr'], 'source_checkpoint': None, 'optimizer_step': 1,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        write_json(out / 'first_step.json', record)
        print('FIRST_STEP ' + json.dumps(record), flush=True)
    hook = register_optimizer_step_post_hook(first_step)
    try:
        return run(plan_path)
    finally:
        hook.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    plan_path = parser.parse_args().plan
    code, error, started = 1, None, time.time()
    try:
        code = main(plan_path)
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        with (plan_path.parent / 'launcher_exit.json').open('x') as stream:
            json.dump({'exit_code': code, 'error': error, 'elapsed_seconds': time.time() - started}, stream, indent=2)
    raise SystemExit(code)
