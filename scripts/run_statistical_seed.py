"""Launch one frozen seed (or resume one category) into a new attempt directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tool.experiment import digest_file, write_json

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--dataset',choices=['mvtec','visa'],required=True)
    parser.add_argument('--seed',type=int,choices=[111,222,333,444,555],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--category')
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    if not protocol['paper_protocol_verified'] or not protocol['execution_authorized']:
        parser.error('Formal execution is disabled: paper protocol is unverified or execution is not authorized')
    if args.resume and not args.category:
        parser.error('Resume must name exactly the failed category')
    for name,expected in protocol['code_sha256'].items():
        if digest_file(ROOT/name)!=expected:
            parser.error('Frozen code changed: '+name)
    inventory=args.protocol.parent/'dataset_files.jsonl'
    if digest_file(inventory)!=protocol['dataset_inventory_sha256']:
        parser.error('Frozen inventory was changed')
    for line in inventory.read_text().splitlines():
        row=json.loads(line)
        if row['dataset']!=args.dataset:continue
        if digest_file(Path(protocol['dataset_roots'][args.dataset])/row['path'])!=row['sha256']:
            parser.error('Frozen data changed: '+row['path'])
    args.output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-u',str(ROOT/'main.py')]
    config={**protocol['training'], 'dataset':args.dataset, 'seed':args.seed,
            'save_path':str(args.output/'artifacts')}
    if args.category:config['category']=[args.category]
    if args.resume:config['resume']=str(args.resume.resolve())
    for name,value in config.items():
        command.append('--'+name)
        command.extend(str(v) for v in value) if isinstance(value,list) else command.append(str(value).lower() if isinstance(value,bool) else str(value))
    write_json(args.output/'launch.json',{'argv':command,'cwd':str(ROOT),'protocol':str(args.protocol.resolve()),'pid':os.getpid()})
    with (args.output/'train.log').open('x') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    write_json(args.output/'exit_status.json',{'exit_code':result.returncode})
    raise SystemExit(result.returncode)
