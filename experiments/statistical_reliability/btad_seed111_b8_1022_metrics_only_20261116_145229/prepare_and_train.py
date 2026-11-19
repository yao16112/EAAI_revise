"""Official BTAD download, isolated extraction, protocol audit, then authorized training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path('/opt/data/private/VarAD_EAAI')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
CONFIG=json.loads((OUT/'preparation_config.json').read_text())
STORAGE=Path(CONFIG['download']['storage'])
started=time.time()


def write(name,value):
    with (OUT/name).open('x') as stream:json.dump(value,stream,indent=2)


def digest(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def validate_member(destination,member):
    target=(destination/member.filename).resolve()
    if not target.is_relative_to(destination.resolve()) or stat.S_ISLNK(member.external_attr>>16):
        raise ValueError('Unsafe archive member: '+member.filename)


def prepare():
    archive=STORAGE/'btad.zip';partial=STORAGE/'btad.zip.part'
    assert not archive.exists() and not partial.exists()
    command=['curl','--fail','--location','--retry','3','--connect-timeout','20',
             '--speed-limit','1024','--speed-time','120','--dump-header',str(STORAGE/'http_headers.txt'),
             '--output',str(partial),CONFIG['download']['url']]
    write('download_command.json',{'argv':command,'cwd':str(ROOT),'pid':os.getpid()})
    print('STAGE DOWNLOAD '+CONFIG['download']['url'],flush=True)
    with (OUT/'download.log').open('x') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    write('download_exit.json',{'exit_code':result.returncode})
    if result.returncode:raise RuntimeError('BTAD download failed; see download.log')
    if partial.stat().st_size!=CONFIG['download']['expected_content_length']:
        raise ValueError('Archive length differs from verified official HTTP response; inspect before proceeding')
    partial.rename(archive)
    extracted=STORAGE/'extracted';extracted.mkdir(exist_ok=False)
    print('STAGE VERIFY_ARCHIVE_AND_EXTRACT',flush=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():validate_member(extracted,member)
        if shutil.disk_usage(STORAGE).free < sum(m.file_size for m in z.infolist())+1024**3:
            raise RuntimeError('Not enough space for isolated extraction')
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failure: '+bad)
        z.extractall(extracted)
    write('download_provenance.json',{'url':CONFIG['download']['url'],'archive':str(archive),
          'bytes':archive.stat().st_size,'sha256':digest(archive),'zip_crc':'PASS'})
    roots={p.parent for p in extracted.rglob('01') if p.is_dir()
           and all((p.parent/c/'train').is_dir() and (p.parent/c/'test').is_dir() for c in ['01','02','03'])}
    if len(roots)!=1:raise ValueError('Expected exactly one standard BTAD01/02/03 dataset root: '+str(roots))
    data_root=next(iter(roots))
    print('STAGE DATASET_AUDIT '+str(data_root),flush=True)
    from dataset.base import BaseSolver
    from PIL import Image
    import numpy as np
    meta=BaseSolver(str(data_root),['01','02','03']).run()
    train_hashes=set();test_hashes=set();summary={};errors=[]
    with (OUT/'dataset_inventory.jsonl').open('x') as inventory:
        for category in ['01','02','03']:
            counts={}
            for split in ['train','test']:
                counts[split]={'normal':0,'anomalous':0}
                for row in meta[split][category]:
                    image=data_root/row['img_path']
                    with Image.open(image) as source:size=source.size;source.verify()
                    image_hash=digest(image)
                    (train_hashes if split=='train' else test_hashes).add(image_hash)
                    counts[split]['anomalous' if row['anomaly'] else 'normal']+=1
                    if split=='train' and row['anomaly']:raise ValueError('Anomaly in training split')
                    files=[(row['img_path'],'image',image_hash)]
                    if row['anomaly']:
                        mask=data_root/row['mask_path']
                        if not row['mask_path'] or not mask.is_file():raise ValueError('Missing anomaly mask: '+row['img_path'])
                        with Image.open(mask) as source:
                            a=np.asarray(source.convert('L'))
                            if source.size!=size or not np.any(a>0):raise ValueError('Empty/misaligned mask: '+str(mask))
                        files.append((row['mask_path'],'mask',digest(mask)))
                    for path,kind,sha in files:
                        inventory.write(json.dumps({'dataset':'btad','category':category,'split':split,
                          'label':row['anomaly'],'path':path,'kind':kind,'sha256':sha})+'\n')
            assert counts['train']['normal'] and counts['test']['normal'] and counts['test']['anomalous'],counts
            summary[category]=counts
        if train_hashes & test_hashes:raise ValueError('Byte-identical images cross train/test splits')
    write('dataset_audit.json',{'status':'PASS','root':str(data_root),'counts':summary,
          'cross_split_identical_bytes':0,'anomaly_masks':'all decoded, nonempty, original dimensions match',
          'split_policy':'unaltered official directory split','paper_protocol_verified':False})
    profile=CONFIG.copy();profile['dataset_roots']={**CONFIG['dataset_roots'],'btad':str(data_root)}
    profile['dataset_inventory_sha256']=digest(OUT/'dataset_inventory.jsonl')
    write('protocol_snapshot.json',profile)
    paths=json.loads((ROOT/'configs/paths.json').read_text());paths['datasets']['btad']=str(data_root)
    write('paths.json',paths)
    environment=os.environ.copy();environment['MARS_PATHS_CONFIG']=str(OUT/'paths.json')
    command=[sys.executable,'-u',str(OUT/'runner.py')]
    print('STAGE TRAINING_ON_GPU2 '+json.dumps(summary),flush=True)
    with (OUT/'train.log').open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        write('training_process.json',{'pid':process.pid,'argv':command,'physical_gpu':2,
              'CUDA_VISIBLE_DEVICES':environment.get('CUDA_VISIBLE_DEVICES'),'MARS_PATHS_CONFIG':environment['MARS_PATHS_CONFIG'],
              'log':str(OUT/'train.log')})
        return process.wait()


if __name__=='__main__':
    # Focused archive boundary checks; no extraction or training in this self-check.
    destination=STORAGE/'test_boundary'
    validate_member(destination,zipfile.ZipInfo('01/train/ok/image.bmp'))
    try:validate_member(destination,zipfile.ZipInfo('../escape'))
    except ValueError:pass
    else:raise AssertionError('Archive traversal was accepted')
    code=1;error=None
    try:code=prepare()
    except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)};traceback.print_exc()
    finally:write('pipeline_exit.json',{'exit_code':code,'error':error,'elapsed_seconds':time.time()-started})
    raise SystemExit(code)
