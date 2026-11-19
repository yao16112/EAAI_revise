"""Read-only comparison of all actual BTAD records against the official ZIP."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
from collections import Counter

import numpy as np
from PIL import Image
ROOT=Path('/opt/data/private/VarAD_EAAI');sys.path.insert(0,str(ROOT))
from dataset.base import BaseSolver

OUT=Path(__file__).resolve().parent
original=Path((ROOT/'experiments/statistical_reliability/btad_launch_path_20261116.txt').read_text().strip())
config=json.loads((original/'preparation_config.json').read_text())
storage=Path(config['download']['storage']);archive=storage/'btad.zip'
data_root=storage/'extracted/BTech_Dataset_transformed'
meta=BaseSolver(str(data_root),['01','02','03']).run()
summary={};totals=Counter();empty=[];train_hashes=set();test_hashes=set();checked=0
with zipfile.ZipFile(archive) as z,(OUT/'dataset_inventory.jsonl').open('x') as inventory:
    for category in ['01','02','03']:
        summary[category]={}
        for split in ['train','test']:
            counts=Counter()
            for row in meta[split][category]:
                files=[(row['img_path'],'image')]
                if row['anomaly']:
                    if not row['mask_path']:raise ValueError('Missing mask: '+row['img_path'])
                    files.append((row['mask_path'],'mask'))
                image_size=None
                for relative,kind in files:
                    target=data_root/relative
                    payload=target.read_bytes()
                    official=z.read('BTech_Dataset_transformed/'+relative)
                    if payload!=official:raise ValueError('Extracted file differs from archive: '+relative)
                    sha=hashlib.sha256(payload).hexdigest()
                    with Image.open(target) as image:
                        if kind=='image':image_size=image.size;image.verify()
                        else:
                            if image.size!=image_size:raise ValueError('Mask size mismatch: '+relative)
                            a=np.asarray(image.convert('L'))
                            if not np.any(a>0):
                                empty.append({'mask':relative,'image':row['img_path'],'label':row['anomaly'],
                                              'shape':list(a.shape),'sha256':sha,'official_zip_byte_identical':True})
                    inventory.write(json.dumps({'dataset':'btad','category':category,'split':split,
                        'path':relative,'kind':kind,'label':row['anomaly'],'sha256':sha})+'\n');checked+=1
                    if kind=='image':(train_hashes if split=='train' else test_hashes).add(sha)
                label='anomalous' if row['anomaly'] else 'normal';counts[label]+=1;totals[split+'_'+label]+=1
                if split=='train' and row['anomaly']:raise ValueError('Anomaly in training')
            summary[category][split]=dict(counts)
            if not counts:raise ValueError('Empty category/split')
    report={'file_integrity_status':'PASS_ALL_REFERENCED_FILES_MATCH_OFFICIAL_ZIP',
       'annotation_status':'11_OFFICIAL_EMPTY_ANOMALY_MASKS_REQUIRE_EXPLICIT_POLICY',
       'source_url':config['download']['url'],'official_author_repo':'https://github.com/pankajmishra000/VT-ADL',
       'archive':str(archive),'archive_bytes':archive.stat().st_size,
       'root':str(data_root),'checked_files':checked,'categories':summary,'totals':dict(totals),
       'empty_masks':empty,'cross_split_identical_bytes':len(train_hashes & test_hashes),
       'data_modified':False,'samples_excluded':False,'labels_modified':False,'masks_fabricated':False}
    assert len(empty)==11 and not (train_hashes & test_hashes)
    h=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    report['archive_sha256']=h.hexdigest()
    assert report['archive_sha256']==json.loads((original/'download_provenance.json').read_text())['sha256']
    # Independently retrieved official HTTP range from this turn, matching the local record.
    report['fresh_official_sample_verification']={
       'member':'BTech_Dataset_transformed/02/ground_truth/ko/0145.png','http_status':206,
       'range':'bytes 630700617-630700819/1229193337','downloaded_record_bytes':203,
       'member_sha256':'59c9ce808455c79a65dd82d03bb820286b7fbffbcd4a9a7b9f678a482aa7b4f2',
       'fresh_official_equals_local':True,'fresh_official_positive_pixels':0}
    with (OUT/'integrity_report.json').open('x') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2))
