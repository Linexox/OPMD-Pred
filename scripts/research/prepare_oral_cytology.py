"""Patient-separated cell crops in an indexed training tar; raw images stay unchanged."""
import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sysu_binary_common import sha, write


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new preparation directory')
    patches=rows(a.audit/'patch_inventory.csv');objects=rows(a.audit/'nucleus_inventory.csv')
    audit=json.loads((a.audit/'summary.json').read_text(encoding='utf-8'))
    assert len(patches)==739 and len(objects)==20021 and not audit['bad_images']
    labels=['I','II','III','IV']; by_patch=defaultdict(list);patient_labels=defaultdict(set)
    for r in objects:
        by_patch[r['patch']].append(r);patient_labels[r['patient_id']].add(labels.index(r['nucleus_category']))
    rng=np.random.default_rng(a.seed);splits={};strata=defaultdict(list)
    for patient,grades in sorted(patient_labels.items()):strata[max(grades)].append(patient)
    for grade,patients in sorted(strata.items()):
        shuffled=rng.permutation(sorted(patients)).tolist();n=max(2,int(round(len(patients)*.15)))
        for split,ids in [('test',shuffled[:n]),('validation',shuffled[n:2*n]),('train',shuffled[2*n:])]:
            for patient in ids:splits[patient]=split
    assert len(splits)==116
    a.output.mkdir(parents=True)
    samples=[];excluded=[];seen=set(); tar_path=a.output/'crops.tar'
    with tarfile.open(tar_path,'w',format=tarfile.USTAR_FORMAT) as archive:
        for index,patch in enumerate(patches):
            name=patch['patch'];source=a.root/'patches'/(name+'.png')
            assert sha(source)==patch['sha256']
            origin=name.rsplit('_',2);ox,oy=int(origin[-2]),int(origin[-1])
            with Image.open(source) as image:
                image=image.convert('RGB')
                for r in by_patch[name]:
                    x0,y0,x1,y1=[float(r[k]) for k in ['x0','y0','x1','y1']]
                    if x1-x0<8 or y1-y0<8:
                        excluded.append({'patch':name,'object_index':r['object_index'],'reason':'bbox_side_below_8'});continue
                    identity=(r['slide_id'],round(ox+x0,2),round(oy+y0,2),round(ox+x1,2),round(oy+y1,2),r['nucleus_category'])
                    if identity in seen:
                        excluded.append({'patch':name,'object_index':r['object_index'],'reason':'repeated_absolute_bbox_same_slide_and_class'});continue
                    seen.add(identity)
                    cx=(max(0,x0)+min(image.width,x1))/2;cy=(max(0,y0)+min(image.height,y1))/2
                    left=math.floor(cx-128);top=math.floor(cy-128)
                    crop=Image.new('RGB',(256,256),'white')
                    clipped=(max(0,left),max(0,top),min(image.width,left+256),min(image.height,top+256))
                    crop.paste(image.crop(clipped),(clipped[0]-left,clipped[1]-top))
                    encoded=io.BytesIO();crop.save(encoded,format='JPEG',quality=95,subsampling=0)
                    payload=encoded.getvalue();member=f'crops/{name}/{int(r["object_index"]):04d}.jpg'
                    item=tarfile.TarInfo(member);item.size=len(payload);item.mtime=0;item.mode=0o600
                    offset=archive.fileobj.tell()+512
                    archive.addfile(item,io.BytesIO(payload))
                    samples.append({'member':member,'offset':offset,'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
                        'patient_id':r['patient_id'],'slide_id':r['slide_id'],'patch':name,'object_index':int(r['object_index']),
                        'class':labels.index(r['nucleus_category']),'stain':r['stain'],'center':patch['center'],
                        'split':splits[r['patient_id']],'source_image_sha256':patch['sha256'],
                        'left':left,'top':top,'bbox_clipped_to_image':r['outside_image']=='True'})
            if (index+1)%150==0:print(f'Prepared {index+1}/739 source patches',flush=True)
    # Validate byte offsets against the actual tar directory and decode examples.
    with tarfile.open(tar_path) as archive:
        members=archive.getmembers();assert len(members)==len(samples)
        for m,r in zip(members,samples):assert m.offset_data==r['offset'] and m.size==r['size'] and m.name==r['member']
    with tar_path.open('rb') as stream:
        for r in samples[::max(1,len(samples)//64)]:
            stream.seek(r['offset']);raw=stream.read(r['size']);assert hashlib.sha256(raw).hexdigest()==r['sha256']
            with Image.open(io.BytesIO(raw)) as image: image.load();assert image.size==(256,256)
    statistics={}
    for split in ['train','validation','test']:
        selected=[r for r in samples if r['split']==split]
        counts=Counter(r['class'] for r in selected)
        assert set(counts)==set(range(4))
        assert all(len({r['patient_id'] for r in selected if r['class']==c})>=2 for c in range(4))
        statistics[split]={'patients':len({r['patient_id'] for r in selected}),'crops':len(selected),
                           'class_counts':{labels[k]:v for k,v in sorted(counts.items())},
                           'stains':dict(Counter(r['stain'] for r in selected)),
                           'centers':dict(Counter(r['center'] for r in selected))}
    write(a.output/'patient_split.json',{'seed':a.seed,'patient_splits':splits,'stratification':'highest local category per patient','statistics':statistics})
    write(a.output/'manifest.json',{'labels':labels,'crop_size':256,'encoding':'JPEG quality95 subsampling0; original RGB PNG unchanged',
          'preprocessing':'Center on clipped nucleus bounding box, 256x256, white padding; no text or polygon drawn into crops.',
          'tar_sha256':sha(tar_path),'audit_sha256':sha(a.audit/'summary.json'),'samples':samples})
    write(a.output/'excluded.json',excluded)
    write(a.output/'summary.json',{'images':739,'crop_candidates':20021,'prepared_crops':len(samples),
          'excluded':dict(Counter(r['reason'] for r in excluded)),'tar_bytes':tar_path.stat().st_size,'splits':statistics})
    print(json.dumps({'prepared_crops':len(samples),'tar_bytes':tar_path.stat().st_size,'splits':statistics}),flush=True)


if __name__=='__main__':main()
