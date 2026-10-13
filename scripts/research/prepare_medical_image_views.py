"""Paired frozen DINO features: existing augmentation vs conservative medical policy."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision.transforms import functional as F

from opmd_pred.data.sysu import load_sysu_records
from opmd_pred.data.sysu_protocol import file_sha256
from opmd_pred.model.vit import ImageEncoder
from opmd_pred.data.transforms import make_image_transform

ENCODER_SHA = '58a3fcc724155c8ecfc7bf0e955b2bf022f0e2aa4bfed06a51a5b99c850a4a65'
POLICY = {'name':'MedAugment-inspired conservative oral-photo policy v1', 'level':2,
    'strategies':[[1,2],[0,3],[1,1],[0,2]], 'pixel':['brightness','contrast','sharpness'],
    'spatial':['rotate_expand','horizontal_flip','translate_x_expand','translate_y_expand'],
    'probability':.4,'brightness_contrast_delta':.08,'rotation_max_degrees':8,
    'translation_max_image_fraction':.04,'sharpness_factor':[.95,1.1],
    'all_spatial_operations_preserve_full_image':True,'ROI_annotations_used':False,
    'paper':'https://arxiv.org/abs/2306.17466v5','exact_paper_reproduction':False}


def uniform(a,b):
    return a+(b-a)*torch.rand(()).item()


def medical(image, view):
    pixel, spatial = POLICY['strategies'][view-1]
    pix = [POLICY['pixel'][i] for i in torch.randperm(3)[:pixel].tolist()]
    geo = [POLICY['spatial'][i] for i in torch.randperm(4)[:spatial].tolist()]
    operations = pix+geo
    for i in torch.randperm(len(operations)).tolist():
        op = operations[i]
        if torch.rand(()).item() >= POLICY['probability']:
            continue
        if op in {'brightness','contrast','sharpness'}:
            cls = {'brightness':ImageEnhance.Brightness,'contrast':ImageEnhance.Contrast,'sharpness':ImageEnhance.Sharpness}[op]
            factor = uniform(.95,1.1) if op=='sharpness' else uniform(.92,1.08)
            image = cls(image).enhance(factor)
        elif op=='rotate_expand':
            image = image.rotate(uniform(-8,8),resample=Image.Resampling.BILINEAR,expand=True,fillcolor=(124,116,104))
        elif op=='horizontal_flip':
            image = ImageOps.mirror(image)
        else:
            axis = 0 if op=='translate_x_expand' else 1
            extent = max(1,round(image.size[axis]*.04))
            offset = round(uniform(-extent,extent))
            w,h = image.size
            canvas = Image.new('RGB',(w+2*extent if axis==0 else w,h+2*extent if axis==1 else h),(124,116,104))
            canvas.paste(image,(extent+offset,0) if axis==0 else (0,extent+offset))
            image = canvas
    return image


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input','prior-cache','encoder','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not replace a frozen image cache')
    assert args.output.resolve().is_relative_to(Path('/home/zjkj/24334067'))
    protocol = json.loads((args.input/'development_nested_cv_v3.json').read_text())
    assert protocol['format']=='sysu-raw-rebuilt-development-cv-v3'
    assert file_sha256(args.input/'sysu_samples.csv')==protocol['csv_sha256']
    assert file_sha256(args.encoder)==ENCODER_SHA
    entries = protocol['development_entries']
    ids = [e['sample_id'] for e in entries]
    assert not set(ids)&set(protocol['locked_previous_test_ids']+protocol['additional_related_locked_ids'])
    records = {r.sample_id:r for r in load_sysu_records(args.input/'sysu_samples.csv')}
    oldmeta = json.loads((args.prior_cache/'metadata.json').read_text())
    assert oldmeta['encoder_sha256']==ENCODER_SHA
    assert file_sha256(args.prior_cache/'image_views.npz')==oldmeta['image_views_sha256']
    with np.load(args.prior_cache/'image_views.npz',allow_pickle=False) as prior:
        oldids,oldfeatures = prior['sample_ids'].tolist(),prior['features'].copy()
    by_sha = {oldmeta['source_image_sha_by_id'][sid]:oldfeatures[i] for i,sid in enumerate(oldids) if oldmeta['source_image_sha_by_id'][sid]}
    present = np.asarray([bool(records[sid].image_path) for sid in ids])
    baseline = np.zeros((len(ids),5,1536),dtype=np.float32)
    for i,e in enumerate(entries):
        if present[i]:
            assert file_sha256(args.input/records[ids[i]].image_path)==e['image_sha256']
            baseline[i] = by_sha[e['image_sha256']]
    medical_features = baseline.copy()
    gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
    if gpu:
        raise RuntimeError('Other GPU job is present')
    encoder = ImageEncoder(pretrained=False,kind='dinov2_b14')
    encoder.load_artifact(args.encoder)
    encoder.freeze()
    encoder.to('cuda').eval()
    evaluation_transform = make_image_transform(False,'dinov2_b14')
    grouped = {}
    for i,e in enumerate(entries):
        if present[i]:
            grouped.setdefault(e['image_sha256'],[]).append(i)
    args.output.mkdir(parents=True)
    for step,(digest,indices) in enumerate(grouped.items(),1):
        with Image.open(args.input/records[ids[indices[0]]].image_path) as opened:
            image = opened.convert('RGB')
        tensors, previews = [],[]
        for view in range(1,5):
            seed = int(hashlib.sha256(f'42:{digest}:{view}'.encode()).hexdigest()[:8],16)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                transformed = medical(image.copy(),view)
                tensors.append(evaluation_transform(transformed))
                previews.append(ImageOps.pad(transformed,(300,300),color=(124,116,104)))
        values = encoder(torch.stack(tensors).to('cuda')).cpu().numpy()
        assert np.isfinite(values).all() and values.shape==(4,1536)
        for view in range(1,5):
            medical_features[indices,view] = values[view-1]
        if step<=8:
            board = Image.new('RGB',(1500,300),(255,255,255))
            board.paste(ImageOps.pad(image,(300,300),color=(124,116,104)),(0,0))
            for j,img in enumerate(previews,1):
                board.paste(img,(j*300,0))
            board.save(args.output/f'preview_{step:02d}.jpg')
        if step%10==0:
            print(json.dumps({'images_encoded':step,'total_unique_images':len(grouped)}),flush=True)
    assert np.array_equal(baseline[:,0],medical_features[:,0])
    assert np.count_nonzero(medical_features[~present])==0
    for name,features in [('baseline',baseline),('medical_aug',medical_features)]:
        np.savez_compressed(args.output/f'{name}.npz',features=features,sample_ids=np.asarray(ids),
                            image_present=present,labels=np.asarray([e['label'] for e in entries]))
    metadata = {'format':'sysu-paired-medical-image-views-v1','csv_sha256':protocol['csv_sha256'],
        'cv_protocol_sha256':file_sha256(args.input/'development_nested_cv_v3.json'),'encoder_sha256':ENCODER_SHA,
        'script_sha256':file_sha256(Path(__file__)),'policy':POLICY,'evaluation_view':0,'training_views':[1,2,3,4],
        'source_image_sha_by_id':{e['sample_id']:e['image_sha256'] for e in entries},
        'cache_sha256':{n:file_sha256(args.output/f'{n}.npz') for n in ('baseline','medical_aug')},
        'prior_cache_sha256':oldmeta['image_views_sha256'],'evaluation_features_reused_by_exact_image_SHA':True,
        'previous_test_inference':False,'trainable_encoder_parameters':0,
        'record_count':len(ids),'image_record_count':int(present.sum()),'unique_image_count':len(grouped)}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'records':len(ids),'image_records':int(present.sum()),'unique_images':len(grouped)}),flush=True)


if __name__=='__main__':
    main()
