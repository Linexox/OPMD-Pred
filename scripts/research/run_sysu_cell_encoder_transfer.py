"""Controlled SYSU transfer: identical tiling, GSE encoder, folds and fusion settings."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as TF

from run_methylation_cell_transfer import Fusion, Inputs, fit, predict
from sysu_binary_common import score, sha, write


def starts(length):
    last=max(0,length-256)
    return sorted(set(list(range(0,last+1,128))+[last]))


@torch.no_grad()
def encode_fields(model, bundle, data, out, device):
    model.eval();features={};tile_counts={};field_rows=[]
    for name in sorted({f for r in bundle['samples'] for f in r['cells']}):
        path=data/name;assert sha(path)==Path(name).stem
        with Image.open(path) as im:
            im=im.convert('RGB');tiles=[]
            for top in starts(im.height):
                for left in starts(im.width):
                    tile=Image.new('RGB',(256,256),'white')
                    tile.paste(im.crop((left,top,min(left+256,im.width),min(top+256,im.height))),(0,0))
                    tiles.append(TF.normalize(TF.to_tensor(tile.resize((224,224),Image.Resampling.BILINEAR)),[.485,.456,.406],[.229,.224,.225]))
        values=[]
        for start in range(0,len(tiles),32):values.append(model(torch.stack(tiles[start:start+32]).to(device)).cpu())
        values=torch.cat(values).numpy();features[name]=values.mean(0);tile_counts[name]=len(tiles)
        field_rows.append({'field':name,'tiles':len(tiles),'width':im.width,'height':im.height})
    # Equal field weight: first average windows within each field, then fields in a sample.
    array=np.array([np.mean([features[f] for f in r['cells']],0) if r['cells'] else np.zeros(512)
                    for r in bundle['samples']],dtype=np.float32)
    assert np.isfinite(array).all()
    np.savez_compressed(out/'cell_features.npz',values=array,sample_ids=[r['sample_id'] for r in bundle['samples']])
    write(out/'fields.json',{'fields':field_rows,'total_tiles':sum(tile_counts.values()),'aggregation':'mean windows per field, then mean fields per sample'})
    return array


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cell-checkpoint',type=Path,required=True);p.add_argument('--methyl-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();c=json.loads(a.config.read_text());b=json.loads((a.data/'bundle.json').read_text(encoding='utf-8'))
    assert sha(a.data/'photos.npz')==b['sources']['photos_sha256']
    a.output.mkdir(parents=True,exist_ok=False);torch.hub.set_dir(str(a.cache/'torch'))
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    methyl=torch.load(a.methyl_checkpoint,map_location='cpu',weights_only=True)
    assert methyl['genes']==b['genes'] and methyl['source_sha256']==b['sources']['external_csv_sha256']
    cell=torch.load(a.cell_checkpoint,map_location='cpu',weights_only=True)
    write(a.output/'identity.json',{'config':c,'sources':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('run_methylation_cell_transfer.py'),Path(__file__).with_name('sysu_binary_common.py')]},
        'bundle_sha256':sha(a.data/'bundle.json'),'cell_checkpoint_sha256':sha(a.cell_checkpoint),'methyl_checkpoint_sha256':sha(a.methyl_checkpoint),
        'scope':'Same original five SYSU folds, seed42. Frozen image encoders. No APP deployment. Compare encoders, not field-cropping policies.'})
    photos=np.load(a.data/'photos.npz',allow_pickle=False);results={}
    for condition in ['imagenet','oral_pretrained']:
        folder=a.output/condition;folder.mkdir()
        model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if condition=='oral_pretrained':
            model.fc=nn.Linear(512,4);model.load_state_dict(cell['model_state'],strict=True)
        model.fc=nn.Identity();model.to(a.device);model.requires_grad_(False)
        features=encode_fields(model,b,a.data,folder,a.device);del model
        predictions=[];folds=[]
        for fold in b['folds']:
            n=fold['fold'];path=folder/f'fold{n}'
            train=fold['inner_train']['sample_ids'];val=fold['inner_val']['sample_ids']
            outer=fold['outer_train']['sample_ids'];held=fold['outer_holdout']['sample_ids']
            inputs=Inputs(b,photos,features,train,True)
            _,epoch=fit(inputs,train,val,methyl['encoder'],c,c['seed']+n,c['fusion_epochs'],a.device,path/'inner')
            inputs=Inputs(b,photos,features,outer,True)
            fusion,_=fit(inputs,outer,None,methyl['encoder'],c,c['seed']+100+n,epoch,a.device,path/'refit')
            rows=predict(fusion,inputs,held,a.device,n)
            restored=Fusion(inputs.table.shape[1],c).to(a.device)
            restored.load_state_dict(torch.load(path/'refit/classifier.pt',map_location='cpu',weights_only=True)['state'],strict=True)
            reloaded=predict(restored,inputs,held,a.device,n)
            error=max([abs(v-w) for r,s in zip(rows,reloaded) if r['probabilities'] for v,w in zip(r['probabilities'],s['probabilities'])] or [0.])
            assert error<1e-6
            m=score(rows);item={'fold':n,'selected_epoch':epoch,'metrics':m,'reload_error':error}
            folds.append(item);predictions.extend(rows);write(path/'predictions.json',{'rows':rows,**item})
            print(f'SYSU {condition} fold={n} epoch={epoch} accuracy={m["accuracy"]:.4f}',flush=True)
        assert len(predictions)==len({r['sample_id'] for r in predictions})==117
        results[condition]={'metrics':score(predictions),'fold_mean_accuracy':float(np.mean([f['metrics']['accuracy'] for f in folds])),
                            'folds':folds,'rows':predictions}
        write(folder/'results.json',results[condition])
    write(a.output/'comparison.json',results)
    print('SYSU CELL TRANSFER COMPLETED',flush=True)


if __name__=='__main__':main()
