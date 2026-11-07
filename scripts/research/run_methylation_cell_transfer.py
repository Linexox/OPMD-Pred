"""Seed-controlled methylation transfer and SYSU cell-feature ablation."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision import transforms as T

from sysu_binary_common import score, sha, write


def seed(value):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class MethylEncoder(nn.Sequential):
    def __init__(self, dimension=32):
        super().__init__(nn.Linear(20, dimension), nn.ReLU(), nn.Linear(dimension, dimension), nn.ReLU())


class Fusion(nn.Module):
    def __init__(self, table_size, c):
        super().__init__()
        d = c['dimension']
        self.methyl = MethylEncoder(d)
        self.photo = nn.Sequential(nn.Linear(512, d), nn.LayerNorm(d), nn.GELU())
        self.cell = nn.Sequential(nn.Linear(512, d), nn.LayerNorm(d), nn.GELU())
        self.table = nn.Sequential(nn.Linear(table_size, d), nn.ReLU(), nn.Linear(d, d))
        self.missing = nn.Parameter(torch.zeros(4, d))
        self.kind = nn.Parameter(torch.randn(4, d) * .02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 4, d * 4, dropout=c['dropout'], activation='gelu',
                                          batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, batch):
        # Zero hidden raw values before encoding, then replace absent branch outputs.
        functions = [self.photo, self.methyl, self.table, self.cell]
        tokens = []
        for j, (key, encoder) in enumerate(zip(('photo', 'methyl', 'table', 'cell'), functions)):
            present = batch['present'][:, j, None]
            value = torch.where(present, batch[key], torch.zeros_like(batch[key]))
            tokens.append(torch.where(present, encoder(value), self.missing[j]) + self.kind[j])
        z = torch.cat([self.cls.expand(len(tokens[0]), -1, -1), torch.stack(tokens, 1)], 1)
        return self.head(self.fusion(z)[:, 0])


def weights(y, classes, device):
    counts = np.bincount(y, minlength=classes)
    assert (counts > 0).all()
    return torch.tensor(len(y) / (classes * counts), dtype=torch.float32, device=device)


def pretrain(bundle, c, out, device):
    x = np.array([r['genes'] + [1.] * 10 for r in bundle['external']], dtype=np.float32)
    y = np.array([r['target'] for r in bundle['external']])
    rng = np.random.default_rng(c['seed'])
    train, val = [], []
    for cls in range(3):
        ids = rng.permutation(np.flatnonzero(y == cls)); n = int(np.ceil(len(ids) * .2))
        val.extend(ids[:n]); train.extend(ids[n:])
    x = torch.tensor(x, device=device); target = torch.tensor(y, device=device)
    def fit(ids, epochs, validate):
        seed(c['seed'])
        model = nn.Sequential(MethylEncoder(c['dimension']), nn.Linear(c['dimension'], 3)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        w = weights(y[ids], 3, device)
        history, best, selected, stale, best_pred = [], (-1., -float('inf')), 1, 0, None
        for epoch in range(1, epochs + 1):
            model.train(); total = 0.
            order = np.asarray(ids)[torch.randperm(len(ids)).numpy()]
            for start in range(0, len(ids), c['batch_size']):
                ix = order[start:start + c['batch_size']]
                loss = F.cross_entropy(model(x[ix]), target[ix], weight=w)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); total += float(loss.detach()) * len(ix)
            entry = {'epoch': epoch, 'train_loss': total / len(ids)}
            if validate:
                model.eval()
                with torch.no_grad():
                    logits = model(x[val]); pred = logits.argmax(1).cpu().numpy()
                    ce = float(F.cross_entropy(logits, target[val], weight=w))
                recalls = [float((pred[y[val] == k] == k).mean()) for k in range(3)]
                key = (float(np.mean(recalls)), -ce)
                entry.update(validation_loss=ce, validation_recalls=recalls, validation_accuracy=float((pred == y[val]).mean()))
                if key > best:
                    best, selected, stale = key, epoch, 0
                    best_pred = {'targets': y[val].tolist(), 'predictions': pred.tolist(), 'recalls': recalls,
                                 'accuracy': float((pred == y[val]).mean())}
                else:
                    stale += 1
            history.append(entry)
            if validate and stale >= c['pretrain_patience']:
                break
        return model, history, selected, best_pred
    _, history, selected, best_pred = fit(train, c['pretrain_epochs'], True)
    model, refit, _, _ = fit(list(range(len(y))), selected, False)
    encoder = {k: v.detach().cpu() for k, v in model[0].state_dict().items()}
    torch.save({'encoder': encoder, 'genes': bundle['genes'], 'config': c,
                'selected_epoch': selected, 'source_sha256': bundle['sources']['external_csv_sha256']}, out / 'methylation_encoder.pt')
    write(out / 'pretraining.json', {'train_count': len(train), 'validation_count': len(val), 'selected_epoch': selected,
        'train_ids': [bundle['external'][i]['id'] for i in train], 'validation_ids': [bundle['external'][i]['id'] for i in val],
        'validation_at_selected_epoch': best_pred, 'selection_history': history, 'refit_history': refit,
        'note': 'Validation selected epoch; not an independent external test. Final encoder refit on all 118 external records.'})
    print(f'External pretraining done: selected epoch {selected}', flush=True)
    return encoder


def cell_features(bundle, data, cache, device):
    files = sorted({p for r in bundle['samples'] for p in r['cells']})
    torch.hub.set_dir(str(cache / 'torch'))
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.fc = nn.Identity(); model.eval(); model.requires_grad_(False)
    # Full fields are padded to square before resizing so edge cells are retained.
    arrays = []
    with torch.no_grad():
        for start in range(0, len(files), 16):
            images = []
            for name in files[start:start + 16]:
                with Image.open(data / name) as im:
                    im = im.convert('RGB'); side = max(im.size)
                    square = Image.new('RGB', (side, side), (255, 255, 255))
                    square.paste(im, ((side-im.width)//2, (side-im.height)//2))
                    # Resize whole padded field directly; no crop.
                    images.append(T.functional.normalize(T.functional.to_tensor(square.resize((224,224), Image.Resampling.BILINEAR)), [.485,.456,.406],[.229,.224,.225]))
            arrays.extend(model(torch.stack(images).to(device)).cpu().numpy())
    lookup = dict(zip(files, arrays))
    values = np.array([np.mean([lookup[p] for p in r['cells']], axis=0) if r['cells'] else np.zeros(512)
                       for r in bundle['samples']], dtype=np.float32)
    assert np.isfinite(values).all()
    np.savez_compressed(cache / 'cell_features.npz', values=values)
    weight_path = cache / 'torch/checkpoints/resnet18-f37072fd.pth'
    write(cache / 'cell_encoder.json', {'weights': 'ResNet18_Weights.IMAGENET1K_V1', 'weight_sha256': sha(weight_path),
          'images': len(files), 'trainable_backbone': False, 'preprocessing': 'RGB, white square pad, bilinear 224x224, ImageNet normalization; mean over sample fields'})
    return values


class Inputs:
    def __init__(self, bundle, photos, cells, fit_ids, use_cells):
        self.samples = bundle['samples']; self.ids = [r['sample_id'] for r in self.samples]
        self.index = {s:i for i,s in enumerate(self.ids)}
        self.grade = np.array([r['grade'] for r in self.samples]); self.y = (self.grade > 0).astype(np.int64)
        self.numeric = np.array([r['numeric'] for r in self.samples], dtype=np.float32)
        ix = [self.index[s] for s in fit_ids]
        means, scales = [], []
        for j in range(self.numeric.shape[1]):
            v = self.numeric[ix,j]; v = v[np.isfinite(v)]
            means.append(float(v.mean()) if len(v) else 0.)
            scales.append(float(v.std()) if len(v) and v.std() > 1e-6 else 1.)
        self.mean = np.array(means, dtype=np.float32); self.scale = np.array(scales,dtype=np.float32)
        missing = {'', '/', 'NA', 'nan', 'N/A', 'None'}
        self.tct = ['' if r['tct'] in missing else r['tct'] for r in self.samples]
        self.vocab = sorted({self.tct[i] for i in ix if self.tct[i]})
        table = np.zeros((len(self.ids), 20+len(self.vocab)+1),dtype=np.float32)
        observed = np.isfinite(self.numeric)
        table[:,:10] = np.where(observed, np.clip((self.numeric-self.mean)/self.scale,-5,5), 0)
        table[:,10:20] = observed
        for i,text in enumerate(self.tct):
            if text:
                table[i,20+(self.vocab.index(text)+1 if text in self.vocab else 0)] = 1
        g = np.array([r['genes'] for r in self.samples],dtype=np.float32); gm = np.isfinite(g)
        self.genes = np.concatenate([np.nan_to_num(g),gm],1).astype(np.float32)
        self.table = table
        order = [list(photos['sample_ids']).index(s) for s in self.ids]
        self.photo = photos['features'][order].astype(np.float32)
        self.cells = cells
        self.present = np.stack([photos['image_present'][order], gm.any(1), observed.any(1) | np.array([bool(t) for t in self.tct]),
                                 np.array([bool(r['cells']) and use_cells for r in self.samples])],1).astype(bool)
        self.common = self.present[:,:3].any(1)

    def eligible(self, ids):
        return [s for s in ids if self.present[self.index[s]].any()]

    def pack(self, ids, device, train=False):
        ix = np.array([self.index[s] for s in ids]); views = torch.randint(1,5,(len(ix),)).numpy() if train else np.zeros(len(ix),dtype=int)
        return {k:torch.as_tensor(v, device=device) for k,v in {'photo':self.photo[ix,views], 'methyl':self.genes[ix],
                'table':self.table[ix], 'cell':self.cells[ix], 'present':self.present[ix], 'target':self.y[ix]}.items()}


@torch.no_grad()
def predict(model, inputs, ids, device, fold):
    model.eval(); rows=[]
    for sid in ids:
        i=inputs.index[sid]
        p = model(inputs.pack([sid],device)).softmax(1)[0].cpu().tolist() if inputs.present[i].any() else None
        rows.append({'sample_id':sid, 'grade':int(inputs.grade[i]), 'fold':fold, 'probabilities':p,
                     'cell_present':bool(inputs.present[i,3]), 'common_evaluable':bool(inputs.common[i])})
    return rows


def fit(inputs, train_ids, val_ids, pretrained, c, initial_seed, epochs, device, folder):
    seed(initial_seed)
    model = Fusion(inputs.table.shape[1],c).to(device)
    if pretrained is not None:
        model.methyl.load_state_dict(pretrained,strict=True)
    methyl_ids={id(p) for p in model.methyl.parameters()}
    optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if id(p) not in methyl_ids], 'lr':c['lr']},
                                {'params':model.methyl.parameters(), 'lr':c['methylation_lr']}],weight_decay=c['weight_decay'])
    ids=inputs.eligible(train_ids); w=weights([inputs.y[inputs.index[s]] for s in ids],2,device)
    history=[]; best=(-1.,-float('inf')); selected=1; stale=0
    for epoch in range(1,epochs+1):
        model.methyl.requires_grad_(epoch>c['warmup_epochs'])
        model.train(); order=torch.randperm(len(ids)).tolist(); total=0.
        for start in range(0,len(ids),c['batch_size']):
            chunk=[ids[i] for i in order[start:start+c['batch_size']]]
            b=inputs.pack(chunk,device,True); loss=F.cross_entropy(model(b),b['target'],weight=w)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.); assert torch.isfinite(norm)
            optimizer.step();total+=float(loss.detach())*len(chunk)
        entry={'epoch':epoch,'train_loss':total/len(ids)}
        if val_ids is not None:
            rows=predict(model,inputs,val_ids,device,-1); m=score(rows)
            valid=[r for r in rows if r['probabilities'] is not None]
            p=torch.tensor([r['probabilities'] for r in valid],device=device)
            y=torch.tensor([int(r['grade']>0) for r in valid],device=device)
            ce=float(F.nll_loss(p.clamp_min(1e-8).log(),y,weight=w))
            key=(m['balanced_accuracy'],-ce);entry.update(validation=m,validation_loss=ce)
            # Select only after the methylation network has started fine-tuning.
            if epoch > c['warmup_epochs']:
                if key>best:best,selected,stale=key,epoch,0
                else:stale+=1
        else:selected=epoch
        history.append(entry)
        if val_ids is not None and stale>=c['fusion_patience'] and epoch>c['warmup_epochs']:break
    folder.mkdir(parents=True,exist_ok=True)
    write(folder/'training.json',{'history':history,'selected_epoch':selected,'training_ids':ids,'validation_ids':val_ids,
                                'normalization_mean':inputs.mean.tolist(),'normalization_scale':inputs.scale.tolist(),'tct_vocab':inputs.vocab})
    if val_ids is None:
        torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':c,
                    'table_size':inputs.table.shape[1], 'normalization_mean':inputs.mean.tolist(),
                    'normalization_scale':inputs.scale.tolist(),'tct_vocab':inputs.vocab,'training_ids':ids},folder/'classifier.pt')
    return model,selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    c=json.loads(a.config.read_text());bundle=json.loads((a.data/'bundle.json').read_text(encoding='utf-8'))
    # Refuse incomplete transfers before loading any model or starting training.
    assert sha(a.data/'photos.npz') == bundle['sources']['photos_sha256']
    for name in sorted({p for r in bundle['samples'] for p in r['cells']}):
        assert sha(a.data/name) == Path(name).stem, f'Incomplete or changed image: {name}'
    a.output.mkdir(parents=True,exist_ok=False);cache=a.output.parent.parent/'cache';cache.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    start=time.time()
    write(a.output/'configuration.json',{'config':c,'inputs':bundle['sources'],'bundle_sha256':sha(a.data/'bundle.json'),
        'sources':{Path(__file__).name:sha(Path(__file__)), 'sysu_binary_common.py':sha(Path(__file__).with_name('sysu_binary_common.py'))},
        'versions':{'torch':str(torch.__version__),'numpy':np.__version__}})
    encoder=pretrain(bundle,c,a.output,a.device)
    cells=cell_features(bundle,a.data,cache,a.device)
    photos=np.load(a.data/'photos.npz',allow_pickle=False)
    reports={}
    for with_cells in (False,True):
        for with_pretrain in (False,True):
            name=('cells' if with_cells else 'no_cells')+('_pretrained' if with_pretrain else '_random')
            all_rows=[]; folds=[]
            for f in bundle['folds']:
                n=f['fold'];folder=a.output/name/f'fold{n}'
                tr=f['inner_train']['sample_ids'];va=f['inner_val']['sample_ids'];outer=f['outer_train']['sample_ids'];held=f['outer_holdout']['sample_ids']
                inputs=Inputs(bundle,photos,cells,tr,with_cells)
                initial=encoder if with_pretrain else None
                _,epoch=fit(inputs,tr,va,initial,c,c['seed']+n,c['fusion_epochs'],a.device,folder/'inner')
                inputs=Inputs(bundle,photos,cells,outer,with_cells)
                model,_=fit(inputs,outer,None,initial,c,c['seed']+100+n,epoch,a.device,folder/'refit')
                rows=predict(model,inputs,held,a.device,n)
                restored=Fusion(inputs.table.shape[1],c).to(a.device)
                restored.load_state_dict(torch.load(folder/'refit/classifier.pt',weights_only=True,map_location='cpu')['state'],strict=True)
                again=predict(restored,inputs,held,a.device,n)
                error=max([abs(x-y) for r,s in zip(rows,again) if r['probabilities'] is not None for x,y in zip(r['probabilities'],s['probabilities'])] or [0.])
                assert error<1e-6
                # Missing branches must not use their hidden values.
                example=inputs.pack(inputs.eligible(held),a.device); model.eval()
                with torch.no_grad():
                    original=model(example); changed={k:v.clone() for k,v in example.items()}
                    for j,key in enumerate(('photo','methyl','table','cell')):changed[key][~changed['present'][:,j]]=999.
                    assert torch.allclose(original,model(changed),atol=1e-6)
                metrics=score(rows);folds.append({'fold':n,'selected_epoch':epoch,'metrics':metrics,'reload_error':error})
                write(folder/'predictions.json',{'rows':rows,**folds[-1]});all_rows.extend(rows)
                print(f'{name} fold{n} epoch={epoch} acc={metrics["accuracy"]:.4f}',flush=True)
            assert len(all_rows)==117 and len({r['sample_id'] for r in all_rows})==117
            reports[name]={'metrics':score(all_rows),'common_input_metrics':score([r for r in all_rows if r['common_evaluable']]),
                           'fold_mean_accuracy':float(np.mean([f['metrics']['accuracy'] for f in folds])),
                           'folds':folds,'rows':all_rows}
            write(a.output/name/'results.json',reports[name])
    write(a.output/'comparison.json',{'results':reports,'elapsed_seconds':time.time()-start,'scope':c['scope']})
    print('COMPLETED',flush=True)


if __name__=='__main__':main()
