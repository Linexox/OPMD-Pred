"""Four-category oral-cell ResNet18 pretraining with patient-held-out evaluation."""
import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as TF

from sysu_binary_common import sha, write


class Crops(Dataset):
    def __init__(self, data, records, augment=False):
        self.tar=data/'crops.tar';self.records=records;self.augment=augment;self.stream=None

    def __len__(self):return len(self.records)

    def __getitem__(self, index):
        if self.stream is None:self.stream=self.tar.open('rb')
        r=self.records[index];self.stream.seek(r['offset']);raw=self.stream.read(r['size'])
        with Image.open(io.BytesIO(raw)) as im:
            im=im.convert('RGB')
            if self.augment:
                if torch.rand(())<.5:im=TF.hflip(im)
                if torch.rand(())<.5:im=TF.vflip(im)
                im=im.rotate(90*int(torch.randint(4,()).item()))
                im=TF.adjust_brightness(im,float(.9+.2*torch.rand(())))
                im=TF.adjust_contrast(im,float(.9+.2*torch.rand(())))
            im=im.resize((224,224),Image.Resampling.BILINEAR)
            tensor=TF.normalize(TF.to_tensor(im),[.485,.456,.406],[.229,.224,.225])
        return tensor,r['class'],index


class PatientClassSampler(Sampler):
    def __init__(self, records, count, seed):
        self.buckets=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(records):self.buckets[r['class']][r['patient_id']].append(i)
        self.patients={k:sorted(v) for k,v in self.buckets.items()}
        assert set(self.patients)==set(range(4))
        self.count=count;self.seed=seed;self.epoch=0

    def __len__(self):return self.count

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch*100003)
        for _ in range(self.count):
            category=int(rng.integers(4));patients=self.patients[category]
            patient=patients[int(rng.integers(len(patients)))];items=self.buckets[category][patient]
            yield items[int(rng.integers(len(items)))]


def metrics(target, prediction):
    cm=np.zeros((4,4),dtype=int)
    for y,p in zip(target,prediction):cm[int(y),int(p)]+=1
    recall=[float(cm[k,k]/cm[k].sum()) if cm[k].sum() else None for k in range(4)]
    return {'accuracy':float(np.trace(cm)/cm.sum()),'balanced_accuracy':float(np.mean([v for v in recall if v is not None])),
            'recalls':recall,'confusion':cm.tolist(),'count':int(cm.sum())}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval();targets=[];probabilities=[];indices=[];loss=0.
    for x,y,index in loader:
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        logits=model(x);loss+=float(F.cross_entropy(logits,y,reduction='sum'))
        probabilities.extend(logits.softmax(1).cpu().tolist());targets.extend(y.cpu().tolist());indices.extend(index.tolist())
    p=np.array(probabilities);m=metrics(targets,p.argmax(1));m['loss']=loss/len(targets)
    return m,[{'index':int(i),'target':int(y),'probabilities':v} for i,y,v in zip(indices,targets,probabilities)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();c=json.loads(a.config.read_text());manifest=json.loads((a.data/'manifest.json').read_text())
    assert sha(a.data/'crops.tar')==manifest['tar_sha256'],'Incomplete training data transfer'
    groups={s:{r['patient_id'] for r in manifest['samples'] if r['split']==s} for s in ['train','validation','test']}
    assert not groups['train']&groups['test'] and not groups['train']&groups['validation'] and not groups['validation']&groups['test']
    records={s:[r for r in manifest['samples'] if r['split']==s] for s in groups}
    identity={'config':c,'manifest_sha256':sha(a.data/'manifest.json'),'split_sha256':sha(a.data/'patient_split.json'),
              'source_sha256':sha(Path(__file__)),'patients':{k:sorted(v) for k,v in groups.items()}}
    a.output.mkdir(parents=True,exist_ok=False);a.cache.mkdir(parents=True,exist_ok=True);torch.hub.set_dir(str(a.cache/'torch'))
    torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    datasets={s:Crops(a.data,r,s=='train') for s,r in records.items()}
    sampler=PatientClassSampler(records['train'],c['samples_per_epoch'],c['seed'])
    loaders={s:DataLoader(ds,batch_size=c['batch_size'],sampler=sampler if s=='train' else None,
                         num_workers=c['workers'],pin_memory=True,persistent_workers=c['workers']>0,
                         generator=torch.Generator().manual_seed(c['seed'])) for s,ds in datasets.items()}
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);model.fc=nn.Linear(512,4);model.to(a.device)
    optimizer=torch.optim.AdamW([{'params':[v for k,v in model.named_parameters() if not k.startswith('fc.')],'lr':c['backbone_lr']},
                                {'params':model.fc.parameters(),'lr':c['classifier_lr']}],weight_decay=c['weight_decay'])
    initial_conv=model.conv1.weight.detach().cpu().clone();best=(-1.,-float('inf'));stale=0;history=[];best_epoch=0
    write(a.output/'identity.json',identity)
    for epoch in range(1,c['epochs']+1):
        sampler.epoch=epoch;warmup=epoch<=c['warmup_epochs']
        for name,param in model.named_parameters():param.requires_grad_(not warmup or name.startswith('fc.'))
        model.train()
        if warmup:
            model.eval();model.fc.train()
        total=0.;correct=0;count=0
        for x,y,_ in loaders['train']:
            x=x.to(a.device,non_blocking=True);y=y.to(a.device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda')):
                logits=model(x);loss=F.cross_entropy(logits,y)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            assert torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(),1.))
            optimizer.step();total+=float(loss.detach())*len(y);correct+=int((logits.argmax(1)==y).sum());count+=len(y)
        validation,_=evaluate(model,loaders['validation'],a.device)
        entry={'epoch':epoch,'training_loss':total/count,'training_accuracy_balanced_sampling':correct/count,'validation':validation}
        history.append(entry);key=(validation['balanced_accuracy'],-validation['loss'])
        if not warmup:
            if key>best:
                best=key;best_epoch=epoch;stale=0
                torch.save({'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'identity':identity,
                            'epoch':epoch,'validation':validation,'labels':manifest['labels']},a.output/'best.pt')
            else:stale+=1
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'identity':identity,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},a.output/'last.pt')
        write(a.output/'history.json',history)
        print(f'Cell pretraining epoch={epoch} train_loss={total/count:.4f} val_acc={validation["accuracy"]:.4f} val_macro_recall={validation["balanced_accuracy"]:.4f}',flush=True)
        if not warmup and stale>=c['patience']:break
    best_state=torch.load(a.output/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(best_state['model_state'],strict=True)
    assert float((model.conv1.weight.detach().cpu()-initial_conv).abs().max())>0,'Backbone did not train'
    validation,val_rows=evaluate(model,loaders['validation'],a.device)
    test,test_rows=evaluate(model,loaders['test'],a.device)
    def attach(rs,split):
        return [{'patient_id':records[split][r['index']]['patient_id'],'slide_id':records[split][r['index']]['slide_id'],
                 'member':records[split][r['index']]['member'],'stain':records[split][r['index']]['stain'],**r} for r in rs]
    val_rows=attach(val_rows,'validation');test_rows=attach(test_rows,'test')
    by_stain={}
    for stain in ['PAP','MGG']:
        chosen=[r for r in test_rows if r['stain']==stain]
        if chosen:by_stain[stain]=metrics([r['target'] for r in chosen],[np.argmax(r['probabilities']) for r in chosen])
    write(a.output/'validation_predictions.json',val_rows);write(a.output/'test_predictions.json',test_rows)
    write(a.output/'results.json',{'identity':identity,'best_epoch':best_epoch,'last_epoch':len(history),
        'validation':validation,'test':test,'test_by_stain':by_stain,'best_checkpoint_sha256':sha(a.output/'best.pt'),
        'backbone_trained':True,'note':'Patient-held-out crop classification; no SYSU inputs used; test not used for model selection.'})
    print('CELL PRETRAINING COMPLETED '+json.dumps({'best_epoch':best_epoch,'test':test}),flush=True)


if __name__=='__main__':main()
