"""Four-class full-photo training with matched two-view manual-ROI controls."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from opmd_pred.data.sysu_protocol import file_sha256
from zenodo_supplement_common import (ClassificationPhotos, PhotoClassifier, configure,
    evaluate_classifier, load_rows, read, save, write)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backbone", choices=["resnet18","vit_b16"], required=True)
    parser.add_argument("--class-weighting", choices=["none","balanced"], required=True)
    parser.add_argument("--views", choices=["global","global_pair","manual_roi"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = dict(read(args.config),backbone=args.backbone,class_weighting=args.class_weighting,views=args.views)
    configure(config["seed"])
    if args.output.exists() and not args.resume:
        raise ValueError("Output exists; use explicit resume with an identical configuration")
    args.output.mkdir(parents=True,exist_ok=True)
    rows = load_rows(config)
    train_labels = [r["label"] for r in rows if r["split"] == "train"]
    counts = [train_labels.count(i) for i in range(4)]
    weights = [len(train_labels)/(4*n) for n in counts] if args.class_weighting == "balanced" else [1.]*4
    config.update(class_counts=counts,class_weights=weights)
    config_path=args.output/"resolved_config.json"
    if config_path.exists() and read(config_path)!=config:
        raise ValueError("Resume configuration changed")
    write(config_path,config)
    write(args.output/"source_receipt.json",{"source_sha256":{str(p):file_sha256(p) for p in
        (Path(__file__),Path(__file__).with_name('zenodo_supplement_common.py'))},
        "records_sha256":file_sha256(config["records"]),"manifest_sha256":file_sha256(config["split_manifest"]),
        "packages":{n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','pillow')}})
    # Separate model initialization from loader/random view generation.
    model=PhotoClassifier(args.backbone,pretrained=not args.resume).to(config["device"])
    device=config["device"]
    if not args.resume:
        fingerprint=hashlib.sha256()
        for name, value in model.state_dict().items():
            fingerprint.update(name.encode());fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
        write(args.output/'initialization.json',{'tensor_state_sha256':fingerprint.hexdigest(),
                                                'backbone':args.backbone,'seed':config['seed']})
    datasets={s:ClassificationPhotos(config,rows,s,training=s=='train') for s in ('train','validation')}
    generator=torch.Generator().manual_seed(config["seed"])
    loaders={s:DataLoader(d,batch_size=config['batch_size'],shuffle=s=='train',num_workers=config['workers'],
                         pin_memory=device=='cuda',generator=generator if s=='train' else None) for s,d in datasets.items()}
    if args.check_only:
        batch, y, _=next(iter(loaders['train']))
        model.train()
        out=model(batch[:2,0].to(device))
        loss=torch.nn.functional.cross_entropy(out,y[:2].to(device))
        loss.backward()
        assert out.shape==(2,4) and bool(torch.isfinite(loss))
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        model.eval()
        with torch.no_grad(): before=model(batch[:2,0].to(device)).cpu()
        save(args.output/'check-model.pt',{'state':model.state_dict()})
        clone=PhotoClassifier(args.backbone).to(device)
        clone.load_state_dict(torch.load(args.output/'check-model.pt',weights_only=True,map_location=device)['state'],strict=True)
        clone.eval()
        with torch.no_grad(): after=clone(batch[:2,0].to(device)).cpu()
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        # Freeze a complete model copy to check all parameters and BatchNorm buffers.
        snapshot={k:v.clone() for k,v in clone.encoder.state_dict().items()}
        clone.encoder.requires_grad_(False);clone.encoder.eval()
        with torch.no_grad(): clone.encoder(batch[:2,0].to(device))
        assert all(torch.equal(v,clone.encoder.state_dict()[k]) for k,v in snapshot.items())
        write(args.output/'checks.json',{'passed':True,'four_class_gradients_finite':True,'reload_exact':True,
                                      'frozen_encoder_parameters_and_buffers_unchanged':True})
        print('checks passed',flush=True)
        return
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':config['encoder_lr']},
        {'params':model.classifier.parameters(),'lr':config['head_lr']}],weight_decay=config['weight_decay'])
    best_key,best_epoch,stale,history,start=(-1.,-1.),0,0,[],1
    if args.resume:
        state=torch.load(args.output/'last.pt',weights_only=True,map_location=device)
        assert state['config']==config
        model.load_state_dict(state['state'],strict=True);optimizer.load_state_dict(state['optimizer'])
        best_key,best_epoch,stale,history,start=tuple(state['best_key']),state['best_epoch'],state['stale'],state['history'],state['epoch']+1
        torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        generator.set_state(state['loader_rng'].cpu());random.setstate(state['python_rng'])
    else:
        configure(config['seed'])
    criterion=torch.tensor(weights,device=device)
    for epoch in range(start,config['epochs']+1):
        if stale>=config['patience']:break
        datasets['train'].epoch=epoch
        model.train(); total,count=0.,0;started=time.monotonic()
        for images,y,_ in loaders['train']:
            images,y=images.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            first=model(images[:,0])
            loss=torch.nn.functional.cross_entropy(first,y,weight=criterion)
            if images.shape[1]==2:
                second=model(images[:,1])
                loss=loss+config['auxiliary_classification_weight']*torch.nn.functional.cross_entropy(second,y,weight=criterion)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite classification loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(y);count+=len(y)
        ps,ys,indices,m=evaluate_classifier(model,loaders['validation'],device)
        item={'epoch':epoch,'training_loss':total/count,'validation':m,'seconds':time.monotonic()-started}
        history.append(item);key=(m['accuracy'],m['macro_f1'])
        if key>best_key:
            best_key,best_epoch,stale=key,epoch,0
            save(args.output/'best.pt',{'format':'zenodo-supplement-classifier-v1','config':config,
                'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'validation':m})
            prediction_rows=[{'image_name':rows[i]['image_name'],'label':y,
                'probabilities':p.tolist()} for p,y,i in zip(ps,ys,indices)]
            (args.output/'validation_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prediction_rows),encoding='utf-8')
            write(args.output/'validation_metrics.json',m)
        else:stale+=1
        with (args.output/'history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(item)+'\n')
        save(args.output/'last.pt',{'state':model.state_dict(),'optimizer':optimizer.state_dict(),'config':config,
            'epoch':epoch,'best_key':best_key,'best_epoch':best_epoch,'stale':stale,'history':history,
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'loader_rng':generator.get_state(),
            'python_rng':random.getstate()})
        print(json.dumps(item),flush=True)
    saved=torch.load(args.output/'best.pt',weights_only=True,map_location=device)
    model.load_state_dict(saved['state'],strict=True)
    ps,ys,indices,m=evaluate_classifier(model,loaders['validation'],device)
    assert m==saved['validation']
    old=[json.loads(line) for line in (args.output/'validation_predictions.jsonl').read_text().splitlines()]
    torch.testing.assert_close(ps,torch.tensor([r['probabilities'] for r in old]),rtol=0,atol=0)
    artifact={'format':'zenodo-supplement-encoder-v1','backbone':args.backbone,'feature_dim':model.feature_dim,
        'state':{k:v.detach().cpu() for k,v in model.encoder.state_dict().items()},'preprocessing':config['preprocessing'],
        'source_classifier_sha256':file_sha256(args.output/'best.pt')}
    save(args.output/'encoder.pt',artifact)
    write(args.output/'encoder_manifest.json',{k:v for k,v in artifact.items() if k!='state'})
    write(args.output/'done.json',{'completed':True,'best_epoch':best_epoch,'epochs_run':len(history),'validation':m,
        'reload_probabilities_exact':True,'encoder_sha256':file_sha256(args.output/'encoder.pt')})
    print(json.dumps({'completed':True,'best_epoch':best_epoch,'validation':m}),flush=True)


if __name__=='__main__':
    main()
