"""Paired BSDA-inspired image-token augmentation with matched image-supervision control."""
import argparse
import csv
import gc
import json
import math
import os
import subprocess
import tempfile
import importlib.metadata
import importlib.util
from pathlib import Path

import numpy as np
import torch

from opmd_pred.analysis.evaluation import classification_metrics
from opmd_pred.data.sysu import MODALITIES, _value_mask, load_sysu_records
from opmd_pred.data.sysu_protocol import file_sha256
from opmd_pred.scripts.common import set_seed
from opmd_pred.scripts.train_multimodal import apply_modality_dropout
from opmd_pred.scripts.train_sysu import model_from_config as base_model_from_config
from opmd_pred.model.fusion import NumericEncoder
from bsda_image_token import ImageTokenBSDA, training_objective, POLICY

ENCODER_SHA = '58a3fcc724155c8ecfc7bf0e955b2bf022f0e2aa4bfed06a51a5b99c850a4a65'



def model_from_config(config):
    require(config.get('dna_input_size') == 2, 'Use the explicit threshold-aware classifier config')
    model = base_model_from_config(config)
    model.fusion.dna_encoder = NumericEncoder(2, config['dimension'])
    require(config.get('bsda_policy')==POLICY, 'Semantic augmentation config differs')
    model.image_bsda=ImageTokenBSDA(config['dimension'])
    return model


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, data):
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp',
                                     mode='w', encoding='utf-8', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_torch(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_classifier(path, trained):
    require(file_sha256(path)==trained['classifier_sha256'], 'Classifier bytes differ from committed stage')
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    require(artifact['format']=='sysu-frozen-feature-cv-classifier-v1', 'Wrong classifier format')
    for key in ('config','fit','receipt','stage_identity','image_encoder_sha256'):
        require(artifact[key]==trained[key], f'Classifier {key} differs from stage')
    require(artifact['epoch']==trained['selected_epoch'], 'Wrong classifier epoch')
    return artifact


def check_oof(saved, fold, recipe, held, data, receipt, trained):
    require(saved['receipt']==receipt and saved['fold']==fold and saved['recipe']==recipe, 'Wrong OOF identity')
    require(saved['classifier_sha256']==trained['classifier_sha256'] and
            saved['selected_epoch']==trained['selected_epoch'], 'OOF classifier differs from committed stage')
    require([row['sample_id'] for row in saved['records']]==held, 'Wrong OOF held-out IDs or order')
    for row in saved['records']:
        sid = row['sample_id']
        eligible = bool(data.eligible[data.index[sid]])
        require(row['label']==data.records[sid].label, 'Wrong OOF label')
        require(row['status']==('predicted' if eligible else 'abstain'), 'Wrong OOF coverage')
        if eligible:
            p = np.asarray(row['probabilities'], dtype=np.float64)
            require(p.shape==(3,) and np.isfinite(p).all() and (p>=0).all() and
                    math.isclose(float(p.sum()),1.,abs_tol=1e-6), 'Invalid OOF probabilities')
        else:
            require(row['probabilities'] is None and row['reason']=='no_selected_input', 'Invalid abstention')


def median(p):
    return (p[:, 1:].sum(1) > .5).long() + (p[:, 2] > .5).long()


def score(p, labels, assigned_labels):
    results = {}
    for rule, predictions in (('fixed_median_0.5', median(p)), ('argmax', p.argmax(1))):
        m = classification_metrics(predictions, labels)
        cm = m['confusion_matrix']
        support = [assigned_labels.count(y) for y in range(3)]
        correct = sum(cm[y][y] for y in range(3))
        m.update(correct_count=correct, assigned_count=len(assigned_labels), eligible_count=len(labels),
                 abstained_count=len(assigned_labels)-len(labels), coverage=len(labels)/len(assigned_labels),
                 accuracy_all_assigned=correct/len(assigned_labels),
                 class_recall_all_assigned=[cm[y][y]/support[y] if support[y] else None for y in range(3)],
                 class_abstained_count=[support[y]-m['support'][y] for y in range(3)],
                 two_level_error_count=int((predictions-labels).abs().eq(2).sum()),
                 mae_scope='eligible predictions only; abstention distance undefined')
        m['severe_recall_all_assigned'] = m['class_recall_all_assigned'][2]
        results[rule] = m
    return results


class Data:
    def __init__(self, csv_path, protocol, cache):
        self.protocol = protocol
        self.ids = [e['sample_id'] for e in protocol['development_entries']]
        self.index = {sid:i for i,sid in enumerate(self.ids)}
        self.records = {r.sample_id:r for r in load_sysu_records(csv_path) if r.sample_id in self.index}
        require(len(self.records)==protocol['development_count'], 'Development record mismatch')
        self.images = {}
        for recipe in protocol['recipes']:
            with np.load(cache/f'{recipe}.npz', allow_pickle=False) as loaded:
                require(loaded['sample_ids'].tolist()==self.ids, 'Wrong cache order')
                self.images[recipe] = torch.from_numpy(loaded['features'].copy())
                mask = torch.from_numpy(loaded['image_present'].copy())
                if hasattr(self,'present'):
                    require(torch.equal(self.present,mask), 'Different modality coverage across recipes')
                self.present = mask
                require(loaded['labels'].tolist()==[self.records[s].label for s in self.ids], 'Wrong cached labels')
            require(tuple(self.images[recipe].shape)==(len(self.ids),5,1536) and torch.isfinite(self.images[recipe]).all(), 'Invalid cache tensor')
        require(torch.equal(self.images['identity_image_repeat'][:,0],self.images['bsda_image'][:,0]), 'Different evaluation features')
        with csv_path.open(encoding='utf-8-sig',newline='') as stream:
            extra = {r['sample_id']:r for r in csv.DictReader(stream)}
        thresholds = {sid:float(extra[sid]['dna_threshold']) if extra[sid]['dna_threshold'] else None for sid in self.ids}
        self.base = {k:[] for k in ('history','history_mask','dna','dna_mask','methylation','methylation_mask','label')}
        for sid in self.ids:
            r = self.records[sid]
            history, mask = _value_mask([r.smoking,r.smoking_years,r.smoking_amount,r.alcohol,r.alcohol_years,
                r.alcohol_amount,r.betel,r.betel_years,r.betel_amount],[1,50,100,1,50,1000,1,20,50])
            threshold = thresholds[sid]
            require((threshold is None)==(r.dna is None) and (threshold is None or threshold in {2.3,2.5}), 'Wrong DNA definition')
            dna, dna_mask = _value_mask([r.dna, None if threshold is None else float(threshold==2.5)],[1,1])
            methylation, methylation_mask = _value_mask([r.methylation],[10])
            for key,value in zip(self.base,(history,mask,dna,dna_mask,methylation,methylation_mask,torch.tensor(r.label))):
                self.base[key].append(value)
        self.base = {key:torch.stack(values) for key,values in self.base.items()}
        self.eligible = self.present | self.base['history_mask'].any(1) | self.base['dna_mask'].any(1) | self.base['methylation_mask'].any(1)
        self.eligible |= torch.tensor([bool(self.records[s].tct) for s in self.ids])
        require(int(self.eligible.sum())==protocol['development_eligible_count'], 'Wrong no-input coverage')

    def indices(self, ids, eligible=False):
        require(set(ids)<=set(self.index), 'Old test or unknown ID reached the runner')
        ix = torch.tensor([self.index[s] for s in ids])
        return ix[self.eligible[ix]] if eligible else ix

    def fit(self, train_ids, recipe):
        ix = self.indices(train_ids, True)
        ids = [self.ids[i] for i in ix.tolist()]
        counts = [sum(self.records[s].label==y for s in ids) for y in range(3)]
        require(all(counts), 'Training fold lacks a class')
        vocab = {text:i for i,text in enumerate(sorted({self.records[s].tct for s in ids if self.records[s].tct}),start=2)}
        fit_ids = [s for s in ids if self.records[s].dna is not None]
        log = np.log1p(np.asarray([self.records[s].dna for s in fit_ids],dtype=np.float64)*10000)
        require(len(log)>0 and np.isfinite(log).all(), 'Invalid DNA training values')
        std = float(log.std())
        return {'recipe':recipe,'fit_ids':ids,'dna_fit_ids':fit_ids,'dna_log_mean':float(log.mean()),
                'dna_log_population_std':std if std>0 else 1.,'dna_constant':std==0,
                'tct_vocab':vocab,'class_counts':counts,'class_weights':[len(ids)/(3*n) for n in counts]}

    def pack(self, indices, fit, training):
        batch = {key:value[indices].clone().to('cuda') for key,value in self.base.items()}
        views = torch.randint(1,5,(len(indices),)) if training else torch.zeros(len(indices),dtype=torch.long)
        batch['image_features'] = self.images[fit['recipe']][indices,views].to('cuda')
        batch['image_present'] = self.present[indices].to('cuda')
        ids = [self.ids[i] for i in indices.tolist()]
        batch['tct'] = torch.tensor([fit['tct_vocab'].get(self.records[s].tct,1) if self.records[s].tct else 0 for s in ids],device='cuda')
        batch['tct_present'] = torch.tensor([bool(self.records[s].tct) for s in ids],device='cuda')
        batch['sample_id'] = ids
        return batch


def logits(model, batch):
    fusion = model.fusion
    embeddings = {'image':fusion.image_projection(batch['image_features']),'tct':fusion.tct_encoder(batch['tct'])}
    present = {'image':batch['image_present'],'tct':batch['tct_present']}
    for name in ('history','dna','methylation'):
        embeddings[name] = getattr(fusion,f'{name}_encoder')(batch[name],batch[f'{name}_mask'])
        present[name] = batch[f'{name}_mask'].any(1)
    return model.forward_from_embeddings(embeddings,present)


@torch.no_grad()
def evaluate(model, data, ids, fit):
    model.eval()
    ix = data.indices(ids, True)
    probabilities = []
    for chunk in ix.split(8):
        probabilities.append(logits(model,data.pack(chunk,fit,False)).softmax(1).cpu())
    p = torch.cat(probabilities)
    labels = data.base['label'][ix]
    require(torch.isfinite(p).all(), 'Nonfinite validation predictions')
    return p,score(p,labels,[data.records[s].label for s in ids])


def stage(data, role, folder, recipe, seed, max_epochs, select_epoch, receipt, fold):
    folder.mkdir(parents=True,exist_ok=True)
    ids = role['sample_ids']
    fit = data.fit(ids,recipe)
    require(set(fit['fit_ids'])<=set(ids) and set(fit['dna_fit_ids'])<=set(ids), 'Invalid fit scope')
    expected = role['candidate_dna_fit_statistics']
    require(set(fit['dna_fit_ids'])==set(expected['fit_sample_ids']), 'DNA fit ID differs from protocol')
    require(math.isclose(fit['dna_log_mean'],expected['log_mean'],abs_tol=1e-12), 'DNA mean differs from protocol')
    require(len(fit['dna_fit_ids'])==expected['count'] and fit['dna_constant']==expected['constant_input'], 'Wrong DNA count or constant flag')
    require(math.isclose(fit['dna_log_population_std'],expected['log_population_std'] or 1.,abs_tol=1e-12), 'Wrong DNA scale')
    require(fit['fit_ids']==[data.ids[i] for i in data.indices(ids,True).tolist()] and
            len(fit['fit_ids'])==role['eligible_count'] and fit['class_counts']==role['eligible_class_support'], 'Wrong eligible training scope')
    identity = {'fold':fold,'role':'inner' if select_epoch is not None else 'refit','recipe':recipe,
                'seed':seed,'max_epochs':max_epochs,'training_ids':ids,'validation_ids':select_epoch}
    config = {'dimension':128,'tct_categories':max(fit['tct_vocab'].values(),default=1),
              'image_encoder':'dinov2_b14','modalities':list(MODALITIES),'head':'softmax','fusion':'transformer','dna_input_size':2,'bsda_policy':POLICY}
    done = folder/'done.json'
    if done.exists():
        record = json.loads(done.read_text())
        require(record['receipt']==receipt and record['fit']==fit and record['config']==config and
                record['stage_identity']==identity, 'Cannot reuse unrelated stage')
        if select_epoch is None:
            check_classifier(folder/'classifier.pt',record)
        return record
    own = os.getpid()
    gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
    require(not [line for line in gpu.splitlines() if line.strip() and int(line.split(',')[0].strip())!=own], 'Other GPU job present')
    set_seed(seed)
    model = model_from_config(config)
    model.fusion.freeze_image()
    model.fusion.image_encoder = None
    gc.collect()
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    weights = torch.tensor(fit['class_weights'],device='cuda')
    history,best_key,best_epoch,stale = [],(-1.,-1.),None,0
    last = folder/'last.pt'
    start = 1
    if last.exists():
        saved = torch.load(last,map_location='cuda',weights_only=True)
        require(saved['format']=='sysu-frozen-feature-cv-training-state-v1' and saved['receipt']==receipt and
                saved['fit']==fit and saved['config']==config and saved['stage_identity']==identity, 'Wrong resume state')
        model.load_state_dict(saved['state'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        history,best_key,best_epoch,stale = saved['history'],tuple(saved['best_key']),saved['best_epoch'],saved['stale']
        start = saved['epoch']+1
        torch.set_rng_state(saved['cpu_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    train_ix = data.indices(ids,True)
    for epoch in range(start,max_epochs+1):
        if select_epoch is not None and stale>=5:
            break
        model.train()
        total = 0.
        shuffled = train_ix[torch.randperm(len(train_ix))]
        for chunk in shuffled.split(8):
            batch = data.pack(chunk,fit,True)
            batch = apply_modality_dropout(batch,.15,'bernoulli',model.fusion.names)
            loss = training_objective(model,batch,weights,recipe,epoch)
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(chunk)
        item = {'epoch':epoch,'training_batch_weighted_loss':total/len(train_ix)}
        if select_epoch is not None:
            _,m = evaluate(model,data,select_epoch,fit)
            item['inner_validation'] = m
            primary = m['fixed_median_0.5']
            key = (primary['accuracy_all_assigned'],primary['macro_f1'])
            if key>best_key:
                best_key,best_epoch,stale = key,epoch,0
            else:
                stale+=1
        else:
            best_epoch=epoch
        history.append(item)
        save_torch(last,{'format':'sysu-frozen-feature-cv-training-state-v1','state':model.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch,'fit':fit,'config':config,'receipt':receipt,'history':history,'best_key':best_key,
            'best_epoch':best_epoch,'stale':stale,'stage_identity':identity,
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
    require(best_epoch is not None,'No trained epoch')
    result = {'fit':fit,'config':config,'receipt':receipt,'seed':seed,'selected_epoch':best_epoch,'history':history,
              'artifact_scope':'frozen-feature classifier; image encoder externally pinned; not legacy raw-image checkpoint',
              'fresh_SYSU_fusion_init':True,'image_encoder_sha256':ENCODER_SHA,'stage_identity':identity}
    if select_epoch is None:
        save_torch(folder/'classifier.pt',{'format':'sysu-frozen-feature-cv-classifier-v1','state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'config':config,'fit':fit,'receipt':receipt,'epoch':best_epoch,
                    'stage_identity':identity,'image_encoder_sha256':ENCODER_SHA})
        result['classifier_sha256'] = file_sha256(folder/'classifier.pt')
    write_json(done,result)
    del optimizer,model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('csv','cv-protocol','cache','output'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    require(args.output.resolve().is_relative_to(Path('/home/zjkj/24334067')), 'Output outside personal directory')
    protocol = json.loads(args.cv_protocol.read_text())
    metadata = json.loads((args.cache/'metadata.json').read_text())
    require(protocol['format']=='sysu-raw-rebuilt-development-cv-v3','Use reviewed v2 protocol')
    require(file_sha256(args.csv)==protocol['csv_sha256']==metadata['csv_sha256'],'CSV fingerprint mismatch')
    require(file_sha256(args.cv_protocol)==metadata['cv_protocol_sha256'],'CV fingerprint mismatch')
    require(metadata['format']=='sysu-frozen-image-semantic-pair-v1','Wrong cache type')
    for recipe in protocol['recipes']:
        require(file_sha256(args.cache/f'{recipe}.npz')==metadata['cache_sha256'][recipe], 'Cache fingerprint mismatch')
    require(metadata['encoder_sha256']==ENCODER_SHA and metadata['evaluation_view']==0 and metadata['training_views']==[1,2,3,4], 'Wrong encoder or view roles')
    require(metadata['source_image_sha_by_id']=={e['sample_id']:e['image_sha256'] for e in protocol['development_entries']},'Image identities differ from protocol')
    require(not set(e['sample_id'] for e in protocol['development_entries'])&set(protocol['locked_previous_test_ids']+protocol['additional_related_locked_ids']),'Old test leaked into development')
    data = Data(args.csv,protocol,args.cache)
    args.output.mkdir(parents=True,exist_ok=True)
    receipt = {'csv_sha256':file_sha256(args.csv),'cv_sha256':file_sha256(args.cv_protocol),
        'cache_sha256':metadata['cache_sha256'],'DNA_features':['raw_report_ratio','DI_threshold_2.3_or_2.5'], 'augmentation_policy':metadata['policy'],'bsda_module_sha256':file_sha256(Path(__file__).with_name('bsda_image_token.py')),'encoder_sha256':ENCODER_SHA,'script_sha256':file_sha256(__file__),
        'recipes':protocol['recipes'],'primary_rule':'fixed_median_0.5','max_inner_epochs':50,'patience':5,
        'per_fold_init_seeds':'inner42+fold; refit142+fold; paired recipes use identical seeds',
        'previous_test_inference':False,'SYSU_fusion_weights_reused':False,
        'code_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'packages':{name:importlib.metadata.version(name) for name in ('torch','timm','numpy')},
        'source_sha256':{name:file_sha256(importlib.util.find_spec(name).origin) for name in (
            'opmd_pred.data.sysu','opmd_pred.data.sysu_protocol','opmd_pred.scripts.common',
            'opmd_pred.scripts.train_multimodal','opmd_pred.scripts.train_sysu','opmd_pred.model.fusion',
            'opmd_pred.model.vit','opmd_pred.model.ordinal','opmd_pred.analysis.evaluation')}}
    freeze = args.output/'freeze.json'
    if freeze.exists():
        require(json.loads(freeze.read_text())==receipt,'Do not change a started CV protocol')
    else:
        write_json(freeze,receipt)
    oof = {name:[] for name in receipt['recipes']}
    for fold in protocol['folds']:
        train,held = fold['outer_train']['sample_ids'],fold['outer_holdout']['sample_ids']
        require(not set(train)&set(held),'Outer role overlap')
        require(set(train)|set(held)==set(data.ids),'Outer partition incomplete')
        inner_train,inner_val=fold['inner_train']['sample_ids'],fold['inner_val']['sample_ids']
        require(not set(inner_train)&set(inner_val) and set(inner_train)|set(inner_val)==set(train),'Inner role overlap or incomplete partition')
        train_groups={e['cv_component'] for e in protocol['development_entries'] if e['sample_id'] in train}
        held_groups={e['cv_component'] for e in protocol['development_entries'] if e['sample_id'] in held}
        require(not train_groups&held_groups,'Cross-group overlap')
        for left,right in ((train,held),(inner_train,inner_val)):
            left_entries=[e for e in protocol['development_entries'] if e['sample_id'] in left]
            right_entries=[e for e in protocol['development_entries'] if e['sample_id'] in right]
            require(not {e['cv_component'] for e in left_entries}&{e['cv_component'] for e in right_entries},'Inner/outer component overlap')
            require(not {e['image_sha256'] for e in left_entries if e['image_sha256']}&
                        {e['image_sha256'] for e in right_entries if e['image_sha256']},'Inner/outer image overlap')
        for recipe in receipt['recipes']:
            path=args.output/f"fold{fold['fold']}"/recipe
            final=path/'outer_predictions.json'
            inner=stage(data,fold['inner_train'],path/'inner',recipe,42+fold['fold'],50,fold['inner_val']['sample_ids'],receipt,fold['fold'])
            trained=stage(data,fold['outer_train'],path/'refit',recipe,142+fold['fold'],inner['selected_epoch'],None,receipt,fold['fold'])
            if final.exists():
                saved=json.loads(final.read_text())
                check_oof(saved,fold['fold'],recipe,held,data,receipt,trained)
                oof[recipe].extend(saved['records']); continue
            artifact=check_classifier(path/'refit/classifier.pt',trained)
            model=model_from_config(trained['config']); model.fusion.freeze_image(); model.fusion.image_encoder=None
            model.load_state_dict(artifact['state'],strict=True); model.to('cuda')
            p,m=evaluate(model,data,held,trained['fit'])
            eligible=data.indices(held,True); by_id={data.ids[i]:probs for i,probs in zip(eligible.tolist(),p.tolist())}
            rows=[{'sample_id':sid,'label':data.records[sid].label,'probabilities':by_id.get(sid),
                   'status':'predicted' if sid in by_id else 'abstain','reason':None if sid in by_id else 'no_selected_input'} for sid in held]
            saved={'fold':fold['fold'],'recipe':recipe,'selected_epoch':inner['selected_epoch'],'metrics':m,
                   'records':rows,'receipt':receipt,'classifier_sha256':trained['classifier_sha256']}
            check_oof(saved,fold['fold'],recipe,held,data,receipt,trained)
            write_json(final,saved); oof[recipe].extend(rows)
            print(json.dumps({'fold':fold['fold'],'recipe':recipe,'selected_epoch':inner['selected_epoch'],
                              'metrics':m['fixed_median_0.5']}),flush=True)
            del model;torch.cuda.empty_cache()
    summary={}
    for recipe,rows in oof.items():
        require(len(rows)==len({r['sample_id'] for r in rows})==len(data.ids),'Incomplete or duplicated OOF')
        require({r['sample_id'] for r in rows}==set(data.ids),'OOF scope mismatch')
        chosen=[r for r in rows if r['status']=='predicted']
        m=score(torch.tensor([r['probabilities'] for r in chosen]),torch.tensor([r['label'] for r in chosen]),[r['label'] for r in rows])
        summary[recipe]={'metrics':m,'records':rows}
    write_json(args.output/'oof_results.json',{'receipt':receipt,'results':summary,'selection_bias':'development OOF; external new cases still required'})
    print(json.dumps({'completed':True,'summary':{k:v['metrics']['fixed_median_0.5'] for k,v in summary.items()}}),flush=True)


if __name__=='__main__':
    main()
