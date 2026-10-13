"""Freeze raw-rebuilt development CV, excluding the old test and related groups."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def present(row):
    return bool(row['image_path'] or row['TCT意见'] or row['DNA倍体分析结果'] or row['甲基化总数'] or
                any(row[k] for k in ('是否吸烟','吸烟年限（年）','吸烟量（支/天）','是否饮酒','饮酒年限(年)',
                                    '饮酒量(ml/天)','是否嚼槟榔','嚼槟榔年限(年)','嚼槟榔量（颗/天）')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    args = p.parse_args()
    root = args.input.resolve()
    output = root/'development_nested_cv_v3.json'
    if output.exists():
        raise FileExistsError('Never overwrite a frozen CV protocol')
    source = json.loads((root/'group_input.json').read_text())
    with (root/'sysu_samples.csv').open(encoding='utf-8-sig', newline='') as f:
        records = {r['sample_id']:r for r in csv.DictReader(f)}
    entries = source['records']
    ids_all = sorted(records)
    parent = {sid:sid for sid in ids_all}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owners = {}
    for e in entries:
        for kind in ('previous_source_group','candidate_patient_id','image_sha256','dna_report_sha256'):
            if e.get(kind):
                key = (kind,e[kind])
                previous = owners.setdefault(key,e['sample_id'])
                a,b = find(previous),find(e['sample_id'])
                if a != b:
                    parent[max(a,b)] = min(a,b)
    locked = set(source['locked_previous_test_ids'])
    locked_groups = {find(s) for s in locked}
    dev = [dict(e, cv_component=find(e['sample_id'])) for e in entries if find(e['sample_id']) not in locked_groups]
    newly_locked = sorted(set(ids_all)-locked-{e['sample_id'] for e in dev})
    ids = np.asarray([e['sample_id'] for e in dev])
    labels = np.asarray([e['label'] for e in dev])
    groups = np.asarray([e['cv_component'] for e in dev])
    def describe(ix, training=False):
        chosen = [dev[int(i)] for i in ix]
        observed = [e for e in chosen if present(records[e['sample_id']])]
        result = {'sample_ids':[e['sample_id'] for e in chosen], 'assigned_count':len(chosen),
            'eligible_count':len(observed), 'class_support':[sum(e['label']==y for e in chosen) for y in range(3)],
            'eligible_class_support':[sum(e['label']==y for e in observed) for y in range(3)]}
        if training:
            fit_ids = [e['sample_id'] for e in observed if records[e['sample_id']]['DNA倍体分析结果']]
            log = np.log1p(np.asarray([float(records[s]['DNA倍体分析结果']) for s in fit_ids])*10000)
            std = float(log.std())
            result['candidate_dna_fit_statistics'] = {'fit_sample_ids':fit_ids,'count':len(log),
                'log_mean':float(log.mean()),'log_population_std':std if std>0 else 1.,'constant_input':std==0}
        return result
    folds, seen = [],[]
    for fold,(train,held) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(ids,labels,groups)):
        it,iv = next(StratifiedGroupKFold(4,shuffle=True,random_state=1042+fold).split(ids[train],labels[train],groups[train]))
        it,iv = train[it],train[iv]
        for left,right in ((train,held),(it,iv)):
            if set(groups[left]) & set(groups[right]):
                raise ValueError('Cross-group overlap')
        item = {'fold':fold,'outer_train':describe(train,True),'outer_holdout':describe(held),
                'inner_train':describe(it,True),'inner_val':describe(iv)}
        if not all(item['inner_train']['eligible_class_support']):
            raise ValueError('Missing training class')
        folds.append(item)
        seen.extend(item['outer_holdout']['sample_ids'])
    if Counter(seen) != Counter(ids.tolist()) or set(ids)&locked:
        raise ValueError('Invalid OOF coverage')
    protocol = {'format':'sysu-raw-rebuilt-development-cv-v3','csv_sha256':sha(root/'sysu_samples.csv'),
        'group_input_sha256':sha(root/'group_input.json'),'script_sha256':sha(Path(__file__)),
        'development_count':len(dev),'development_eligible_count':sum(present(records[s]) for s in ids),
        'development_component_count':len(set(groups)), 'locked_previous_test_ids':sorted(locked),
        'additional_related_locked_ids':newly_locked, 'development_entries':dev,'folds':folds,
        'seed':42,'inner_rule':'first of four group-stratified folds seed1042+outer fold',
        'patient_identity_status':'conservative source-derived candidate groups; annotations removed for grouping only',
        'previous_test_new_inference':False,'recipes':['baseline','medical_aug'],
        'DNA_model_features':['report abnormal ratio','report DI threshold: 2.3=0, 2.5=1'],
        'primary_rule':'fixed_median_0.5','inner_max_epochs':50,'patience':5,
        'epoch_selection':'inner median accuracy_all then macro_f1','seeds':[42],
        'image_views':'1 common fixed evaluation + 4 training-only cached views, matching update budget'}
    output.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:protocol[k] for k in ('development_count','development_eligible_count','development_component_count',
                                           'additional_related_locked_ids')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
