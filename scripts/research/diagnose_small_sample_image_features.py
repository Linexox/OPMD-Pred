"""Image-only frozen-feature OOF diagnosis, with training-fold-only normalization."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from opmd_pred.data.sysu_protocol import file_sha256


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def metrics(rows):
    cm = np.zeros((3,3),dtype=int)
    for r in rows:
        cm[r['label'],r['prediction']] += 1
    support = cm.sum(1)
    return {'records':len(rows),'correct':int(np.trace(cm)),'accuracy':float(np.trace(cm)/len(rows)),
            'confusion_matrix':cm.tolist(),'recall':[float(cm[i,i]/support[i]) if support[i] else None for i in range(3)],
            'severe_to_mild':int(cm[2,0]),'cross_two_levels':int(cm[0,2]+cm[2,0]),
            'mean_error_distance':float(sum(abs(r['label']-r['prediction']) for r in rows)/len(rows))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite diagnosis')
    protocol = json.loads((args.input/'development_nested_cv_v3.json').read_text())
    metadata = json.loads((args.cache/'metadata.json').read_text())
    assert metadata['cv_protocol_sha256']==file_sha256(args.input/'development_nested_cv_v3.json')
    assert file_sha256(args.cache/'baseline.npz')==metadata['cache_sha256']['baseline']
    with np.load(args.cache/'baseline.npz',allow_pickle=False) as loaded:
        ids = loaded['sample_ids'].tolist()
        features = loaded['features'][:,0].copy()
        present = loaded['image_present'].copy()
        labels = loaded['labels'].copy()
    index = {s:i for i,s in enumerate(ids)}
    results = {name:[] for name in ('centered_normalized_prototypes','centered_normalized_1NN','linear_probe_C1_balanced')}
    fit_records = []
    for fold in protocol['folds']:
        train = [index[s] for s in fold['outer_train']['sample_ids'] if present[index[s]]]
        held = [index[s] for s in fold['outer_holdout']['sample_ids'] if present[index[s]]]
        assert set(labels[train])=={0,1,2}
        center = features[train].mean(0,keepdims=True)
        x,z = normalize(features[train]-center),normalize(features[held]-center)
        y = labels[train]
        prototypes = np.stack([x[y==k].mean(0) for k in range(3)])
        distances = ((z[:,None]-prototypes[None])**2).sum(2)
        nearest_distances = ((z[:,None]-x[None])**2).sum(2)
        lr = LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=42).fit(x,y)
        predictions = {'centered_normalized_prototypes':distances.argmin(1),
                       'centered_normalized_1NN':y[nearest_distances.argmin(1)],
                       'linear_probe_C1_balanced':lr.predict(z)}
        for name,preds in predictions.items():
            results[name].extend({'sample_id':ids[i],'label':int(labels[i]),'prediction':int(pred), 'fold':fold['fold']}
                                 for i,pred in zip(held,preds))
        fit_records.append({'fold':fold['fold'],'normalization_fit_ids':[ids[i] for i in train],
                            'held_image_ids':[ids[i] for i in held], 'class_counts':[int((y==k).sum()) for k in range(3)]})
    for rows in results.values():
        assert len(rows)==len({r['sample_id'] for r in rows})==int(present.sum())
    payload = {'scope':'only identical image-present development OOF records; not full multimodal accuracy',
               'csv_sha256':protocol['csv_sha256'],'cv_sha256':metadata['cv_protocol_sha256'],
               'script_sha256':file_sha256(Path(__file__)),'fits':fit_records,
               'methods':{k:{'metrics':metrics(rows),'records':rows} for k,rows in results.items()},
               'paper':'https://arxiv.org/abs/1911.04623v2','protocol_adaptation_not_fewshot_paper_replication':True,
               'old_test_inference':False}
    args.output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({k:v['metrics'] for k,v in payload['methods'].items()}),flush=True)


if __name__=='__main__':
    main()
