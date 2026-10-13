"""Collect completed classifiers/metadata and independently recount saved OOF."""
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

P=Path('/home/zjkj/24334067')
INPUT=P/'sysu-research-inputs/raw-rebuild-20261009'
OUTPUT=P/'OPMD-Pred-SYSU-Proxy-Balanced/outputs'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def recount(rows,expected):
    if len(rows)!=len({r['sample_id'] for r in rows}) or {r['sample_id'] for r in rows}!=set(expected):
        raise ValueError('OOF identity scope changed')
    cm=[[0]*3 for _ in range(3)]
    abstained=[0]*3
    for r in rows:
        if r['label']!=expected[r['sample_id']]:
            raise ValueError('OOF label mismatch')
        if r['status']=='abstain':
            assert r['probabilities'] is None
            abstained[r['label']]+=1
            continue
        probs=r['probabilities']
        assert len(probs)==3 and all(math.isfinite(p) and 0<=p<=1 for p in probs) and abs(sum(probs)-1)<1e-6
        pred=int(sum(probs[1:])>.5)+int(probs[2]>.5)
        cm[r['label']][pred]+=1
    correct=sum(cm[i][i] for i in range(3))
    n=sum(sum(row) for row in cm)
    supports=[sum(cm[i])+abstained[i] for i in range(3)]
    return {'correct':correct,'assigned':len(rows),'predicted':n,'accuracy_all':correct/len(rows),
        'accuracy_predicted':correct/n,'recall_all':[cm[i][i]/supports[i] for i in range(3)],
        'severe_to_mild':cm[2][0],'cross_two_levels':cm[0][2]+cm[2][0],
        'MAE_predicted':sum(cm[i][j]*abs(i-j) for i in range(3) for j in range(3))/n,
        'confusion_matrix':cm,'abstained_class_counts':abstained}


def main():
    protocol=json.loads((INPUT/'development_nested_cv_v3.json').read_text())
    expected={e['sample_id']:e['label'] for e in protocol['development_entries']}
    runs={'seed42':OUTPUT/'sysu_raw_medical_cv_20261009',
          'seed43':OUTPUT/'sysu_raw_medical_multiseed_20261009/seed43',
          'seed44':OUTPUT/'sysu_raw_medical_multiseed_20261009/seed44',
          'bsda_seed42':OUTPUT/'sysu_bsda_image_cv_20261009'}
    results={}
    for name,root in runs.items():
        data=json.loads((root/'oof_results.json').read_text())
        assert data['receipt']['csv_sha256']==protocol['csv_sha256']
        results[name]={}
        for recipe,value in data['results'].items():
            counts=recount(value['records'],expected)
            recorded=value['metrics']['fixed_median_0.5']
            assert recorded['correct_count']==counts['correct']
            assert recorded['severe_to_mild_count']==counts['severe_to_mild']
            assert recorded['two_level_error_count']==counts['cross_two_levels']
            assert abs(recorded['mae']-counts['MAE_predicted'])<1e-6
            results[name][recipe]=counts
        classifiers=list(root.glob('fold*/*/refit/classifier.pt'))
        assert len(classifiers)==10
        for path in classifiers:
            done=json.loads(path.with_name('done.json').read_text())
            assert sha(path)==done['classifier_sha256']
    aggregate={}
    for recipe in ('baseline','medical_aug'):
        values=[results[f'seed{s}'][recipe] for s in (42,43,44)]
        aggregate[recipe]={'accuracy_all_mean':statistics.mean(v['accuracy_all'] for v in values),
            'accuracy_all_sample_sd':statistics.stdev(v['accuracy_all'] for v in values),
            'recall_all_mean':[statistics.mean(v['recall_all'][i] for v in values) for i in range(3)],
            'severe_to_mild_mean':statistics.mean(v['severe_to_mild'] for v in values),
            'cross_two_levels_mean':statistics.mean(v['cross_two_levels'] for v in values),
            'MAE_predicted_mean':statistics.mean(v['MAE_predicted'] for v in values)}
    payload={'csv_sha256':protocol['csv_sha256'],'cv_sha256':sha(INPUT/'development_nested_cv_v3.json'),
        'runs':results,'three_seed_aggregate':aggregate,'classifiers_verified':40,
        'old_test_inference':False,'model_forward_during_collection':False,
        'interpretation':'development OOF, not independent new-cohort validation; 3 seeds do not triple patient count'}
    (INPUT/'completed_results.json').write_text(json.dumps(payload,indent=2))
    with tarfile.open(INPUT/'completed_evidence.tar.gz','w:gz') as archive:
        for name,root in runs.items():
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.name in {'oof_results.json','freeze.json','done.json','classifier.pt','outer_predictions.json'}:
                    archive.add(path,arcname=str(Path('runs')/name/path.relative_to(root)))
        for name in ('completed_results.json','development_nested_cv_v3.json','sysu_samples.csv','image_feature_diagnosis.json'):
            archive.add(INPUT/name,arcname=name)
        for folder in ('paired_views_final','bsda_pair'):
            for path in sorted((INPUT/folder).rglob('*')):
                if path.is_file() and path.suffix in {'.npz','.json'}:
                    archive.add(path,arcname=str(Path(folder)/path.relative_to(INPUT/folder)))
    print(json.dumps({'archive_bytes':(INPUT/'completed_evidence.tar.gz').stat().st_size,
                     'archive_sha256':sha(INPUT/'completed_evidence.tar.gz'),'classifiers_verified':40,
                     'three_seed_aggregate':aggregate,'BSDA':results['bsda_seed42']}),flush=True)


if __name__=='__main__':
    main()
