"""Recount six OOF runs, verify classifier bytes, and archive all outcomes including failure."""
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

P=Path('/home/zjkj/24334067')
R=P/'OPMD-Pred-SYSU-Proxy-Balanced/outputs/sysu_distance_20261010_attempt2'
I=P/'sysu-research-inputs/raw-rebuild-20261009'


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    protocol=json.loads((I/'development_nested_cv_v3.json').read_text())
    expected={r['sample_id']:r['label'] for r in protocol['development_entries']}
    runs={}
    for seed in (42,43,44):
        runs[str(seed)]={}
        for name in ('control','distance05'):
            root=R/f'seed{seed}'/name
            data=json.loads((root/'oof_results.json').read_text())['results']['baseline']
            rows=data['records']; assert len(rows)==len({r['sample_id'] for r in rows})==117
            assert {r['sample_id']:r['label'] for r in rows}==expected
            cm=[[0]*3 for _ in range(3)]; abstained=0
            for row in rows:
                if row['status']=='abstain':
                    assert row['probabilities'] is None; abstained+=1; continue
                p=row['probabilities']; assert len(p)==3 and all(math.isfinite(x) and 0<=x<=1 for x in p) and abs(sum(p)-1)<1e-6
                pred=int(sum(p[1:])>.5)+int(p[2]>.5); cm[row['label']][pred]+=1
            correct=sum(cm[i][i] for i in range(3)); severe=cm[2][0]; cross=severe+cm[0][2]
            m=data['metrics']['fixed_median_0.5']
            assert m['correct_count']==correct and m['severe_to_mild_count']==severe and m['two_level_error_count']==cross and abstained==5
            models=list(root.glob('fold*/baseline/refit/classifier.pt')); assert len(models)==5
            for p in models: assert sha(p)==json.loads(p.with_name('done.json').read_text())['classifier_sha256']
            runs[str(seed)][name]={'correct':correct,'accuracy':correct/117,'severe_to_mild':severe,'cross_two':cross,
                'severe_recall':cm[2][2]/31,'MAE':sum(abs(i-j)*cm[i][j] for i in range(3) for j in range(3))/112,
                'confusion':cm,'abstained':abstained}
        print(seed,json.dumps(runs[str(seed)]),flush=True)
    aggregate={}
    for name in ('control','distance05'):
        values=[runs[str(s)][name] for s in (42,43,44)]
        aggregate[name]={k:statistics.mean(x[k] for x in values) for k in ('correct','accuracy','severe_to_mild','cross_two','severe_recall','MAE')}
        aggregate[name]['accuracy_sample_sd']=statistics.stdev(x['accuracy'] for x in values)
    payload={'runs':runs,'aggregate':aggregate,'classifiers_verified':30,'old_test_used':False,
             'data_sha256':sha(I/'sysu_samples.csv'),'cv_sha256':sha(I/'development_nested_cv_v3.json')}
    (R/'verified_results.json').write_text(json.dumps(payload,indent=2))
    archive=I/'distance-study-evidence-20261010.tar.gz'
    with tarfile.open(archive,'w:gz') as z:
        for root in (R,R.with_name('sysu_distance_20261010')):
            for p in root.rglob('*'):
                if p.is_file() and p.name!='last.pt': z.add(p,arcname=str(p.relative_to(R.parent)))
        for name in ('train_distance_cv.py','run_distance_study.py','collect_distance_results.py'):
            z.add(R.parents[1]/'scripts/research'/name,arcname='source/'+name)
    receipt={'bytes':archive.stat().st_size,'sha256':sha(archive),'classifiers_verified':30}
    (I/'distance-archive-receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
