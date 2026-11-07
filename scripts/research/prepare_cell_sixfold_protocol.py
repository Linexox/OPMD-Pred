"""Freeze the exact historical APP six test folds and thirty expert fit lists."""
import argparse
import json
from pathlib import Path

from sysu_binary_common import sha, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--main-root',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Do not overwrite a frozen protocol')
    baseline_path=a.main_root/'outputs/experiments/sysu/sixfold-five-expert-stability-20261104/attempt1/oof_results.json'
    original=json.loads(baseline_path.read_text());baseline=original['methods']['mean']
    b=json.loads(a.bundle.read_text(encoding='utf-8'));groups={r['sample_id']:r['group'] for r in b['samples']}
    scope=set(groups);folds=[];sources={str(baseline_path):sha(baseline_path)};all_held=[]
    for n,root in enumerate(original['fold_roots']):
        candidates=[a.main_root/root/'assessment'/name for name in ['inference-results.json','results.json']]
        source=next(p for p in candidates if p.exists());r=json.loads(source.read_text())
        sources[str(source)]=sha(source);held=r['evaluation_ids'];all_held.extend(held)
        train_scope=scope-set(held);specs=r['expert_specs']
        omissions=[train_scope-set(e['fit_ids']) for e in specs]
        assert set.union(*omissions)==train_scope and sum(map(len,omissions))==len(train_scope)
        entries=[]
        for e,spec in enumerate(specs):
            fit=spec['fit_ids'];val=[s for s in fit if s in omissions[(e+1)%5]]
            inner=[s for s in fit if s not in set(val)]
            assert set(inner)|set(val)==set(fit) and not set(inner)&set(val)
            for left,right in [(fit,held),(inner,val)]:
                assert not {groups[s] for s in left}&{groups[s] for s in right}
            entries.append({'expert':e,'fit_ids':fit,'inner_train':inner,'inner_validation':val,
                            'initial_seed':42+e,'refit_seed':142+e})
        assert len(entries)==5
        assert held==[row['sample_id'] for row in baseline['records'] if row['fold']==n]
        folds.append({'fold':n,'held_ids':held,'experts':entries})
    assert len(all_held)==len(set(all_held))==117 and set(all_held)==scope
    write(a.output,{'folds':folds,'historical_app':baseline,'source_sha256':sources,
         'bundle_sha256':sha(a.bundle),'selection':'new-model hyperparameters unchanged; epoch selected within each expert fit list; six-fold tests never used for selection',
         'ensemble':'five complete multimodal models; fixed probability weights 0.2 each; no Router training',
         'historical_app_training':'fixed 7 epochs, lr 1e-4, weight_decay 1e-4, batch8, modality_dropout .15',
         'new_model_training':'reuse methylation_cell_transfer.json; nested epoch selection; architecture/inputs/optimization differ from historical APP'})
    print('PASS: exact six historical test lists and 30 expert fit lists frozen; group isolation verified.')


if __name__=='__main__':main()
