"""Freeze fresh stratified patient-group six-fold protocols for downstream seeds."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from sysu_binary_common import sha, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[43,44,45]);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new protocol directory')
    b=json.loads(a.bundle.read_text(encoding='utf-8'));ids=[r['sample_id'] for r in b['samples']]
    groups={r['sample_id']:r['group'] for r in b['samples']};grades={r['sample_id']:r['grade'] for r in b['samples']}
    def parts(selected,count,seed):
        splitter=StratifiedGroupKFold(count,shuffle=True,random_state=seed)
        result=[[selected[i] for i in held] for _,held in splitter.split(np.zeros(len(selected)),[grades[s] for s in selected],[groups[s] for s in selected])]
        assert set(s for part in result for s in part)==set(selected)
        for i,left in enumerate(result):
            for right in result[i+1:]:assert not {groups[s] for s in left}&{groups[s] for s in right}
        return result
    a.output.mkdir(parents=True);fingerprints=[]
    for seed in a.seeds:
        outer=parts(ids,6,seed);folds=[]
        for n,held in enumerate(outer):
            scope=[s for s in ids if s not in set(held)];five=parts(scope,5,seed);experts=[]
            for e,omitted in enumerate(five):
                fit=[s for s in scope if s not in set(omitted)]
                val=[s for s in fit if s in set(five[(e+1)%5])];inner=[s for s in fit if s not in set(val)]
                for left,right in [(fit,held),(inner,val)]:assert not {groups[s] for s in left}&{groups[s] for s in right}
                assert set(inner)|set(val)==set(fit)
                experts.append({'expert':e,'fit_ids':fit,'inner_train':inner,'inner_validation':val,
                                'initial_seed':seed+e,'refit_seed':seed+100+e})
            folds.append({'fold':n,'held_ids':held,'experts':experts})
        signature={frozenset(part) for part in outer};assert all(signature!=v for v in fingerprints);fingerprints.append(signature)
        write(a.output/f'seed{seed}.json',{'seed':seed,'folds':folds,'bundle_sha256':sha(a.bundle),
              'stratification':'original three-grade labels, grouped by existing patient/repetition component',
              'changed':'outer split, remaining five-group split, model initialization and photo-view/training RNG',
              'fixed':'external cell and methylation pretrained weights, oral-photo weights, inputs, hyperparameters, equal 20% ensemble',
              'test_used_for_selection':False})
        print(f'Frozen seed{seed}: outer sizes {[len(p) for p in outer]}',flush=True)


if __name__=='__main__':main()
