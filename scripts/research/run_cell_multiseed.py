"""Additional random-split seeds for the externally pretrained multimodal mean5 model."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_methylation_cell_transfer import Fusion, Inputs, fit, predict
from sysu_binary_common import score, sha, write


def run_seed(seed,protocol,bundle,photo,cells,methyl,c,folder,device):
    folder.mkdir(parents=True,exist_ok=False);folds=[];all_rows=[]
    for fold in protocol['folds']:
        n=fold['fold'];held=fold['held_ids'];where=folder/f'fold{n}';members=[];specs=[]
        for spec in fold['experts']:
            e=spec['expert'];path=where/f'expert{e}'
            inputs=Inputs(bundle,photo,cells,spec['inner_train'],True)
            _,epochs=fit(inputs,spec['inner_train'],spec['inner_validation'],methyl,c,spec['initial_seed'],c['fusion_epochs'],device,path/'inner')
            inputs=Inputs(bundle,photo,cells,spec['fit_ids'],True)
            model,_=fit(inputs,spec['fit_ids'],None,methyl,c,spec['refit_seed'],epochs,device,path/'refit')
            rows=predict(model,inputs,held,device,n)
            saved=torch.load(path/'refit/classifier.pt',weights_only=True,map_location='cpu')
            assert saved['training_ids']==inputs.eligible(spec['fit_ids']) and not set(held)&set(saved['training_ids'])
            restored=Fusion(inputs.table.shape[1],c).to(device);restored.load_state_dict(saved['state'],strict=True)
            reread=predict(restored,inputs,held,device,n)
            error=max([abs(x-y) for r,s in zip(rows,reread) if r['probabilities'] is not None for x,y in zip(r['probabilities'],s['probabilities'])] or [0.])
            assert error<1e-6
            members.append(rows);specs.append({'expert':e,'selected_epoch':epochs,'classifier_sha256':sha(path/'refit/classifier.pt'),'reload_error':error})
            write(path/'predictions.json',{'records':rows,**specs[-1]})
        averaged=[]
        for i,sid in enumerate(held):
            rs=[r[i] for r in members];assert all(r['sample_id']==sid for r in rs)
            available=[r['probabilities'] is not None for r in rs];assert len(set(available))==1
            p=np.mean([r['probabilities'] for r in rs],0).tolist() if available[0] else None
            averaged.append({'sample_id':sid,'grade':rs[0]['grade'],'fold':n,'probabilities':p,
                             'member_probabilities':[r['probabilities'] for r in rs],'weights':[.2]*5})
        result={'fold':n,'metrics':score(averaged),'experts':specs,'records':averaged}
        folds.append(result);all_rows.extend(averaged);write(where/'results.json',result)
        write(folder/'state.json',{'state':'running','seed':seed,'completed_fold':n+1})
        print(f'seed{seed} fold{n+1}: accuracy={result["metrics"]["accuracy"]:.4f}',flush=True)
    assert len(all_rows)==len({r['sample_id'] for r in all_rows})==117
    accuracies=[r['metrics']['accuracy'] for r in folds]
    result={'seed':seed,'metrics':score(all_rows),'folds':folds,'records':all_rows,'fold_accuracy_mean':float(np.mean(accuracies)),
            'fold_accuracy_max':float(max(accuracies)),'fold_accuracy_std':float(np.std(accuracies,ddof=1))}
    write(folder/'results.json',result);write(folder/'state.json',{'state':'completed','seed':seed,'trained_models':30})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--protocols',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--methyl-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[43,44,45]);p.add_argument('--device',default='cuda');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);b=json.loads((a.data/'bundle.json').read_text(encoding='utf-8'));c=json.loads(a.config.read_text(encoding='utf-8'))
    assert sha(a.data/'photos.npz')==b['sources']['photos_sha256']
    photo=np.load(a.data/'photos.npz',allow_pickle=False);features=np.load(a.features,allow_pickle=False)
    assert list(features['sample_ids'])==[r['sample_id'] for r in b['samples']]
    cells=features['values'];assert cells.shape==(117,512) and np.isfinite(cells).all()
    methyl=torch.load(a.methyl_checkpoint,weights_only=True,map_location='cpu')
    assert methyl['genes']==b['genes'] and methyl['source_sha256']==b['sources']['external_csv_sha256']
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(a.output/'identity.json',{'base_config':c,'seeds':a.seeds,'bundle_sha256':sha(a.data/'bundle.json'),
          'methyl_sha256':sha(a.methyl_checkpoint),'cell_features_sha256':sha(a.features),
          'sources':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('run_methylation_cell_transfer.py'),Path(__file__).with_name('sysu_binary_common.py')]},
          'protocol_sha256':{str(seed):sha(a.protocols/f'seed{seed}.json') for seed in a.seeds},'test_used_for_selection':False})
    results={};started=time.time()
    for seed in a.seeds:
        protocol=json.loads((a.protocols/f'seed{seed}.json').read_text(encoding='utf-8'))
        assert protocol['bundle_sha256']==sha(a.data/'bundle.json') and protocol['seed']==seed
        results[str(seed)]=run_seed(seed,protocol,b,photo,cells,methyl['encoder'],dict(c,seed=seed),a.output/f'seed{seed}',a.device)
    write(a.output/'results.json',{'results':results,'elapsed_seconds':time.time()-started,'app_replaced':False})
    print('ADDITIONAL SEEDS COMPLETED',flush=True)


if __name__=='__main__':main()
