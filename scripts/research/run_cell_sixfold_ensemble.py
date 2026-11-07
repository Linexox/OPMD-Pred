"""Matched historical six-fold evaluation of five new multimodal models at 20% each."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_methylation_cell_transfer import Fusion, Inputs, fit, predict
from sysu_binary_common import score, sha, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--methyl-checkpoint',type=Path,required=True)
    p.add_argument('--features-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    protocol=json.loads(a.protocol.read_text());b=json.loads((a.data/'bundle.json').read_text(encoding='utf-8'));c=json.loads(a.config.read_text())
    assert sha(a.data/'bundle.json')==protocol['bundle_sha256']
    assert sha(a.data/'photos.npz')==b['sources']['photos_sha256']
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    methyl=torch.load(a.methyl_checkpoint,map_location='cpu',weights_only=True)
    assert methyl['genes']==b['genes'] and methyl['source_sha256']==b['sources']['external_csv_sha256']
    photo=np.load(a.data/'photos.npz',allow_pickle=False);sample_ids=[r['sample_id'] for r in b['samples']]
    sources={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('run_methylation_cell_transfer.py'),Path(__file__).with_name('sysu_binary_common.py')]}
    identity={'config':c,'protocol_sha256':sha(a.protocol),'methyl_sha256':sha(a.methyl_checkpoint),'sources':sources,
              'test_used_for_selection':False,'ensemble_weights':[.2]*5}
    write(a.output/'identity.json',identity);results={};started=time.time()
    for condition in ['imagenet','oral_pretrained']:
        cache=a.features_root/condition/'cell_features.npz';feature_file=np.load(cache,allow_pickle=False)
        assert list(feature_file['sample_ids'])==sample_ids
        cells=feature_file['values'];assert cells.shape==(117,512) and np.isfinite(cells).all()
        folds=[];all_rows=[]
        for fold in protocol['folds']:
            n=fold['fold'];held=fold['held_ids'];folder=a.output/condition/f'fold{n}'
            members=[];expert_details=[]
            for spec in fold['experts']:
                e=spec['expert'];path=folder/f'expert{e}'
                inputs=Inputs(b,photo,cells,spec['inner_train'],True)
                _,epochs=fit(inputs,spec['inner_train'],spec['inner_validation'],methyl['encoder'],c,
                             spec['initial_seed'],c['fusion_epochs'],a.device,path/'inner')
                inputs=Inputs(b,photo,cells,spec['fit_ids'],True)
                model,_=fit(inputs,spec['fit_ids'],None,methyl['encoder'],c,spec['refit_seed'],epochs,a.device,path/'refit')
                rows=predict(model,inputs,held,a.device,n)
                saved=torch.load(path/'refit/classifier.pt',map_location='cpu',weights_only=True)
                assert saved['training_ids']==inputs.eligible(spec['fit_ids'])
                assert not set(held)&set(saved['training_ids'])
                restored=Fusion(inputs.table.shape[1],c).to(a.device);restored.load_state_dict(saved['state'],strict=True)
                check=predict(restored,inputs,held,a.device,n)
                error=max([abs(v-w) for r,s in zip(rows,check) if r['probabilities'] is not None for v,w in zip(r['probabilities'],s['probabilities'])] or [0.])
                assert error<1e-6
                members.append(rows);expert_details.append({'expert':e,'epoch':epochs,'reload_error':error,'classifier_sha256':sha(path/'refit/classifier.pt')})
                write(path/'predictions.json',{'rows':rows,**expert_details[-1]})
            averaged=[]
            for i,sid in enumerate(held):
                chosen=[m[i] for m in members];assert all(r['sample_id']==sid for r in chosen)
                available=[r['probabilities'] is not None for r in chosen];assert len(set(available))==1
                p=np.mean([r['probabilities'] for r in chosen],0).tolist() if available[0] else None
                averaged.append({'sample_id':sid,'grade':chosen[0]['grade'],'fold':n,'probabilities':p,
                                 'member_probabilities':[r['probabilities'] for r in chosen],'weights':[.2]*5})
            item={'fold':n,'metrics':score(averaged),'experts':expert_details,'records':averaged}
            folds.append(item);all_rows.extend(averaged);write(folder/'results.json',item)
            print(f'{condition} fold{n+1}: ensemble accuracy={item["metrics"]["accuracy"]:.4f}',flush=True)
            write(a.output/'state.json',{'state':'running','condition':condition,'completed_fold':n+1})
        assert len(all_rows)==len({r['sample_id'] for r in all_rows})==117
        historical=protocol['historical_app']['records']
        assert {r['sample_id']:r['grade'] for r in all_rows if r['probabilities'] is not None}=={r['sample_id']:r['grade'] for r in historical if r['probabilities'] is not None}
        accuracies=[r['metrics']['accuracy'] for r in folds]
        results[condition]={'records':all_rows,'metrics':score(all_rows),'folds':folds,'fold_accuracy_mean':float(np.mean(accuracies)),
                            'fold_accuracy_max':float(max(accuracies)),'fold_accuracy_std':float(np.std(accuracies,ddof=1)),
                            'cell_features_sha256':sha(cache)}
        write(a.output/condition/'results.json',results[condition])
    write(a.output/'comparison.json',{'historical_app':protocol['historical_app'],'new_methods':results,
                                    'identity':identity,'elapsed_seconds':time.time()-started,'app_replaced':False})
    write(a.output/'state.json',{'state':'completed','new_multimodal_classifiers':60,'test_used_for_selection':False})
    print('SIX-FOLD FIVE-MODEL COMPARISON COMPLETED',flush=True)


if __name__=='__main__':main()
