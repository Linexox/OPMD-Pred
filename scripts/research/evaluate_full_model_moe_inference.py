"""Recompute complete expert inference with ALL five table contexts for every row.

Training cross-fit table scores are not deployment predictions on training rows.
Existing results are preserved; corrected all-context results are separate artifacts.
"""
import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tabpfn.model_loading import load_fitted_tabpfn_model

from sysu_binary_common import BinaryData, read, write, sha, score
from sysu_tabular_common import offline_predictions
from opmd_pred.model.full_model_moe import FullModelRouter, full_model_features
from opmd_pred.model.two_branch_fusion import TwoBranchFusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/sysu_moe/full_model_router.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read(args.config)
    base = read(config['base_config'])
    assert importlib.metadata.version('tabpfn') == base['tabpfn']['tabpfn_package_version']
    data = BinaryData(base, genes=False, threshold=False)
    images, image_present = data.photos('resnet18')
    for scope in ('assessment', 'deployment'):
        folder = Path(config['output_root']) / scope
        old = read(folder / 'results.json')
        ids = old['evaluation_ids']
        frame = data.frame(ids)
        table_present = frame.notna().any(axis=1).to_numpy()
        mask = frame.notna().astype(np.float32).to_numpy()
        image_flags = image_present[[data.index[s] for s in ids]]
        available = table_present | image_flags
        all_h, all_p, context_checks = [], [], []
        for e, spec in enumerate(old['expert_specs']):
            cache = Path(spec['table_cache'])
            branch = read(cache / 'done.json')
            assert sha(spec['model']) == spec['model_sha256']
            outputs = []
            for k in range(5):
                path = cache / f'part{k}/classifier.tabpfn_fit'
                assert sha(path) == branch['files'][f'part{k}/classifier.tabpfn_fit']
                with offline_predictions():
                    reference = load_fitted_tabpfn_model(path, device=args.device)
                    assert list(reference.classes_) == [0, 1]
                    probabilities = np.asarray(reference.predict_proba(frame.loc[table_present]), dtype=np.float64)
                    # Compare two isolated 24-row queries with the all-record query batch.
                    errors = []
                    for row in range(min(2, len(probabilities))):
                        single = frame.loc[table_present].iloc[[row]]
                        repeated = pd.concat([single] * 24, ignore_index=True)
                        one = np.asarray(reference.predict_proba(repeated))[0]
                        errors.append(float(np.max(np.abs(one - probabilities[row]))))
                if max(errors, default=0) > 1e-4:
                    raise ValueError('Table inference batch paths differ by more than 1e-4')
                outputs.append(probabilities)
                context_checks.append({'expert': e, 'context': k,
                                       'single_24_vs_batch_max_error': max(errors, default=0)})
                del reference
            table = np.zeros((len(ids), 14), np.float32)
            table[table_present, :2] = np.mean(outputs, axis=0)
            table[table_present, 2:] = mask[table_present]
            artifact = torch.load(spec['model'], weights_only=True, map_location='cpu')
            model = TwoBranchFusion(**artifact['model_config']).to(args.device).eval().requires_grad_(False)
            model.load_state_dict(artifact['state'], strict=True)
            h0 = np.zeros((len(ids), 3, 128), np.float32)
            p = np.full((len(ids), 2), np.nan, np.float32)
            ix = np.flatnonzero(available)
            for start in range(0, len(ix), 8):
                chosen = ix[start:start + 8]
                batch = {'image': torch.tensor(images[[data.index[ids[i]] for i in chosen], 0], device=args.device),
                         'table': torch.tensor(table[chosen], device=args.device),
                         'image_present': torch.tensor(image_flags[chosen], device=args.device),
                         'table_present': torch.tensor(table_present[chosen], device=args.device)}
                h, result = full_model_features(model, batch)
                h0[chosen], p[chosen] = h.cpu().numpy(), result.cpu().numpy()
            all_h.append(h0); all_p.append(p)
            del model
            print(scope, 'recomputed complete expert', e, flush=True)
        h0, probabilities = np.stack(all_h, 1), np.stack(all_p, 1)
        np.savez_compressed(folder / 'inference-inputs.npz', sample_ids=np.asarray(ids),
                            h0=h0, probabilities=probabilities, available=available)
        revised = {k: v for k, v in old.items() if k != 'methods'}
        revised['methods'] = {}
        revised['table_prediction_rule'] = 'mean of all five fitted contexts for every row, including previously seen rows'
        revised['postprocessing_source_sha256'] = sha(__file__)
        revised['context_batch_checks'] = context_checks
        for mode in ('mean', 'constant', 'dynamic'):
            artifact = torch.load(folder / f'{mode}.pt', weights_only=True, map_location='cpu')
            gate = FullModelRouter(**artifact['model_config']).to(args.device).eval()
            gate.load_state_dict(artifact['state'], strict=True)
            with torch.no_grad():
                mixed, alpha = gate(torch.tensor(h0[available], device=args.device),
                                    torch.tensor(probabilities[available], device=args.device))
            lookup = dict(zip([s for s, flag in zip(ids, available) if flag], mixed.cpu().tolist()))
            rows = data.prediction_rows(ids, lookup, -1)
            for position, a, expert_p in zip(np.flatnonzero(available), alpha.cpu().tolist(), probabilities[available].tolist()):
                rows[position].update(router_weights=a, complete_expert_probabilities=expert_p)
            revised['methods'][mode] = {'records': rows, 'metrics': score(rows, 'binary'),
                'mean_router_weights': alpha.mean(0).cpu().tolist(),
                'maximum_weight_mean': float(alpha.max(1).values.mean()),
                'honest_training_metrics': old['methods'][mode]['honest_training_metrics']}
        if scope == 'assessment':
            error = max(abs(a - b) for mode in revised['methods']
                        for r, s in zip(revised['methods'][mode]['records'], old['methods'][mode]['records'])
                        if r['probabilities'] is not None for a, b in zip(r['probabilities'], s['probabilities']))
            assert error < 1e-5
            revised['original_assessment_max_absolute_error'] = error
        write(folder / 'inference-results.json', revised)
        print(scope, {m: v['metrics']['accuracy'] for m, v in revised['methods'].items()}, flush=True)


if __name__ == '__main__':
    main()
