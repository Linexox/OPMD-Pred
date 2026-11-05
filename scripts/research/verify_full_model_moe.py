"""Independently verify exclusion, complete-expert arithmetic, metrics and reload."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch

from opmd_pred.model.full_model_moe import FullModelRouter
from opmd_pred.model.two_branch_fusion import TwoBranchFusion
from sysu_binary_common import read, write, score, sha


def verify(root, scope, groups, grades, expected_ids, fusion_device):
    folder = root / scope
    corrected = (folder / 'inference-results.json').exists()
    result = read(folder / ('inference-results.json' if corrected else 'results.json'))
    if corrected:
        assert result['postprocessing_source_sha256'] == sha('scripts/research/evaluate_full_model_moe_inference.py')
        assert max(c['single_24_vs_batch_max_error'] for c in result['context_batch_checks']) <= 1e-4
    assert set(result['training_ids']) == set(expected_ids)
    test = set(result['evaluation_ids'])
    if scope == 'assessment':
        assert not (test & set(expected_ids))
        assert not ({groups[s] for s in test} & {groups[s] for s in expected_ids})
    audit = read(folder / 'training_exclusion_audit.json')
    assert len(audit) == 25
    for part in audit:
        assert not ({groups[s] for s in part['fit_ids']} & {groups[s] for s in part['unseen_ids']})
        if scope == 'assessment':
            assert not (test & set(part['fit_ids']))
    covered = Counter((s, part['expert']) for part in audit for s in part['unseen_ids'])
    assert len(covered) == len(expected_ids) * 5 and set(covered.values()) == {1}
    specs = result['expert_specs']
    assert len(specs) == 5
    features = [np.load(folder / f'experts/expert{e}/evaluation.npz', allow_pickle=False) for e in range(5)]
    for e, (spec, values) in enumerate(zip(specs, features)):
        assert spec['expert_number'] == e
        assert list(values['sample_ids']) == result['evaluation_ids']
        assert sha(spec['model']) == spec['model_sha256']
        complete = torch.load(spec['model'], weights_only=True, map_location='cpu')
        assert set(complete['identity']['training_ids']) == set(spec['fit_ids'])
        assert complete['model_config']['family'] == 'transformer'
        assert complete['model_config']['num_classes'] == 2
        cache = read(Path(spec['table_cache']) / 'done.json')
        assert set(cache['identity']['fit_ids']) == set(spec['fit_ids'])
        for assignment in cache['assignments']:
            assert set(assignment['fit_ids']) <= set(spec['fit_ids'])
            assert not ({groups[s] for s in assignment['fit_ids']} &
                        {groups[s] for s in assignment['unseen_training_ids']})
        if scope == 'assessment':
            assert not (set(spec['fit_ids']) & test)
    a = features[0]['available']
    assert all(np.array_equal(f['available'], a) for f in features)
    if corrected:
        values = np.load(folder / 'inference-inputs.npz', allow_pickle=False)
        assert list(values['sample_ids']) == result['evaluation_ids']
        assert np.array_equal(values['available'], a)
        x, p = torch.tensor(values['h0'][a]), torch.tensor(values['probabilities'][a])
    else:
        x = torch.tensor(np.stack([f['h0'][a] for f in features], 1))
        p = torch.tensor(np.stack([f['probabilities'][a] for f in features], 1))
    if not torch.isfinite(x).all() or not torch.isfinite(p).all():
        raise ValueError('Nonfinite complete expert output')
    torch.testing.assert_close(p.sum(2), torch.ones(p.shape[:2]))
    full_forward_error = None
    if fusion_device != 'skip':
        full_forward_error = 0.
        with torch.no_grad():
            for e, spec in enumerate(specs):
                artifact = torch.load(spec['model'], weights_only=True, map_location='cpu')
                full = TwoBranchFusion(**artifact['model_config']).to(fusion_device).eval()
                full.load_state_dict(artifact['state'], strict=True)
                actual = torch.cat([full.head(full.fusion(x[start:start + 8, e].to(fusion_device))[:, 0]).softmax(1).cpu()
                                    for start in range(0, len(x), 8)])
                error = float((actual - p[:, e]).abs().max())
                assert error < 1e-5, f'{scope} complete expert {e} {fusion_device} forward difference: {error}'
                full_forward_error = max(full_forward_error, error)
    maximum_error = 0.
    summaries = {}
    for mode, payload in result['methods'].items():
        assert [r['sample_id'] for r in payload['records']] == result['evaluation_ids']
        assert all(r['grade'] == grades[r['sample_id']] for r in payload['records'])
        assert score(payload['records'], 'binary') == payload['metrics']
        artifact = torch.load(folder / f'{mode}.pt', weights_only=True, map_location='cpu')
        assert set(artifact['training_ids']) <= set(expected_ids)
        gate = FullModelRouter(**artifact['model_config']).eval()
        gate.load_state_dict(artifact['state'], strict=True)
        with torch.no_grad():
            q, alpha = gate(x, p)
        saved = [r for r in payload['records'] if r['probabilities'] is not None]
        error = float(np.max(np.abs(q.numpy() - np.asarray([r['probabilities'] for r in saved]))))
        maximum_error = max(error, maximum_error)
        assert error < 1e-6
        np.testing.assert_allclose(alpha.numpy(), [r['router_weights'] for r in saved], atol=1e-6, rtol=0)
        manual = np.sum(p.numpy().astype(np.float64) * alpha.numpy().astype(np.float64)[:, :, None], axis=1)
        # Five FP32 products/sums can differ between NumPy and Torch by several ULPs.
        np.testing.assert_allclose(manual, q.numpy(), atol=5 * np.finfo(np.float32).eps, rtol=0)
        if mode == 'mean':
            np.testing.assert_allclose(q.numpy(), p.numpy().astype(np.float64).mean(1),
                                       atol=5 * np.finfo(np.float32).eps, rtol=0)
        summaries[mode] = {'metrics': payload['metrics'], 'mean_weights': payload['mean_router_weights'],
                           'max_weight_mean': payload['maximum_weight_mean']}
    labels = np.asarray([int(r['grade'] > 0) for r in result['methods']['mean']['records'] if r['probabilities'] is not None])
    expert_predictions = p.numpy()[:, :, 1] >= .5
    disagreement = (expert_predictions.any(1) & ~expert_predictions.all(1))
    meta = np.load(folder / 'router_training_inputs.npz', allow_pickle=False)
    return {'passed': True, 'evaluation_kind': result['evaluation_kind'],
            'all_table_contexts_used_for_every_inference_row': corrected,
            'assigned': len(test), 'predictable': int(a.sum()), 'no_input': int((~a).sum()),
            'gate_training_predictable': int(meta['available'].sum()),
            'five_unseen_experts_per_meta_record': True, 'cpu_reload_max_absolute_error': maximum_error,
            'complete_fusion_forward_checked': fusion_device != 'skip',
            'complete_fusion_forward_device': fusion_device,
            'complete_fusion_forward_max_absolute_error': full_forward_error,
            'complete_expert_accuracy': (expert_predictions == labels[:, None]).mean(0).tolist(),
            'complete_expert_disagreement_records': int(disagreement.sum()),
            'any_expert_correct_records': int((expert_predictions == labels[:, None]).any(1).sum()),
            'methods': summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/research/sysu_moe/full_model_router.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fusion-device', choices=['cpu', 'cuda', 'skip'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = read(args.config)
    base = read(cfg['base_config'])
    old = read(base['protocol'])
    assert sha(Path(base['csv'])) == base['csv_sha256']
    assert sha(Path(base['protocol'])) == base['protocol_sha256']
    with Path(base['csv']).open(encoding='utf-8-sig', newline='') as stream:
        grades = {r['sample_id']: int(r['grade']) for r in csv.DictReader(stream)}
    groups = {r['sample_id']: r['cv_component'] for r in old['development_entries']}
    locked = set(old['locked_previous_test_ids'] + old['additional_related_locked_ids'])
    assert not (set(groups) & locked)
    prepared = read(Path(cfg['prepared_root']) / 'partitions.json')
    assessment_ids = [s for p in prepared['assessment_parts'] for s in p]
    result = {'passed': True, 'old_locked_tests_excluded': True,
              'six_fold_sizes': [len(x) for x in prepared['six_parts']],
              'assessment_training_part_sizes': [len(x) for x in prepared['assessment_parts']],
              'original_five_fold_sizes': [len(x) for x in prepared['deployment_parts']]}
    for scope, ids in [('assessment', assessment_ids), ('deployment', list(groups))]:
        result[scope] = verify(Path(cfg['output_root']), scope, groups, grades, ids, args.fusion_device)
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('assessment', 'deployment')}))
    for scope in ('assessment', 'deployment'):
        print(scope, result[scope]['assigned'], result[scope]['predictable'],
              {m: r['metrics']['accuracy'] for m, r in result[scope]['methods'].items()})


if __name__ == '__main__':
    main()
