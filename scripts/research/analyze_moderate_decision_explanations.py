"""Locate the decision step at which saved models misclassify moderate records."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/experiments/sysu/explanation-acquisition-20261017/attempt1'
OUT = ROOT / 'outputs/experiments/sysu/moderate-decision-explanations-20261017/attempt1'
NAMES = ('image', 'history', 'tct', 'dna', 'methylation')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contribution(values):
    result = np.zeros(5, dtype=float)
    for index in range(5):
        for mask in range(32):
            if mask & (1 << index):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(4 - size) / math.factorial(5)
            result[index] += weight * (values[mask | (1 << index)] - values[mask])
    return result.tolist()


def main():
    protocol = read(SOURCE / 'protocol.json')
    by_id = {case['sample_id']: case for case in protocol['cases']}
    detailed = defaultdict(list)
    caps = defaultdict(list)
    receipts = []
    for model in protocol['models']:
        name = f"{model['method']}_seed{model['seed']}_fold{model['fold']}"
        cache_path = SOURCE / 'coalitions' / (name + '.npz')
        receipt = read(cache_path.with_suffix('.json'))
        if sha(cache_path) != receipt['coalitions_sha256'] or sha(Path(model['path'])) != model['sha256']:
            raise ValueError('Saved evidence changed')
        with np.load(cache_path, allow_pickle=False) as cache:
            probabilities = cache['probabilities'].astype(float)
            positions = {sid: i for i, sid in enumerate(cache['sample_ids'].tolist())}
        if model['method'] == 'ordinal_pretrain':
            state = torch.load(model['path'], map_location='cpu', weights_only=True)['state']
            gap = float(torch.nn.functional.softplus(state['ordinal_head.threshold_gap']))
            caps[model['seed']].append(dict(fold=model['fold'], threshold_gap=gap, maximum_middle_score=math.tanh(gap / 4)))
        for sid in model['held_ids']:
            case = by_id[sid]
            if case['label'] != 1:
                continue
            row = dict(sample_id=sid, seed=model['seed'], fold=model['fold'], available=bool(case['observed_mask']),
                       quality_flags=case['quality_flags'], observed=case['observed'])
            if case['observed_mask']:
                values = probabilities[positions[sid]]
                full = case['observed_mask']
                q1, q2 = values[:, 1:].sum(1), values[:, 2]
                prediction = int(q1[full] > .5) + int(q2[full] > .5)
                row.update(prediction=prediction, probabilities=values[full].tolist(),
                           above_mild=float(q1[full]), above_moderate=float(q2[full]),
                           above_mild_contributions=contribution(q1), above_moderate_contributions=contribution(q2))
                removed = {}
                for index, modality in enumerate(NAMES):
                    if full & (1 << index):
                        mask = full ^ (1 << index)
                        removed[modality] = dict(prediction=int(q1[mask] > .5) + int(q2[mask] > .5) if mask else None,
                                                 above_mild=float(q1[mask]), above_moderate=float(q2[mask]))
                row['remove_one_observed_input'] = removed
            detailed[model['method']].append(row)
        receipts.append(dict(model=name, checkpoint_sha256=model['sha256'], coalition_sha256=receipt['coalitions_sha256']))
    methods = {}
    for method, rows in detailed.items():
        per_seed = []
        groups = {}
        for seed in (42, 43, 44):
            selected = [row for row in rows if row['seed'] == seed]
            n = len(selected)
            counts = [sum(row.get('prediction') == index for row in selected) for index in range(3)]
            per_seed.append(dict(seed=seed, records=n, predicted_counts=counts,
                                 low_fraction=counts[0]/n, moderate_recall=counts[1]/n, high_fraction=counts[2]/n,
                                 abstained=sum(not row['available'] for row in selected)))
        for index, label in enumerate(('predicted_low', 'predicted_moderate', 'predicted_severe')):
            selected = [row for row in rows if row.get('prediction') == index]
            groups[label] = dict(record_model_observations=len(selected))
            if selected:
                for target in ('above_mild', 'above_moderate'):
                    groups[label][target + '_mean_score'] = float(np.mean([row[target] for row in selected]))
                    groups[label][target + '_mean_signed_contributions'] = np.mean([row[target + '_contributions'] for row in selected], axis=0).tolist()
        methods[method] = dict(per_seed=per_seed, mean={key: float(np.mean([row[key] for row in per_seed])) for key in ('low_fraction', 'moderate_recall', 'high_fraction')}, groups=groups)
    result = dict(scope='All 31 moderate development records, existing five-fold held-out predictions, three saved seeds; all available inputs.',
                  methods=methods, ordinal_threshold_caps=dict(caps), modalities=list(NAMES), new_training=False,
                  new_inference=False, locked_22_evaluated=False, protocol_sha256=sha(SOURCE/'protocol.json'),
                  limitations=['Existing model comparison changes both pretraining and classification head; no causal attribution to head alone.',
                               'Middle-class score cap does not bound recall under the cumulative median rule.',
                               'Original grading and collection-time questions remain; no source records changed.',
                               'Repeated seeds are repeated model observations, not additional patients.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'records.private.json').write_text(json.dumps(dict(detailed), ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'sources.json').write_text(json.dumps(dict(models=receipts, source_sha256=sha(Path(__file__))), indent=2), encoding='utf-8')
    print(json.dumps({key: value['mean'] for key, value in methods.items()}, ensure_ascii=True))
    print(json.dumps({'ordinal_middle_score_cap_range': [min(row['maximum_middle_score'] for seed in caps.values() for row in seed), max(row['maximum_middle_score'] for seed in caps.values() for row in seed)]}))


if __name__ == '__main__':
    main()
