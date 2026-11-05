"""Grouped cross-fitted table probabilities with saved, auditable reference contexts."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
from pathlib import Path
import time

import numpy as np

from sysu_binary_common import BinaryData, digest, read, sha, write
from sysu_tabular_common import logistic_model, offline_predictions


def partition(data, ids, task, seed, max_folds=5):
    from sklearn.model_selection import StratifiedGroupKFold
    y, groups = data.target(ids, task), np.asarray([data.groups[s] for s in ids])
    expected = set(range(2 if task == 'binary' else 3))
    for count in range(max_folds, 1, -1):
        if len(set(groups)) < count:
            continue
        splitter = StratifiedGroupKFold(count, shuffle=True, random_state=seed)
        parts = list(splitter.split(np.zeros(len(ids)), y, groups))
        if all(set(y[train]) == expected and not (set(groups[train]) & set(groups[test])) for train, test in parts):
            return parts
    raise ValueError('Insufficient groups/classes for honest table cross-fitting')


def build_branch(data, task, family, seed, fit_ids, query_ids, scope):
    import joblib
    import torch
    full_fit, full_query = list(fit_ids), list(query_ids)
    if set(full_fit) & set(full_query) or {data.groups[s] for s in full_fit} & {data.groups[s] for s in full_query}:
        raise ValueError('Outer/inner table reference reaches evaluation patients')
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'torch')}
    if family == 'tabpfn':
        versions['tabpfn'] = importlib.metadata.version('tabpfn')
    identity = {'task': task, 'family': family, 'seed': seed, 'fit_ids': full_fit, 'query_ids': full_query,
        'scope': scope, 'genes': data.genes, 'threshold': data.threshold, 'csv_sha256': data.config['csv_sha256'],
        'protocol_sha256': data.config['protocol_sha256'], 'crossfit_folds': data.config['table_crossfit_folds'],
        'tabpfn': data.config['tabpfn'], 'versions': versions,
        'source_sha256': {str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('sysu_binary_common.py'),
                                            Path(__file__).with_name('sysu_tabular_common.py')]}}
    folder = Path(data.config['cache_root']) / digest(identity)
    if (folder / 'done.json').exists():
        done = read(folder / 'done.json')
        if done['identity'] != identity or any(sha(folder / p) != v for p, v in done['files'].items()):
            raise ValueError('Crossfit cache content differs')
        return done, folder
    folder.mkdir(parents=True, exist_ok=True)
    frame = data.frame(data.ids)
    available = dict(zip(data.ids, frame.notna().any(axis=1).tolist()))
    frame.index = data.ids
    fit = [s for s in full_fit if available[s]]
    query = [s for s in full_query if available[s]]
    parts = partition(data, fit, task, seed, data.config['table_crossfit_folds'])
    classes = 2 if task == 'binary' else 3
    probabilities, contributors, assignments = {}, {s: [] for s in query}, []
    fingerprints = {}
    started = time.monotonic()
    for n, (a, b) in enumerate(parts):
        train, test = [fit[i] for i in a], [fit[i] for i in b]
        paths = folder / f'part{n}'
        paths.mkdir(exist_ok=True)
        y = data.target(train, task)
        X = frame.loc[train]
        target_ids = test + query
        marker = paths / 'done.json'
        record_identity = {'train': train, 'test': test, 'query': query, 'parent': digest(identity)}
        if marker.exists():
            item = read(marker)
            if item['identity'] != record_identity or any(sha(paths / p) != v for p, v in item['files'].items()):
                raise ValueError('Saved table context changed')
            predictions = np.asarray(item['probabilities'])
        else:
            if family == 'logistic':
                model = logistic_model(X, seed + n)
            else:
                from tabpfn import TabPFNClassifier
                from tabpfn.constants import ModelVersion
                cfg = data.config['tabpfn']
                if versions['tabpfn'] != cfg['tabpfn_package_version'] or sha(cfg['tabpfn_weights']) != cfg['tabpfn_weights_sha256']:
                    raise ValueError('Pinned TabPFN changed')
                model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, model_path=cfg['tabpfn_weights'],
                    n_estimators=cfg['n_estimators'], random_state=seed + n, device='cuda',
                    n_preprocessing_jobs=1, categorical_features_indices=[list(X.columns).index('TCT')],
                    inference_precision=torch.float32, show_progress_bar=False)
            with offline_predictions():
                model.fit(X, y)
                predictions = model.predict_proba(frame.loc[target_ids])
            if list(model.classes_) != list(range(classes)) or not np.isfinite(predictions).all():
                raise ValueError('Table class order/probabilities invalid')
            if family == 'logistic':
                filename = 'classifier.joblib'
                joblib.dump(model, paths / filename)
                restored = joblib.load(paths / filename)
            else:
                from tabpfn.model_loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model
                filename = 'classifier.tabpfn_fit'
                save_fitted_tabpfn_model(model, paths / filename)
                with offline_predictions():
                    restored = load_fitted_tabpfn_model(paths / filename, device='cuda')
            with offline_predictions():
                reloaded = restored.predict_proba(frame.loc[target_ids])
            error = float(np.max(np.abs(reloaded - predictions)))
            if error > 1e-5:
                raise ValueError('Table reload changed probabilities')
            item = {'identity': record_identity, 'probabilities': predictions.tolist(), 'columns': list(X.columns),
                    'classes': list(range(classes)), 'reload_max_absolute_error': error,
                    'files': {filename: sha(paths / filename)}}
            write(marker, item)
            del model, restored
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for s, value in zip(target_ids, predictions):
            if s in test:
                if s in probabilities:
                    raise ValueError('Duplicate crossfit training prediction')
                probabilities[s] = value.tolist()
            else:
                contributors[s].append(value)
        assignments.append({'part': n, 'fit_ids': train, 'unseen_training_ids': test, 'unseen_query_ids': query})
        for filename in list(item['files']) + ['done.json']:
            fingerprints[f'part{n}/{filename}'] = sha(paths / filename)
    for sid, values in contributors.items():
        if len(values) != len(parts):
            raise ValueError('Missing context contribution')
        probabilities[sid] = np.mean(values, axis=0).tolist()
    for s in fit:
        context = next(a for a in assignments if s in a['unseen_training_ids'])
        if data.groups[s] in {data.groups[t] for t in context['fit_ids']}:
            raise ValueError('Training probability has seen related answer')
    if set(probabilities) != set(fit + query):
        raise ValueError('Incomplete crossfit outputs')
    result = {'identity': identity, 'probabilities': probabilities, 'assignments': assignments,
              'files': fingerprints, 'fold_count': len(parts), 'columns': list(frame.columns),
              'field_masks': {s: frame.loc[s].notna().astype(float).tolist() for s in full_fit + full_query},
              'available': {s: available[s] for s in full_fit + full_query},
              'seconds': time.monotonic() - started, 'network_connections_during_processing': 0}
    write(folder / 'done.json', result)
    return result, folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--task', choices=['binary', 'threeclass'], default='binary')
    p.add_argument('--table', choices=['logistic', 'tabpfn'], default='logistic')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fold', type=int, default=0)
    a = p.parse_args()
    c = read(a.config)
    data = BinaryData(c, c['fields'] == 'genes', c['threshold_input'])
    fold = next(f for f in data.protocol['folds'] if f['fold'] == a.fold)
    for scope, train, val in [('inner', 'inner_train', 'inner_val'), ('outer', 'outer_train', 'outer_holdout')]:
        result, path = build_branch(data, a.task, a.table, a.seed + a.fold, fold[train]['sample_ids'],
                                    fold[val]['sample_ids'], f'fold{a.fold}/{scope}')
        print(scope, path, result['seconds'], flush=True)


if __name__ == '__main__':
    main()
