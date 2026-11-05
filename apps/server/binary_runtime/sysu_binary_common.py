"""Shared binary/three-class research data and explicit grade-destination metrics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

GENES = ['CDKN2A', 'TIMP3', 'ZNF582', 'TMEFF2', 'TFPI2', 'SHOX2', 'PAX1', 'MGMT', 'SOX1', 'DAPK1']
HISTORY = ['是否吸烟', '吸烟年限（年）', '吸烟量（支/天）', '是否饮酒', '饮酒年限(年)',
           '饮酒量(ml/天)', '是否嚼槟榔', '嚼槟榔年限(年)', '嚼槟榔量（颗/天）']
HISTORY_KEYS = ['smoking', 'smoking_years', 'smoking_amount', 'alcohol', 'alcohol_years',
                'alcohol_amount', 'betel', 'betel_years', 'betel_amount']
MISSING = {'', 'nan', 'NAN', 'NA', 'N/A', '/', '#VALUE!', 'None', 'null', 'NULL'}
LABELS = {'level1_mild_or_normal': 0, 'level2_moderate': 1, 'level3_severe': 2}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def number(value):
    if value is None or str(value).strip() in MISSING:
        return None
    if str(value).strip() in {'是', '有', 'yes', 'Yes'}:
        return 1.
    if str(value).strip() in {'否', '无', 'no', 'No'}:
        return 0.
    value = float(value)
    return value if np.isfinite(value) else None


def score(rows, task='binary', rule='probability_sum'):
    selected = [r for r in rows if r.get('probabilities') is not None]
    out = {'assigned_count': len(rows), 'eligible_count': len(selected),
           'coverage': len(selected) / len(rows) if rows else 0.,
           'abstain_by_grade': [sum(r.get('probabilities') is None and r['grade'] == g for r in rows) for g in range(3)]}
    if not selected:
        return dict(out, accuracy=None)
    p = np.asarray([r['probabilities'] for r in selected], dtype=float)
    if p.ndim != 2 or p.shape[1] not in (2, 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-5):
        raise ValueError('Invalid probabilities')
    grades = np.asarray([r['grade'] for r in selected], dtype=int)
    if task == 'binary':
        target = (grades > 0).astype(int)
        pred = (p[:, 1:].sum(1) >= .5).astype(int) if rule == 'probability_sum' else (p.argmax(1) > 0).astype(int)
        nclasses = 2
    else:
        if p.shape[1] != 3:
            raise ValueError('Three-class metrics require three probabilities')
        target, pred, nclasses = grades, p.argmax(1), 3
    cm = np.zeros((nclasses, nclasses), dtype=int)
    dest = np.zeros((3, nclasses), dtype=int)
    for g, y, h in zip(grades, target, pred):
        cm[y, h] += 1
        dest[g, h] += 1
    recalls = [float(cm[k, k] / cm[k].sum()) if cm[k].sum() else None for k in range(nclasses)]
    out.update(accuracy=float((pred == target).mean()), recalls=recalls,
               balanced_accuracy=float(np.mean([v for v in recalls if v is not None])),
               confusion=cm.tolist(), original_grade_destinations=dest.tolist(),
               moderate_detection=float((pred[grades == 1] > 0).mean()) if (grades == 1).any() else None,
               severe_detection=float((pred[grades == 2] > 0).mean()) if (grades == 2).any() else None,
               moderate_missed=int(((grades == 1) & (pred == 0)).sum()),
               severe_missed=int(((grades == 2) & (pred == 0)).sum()))
    if task == 'binary':
        out.update(sensitivity=recalls[1], specificity=recalls[0],
                   positive_predictive_value=float(cm[1, 1] / cm[:, 1].sum()) if cm[:, 1].sum() else None,
                   false_positives=int(cm[0, 1]))
    return out


class BinaryData:
    def __init__(self, config, genes=True, threshold=False):
        self.config = config
        if sha(config['csv']) != config['csv_sha256'] or sha(config['protocol']) != config['protocol_sha256']:
            raise ValueError('Prepared inputs changed')
        self.protocol = read(config['protocol'])
        with Path(config['csv']).open(encoding='utf-8-sig', newline='') as stream:
            self.rows = {r['sample_id']: r for r in csv.DictReader(stream)}
        self.ids = [r['sample_id'] for r in self.protocol['development_entries']]
        self.entries = {r['sample_id']: r for r in self.protocol['development_entries']}
        self.index = {sid: j for j, sid in enumerate(self.ids)}
        self.groups = {s: self.entries[s]['cv_component'] for s in self.ids}
        self.grades = {s: int(self.rows[s]['grade']) for s in self.ids}
        self.genes, self.threshold = genes, threshold
        if set(self.rows) != set(self.ids) or len(self.ids) != 117:
            raise ValueError('Unexpected development scope')
        locked = set(self.protocol['locked_previous_test_ids'] + self.protocol['additional_related_locked_ids'])
        if locked & set(self.ids):
            raise ValueError('Locked records included')
        held = []
        for fold in self.protocol['folds']:
            roles = {k: v['sample_ids'] for k, v in fold.items() if isinstance(v, dict) and 'sample_ids' in v}
            if set(roles['inner_train']) | set(roles['inner_val']) != set(roles['outer_train']):
                raise ValueError('Inner membership changed')
            for a, b in [('outer_train', 'outer_holdout'), ('inner_train', 'inner_val')]:
                if {self.groups[s] for s in roles[a]} & {self.groups[s] for s in roles[b]}:
                    raise ValueError('Patient group leakage')
            held.extend(roles['outer_holdout'])
        if len(held) != 117 or set(held) != set(self.ids):
            raise ValueError('Invalid five folds')

    def target(self, ids, task):
        return np.asarray([int(self.grades[s] > 0) if task == 'binary' else self.grades[s] for s in ids])

    def frame(self, ids):
        import pandas as pd
        columns = HISTORY_KEYS + ['TCT', 'dna_report_abnormal_ratio', 'methylation_total']
        if self.genes:
            columns += GENES
        if self.threshold:
            columns += ['DNA_DI_2_5']
        values = []
        for sid in ids:
            r = self.rows[sid]
            item = {k: number(r[f]) for k, f in zip(HISTORY_KEYS, HISTORY)}
            text = r['TCT意见'].strip()
            item.update(TCT=np.nan if text in MISSING else text,
                        dna_report_abnormal_ratio=number(r['dna_report_abnormal_ratio']),
                        methylation_total=number(r['甲基化总数']))
            if self.genes:
                item.update({g: number(r[g]) for g in GENES})
            if self.threshold:
                raw = number(r['dna_threshold'])
                item['DNA_DI_2_5'] = None if raw is None else float(raw >= 2.5)
            values.append(item)
        frame = pd.DataFrame(values, columns=columns)
        frame['TCT'] = frame['TCT'].astype(object)
        return frame

    def photos(self, kind):
        cfg = self.config['images'][kind]
        path = Path(cfg['features'])
        meta = read(path.parent / 'metadata.json')
        if sha(path) != cfg['features_sha256'] or meta['encoder']['sha256'] != cfg['encoder_sha256']:
            raise ValueError('Image cache/encoder differs')
        if sha(cfg['encoder']) != cfg['encoder_sha256']:
            raise ValueError('Encoder changed')
        cache = np.load(path, allow_pickle=False)
        ids = list(cache['sample_ids'])
        if set(ids) != set(self.ids) or len(ids) != len(self.ids):
            raise ValueError('Photo scope differs')
        order = [ids.index(s) for s in self.ids]
        x, present = cache['features'][order], cache['image_present'][order].astype(bool)
        if not np.isfinite(x).all() or x.shape != (117, 5, cfg['feature_dim']):
            raise ValueError('Invalid photo features')
        if not np.array_equal(cache['labels'][order], np.asarray([self.grades[s] for s in self.ids])):
            raise ValueError('Photo labels differ')
        return x, present

    def prediction_rows(self, ids, probabilities, fold):
        return [{'sample_id': s, 'grade': self.grades[s], 'fold': fold,
                 'probabilities': probabilities.get(s),
                 'status': 'predicted' if s in probabilities else 'abstain'} for s in ids]
