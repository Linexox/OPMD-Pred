"""Prepare private fixed inputs for deployment parity, not a new accuracy evaluation."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/checks/daily/2026-11-05/app-mean-deployment'
MISSING = {'', 'nan', 'NAN', 'NA', 'N/A', '/', '#VALUE!', 'None', 'null', 'NULL'}


def number(value):
    if value is None or str(value).strip() in MISSING:
        return None
    if str(value).strip() in {'是', '有', 'yes', 'Yes'}:
        return 1.
    if str(value).strip() in {'否', '无', 'no', 'No'}:
        return 0.
    return float(value)


def main():
    source = ROOT / 'outputs/experiments/sysu/full-model-moe-20261104/attempt1/deployment/inference-results.json'
    expected = json.loads(source.read_text(encoding='utf-8'))['methods']['mean']['records']
    h0 = np.load(source.parent / 'inference-inputs.npz', allow_pickle=False)
    images = np.load(ROOT / 'data/processed/zenodo/first-step-supplement/features/resnet_balance/candidate/baseline.npz', allow_pickle=False)
    csv_path = ROOT / 'data/processed/sysu/binary-table-fusion-20261029/attempt1/sysu_samples.csv'
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = {r['sample_id']: r for r in csv.DictReader(stream)}
    history = ['是否吸烟', '吸烟年限（年）', '吸烟量（支/天）', '是否饮酒', '饮酒年限(年)',
               '饮酒量(ml/天)', '是否嚼槟榔', '嚼槟榔年限(年)', '嚼槟榔量（颗/天）']
    keys = ['smoking', 'smoking_years', 'smoking_amount', 'alcohol', 'alcohol_years',
            'alcohol_amount', 'betel', 'betel_years', 'betel_amount']
    cases = []
    OUT.mkdir(parents=True, exist_ok=True)
    for grade in range(3):
        for row in expected:
            raw = rows[row['sample_id']]
            photo = (csv_path.parent / raw['image_path']).resolve()
            if row['grade'] != grade or row['probabilities'] is None or not photo.is_file():
                continue
            fields = {k: number(raw[h]) for k, h in zip(keys, history)}
            fields.update(TCT=None if raw['TCT意见'].strip() in MISSING else raw['TCT意见'].strip(),
                          dna_report_abnormal_ratio=number(raw['dna_report_abnormal_ratio']),
                          methylation_total=number(raw['甲基化总数']))
            count, total = number(raw['dna_abnormal_cells']), number(raw['dna_analyzed_cells'])
            ratio = fields['dna_report_abnormal_ratio']
            if ratio is not None and (count is None or not total or abs(count / total - ratio) > 1e-10):
                continue
            target = OUT / f'fixed-{grade}{photo.suffix.lower()}'
            shutil.copyfile(photo, target)
            payload = {k: fields[k] for k in keys}
            payload.update(tct=fields['TCT'], dna_count=count, dna_total=total,
                           dna_threshold=number(raw['dna_threshold']), methylation=fields['methylation_total'])
            cases.append({'sample_id': row['sample_id'], 'grade': grade, 'fields': fields,
                          'payload': payload, 'photo': target.relative_to(ROOT).as_posix(),
                          'photo_sha256': hashlib.sha256(photo.read_bytes()).hexdigest(),
                          'expected_h0': h0['h0'][h0['sample_ids'].tolist().index(row['sample_id'])].tolist(),
                          'expected_photo_feature': images['features'][images['sample_ids'].tolist().index(row['sample_id']), 0].tolist(),
                          'expected_member_probabilities': row['complete_expert_probabilities'],
                          'expected_probabilities': row['probabilities']})
            break
    assert len(cases) == 3, 'Need one available photo case per original grade'
    (OUT / 'fixed-inputs.private.json').write_text(json.dumps({'cases': cases,
        'scope': 'Numerical deployment parity on existing development inputs, not an accuracy study',
        'probability_tolerance': 1e-4, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print({'fixed_cases': len(cases), 'contains_patient_material': True, 'server_only': True})


if __name__ == '__main__':
    main()
