"""Prepare private, portable inputs; never modify the APP data or weights."""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from sysu_binary_common import GENES, HISTORY, number, sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main-root', type=Path, required=True)
    p.add_argument('--external-csv', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new preparation directory')
    conf = json.loads((a.main_root / 'configs/research/sysu_binary_fusion/methylation_total.json').read_text())
    csv_path = a.main_root / conf['csv']
    protocol_path = a.main_root / conf['protocol']
    assert sha(csv_path) == conf['csv_sha256']
    assert sha(protocol_path) == conf['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        rows = {r['sample_id']: r for r in csv.DictReader(f)}
    cells_root = a.main_root / 'data/processed/sysu/cell-image-samples/20261106'
    folders = json.loads((cells_root / 'samples.private.json').read_text(encoding='utf-8'))
    by_report, image_folders = defaultdict(dict), defaultdict(set)
    for folder in folders:
        for image in folder['candidates']:
            image_folders[image['image_sha256']].add(folder['sample_id'])
            for source in image['sources']:
                assert source['original_role'] in ('tct/fields', 'dna/fields', 'combined/fields')
                by_report[source['pdf_sha256']][folder['sample_id']] = folder
    a.output.mkdir(parents=True)
    (a.output / 'cells').mkdir()
    samples, matching = [], []
    for entry in protocol['development_entries']:
        sid = entry['sample_id']
        r = rows[sid]
        matches = by_report.get(entry['dna_report_sha256'], {})
        selected = []
        if len(matches) == 1:
            for image in next(iter(matches.values()))['candidates']:
                digest = image['image_sha256']
                if len(image_folders[digest]) != 1:
                    continue
                source = cells_root / image['path']
                assert sha(source) == digest
                dest = a.output / 'cells' / (digest + source.suffix)
                if not dest.exists():
                    shutil.copyfile(source, dest)
                selected.append(dest.relative_to(a.output).as_posix())
        samples.append({'sample_id': sid, 'grade': int(r['grade']), 'group': entry['cv_component'],
                        'genes': [number(r[g]) for g in GENES],
                        'numeric': [number(r[h]) for h in HISTORY] + [number(r['dna_report_abnormal_ratio'])],
                        'tct': r['TCT意见'].strip(), 'cells': sorted(set(selected))})
        matching.append({'sample_id': sid, 'report_folder_matches': len(matches), 'accepted_images': len(selected)})
    features_path = a.main_root / conf['images']['resnet18']['features']
    assert sha(features_path) == conf['images']['resnet18']['features_sha256']
    shutil.copyfile(features_path, a.output / 'photos.npz')
    with a.external_csv.open(encoding='utf-8-sig', newline='') as f:
        ext = list(csv.DictReader(f))
    labels = {'正常对照': 0, '白斑 OPL': 1, '口腔鳞癌 OSCC': 2}
    states = {'非甲基化': 0., '甲基化': 1.}
    external = [{'id': r['样本编号'], 'target': labels[r['分组']], 'genes': [states[r[g]] for g in GENES]} for r in ext]
    locked = set(protocol['locked_previous_test_ids'] + protocol['additional_related_locked_ids'])
    assert not locked.intersection(rows)
    # Check identities and existing patient grouping before any training.
    group = {r['sample_id']: r['group'] for r in samples}
    for fold in protocol['folds']:
        for left, right in [('outer_train', 'outer_holdout'), ('inner_train', 'inner_val')]:
            assert not {group[s] for s in fold[left]['sample_ids']} & {group[s] for s in fold[right]['sample_ids']}
    bundle = {'genes': GENES, 'samples': samples, 'external': external, 'folds': protocol['folds'],
              'sources': {'sysu_csv_sha256': sha(csv_path), 'protocol_sha256': sha(protocol_path),
                          'external_csv_sha256': sha(a.external_csv), 'photos_sha256': sha(features_path),
                          'cell_manifest_sha256': sha(cells_root / 'samples.private.json')},
              'cell_policy': 'Unique exact DNA-report SHA folder match; all unique fields; exclude images shared by folders; no montage or nucleus crops.'}
    write(a.output / 'bundle.json', bundle)
    write(a.output / 'matching.private.json', matching)
    summary = {'samples': len(samples), 'with_cells': sum(bool(r['cells']) for r in samples),
               'cell_images': len(list((a.output / 'cells').iterdir())),
               'ambiguous_report_matches': sum(r['report_folder_matches'] > 1 for r in matching),
               'external_samples': len(external)}
    write(a.output / 'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
