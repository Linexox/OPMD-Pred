"""Package five complete classifiers, each with its own five TabPFN references."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/experiments/sysu/full-model-moe-20261104/attempt1'
DEST = ROOT / 'models/app/binary-mean'
MODEL_ID = 'sysu-resnet18-tabpfn-transformer-mean5-seed42-20261105'
MISSING = {'', 'nan', 'NAN', 'NA', 'N/A', '/', '#VALUE!', 'None', 'null', 'NULL'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise ValueError('Existing package file differs: ' + str(path))
    path.write_text(content, encoding='utf-8')


def copy(source, destination, expected):
    assert sha(source) == expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_name(destination.name + '.incoming')
        shutil.copyfile(source, temporary)
        assert sha(temporary) == expected
        temporary.replace(destination)
    assert sha(destination) == expected


def main():
    original = read(ROOT / 'models/app/binary/manifest.json')
    frozen = read(SOURCE / 'freeze.json')
    base = frozen['base_config']
    assert sha(ROOT / 'src/opmd_pred/model/two_branch_fusion.py') == frozen['sources']['src/opmd_pred/model/two_branch_fusion.py']
    result_path = SOURCE / 'deployment/inference-results.json'
    source = read(result_path)
    assert len(source['training_ids']) == 117 and len(source['expert_specs']) == 5
    assert frozen['config']['expert_epochs'] == 7 and frozen['config']['seed'] == 42
    assert sha(ROOT / base['csv']) == base['csv_sha256']
    with (ROOT / base['csv']).open(encoding='utf-8-sig', newline='') as stream:
        rows = {row['sample_id']: row for row in csv.DictReader(stream)}
    shared = ['encoder.pt', 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt']
    for name in shared:
        copy(ROOT / 'models/app/binary' / name, DEST / name, original['sources_sha256'][name])
    members, provenance = [], []
    for e, spec in enumerate(source['expert_specs']):
        folder = DEST / f'expert{e}'
        model = ROOT / spec['model']
        assert sha(model) == spec['model_sha256']
        artifact = torch.load(model, map_location='cpu', weights_only=True)
        assert set(artifact['identity']['training_ids']) == set(spec['fit_ids'])
        copy(model, folder / 'fusion.pt', spec['model_sha256'])
        cache = ROOT / spec['table_cache']; table = read(cache / 'done.json')
        assert set(table['identity']['fit_ids']) == set(spec['fit_ids'])
        assert len(table['assignments']) == 5
        refs, vocabulary, fingerprints = [], [], {'../' + p: original['sources_sha256'][p] for p in shared}
        fingerprints['fusion.pt'] = spec['model_sha256']
        for assignment in table['assignments']:
            k = assignment['part']; name = f'table-{k}.tabpfn_fit'
            expected = table['files'][f'part{k}/classifier.tabpfn_fit']
            copy(cache / f'part{k}/classifier.tabpfn_fit', folder / name, expected)
            refs.append(name); fingerprints[name] = expected
            values = sorted({rows[s]['TCT意见'].strip() for s in assignment['fit_ids']
                             if rows[s]['TCT意见'].strip() not in MISSING})
            vocabulary.append({'model_sha256': expected, 'tct_values': values})
        vocab = {'format': 'binary-tct-reference-vocabulary-v1', 'references': vocabulary,
                 'source_csv_sha256': base['csv_sha256'], 'branch_sha256': sha(cache / 'done.json')}
        write(folder / 'tct-reference-vocabulary.json', vocab)
        fingerprints['tct-reference-vocabulary.json'] = sha(folder / 'tct-reference-vocabulary.json')
        manifest = {'seed': 42 + e, 'fold_directory': f'expert{e}', 'columns': original['columns'],
            'tabpfn_version': original['tabpfn_version'], 'encoder': '../encoder.pt', 'fusion': 'fusion.pt',
            'table_reference_models': refs, 'sources_sha256': fingerprints, 'server_only': True,
            'identity_extra_sources': ['../tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'],
            'tct_vocabulary': 'tct-reference-vocabulary.json', 'path_base': 'manifest'}
        write(folder / 'manifest.json', manifest)
        members.append({'manifest': f'expert{e}/manifest.json', 'seed': 42 + e,
                        'fold_directory': f'expert{e}', 'weight': .2})
        provenance.append({'slot': e, 'source_fusion': spec['model'], 'source_fusion_sha256': spec['model_sha256'],
                           'source_table_cache': spec['table_cache'], 'training_records': len(spec['fit_ids'])})
    cv_path = ROOT / 'outputs/experiments/sysu/moe-regularization-20261105/attempt2/oof_results.json'
    cv = read(cv_path)['methods']['original_mean']
    anchors = []
    for g, label in enumerate(['正常/轻度', '中度', '重度']):
        values = [row['probabilities'][1] for row in cv['records'] if row['grade'] == g and row['probabilities'] is not None]
        anchors.append({'label': label, 'count': len(values), 'score': statistics.median(values)})
    files = {p.relative_to(DEST).as_posix(): sha(p) for p in DEST.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    for member in members:
        files[member['manifest']] = sha(DEST / member['manifest'])
    manifest = {'format': 'sysu-five-complete-model-mean-v1', 'model_id': MODEL_ID,
        'seed': 42, 'members': members, 'columns': original['columns'], 'sources_sha256': files,
        'classifier_dropout_during_training': .1, 'decision_threshold': .5, 'server_only': True,
        'training': {'development_records': 117, 'five_original_partitions': True,
                     'each_member_uses_four_partitions': True, 'epochs': 7},
        'grade_reference': {'method': 'original_grade_median', 'reference_id': 'mean5-sixfold-oof-20261105',
            'score_index': 1, 'records': 112, 'scope': 'six-fold held-out development scores; pooled across folds',
            'anchors': anchors},
        'evaluation': {'scope': 'same architecture assessed by six outer folds; not a score of the final deployment weights',
            'source_sha256': sha(cv_path), 'fold_mean_accuracy': cv['fold_accuracy_mean'],
            'pooled_metrics': cv['metrics']}}
    write(DEST / 'manifest.json', manifest)
    audit = {'passed': True, 'model_id': MODEL_ID, 'artifact_files': len(files),
             'manifest_sha256': sha(DEST / 'manifest.json'), 'source_result_sha256': sha(result_path),
             'provenance': provenance, 'router_used': False, 'best_test_fold_not_selected': True}
    write(ROOT / 'outputs/checks/daily/2026-11-05/app-mean-deployment/package.json', audit)
    print({k: v for k, v in audit.items() if k != 'provenance'})


if __name__ == '__main__':
    main()
