"""One prespecified distance-loss pair; continue seeds only if the serious-error gate passes."""
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PERSONAL = Path('/home/zjkj/24334067')
WORK = PERSONAL / 'OPMD-Pred-SYSU-Proxy-Balanced'
INPUT = PERSONAL / 'sysu-research-inputs/raw-rebuild-20261009'
OUTPUT = WORK / 'outputs/sysu_distance_20261010_attempt2'
PYTHON = PERSONAL / 'OPMD-Pred/.venv/bin/python'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def result(path):
    data = json.loads((path / 'oof_results.json').read_text())['results']['baseline']
    return data['metrics']['fixed_median_0.5'], data['records']


def reference(seed):
    if seed == 42:
        return WORK / 'outputs/sysu_raw_medical_cv_20261009'
    return WORK / f'outputs/sysu_raw_medical_multiseed_20261009/seed{seed}'


def main():
    assert WORK.resolve().is_relative_to(PERSONAL)
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    gate = {'correct_count_drop_max': 2, 'severe_to_mild_reduction_min': 1,
            'cross_two_levels_must_not_increase': True, 'seed42_min_correct': 63}
    study = {'date': '2026-10-10', 'distance_weights': [0., .5], 'initial_seed': 42,
             'conditional_seeds': [43, 44], 'gate': gate, 'budget_seconds': 7200,
             'old_test_used': False, 'status': 'running', 'runs': [],
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(OUTPUT / 'study.json', study)
    for seed in (42, 43, 44):
        pair = {}
        for weight in (0., .5):
            path = OUTPUT / f'seed{seed}' / ('control' if weight == 0 else 'distance05')
            path.mkdir(parents=True)
            remaining = 7200 - (time.monotonic() - started)
            if remaining <= 0:
                study['status'] = 'budget_stopped'; write(OUTPUT / 'study.json', study); return
            command = [str(PYTHON), str(WORK / 'scripts/research/train_distance_cv.py'),
                       '--csv', str(INPUT / 'sysu_samples.csv'), '--cv-protocol', str(INPUT / 'development_nested_cv_v3.json'),
                       '--cache', str(INPUT / 'paired_views_final'), '--output', str(path),
                       '--distance-weight', str(weight), '--seed-offset', str(seed - 42),
                       '--time-budget-seconds', str(remaining)]
            write(path / 'command.json', command)
            with (path / 'run.log').open('w') as log:
                environment = dict(os.environ, PYTHONPATH=str(WORK/'src'), OMP_NUM_THREADS='2')
                code = subprocess.call(command, cwd=WORK, env=environment, stdout=log, stderr=subprocess.STDOUT)
            study['runs'].append({'seed': seed, 'weight': weight, 'return_code': code, 'path': str(path)})
            if code:
                study['status'] = 'budget_stopped' if code == 75 else 'failed'
                write(OUTPUT / 'study.json', study); return
            metrics, rows = result(path)
            pair[weight] = metrics
            if weight == 0:
                expected, previous_rows = result(reference(seed))
                by_id = {r['sample_id']: r for r in previous_rows}
                errors = []
                for row in rows:
                    previous = by_id[row['sample_id']]
                    assert row['status'] == previous['status'] and row['label'] == previous['label']
                    if row['probabilities']:
                        errors.append(max(abs(a-b) for a,b in zip(row['probabilities'], previous['probabilities'])))
                checks = {k: metrics[k] == expected[k] for k in (
                    'correct_count', 'severe_to_mild_count', 'two_level_error_count', 'assigned_count', 'abstained_count')}
                checks['probabilities_aligned'] = max(errors, default=0.) <= 1e-5
                write(path / 'reproduction.json', {'checks': checks, 'max_probability_error': max(errors, default=0.)})
                if not all(checks.values()):
                    study['status'] = 'control_mismatch'; write(OUTPUT / 'study.json', study); return
        c, n = pair[0.], pair[.5]
        passed = (n['correct_count'] >= c['correct_count']-2 and
                  n['severe_to_mild_count'] <= c['severe_to_mild_count']-1 and
                  n['two_level_error_count'] <= c['two_level_error_count'])
        write(OUTPUT / f'seed{seed}/comparison.json', {'control': c, 'distance05': n, 'gate_passed': passed})
        if seed == 42 and not passed:
            study['status'] = 'completed_negative'; write(OUTPUT / 'study.json', study); return
    means = {}
    for name in ('control', 'distance05'):
        values = [result(OUTPUT / f'seed{s}' / name)[0] for s in (42, 43, 44)]
        means[name] = {k: statistics.mean(v[k] for v in values) for k in (
            'correct_count', 'accuracy_all_assigned', 'severe_to_mild_count', 'two_level_error_count', 'mae')}
    c, n = means['control'], means['distance05']
    passed = n['correct_count'] >= c['correct_count']-2 and n['severe_to_mild_count'] <= c['severe_to_mild_count']-1 and n['two_level_error_count'] <= c['two_level_error_count']
    study.update(status='completed', mean_metrics=means, final_gate_passed=passed)
    write(OUTPUT / 'study.json', study)


if __name__ == '__main__': main()
