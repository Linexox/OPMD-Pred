"""Verify saved Android runs, explanation arithmetic and missing-input behavior."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import urllib.error
import urllib.request

import server
from explanation import compare_saved_runs
from inference import ResearchEngine, load_engine_config, LABELS


def verify_result(result):
    assert result['status'] == 'predicted'
    explanation = result['explanation']
    assert explanation['status'] == 'ready', explanation
    assert explanation['classifier_sha256'] == result['classifier_sha256']
    assert explanation['rule'] == result['rule']
    assert explanation['replay_error'] <= 1e-4
    errors = []
    for index, target in enumerate(explanation['targets']):
        assert abs(target['value'] - result['probabilities'][index]) < 1e-10
        errors.append(abs(target['reference_value'] + sum(c['value'] for c in target['contributions']) - target['value']))
        assert errors[-1] < 1e-6
        assert all(abs(c['value']) < 1e-12 for c in target['contributions'] if not c['present'])
    checks = explanation['decision_checks']
    assert abs(checks[0]['value'] - sum(result['probabilities'][1:])) < 1e-10
    assert abs(checks[1]['value'] - result['probabilities'][2]) < 1e-10
    for check in checks:
        assert check['passed'] == (check['value'] > .5)
        assert abs(check['reference_value'] + sum(c['value'] for c in check['contributions']) - check['value']) < 1e-6
    assert result['median_class'] == LABELS[sum(check['passed'] for check in checks)]
    return max(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--service-data', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    token = (args.service_data/'pairing-token.txt').read_text(encoding='utf-8').strip()

    def get(path, authenticated=True, status=200):
        request = urllib.request.Request(args.url+path, headers={'Authorization': 'Bearer '+token} if authenticated else {})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                assert response.status == status
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            assert exc.code == status, exc
            return json.loads(exc.read())

    checks = {}
    get('/api/health', authenticated=False, status=401)
    example = next(x for x in get('/api/demo-examples') if x['key'] == 'photo')
    vid = example['visit_id']
    visit = get('/api/visits/'+vid)
    runs = get('/api/visits/'+vid+'/runs')
    actual = [r for r in runs if r['result'].get('explanation', {}).get('status') == 'ready']
    assert len(actual) >= 2
    current = actual[0]
    previous = next(r for r in actual if r['revision'] < current['revision'])
    arithmetic = [verify_result(r['result']) for r in (current, previous)]
    compared = get(f'/api/visits/{vid}/run-comparison?current_run_id={current["id"]}&previous_run_id={previous["id"]}')
    assert compared['status'] == 'compared'
    assert {x['modality'] for x in compared['changed_inputs'] if x['kind'] == 'added'} == {'tct', 'dna', 'methylation'}
    assert next(x for x in compared['changed_inputs'] if x['modality'] == 'image')['kind'] == 'unchanged'
    assert compared['probability_deltas'] == [a-b for a,b in zip(current['result']['probabilities'], previous['result']['probabilities'])]
    assert visit['reports'][0]['name'] == 'opmd-demo-report-20261017.png', visit['reports']
    checks['live_saved_explanations_and_original_report_name'] = True
    checks['same_revision_chain_and_unchanged_photo'] = True
    checks['pairing_required'] = True
    get(f'/api/visits/{vid}/run-comparison?current_run_id=not-a-run', status=404)
    get(f'/api/visits/{vid}/run-comparison?current_run_id={current["id"]}&previous_run_id=not-a-run', status=404)
    for field, value in [('classifier_sha256', 'other-model'), ('image_encoder_sha256', 'other-encoder'), ('rule', 'other-rule')]:
        changed = deepcopy(previous)
        changed['result'][field] = value
        assert compare_saved_runs(current, changed)['status'] == 'unavailable'
    for field, value in [('patient_id', 'other-patient'), ('site', '右舌侧缘'), ('visit_date', '2026-10-18')]:
        changed = deepcopy(previous)
        changed['result']['input_context'][field] = value
        assert compare_saved_runs(current, changed)['status'] == 'unavailable'
    changed = deepcopy(previous)
    changed['visit_id'] = changed['result']['input_context']['visit_id'] = 'unrelated-visit'
    assert compare_saved_runs(current, changed)['status'] == 'unavailable'
    assert compare_saved_runs(previous, current)['status'] == 'unavailable'
    legacy = deepcopy(previous)
    legacy['result'].pop('model_inputs')
    assert compare_saved_runs(current, legacy)['status'] == 'unavailable'
    checks['different_models_sites_people_dates_and_legacy_results_not_mixed'] = True

    # Explicit source task comparison, without changing the actual saved records.
    task_current = deepcopy(current)
    task_current['visit_id'] = task_current['result']['input_context']['visit_id'] = 'synthetic-task-submission'
    task_current['result']['input_context']['task_id'] = 'synthetic-material-task'
    task_result = compare_saved_runs(task_current, previous, previous['visit_id'], previous['revision'])
    assert task_result['status'] == 'compared' and task_result['previous']['relation'] == 'material_task_source'
    assert compare_saved_runs(task_current, previous, previous['visit_id'], previous['revision']+1)['status'] == 'unavailable'
    checks['task_comparison_requires_exact_source_revision'] = True

    # No-image synthetic inputs use the real pinned classifier. No hospital case or label is changed.
    import numpy as np
    engine = ResearchEngine(load_engine_config(server.HERE/'models.local.json'))
    features = np.zeros(1536, dtype=np.float32)
    assert engine.predict_features({}, features, False)['status'] == 'abstain'
    assert engine.predict_features({'tct': '未见核异质细胞', 'report_confirmed': False}, features, False)['status'] == 'abstain'
    zero = engine.predict_features({'dna_count': 0, 'dna_total': 1000, 'dna_threshold': 2.5, 'report_confirmed': True}, features, False)
    assert zero['present']['dna'] and zero['status'] == 'predicted'
    verify_result(zero)
    unknown = engine.predict_features({'tct': '验收专用未知表述', 'report_confirmed': True}, features, False)
    assert unknown['unknown_tct'] and unknown['present']['tct']
    verify_result(unknown)
    checks['all_missing_unconfirmed_zero_and_unknown_tct'] = True

    # Compare one previously held-out development record to its archived probability vector.
    from workspace_paths import SYSU_BASELINE_EVIDENCE, SYSU_DATASET
    raw = SYSU_DATASET
    from verification_helpers import payload_from_csv
    with (raw/'sysu_samples.csv').open(encoding='utf-8-sig') as stream:
        rows = {r['sample_id']: r for r in csv.DictReader(stream)}
    cv = json.loads((raw/'development_nested_cv_v3.json').read_text(encoding='utf-8'))
    locked = set(cv['locked_previous_test_ids']+cv['additional_related_locked_ids'])
    held = json.loads((SYSU_BASELINE_EVIDENCE/'runs/seed42/fold0/baseline/outer_predictions.json').read_text(encoding='utf-8'))['records']
    selected = next(r for r in held if r['status'] != 'abstain')
    sid = selected['sample_id']
    assert sid not in locked and sid not in engine.fit['fit_ids']
    with np.load(SYSU_BASELINE_EVIDENCE/'paired_views_final/baseline.npz', allow_pickle=False) as cache:
        index = cache['sample_ids'].tolist().index(sid)
        replay = engine.predict_features(payload_from_csv(rows[sid]), cache['features'][index, 0], bool(cache['image_present'][index]))
    parity_error = float(np.max(np.abs(np.asarray(replay['probabilities'])-selected['probabilities'])))
    assert parity_error < 1e-4
    verify_result(replay)
    checks['one_frozen_development_holdout_probability_parity'] = True
    evidence = dict(checks=checks, live_current=current, live_previous=previous, comparison=compared,
                    maximum_reconstruction_error=max(arithmetic), heldout_probability_error=parity_error,
                    selected_development_record=sid, locked_22_evaluated=False, training=False,
                    real_doctor_or_patient_trial=False, live_api_writes=False)
    (args.output/'verification.private.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(checks=checks, maximum_reconstruction_error=max(arithmetic), heldout_probability_error=parity_error), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
