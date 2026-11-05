"""Run actual CPU model parity and the mobile API against an isolated private store."""
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--reuse-component-parity', action='store_true')
    args = parser.parse_args()
    project = args.project.resolve()
    out = project / 'outputs/checks/daily/2026-11-05/app-mean-deployment'
    code = project / 'apps/server/binary_runtime'
    server = project / 'apps/server'
    for path in reversed([server, code, *[project / 'runtime' / name for name in
                         ('binary-research-deps-v3', 'binary-research-deps-v2', 'binary-research-deps')]]):
        sys.path.insert(0, str(path))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_HOME', str(project / 'runtime/cache/huggingface'))
    os.chdir(code)
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import opmd_pred.model.two_branch_fusion  # Pin frozen implementation before server imports src.
    from mean_ensemble_predictor import MeanEnsemblePredictor
    from sysu_tabular_common import offline_predictions
    from sysu_binary_common import write, sha
    from binary_inputs import binary_fields_from_visit
    from server import create_app
    from fastapi.testclient import TestClient
    engine_file = out / 'staged/binary_inference.py' if args.staged else server / 'binary_inference.py'
    module_spec = importlib.util.spec_from_file_location('checked_binary_engine', engine_file)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    fixtures = json.loads((out / 'fixed-inputs.private.json').read_text(encoding='utf-8'))
    manifest = project / 'models/app/binary-mean/manifest.json'
    model_id = json.loads(manifest.read_text(encoding='utf-8'))['model_id']
    start = time.perf_counter(); checks = []
    with offline_predictions():
        predictor = MeanEnsemblePredictor(manifest, device='cpu')
        assert len(predictor.members) == 5 and len(predictor.tct_reference_vocabularies) == 25
        assert all(p.encoder is predictor.members[0].encoder for p in predictor.members)
        if args.reuse_component_parity:
            diagnosis = json.loads((out / 'cpu-parity-diagnosis.json').read_text())
            assert len(diagnosis['records']) == 3
            assert all(r['same_binary_class'] and r['mean_probability_error'] <= 1e-3 for r in diagnosis['records'])
            checks = [{'original_grade': r['original_grade'], 'mean_probability_max_error': r['mean_probability_error'],
                       'reused_completed_component_check': True} for r in diagnosis['records']]
        for case in ([] if args.reuse_component_parity else fixtures['cases']):
            photo = project / case['photo']
            assert sha(photo) == case['photo_sha256']
            mapped = binary_fields_from_visit(case['payload'])
            assert mapped == case['fields']
            result = predictor.predict(mapped, photo)
            member_error = float(np.abs(np.asarray(result['member_probabilities']) - case['expected_member_probabilities']).max())
            error = float(np.abs(np.asarray(result['probabilities']) - case['expected_probabilities']).max())
            assert error <= 1e-3, f'CPU/GPU probability difference exceeds 0.1 percentage point: {error}'
            assert result['prediction'] == int(case['expected_probabilities'][1] >= .5)
            manual = np.mean(result['member_probabilities'], axis=0)
            np.testing.assert_allclose(manual, result['probabilities'], rtol=0, atol=1e-12)
            checks.append({'original_grade': case['grade'], 'mean_probability_max_error': error,
                           'member_probability_max_error': member_error, 'seconds': result['timings']['total_seconds']})
        assert predictor.predict({}, None)['status'] == 'no_input'
        zero = predictor.predict({'dna_report_abnormal_ratio': 0.}, None)
        assert zero['status'] == 'predicted' and zero['table_present']
        photo_only = predictor.predict({}, project / fixtures['cases'][0]['photo'])
        assert photo_only['status'] == 'predicted' and not photo_only['table_present']
        engine = module.BinaryResearchEngine(predictor, model_id)
        http_root = out / 'http-check-store'
        app = create_app(http_root, 'isolated-mean-model-check', {}, research_engine=engine)
        headers = {'Authorization': 'Bearer isolated-mean-model-check', 'X-Result-Formats': 'binary-v1'}
        with TestClient(app) as client:
            def post(path, body):
                response = client.post(path, json=body, headers={**headers, 'X-Request-Id': uuid.uuid4().hex})
                assert response.status_code == 200, (path, response.status_code)
                return response.json()
            def wait(job):
                deadline = time.monotonic() + 240
                while time.monotonic() < deadline:
                    response = client.get('/api/runs/' + job['id'], headers=headers)
                    assert response.status_code == 200
                    value = response.json()
                    if value['status'] not in ('queued', 'running'):
                        assert value['status'] == 'predicted', value.get('error')
                        return value
                    time.sleep(.1)
                raise TimeoutError('Analysis did not finish')
            patient = post('/api/patients', {'alias': '接口核对'})
            case = fixtures['cases'][0]
            image = project / case['photo']
            asset = post('/api/assets', {'kind': 'photo', 'name': 'fixed-photo' + image.suffix,
                                         'data': base64.b64encode(image.read_bytes()).decode('ascii')})
            visit = post('/api/visits', {'patient_id': patient['id'], 'scenario': 'followup', 'site': '右舌侧缘',
                'visit_date': '2026-11-05', 'symptoms': '', 'payload': case['payload'], 'photo_id': asset['id']})
            first = wait(post('/api/visits/' + visit['id'] + '/runs', {'revision': visit['revision']}))
            result = first['result']
            assert result['model_id'] == model_id and result['model_fingerprint']
            assert result['class_labels'] == ['正常/轻度', '中度及以上']
            assert result['grade_reference']['reference_id'] == 'mean5-sixfold-oof-20261105'
            np.testing.assert_allclose(result['probabilities'], case['expected_probabilities'], atol=1e-3, rtol=0)
            assert 'photo_feature' not in result and 'member_probabilities' not in result
            changed = {**case['payload'], 'methylation': 2 if case['payload'].get('methylation') != 2 else 1}
            response = client.patch('/api/visits/' + visit['id'], headers={**headers, 'X-Request-Id': uuid.uuid4().hex},
                json={'revision': visit['revision'], 'patient_id': patient['id'], 'scenario': 'followup',
                      'site': '右舌侧缘', 'visit_date': '2026-11-05', 'symptoms': '', 'payload': changed, 'photo_id': asset['id']})
            assert response.status_code == 200
            second_visit = response.json()
            second = wait(post('/api/visits/' + visit['id'] + '/runs', {'revision': second_visit['revision']}))
            history = client.get('/api/visits/' + visit['id'] + '/runs', headers=headers).json()
            saved_current = next(r for r in history if r['revision'] == second_visit['revision'])
            compared = client.get('/api/visits/' + visit['id'] + '/run-comparison?current_run_id=' + saved_current['id'], headers=headers)
            assert compared.status_code == 200 and compared.json()['status'] == 'compared'
            assert any(v['modality'] == 'methylation' and v['kind'] in ('added', 'changed') for v in compared.json()['changed_inputs'])
            assert client.get('/api/health').status_code == 401
    report = {'passed': True, 'fixed_inputs': checks, 'all_missing_rejected': True,
        'real_zero_used': True, 'photo_only_supported': True, 'complete_models': 5, 'table_references': 25,
        'equal_weights': [.2] * 5, 'api_analysis_and_revision_comparison': True,
        'grade_reference_updated': True, 'private_features_not_returned': True,
        'cpu_gpu_probability_tolerance': 1e-3, 'same_class_required': True,
        'model_id': model_id, 'manifest_sha256': sha(manifest), 'elapsed_seconds': time.perf_counter() - start}
    write(out / 'real-cpu-and-http-check.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
