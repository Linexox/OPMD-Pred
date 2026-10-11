"""Synthetic API workflows and probability parity on development fold0 only."""
import argparse
import base64
import csv
import io
import json
import time
import uuid
from pathlib import Path
import server
from fastapi.testclient import TestClient
from core import Store
from inference import load_engine_config
from PIL import Image


def verify(output, raw=None):
    output.mkdir(parents=True, exist_ok=False)
    config = load_engine_config(server.DEMO / 'models.local.json')
    app = server.create_app(output / 'records', 'acceptance-only', config)
    client = TestClient(app)
    client.headers['Authorization'] = 'Bearer acceptance-only'
    checks = {}

    def write(path, body, method='POST', key=None, expected=200):
        r = client.request(method, path, json=body, headers={'X-Request-Id': key or uuid.uuid4().hex})
        assert r.status_code == expected, (path, r.status_code, r.text)
        return r.json()

    assert TestClient(app).get('/api/health').status_code == 401
    checks['pairing_required'] = True
    pid = write('/api/patients', {'alias': '工程验收（模拟）'})['id']
    body = {'patient_id': pid, 'scenario': 'self_check', 'site': '右舌侧缘', 'visit_date': '2026-10-11', 'symptoms': '模拟资料', 'payload': {'smoking': 0}}
    key = uuid.uuid4().hex
    a = write('/api/visits', body, key=key)
    assert write('/api/visits', body, key=key)['id'] == a['id']
    write('/api/visits', dict(body, site='舌背'), key=key, expected=409)
    checks['idempotency_and_content_conflict'] = True
    write(f"/api/visits/{a['id']}/submissions", {'revision': a['revision']})
    a = write(f"/api/visits/{a['id']}/reviews", {'revision': a['revision'], 'reviewer': '模拟医生', 'feedback': '模拟：请补照片', 'requested_items': ['photo'], 'material_note': '模拟：补右舌侧缘照片'})
    task = client.get(f'/api/patients/{pid}/tasks').json()[0]
    sub = write(f"/api/tasks/{task['id']}/submissions", {})
    assert sub['site'] == a['site'] and sub['scenario'] == 'self_check' and sub['id'] != a['id']
    write(f"/api/visits/{sub['id']}/submissions", {'revision': sub['revision']}, expected=422)
    # Data URLs and desktop file paths never cross the domain save API.
    image = io.BytesIO(); Image.new('RGB', (600, 700), (130, 150, 170)).save(image, format='PNG')
    photo = write('/api/assets', {'name': '模拟照片.png', 'kind': 'photo', 'data': base64.b64encode(image.getvalue()).decode()})
    write('/api/assets', {'name': '坏照片.jpg', 'kind': 'photo', 'data': base64.b64encode(b'broken').decode()}, expected=422)
    write('/api/assets', {'name': '坏报告.pdf', 'kind': 'report', 'data': base64.b64encode(b'%PDF-broken').decode()}, expected=422)
    edit = dict(body, revision=sub['revision'], task_id=task['id'], photo_id=photo['id'])
    sub = write(f"/api/visits/{sub['id']}", edit, 'PATCH')
    write(f"/api/visits/{sub['id']}", edit, 'PATCH', expected=409)
    sub = write(f"/api/visits/{sub['id']}/submissions", {'revision': sub['revision']})
    assert client.get(f'/api/patients/{pid}/tasks').json()[0]['status'] == 'submitted'
    sub = write(f"/api/visits/{sub['id']}/reviews", {'revision': sub['revision'], 'reviewer': '模拟医生', 'feedback': '模拟：照片已收到'})
    assert client.get(f'/api/patients/{pid}/tasks').json()[0]['status'] == 'confirmed'
    checks['photo_task_closed_loop'] = True
    edit.update(revision=sub['revision'], symptoms='模拟：修改后重新复核')
    sub = write(f"/api/visits/{sub['id']}", edit, 'PATCH')
    assert not sub['feedback'] and len(sub['feedback_history']) == 1
    assert client.get(f'/api/patients/{pid}/tasks').json()[0]['status'] == 'pending'
    checks['edit_reopens_task_and_preserves_feedback'] = True
    from reportlab.pdfgen import canvas
    pdf = io.BytesIO(); c = canvas.Canvas(pdf); c.drawString(30, 700, 'Synthetic report - not a patient'); c.save()
    report = write('/api/assets', {'name': '模拟报告.pdf', 'kind': 'report', 'data': base64.b64encode(pdf.getvalue()).decode()})
    b = write('/api/visits', dict(body, scenario='followup', site='舌背', payload={}))
    write(f"/api/visits/{b['id']}/submissions", {'revision': b['revision']})
    write(f"/api/visits/{b['id']}/reviews", {'revision': b['revision'], 'reviewer': '模拟医生', 'feedback': '模拟：补报告', 'requested_items': ['report'], 'material_note': '模拟：补本次TCT+DNA报告'})
    t2 = next(t for t in client.get(f'/api/patients/{pid}/tasks').json() if t['visit_id'] == b['id'])
    r2 = write(f"/api/tasks/{t2['id']}/submissions", {})
    zero = {'dna_count': 0, 'dna_total': 10000, 'dna_threshold': 2.3, 'report_site': '舌背', 'report_confirmed': True}
    rb = dict(body, site='舌背', scenario='followup', revision=r2['revision'], payload=zero, report_ids=[report['id']])
    r2 = write(f"/api/visits/{r2['id']}", rb, 'PATCH')
    assert r2['payload']['dna_ratio'] == 0 and not r2['payload']['report_confirmed']
    rb.update(revision=r2['revision'], report_ids=[])
    r2 = write(f"/api/visits/{r2['id']}", rb, 'PATCH')
    assert r2['payload']['report_confirmed']
    write(f"/api/visits/{r2['id']}/submissions", {'revision': r2['revision']})
    write(f"/api/visits/{r2['id']}/reviews", {'revision': r2['revision'], 'reviewer': '模拟医生', 'feedback': '模拟：报告收到', 'due': '2026-10-20', 'task_note': '模拟：线下复诊'})
    tt = client.get(f'/api/patients/{pid}/tasks').json()
    assert next(t for t in tt if t['id'] == t2['id'])['status'] == 'confirmed'
    appointment = next(t for t in tt if t['kind'] == 'appointment')
    assert write(f"/api/tasks/{appointment['id']}/completions", {})['status'] == 'patient_reported'
    checks['report_task_zero_confirmation_and_appointment'] = True
    assert client.get('/api/assets/' + r2['reports'][0]['id']).content.startswith(b'%PDF-')
    assert r2['reports'][0]['name'] == '模拟报告.pdf'
    assert 'image' not in r2 and 'report' not in r2
    checks['attachment_access_names_and_path_isolation'] = True
    reopened = Store(output / 'records')
    assert len(reopened.visits(pid)) == 4
    assert reopened.task(task['id'])['status'] == 'pending'
    reopened.close()
    checks['restart_persistence_and_separate_sites'] = True

    def job_result(vid, revision):
        j = write(f'/api/visits/{vid}/runs', {'revision': revision})
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result = client.get('/api/runs/' + j['id']).json()
            if result['status'] not in ('queued', 'running'):
                return result
            time.sleep(.05)
        raise AssertionError('Inference timed out')

    parity = []
    if raw:
        import numpy as np
        from verify_demo import payload_from_csv
        from inference import ResearchEngine
        with (raw / 'sysu_samples.csv').open(encoding='utf-8-sig') as f:
            rows = {r['sample_id']: r for r in csv.DictReader(f)}
        held = json.loads((raw / 'completed_evidence/runs/seed42/fold0/baseline/outer_predictions.json').read_text(encoding='utf-8'))['records']
        engine = ResearchEngine(config); engine._load()
        with np.load(raw / 'completed_evidence/paired_views_final/baseline.npz', allow_pickle=False) as cache:
            ids = cache['sample_ids'].tolist()
            for record in held:
                if record['status'] == 'abstain': continue
                sid = record['sample_id']; i = ids.index(sid)
                assert sid not in engine.fit['fit_ids']
                cached = engine.predict_features(payload_from_csv(rows[sid]), cache['features'][i, 0], bool(cache['image_present'][i]))
                err = float(np.max(np.abs(np.array(cached['probabilities']) - record['probabilities'])))
                assert err < 1e-4
                parity.append({'sample_id': sid, 'cached_probability_error': err})
        selected = [r for r in held if r['status'] != 'abstain' and rows[r['sample_id']]['image_path']][:2]
        for record in selected:
            sid = record['sample_id']; row = rows[sid]
            path = raw / row['image_path']
            image_asset = write('/api/assets', {'name': path.name, 'kind': 'photo', 'data': base64.b64encode(path.read_bytes()).decode()})
            p = payload_from_csv(row)
            p.update(report_site='右舌侧缘', source_note='开发留出数据的已核对字段，仅作接口工程比较')
            v = write('/api/visits', dict(body, payload=p, photo_id=image_asset['id']))
            live = job_result(v['id'], v['revision'])
            assert live['status'] == 'predicted', live
            err = float(np.max(np.abs(np.array(live['result']['probabilities']) - record['probabilities'])))
            assert err < 1e-4
            next(x for x in parity if x['sample_id'] == sid)['api_raw_image_probability_error'] = err
            updated = write('/api/visits/' + v['id'], dict(body, payload=p, revision=v['revision'], symptoms='工程核对：版本更新'), 'PATCH')
            assert client.get('/api/runs/' + live['id']).json()['stale']
            assert len(client.get('/api/visits/' + v['id'] + '/runs').json()) == 1
        checks['background_real_raw_image_parity_and_old_results'] = True
    empty = write('/api/visits', dict(body, payload={}))
    assert job_result(empty['id'], empty['revision'])['status'] == 'abstain'
    unknown = write('/api/visits', dict(body, payload={'tct': '工程验收未知TCT', 'report_confirmed': True, 'report_site': '右舌侧缘', 'source_note': '模拟：未知词表输入'}))
    u = job_result(unknown['id'], unknown['revision'])
    assert u['status'] == 'predicted' and u['result']['unknown_tct']
    checks['missing_vs_unknown_tct'] = True
    badapp = server.create_app(output / 'failed-model', 'acceptance-only', {'classifier': 'missing', 'encoder': 'missing'})
    badclient = TestClient(badapp); badclient.headers['Authorization'] = 'Bearer acceptance-only'
    def badwrite(path, body):
        r = badclient.post(path, json=body, headers={'X-Request-Id': uuid.uuid4().hex}); assert r.status_code == 200, r.text; return r.json()
    bp = badwrite('/api/patients', {'alias': '失败验收（模拟）'})['id']
    bv = badwrite('/api/visits', dict(body, patient_id=bp))
    bj = badwrite('/api/visits/' + bv['id'] + '/runs', {'revision': bv['revision']})
    for _ in range(100):
        result = badclient.get('/api/runs/' + bj['id']).json()
        if result['status'] == 'failed': break
        time.sleep(.05)
    assert result['status'] == 'failed' and 'probabilities' not in result['result']
    checks['model_failure_without_placeholder'] = True
    z = write('/api/visits', dict(body, payload={'dna_count':0,'dna_total':0,'dna_threshold':2.5}))
    assert z['payload']['dna_ratio'] is None
    write('/api/visits', dict(body, payload={'dna_count':11,'dna_total':10,'dna_threshold':2.3}), expected=422)
    write('/api/visits', dict(body, payload={'dna_threshold':3}), expected=422)
    checks['undefined_ratio_and_invalid_inputs'] = True
    result = {'date': '2026-10-11', 'checks': checks, 'holdout_comparisons': parity, 'old_test22_evaluated': False,
              'real_doctor_reviews': 0, 'real_user_trials': 0, 'synthetic_records_are_engineering_evidence': True}
    (output / 'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'checks':checks,'compared':len(parity)}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--raw-data', type=Path); args = parser.parse_args()
    verify(args.output, args.raw_data)

if __name__ == '__main__': main()
