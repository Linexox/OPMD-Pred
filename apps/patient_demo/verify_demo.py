"""Engineering acceptance on synthetic records and frozen fold0 holdouts, never old test22."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import bootstrap
import numpy as np
from PIL import Image
from core import Store, digest, quality, validate_payload
from inference import ResearchEngine, load_engine_config, ENCODER_SHA, CLASSIFIER_SHA
from export_pdf import export_summary


def expect_error(action):
    try:
        action()
    except (ValueError, OSError):
        return
    raise AssertionError('Expected explicit failure')


def workflows(output):
    root = output / 'synthetic_records'
    s = Store(root)
    p = s.add_patient('验收用户（模拟）')
    photo = output / 'synthetic-quality-pattern.png'
    yy, xx = np.indices((520, 700))
    pattern = (((xx // 16 + yy // 16) % 2) * 180 + 40).astype(np.uint8)
    Image.fromarray(pattern).convert('RGB').save(photo)
    bad = output / 'corrupt-image.jpg'; bad.write_bytes(b'not an image')
    expect_error(lambda: quality(bad))
    expect_error(lambda: validate_payload({'dna_count': 11, 'dna_total': 10, 'dna_threshold': 2.3}))
    expect_error(lambda: validate_payload({'dna_threshold': 3.0}))
    expect_error(lambda: validate_payload({'age': 42}))
    assert validate_payload({'dna_count': 0, 'dna_total': 10000, 'dna_threshold': 2.3})['dna_ratio'] == 0
    assert validate_payload({'dna_count': 0, 'dna_total': 0, 'dna_threshold': 2.3})['dna_ratio'] is None
    first = s.save_visit(p, 'self_check', '左颊黏膜', '2026-10-10', '模拟：准备首次就诊资料', {}, str(photo))
    original = s.visit(first)
    assert original['original_sha'] == digest(photo)
    s.submit(first)
    expect_error(lambda: s.review(first, '', '医生演示'))
    s.review(first, '演示意见：请补充资料并按约定就诊。', '医生演示', True)
    assert s.visit(first)['status'] == 'needs_more'
    s.save_visit(p, 'self_check', '左颊黏膜', '2026-10-10', '模拟：已补充文字', {}, visit_id=first)
    assert len(s.feedback_history(first)) == 1 and s.visit(first)['revision'] == 2
    s.submit(first)
    s.review(first, '演示意见：请按任务准备复诊资料。', '医生演示', due='2026-10-20', task_note='同部位照片，按医生约定就诊')
    follow = s.save_visit(p, 'followup', '左颊黏膜', '2026-10-20', '模拟：复诊记录', {})
    another_site = s.save_visit(p, 'followup', '右舌侧缘', '2026-10-20', '模拟：另一部位', {})
    assert len(s.visits(p)) == 3 and len({first, follow, another_site}) == 3
    task = s.tasks(p)[0]; s.complete_task(task['id']); assert s.tasks(p)[0]['status'] == 'patient_reported'
    # New sources are saved unconfirmed; confirmation is a separate explicit action.
    export_summary(s, first, output / 'summary-preview.pdf')
    zero = {'dna_count': 0, 'dna_total': 10000, 'dna_threshold': 2.3,
            'report_confirmed': True, 'report_site': '左颊黏膜'}
    s.save_visit(p, 'followup', '左颊黏膜', '2026-10-20', '模拟', zero,
                 report_source=[str(output / 'summary-preview.pdf')], visit_id=follow)
    assert not s.visit(follow)['payload']['report_confirmed']
    s.save_visit(p, 'followup', '左颊黏膜', '2026-10-20', '模拟', zero, visit_id=follow)
    assert s.visit(follow)['payload']['report_confirmed'] and s.visit(follow)['payload']['dna_ratio'] == 0
    s.save_visit(p, 'followup', '右舌侧缘', '2026-10-20', '模拟', zero, visit_id=follow)
    assert not s.visit(follow)['payload']['report_confirmed']
    s.save_visit(p, 'followup', '左颊黏膜', '2026-10-20', '模拟', zero, visit_id=follow)
    s.save_visit(p, 'followup', '左颊黏膜', '2026-10-20', '模拟', zero, visit_id=follow)
    assert s.visit(follow)['payload']['report_confirmed']
    changed = dict(zero, dna_count=1)
    s.save_visit(p, 'followup', '左颊黏膜', '2026-10-20', '模拟', changed, visit_id=follow)
    assert not s.visit(follow)['payload']['report_confirmed']
    s.add_run(first, 1, {'status': 'failed', 'reason': 'old revision', 'probabilities': None})
    assert s.latest_run(first) is None
    s.close()
    s = Store(root)
    assert len(s.visits(p)) == 3 and len(s.feedback_history(first)) == 2
    s.close()
    from PyQt6.QtWidgets import QApplication
    from app import Window, configure_application
    a = QApplication.instance() or QApplication([]); configure_application(a)
    s = Store(root); w = Window(s, ResearchEngine({'classifier': '', 'encoder': ''}))
    w.show(); a.processEvents()
    w.visit_id = first; w.refresh_visits(first); w.load_visit()
    assert '当前：2026-10-10' in w.after_caption.text() and '此前' not in w.before_caption.text()
    assert str(task['source_revision']) in w.task_list.item(0).text()
    w.grab().save(str(output / 'patient-entry.png'))
    w.tabs.setCurrentIndex(2); a.processEvents(); w.grab().save(str(output / 'patient-feedback.png'))
    w.role.setCurrentIndex(1); a.processEvents(); w.grab().save(str(output / 'doctor-view.png'))
    w.close()
    return {'storage_restart': True, 'separate_visits_and_sites': True, 'feedback_history': True,
            'source_binding': True, 'zero_vs_missing': True, 'invalid_inputs': True,
            'patient_pdf': True, 'synthetic_users_are_not_real_usability_participants': True}


def payload_from_csv(row):
    from opmd_pred.data.sysu import _number, _binary, MISSING
    names = ['是否吸烟', '吸烟年限（年）', '吸烟量（支/天）', '是否饮酒', '饮酒年限(年)',
             '饮酒量(ml/天)', '是否嚼槟榔', '嚼槟榔年限(年)', '嚼槟榔量（颗/天）']
    p = {key: (_binary(row[name]) if scale == 1 else _number(row[name]))
         for (key, _, scale), name in zip(__import__('core').HISTORY, names)}
    p.update(tct='' if row['TCT意见'] in MISSING else row['TCT意见'], methylation=_number(row['甲基化总数']),
             dna_count=_number(row['dna_abnormal_cells']), dna_total=_number(row['dna_analyzed_cells']),
             dna_threshold=_number(row['dna_threshold']), report_confirmed=True)
    return p


def model_checks(raw, output):
    engine = ResearchEngine(load_engine_config(bootstrap.APP_DIR / 'models.local.json'))
    torch = __import__('torch'); torch.set_num_threads(min(8, os.cpu_count() or 1))
    root = raw / 'completed_evidence'
    held = json.loads((root / 'runs/seed42/fold0/baseline/outer_predictions.json').read_text(encoding='utf-8'))['records']
    with (raw / 'sysu_samples.csv').open(encoding='utf-8-sig') as f: rows = {r['sample_id']: r for r in csv.DictReader(f)}
    with np.load(root / 'paired_views_final/baseline.npz', allow_pickle=False) as cache:
        ids, features, present = cache['sample_ids'].tolist(), cache['features'].copy(), cache['image_present'].copy()
    engine._load()
    comparisons, original_images = [], []
    for r in held:
        if r['status'] == 'abstain': continue
        sid = r['sample_id']; i = ids.index(sid); p = payload_from_csv(rows[sid])
        assert sid not in engine.fit['fit_ids']
        result = engine.predict_features(p, features[i, 0], bool(present[i]))
        error = float(np.abs(np.asarray(result['probabilities']) - r['probabilities']).max())
        assert error < 1e-4, (sid, error)
        comparisons.append({'sample_id': sid, 'max_probability_error': error})
        if present[i] and len(original_images) < 2:
            image_path = rows[sid]['image_path']
            v = {'payload': p, 'image': image_path, 'image_sha': digest(raw / image_path),
                 'report': '', 'revision': 1, 'quality': {}}
            actual = engine.predict(v, raw)
            original_error = float(np.abs(np.asarray(actual['probabilities']) - r['probabilities']).max())
            assert original_error < 1e-4, (sid, original_error)
            original_images.append({'sample_id': sid, 'max_probability_error': original_error,
                                    'elapsed_seconds': actual['elapsed_seconds']})
    assert len(original_images) == 2
    empty = engine.predict_features({}, np.zeros(1536), False)
    assert empty['status'] == 'abstain' and empty['probabilities'] is None
    unknown = engine.predict_features({'tct': '不在训练词表中的描述', 'report_confirmed': True}, np.zeros(1536), False)
    assert unknown['status'] == 'predicted' and unknown['present']['tct']
    expect_error(lambda: ResearchEngine({'classifier': 'not-found', 'encoder': 'not-found'})._load())
    # Exercise a real QThread on an explicitly synthetic input; no clinical inference claim.
    from PyQt6.QtCore import QEventLoop, QTimer
    from app import InferenceJob
    loop = QEventLoop(); delivered = []
    # Use a fresh engine to test loading and raw-image inference inside the worker.
    sample = rows[original_images[0]['sample_id']]
    row = {'payload': payload_from_csv(sample), 'image': sample['image_path'],
           'image_sha': digest(raw / sample['image_path']), 'report': '', 'revision': 7, 'quality': {}}
    job = InferenceJob(ResearchEngine(load_engine_config(bootstrap.APP_DIR / 'models.local.json')), row, raw)
    job.ready.connect(lambda value: delivered.append(value)); job.failed.connect(lambda msg: delivered.append({'status': 'failed', 'reason': msg}))
    job.finished.connect(loop.quit)
    timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit); timer.start(30000)
    job.start(); loop.exec()
    assert job.isFinished() and delivered[0]['status'] == 'predicted' and delivered[0]['input_revision'] == 7
    expected = next(r['probabilities'] for r in held if r['sample_id'] == sample['sample_id'])
    background_error = float(np.abs(np.asarray(delivered[0]['probabilities']) - expected).max())
    assert background_error < 1e-4
    return {'fold0_holdout_cached_features': comparisons, 'raw_image_live_inference': original_images,
            'encoder_sha256': ENCODER_SHA, 'classifier_sha256': CLASSIFIER_SHA,
            'unknown_tct_id': 1, 'all_missing_abstains': True, 'old_test22_evaluated': False,
            'new_training': False, 'probability_tolerance': 1e-4, 'background_qthread_delivery': True,
            'background_raw_image_max_error': background_error}


def main():
    import faulthandler
    faulthandler.dump_traceback_later(20, repeat=True)
    from PyQt6.QtWidgets import QApplication
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--raw-data', type=Path); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new acceptance output directory')
    args.output.mkdir(parents=True)
    result = {'date': '2026-10-10', 'engineering_workflows': workflows(args.output),
              'real_doctors': 0, 'real_usability_users': 0}
    if args.raw_data: result['model'] = model_checks(args.raw_data, args.output)
    (args.output / 'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__': main()
