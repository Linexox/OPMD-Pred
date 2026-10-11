"""Task transitions, legacy migration, three-step UI, scaling and save/discard guards."""
import argparse
import json
import os
import shutil
import hashlib
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import bootstrap
from core import Store, digest
from PIL import Image
from app import Window, configure_application
from inference import ResearchEngine
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton
from PyQt6.QtCore import QTimer


def fails(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Expected validation failure')


def dialog_answer(text):
    def choose():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget,QMessageBox):
                next(b for b in widget.buttons() if b.text()==text).click(); return
        raise AssertionError('Expected unsaved-input dialog')
    QTimer.singleShot(0,choose)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--legacy',type=Path); args=parser.parse_args()
    assert not args.output.exists(); args.output.mkdir(parents=True)
    a=QApplication([]); a.setQuitOnLastWindowClosed(False); configure_application(a)
    if args.legacy:
        shutil.copytree(args.legacy,args.output/'legacy-copy')
        legacy=Store(args.output/'legacy-copy')
        assert len(legacy.patients())>=1 and all(t['kind']=='appointment' for p in legacy.patients() for t in legacy.tasks(p['id']))
        assert list((args.output/'legacy-copy').glob('before-task-migration-*.sqlite3')); legacy.close()
    store=Store(args.output/'records'); pid=store.add_patient('模拟：小陈')
    source=store.save_visit(pid,'self_check','右舌侧缘','2026-10-10','【模拟】准备首次就诊',{})
    store.submit(source); store.review(source,'【模拟反馈】请补右舌侧缘照片','模拟医生',True,
        requested_items=['photo'],material_note='【模拟任务】补右舌侧缘照片')
    task=store.tasks(pid)[0]; assert task['kind']=='material' and task['status']=='pending'
    supplement=store.begin_material_task(task['id']); assert supplement!=source
    assert store.begin_material_task(task['id'])==supplement
    assert store.visit(source)['revision']==1
    fails(lambda: store.submit(supplement)); fails(lambda: store.complete_task(task['id']))
    wrong=store.add_patient('模拟：另一档案')
    fails(lambda: store.save_visit(wrong,'followup','右舌侧缘','2026-10-10','',{},visit_id=supplement))
    fails(lambda: store.save_visit(pid,'followup','左颊黏膜','2026-10-10','',{},visit_id=supplement))
    photo=args.output/'technical-pattern.png'; Image.new('RGB',(600,450),(65,140,160)).save(photo)
    assert store.visit(supplement)['scenario']=='self_check'
    store.save_visit(pid,'self_check','右舌侧缘','2026-10-10','【模拟】补充照片',{},str(photo),visit_id=supplement)
    store.submit(supplement); store.submit(supplement)
    assert store.task(task['id'])['status']=='submitted'
    fails(lambda: store.begin_material_task(task['id']))
    store.review(supplement,'【模拟确认】该任务资料已核对','模拟医生')
    assert store.task(task['id'])['status']=='confirmed'
    store.save_visit(pid,'self_check','右舌侧缘','2026-10-10','【模拟】修改后须重新核对',{},visit_id=supplement)
    assert store.task(task['id'])['status']=='pending' and not store.task(task['id'])['confirmed']
    store.submit(supplement); store.review(supplement,'【模拟确认】已重新核对','模拟医生')
    store.review(source,'【模拟反馈】再补报告','模拟医生',True,requested_items=['report'],material_note='【模拟任务】对应右舌侧缘的报告')
    report_task=next(t for t in store.tasks(pid) if t['requested_items']==['report'])
    report_visit=store.begin_material_task(report_task['id']); fails(lambda: store.submit(report_visit))
    store.save_visit(pid,'self_check','右舌侧缘','2026-10-10','【模拟】报告附件',{},report_source=[str(photo)],visit_id=report_visit)
    store.submit(report_visit); store.review(report_visit,'【模拟确认】报告任务资料已核对','模拟医生')
    store.review(source,'【模拟反馈】安排复诊','模拟医生',due='2026-10-20',task_note='【模拟任务】按指定部位复诊')
    appointment=next(t for t in store.tasks(pid) if t['kind']=='appointment')
    store.complete_task(appointment['id']); assert store.task(appointment['id'])['status']=='patient_reported'
    assert len(store.visits(pid))==3 and store.visit(source)['revision']==1
    store.close(); store=Store(args.output/'records')
    assert store.task(task['id'])['status']=='confirmed' and store.task(report_task['id'])['status']=='confirmed'
    w=Window(store,ResearchEngine({'classifier':'','encoder':''})); w.resize(1024,768); w.show(); a.processEvents()
    w.patient.setCurrentIndex(w.patient.findData(pid))
    w.visit_id=source; w.refresh_visits(source); w.load_visit(); w.tabs.setCurrentIndex(1)
    w.symptoms.setPlainText('【模拟】分步输入不会丢失'); assert w.dirty
    w.move_step(1); w.move_step(-1); assert '不会丢失' in w.symptoms.toPlainText()
    w.photo_date.setText('not-a-date'); w.move_step(1)
    assert not w.error_labels['photo_date'].isHidden() and w.entry_stack.currentIndex()==1
    w.photo_date.clear()
    assert w.error_labels['photo_date'].isHidden()
    dialog_answer('继续编辑'); assert not w.guard_changes()
    dialog_answer('保存草稿'); assert w.guard_changes() and not w.dirty
    w.symptoms.setPlainText('【模拟】放弃这段修改'); dialog_answer('放弃修改'); assert w.guard_changes()
    w.load_visit(); assert '放弃这段修改' not in w.symptoms.toPlainText()
    store.submit(source); store.review(source,'【模拟反馈】补指定部位照片','模拟医生',True,
        requested_items=['photo'],material_note='【模拟任务】右舌侧缘照片，拍摄日期请填写真实日期')
    w.load_visit()
    w.entry_stack.setCurrentIndex(0); a.processEvents(); w.grab().save(str(args.output/'after-step1.png'))
    w.entry_stack.setCurrentIndex(1); a.processEvents(); w.grab().save(str(args.output/'after-step2.png'))
    w.entry_stack.setCurrentIndex(2); a.processEvents(); w.grab().save(str(args.output/'after-step3.png'))
    w.tabs.setCurrentIndex(0); a.processEvents(); w.grab().save(str(args.output/'after-next.png'))
    w.role.setCurrentIndex(1)
    w.doctor_tabs.setCurrentIndex(1)
    w.show_research({'status':'predicted','simulation':True,'probabilities':[.2,.3,.5],'present':{'image':True,'history':True,'tct':False,'dna':False,'methylation':False}})
    a.processEvents(); w.grab().save(str(args.output/'after-doctor.png'))
    assert '正常/轻度' in w.scores_card.text() and 'probabilities' not in w.research.toPlainText()
    assert w.raw_log.isHidden()
    w.tabs.setCurrentIndex(1); w.entry_stack.setCurrentIndex(0); a.processEvents()
    assert w.next_button.height()>=44 and w.width()==1024
    assert not w.next_button.visibleRegion().isEmpty()
    receipt={'task_photo_and_report_closed':True,'duplicate_submission_safe':True,'task_revision_reopens':True,
        'appointments_patient_reported_only':True,'wrong_patient_site_rejected':True,'restart_preserved':True,
        'legacy_migration':bool(args.legacy),'three_steps_preserve_input':True,'inline_errors':True,
        'save_discard_cancel':True,'doctor_cards_no_primary_json':True,'window_logical_size':[w.width(),w.height()],
        'scale_factor':os.environ.get('QT_SCALE_FACTOR','1'),'doctor_participants':0,'user_participants':0}
    receipt['source_sha256']={name:hashlib.sha256((bootstrap.APP_DIR/name).read_bytes()).hexdigest()
                              for name in ('app.py','core.py','verify_day2.py')}
    w.close(); (args.output/'acceptance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
