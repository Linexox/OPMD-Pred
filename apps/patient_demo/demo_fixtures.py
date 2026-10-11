"""Explicitly fictional examples; never stand in for doctor/user validation."""
from datetime import date, timedelta


def seed(store):
    existing = {p['alias']: p['id'] for p in store.patients()}
    if '模拟：补照片' in existing:
        return existing['模拟：补照片']
    today = date.today()
    a = store.add_patient('模拟：补照片')
    v = store.save_visit(a, 'self_check', '右舌侧缘', today.isoformat(),
        '【虚构案例】准备首次就诊资料。没有检测报告，照片可后续补充。', {'smoking': 0})
    store.submit(v)
    store.review(v,'【模拟反馈，非真实医生意见】请补一张右舌侧缘照片。','模拟复核人',True,
        requested_items=['photo'],material_note='【模拟任务】补右舌侧缘照片，核对实际拍摄日期')
    b = store.add_patient('模拟：复诊')
    prior = store.save_visit(b, 'followup', '右舌侧缘', (today - timedelta(days=7)).isoformat(),
        '【虚构案例】用来演示就诊记录、人工反馈和待办；不是实际患者。', {})
    store.submit(prior)
    store.review(prior, '【模拟反馈，非真实医生意见】请按指定部位准备复诊资料。', '模拟复核人',
        due=(today + timedelta(days=7)).isoformat(), task_note='【模拟任务】右舌侧缘照片；按约定复诊')
    store.save_visit(b, 'followup', '右舌侧缘', today.isoformat(),
        '【虚构案例】本次随访草稿，查看此前任务后可继续完善。', {})
    return a
