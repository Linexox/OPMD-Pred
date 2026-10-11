"""Local visit workflow. No names, clinical decisions, or patient diagnoses are generated."""
import hashlib
import json
import math
import shutil
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HISTORY = (
    ('smoking', '是否吸烟（0否/1是）', 1), ('smoking_years', '吸烟年限（年）', 50),
    ('smoking_amount', '吸烟量（支/天）', 100), ('alcohol', '是否饮酒（0否/1是）', 1),
    ('alcohol_years', '饮酒年限（年）', 50), ('alcohol_amount', '饮酒量（ml/天）', 1000),
    ('betel', '是否嚼槟榔（0否/1是）', 1), ('betel_years', '槟榔年限（年）', 20),
    ('betel_amount', '槟榔量（颗/天）', 50),
)
SITES = ('上唇', '下唇', '左颊黏膜', '右颊黏膜', '牙龈', '上腭', '舌背',
         '左舌侧缘', '右舌侧缘', '舌腹', '口底', '其他（备注说明）')
SCENARIOS = {'self_check': '首次自查', 'followup': '已就诊随访'}
STATUSES = {'draft': '草稿', 'submitted': '等待复核', 'needs_more': '待补充资料', 'reviewed': '医生已反馈'}


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def number(value, name, maximum=None):
    if value is None or str(value).strip() == '':
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}须为数值；不确定请留空') from exc
    if not math.isfinite(result) or result < 0 or (maximum is not None and result > maximum):
        raise ValueError(f'{name}超出支持范围')
    return result


def validate_payload(payload):
    result = dict(payload)
    if 'age' in result or 'sex' in result or 'gender' in result:
        raise ValueError('本版不收集年龄、性别模型输入')
    for key, title, scale in HISTORY:
        result[key] = number(result.get(key), title, 1 if scale == 1 else None)
        if scale == 1 and result[key] not in (None, 0, 1):
            raise ValueError(f'{title}仅支持0或1')
    count = number(result.get('dna_count'), 'DNA异常细胞数')
    total = number(result.get('dna_total'), 'DNA总细胞数')
    threshold = number(result.get('dna_threshold'), 'DNA阈值')
    if count is not None and not count.is_integer() or total is not None and not total.is_integer():
        raise ValueError('细胞数须为整数')
    if threshold not in (None, 2.3, 2.5):
        raise ValueError('当前模型仅支持DI≥2.3或DI≥2.5，其他定义请留存附件')
    if count is not None and total is not None and count > total:
        raise ValueError('DNA异常细胞数不能大于总数')
    result.update(dna_count=count, dna_total=total, dna_threshold=threshold)
    result['dna_ratio'] = count / total if count is not None and total and threshold is not None else None
    result['methylation'] = number(result.get('methylation'), '甲基化总数')
    result['tct'] = str(result.get('tct', '')).strip()
    result['report_confirmed'] = bool(result.get('report_confirmed', False))
    if result.get('report_date'):
        date.fromisoformat(result['report_date'])
    if result.get('photo_date'):
        date.fromisoformat(result['photo_date'])
    # Reports may be stored before verification; only confirmed measurements enter inference.
    return result


def quality(path):
    with Image.open(path) as source:
        source.load()
        rgb = source.convert('RGB')
        gray = np.asarray(ImageOps.contain(rgb.convert('L'), (512, 512)), dtype=np.float32)
        lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        sharpness = float(lap.var()) if lap.size else 0.
        dark, bright = float((gray < 20).mean()), float((gray > 245).mean())
        messages = []
        if min(rgb.size) < 256:
            messages.append('图片尺寸较小，建议重新拍摄')
        if sharpness < 40:
            messages.append('可能模糊或纹理不足，请检查对焦')
        if dark > .4:
            messages.append('暗部较多，请检查光照')
        if bright > .25:
            messages.append('亮部较多，请检查曝光或反光')
        return {'width': rgb.width, 'height': rgb.height, 'laplacian_variance': sharpness,
                'dark_fraction': dark, 'bright_fraction': bright, 'messages': messages,
                'policy': 'technical-demo-v1; not validated for lesion visibility',
                'thresholds': {'min_side': 256, 'laplacian_variance': 40, 'dark_fraction': .4, 'bright_fraction': .25}}


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'attachments').mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / 'records.sqlite3')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS patients(id TEXT PRIMARY KEY, alias TEXT NOT NULL, created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS visits(id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id),
          scenario TEXT NOT NULL, site TEXT NOT NULL, visit_date TEXT NOT NULL, symptoms TEXT NOT NULL,
          payload TEXT NOT NULL, image TEXT NOT NULL DEFAULT '', image_sha TEXT NOT NULL DEFAULT '',
          original_image TEXT NOT NULL DEFAULT '', original_sha TEXT NOT NULL DEFAULT '',
          report TEXT NOT NULL DEFAULT '', quality TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'draft',
          feedback TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL DEFAULT '', reviewed TEXT NOT NULL DEFAULT '',
          created TEXT NOT NULL, updated TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id),
          visit_id TEXT NOT NULL REFERENCES visits(id), due TEXT NOT NULL, site TEXT NOT NULL, note TEXT NOT NULL,
          source_revision INTEGER NOT NULL, source_date TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending', created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, visit_id TEXT NOT NULL REFERENCES visits(id),
          revision INTEGER NOT NULL, result TEXT NOT NULL, created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL,
          object_id TEXT NOT NULL, detail TEXT NOT NULL, created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY AUTOINCREMENT,
          visit_id TEXT NOT NULL REFERENCES visits(id), revision INTEGER NOT NULL, event TEXT NOT NULL,
          record TEXT NOT NULL, created TEXT NOT NULL);
        ''')
        additions = {'visits': {'task_id': "TEXT NOT NULL DEFAULT ''"}, 'tasks': {
            'kind': "TEXT NOT NULL DEFAULT 'appointment'", 'requested_items': "TEXT NOT NULL DEFAULT '[]'",
            'submission_id': "TEXT NOT NULL DEFAULT ''", 'submission_revision': 'INTEGER',
            'confirmed': "TEXT NOT NULL DEFAULT ''", 'confirmation_reviewer': "TEXT NOT NULL DEFAULT ''"}}
        pending = [(table, col, definition) for table, columns in additions.items() for col, definition in columns.items()
                   if col not in {r['name'] for r in self.db.execute(f'PRAGMA table_info({table})')}]
        if pending:
            backup_path = self.root / ('before-task-migration-' + uuid.uuid4().hex + '.sqlite3')
            with sqlite3.connect(backup_path) as backup:
                self.db.backup(backup)
            with self.db:
                for table, col, definition in pending:
                    self.db.execute(f'ALTER TABLE {table} ADD COLUMN {col} {definition}')
                self.audit('task_schema_migration', '', {'backup': backup_path.name, 'columns': [c for _, c, _ in pending]})

    def audit(self, event, object_id, detail=None):
        self.db.execute('INSERT INTO audit(event,object_id,detail,created) VALUES(?,?,?,?)',
                        (event, object_id, json.dumps(detail or {}, ensure_ascii=False), now()))

    def snapshot(self, visit_id, event):
        v = self.visit(visit_id)
        self.db.execute('INSERT INTO snapshots(visit_id,revision,event,record,created) VALUES(?,?,?,?,?)',
                        (visit_id, v['revision'], event, json.dumps(v, ensure_ascii=False), now()))

    def feedback_history(self, visit_id):
        records = [json.loads(r['record']) for r in self.db.execute(
            "SELECT record FROM snapshots WHERE visit_id=? AND event='review' ORDER BY id", (visit_id,))]
        return records

    def patients(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM patients ORDER BY created,id')]

    def add_patient(self, alias):
        if not alias.strip():
            raise ValueError('请填写匿名代号')
        pid = uuid.uuid4().hex
        with self.db:
            self.db.execute('INSERT INTO patients VALUES(?,?,?)', (pid, alias.strip(), now()))
            self.audit('create_patient', pid)
        return pid

    def visits(self, patient_id):
        return [self.visit(r['id']) for r in self.db.execute(
            'SELECT id FROM visits WHERE patient_id=? ORDER BY visit_date DESC,created DESC', (patient_id,))]

    def visit(self, visit_id):
        row = self.db.execute('SELECT * FROM visits WHERE id=?', (visit_id,)).fetchone()
        if row is None:
            raise ValueError('记录不存在')
        value = dict(row)
        value['payload'] = json.loads(value['payload'])
        value['quality'] = json.loads(value['quality'])
        return value

    def _copy(self, source, visit_id, image=False):
        p = Path(source)
        folder = self.root / 'attachments' / visit_id
        folder.mkdir(exist_ok=True)
        target = folder / (uuid.uuid4().hex + ('.png' if image else p.suffix.lower()))
        if image:
            with Image.open(p) as opened:
                opened.load()
                opened.convert('RGB').save(target)  # Fresh pixels remove identifying EXIF.
        else:
            if p.suffix.lower() not in {'.pdf', '.jpg', '.jpeg', '.png'}:
                raise ValueError('报告支持PDF、JPG或PNG')
            if p.suffix.lower() == '.pdf' and not p.read_bytes().startswith(b'%PDF-'):
                raise ValueError('PDF文件头无效，请重新选择原报告')
            if p.suffix.lower() != '.pdf':
                with Image.open(p) as report_image:
                    report_image.verify()
            shutil.copyfile(p, target)
        return str(target.relative_to(self.root))

    def save_visit(self, patient_id, scenario, site, visit_date, symptoms, payload,
                   image_source='', report_source='', visit_id=None, task_id=''):
        if scenario not in SCENARIOS or site not in SITES:
            raise ValueError('请确认场景和部位')
        date.fromisoformat(visit_date)
        payload = validate_payload(payload)
        old = self.visit(visit_id) if visit_id else None
        task_id = old['task_id'] if old else task_id
        task = self.task(task_id) if task_id else None
        if task:
            if task['kind'] != 'material' or task['patient_id'] != patient_id or task['site'] != site:
                raise ValueError('补充资料必须对应任务指定的档案和部位')
            if scenario != self.task_scenario(task):
                raise ValueError('补充资料须沿用来源记录的使用场景')
            if old and task['submission_id'] and task['submission_id'] != old['id']:
                raise ValueError('这是历史提交，请打开当前任务的资料记录')
        if old and old['patient_id'] != patient_id:
            raise ValueError('不能把一次就诊转移到另一患者')
        vid = visit_id or uuid.uuid4().hex
        image = old['image'] if old else ''
        original_image = old['original_image'] if old else ''
        original_sha = old['original_sha'] if old else ''
        report = old['report'] if old else ''
        q = old['quality'] if old else {}
        if image_source:
            q = quality(image_source)
            original_image = self._copy(image_source, vid)
            original_sha = digest(self.root / original_image)
            image = self._copy(image_source, vid, True)
        if report_source:
            sources = [report_source] if isinstance(report_source, str) else report_source
            existing = json.loads(report) if report else []
            hashes = {digest(self.root / p) for p in existing}
            for p in sources:
                sha = digest(p)
                if sha not in hashes:
                    existing.append(self._copy(p, vid)); hashes.add(sha)
            report = json.dumps(existing, ensure_ascii=False)
            payload['report_confirmed'] = False  # Recheck the new source after saving.
        paths = json.loads(report) if report else []
        payload['report_sha256s'] = [digest(self.root / p) for p in paths]
        confirmation_keys = ('dna_count', 'dna_total', 'dna_threshold', 'methylation', 'tct',
                             'report_date', 'report_site', 'source_note', 'report_sha256s')
        if old and (any(payload.get(k) != old['payload'].get(k) for k in confirmation_keys)
                    or site != old['site'] or visit_date != old['visit_date']):
            payload['report_confirmed'] = False
        if payload['report_confirmed'] and not paths and not str(payload.get('source_note', '')).strip():
            raise ValueError('确认检测字段需上传来源报告或填写来源说明')
        if payload['report_confirmed'] and payload.get('report_site') != site:
            raise ValueError('检测取材部位须与本记录部位对应；不确定时保留资料、取消确认')
        values = (scenario, site, visit_date, symptoms.strip(), json.dumps(payload, ensure_ascii=False),
                  image, digest(self.root / image) if image else '', original_image, original_sha,
                  report, json.dumps(q, ensure_ascii=False), now())
        with self.db:
            if old:
                self.snapshot(vid, 'before_edit')
                self.db.execute('''UPDATE visits SET scenario=?,site=?,visit_date=?,symptoms=?,payload=?,image=?,
                   image_sha=?,original_image=?,original_sha=?,report=?,quality=?,updated=?,revision=revision+1,status='draft',feedback='',reviewer='',reviewed=''
                   WHERE id=?''', values + (vid,))
            else:
                self.db.execute('''INSERT INTO visits(scenario,site,visit_date,symptoms,payload,image,image_sha,
                   original_image,original_sha,report,quality,updated,id,patient_id,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                                values + (vid, patient_id, now()))
            if task:
                self.db.execute('UPDATE visits SET task_id=? WHERE id=?', (task_id, vid))
                self.db.execute("UPDATE tasks SET submission_id=?,submission_revision=?,status='pending',confirmed='',confirmation_reviewer='' WHERE id=?",
                                (vid, old['revision'] + 1 if old else 1, task_id))
            self.snapshot(vid, 'save')
            self.audit('save_visit', vid, {'revision': old['revision'] + 1 if old else 1})
        return vid

    def submit(self, visit_id):
        with self.db:
            visit = self.visit(visit_id)
            if visit['status'] == 'submitted': return
            if visit['status'] == 'reviewed': raise ValueError('该版本已复核；如有补充请先保存修改')
            if visit['task_id']:
                task = self.task(visit['task_id'])
                if task['submission_id'] != visit_id or task['submission_revision'] != visit['revision']:
                    raise ValueError('任务对应的提交版本已改变')
                present = {'photo': bool(visit['image']), 'report': bool(json.loads(visit['report'] or '[]')),
                           'history': any(visit['payload'].get(k) is not None for k, _, _ in HISTORY),
                           'other': bool(visit['symptoms'].strip())}
                missing = [k for k in task['requested_items'] if not present[k]]
                if missing:
                    titles = {'photo': '照片', 'report': '检测报告', 'history': '病史', 'other': '文字说明'}
                    raise ValueError('尚未补充：' + '、'.join(titles[k] for k in missing))
                self.db.execute("UPDATE tasks SET status='submitted' WHERE id=?", (task['id'],))
            self.db.execute("UPDATE visits SET status='submitted',updated=? WHERE id=?", (now(), visit_id))
            self.audit('submit', visit_id)

    def review(self, visit_id, feedback, reviewer, needs_more=False, due='', task_note='',
               requested_items=None, material_note=''):
        visit = self.visit(visit_id)
        if visit['status'] not in ('submitted', 'needs_more'):
            raise ValueError('请先提交资料再复核')
        if not feedback.strip() or not reviewer.strip():
            raise ValueError('请填写复核人代号和确认意见')
        if due:
            date.fromisoformat(due)
            if not task_note.strip():
                raise ValueError('设置复诊日期时请填写任务说明')
        requested_items = sorted(set(requested_items or []))
        if not set(requested_items) <= {'photo', 'report', 'history', 'other'}:
            raise ValueError('不支持的待补资料类型')
        if requested_items and not material_note.strip():
            raise ValueError('请具体说明要补什么、拍哪里或核对哪份报告')
        needs_more = bool(needs_more or requested_items)
        linked = self.task(visit['task_id']) if visit['task_id'] else None
        if linked and (linked['submission_id'] != visit_id or linked['submission_revision'] != visit['revision']):
            raise ValueError('仅能确认当前任务的最新提交版本')
        with self.db:
            self.db.execute('UPDATE visits SET status=?,feedback=?,reviewer=?,reviewed=?,updated=? WHERE id=?',
                            ('needs_more' if needs_more else 'reviewed', feedback.strip(), reviewer.strip(), now(), now(), visit_id))
            if due:
                self._create_task(visit, 'appointment', [], due, task_note.strip())
            if requested_items:
                self._create_task(visit, 'material', requested_items, '', material_note.strip())
            if linked:
                self.db.execute('UPDATE tasks SET status=?,confirmed=?,confirmation_reviewer=? WHERE id=?',
                    ('pending' if needs_more else 'confirmed', '' if needs_more else now(), '' if needs_more else reviewer, linked['id']))
            self.audit('review', visit_id, {'reviewer': reviewer, 'needs_more': needs_more, 'due': due})
            self.snapshot(visit_id, 'review')

    def tasks(self, patient_id):
        return [self.task(r['id']) for r in self.db.execute('SELECT id FROM tasks WHERE patient_id=? ORDER BY created,id', (patient_id,))]

    def task(self, task_id):
        row = self.db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
        if row is None: raise ValueError('任务不存在')
        t = dict(row); t['requested_items'] = json.loads(t['requested_items']); return t

    def _create_task(self, visit, kind, items, due, note):
        identical = self.db.execute('SELECT id FROM tasks WHERE visit_id=? AND source_revision=? AND kind=? AND requested_items=? AND due=? AND note=?',
            (visit['id'], visit['revision'], kind, json.dumps(items), due, note)).fetchone()
        if identical: return identical['id']
        tid = uuid.uuid4().hex
        self.db.execute('INSERT INTO tasks(id,patient_id,visit_id,due,site,note,source_revision,source_date,created,kind,requested_items) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (tid, visit['patient_id'], visit['id'], due, visit['site'], note, visit['revision'], visit['visit_date'], now(), kind, json.dumps(items)))
        return tid

    def begin_material_task(self, task_id):
        task = self.task(task_id)
        if task['kind'] != 'material': raise ValueError('这是一项线下复诊任务')
        if task['status'] == 'confirmed': raise ValueError('资料已由医生确认')
        if task['status'] == 'submitted': raise ValueError('资料已经提交，正在等待医生复核')
        if task['submission_id']:
            return task['submission_id']
        return self.save_visit(task['patient_id'], self.task_scenario(task), task['site'], date.today().isoformat(), '', {}, task_id=task_id)

    def task_scenario(self, task):
        row=self.db.execute('SELECT record FROM snapshots WHERE visit_id=? AND revision=? ORDER BY id DESC LIMIT 1',
                            (task['visit_id'],task['source_revision'])).fetchone()
        return json.loads(row['record'])['scenario'] if row else self.visit(task['visit_id'])['scenario']

    def complete_task(self, task_id):
        task = self.task(task_id)
        if task['kind'] != 'appointment': raise ValueError('补资料任务需先提交，再由医生确认')
        with self.db:
            self.db.execute("UPDATE tasks SET status='patient_reported' WHERE id=?", (task_id,))
            self.audit('complete_task', task_id)

    def add_run(self, visit_id, revision, result):
        with self.db:
            self.db.execute('INSERT INTO runs VALUES(?,?,?,?,?)',
                            (uuid.uuid4().hex, visit_id, revision, json.dumps(result, ensure_ascii=False, allow_nan=False), now()))
            self.audit('research_inference', visit_id, {'revision': revision, 'status': result['status']})

    def latest_run(self, visit_id):
        row = self.db.execute('SELECT * FROM runs WHERE visit_id=? AND revision=? ORDER BY created DESC,id DESC LIMIT 1',
                              (visit_id, self.visit(visit_id)['revision'])).fetchone()
        return json.loads(row['result']) if row else None

    def close(self):
        self.db.close()
