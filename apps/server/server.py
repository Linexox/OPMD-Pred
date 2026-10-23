"""Mobile APP service with visit storage and frozen-model inference."""
import argparse
import base64
import hashlib
import json
import re
import secrets
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / '.deps' / 'api'), str(HERE)]
import bootstrap
from core import HISTORY, SCENARIOS, SITES, Store, digest, now
from inference import ResearchEngine, load_engine_config
from explanation import compare_saved_runs, run_summary
sys.path.insert(0, str(HERE / '.deps' / 'api'))
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn


class ApiError(Exception):
    def __init__(self, status, code, message, field=''):
        self.status, self.code, self.message, self.field = status, code, message, field


def create_app(root, token, model_config):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'uploads').mkdir(exist_ok=True)
    lock = threading.RLock()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='research')
    engine = ResearchEngine(model_config)

    @contextmanager
    def store():
        with lock:
            s = Store(root)
            try:
                yield s
            finally:
                s.close()

    with store() as s:
        s.db.executescript('''
          CREATE TABLE IF NOT EXISTS mobile_operations(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, response TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS mobile_assets(id TEXT PRIMARY KEY, path TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL, sha TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS mobile_jobs(id TEXT PRIMARY KEY, visit_id TEXT NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL, result TEXT, created TEXT NOT NULL);
        ''')
        # Interrupted jobs remain visible; never pretend they finished.
        s.db.execute("UPDATE mobile_jobs SET status='failed',result=? WHERE status IN ('queued','running')",
                     (json.dumps({'status': 'failed', 'reason': '服务已重启，请重试'}, ensure_ascii=False),))
        s.db.commit()

    app = FastAPI(title='口腔随访服务', version='1.0.0')
    app.add_middleware(CORSMiddleware, allow_origins=['http://localhost', 'https://localhost', 'http://localhost:5173',
                                                   'http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])

    @app.exception_handler(ApiError)
    async def api_error(_, exc):
        return JSONResponse(status_code=exc.status, content={'error': {'code': exc.code, 'message': exc.message, 'field': exc.field}})

    @app.exception_handler(ValueError)
    async def invalid(_, exc):
        return JSONResponse(status_code=422, content={'error': {'code': 'INVALID_INPUT', 'message': str(exc), 'field': ''}})

    @app.middleware('http')
    async def authorize(request, call_next):
        if request.method != 'OPTIONS' and request.url.path not in ('/download', '/download/app.apk') and request.headers.get('Authorization') != 'Bearer ' + token:
            return JSONResponse(status_code=401, content={'error': {'code': 'PAIRING_REQUIRED', 'message': '请检查连接口令', 'field': ''}})
        return await call_next(request)

    def mutate(s, request, body, action):
        key = request.headers.get('X-Request-Id', '')
        if not key or len(key) > 100:
            raise ApiError(400, 'REQUEST_ID_REQUIRED', '请重试这次操作')
        fp = hashlib.sha256((request.method + request.url.path + json.dumps(body, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
        old = s.db.execute('SELECT * FROM mobile_operations WHERE id=?', (key,)).fetchone()
        if old:
            if old['fingerprint'] != fp:
                raise ApiError(409, 'REQUEST_CONFLICT', '这次操作的内容已改变，请重新提交')
            return json.loads(old['response'])
        result = action()
        with s.db:
            s.db.execute('INSERT INTO mobile_operations VALUES(?,?,?)', (key, fp, json.dumps(result, ensure_ascii=False, allow_nan=False)))
        return result

    def version(s, vid, expected):
        v = s.visit(vid)
        if expected != v['revision']:
            raise ApiError(409, 'STALE_REVISION', '资料已经更新，请重新打开记录', 'revision')
        return v

    def asset(s, relative, kind, name=''):
        row = s.db.execute('SELECT id FROM mobile_assets WHERE path=?', (relative,)).fetchone()
        aid = row['id'] if row else uuid.uuid4().hex
        if not row:
            sha = digest(root / relative)
            source_name = s.db.execute("SELECT name FROM mobile_assets WHERE sha=? AND kind=? ORDER BY CASE WHEN path LIKE 'uploads%' THEN 0 ELSE 1 END,rowid DESC LIMIT 1", (sha, kind)).fetchone()
            name = name or (source_name['name'] if source_name else '')
            with s.db:
                s.db.execute('INSERT INTO mobile_assets VALUES(?,?,?,?,?)',
                             (aid, relative, name or ('病损照片.jpg' if kind == 'photo' else Path(relative).name), kind, digest(root / relative)))
        else:
            name = s.db.execute('SELECT name FROM mobile_assets WHERE id=?', (aid,)).fetchone()['name']
            if kind == 'report' and re.fullmatch(r'[0-9a-f]{32}\.(?:pdf|png|jpe?g)', name, re.IGNORECASE):
                original = s.db.execute("SELECT name FROM mobile_assets WHERE sha=? AND kind='report' AND path LIKE 'uploads%' ORDER BY rowid DESC LIMIT 1", (digest(root / relative),)).fetchone()
                if original and original['name'] != name:
                    name = original['name']
                    with s.db:
                        s.db.execute('UPDATE mobile_assets SET name=? WHERE id=?', (name, aid))
        return {'id': aid, 'name': name or ('病损照片' if kind == 'photo' else Path(relative).name), 'kind': kind}

    def public_visit(s, vid):
        v = s.visit(vid)
        v['photo'] = asset(s, v['image'], 'photo') if v['image'] else None
        v['reports'] = [asset(s, p, 'report') for p in json.loads(v['report'] or '[]')]
        v['feedback_history'] = [{k: row[k] for k in ('revision', 'feedback', 'reviewer', 'reviewed')}
                                 for row in s.feedback_history(vid)]
        for k in ('image', 'original_image', 'report'):
            v.pop(k, None)
        return v

    @app.get('/api/health')
    def health():
        return {'status': 'ready', 'name': '口腔随访助手', 'version': 1}

    @app.get('/download', response_class=HTMLResponse)
    def download_page():
        return HTMLResponse('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>安装口腔随访</title><style>body{font:17px system-ui;background:#f5f4fa;color:#292833;padding:48px 24px;max-width:480px;margin:auto}h1{font-size:28px}a{display:block;background:#292833;color:white;padding:18px;text-align:center;border-radius:18px;text-decoration:none;margin:32px 0}p{line-height:1.8;color:#777080}</style><h1>口腔随访</h1><p>安装后打开即可连接服务。</p><a href="/download/app.apk">下载 Android 安装包</a></html>''', headers={'Cache-Control': 'no-store'})

    @app.get('/download/app.apk')
    def download_apk():
        apk = root / 'distribution' / 'app-debug.apk'
        if not apk.is_file():
            raise ApiError(404, 'APK_NOT_READY', '安装包尚未准备好')
        return FileResponse(apk, media_type='application/vnd.android.package-archive',
                            filename='口腔随访.apk', headers={'Cache-Control': 'no-store'})

    @app.get('/api/options')
    def options():
        return {'sites': SITES, 'scenarios': SCENARIOS, 'history': [{'key': k, 'title': title, 'scale': scale} for k, title, scale in HISTORY]}

    @app.get('/api/patients')
    def patients():
        with store() as s:
            return [p for p in s.patients() if not p['alias'].startswith('模拟：')]

    @app.post('/api/patients')
    async def add_patient(request: Request):
        body = await request.json()
        with store() as s:
            return mutate(s, request, body, lambda: {'id': s.add_patient(str(body.get('alias', '')))})

    # Compatibility for older APKs: no generated visits or measurements.
    @app.post('/api/demo-fixtures')
    async def fixtures(request: Request):
        raise ApiError(410, 'ENTRY_REMOVED', '请新建档案并填写资料')

    @app.get('/api/demo-examples')
    def demo_examples():
        return []

    @app.get('/api/patients/{pid}/visits')
    def visits(pid: str):
        with store() as s:
            return [public_visit(s, v['id']) for v in s.visits(pid)]

    @app.get('/api/patients/{pid}/tasks')
    def tasks(pid: str):
        with store() as s:
            return s.tasks(pid)

    @app.get('/api/visits/{vid}')
    def visit(vid: str):
        with store() as s:
            return public_visit(s, vid)

    @app.post('/api/assets')
    async def upload(request: Request):
        if int(request.headers.get('content-length', '0')) > 29_000_000:
            raise ApiError(413, 'FILE_TOO_LARGE', '文件超过20MB，请选择较小文件')
        body = await request.json()
        with store() as s:
            def action():
                name = str(body.get('name', ''))
                kind = body.get('kind')
                ext = Path(name).suffix.lower()
                if kind not in ('photo', 'report') or ext not in ('.jpg', '.jpeg', '.png', '.pdf') or kind == 'photo' and ext == '.pdf':
                    raise ValueError('照片支持JPG/PNG；报告支持PDF、JPG、PNG')
                try:
                    data = base64.b64decode(body.get('data', ''), validate=True)
                except Exception as exc:
                    raise ValueError('文件读取失败，请重新选择') from exc
                if not data or len(data) > 20_000_000:
                    raise ApiError(413, 'FILE_TOO_LARGE', '文件为空或超过20MB')
                aid = uuid.uuid4().hex
                path = root / 'uploads' / (aid + ext)
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                except OSError as exc:
                    raise ApiError(503, 'UPLOAD_STORAGE_UNAVAILABLE', '电脑暂时无法保存文件，资料已保留，请检查电脑存储后重试') from exc
                try:
                    if ext == '.pdf':
                        from pypdf import PdfReader
                        if not data.startswith(b'%PDF-') or not PdfReader(path).pages:
                            raise ValueError('报告PDF不可读取')
                    else:
                        from PIL import Image
                        with Image.open(path) as image:
                            image.verify()
                    with s.db:
                        s.db.execute('INSERT INTO mobile_assets VALUES(?,?,?,?,?)',
                                     (aid, str(path.relative_to(root)), Path(name).name, kind, digest(path)))
                except Exception as exc:
                    path.unlink(missing_ok=True)
                    raise ValueError('文件损坏，请重新选择原文件') from exc
                return {'id': aid, 'name': Path(name).name, 'kind': kind}
            return mutate(s, request, body, action)

    @app.get('/api/assets/{aid}')
    def download(aid: str):
        with store() as s:
            row = s.db.execute('SELECT * FROM mobile_assets WHERE id=?', (aid,)).fetchone()
            if not row:
                raise ApiError(404, 'ASSET_NOT_FOUND', '附件不存在')
            path = root / row['path']
            if digest(path) != row['sha']:
                raise ApiError(409, 'ASSET_CHANGED', '附件已改变，请重新保存资料')
            return FileResponse(path, filename=row['name'])

    def save(s, body, vid=None):
        if vid:
            version(s, vid, body.get('revision'))
        def source(aid, kind):
            if not aid:
                return ''
            row = s.db.execute('SELECT * FROM mobile_assets WHERE id=?', (aid,)).fetchone()
            if not row or row['kind'] != kind:
                raise ValueError('附件类型或编号无效，请重新选择')
            if digest(root / row['path']) != row['sha']:
                raise ValueError('附件已改变，请重新上传')
            return str(root / row['path'])
        saved = s.save_visit(body['patient_id'], body['scenario'], body['site'], body['visit_date'], body.get('symptoms', ''),
                             body.get('payload', {}), image_source=source(body.get('photo_id'), 'photo'),
                             report_source=[source(aid, 'report') for aid in body.get('report_ids', [])],
                             visit_id=vid, task_id=body.get('task_id', ''))
        return public_visit(s, saved)

    @app.post('/api/visits')
    async def new_visit(request: Request):
        body = await request.json()
        with store() as s:
            return mutate(s, request, body, lambda: save(s, body))

    @app.patch('/api/visits/{vid}')
    async def edit_visit(vid: str, request: Request):
        body = await request.json()
        with store() as s:
            return mutate(s, request, body, lambda: save(s, body, vid))

    @app.post('/api/visits/{vid}/submissions')
    async def submit(vid: str, request: Request):
        body = await request.json()
        with store() as s:
            def action():
                version(s, vid, body.get('revision'))
                s.submit(vid)
                return public_visit(s, vid)
            return mutate(s, request, body, action)

    @app.post('/api/visits/{vid}/reviews')
    async def review(vid: str, request: Request):
        body = await request.json()
        with store() as s:
            def action():
                version(s, vid, body.get('revision'))
                s.review(vid, body.get('feedback', ''), body.get('reviewer', ''), requested_items=body.get('requested_items', []),
                         material_note=body.get('material_note', ''), due=body.get('due', ''), task_note=body.get('task_note', ''))
                return public_visit(s, vid)
            return mutate(s, request, body, action)

    @app.post('/api/tasks/{tid}/submissions')
    async def begin(tid: str, request: Request):
        body = await request.json()
        with store() as s:
            return mutate(s, request, body, lambda: public_visit(s, s.begin_material_task(tid)))

    @app.post('/api/tasks/{tid}/completions')
    async def complete(tid: str, request: Request):
        body = await request.json()
        with store() as s:
            def action():
                s.complete_task(tid)
                return s.task(tid)
            return mutate(s, request, body, action)

    def run_job(jid, snapshot):
        with store() as s:
            s.db.execute("UPDATE mobile_jobs SET status='running' WHERE id=?", (jid,))
            s.db.commit()
        try:
            result = engine.predict(snapshot, root)
        except Exception as exc:
            result = {'status': 'failed', 'reason': str(exc), 'input_revision': snapshot['revision']}
        result['analysis_id'] = jid
        with store() as s:
            s.add_run(snapshot['id'], snapshot['revision'], result)
            s.db.execute('UPDATE mobile_jobs SET status=?,result=? WHERE id=?',
                         (result['status'], json.dumps(result, ensure_ascii=False, allow_nan=False), jid))
            s.db.commit()

    @app.post('/api/visits/{vid}/runs')
    async def start_run(vid: str, request: Request):
        body = await request.json()
        with store() as s:
            def action():
                snapshot = version(s, vid, body.get('revision'))
                pending = s.db.execute("SELECT id,status FROM mobile_jobs WHERE visit_id=? AND revision=? AND status IN ('queued','running') ORDER BY created DESC LIMIT 1",
                                       (vid, snapshot['revision'])).fetchone()
                if pending:
                    return {'id': pending['id'], 'status': pending['status'], 'input_revision': snapshot['revision']}
                jid = uuid.uuid4().hex
                with s.db:
                    s.db.execute('INSERT INTO mobile_jobs VALUES(?,?,?,?,?,?)', (jid, vid, snapshot['revision'], 'queued', None, now()))
                pool.submit(run_job, jid, snapshot)
                return {'id': jid, 'status': 'queued', 'input_revision': snapshot['revision']}
            return mutate(s, request, body, action)

    @app.get('/api/runs/{jid}')
    def job(jid: str):
        with store() as s:
            row = s.db.execute('SELECT * FROM mobile_jobs WHERE id=?', (jid,)).fetchone()
            if not row:
                raise ApiError(404, 'RUN_NOT_FOUND', '运行记录不存在')
            result = dict(row)
            result['result'] = json.loads(result['result']) if result['result'] else None
            result['stale'] = s.visit(row['visit_id'])['revision'] != row['revision']
            return result

    def saved_runs(s, vid):
        return [dict(id=r['id'], visit_id=r['visit_id'], revision=r['revision'], created=r['created'], result=json.loads(r['result']))
                for r in s.db.execute('SELECT * FROM runs WHERE visit_id=? ORDER BY created DESC,id DESC', (vid,))]

    @app.get('/api/visits/{vid}/runs')
    def history(vid: str):
        with store() as s:
            return saved_runs(s, vid)

    @app.get('/api/visits/{vid}/run-comparison')
    def comparison(vid: str, current_run_id: str, previous_run_id: str = ''):
        with store() as s:
            s.visit(vid)
            local_runs = saved_runs(s, vid)
            current = next((r for r in local_runs if r['id'] == current_run_id), None)
            if current is None:
                raise ApiError(404, 'RUN_NOT_FOUND', '分析记录不存在，请重新打开')
            context = current['result'].get('input_context', {})
            source_visit_id, source_revision = '', None
            if context.get('task_id'):
                task = s.task(context['task_id'])
                if task['kind'] == 'material' and task['submission_id'] == vid and task['patient_id'] == context.get('patient_id') and task['site'] == context.get('site'):
                    source_visit_id, source_revision = task['visit_id'], task['source_revision']
            possible = [r for r in local_runs if r['id'] != current_run_id]
            if source_visit_id and source_visit_id != vid:
                possible += [r for r in saved_runs(s, source_visit_id) if r['revision'] == source_revision]
            possible.sort(key=lambda r: (r['created'], r['id']), reverse=True)
            comparisons = {r['id']: compare_saved_runs(current, r, source_visit_id, source_revision) for r in possible}
            candidates = [run_summary(r, comparisons[r['id']]['previous']['relation']) for r in possible if comparisons[r['id']]['status'] == 'compared']
            if previous_run_id:
                if previous_run_id not in comparisons:
                    raise ApiError(404, 'COMPARISON_NOT_FOUND', '该记录不属于本次可对照的资料')
                result = comparisons[previous_run_id]
            elif candidates:
                result = comparisons[candidates[0]['id']]
            else:
                reason = next((comparisons[r['id']]['reason'] for r in possible if r['result'].get('status') == 'predicted'),
                              '尚无可对照的较早分析。资料补充后重新分析，即可在这里查看变化。')
                result = {'status': 'unavailable', 'reason': reason}
            return dict(result, candidates=candidates)

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', default=str(HERE.parents[2] / 'OPMDMobileDemo'))
    parser.add_argument('--model-config', default=str(HERE / 'models.local.json'))
    args = parser.parse_args()
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    token_path = root / 'pairing-token.txt'
    if not token_path.exists():
        token_path.write_text(secrets.token_urlsafe(24), encoding='utf-8')
    token = token_path.read_text(encoding='utf-8').strip()
    print('连接口令：' + token, flush=True)
    print(f'手机连接：http://电脑的Wi-Fi地址:{args.port}', flush=True)
    uvicorn.run(create_app(root, token, load_engine_config(args.model_config)), host=args.host, port=args.port)
