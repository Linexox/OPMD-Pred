"""Switch the verified personal-PC service, with an explicit old-model fallback."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import urllib.request


ROOT = Path('/home/nexox/workspace/OPMD_Pred')
OUT = ROOT / 'outputs/checks/daily/2026-11-05/app-mean-deployment'
MODEL = 'models/app/binary-mean/manifest.json'
OLD_MODEL = 'models/app/binary/manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_selection(relative):
    target = ROOT / 'runtime/binary-model-manifest.txt'
    incoming = target.with_name(target.name + '.incoming')
    incoming.write_text(relative + '\n', encoding='utf-8'); incoming.chmod(0o600); incoming.replace(target)


def health(url, token):
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def wait_ready(expected, token):
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        try:
            result = health('http://127.0.0.1:8765/api/health', token)
            if result.get('status') == 'ready' and result.get('model_id') == expected:
                return result
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError('Service did not report the selected model')


def main():
    assert Path.cwd().resolve() == ROOT and ROOT.is_relative_to('/home/nexox/workspace')
    checked = json.loads((OUT / 'real-cpu-and-http-check.json').read_text())
    model = json.loads((ROOT / MODEL).read_text())
    assert checked['passed'] and checked['manifest_sha256'] == sha(ROOT / MODEL)
    token = (ROOT / 'runtime/pairing-token.txt').read_text().strip()
    assert token
    with sqlite3.connect('file:' + str(ROOT / 'runtime/records.sqlite3') + '?mode=ro', uri=True) as db:
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    sources = ['binary_linux_service.py', 'binary_inference.py', 'server.py']
    assert all((OUT / 'staged' / name).is_file() for name in sources)
    started = time.time()
    subprocess.run(['systemctl', '--user', 'stop', 'opmd-pred.service'], check=True)
    try:
        for name in sources:
            source = OUT / 'staged' / name
            target = ROOT / 'apps/server' / name
            incoming = target.with_name(target.name + '.incoming')
            shutil.copyfile(source, incoming); incoming.chmod(0o600)
            assert sha(source) == sha(incoming)
            incoming.replace(target)
        write_selection(MODEL)
        subprocess.run(['systemctl', '--user', 'start', 'opmd-pred.service'], check=True)
        local = wait_ready(model['model_id'], token)
        public = health('http://www.ming56.cn:9020/api/health', token)
        assert public['model_id'] == model['model_id'] and public['model_task'] == 'binary'
        result = {'passed': True, 'model_id': model['model_id'], 'local_ready': local['status'],
                  'public_ready': public['status'], 'manifest_sha256': sha(ROOT / MODEL),
                  'service_active': subprocess.check_output(['systemctl', '--user', 'is-active', 'opmd-pred.service'], text=True).strip(),
                  'activated_seconds': time.time() - started, 'database_quick_check': 'ok',
                  'database_migration': False, 'pairing_token_changed': False, 'public_address_changed': False,
                  'old_model_retained_at_original_location': True}
        (OUT / 'activation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(result, flush=True)
    except Exception:
        write_selection(OLD_MODEL)
        subprocess.run(['systemctl', '--user', 'restart', 'opmd-pred.service'], check=True)
        wait_ready('sysu-resnet18-tabpfn-transformer-seed42-fold0', token)
        (OUT / 'activation-failed.json').write_text(json.dumps({'passed': False, 'old_model_restored': True,
                                                             'started': started}) + '\n')
        raise


if __name__ == '__main__':
    os.umask(0o077)
    main()
