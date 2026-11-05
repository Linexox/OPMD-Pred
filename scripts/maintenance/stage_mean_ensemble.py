"""Transfer only the selected server package and isolated checks to the personal PC."""
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/home/nexox/workspace/OPMD_Pred'
CHECKS = Path('outputs/checks/daily/2026-11-05/app-mean-deployment')


def main():
    bundle = ROOT / 'models/app/binary-mean'
    files = [(p, p.relative_to(ROOT).as_posix()) for p in bundle.rglob('*') if p.is_file()]
    for name in ['apps/server/binary_runtime/mean_ensemble_predictor.py',
                 'scripts/maintenance/check_mean_ensemble_deployment.py']:
        files.append((ROOT / name, name))
    files.append((ROOT / 'apps/server/binary_inference.py', (CHECKS / 'staged/binary_inference.py').as_posix()))
    for p in (ROOT / CHECKS).iterdir():
        if p.is_file() and (p.name == 'fixed-inputs.private.json' or p.name.startswith('fixed-')):
            files.append((p, p.relative_to(ROOT).as_posix()))
    command = f'cd {REMOTE} && test "$(pwd)" = {REMOTE} && umask 077 && tar -xf -'
    process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'lnx', command],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def private(info):
        info.mode = 0o700 if info.isdir() else 0o600
        return info
    with tarfile.open(fileobj=process.stdin, mode='w|') as archive:
        for source, target in files:
            archive.add(source, arcname=target, recursive=False, filter=private)
    process.stdin.close()
    error = process.stderr.read().decode('utf-8', errors='replace')
    process.stdout.close()
    if process.wait() != 0:
        raise RuntimeError(error)
    print({'transferred_files': len(files), 'bytes': sum(p.stat().st_size for p, _ in files),
           'target': REMOTE, 'running_service_not_changed': True})


if __name__ == '__main__':
    main()
