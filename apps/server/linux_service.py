"""Run the existing APP backend on Linux without printing its pairing token."""
import argparse
import os
import secrets
from pathlib import Path

from server import create_app, load_engine_config
import uvicorn


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=project / 'runtime')
    parser.add_argument('--model-config', type=Path,
                        default=project / 'apps/server/models.local.json')
    args = parser.parse_args()
    os.umask(0o077)
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    token_path = root / 'pairing-token.txt'
    if not token_path.exists():
        with token_path.open('x', encoding='utf-8') as stream:
            stream.write(secrets.token_urlsafe(24))
    token_path.chmod(0o600)
    token = token_path.read_text(encoding='utf-8').strip()
    if not token:
        raise ValueError('连接口令文件为空')
    app = create_app(root, token, load_engine_config(args.model_config))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
