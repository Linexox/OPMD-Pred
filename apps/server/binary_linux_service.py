"""Run the verified binary model with the existing mobile API and data store."""
import argparse
import os
import sys
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[2]
    selection = project/'runtime/binary-model-manifest.txt'
    default_manifest = (project/selection.read_text(encoding='utf-8').strip()).resolve() if selection.is_file() else project/'models/app/binary-mean/manifest.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=project/'runtime')
    parser.add_argument('--token-file', type=Path, default=project/'runtime/pairing-token.txt')
    parser.add_argument('--manifest', type=Path, default=default_manifest)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.manifest = args.manifest.resolve()
    args.token_file = args.token_file.resolve()
    os.umask(0o077)
    runtime = project/'runtime'
    code = project/'apps/server/binary_runtime'
    sys.path[1:1] = [str(runtime/name) for name in ('binary-research-deps-v3','binary-research-deps-v2','binary-research-deps')]+[str(code)]
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('XDG_CACHE_HOME', str(runtime/'cache'))
    os.environ.setdefault('HF_HOME', str(runtime/'cache/huggingface'))
    # Retain the frozen model source directory and its relative manifest paths.
    # Import the frozen network package before the older service bootstrap adds src.
    import importlib
    importlib.import_module('opmd_pred.model.two_branch_fusion')
    from server import create_app
    from binary_inference import BinaryResearchEngine
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(args.manifest.parent))
    os.chdir(code)
    from prototype_binary_predictor import PrivateBinaryPredictor
    from mean_ensemble_predictor import MeanEnsemblePredictor
    from sysu_tabular_common import offline_predictions
    import torch
    import uvicorn
    torch.set_num_threads(2)
    token = args.token_file.read_text(encoding='utf8').strip()
    if not token:
        raise ValueError('连接口令文件为空')
    with offline_predictions():
        import json
        model_manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        if model_manifest.get('format') == 'sysu-five-complete-model-mean-v1':
            predictor = MeanEnsemblePredictor(args.manifest, device='cpu')
            model_id = model_manifest['model_id']
        else:
            predictor = PrivateBinaryPredictor(args.manifest, device='cpu')
            model_id = 'sysu-resnet18-tabpfn-transformer-seed42-fold0'
        engine = BinaryResearchEngine(predictor, model_id)
        app = create_app(args.data_dir, token, {}, research_engine=engine)
        uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
