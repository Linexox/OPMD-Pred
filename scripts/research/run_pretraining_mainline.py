"""Portable master entry; reuse the frozen training algorithms and private assets."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return Path(path).read_text(encoding='utf-8')


def load(path):
    return json.loads(read(path))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve_input(root, path):
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('路径须位于当前仓库内')
    return resolved


def training_commands(action, root, config, output, device):
    if action not in {'fivefold', 'sixfold', 'multiseed'}:
        raise ValueError('未知训练任务')
    target = resolve_input(root, output)
    if not target.is_relative_to((root / 'outputs/experiments').resolve()):
        raise ValueError('新实验结果须写入outputs/experiments下的新目录')
    if target.exists():
        raise FileExistsError('结果目录已存在，请指定新的--output；不会覆盖历史实验')
    path = lambda key: str(resolve_input(root, config[key]))
    common = ['--data', path('data'), '--config', path('training_config'),
              '--methyl-checkpoint', path('methyl_checkpoint'), '--output', str(target), '--device', device]
    if action == 'fivefold':
        # The unchanged original runner retains both pretrained cells and its ImageNet control.
        return [[sys.executable, str(root / 'scripts/research/run_sysu_cell_encoder_transfer.py'), *common,
                 '--cell-checkpoint', path('cell_checkpoint'), '--cache', path('cache')]]
    if action == 'sixfold':
        return [[sys.executable, str(root / 'scripts/research/run_cell_sixfold_ensemble.py'), *common,
                 '--protocol', path('sixfold_protocol'), '--features-root', path('feature_root')]]
    return [[sys.executable, str(root / 'scripts/research/run_cell_multiseed.py'), *common,
             '--protocols', path('multiseed_protocols'), '--seeds', '43', '44', '45',
             '--features', str(resolve_input(root, config['feature_root']) / 'oral_pretrained/cell_features.npz')]]


def check_assets(root, config):
    for name, expected in config['artifacts_sha256'].items():
        actual = digest(resolve_input(root, name))
        if actual != expected:
            raise ValueError(f'资料校验失败：{name}')
    for key in ['data', 'external_cell_data', 'training_config', 'cell_pretraining_config',
                'sixfold_protocol', 'multiseed_protocols', 'fivefold_results', 'sixfold_results', 'multiseed_results']:
        if not resolve_input(root, config[key]).exists():
            raise FileNotFoundError(f'缺少私有资料或配置：{config[key]}')


def verify(root, config):
    import numpy as np
    import torch
    from run_methylation_cell_transfer import Fusion, Inputs, predict
    from sysu_binary_common import score

    torch.set_num_threads(2)
    path = lambda key: resolve_input(root, config[key])
    five = load(path('fivefold_results') / 'comparison.json')
    six = load(path('sixfold_results') / 'comparison.json')
    multi = load(path('multiseed_results') / 'results.json')
    methods = {f'fivefold/{k}': (r, 'rows') for k, r in five.items()}
    methods.update({f'sixfold/{k}': (r, 'records') for k, r in six['new_methods'].items()})
    methods.update({f'multiseed/{k}': (r, 'records') for k, r in multi['results'].items()})
    expected = None
    summary = {}
    for name, (result, key) in methods.items():
        rows = result[key]
        if len(rows) != 117 or len({r['sample_id'] for r in rows}) != 117:
            raise ValueError(f'评价名单不完整：{name}')
        metrics = score(rows)
        if metrics != result['metrics']:
            raise ValueError(f'归档指标与逐份预测不一致：{name}')
        evaluated = {r['sample_id']: r['grade'] for r in rows if r['probabilities'] is not None}
        if expected is None:
            expected = evaluated
        if evaluated != expected:
            raise ValueError('各次评价使用的有效资料不同')
        summary[name] = {'accuracy': metrics['accuracy'], 'normal_recall': metrics['recalls'][0],
                         'moderate_recall': metrics['moderate_detection'], 'severe_recall': metrics['severe_detection']}
    bundle = load(path('data') / 'bundle.json')
    photos = np.load(path('data') / 'photos.npz', allow_pickle=False)
    if digest(path('data') / 'photos.npz') != bundle['sources']['photos_sha256']:
        raise ValueError('口腔照片特征已改变')
    cache = np.load(path('feature_root') / 'oral_pretrained/cell_features.npz', allow_pickle=False)
    if list(cache['sample_ids']) != [r['sample_id'] for r in bundle['samples']]:
        raise ValueError('细胞图像特征名单不一致')
    folder = path('fivefold_results') / 'oral_pretrained/fold0'
    checkpoint = torch.load(folder / 'refit/classifier.pt', weights_only=True, map_location='cpu')
    fold = bundle['folds'][0]
    inputs = Inputs(bundle, photos, cache['values'], fold['outer_train']['sample_ids'], True)
    if checkpoint['training_ids'] != inputs.eligible(fold['outer_train']['sample_ids']):
        raise ValueError('保存模型的训练名单与原划分不同')
    if (checkpoint['normalization_mean'] != inputs.mean.tolist()
            or checkpoint['normalization_scale'] != inputs.scale.tolist()
            or checkpoint['tct_vocab'] != inputs.vocab):
        raise ValueError('表格处理与保存模型不一致')
    model = Fusion(checkpoint['table_size'], checkpoint['config'])
    model.load_state_dict(checkpoint['state'], strict=True)
    actual = predict(model, inputs, fold['outer_holdout']['sample_ids'], 'cpu', 0)
    archived = load(folder / 'predictions.json')['rows']
    if [r['sample_id'] for r in actual] != [r['sample_id'] for r in archived]:
        raise ValueError('恢复推理的病例顺序不同')
    errors = []
    for a, b in zip(actual, archived, strict=True):
        if (a['probabilities'] is None) != (b['probabilities'] is None):
            raise ValueError('可评价状态不一致')
        if a['probabilities'] is not None:
            errors.append(float(np.abs(np.asarray(a['probabilities']) - b['probabilities']).max()))
            if (a['probabilities'][1] >= .5) != (b['probabilities'][1] >= .5):
                raise ValueError('恢复推理类别不一致')
    if max(errors, default=0.) >= 1e-3:
        raise ValueError('跨设备恢复推理的概率误差过大')
    return {'status': 'verified', 'evaluated_records': len(expected), 'metrics': summary,
            'restored_fold': 1, 'restored_predictions': len(errors), 'cpu_probability_max_error': max(errors, default=0.)}


def main():
    parser = argparse.ArgumentParser(description='当前甲基化与细胞图像预训练研究入口；默认只检查资料，不训练')
    parser.add_argument('action', choices=['inspect', 'verify', 'fivefold', 'sixfold', 'multiseed'], nargs='?', default='inspect')
    parser.add_argument('--config', default='configs/research/pretraining_mainline.json')
    parser.add_argument('--output', help='训练必须指定outputs/experiments下尚不存在的目录')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry-run', action='store_true', help='仅展示命令，不运行、不创建结果目录')
    args = parser.parse_args()
    config = load(resolve_input(ROOT, args.config))
    try:
        if args.action in {'inspect', 'verify'}:
            check_assets(ROOT, config)
            if args.action == 'verify':
                result = verify(ROOT, config)
            else:
                versions = {}
                for name in ['torch', 'torchvision', 'numpy', 'pillow', 'scikit-learn', 'matplotlib']:
                    try:
                        versions[name] = importlib.metadata.version(name)
                    except importlib.metadata.PackageNotFoundError:
                        versions[name] = 'not installed'
                result = {'status': 'assets_verified', 'root': str(ROOT), 'model': config['model'],
                          'versions': versions, 'note': 'scikit-learn用于生成新随机划分，matplotlib用于绘图；本次不改运行环境。'}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if not args.output:
            parser.error('训练须提供--output；默认不会运行训练')
        commands = training_commands(args.action, ROOT, config, args.output, args.device)
        if args.dry_run:
            print(json.dumps({'cwd': str(ROOT), 'commands': commands}, ensure_ascii=False, indent=2))
            return
        check_assets(ROOT, config)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env['PYTHONPATH'] = str(ROOT / 'scripts/research') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        cache = resolve_input(ROOT, config['cache'])
        for name, folder in [('TORCH_HOME', cache / 'torch'), ('MPLCONFIGDIR', cache / 'matplotlib'), ('TMPDIR', cache / 'tmp'), ('TMP', cache / 'tmp'), ('TEMP', cache / 'tmp')]:
            folder.mkdir(parents=True, exist_ok=True)
            env[name] = str(folder)
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
