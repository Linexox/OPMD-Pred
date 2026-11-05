"""Compare training dropout with the frozen six-fold complete-model mean baseline.

Reuse original training implementation and immutable TabPFN/image caches.
Inject only the Transformer dropout constructor in this experiment process.
"""
import argparse
import gc
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import run_sysu_binary_fusion as training
from opmd_pred.model.moe_regularization import set_transformer_dropout
from opmd_pred.model.two_branch_fusion import TwoBranchFusion
from sysu_binary_common import BinaryData, digest, read, score, sha, write


def configured_model(config, probability):
    model = set_transformer_dropout(TwoBranchFusion(**config), probability)
    controls = [m.p for m in model.fusion.modules() if isinstance(m, nn.Dropout)]
    controls += [m.dropout for m in model.fusion.modules() if isinstance(m, nn.MultiheadAttention)]
    if not controls or any(value != probability for value in controls):
        raise ValueError('Transformer dropout controls were not all set')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/sysu_moe/dropout_zero_sixfold.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = read(args.config)
    base = read(config['base_config'])
    data = BinaryData(base, genes=False, threshold=False)
    baseline = read(config['baseline_results'])
    parts = read(config['partition_file'])['six_parts']
    root = Path(config['output_root'])
    originals = {}
    sources = [Path(__file__), Path(training.__file__),
               Path('src/opmd_pred/model/moe_regularization.py'),
               Path('src/opmd_pred/model/two_branch_fusion.py'),
               Path('scripts/research/sysu_binary_common.py')]
    frozen = {'config': config, 'base_config': base, 'device': args.device,
              'torch': torch.__version__, 'source_sha256': {str(p): sha(p) for p in sources},
              'baseline_sha256': sha(config['baseline_results']),
              'partitions_sha256': sha(config['partition_file']), 'test_used_for_selection': False}
    if (root/'freeze.json').exists() and read(root/'freeze.json') != frozen:
        raise ValueError('Experiment inputs changed; use a new attempt')
    write(root/'freeze.json', frozen)
    started = time.time()

    def progress(**values):
        state = {'status': 'running', 'started': started, 'updated': time.time(),
                 'app_replaced': False, **values}
        write(root/'state.json', state)
        print(state, flush=True)

    def train(old, inputs, folder, probability, fold):
        original = torch.load(old['model'], map_location='cpu', weights_only=True)
        expected = original['identity']['training_ids']
        if expected != old['fit_ids']:
            raise ValueError('Original training order differs')
        identity = {'experiment_sha256': digest(frozen), 'fold': fold,
                    'source_classifier_sha256': old['model_sha256'],
                    'fusion_dropout': probability}
        with patch.object(training, 'TwoBranchFusion',
                          lambda **kwargs: configured_model(kwargs, probability)):
            training.train_stage(inputs, expected, None, folder, original['model_config'],
                                 base, original['identity']['seed'], config['expert_epochs'],
                                 identity, args.device)
        saved = torch.load(folder/'classifier.pt', map_location='cpu', weights_only=True)
        return original, saved

    folds = []
    try:
        for fold, old_root in enumerate(baseline['fold_roots']):
            old_folder = Path(old_root)/'assessment'
            old_path = old_folder/'inference-results.json'
            if not old_path.exists():
                old_path = old_folder/'results.json'
            originals[str(old_path)] = sha(old_path)
            old_result = read(old_path)
            ids = old_result['evaluation_ids']
            if ids != parts[fold]:
                raise ValueError('Held-out partition differs')
            member_outputs, member_specs, available = [], [], None
            for expert, old in enumerate(old_result['expert_specs']):
                progress(fold=fold, expert=expert, stage='train')
                if sha(old['model']) != old['model_sha256']:
                    raise ValueError('Original classifier changed')
                originals[old['model']] = old['model_sha256']
                held_groups = {data.groups[s] for s in ids}
                if held_groups & {data.groups[s] for s in old['fit_ids']}:
                    raise ValueError('Held-out group appears in training')
                branch = read(Path(old['table_cache'])/'done.json')
                if set(branch['identity']['fit_ids']) != set(old['fit_ids']):
                    raise ValueError('TabPFN training membership differs')
                for name, fingerprint in branch['files'].items():
                    path = str(Path(old['table_cache'])/name)
                    if sha(path) != fingerprint:
                        raise ValueError('TabPFN cache changed')
                    originals[path] = fingerprint
                for s in ids:
                    if branch['available'][s] and not all(s in a['unseen_query_ids'] for a in branch['assignments']):
                        raise ValueError('Held-out row does not use all five unseen contexts')
                inputs = training.Inputs(data, 'resnet18', branch, 'transformer', 'binary')
                if fold == 0 and expert == 0:
                    original, check = train(old, inputs, root/'baseline-replay', 0.1, fold)
                    if any(not torch.equal(v, check['state'][k]) for k, v in original['state'].items()):
                        raise ValueError('Dropout 0.1 replay differs from original training')
                    write(root/'baseline-replay/check.json', {'passed': True, 'all_parameters_exactly_equal': True})
                folder = root/f'fold{fold}/expert{expert}'
                original, saved = train(old, inputs, folder, config['fusion_dropout'], fold)
                model = configured_model(saved['model_config'], config['fusion_dropout']).to(args.device)
                model.load_state_dict(saved['state'], strict=True)
                output = training.predict(model, inputs, ids, args.device)
                if available is not None and set(output) != available:
                    raise ValueError('Member input availability differs')
                available = set(output)
                member_outputs.append(output)
                spec = {'source_expert': old, 'classifier': str(folder/'classifier.pt'),
                        'sha256': sha(folder/'classifier.pt'), 'training_dropout': config['fusion_dropout']}
                member_specs.append(spec)
                write(folder/'predictions.json', {'probabilities': output, 'spec': spec})
                del model, saved, original
                gc.collect()
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()
            means = {s: np.mean([output[s] for output in member_outputs], axis=0).tolist() for s in available}
            rows = data.prediction_rows(ids, means, fold)
            for row in rows:
                if row['probabilities'] is not None:
                    row['complete_expert_probabilities'] = [output[row['sample_id']] for output in member_outputs]
            result = {'fold': fold, 'records': rows, 'metrics': score(rows, 'binary'), 'experts': member_specs}
            folds.append(result)
            write(root/f'fold{fold}/results.json', result)
            print('FOLD_COMPLETED', fold + 1, result['metrics'], flush=True)
        rows = [r for fold in folds for r in fold['records']]
        if len(rows) != 117 or len({r['sample_id'] for r in rows}) != 117:
            raise ValueError('Incomplete held-out predictions')
        if set(r['sample_id'] for r in rows) != set(r['sample_id'] for r in baseline['methods']['mean']['records']):
            raise ValueError('Baseline cohort differs')
        for path, fingerprint in originals.items():
            if sha(path) != fingerprint:
                raise ValueError('Original artifact changed during experiment')
        accuracies = [fold['metrics']['accuracy'] for fold in folds]
        result = {'config': config, 'records': rows, 'metrics': score(rows, 'binary'),
                  'per_fold': [fold['metrics'] for fold in folds],
                  'fold_accuracy_mean': float(np.mean(accuracies)),
                  'fold_accuracy_sample_std': float(np.std(accuracies, ddof=1)),
                  'fold_accuracy_max': float(np.max(accuracies)),
                  'baseline': {k: v for k, v in baseline['methods']['mean'].items() if k != 'records'},
                  'source_artifacts_sha256': originals, 'app_replaced': False,
                  'test_used_for_selection': False, 'new_table_models': 0}
        write(root/'oof_results.json', result)
        progress(status='completed', trained_classifiers=30, stage='completed')
        print('COMPLETE', result['metrics'], flush=True)
    except Exception as error:
        progress(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
