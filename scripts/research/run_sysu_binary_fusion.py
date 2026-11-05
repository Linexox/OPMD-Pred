"""Nested, fixed-fold photo/table fusion with separate binary and three-class targets."""
from __future__ import annotations

import argparse
import gc
import itertools
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from opmd_pred.model.two_branch_fusion import TwoBranchFusion, drop_branches
from build_sysu_table_branch import build_branch
from sysu_binary_common import BinaryData, digest, read, score, sha, write


def save_torch(path, value):
    temp = path.with_name(path.name + '.writing')
    torch.save(value, temp)
    temp.replace(path)


class Inputs:
    def __init__(self, data, image, branch, family, task):
        self.data, self.task, self.family = data, task, family
        self.images, self.image_present = data.photos(image)
        self.branch = branch
        self.classes = 2 if task == 'binary' else 3
        self.table_dim = self.classes + len(branch['columns'])

    def eligible(self, ids):
        return [s for s in ids if self.image_present[self.data.index[s]] or
                (self.family != 'image_only' and self.branch['available'][s])]

    def pack(self, ids, device, training=False):
        ix = np.asarray([self.data.index[s] for s in ids])
        views = torch.randint(1, 5, (len(ids),)).numpy() if training else np.zeros(len(ids), dtype=int)
        photo = torch.tensor(self.images[ix, views], device=device)
        tab = np.zeros((len(ids), self.table_dim), dtype=np.float32)
        present = np.zeros(len(ids), dtype=bool)
        for i, s in enumerate(ids):
            if self.family != 'image_only' and self.branch['available'][s]:
                tab[i] = self.branch['probabilities'][s] + self.branch['field_masks'][s]
                present[i] = True
        return {'image': photo, 'table': torch.tensor(tab, device=device),
                'image_present': torch.tensor(self.image_present[ix], device=device),
                'table_present': torch.tensor(present, device=device),
                'target': torch.tensor(self.data.target(ids, self.task), device=device)}


@torch.no_grad()
def predict(model, inputs, ids, device):
    model.eval()
    probabilities = {}
    chosen = inputs.eligible(ids)
    for start in range(0, len(chosen), 8):
        chunk = chosen[start:start + 8]
        p = model(inputs.pack(chunk, device)).softmax(1).cpu().numpy()
        if not np.isfinite(p).all():
            raise ValueError('Nonfinite network prediction')
        probabilities.update(zip(chunk, p.tolist()))
    return probabilities


def train_stage(inputs, train_ids, validation_ids, folder, model_config, config, seed, max_epochs, identity, device):
    folder.mkdir(parents=True, exist_ok=True)
    own_identity = dict(identity, training_ids=train_ids, validation_ids=validation_ids,
                        model_config=model_config, seed=seed, max_epochs=max_epochs)
    if (folder / 'done.json').exists():
        result = read(folder / 'done.json')
        if result['identity'] != own_identity or (validation_ids is None and sha(folder / 'classifier.pt') != result['classifier_sha256']):
            raise ValueError('Completed training identity differs')
        return result
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    chosen = inputs.eligible(train_ids)
    y = inputs.data.target(chosen, inputs.task)
    counts = np.bincount(y, minlength=inputs.classes)
    if (counts == 0).any():
        raise ValueError('Missing training class')
    weights = torch.tensor(len(y) / (inputs.classes * counts), dtype=torch.float32, device=device)
    model = TwoBranchFusion(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    history, best, selected_epoch, stale, first = [], (-1., -1., -float('inf')), 0, 0, 1
    if (folder / 'last.pt').exists():
        previous = torch.load(folder / 'last.pt', map_location=device, weights_only=True)
        if previous['identity'] != own_identity:
            raise ValueError('Resume training identity differs')
        model.load_state_dict(previous['state'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        history, best = previous['history'], tuple(previous['best'])
        selected_epoch, stale, first = previous['selected_epoch'], previous['stale'], previous['epoch'] + 1
        torch.set_rng_state(previous['cpu_rng'].cpu())
        if device.startswith('cuda'):
            torch.cuda.set_rng_state(previous['cuda_rng'].cpu())
    started = time.monotonic()
    for epoch in range(first, max_epochs + 1):
        if validation_ids is not None and stale >= config['patience']:
            break
        model.train()
        order = torch.randperm(len(chosen)).tolist()
        total = 0.
        for start in range(0, len(order), config['batch_size']):
            chunk = [chosen[i] for i in order[start:start + config['batch_size']]]
            batch = inputs.pack(chunk, device, training=True)
            if inputs.family != 'image_only':
                batch = drop_branches(batch, config['modality_dropout'])
            loss = model.supervised_loss(batch, weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite gradients')
            optimizer.step()
            total += float(loss.detach()) * len(chunk)
        item = {'epoch': epoch, 'training_loss': total / len(chosen)}
        if validation_ids is not None:
            probabilities = predict(model, inputs, validation_ids, device)
            rows = inputs.data.prediction_rows(validation_ids, probabilities, identity['fold'])
            metrics = score(rows, inputs.task)
            eligible = [s for s in validation_ids if s in probabilities]
            probs = torch.tensor([probabilities[s] for s in eligible], dtype=torch.float32, device=device)
            targets = torch.tensor(inputs.data.target(eligible, inputs.task), device=device)
            val_loss = float(F.nll_loss(probs.clamp_min(1e-12).log(), targets, weight=weights))
            item.update(validation=metrics, validation_loss=val_loss)
            key = (metrics['balanced_accuracy'], metrics['severe_detection'], -val_loss)
            if key > best:
                best, selected_epoch, stale = key, epoch, 0
            else:
                stale += 1
        else:
            selected_epoch = epoch
        history.append(item)
        save_torch(folder / 'last.pt', {'identity': own_identity, 'state': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'history': history, 'best': best,
            'selected_epoch': selected_epoch, 'stale': stale, 'epoch': epoch,
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state() if device.startswith('cuda') else None})
    result = {'identity': own_identity, 'history': history, 'selected_epoch': selected_epoch,
              'class_counts': counts.tolist(), 'class_weights': weights.cpu().tolist(),
              'seconds_this_invocation': time.monotonic() - started,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if validation_ids is None:
        save_torch(folder / 'classifier.pt', {'format': 'sysu-two-branch-classifier-v1',
            'model_config': model_config, 'identity': own_identity,
            'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}})
        result['classifier_sha256'] = sha(folder / 'classifier.pt')
    write(folder / 'done.json', result)
    del optimizer, model
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def run_combination(data, config, task, image, table, family, seed, folds, device):
    fields = config['fields'] + ('_with_di' if config['threshold_input'] else '_mixed_di')
    path = Path(config['output_root']) / fields / task / image / table / family / f'seed{seed}' / 'attempt1'
    source_files = [Path(__file__), Path(__file__).with_name('sysu_binary_common.py'),
        Path(__file__).with_name('build_sysu_table_branch.py'), Path(__file__).with_name('sysu_tabular_common.py'),
        Path('src/opmd_pred/model/two_branch_fusion.py')]
    receipt = {'data_sha256': config['csv_sha256'], 'protocol_sha256': config['protocol_sha256'],
               'config': config, 'sources': {p.as_posix(): sha(p) for p in source_files},
               'task': task, 'image': image, 'table': table, 'fusion': family, 'seed': seed,
               'image_source': config['images'][image]}
    if (path / 'freeze.json').exists() and read(path / 'freeze.json') != receipt:
        raise ValueError('Combination source/configuration changed')
    write(path / 'freeze.json', receipt)
    for fold in data.protocol['folds']:
        n = fold['fold']
        if n not in folds:
            continue
        folder = path / f'fold{n}'
        held = fold['outer_holdout']['sample_ids']
        started = time.monotonic()
        if (folder / 'outer_predictions.json').exists():
            previous = read(folder / 'outer_predictions.json')
            if previous['receipt_sha256'] != digest(receipt) or sha(folder / 'refit/classifier.pt') != previous['classifier_sha256']:
                raise ValueError('Completed fold changed')
            continue
        if family == 'image_only':
            # No table model or patient labels are needed to form image-only input.
            empty = {'columns': [], 'probabilities': {}, 'field_masks': {s: [] for s in data.ids},
                     'available': {s: False for s in data.ids}}
            inner_branch = outer_branch = empty
            cache_paths = []
        else:
            inner_branch, inner_path = build_branch(data, task, table, seed + n,
                fold['inner_train']['sample_ids'], fold['inner_val']['sample_ids'], f'fold{n}/inner')
            outer_branch, outer_path = build_branch(data, task, table, seed + n,
                fold['outer_train']['sample_ids'], held, f'fold{n}/outer')
            cache_paths = [inner_path.as_posix(), outer_path.as_posix()]
        inner_inputs = Inputs(data, image, inner_branch, family, task)
        outer_inputs = Inputs(data, image, outer_branch, family, task)
        model_config = {'feature_dim': config['images'][image]['feature_dim'], 'table_dim': inner_inputs.table_dim,
            'num_classes': inner_inputs.classes, 'family': family, 'dimension': config['dimension'], 'state_size': config['state_size']}
        identity = {'receipt_sha256': digest(receipt), 'fold': n, 'cache_paths': cache_paths,
            'table_cache_sha256': [sha(Path(p) / 'done.json') for p in cache_paths]}
        inner = train_stage(inner_inputs, fold['inner_train']['sample_ids'], fold['inner_val']['sample_ids'],
            folder / 'inner', model_config, config, seed + n, config['inner_epochs'], identity, device)
        refit = train_stage(outer_inputs, fold['outer_train']['sample_ids'], None, folder / 'refit',
            model_config, config, seed + 100 + n, inner['selected_epoch'], identity, device)
        artifact = torch.load(folder / 'refit/classifier.pt', map_location='cpu', weights_only=True)
        model = TwoBranchFusion(**artifact['model_config']).to(device)
        model.load_state_dict(artifact['state'], strict=True)
        probabilities = predict(model, outer_inputs, held, device)
        # Independently reloading the serialized classifier must preserve its output.
        restored = TwoBranchFusion(**artifact['model_config']).to(device)
        restored.load_state_dict(torch.load(folder / 'refit/classifier.pt', map_location='cpu', weights_only=True)['state'], strict=True)
        reread = predict(restored, outer_inputs, held, device)
        error = max((abs(a - b) for s in probabilities for a, b in zip(probabilities[s], reread[s])), default=0.)
        if error > 1e-6:
            raise ValueError('Fusion reload changed probabilities')
        rows = data.prediction_rows(held, probabilities, n)
        out = {'records': rows, 'metrics': score(rows, task), 'receipt_sha256': digest(receipt),
               'selected_epoch': inner['selected_epoch'], 'classifier_sha256': refit['classifier_sha256'],
               'reload_max_absolute_error': error, 'table_cache_paths': cache_paths,
               'seconds': time.monotonic() - started}
        if family != 'image_only':
            table_rows = data.prediction_rows(held, outer_branch['probabilities'], n)
            out.update(table_records=table_rows, table_metrics=score(table_rows, task))
        write(folder / 'outer_predictions.json', out)
        print(f'{fields} {task} {image} {table} {family} seed{seed} fold{n} acc={out["metrics"]["accuracy"]:.4f} seconds={out["seconds"]:.1f}', flush=True)
        del model, restored
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    available = [path / f'fold{n}/outer_predictions.json' for n in range(5)]
    if all(p.exists() for p in available):
        payloads = [read(p) for p in available]
        rows = [r for d in payloads for r in d['records']]
        if len(rows) != 117 or len({r['sample_id'] for r in rows}) != 117:
            raise ValueError('Incomplete development OOF')
        out = {'records': rows, 'metrics': score(rows, task), 'receipt': receipt,
               'fold_files': {p.as_posix(): sha(p) for p in available}}
        if family != 'image_only':
            table_rows = [r for d in payloads for r in d['table_records']]
            out.update(table_records=table_rows, table_metrics=score(table_rows, task))
        write(path / 'oof_results.json', out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--task', choices=['binary', 'threeclass'])
    p.add_argument('--image', choices=['resnet18', 'vit_b16'])
    p.add_argument('--table', choices=['logistic', 'tabpfn'])
    p.add_argument('--fusion', choices=['transformer', 'multimodn', 'image_only'])
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    c = read(a.config)
    data = BinaryData(c, c['fields'] == 'genes', c['threshold_input'])
    choices = [([val] if val else c[key]) for val, key in
               [(a.task, 'task_choices'), (a.image, 'image_choices'), (a.table, 'table_choices'), (a.fusion, 'fusion_choices')]]
    for seed in a.seeds:
        for task, image, table, family in itertools.product(*choices):
            run_combination(data, c, task, image, table, family, seed, a.folds, a.device)
        if c['photo_baselines'] and not a.fusion:
            for task, image in itertools.product(choices[0], choices[1]):
                run_combination(data, c, task, image, 'none', 'image_only', seed, a.folds, a.device)


if __name__ == '__main__':
    main()
