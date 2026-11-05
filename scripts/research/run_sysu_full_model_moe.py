"""Compare complete-model mean/constant/dynamic gating with honest meta inputs."""
from __future__ import annotations

import argparse
import gc
import itertools
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold

from build_sysu_table_branch import build_branch
from run_sysu_binary_fusion import Inputs, train_stage, save_torch
from sysu_binary_common import BinaryData, read, write, sha, digest, score
from opmd_pred.model.two_branch_fusion import TwoBranchFusion
from opmd_pred.model.full_model_moe import FullModelRouter, full_model_features


def split_groups(data, ids, count, seed):
    grades = [data.grades[s] for s in ids]
    groups = [data.groups[s] for s in ids]
    splitter = StratifiedGroupKFold(count, shuffle=True, random_state=seed)
    parts = [[ids[i] for i in b] for _, b in splitter.split(np.zeros(len(ids)), grades, groups)]
    for a, b in itertools.combinations(parts, 2):
        if {data.groups[s] for s in a} & {data.groups[s] for s in b}:
            raise ValueError('Group split overlap')
    if sorted(s for p in parts for s in p) != sorted(ids):
        raise ValueError('Incomplete group partition')
    return parts


def prepare(config, data):
    root = Path(config['prepared_root'])
    path = root / 'partitions.json'
    six = split_groups(data, data.ids, 6, config['seed'])
    test = six[config['assessment_test_fold']]
    assessment_ids = [s for s in data.ids if s not in set(test)]
    assessment_parts = split_groups(data, assessment_ids, 5, config['seed'])
    deployment_parts = [f['outer_holdout']['sample_ids'] for f in data.protocol['folds']]
    result = {'seed': config['seed'], 'six_parts': six, 'assessment_test': test,
              'assessment_parts': assessment_parts, 'deployment_parts': deployment_parts,
              'csv_sha256': data.config['csv_sha256'], 'original_protocol_sha256': data.config['protocol_sha256']}
    if path.exists() and read(path) != result:
        raise ValueError('Previously frozen partitions changed')
    write(path, result)
    return result


class Study:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.base = read(config['base_config'])
        self.base['cache_root'] = str(Path(config['prepared_root']) / 'table-cache')
        self.data = BinaryData(self.base, genes=False, threshold=False)
        self.partitions = prepare(config, self.data)
        self.root = Path(config['output_root'])
        self.started = time.time()
        sources = [Path(__file__), Path('src/opmd_pred/model/full_model_moe.py'),
                   Path('src/opmd_pred/model/two_branch_fusion.py'),
                   *[Path('scripts/research') / name for name in
                     ('build_sysu_table_branch.py', 'run_sysu_binary_fusion.py',
                      'sysu_binary_common.py', 'sysu_tabular_common.py')]]
        receipt = {'config': config, 'base_config': self.base, 'partitions_sha256': digest(self.partitions),
                   'sources': {p.as_posix(): sha(p) for p in sources}, 'device': device,
                   'torch': torch.__version__, 'test_used_for_selection': False}
        if (self.root / 'freeze.json').exists() and read(self.root / 'freeze.json') != receipt:
            raise ValueError('Study sources/configuration changed; use a new attempt')
        write(self.root / 'freeze.json', receipt)
        self.receipt_sha = digest(receipt)

    def progress(self, **kwargs):
        stage = kwargs.get('stage')
        value = dict(status=stage if stage in ('completed', 'failed') else 'running',
                     started=self.started, updated=time.time(), **kwargs)
        write(self.root / 'state.json', value)
        print({k: v for k, v in value.items() if k not in ('started', 'updated')}, flush=True)

    def train_expert(self, fit, folder, expert_number, scope):
        query = [s for s in self.data.ids if s not in set(fit)]
        seed = self.config['seed'] + expert_number
        branch, cache = build_branch(self.data, 'binary', 'tabpfn', seed, fit, query, scope)
        inputs = Inputs(self.data, 'resnet18', branch, 'transformer', 'binary')
        model_config = dict(feature_dim=512, table_dim=14, num_classes=2,
                            family='transformer', dimension=128, state_size=64)
        identity = {'study_sha256': self.receipt_sha, 'fold': expert_number,
                    'scope': scope, 'table_cache_sha256': sha(cache / 'done.json')}
        done = train_stage(inputs, fit, None, folder / 'fusion', model_config, self.base,
                           seed + 100, self.config['expert_epochs'], identity, self.device)
        artifact = folder / 'fusion/classifier.pt'
        record = {'fit_ids': fit, 'fit_groups': sorted({self.data.groups[s] for s in fit}),
                  'table_cache': str(cache), 'model': str(artifact),
                  'model_sha256': done['classifier_sha256'], 'expert_number': expert_number,
                  'selected_epochs': self.config['expert_epochs']}
        write(folder / 'expert.json', record)
        return record, branch, inputs

    @torch.no_grad()
    def extract(self, expert, inputs, ids, destination):
        if destination.exists():
            saved = np.load(destination, allow_pickle=False)
            if list(saved['sample_ids']) != ids or str(saved['model_sha256']) != expert['model_sha256']:
                raise ValueError('Expert feature cache differs')
            return saved['h0'], saved['probabilities'], saved['available']
        artifact = torch.load(expert['model'], map_location='cpu', weights_only=True)
        model = TwoBranchFusion(**artifact['model_config']).to(self.device).eval()
        model.load_state_dict(artifact['state'], strict=True)
        model.requires_grad_(False)
        h0 = np.zeros((len(ids), 3, 128), np.float32)
        probabilities = np.full((len(ids), 2), np.nan, np.float32)
        positions = {s: i for i, s in enumerate(ids)}
        available = inputs.eligible(ids)
        for start in range(0, len(available), 8):
            chunk = available[start:start + 8]
            initial, p = full_model_features(model, inputs.pack(chunk, self.device))
            ix = [positions[s] for s in chunk]
            h0[ix], probabilities[ix] = initial.cpu().numpy(), p.cpu().numpy()
        flags = np.asarray([s in set(available) for s in ids])
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, sample_ids=np.asarray(ids), h0=h0, probabilities=probabilities,
                            available=flags, model_sha256=np.asarray(expert['model_sha256']))
        del model
        gc.collect()
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        return h0, probabilities, flags

    def fit_gate(self, mode, ids, h0, probabilities, available, folder):
        selected = np.flatnonzero(available)
        if not len(selected):
            raise ValueError('No gate training records')
        x = torch.tensor(h0[selected], device=self.device)
        p = torch.tensor(probabilities[selected], device=self.device)
        y = torch.tensor(self.data.target([ids[i] for i in selected], 'binary'), device=self.device)
        counts = torch.bincount(y, minlength=2)
        if not counts.all():
            raise ValueError('Missing gate training class')
        weights = len(y) / (2 * counts.float())
        model = FullModelRouter(mode=mode).to(self.device)
        history = []
        if mode != 'mean':
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['router_lr'],
                                          weight_decay=self.config['router_weight_decay'])
            for epoch in range(self.config['router_epochs']):
                output, alpha = model(x, p)
                loss = F.nll_loss(output.clamp_min(1e-12).log(), y, weight=weights)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite Router training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                history.append({'epoch': epoch + 1, 'loss': float(loss.detach())})
        model.eval()
        save_torch(folder / f'{mode}.pt', {'format': 'sysu-full-model-router-v1',
            'model_config': {'experts': 5, 'dimension': 128, 'mode': mode},
            'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'study_sha256': self.receipt_sha, 'training_ids': [ids[i] for i in selected],
            'class_counts': counts.tolist(), 'history': history})
        return model

    @torch.no_grad()
    def evaluate(self, model, ids, h0, probabilities, available):
        selected = np.flatnonzero(available)
        mixed, alpha = model(torch.tensor(h0[selected], device=self.device),
                             torch.tensor(probabilities[selected], device=self.device))
        lookup = {ids[i]: p for i, p in zip(selected, mixed.cpu().tolist())}
        rows = self.data.prediction_rows(ids, lookup, -1)
        for i, a, p in zip(selected, alpha.cpu().tolist(), probabilities[selected].tolist()):
            rows[i].update(router_weights=a, complete_expert_probabilities=p)
        return {'records': rows, 'metrics': score(rows, 'binary'),
                'mean_router_weights': alpha.mean(0).cpu().tolist(),
                'maximum_weight_mean': float(alpha.max(1).values.mean().cpu())}

    def run_scope(self, scope):
        parts = self.partitions[scope + '_parts']
        scope_ids = [s for s in self.data.ids if any(s in p for p in parts)]
        test_ids = self.partitions['assessment_test'] if scope == 'assessment' else scope_ids
        folder = self.root / scope
        specs, query_h, query_p, query_a = [], [], [], []
        # Five final complete experts each exclude a different one of five parts.
        for e, omitted in enumerate(parts):
            fit = [s for s in scope_ids if s not in set(omitted)]
            self.progress(scope=scope, stage='final_expert', expert=e)
            spec, branch, inputs = self.train_expert(fit, folder / f'experts/expert{e}', e, f'{scope}/expert{e}/final')
            h, p, a = self.extract(spec, inputs, test_ids, folder / f'experts/expert{e}/evaluation.npz')
            query_h.append(h); query_p.append(p); query_a.append(a); specs.append(spec)
        # Each row needs FIVE unseen complete predictions, not only its one OOF expert.
        # Temporary copies remove the meta-held group as well as the expert's own omitted group.
        meta_h = np.zeros((len(scope_ids), 5, 3, 128), np.float32)
        meta_p = np.full((len(scope_ids), 5, 2), np.nan, np.float32)
        meta_a = np.zeros((len(scope_ids), 5), bool)
        positions = {s: i for i, s in enumerate(scope_ids)}
        audit = []
        for m, held in enumerate(parts):
            held_groups = {self.data.groups[s] for s in held}
            for e, omitted in enumerate(parts):
                fit = [s for s in scope_ids if s not in set(held + omitted)]
                self.progress(scope=scope, stage='honest_router_inputs', meta_part=m, expert=e)
                if {self.data.groups[s] for s in fit} & held_groups:
                    raise ValueError('Meta expert has seen the held group')
                own = folder / f'meta/part{m}/expert{e}'
                final = folder / f'experts/expert{e}'
                target = final if m == e else own
                model_scope = f'{scope}/expert{e}/final' if m == e else f'{scope}/meta{m}/expert{e}'
                spec, branch, inputs = self.train_expert(fit, target, e, model_scope)
                h, p, a = self.extract(spec, inputs, held, own / 'features.npz')
                ix = [positions[s] for s in held]
                meta_h[ix, e], meta_p[ix, e], meta_a[ix, e] = h, p, a
                audit.append({'meta_part': m, 'expert': e, 'fit_ids': fit, 'unseen_ids': held,
                              'overlap_groups': 0, 'expert_model_sha256': spec['model_sha256']})
        if not np.all(meta_a == meta_a[:, :1]):
            raise ValueError('Experts disagree about input availability')
        query_h = np.stack(query_h, 1)
        query_p = np.stack(query_p, 1)
        query_a = np.stack(query_a, 1)
        if not np.all(query_a == query_a[:, :1]):
            raise ValueError('Expert query availability differs')
        np.savez_compressed(folder / 'router_training_inputs.npz', sample_ids=np.asarray(scope_ids),
                            h0=meta_h, probabilities=meta_p, available=meta_a[:, 0])
        write(folder / 'training_exclusion_audit.json', audit)
        self.progress(scope=scope, stage='fit_routers')
        result = {'scope': scope, 'study_sha256': self.receipt_sha, 'test_used_for_selection': False,
                  'evaluation_kind': 'held_out_development_partition' if scope == 'assessment' else 'training_fit_check',
                  'training_ids': scope_ids, 'evaluation_ids': test_ids, 'methods': {},
                  'training_input_shift': 'temporary experts exclude the meta group; final experts use four of five groups',
                  'expert_specs': specs, 'app_replaced': False}
        for mode in ('mean', 'constant', 'dynamic'):
            gate = self.fit_gate(mode, scope_ids, meta_h, meta_p, meta_a[:, 0], folder)
            result['methods'][mode] = self.evaluate(gate, test_ids, query_h, query_p, query_a[:, 0])
            result['methods'][mode]['honest_training_metrics'] = self.evaluate(
                gate, scope_ids, meta_h, meta_p, meta_a[:, 0])['metrics']
            artifact = torch.load(folder / f'{mode}.pt', map_location='cpu', weights_only=True)
            restored = FullModelRouter(**artifact['model_config']).to(self.device)
            restored.load_state_dict(artifact['state'], strict=True)
            reread = self.evaluate(restored, test_ids, query_h, query_p, query_a[:, 0])
            if reread != {k: v for k, v in result['methods'][mode].items() if k != 'honest_training_metrics'}:
                raise ValueError('Reloaded ensemble predictions changed')
            print(scope, mode, result['methods'][mode]['metrics'], flush=True)
            del gate, restored
        write(folder / 'results.json', result)
        if scope == 'deployment':
            write(folder / 'candidate_manifest.json', {'format': 'sysu-full-model-moe-candidate-v1',
                'encoder': self.base['images']['resnet18']['encoder'],
                'encoder_sha256': self.base['images']['resnet18']['encoder_sha256'],
                'experts': specs, 'router': str(folder / 'dynamic.pt'),
                'router_sha256': sha(folder / 'dynamic.pt'), 'class_labels': ['正常/轻度', '中度及以上'],
                'decision_threshold': .5, 'server_only': True, 'app_replaced': False,
                'scope': 'all development data; probabilities on this set are training-fit results'})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/sysu_moe/full_model_router.json')
    parser.add_argument('--scope', choices=['assessment', 'deployment', 'both'], default='both')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    study = Study(read(args.config), args.device)
    try:
        for scope in ('assessment', 'deployment') if args.scope == 'both' else (args.scope,):
            study.run_scope(scope)
    except Exception as exc:
        study.progress(status_detail='failed', stage='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    study.progress(stage='completed', status_detail='completed', app_replaced=False)


if __name__ == '__main__':
    main()
