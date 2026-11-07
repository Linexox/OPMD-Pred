"""Independently check all held-out predictions and report the added seeds."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_methylation_cell_transfer import Fusion, Inputs, predict
from sysu_binary_common import sha, write


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def independent_metrics(rows):
    destinations = np.zeros((3, 2), dtype=int)
    absent = np.zeros(3, dtype=int)
    for row in rows:
        grade = row['grade']
        if row['probabilities'] is None:
            absent[grade] += 1
            continue
        p = np.asarray(row['probabilities'])
        assert p.shape == (2,) and np.isfinite(p).all()
        assert (p >= 0).all() and np.isclose(p.sum(), 1, atol=1e-6)
        destinations[grade, int(p[1] >= .5)] += 1
    n = int(destinations.sum())
    correct = int(destinations[0, 0] + destinations[1:, 1].sum())
    return destinations, absent, n, correct


def check_metrics(rows, metrics):
    dest, absent, n, correct = independent_metrics(rows)
    assert metrics['assigned_count'] == len(rows)
    assert metrics['eligible_count'] == n
    assert metrics['abstain_by_grade'] == absent.tolist()
    assert metrics['original_grade_destinations'] == dest.tolist()
    assert abs(metrics['accuracy'] - correct / n) < 1e-12
    return dest, n, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--protocols', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--methyl-checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = read(args.results / 'results.json')
    identity = read(args.results / 'identity.json')
    bundle = read(args.data / 'bundle.json')
    samples = {s['sample_id']: s for s in bundle['samples']}
    photo = np.load(args.data / 'photos.npz', allow_pickle=False)
    features = np.load(args.features, allow_pickle=False)
    assert list(features['sample_ids']) == list(samples)
    cells = features['values']
    assert sha(args.data / 'bundle.json') == identity['bundle_sha256']
    assert sha(args.data / 'photos.npz') == bundle['sources']['photos_sha256']
    assert sha(args.features) == identity['cell_features_sha256']
    assert sha(args.methyl_checkpoint) == identity['methyl_sha256']
    for name, expected in identity['sources'].items():
        assert sha(Path(__file__).with_name(name)) == expected
    reference = read(args.reference)['new_methods']['oral_pretrained']
    expected_eligible = {r['sample_id']: r['grade'] for r in reference['records'] if r['probabilities'] is not None}
    expected_missing = {r['sample_id'] for r in reference['records'] if r['probabilities'] is None}
    checked = []
    aggregate = []
    signatures = []
    max_error = 0.
    max_ensemble_error = 0.
    for key, result in data['results'].items():
        seed = int(key)
        protocol_path = args.protocols / f'seed{seed}.json'
        assert sha(protocol_path) == identity['protocol_sha256'][key]
        protocol = read(protocol_path)
        assert protocol['seed'] == seed
        assert protocol['bundle_sha256'] == identity['bundle_sha256']
        signature = {frozenset(f['held_ids']) for f in protocol['folds']}
        assert all(signature != s for s in signatures)
        signatures.append(signature)
        held_all = []
        for fold, spec in zip(result['folds'], protocol['folds'], strict=True):
            n = spec['fold']
            held = spec['held_ids']
            held_all.extend(held)
            assert fold['fold'] == n
            assert [r['sample_id'] for r in fold['records']] == held
            cpu_members = []
            for expert in spec['experts']:
                e = expert['expert']
                folder = args.results / f'seed{seed}' / f'fold{n}' / f'expert{e}'
                fit_ids = expert['fit_ids']
                inner_ids = expert['inner_train']
                val_ids = expert['inner_validation']
                assert set(inner_ids).isdisjoint(val_ids)
                assert set(inner_ids) | set(val_ids) == set(fit_ids)
                for left, right in [(fit_ids, held), (inner_ids, val_ids)]:
                    assert {samples[s]['group'] for s in left}.isdisjoint({samples[s]['group'] for s in right})
                checkpoint = folder / 'refit/classifier.pt'
                saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
                inputs = Inputs(bundle, photo, cells, fit_ids, True)
                assert saved['training_ids'] == inputs.eligible(fit_ids)
                assert sha(checkpoint) == fold['experts'][e]['classifier_sha256']
                assert fold['experts'][e]['reload_error'] < 1e-6
                assert saved['normalization_mean'] == inputs.mean.tolist()
                assert saved['normalization_scale'] == inputs.scale.tolist()
                assert saved['tct_vocab'] == inputs.vocab
                inner = read(folder / 'inner/training.json')
                inner_inputs = Inputs(bundle, photo, cells, inner_ids, True)
                assert inner['training_ids'] == inner_inputs.eligible(inner_ids)
                assert inner['validation_ids'] == val_ids
                assert inner['normalization_mean'] == inner_inputs.mean.tolist()
                assert inner['normalization_scale'] == inner_inputs.scale.tolist()
                assert inner['tct_vocab'] == inner_inputs.vocab
                assert inner['selected_epoch'] == fold['experts'][e]['selected_epoch']
                assert read(folder / 'refit/training.json')['selected_epoch'] == inner['selected_epoch']
                restored = Fusion(saved['table_size'], saved['config'])
                restored.load_state_dict(saved['state'], strict=True)
                actual = predict(restored, inputs, held, 'cpu', n)
                cpu_members.append(actual)
                archived = read(folder / 'predictions.json')['records']
                assert [r['sample_id'] for r in archived] == held
                error = 0.
                for i, (a, b) in enumerate(zip(actual, archived, strict=True)):
                    assert a['grade'] == b['grade'] == samples[a['sample_id']]['grade']
                    assert (a['probabilities'] is None) == (b['probabilities'] is None)
                    assert b['probabilities'] == fold['records'][i]['member_probabilities'][e]
                    if a['probabilities'] is not None:
                        error = max(error, float(np.abs(np.asarray(a['probabilities']) - b['probabilities']).max()))
                # Cross-device kernels need not produce bit-identical floats.
                # Separately require every final ensemble classification to agree.
                assert error < 1e-3, (seed, n, e, error)
                max_error = max(max_error, error)
                checked.append({'seed': seed, 'fold': n, 'expert': e, 'classifier_sha256': sha(checkpoint), 'cpu_probability_max_error': error})
            for i, row in enumerate(fold['records']):
                assert row['weights'] == [.2] * 5
                if row['probabilities'] is not None:
                    assert np.allclose(np.mean(row['member_probabilities'], axis=0), row['probabilities'], atol=1e-12, rtol=0)
                    cpu_mean = np.mean([member[i]['probabilities'] for member in cpu_members], axis=0)
                    max_ensemble_error = max(max_ensemble_error, float(np.abs(cpu_mean - row['probabilities']).max()))
                    assert (cpu_mean[1] >= .5) == (row['probabilities'][1] >= .5), (seed, n, row['sample_id'])
            check_metrics(fold['records'], fold['metrics'])
        assert len(held_all) == len(set(held_all)) == len(samples) == 117
        assert set(held_all) == set(samples)
        rows = [r for fold in result['folds'] for r in fold['records']]
        assert rows == result['records']
        assert {r['sample_id']: r['grade'] for r in rows if r['probabilities'] is not None} == expected_eligible
        assert {r['sample_id'] for r in rows if r['probabilities'] is None} == expected_missing
        dest, n, correct = check_metrics(rows, result['metrics'])
        fold_acc = [f['metrics']['accuracy'] for f in result['folds']]
        assert abs(np.mean(fold_acc) - result['fold_accuracy_mean']) < 1e-12
        assert max(fold_acc) == result['fold_accuracy_max']
        best = next(f for f in result['folds'] if f['metrics']['accuracy'] == max(fold_acc))
        aggregate.append({'seed': seed, 'correct': correct, 'n': n, 'accuracy': correct/n,
                          'fold_mean': float(np.mean(fold_acc)), 'highest_fold': best['fold']+1,
                          'highest_accuracy': max(fold_acc), 'highest_correct': int(round(max(fold_acc)*best['metrics']['eligible_count'])),
                          'highest_n': best['metrics']['eligible_count'], 'destinations': dest.tolist(),
                          'normal_recall': dest[0, 0]/dest[0].sum(), 'positive_recall': dest[1:, 1].sum()/dest[1:].sum(),
                          'moderate_recall': dest[1, 1]/dest[1].sum(), 'severe_recall': dest[2, 1]/dest[2].sum()})
    summary = {name: {'mean': float(np.mean([r[name] for r in aggregate])), 'std': float(np.std([r[name] for r in aggregate], ddof=1)),
                      'min': float(min(r[name] for r in aggregate)), 'max': float(max(r[name] for r in aggregate))}
               for name in ['accuracy', 'fold_mean', 'normal_recall', 'positive_recall', 'moderate_recall', 'severe_recall']}
    write(args.results / 'independent_verification.json', {'checked_classifiers': len(checked), 'checks': checked,
          'same_evaluable_records_as_seed42': True, 'new_patient_group_splits': True, 'equal_mean_verified': True,
          'cpu_probability_max_error': max_error, 'cpu_ensemble_probability_max_error': max_ensemble_error,
          'cpu_ensemble_classification_changes': 0, 'cross_device_member_tolerance': 1e-3, 'app_replaced': False})
    write(args.results / 'summary.json', {'seeds': aggregate, 'new_seed_aggregate': summary, 'reference_seed42_excluded_from_aggregate': True})
    pct = lambda v: f'{v:.2%}'
    mean = summary['accuracy']['mean']
    lines = ['# 外部甲基化＋外部细胞图像预训练：新增三个种子的结果', '', '2026年11月6日完成。', '',
             f'新增种子43、44、45，总体准确率分别为{pct(aggregate[0]["accuracy"])}、{pct(aggregate[1]["accuracy"])}、{pct(aggregate[2]["accuracy"])}，平均{pct(mean)}。重度召回平均{pct(summary["severe_recall"]["mean"])}，中度召回平均{pct(summary["moderate_recall"]["mean"])}。', '',
             '结论：新方案在不同划分下的总体表现接近，但中度样本仍较容易被判轻。保留为研究候选，当前APP模型保持原样。', '',
             '## 实际训练了什么', '',
             '口腔照片使用本地Zenodo预训练的ResNet-18；细胞视野使用外部脱落细胞数据预训练的独立ResNet-18。10基因甲基化使用GSE204943预训练的小型神经网络，其他表格资料使用普通神经网络。四组特征进入Transformer，输出“正常/轻度”和“中度及以上”两个概率。这套新方案没有使用TabPFN。', '',
             '每个种子都重新按患者组划成六折，兼顾原始三个等级的分布。每次留一折测试；其余资料再分五组，五个完整多模态模型各用四组训练。五个概率各占20%，中度及以上的平均概率≥0.5就判为中度及以上。每个种子训练30个最终模型，三个种子共90个。训练轮数只用各模型训练范围内的资料选择。', '',
             '43、44、45同时改变资料划分、模型初始化、批次顺序和训练时的照片视图；外部预训练的初始权重、模型结构和其他参数相同。本次检查的是整套训练结果对随机变化的稳定性，没有单独拆分“换划分”和“换初始化”的影响。', '',
             '每个种子均安排117份开发资料，112份至少有一组可用输入：正常/轻度52份、中度29份、重度31份。其余5份全部输入缺失，保留为未分析记录，不计入准确率。原22份测试资料继续排除。', '',
             '## 准确率和各类召回率', '',
             '“总体准确率”把六次测试的112份预测合起来计算；“六折平均”是六个准确率的算术平均。中度、重度召回分别指：真实属于该等级的资料中，有多少被二分类模型判为“中度及以上”。', '',
             '| 种子 | 总体准确率 | 六折平均 | 最高折准确率 | 正常/轻度召回 | 中度及以上召回 | 原中度召回 | 原重度召回 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f'| {r["seed"]} | {pct(r["accuracy"])}（{r["correct"]}/{r["n"]}） | {pct(r["fold_mean"])} | {pct(r["highest_accuracy"])}（第{r["highest_fold"]}折，{r["highest_correct"]}/{r["highest_n"]}） | {pct(r["normal_recall"])} | {pct(r["positive_recall"])} | {pct(r["moderate_recall"])} | {pct(r["severe_recall"])} |')
    lines += [f'| 三种子平均 | {pct(mean)} | {pct(summary["fold_mean"]["mean"])} | — | {pct(summary["normal_recall"]["mean"])} | {pct(summary["positive_recall"]["mean"])} | {pct(summary["moderate_recall"]["mean"])} | {pct(summary["severe_recall"]["mean"])} |', '',
              f'总体准确率的种子间标准差为{summary["accuracy"]["std"]*100:.2f}个百分点，范围{pct(summary["accuracy"]["min"])}—{pct(summary["accuracy"]["max"])}。这是同一批资料的三次评价，不把它当作336份独立新资料。', '',
              '## 错误去了哪里', '',
              '| 种子 | 正常/轻度→中度及以上（52份中） | 中度→正常/轻度（29份中） | 重度→正常/轻度（31份中） |',
              '|---|---:|---:|---:|']
    for r in aggregate:
        dest = r['destinations']
        lines.append(f'| {r["seed"]} | {dest[0][1]} | {dest[1][0]} | {dest[2][0]} |')
    lines += ['', '新增三种子中，重度有3—4份被判为正常/轻度；中度有11—14份被判为正常/轻度。这仍是当前需要改进的主要部分。', '',
              '## 与此前结果怎样对应', '',
              '同一新方案此前在固定种子42、APP历史六折名单下，总体准确率75.89%、六折平均75.58%、最高折84.21%，正常/轻度召回76.92%、中度及以上召回75.00%、原中度召回65.52%、原重度召回83.87%。该结果作为参考，不计入上面新增三个种子的平均。', '',
              'APP历史方案“ResNet-18＋TabPFN＋Transformer，五个模型各占20%，Dropout 0.1”的总体准确率为75.89%，最高折89.47%。本次没有重跑APP方案。新增种子的测试名单不同，因此最高94.44%说明存在表现较好的小测试折，不能据此认定新方案整体超过APP。', '',
              '## 每折结果', '', '| 种子 | 第1折 | 第2折 | 第3折 | 第4折 | 第5折 | 第6折 |', '|---|---:|---:|---:|---:|---:|---:|']
    for seed, r in data['results'].items():
        lines.append('| '+seed+' | '+' | '.join(f'{pct(f["metrics"]["accuracy"])}（{int(round(f["metrics"]["accuracy"]*f["metrics"]["eligible_count"]))}/{f["metrics"]["eligible_count"]}）' for f in r['folds'])+' |')
    lines += ['', '<details><summary>运行配置与证据位置</summary>', '',
              '下游配置沿用configs/research/methylation_cell_transfer.json：Dropout 0.1，批量16，融合部分学习率0.001，甲基化学习率0.0001，权重衰减0.01，最多50轮，连续8轮无改善停止；甲基化先固定5轮再微调。外部两个预训练编码器没有按三个种子重新训练，口腔及细胞图像骨干在SYSU训练时固定。', '',
              '源码：scripts/research/prepare_cell_random_protocols.py、run_cell_multiseed.py、run_cell_multiseed_remote.sh、summarize_cell_multiseed.py。运行入口：bash cell-pretraining/run_cell_multiseed_remote.sh（个人GPU目录）。', '',
              '划分与完整名单：data/processed/oral-cytology-20686174/random-protocols/seed43.json、seed44.json、seed45.json。', '',
              '本地权重、逐份和逐成员预测、曲线、输入与源码校验：outputs/experiments/cell-sixfold-ensemble-20261106/multiseed-attempt1。results.json保存原始结果；summary.json保存汇总；independent_verification.json保存90个模型的独立核对。', '',
              f'90个模型均核对了训练/测试患者组隔离、归一化与TCT词表只取训练资料、保存权重SHA、恢复推理及五个概率的平均。GPU保存恢复误差均小于1e-6；本机CPU重算单模型最大概率差{max_error:.3g}，五模型平均后最大差{max_ensemble_error:.3g}，全部最终分类一致。', '',
              '远程结果：/home/zjkj/24334067/OPMD-Pred-Model-Experiments-20261106/cell-pretraining/outputs/multiseed-attempt1。多种子下游运行用时'+f'{data["elapsed_seconds"]:.1f}'+'秒，使用已缓存的两个图片编码器特征，不包含之前的外部预训练与图片特征提取时间。', '',
              '</details>', '']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'checked_models': len(checked), 'cpu_max_error': max_error, 'metrics': summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
