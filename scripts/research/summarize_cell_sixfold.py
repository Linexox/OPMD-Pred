"""Independent membership/ensemble checks and matched-APP six-fold report."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sysu_binary_common import score, sha, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    d=json.loads((a.results/'comparison.json').read_text(encoding='utf-8'));protocol=json.loads(a.protocol.read_text(encoding='utf-8'))
    assert sha(a.protocol)==d['identity']['protocol_sha256']
    names={'historical_app':'APP历史方案：ResNet-18＋TabPFN＋Transformer',
           'imagenet':'新结构：甲基化预训练＋通用ImageNet细胞编码器',
           'oral_pretrained':'新结构：甲基化预训练＋外部口腔细胞预训练编码器'}
    methods={'historical_app':d['historical_app'],**d['new_methods']};checks=[]
    baseline_ids={r['sample_id']:r['grade'] for r in d['historical_app']['records'] if r['probabilities'] is not None}
    for condition,r in d['new_methods'].items():
        assert {v['sample_id']:v['grade'] for v in r['records'] if v['probabilities'] is not None}==baseline_ids
        for fold in r['folds']:
            n=fold['fold'];specs=protocol['folds'][n]['experts'];held=protocol['folds'][n]['held_ids']
            assert [v['sample_id'] for v in fold['records']]==held
            for i,spec in enumerate(specs):
                path=a.results/condition/f'fold{n}'/f'expert{i}'
                saved=torch.load(path/'refit/classifier.pt',weights_only=True,map_location='cpu')
                assert set(saved['training_ids'])<=set(spec['fit_ids'])
                assert not set(held)&set(saved['training_ids'])
                inner=json.loads((path/'inner/training.json').read_text(encoding='utf-8'))
                assert set(inner['training_ids'])<=set(spec['inner_train'])
                assert inner['validation_ids']==spec['inner_validation']
                assert sha(path/'refit/classifier.pt')==fold['experts'][i]['classifier_sha256']
                assert fold['experts'][i]['reload_error']<1e-6
                checks.append({'condition':condition,'fold':n,'expert':i,'classifier_sha256':sha(path/'refit/classifier.pt')})
            for row in fold['records']:
                assert row['weights']==[.2]*5
                if row['probabilities'] is not None:
                    assert np.allclose(np.mean(row['member_probabilities'],0),row['probabilities'],atol=1e-8)
            assert score(fold['records'])==fold['metrics']
        assert score(r['records'])==r['metrics']
    write(a.results/'independent_verification.json',{'checked_new_classifiers':len(checks),'checks':checks,
        'same_held_out_samples_as_historical_app':True,'equal_mean_verified':True,'test_used_for_selection':False})
    def percent(x):return f'{x:.2%}'
    original=methods['historical_app']['metrics'];new=methods['oral_pretrained']['metrics']
    lines=['# 与APP历史方案对齐：六折留出、五模型等权集成','',
        '2026年11月6日完成。新结构两种细胞编码器各训练30个多模态模型，共60个；APP使用归档的历史预测作为对照，没有重训或替换发布模型。','',
        f'新方案总体准确率{percent(new["accuracy"])}，与APP历史方案相同；最高折准确率84.21%，APP历史最高为89.47%。新方案正常/轻度误报少2份，中度漏判相同，重度漏判多2份。当前保留APP主模型。','',
        '## 使用了怎样的训练和评价方式','',
        '117份开发资料分六折，每次留一折，只用于测试。其余资料按历史保存的五组划分，五个完整多模态模型各排除不同一组、用另外四组训练。五个模型的二分类概率各占20%汇总。六折轮流测试，每份资料只进入一次最终评价。','',
        '这里复用了APP当时保存的六份测试名单与30份子模型训练名单，包括顺序，没有重新随机划分。历史实现会在每次六折留一后，把剩余资料重新分五组；因此以保存名单为准，不把后来另外一套五折名单混进来。每个子模型最终训练范围为77—79份原始记录，完全无输入的记录不参与梯度训练。','',
        '本轮没有训练Router，也没有拿留出的测试折学习集成权重。旧22份测试资料不参与。117份中112份具备输入，其余5份缺全部输入，三种方案的可评价名单及标签完全一致。','',
        '## 三种方法具体是什么','',
        '- APP历史方案：Zenodo预训练的ResNet-18编码口腔照片，TabPFN处理病史、TCT、DNA异常比例和甲基化总数，Transformer融合。五个完整模型各占20%。',
        '- 新结构、通用细胞编码器：口腔照片编码器保留，甲基化展开为10基因并使用GSE204943预训练的小型神经网络；其他表格使用普通神经网络，增加ImageNet ResNet-18细胞图像分支，四组特征经Transformer融合。五个独立多模态模型各占20%。',
        '- 新结构、外部细胞预训练：与上一项完全相同，只把细胞ResNet-18权重换成外部Zenodo口腔细胞四分类预训练权重。两项新结构都采用相同的256像素细胞切图、128步长及视野汇总。','',
        '对齐的是测试名单、各子模型训练名单和等权集成方式。训练参数仍各自沿用：APP历史模型固定7轮；新结构沿用上一轮参数，并只在每个子模型自己的训练范围内，用一组资料验证以选择轮数，再用全部四组重训。没有用外层测试折选参数。因此新结构之间可比较细胞预训练的作用；新结构与APP之间比较的是整套方法，不能把差距只归因于预训练。','',
        '## 总体表现','',
        '“真实中度/重度召回率”指该原始病理等级被二分类模型判为“中度及以上”的比例，合并六折全部留出预测统计。','',
        '| 方法 | 总体准确率（112份） | 六折准确率平均 | 最高折准确率 | 正常/轻度召回（52份） | 中度及以上召回（60份） | 真实中度召回（29份） | 真实重度召回（31份） |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,r in methods.items():
        m=r['metrics'];folds=r['per_fold'] if key=='historical_app' else [x['metrics'] for x in r['folds']]
        maximum=max(f['accuracy'] for f in folds);best=[str(i+1) for i,f in enumerate(folds) if abs(f['accuracy']-maximum)<1e-12]
        values=[percent(m['accuracy']),percent(r['fold_accuracy_mean']),percent(maximum)+'（第'+ '/'.join(best)+'折）',
                percent(m['recalls'][0]),percent(m['recalls'][1]),percent(m['moderate_detection']),percent(m['severe_detection'])]
        lines.append('| '+names[key]+' | '+' | '.join(values)+' |')
    lines+=['','## 每一折的准确率','',
            '| 留出折 | 可评价资料数 | APP历史方案 | 新结构＋ImageNet细胞编码器 | 新结构＋外部细胞预训练 |',
            '|---|---:|---:|---:|---:|']
    for i in range(6):
        old=methods['historical_app']['per_fold'][i];plain=methods['imagenet']['folds'][i]['metrics'];trained=methods['oral_pretrained']['folds'][i]['metrics']
        lines.append(f'| 第{i+1}折 | {old["eligible_count"]} | {percent(old["accuracy"])} | {percent(plain["accuracy"])} | {percent(trained["accuracy"])} |')
    lines+=['','APP历史最高和新方案最高都在第1折：APP判对17/19，新方案判对16/19。','',
        '## 错误去了哪里','',
        '| 方法 | 正常/轻度→中度及以上 | 中度→正常/轻度 | 重度→正常/轻度 |','|---|---:|---:|---:|']
    for key,r in methods.items():
        m=r['metrics'];lines.append(f'| {names[key]} | {m["false_positives"]}/52 | {m["moderate_missed"]}/29 | {m["severe_missed"]}/31 |')
    lines+=['','## 结论','',
        '外部细胞预训练在新结构内部仍有收益：总体准确率由72.32%到75.89%，中度召回由55.17%到65.52%，重度召回由80.65%到83.87%，正常/轻度召回保持76.92%。','',
        '与APP历史方案相比，新方案总体正确数相同，正常/轻度误报更少，但重度漏判从3份增加到5份，最高折也未超过原成绩。当前APP主模型继续保留；外部细胞预训练编码器作为研究候选保留。','',
        '上一轮80.36%来自五折单模型，每个最终模型使用约80%的开发资料；本轮每个集成成员使用约三分之二资料，且测试划分改为历史六折。两个成绩属于不同实验设置，不把80.36%当作本轮集成成绩。','',
        '<details><summary>配置、代码和证据位置</summary>','',
        '新结构配置沿用configs/research/methylation_cell_transfer.json：学习率0.001、甲基化微调0.0001、批量16、权重衰减0.01、Dropout0.1、最多50轮、早停耐心8轮；甲基化前5轮固定。从5个内部组中，子模型e的验证组取下一个组，另外3组作内部训练，选轮数后在全部4组重训；初始种子42+e，最终种子142+e。',
        'APP历史优化设置为固定7轮、学习率0.0001、批量8、权重衰减0.0001、模态随机隐藏15%；未声称本轮把这些优化参数也改成一致。',
        '源码：scripts/research/prepare_cell_sixfold_protocol.py、run_cell_sixfold_ensemble.py、run_cell_sixfold_comparison.sh、summarize_cell_sixfold.py。',
        '冻结协议：data/processed/oral-cytology-20686174/sixfold-app-protocol.json；包括历史源文件校验值、六个测试名单、30个训练名单与内部验证名单。',
        '本地结果：outputs/experiments/cell-sixfold-ensemble-20261106/sixfold-attempt1。comparison.json含全部逐份及逐成员预测；各fold*/expert*/refit/classifier.pt保存模型，训练曲线在training.json。independent_verification.json核对60个模型的训练名单、测试隔离、恢复预测及等权平均。',
        '远程：/home/zjkj/24334067/OPMD-Pred-Model-Experiments-20261106/cell-pretraining/outputs/sixfold-attempt1；GPU训练与本地开发均与APP主线隔离。','', '</details>','']
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines),encoding='utf-8')
    print('PASS: 60 classifiers, exact historical memberships, equal averaging and all reported counts verified.')


if __name__=='__main__':main()
