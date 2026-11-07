"""Verify completed cell pretraining and transfer, then write one Chinese report."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sysu_binary_common import sha, write


def pct(value):return f'{value:.2%}' if value is not None else '该子集无此类'


def main():
    p=argparse.ArgumentParser();p.add_argument('--pretrain',type=Path,required=True);p.add_argument('--sysu',type=Path,required=True)
    p.add_argument('--data-summary',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();external=json.loads((a.pretrain/'results.json').read_text());down=json.loads((a.sysu/'comparison.json').read_text())
    data=json.loads(a.data_summary.read_text());test_rows=json.loads((a.pretrain/'test_predictions.json').read_text())
    labels=['I','II','III','IV'];cm=np.zeros((4,4),dtype=int)
    groups=external['identity']['patients'];assert not set(groups['train'])&set(groups['test'])
    for row in test_rows:
        assert row['patient_id'] in groups['test'];cm[row['target'],int(np.argmax(row['probabilities']))]+=1
    assert cm.tolist()==external['test']['confusion']
    checks=[];reference_ids=None
    for condition,r in down.items():
        usable=[x for x in r['rows'] if x['probabilities'] is not None]
        ids={x['sample_id'] for x in usable}
        if reference_ids is None:reference_ids=ids
        assert ids==reference_ids and len(r['rows'])==117
        destinations=np.zeros((3,2),dtype=int)
        for row in usable:destinations[row['grade'],int(row['probabilities'][1]>=.5)]+=1
        assert destinations.tolist()==r['metrics']['original_grade_destinations']
        for fold in r['folds']:
            path=a.sysu/condition/f'fold{fold["fold"]}'/'refit/classifier.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            held={x['sample_id'] for x in r['rows'] if x['fold']==fold['fold']}
            assert not held&set(checkpoint['training_ids'])
            assert fold['reload_error']<1e-6
            checks.append({'condition':condition,'fold':fold['fold'],'checkpoint_sha256':sha(path)})
    write(a.sysu/'independent_verification.json',{'checked_models':checks,'test_confusion_recomputed':True,'same_sysu_evaluable_ids':True})
    baseline=down['imagenet']['metrics'];candidate=down['oral_pretrained']['metrics']
    wins=sum(n['metrics']['accuracy']>o['metrics']['accuracy'] for o,n in zip(down['imagenet']['folds'],down['oral_pretrained']['folds']))
    losses=sum(n['metrics']['accuracy']<o['metrics']['accuracy'] for o,n in zip(down['imagenet']['folds'],down['oral_pretrained']['folds']))
    lines=['# 外部口腔细胞预训练与SYSU多模态训练结果','',
        '日期：2026年11月6日。实验分支codex/model-experiments。外部细胞预训练、两组SYSU五折训练已完成，APP当前模型未更换。','',
        f'相同SYSU资料、五折划分及训练设置下，加入外部细胞预训练后，总体准确率由{pct(baseline["accuracy"])}升至{pct(candidate["accuracy"])}；正常/轻度、中度及重度样本召回均提高。五折中{wins}折提高、{losses}折下降。这是固定种子42的首轮对照结果。','',
        '## 这次实际做了什么','',
        '先用Zenodo口腔脱落细胞数据的原始I—IV标签训练一个独立ResNet-18，再把其图像特征接入SYSU多模态模型。下游比较通用ImageNet细胞编码器和外部口腔细胞预训练编码器，其他条件相同。','',
        '两组下游都保留：在本地Zenodo口腔照片上预训练的ResNet-18、GSE204943预训练的10基因甲基化网络、TCT意见、DNA异常比例、吸烟/饮酒/槟榔信息，以及细胞图像。由32维特征投影和Transformer融合，输出正常/轻度与中度及以上两个概率，后者≥0.5判为中度及以上。','',
        '外部细胞预训练时训练图片骨干；SYSU阶段固定两种图片骨干，训练甲基化、其他表格、特征投影、Transformer和二分类层。甲基化沿用上轮同一外部权重，在SYSU前5轮固定、之后微调。','',
        '## 外部细胞数据与预训练结果','',
        f'739张原图来自116人。14个异常小核框排除，28个绝对位置和类别相同的核框去重，得到{data["prepared_crops"]:,}张256×256局部图。28处去重依据是相同位置的裁图，不要求原多边形的每个顶点完全相等。原PNG、GeoJSON均未改动。局部图存为JPEG quality95、无色度降采样，放入带字节索引的crops.tar以避免传输近2万个零碎文件。','',
        '| 资料划分 | 患者数 | 局部图数 | I / II / III / IV |','|---|---:|---:|---|']
    for split,name in [('train','训练'),('validation','验证'),('test','测试')]:
        r=data['splits'][split];lines.append(f'| {name} | {r["patients"]} | {r["crops"]:,} | '+ ' / '.join(str(r['class_counts'].get(c,0)) for c in labels)+' |')
    lines+=['','患者先分组，同一人的切片、染色、局部图均在同一组。每轮从训练池有放回抽4096张局部图：先均匀抽类别，再均匀抽患者及患者内局部图，采用普通交叉熵；验证、测试使用原始分布。前2轮只训练新分类层，之后训练整个ResNet-18。', '',
            f'最多30轮、早停耐心6轮。本次最佳为第{external["best_epoch"]}轮，实际训练到第{external["last_epoch"]}轮；选择依据是验证集四类召回率平均值，测试集不参与选模。','',
            '| 指标 | 验证集 | 测试集 |','|---|---:|---:|',
            f'| 准确率 | {pct(external["validation"]["accuracy"])} | {pct(external["test"]["accuracy"])} |',
            f'| 四类召回率平均 | {pct(external["validation"]["balanced_accuracy"])} | {pct(external["test"]["balanced_accuracy"])} |']
    for i,label in enumerate(labels):lines.append(f'| {label}类召回率 | {pct(external["validation"]["recalls"][i])} | {pct(external["test"]["recalls"][i])} |')
    lines+=['','测试集混淆矩阵：行是真实类别，列是预测类别。','', '| 真实类别 → 预测类别 | I | II | III | IV |','|---|---:|---:|---:|---:|']
    for label,row in zip(labels,cm):lines.append('| '+label+' | '+' | '.join(str(int(v)) for v in row)+' |')
    lines+=['','| 测试染色子集 | 局部图数 | 准确率 | I召回 | II召回 | III召回 | IV召回 |','|---|---:|---:|---:|---:|---:|---:|']
    for stain,r in external['test_by_stain'].items():lines.append(f'| {stain} | {r["count"]} | {pct(r["accuracy"])} | '+' | '.join(pct(v) for v in r['recalls'])+' |')
    lines+=['','I—IV为原始细胞学标签，未直接改成SYSU病理的轻、中、重度；公开资料缺少逐类完整判读标准，因此报告保留原编号。局部图相关性较高，以上指标分母为局部图，不把它写成患者分级准确率。','',
            '## SYSU：细胞预训练是否改善多模态分级','',
            '沿用上轮117份开发资料的原五折名单及种子42。两组都用256像素窗口、128步长覆盖SYSU细胞视野，边缘补白，先平均单视野窗口特征，再平均同一记录的视野特征。为公平比较，两组的切图和汇总方法完全相同。其他超参数与上一轮新结构一致。','',
            '每个外层训练组内部选训练轮数，再用该组全部训练资料重训，预测未见过的那一折。没有使用旧22份测试资料。这里是五折单模型评价，没有重新评价APP的六折五模型集成。','',
            '“真实中度/重度召回”指这些原始标签的资料被二分类模型判为中度及以上的比例；五折合并统计。','',
            '| 细胞图像编码器 | 总体准确率 | 五折准确率平均 | 最高折准确率 | 正常/轻度召回 | 中度及以上召回 | 真实中度召回 | 真实重度召回 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    names={'imagenet':'通用ImageNet ResNet-18；未做外部细胞预训练','oral_pretrained':'ImageNet初始化后，在外部口腔细胞上预训练的ResNet-18'}
    for key,r in down.items():
        m=r['metrics'];highest=max(f['metrics']['accuracy'] for f in r['folds']);best=[str(f['fold']+1) for f in r['folds'] if abs(f['metrics']['accuracy']-highest)<1e-12]
        values=[pct(m['accuracy']),pct(r['fold_mean_accuracy']),pct(highest)+'（第'+ '/'.join(best)+'折）',*map(pct,m['recalls']),pct(m['moderate_detection']),pct(m['severe_detection'])]
        lines.append('| '+names[key]+' | '+' | '.join(values)+' |')
    lines+=['','| 细胞图像编码器 | 可评价资料数 | 正常/轻度误判为中度及以上 | 中度漏判为正常/轻度 | 重度漏判为正常/轻度 |','|---|---:|---:|---:|---:|']
    for key,r in down.items():
        m=r['metrics'];dest=m['original_grade_destinations'];lines.append(f'| {names[key]} | {m["eligible_count"]}/117 | {m["false_positives"]}/{sum(dest[0])} | {m["moderate_missed"]}/{sum(dest[1])} | {m["severe_missed"]}/{sum(dest[2])} |')
    before=down['imagenet']['metrics'];after=down['oral_pretrained']['metrics']
    lines+=['','## 结论','',
            f'本轮只有细胞编码器的预训练来源不同。外部预训练后，SYSU总体准确率变化{(after["accuracy"]-before["accuracy"])*100:+.2f}个百分点，真实中度召回变化{(after["moderate_detection"]-before["moderate_detection"])*100:+.2f}个百分点，真实重度召回变化{(after["severe_detection"]-before["severe_detection"])*100:+.2f}个百分点。',
            '本轮为固定种子42的一次比较；是否继续保留该编码器，要结合这些错误去向及后续重复结果判断。不能仅因外部细胞分类准确率高，就认为SYSU病理分级也提高。APP模型没有自动替换。','',
            '<details><summary>代码、配置、复现位置与核对</summary>','',
            '本地源码均在D:/.Workspace/医工/worktrees/OPMD-Pred-Model-Experiments，分支codex/model-experiments。预处理：scripts/research/prepare_oral_cytology.py；预训练：train_oral_cytology_encoder.py；SYSU对照：run_sysu_cell_encoder_transfer.py；运行：run_oral_cytology_transfer_remote.sh。',
            '外部配置：configs/research/oral_cytology_pretraining.json。批量64，每轮4096张；骨干学习率1e-4，新分类层1e-3，AdamW权重衰减1e-4，混合精度bfloat16。SYSU配置：configs/research/methylation_cell_transfer.json；该阶段FP32。',
            '远程实际运行目录：/home/zjkj/24334067/OPMD-Pred-Model-Experiments-20261106/cell-pretraining；在个人GPU服务器中隔离运行，APP服务和旧权重未修改。',
            '本地输入：data/processed/oral-cytology-20686174/attempt1。结果：outputs/experiments/oral-cytology-transfer-20261106/。外部best.pt保存完整四分类模型及训练患者名单，SYSU各fold/refit/classifier.pt保存融合网络。',
            '独立核对已重新计算外部测试混淆矩阵和SYSU错误计数，核对各折训练与留出名单无交集，保存恢复的概率误差小于1e-6。',
            'GSE204943甲基化网络沿用此前已训练权重；正常对照第95百分位的转换规则及外部验证预处理限制仍按此前记录保留，本轮不重新估计甲基化阈值。','', '</details>','']
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines),encoding='utf-8')
    print('Verified cell held-out test and ten SYSU models; report saved.')


if __name__=='__main__':main()
