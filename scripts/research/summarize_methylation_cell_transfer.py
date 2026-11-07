"""Independently check predictions and render the first transfer experiment report."""
import argparse
import json
from pathlib import Path

import torch
from sysu_binary_common import sha, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args(); d=json.loads((a.results/'comparison.json').read_text());pre=json.loads((a.results/'pretraining.json').read_text())
    names={'no_cells_random':'口腔照片＋10基因及其他表格资料；甲基化网络直接在SYSU训练',
           'no_cells_pretrained':'口腔照片＋10基因及其他表格资料；甲基化网络先在GSE204943预训练',
           'cells_random':'增加SYSU细胞图像；甲基化网络直接在SYSU训练',
           'cells_pretrained':'增加SYSU细胞图像；甲基化网络先在GSE204943预训练'}
    verified=[];common_ids=None
    enc=torch.load(a.results/'methylation_encoder.pt',map_location='cpu',weights_only=True)['encoder']
    for key,result in d['results'].items():
        rows=result['rows'];assert len(rows)==117 and len({r['sample_id'] for r in rows})==117
        usable=[r for r in rows if r['probabilities'] is not None];ids={r['sample_id'] for r in usable}
        if common_ids is None:common_ids=ids
        assert common_ids==ids
        dest=[[0,0] for _ in range(3)]
        for r in usable:
            assert abs(sum(r['probabilities'])-1)<1e-5
            dest[r['grade']][int(r['probabilities'][1]>=.5)]+=1
        assert dest==result['metrics']['original_grade_destinations']
        correct=dest[0][0]+dest[1][1]+dest[2][1]
        assert abs(correct/len(usable)-result['metrics']['accuracy'])<1e-12
        for fold in result['folds']:
            folder=a.results/key/f'fold{fold["fold"]}'
            checkpoint=torch.load(folder/'refit/classifier.pt',map_location='cpu',weights_only=True)
            held={r['sample_id'] for r in rows if r['fold']==fold['fold']}
            assert not held.intersection(checkpoint['training_ids'])
            assert fold['selected_epoch']>5 and fold['reload_error']<1e-6
            change=None
            if key.endswith('_pretrained'):
                change=max(float((checkpoint['state']['methyl.'+k]-v).abs().max()) for k,v in enc.items())
                assert change>0, 'Methylation encoder was not fine-tuned'
            verified.append({'method':key,'fold':fold['fold'],'checkpoint_sha256':sha(folder/'refit/classifier.pt'),
                             'methylation_max_change_after_transfer':change})
    write(a.results/'independent_verification.json',{'checkpoints':verified,'common_evaluable_records':len(common_ids),
          'checks':['manual prediction-count recomputation','same evaluable IDs in all four conditions',
                    'held-out IDs absent from checkpoint training IDs','strict reload probability equality recorded',
                    'methylation weights actually changed during SYSU fine-tuning']})
    lines=['# 甲基化预训练与细胞图像：首轮训练结果','',
        '2026年11月6日。第一轮固定种子42、原五折划分，已完成4种组合、20次外层训练。', '',
        '两项同时加入后，中度及以上召回率从70.00%升到75.00%，但正常/轻度召回率从73.08%降到69.23%。112份可评价资料中多判对1份。当前结果支持继续复核，尚不足以替换APP模型。','',
        '## 实际训练了什么','',
        '所有组合使用同一个新结构：口腔照片特征、10基因甲基化特征、其他表格特征，以及可选的细胞图像特征，分别映射为32维向量，再由2层、4个注意力头的Transformer融合，输出两个类别的概率。中度及以上概率≥0.5归入中度及以上，否则归入正常/轻度。','',
        '- 口腔照片：复用在本地Zenodo照片上预训练的ResNet-18及原有5视图特征缓存。训练随机用4个增强视图之一，验证使用固定视图。',
        '- 甲基化：10个基因的0/1值＋10个是否缺失的标记，经过20→32→32的全连接网络。预训练任务为GSE204943的正常、白斑、口腔鳞癌三分类；迁移时丢弃外部分类层。',
        '- 其他表格：9项吸烟/饮酒/槟榔信息、DNA报告异常比例、TCT意见。连续字段只用当前训练组计算均值和标准差；TCT词表也只从当前训练组建立。甲基化总数、年龄和性别不加入。',
        '- 细胞图像：独立的ImageNet ResNet-18，保留整张视野、补白成正方形并缩放到224像素。每份样本的多张图提取特征后取平均。口腔照片与细胞图像使用不同权重。',
        '- 本轮两个图片骨干均固定；训练特征投影、表格网络、Transformer和分类层。甲基化网络前5轮固定，从第6轮开始用较小学习率微调。此处的微调不包含图片骨干。','',
        '## 数据与比较方式','',
        'GSE204943使用118份外部资料。SYSU使用既有117份开发记录，其中112份有可用模型输入：正常/轻度52份、中度29份、重度31份。另5份无可用输入（正常/轻度3份、中度2份）单独记录，没有计入准确率。四组可评价样本完全一致。','',
        '96份SYSU记录匹配到188张细胞视野图。通过原DNA报告的SHA对应样本文件夹；5份对应多个文件夹的记录不使用细胞图像。跨文件夹重复图像、DI排列拼图与单核小图均排除，其余资料照常参加训练。这里只使用可明确关联到原开发记录的细胞图像。','',
        '每次留出一折评价，其余资料内部再划分训练和验证，以验证结果选训练轮数，随后在该折全部训练资料上重训。旧22份测试资料未使用。这里每份留出资料由一个未训练过它的模型预测，不是APP五模型平均的评价，也不能直接与APP另一套六折结果比较。','',
        '## 结果','',
        '表中“真实中度召回”“真实重度召回”分别指原标签为中度或重度的资料，被二分类模型判为“中度及以上”的比例。召回率合并五折全部留出预测计算，不是只看最高折。最高折准确率是五折中最大的单折结果，展示折序从第1折开始。','',
        '| 方法 | 合并112份准确率 | 五折准确率平均 | 最高折准确率 | 正常/轻度召回（52份） | 中度及以上召回（60份） | 真实中度召回（29份） | 真实重度召回（31份） |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,r in d['results'].items():
        m=r['metrics'];highest=max(f['metrics']['accuracy'] for f in r['folds'])
        best_folds=[str(f['fold']+1) for f in r['folds'] if abs(f['metrics']['accuracy']-highest)<1e-12]
        values=[f'{m["accuracy"]:.2%}',f'{r["fold_mean_accuracy"]:.2%}',
                f'{highest:.2%}（第'+ '/'.join(best_folds)+'折）']
        values += [f'{v:.2%}' for v in [*m['recalls'],m['moderate_detection'],m['severe_detection']]]
        lines.append('| '+names[k]+' | '+' | '.join(values)+' |')
    lines+=['','| 方法 | 正常/轻度→中度及以上 | 中度→正常/轻度 | 重度→正常/轻度 |','|---|---:|---:|---:|']
    for k,r in d['results'].items():
        m=r['metrics'];lines.append(f'| {names[k]} | {m["false_positives"]}/52 | {m["moderate_missed"]}/29 | {m["severe_missed"]}/31 |')
    lines+=['','## 这轮能得出什么','',
        '只加入甲基化预训练时，准确率下降，中度漏判减少、重度漏判增加。只加入细胞图像时，正常/轻度误报减少，但中重度漏判增加。两项同时加入时，中度漏判由13份降到11份、重度由5份降到4份，代价是正常/轻度误报从14份增加到16份。','',
        '这些是同一新结构、相同资料与划分下的结果；只能说明这次配置存在取舍。下一步优先在种子43、44复核两项同时加入的变化，再决定是否调整网络或进行图片骨干微调。转换规则补齐后核对外部数据口径。','',
        '## 外部甲基化训练','',
        '118份外部资料按类别划为93份训练、25份验证。验证集用于选择轮数：第21轮，正确率84%（21/25），正常召回100%（5/5）、白斑20%（1/5）、口腔鳞癌100%（15/15）；三类召回率平均为73.33%。白斑错误去向为正常1份、鳞癌3份。最终以21轮在全部118份上重新训练甲基化编码器，再迁移到SYSU。','',
        '补充的转换记录已核对：使用EPIC B4注释，将TSS1500、TSS200、1stExon、5′UTR中的位点按基因汇总，187个位点生成10个基因平均β值；每个基因以22份正常对照平均β值的第95百分位为阈值，严格大于阈值记为1，否则为0。不是统一0.5。截图中10个基因的白斑、鳞癌阳性数量全部与实际训练CSV一致。尚未取得具体探针清单和连续平均值逐格复算，核对范围为规则、位点数量求和及最终阳性计数。','',
        '这里的1表示高于外部正常对照基线，不直接等于医院报告采用相同区域及标准判出的阳性。全部22份正常资料参与阈值计算，其中5份后来进入外部验证集；外部验证集还用于选择训练轮数。因此84%是存在预处理信息进入验证集的探索性结果，不作为严格独立测试成绩。修正外部验证时，应先划分样本，只用训练组正常资料计算阈值。该问题不代表SYSU留出资料被用于外部阈值计算。现有CSV和训练权重均未修改。','',
        '<details><summary>运行配置与文件位置</summary>','',
        '预训练AdamW学习率0.001、最多200轮、耐心20轮；SYSU最多50轮、耐心8轮，学习率0.001，甲基化微调学习率0.0001。批量16，权重衰减0.01，Transformer Dropout=0.1，按各训练组类别数量加权交叉熵。最优轮数按内部验证平衡准确率选择，相同则比较损失；只从第6轮后选择。','',
        '源码：scripts/research/prepare_methylation_cell_transfer.py、run_methylation_cell_transfer.py、summarize_methylation_cell_transfer.py。配置：configs/research/methylation_cell_transfer.json。',
        '本地结果：outputs/experiments/methylation-cell-transfer-20261106/attempt2/。其中comparison.json含全部指标与留出预测，各方法/fold*/refit/classifier.pt保存模型，training.json保存曲线。independent_verification.json为独立计数、训练名单、权重微调与保存恢复检查。',
        '远程训练位置：/home/zjkj/24334067/OPMD-Pred-Model-Experiments-20261106；主机ubuntu-System-Product-Name，RTX 4090。代码及数据位于用户个人目录，既有APP服务未修改。',
        '首次启动发生图片尚未传完导致读取失败；未产生SYSU结果。修复为全部文件校验后启动，成功结果单独存入attempt2，失败日志保留为failed-attempt1.log。该编号表示执行重试，不是另一套方法。','',
        '</details>','']
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines),encoding='utf-8')
    print(f'Verified {len(verified)} checkpoints; report: {a.report}')


if __name__=='__main__':main()
