"""Render the completed dataset audit and proposed preprocessing as one HTML report."""
import argparse
import base64
import html
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();s=json.loads((a.audit/'summary.json').read_text(encoding='utf-8'))
    default_cache='D:/.Workspace/医工/开发缓存/matplotlib' if os.name=='nt' else str(a.audit/'matplotlib-cache')
    cache=Path(os.environ.get('MPLCONFIGDIR',default_cache));cache.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['I','II','III','IV'];colors=['#7161b6','#8e80ca','#ae9bdd','#c6b4e8']
    fig,axes=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
    for ax,key,title in zip(axes,['patch_category_counts','nucleus_annotation_counts'],['Image patches (739)','Annotated nuclei (20,021)']):
        counts=[s[key][c] for c in labels];bars=ax.barh(labels,counts,color=colors)
        ax.invert_yaxis();ax.set_title(title);ax.set_xlim(0,max(counts)*1.3)
        for bar,n in zip(bars,counts):ax.text(n+max(counts)*.02,bar.get_y()+.4,f'{n:,} ({n/sum(counts):.1%})',va='center',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(a.audit/'class_distribution.png',dpi=160);plt.close(fig)
    def embed(path,mime):return f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode()
    rows=''.join(f'<tr><td>{c}</td><td>{s["patch_category_counts"][c]:,}</td><td>{s["nucleus_annotation_counts"][c]:,}</td><td>{s["patients_by_patch_category"][c]}</td><td>{s["patch_category_by_stain"][c].get("PAP",0)} / {s["patch_category_by_stain"][c].get("MGG",0)}</td></tr>' for c in labels)
    source=html.escape(s['source_root'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>口腔脱落细胞数据检查与预处理计划</title><style>
body{{font:17px/1.75 'Microsoft YaHei',sans-serif;color:#292733;background:#f7f6fa;margin:0}}main{{max-width:1000px;margin:32px auto;padding:32px;background:white;border-radius:20px}}h1{{font-size:29px;line-height:1.4}}h2{{font-size:22px;margin-top:34px}}p,li{{max-width:900px}}table{{width:100%;border-collapse:collapse;font-size:16px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #e6e3ee}}th{{background:#f5f2fb}}img{{max-width:100%;height:auto}}.lead{{padding:18px;background:#f1ecfb;border-radius:14px}}.muted{{color:#656170}}code{{overflow-wrap:anywhere;font-size:14px}}pre{{white-space:pre-wrap;background:#f6f4f8;padding:14px;border-radius:10px;font-size:14px}}details{{margin-top:24px}}a{{color:#6550a5}}</style><main>
<h1>口腔脱落细胞数据检查与预处理计划</h1><p class="muted">2026年11月6日｜已完成本地文件检查与统计；尚未批量裁图、划分训练集或启动细胞预训练。</p>
<p class="lead"><strong>现有下载包足够开展第一轮细胞图像预训练。</strong>可用内容是116人的739张显微图像块及20,021个细胞核标注。建议先按患者分组，再裁取细胞核周边局部图，用原始I—IV标签预训练ResNet-18，最后迁移到SYSU比较。</p>
<h2>一、实际下载了什么</h2>
<table><tr><th>内容</th><th>实际情况</th></tr>
<tr><td>本地文件总体积</td><td>4,206,037,866字节，约4.21 GB（3.92 GiB）</td></tr>
<tr><td>全部登记资料</td><td>242人、380张切片、10个中心；并非每张切片都共享图像块</td></tr>
<tr><td>本地实际图像</td><td>739张PNG，全部2048×2048像素、RGB；来自116人、156张切片、7个中心</td></tr>
<tr><td>染色</td><td>PAP（巴氏染色）323张图；MGG（May-Grünwald-Giemsa染色）416张图</td></tr>
<tr><td>细胞核标注</td><td>739份局部GeoJSON，共20,021个多边形；记录细胞核位置及I—IV标签</td></tr>
<tr><td>全切片上的位置标注</td><td>156份GeoJSON，标出739个图像块在原切片的位置；这些不是另一批细胞核标注</td></tr>
<tr><td>全切片文件</td><td>本目录没有MRXS全切片文件。另附368份全切片压缩包下载清单，合计约532.90 GB；第一轮无需补下载</td></tr></table>
<p>官方入口另行提供全切片分卷下载，当前4 GB包以图像块、标注和资料表为主。<a href="https://zenodo.org/records/20686174">Zenodo数据集说明</a></p>
<h2>二、类别与样本分布</h2>
<table><tr><th>原始类别</th><th>图像块数</th><th>细胞核数</th><th>涉及人数</th><th>PAP / MGG图像块</th></tr>{rows}</table>
<p class="muted">“涉及人数”按各类分别计数；2人出现多种局部类别，因此四行人数不能相加当总人数。数据集原始I—IV属于细胞学类别，本轮保留原标签，不转换成SYSU病理的轻度、中度、重度。</p>
<img alt="各类图像块及细胞核数量" src="{embed(a.audit/'class_distribution.png','image/png')}">
<p>I类占图像块61.43%、细胞核63.68%，明显多于其余类别。另有中心分布偏斜：739张图中580张（78.48%）来自两个中心。患者数也有限，20,021个核并不等于20,021个独立患者样本。</p>
<h2>三、检查到的问题及处理方式</h2>
<ul><li><strong>13处文件名差异：</strong>资料表名称少了文件中的“_1”或“_2”。已按官方文件清单的切片编号和坐标唯一匹配，原文件不改名，索引保留两种名称。</li>
<li><strong>1张切片的标签层级不同：</strong>切片资料为IV，4张局部图及其中43个核均标为III。细胞预训练使用局部标注III，保留切片IV字段，不自动覆盖或改写。</li>
<li><strong>14个极小核框：</strong>至少一边小于8像素，首轮列入排除清单，不删除原标注。</li>
<li><strong>2个核框略微超出边缘：</strong>分别超出右边界1和7像素。读取实际图像范围，裁图不足的部分补白，保留边缘标记。</li>
<li><strong>1对可能重叠的图像块：</strong>位置来自同张切片。划分前按患者归组，确保重叠区域不会跨训练与测试；裁图时再检查重复对象。</li>
<li><strong>图像完整性：</strong>739张均正常解码；无字节完全相同或RGB像素完全相同的图像，表中核数与GeoJSON核数均一致。解码正常不代表所有图都已通过医学可读性审查。</li></ul>
<h2>四、预处理按这四步做</h2>
<h3>1. 建立统一索引</h3><p>新增<code>scripts/research/prepare_oral_cytology.py</code>，读取本次统计清单和原标注，生成每张图、每个核的记录：患者、切片、中心、染色、局部类别、图像路径、标注序号、核位置及原文件SHA。使用官方清单修正13处名称对应；不使用自由文本报告、年龄或性别作为图像模型输入。</p>
<h3>2. 先按患者划分，再产生局部图</h3><p>固定种子42，按患者约70%训练、15%验证、15%测试。同一人的所有切片、染色、图像块、裁图进入同一组。以该患者已有局部标注中的最高类别作分层依据，并检查每组四类、两种染色及中心分布；每组各类至少有2名患者，否则停止生成并报告实际数量。保存<code>patient_split.json</code>，之后所有预训练对照复用，测试集不参与选参数。</p>
<h3>3. 裁取细胞核周边区域</h3><p>围绕每个有效细胞核的中心取<strong>256×256像素</strong>的RGB局部图，保留核及周围细胞信息，输入模型时再缩放到224×224。边缘补白。14个极小框首轮排除，按当前规则预计保留20,007个局部图候选，最终数量以对象去重和裁图检查结果为准。这里不是完整细胞分割，不宣称局部图一定包含完整细胞。</p>
<p>核的中位宽40像素、高41像素。整张2048图缩到224后，典型核宽只有约4.4像素；256局部图缩到224后仍约35像素。保留原RGB，不在首轮强行进行染色归一化；尚无足够精确的逐图物理像素尺寸，先不做微米尺度统一。</p>
<h3>4. 给预训练提供固定输入</h3><p>新增<code>scripts/research/train_oral_cytology_encoder.py</code>，用ImageNet初始化的独立ResNet-18，预测原始I/II/III/IV四类。训练时先等概率抽类别，再从该类等概率抽患者，最后从患者抽局部图；使用普通交叉熵，不再叠加逆频率类别权重。只在训练组使用水平/垂直翻转、90度旋转及轻度亮度/对比度变化，不用大幅改色或会裁掉目标核的随机裁剪。</p>
<p>验证和测试均使用原始分布、固定处理。报告准确率、各类召回率、混淆矩阵和三组各自患者数量，并分别汇报PAP、MGG表现。保留验证集选出的权重；测试集只评价锁定方案。</p>
<p>迁移到SYSU时丢弃四分类层，保留图像编码器。SYSU视野没有这套细胞核标注，因此采用256像素窗口、步长128覆盖视野，边缘补白，汇总多个局部图特征。为公平比较，通用ImageNet权重和新预训练权重必须使用相同的SYSU切图、汇总方法与留出划分；细胞学四类不直接套为SYSU病理类别。</p>
<h2>五、产物与验收</h2><p>原始数据保留在<code>{source}</code>。新索引、患者划分、局部图与排除清单统一放本实验工作树<code>data/processed/oral-cytology-20686174/attempt1/</code>；权重、曲线和指标放<code>outputs/experiments/oral-cytology-20686174/attempt1/</code>。不复制全切片，不修改APP。</p>
<pre>python scripts/research/prepare_oral_cytology.py --root "D:/.Workspace/医工/DATASETS/oral-cytology-zenodo-20686174/oral-cytology-dataset" --audit outputs/checks/oral-cytology-20686174-20261106/resolved --output data/processed/oral-cytology-20686174/attempt1 --seed 42</pre>
<p class="muted">上面是下一阶段计划实现的命令，准备脚本和训练脚本尚未创建。本轮已经实现并执行的是audit_oral_cytology.py（统计检查）和summarize_oral_cytology.py（本报告）。</p>
<p>预处理完成时核对：患者不跨组；每张裁图都能追溯到原PNG和标注；裁图没有画入类别文字或标注轮廓；标签与原局部标注一致；14个小框及边界处理有明细；抽查四类、两种染色的局部图；确认输出数量与最终清单一致。</p>
<details><summary>四类、两种染色的图像示例及检查记录</summary><p>以下为原图缩略展示。类别文字只在报告中，不写入训练图像。</p><img alt="各类别PAP和MGG染色示例" src="{embed(a.audit/'category_stain_examples.jpg','image/jpeg')}"><p>最终统计：<code>outputs/checks/oral-cytology-20686174-20261106/resolved/summary.json</code>；图像清单为patch_inventory.csv，核位置清单为nucleus_inventory.csv。上级目录保存首次直接按名称匹配时的检查结果；resolved使用官方清单核实了13处名称差异，应以resolved为准。核位置清单约2万行仅是审计产物，还没有批量生成局部图。</p></details>
</main></html>'''
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(page,encoding='utf-8')
    print(f'Report saved: {a.report}')


if __name__=='__main__':main()
