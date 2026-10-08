# 私有验证记录预测报告

分支`codex/sysu-explainability`，基于工程基线e2e090f。`predict_sysu.py`读取真实训练后的foundation格式checkpoint，生成`predictions.json`和纯离线`report.html`。当前尚无已审核SYSU模型，**本轮没有推理、生成病例报告或展示伪造数据**；仅代码准备和静态审阅。

## 输入与资料不足

只接受checkpoint内保存的val记录。`--sample-ids`可指定这些记录的子集，train/test自动选择禁止。CPU先验证CSV/image SHA、患者manifest、有效split和排除理由，恢复保存的词表与五模态配置；年龄性别不进入batch或展示。无选定观测输入的记录保留在报告内，输出abstain，没有概率/动作/归因。

模型预测同时保存三类概率、argmax、保存成本矩阵的动作和各动作期望代价；没有成本矩阵时不添加成本动作。概率标注为未校准模型输出，不是已验证的患病概率或误诊率。成本来源是checkpoint保存快照，原CLI文件路径只作追溯，不假装重新核验了已不存在文件。

## SHAP和InterSHAP

显式`--explain`启用最多32个coalition，编码器一次得到embedding后按实际观测mask移除模态。自然未测模态保持缺失，不能补阴性或编造甲基化。默认目标severe，也可at_least_moderate/expected_severity；保存全缺失基准、完整观测目标、原始Shapley、完整SII及加和残差。

`--intershap`额外计算原文global absolute-mean与local逐记录比值，必须同时启用explain。近零分母标记未定义，记录定义数量。所选验证记录的global统计不是整个数据集的全局统计。解释是missing-token游戏，不是背景积分或医学因果证据。完整SII对角为main effect，非对角为成对交互。

## 本地HTML

HTML使用卡片和锚点导航，显示三类概率条图、自然观测/缺失/模型未选模态、两种判定与成本、模态贡献和完整SII矩阵。没有JavaScript、外部CDN或网络调用，所有文本来源经html.escape，原图base64只接受JPEG/PNG/WebP。默认不含原图及真实标签；只有显式`--embed-images`或`--include-labels`才包含它们，生成物始终是私有医疗研究资料，不公开发布。

不接入网站、医生工作流或自动转诊。缺失不显示为阴性；原始来源不能区分未测/未记录时维持未知。资料不足和InterSHAP未定义有独立显示。该报告尚未经渲染或运行验证，不能把离线HTML代码存在视为产品交付完成。

## 未来命令

审核患者ID/重复图/标签与检查时点并训练选定模型后，使用新的私有output目录，GPU任务先检查资源：

```bash
PYTHONPATH=src python -m opmd_pred.scripts.predict_sysu \
  --csv /personal/data/reviewed_sysu_samples.csv --images /personal/data/SYSU_processed \
  --checkpoint /personal/outputs/sysu_selected/checkpoint.pt \
  --output /personal/outputs/sysu_selected/val_report \
  --explain --intershap --target severe --device cpu
```

此命令仅为未来操作说明，本轮未执行。图像IG为同分支另一个`explain_sysu_image`入口；没有把像素热图直接包装成病灶定位或已验证解释。当前分支只支持base中的softmax/ordinal；CORN及一致性模型的完整工具使用需后续集成分支统一源码，而非把另一分支checkpoint硬加载进不同代码。

总代码静态审閱检查概率有效性、成本方向、自然缺失和资料不足路径、checkpoint/CSV变更检测及HTML输出转义。模型恢复、coalition数值、渲染和效果尚未实测；不会自动将验证预测当最终泛化证据。
