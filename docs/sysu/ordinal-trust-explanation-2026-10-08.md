# SYSU 有序三分类、可信预测与解释：证据和实施方案

日期：2026-10-08。范围：研究设计与原文/作者代码核验；本文件没有 SYSU 训练结果，没有上传私有数据。

用户已确认保留吸烟/饮酒/槟榔史与 TCT；删除年龄、性别及其缺失 mask、token 和解释项目。候选模态为完整口腔图片、病史、TCT、DNA 倍体、甲基化。用户已确认类别为 `0=正常/轻度，1=中度，2=重度`；这不意味着三类之间的医学损失等距。检测时点与定量代价仍需核验；“宁可判重”是本项目的用户目标，不能写成普遍医学伦理要求。

**训练前置条件：**协调者当前数据审计报告 139 条记录没有真实 patient ID，存在 2 组同图跨类别，其中 1 组跨现有 train/val。该信息来自协调者的现态审计，本调研代理未独立读取患者记录；需用户核对身份/复诊关联、同图冲突和标签后再固定患者划分。没有真实 patient ID 就不能把目前索引分组结果声明为患者独立，也不能据此承诺 conformal 的患者交换性。

## 1. 建议优先落实的路线

先建立相同患者划分的 softmax 基线，再比较**有序概率头 + 校准 + 不对称期望代价决策**。有序模型负责学习严重程度分布；决策层负责把“宁可判重、跨两级更不可接受”表达为明确代价。二者分别消融，避免把召回上升误归因于更好的特征。

准确率 ≥75% 是实际患者独立评估目标，不能由全预测重度或只报告转人工后的样本达到。必须同时报告所有患者的三分类准确率、重度召回/精确率、低估率、跨两级错误率、预测重度比例、转人工比例。现阶段不声称任何方法已经满足 SYSU 目标。

## 2. 证据矩阵与可迁移边界

| 方法与原始来源 | 已有证据 | SYSU 可实现内容 | 边界与优先级 |
|---|---|---|---|
| Cao, Mirjalili & Raschka, CORAL，Pattern Recognition Letters 2020：[论文](https://arxiv.org/abs/1901.07884)、[作者代码](https://github.com/Raschka-research-group/coral-cnn) | 累计二分类、共享输出权重，解决 rank inconsistency；原实验主要为年龄估计 | 两个累计事件概率；小参数头 | 有序一致性不是重度漏判代价保证；优先对照 |
| Shi, Cao & Raschka, CORN：[论文](https://arxiv.org/abs/2111.08851)、[作者库](https://github.com/Raschka-research-group/coral-pytorch) | 用条件任务与概率链式法则保证 rank consistency，取消 CORAL 输出权重共享限制 | 两个条件二分类；对中/重子集训练第二任务 | 第二任务样本更少；需核验源码，不能直接复制 main；优先候选 |
| Hou, Yu & Samaras, squared EMD：[论文](https://arxiv.org/abs/1611.05916) | 分类概率损失引入类别距离 | softmax + 累计分布距离正则，作为 CORAL/CORN 之外的对照 | 顺序/距离代价仍需定义；对称距离不自动提高重度召回 |
| Guo et al., ICML 2017：[论文](https://proceedings.mlr.press/v70/guo17a.html)、[作者温度校准代码](https://github.com/gpleiss/temperature_scaling) | 温度缩放为低参数后处理校准方法 | 内部校准集拟合一个正温度；报告 NLL/Brier/可靠性图 | 不能保证每个患者或每个缺失模式的可靠性；优先基础模块 |
| Angelopoulos et al., Conformal Risk Control：[论文](https://arxiv.org/abs/2208.02814)、[作者代码](https://github.com/aangelopoulos/conformal-risk) | 对有界、随参数单调的损失控制期望风险 | 单调低估损失与集合输出的研究对照 | 通用不对称代价不是单调损失；总体风险不是重度条件漏判率 |
| Zaffran et al., ICML 2023：[论文](https://proceedings.mlr.press/v202/zaffran23a.html) | 缺失协变量下总体覆盖与缺失模式条件覆盖有差异；研究回归区间 | 支持按实际缺失模式审计，而非只看平均覆盖 | 回归理论/算法不可直接当三分类医学保证 |
| Aaron Ajit, MLMI 2026：[官方论文](https://papers.miccai.org/miccai-2026-sat/paper/MLMI_024.pdf)、[作者代码](https://github.com/aaronjji/Modality-Conditional-Conformal-Prediction-for-Multimodal-Medical-Diagnosis) | MC² 按模态可用模式做 Mondrian 校准；实验为影像+表格二分类 | 已有缺失 mask 可用作校准分组 | 稀疏分组、自然信息性缺失、跨院迁移限制明显；不是首次提出缺失模式校准 |

这里的推荐是对 SYSU 的适配假设，没有把其他任务的数值收益外推到 SYSU，也没有把简单组合声称为首创。

## 3. 三分类概率：必须先保证合法

令累计概率 `q1=P(Y≥1|x,m)`、`q2=P(Y≥2|x,m)`。合法条件为 `1≥q1≥q2≥0`，因此

\[
p_0=1-q_1,\qquad p_1=q_1-q_2,\qquad p_2=q_2.
\]

标签累计编码为 `0→[0,0]，1→[1,0]，2→[1,1]`。直接用两个互不约束的 sigmoid 做普通 BCE，可能出现 `q2>q1`，产生负的中度概率；事后分别 clip 并归一化不能补救训练目标错误。

**CORAL 风格头的工程选择（明确属于适配）：**共享严重度标量 `s=wᵀh`，用有序阈值 `θ2=θ1+softplus(δ)`，令 `qk=sigmoid(s-θk)`。每次 forward 都满足单调性。作者原始 `CoralLayer` 为共享权重 + 独立 bias，默认按降序初始化；论文一致性结论与训练条件有关，不能把“有序初始化”当任意训练状态都不会交叉。我们的显式有序阈值参数化应记录为 CORAL 风格变体，并与原始参考配置区分。[作者 layers.py](https://raw.githubusercontent.com/Raschka-research-group/coral-pytorch/main/coral_pytorch/layers.py)

**CORN：**`r1=P(Y≥1|x,m)`，`r2=P(Y≥2|Y≥1,x,m)`；令 `q1=r1，q2=r1*r2`。第一任务用全部患者；第二任务只监督真实 `y≥1` 的患者，以 `1[y=2]` 为目标。第二任务不是用第一任务的预测结果挑训练样本。条件子集为空时跳过该项，避免 NaN。[CORN 原论文](https://arxiv.org/abs/2111.08851)

对应条件损失可写成

\[
L=\frac{\sum_i\mathrm{BCE}(z_{i1},1[y_i\ge1])+\sum_{i:y_i\ge1}\mathrm{BCE}(z_{i2},1[y_i=2])}{N+N_{y\ge1}}.
\]

源码核验发现：2026-10-08 读取的作者 `coral-pytorch/main/coral_pytorch/losses.py` 的 `corn_loss` 已选出一维 `pred=logits[train_examples,task_index]`，但随后对二维全批 `logits` 调 `F.logsigmoid`，与条件子集 `train_labels` 不匹配。该结论来自静态形状核对，未声称对作者所有发行版复现了问题；实施时对 `pred` 调稳定的 BCEWithLogitsLoss，固定代码版本再核验条件任务形状。Git 远程 SHA 查询本次连接失败，不能把当前 main 当永久可复现版本。[读取的原始源码](https://raw.githubusercontent.com/Raschka-research-group/coral-pytorch/main/coral_pytorch/losses.py)

## 4. “宁可判重”用明确决策表达

以下矩阵仅为研究示例，**导师未确认，不是医学标准**。行是真实 `y`，列是判定 `a`：

\[
C[y,a]=\begin{pmatrix}0&1&4\\2&0&1\\8&3&0\end{pmatrix}.
\]

它表达：同等级零代价、低估比相应过估代价大、跨两级比跨一级更贵。给定校准概率，期望代价与决策为

\[
R(a|x,m)=\sum_{y=0}^{2}p_y(x,m)C[y,a],\qquad a^*=\arg\min_a R(a|x,m).
\]

实现为 `risk = probabilities @ cost_matrix`，再对 action 维求 argmin。若矩阵行列颠倒，会把低估/过估处罚反过来。例 `p=[0.50,0.35,0.15]`，三个风险为 `[1.90,0.95,2.35]`，代价决策判中度，而最大概率判轻度。二者都保存，以展示安全偏好带来的变化。

这个推导来自条件期望最小化：在概率估计准确且 C 合理时，它是给定 C 的 Bayes action；并不保证 accuracy≥75%，也不自动满足某个重度召回下限。

建议先对同一概率模型比较 `argmax` 和固定 C 决策，不重训；再研究损失改动。候选训练正则：

\[
L=L_{\rm probability}+\lambda\sum_a p_a C[y,a].
\]

以及累计距离正则 `λ Σ_{k=0}^1 (F_p(k)-1[y≤k])²`。后者体现有序距离，仍是对称目标。前者和类别加权都会改变概率学习的性质；单独使用线性期望代价训练容易倾向硬决策，不能替代 proper probability loss。须保留 `λ=0` 对照并重新评估校准。

对于加权 CE，理想总体最优解满足 `p_weighted(y|x) ∝ w_y p_true(y|x)`（由加权对数损失的一阶条件推导）。因此重度权重、重采样、asymmetric loss 的输出不能未经校准当自然类别概率，再叠加成本而不检查偏重程度。标量温度通常不能消除所有类依赖偏移；不要把“校准过”当后验必然正确。

## 5. 小样本校准与风险控制

### 5.1 校准协议

固定患者独立测试；开发集内患者分组交叉验证选择模型、损失与预处理。标准化、插补、特征选择、DNA/甲基化变换仅在每个训练 fold 拟合。用未参与模型拟合的预测做校准。正式 split conformal 还需要**与训练/模型选择分离的 calibration 患者**；普通 OOF 汇总后随意 refit 全开发集模型，不自动拥有标准 split-conformal 保证。

OOF 可以节省开发阶段样本，但其用途是选择方案/估计校准收益。若采用 cross-conformal 或 CV+ 变体，需要遵循相应算法与证明，不能把经验交叉拟合结果写成精确有限样本保证。小样本优先低参数校准，记录是否温度拟合、数据量与 fold。测试集不选择温度、C、阈值、转人工规则、模态、损失或展示案例。

softmax 用 `softmax(z/T), T>0`。有序阈值头可共同除以正 T 保持顺序；CORN 对条件 logits 缩放后仍经乘积生成 `q2`。统一以三分类 NLL 拟合低参数校准，验证概率合法；报告 NLL、Brier、可靠性图和带计数的 ECE。ECE 小样本依赖分箱，不能单独作为可信指标。[温度校准原文](https://proceedings.mlr.press/v70/guo17a.html)

### 5.2 缺失模式 conformal

对固定模型可用 LAC 分数 `s(x,y)=1-p_y`。自然缺失模式 `m` 内有 `n_m` 个独立 calibration 患者，取排序位次 `k=ceil((n_m+1)(1-α))`；`k≤n_m` 时取第 k 个分数，否则取 `+∞`。预测集合 `Γ(x,m)={y:1-p_y≤threshold_m}`。只有模式内 calibration/test 可交换时，才能解释为该模式的覆盖结论。

真实缺失与随机 modality dropout 不是同一机制：医生可能依据病情安排 DNA/甲基化，分组患者中的病情分布仍会随时间、科室或检查路径变化。对全模态患者随机遮蔽后构建的校准组，不自动代表自然缺失患者。原 MC² 也明确指出其分组由随机 dropout 构建、信息性缺失可破坏假设，且其组保证不是类条件保证。[MC² 论文 §3、§6](https://papers.miccai.org/miccai-2026-sat/paper/MLMI_024.pdf)

五模态最多 32 种 mask，小样本逐格容易没有数据。建议预定义少数临床有意义的组、显示每组人数与集合大小；稀疏/未见模式输出所有三类并提示人工复核。若退回 pooled threshold，必须标注“无此模式覆盖保证”，不能继承逐模式结论。一个 calibration 患者的多图或多个模拟 mask 不能当多个独立样本扩大 n；按患者聚合或使用另有证明的群组方法。

### 5.3 重度漏判风险与一般 conformal 的区别

`P(Y∈Γ)≥1-α` 是集合覆盖，不意味着重度点预测 recall≥1-α，也不意味着每个病例错误概率≤α。若集合含重度可设计“不能排除重度”的复核提示，但原模型点预测、集合与建议三者需分别展示。

研究 CRC 时先定义单调策略族 `a_λ`：λ 增大只让严重程度上移。则 `L=1[Y=2]1[a_λ<2]` 随 λ 非增且有界。CRC 的有限样本修正控制的是这个**联合漏判事件概率**，并非 `P(a<2|Y=2)`。要研究条件重度漏判，可在独立重度 calibration 患者内施行类条件风险控制；按类再按缺失模式分组会迅速缩小样本量，必须报告人数和假设。未找到可行 λ 时不能强行宣布控制达标。

常用修正形式为 `(n/(n+1))*empirical_risk(λ)+B/(n+1)≤α`，但须遵循损失单调方向、参数集边界、可行性与交换性。一般不对称矩阵包含过估罚项，严重程度上移时总成本可能先降再升；因此不能把任意成本函数直接塞进原始单调 CRC。[CRC 原论文与官方实现](https://github.com/aangelopoulos/conformal-risk)

## 6. SHAP / InterSHAP 之外的图像与病例解释

| 原始方法 | 推荐用途 | 对 DINO / SYSU 的具体限制 |
|---|---|---|
| Integrated Gradients，Sundararajan et al., ICML 2017：[论文](https://proceedings.mlr.press/v70/sundararajan17a.html)、[Captum 官方 API](https://captum.ai/api/integrated_gradients.html) | 分别解释 `P(重度)=q2`、`P(至少中度)=q1`，以及连续 DNA/甲基化变量 | 固定二值 mask，不沿积分路径把缺失伪装成半观测；比较多种图像基线，记录 completeness delta |
| Grad-CAM，Selvaraju et al., ICCV 2017：[论文](https://doi.org/10.1109/ICCV.2017.74)、[ViT 官方库示例](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md) | CNN 对照及 ViT patch 梯度热图 | 518/B14 为 37×37 patch；必须剔除 CLS、还原 padding 并映射原图；不能套224/14×14模板 |
| Chefer, Gur & Wolf, CVPR 2021：[论文](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)、[作者代码](https://github.com/hila-chefer/Transformer-Explainability) | Transformer relevance，比直接画注意力更有方法依据 | 接入 timm DINOv2 与 CLS+patch-mean descriptor 需审计，不是自动即插即用；作为次优先对照 |
| ProtoPNet，Chen et al., NeurIPS 2019：[论文](https://arxiv.org/abs/1806.10574)、[作者代码](https://github.com/cfchen-duke/ProtoPNet) | “当前区域与哪些训练病例区域相似”的病例展示 | 仅检索相近 embedding 是展示辅助，不是内生 prototype 模型；prototype 学习/投影只能用训练患者 |
| IAIA-BL，Barnett et al., Nature Machine Intelligence 2021：[原论文](https://arxiv.org/abs/2103.12308)、[机构出版记录](https://scholars.duke.edu/publication/1504995) | 医工交叉参考：病例原型、医学语义、临床相关区域 | 乳腺任务与口腔不同，方法能用部分像素标注；未经病灶/语义审查，原型可能学习背景捷径 |

IG 公式为 `IG_j=(x_j-baseline_j)∫_0^1 ∂f(baseline+t(x-baseline))/∂x_j dt`。normalized zero 图像是均值颜色，不是黑图；原始黑图需要先按正式 normalization 转换。离散分类变量、病史和 TCT 可用分组遮蔽/替换的 SHAP 对照，连续插值需交代是否临床可行。

Grad-CAM 库的纯 CLS ViT 示例认为最后输出 patch 梯度为零，故选最后 attention 前的 norm。我们的 encoder 用 **CLS + mean patch**，该“必为零”论证不适用；应在实际 forward 图核验 target layer、梯度和热图，再选合适层，不能把库模板判断当本模型结论。若导出 embedding 缓存后训练融合模型，图像解释必须通过同一已保存 encoder 重新建立梯度图，不能对已 detach 的 embedding 伪造像素归因。[库作者 ViT 指南](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md)

### 6.1 和现有 SHAP / InterSHAP 结合

解释目标统一为 `q2`、`q1`、三类概率，或两项 action risk 的差值；hard argmin 本身不可导且边界不稳定。模型概率与成本偏好分别说明：例“预测更像轻度，但由于重度漏判代价，建议复核中/重度”，不能把提高决策级别归因于某种影像征象。

按患者实际观测集合 O 建立 coalition game `S⊆O`，原本未测模态始终保持缺失。对同一模型，遮蔽时同步改值与 mask，使用模型支持的缺失路径。禁止为解释而填入不存在的 DNA/甲基化真实值。五个全观测模态最多需要 32 个 coalition forward（若背景积分则另乘背景数量）；exact 指 coalition/Shapley 求和，没有让背景分布、模型或因果假设变成精确事实。

InterSHAP 将模态主效应和交互分开；局部有符号项用于方向解释，聚合交互占比用其定义，不能把大的交互占比说成更多临床价值，也不能把全局 mean-absolute 与 absolute-mean 混用。分母近零标记未定义，不夸大百分比。[用户参考原论文](https://arxiv.org/abs/2412.15828)、[AAAI 原文](https://ojs.aaai.org/index.php/AAAI/article/download/35452/37607)

建议分两种语义分别报告：A. 保持实际 mask，只解释观测值相对训练背景的贡献；B. 做模态移除干预，解释模型对可用信息的依赖。二者不能共用“某检查的生物学作用”标签。解释背景只取训练患者，并比较相关变量的分组背景/缺失路径；用独立 replacement 可能构造临床上不可能的组合。

### 6.2 解释也需要评价

必要对照：参数随机化、类别目标替换、随机区域遮挡；记录重要 patch 删除/插入时同一目标输出的变化，并比较随机区域与低贡献区域。黑块删除会产生分布外伪影，应预定义模糊/局部替换等干预并做敏感性分析。图像翻转/轻微亮度变化后，把解释映射回原图再比较稳定性；热图好看或稳定本身不等于忠实。参数/标签随机化是诊断手段，也不是完备证明。[Adebayo et al., NeurIPS 2018](https://arxiv.org/abs/1810.03292)

如有导师少量病灶区域或医学概念评审，可作为解释评价数据；不能把解释评价标注混进“只用全图标签”的训练配置而不说明。病例原型展示必须排除同患者训练/验证交叉；比赛外部展示需另作脱敏与授权，现阶段只在项目内生成。

## 7. 实施顺序、分支与消融

建议在工程基线稳定后，从同一 commit 隔离 `codex/sysu-cost-ordinal`、`codex/sysu-explainability`。本报告只创建本文件，未创建训练分支或改动模型。

1. **基线**：softmax proper CE、图像单模态/表格单模态/简单 late fusion；统一去年龄性别，显式缺失 mask。
2. **有序**：CORAL 风格有序阈值、CORN；相同 encoder、fold、seed、训练预算。保存两任务概率并核验合法性。
3. **决策**：同一概率分别 argmax 与固定 C；重度 recall、precision、accuracy 与低估距离的 Pareto 图。C 的候选小网格只在开发集研究，测试冻结。
4. **损失**：累计距离/不对称正则分别加，含 λ=0；不同时叠加 loss、class weight、重采样、阈值以免无法归因。
5. **可信输出**：未校准 vs 低参数校准；pooled vs 预定义缺失模式 conformal，报告集合覆盖、集合大小、空集/全类集比例与分组人数。
6. **解释**：固定已保存模型、SHAP/InterSHAP + IG/patch CAM；同一病例目标、背景、mask、成本版本完整记录。

每个配置报告：accuracy/macro-F1/balanced accuracy；各类 recall/precision；QWK（权重约定写明）；`MAE=mean(|a-y|)`；`mean(1[|a-y|=2])`；重度→轻度人数/重度人数；低估/过估率及 mean cost。同时显示混淆矩阵和患者人数。按自然缺失模式与人工遮蔽模式分开，不把两个分布的结果混成一个均值。患者 bootstrap 对指标/方法差值给区间，复诊/多图一起抽；多次选择模型后区间不能掩盖选择偏差。

## 8. 可提出的研究问题与创新措辞

可以检验：“有序且代价敏感的缺失模态融合，能否在患者独立 accuracy≥75% 约束下，减少重度低估/跨两级错误；其观测信息贡献和交互是否在缺失干预下忠实、稳定？”

潜在贡献需由实验支撑：把**实际缺失机制、严重度错误方向、病例解释稳定性**共同做患者层评价，比仅平均准确率和展示热图更契合比赛。融合、ordinal head、calibration、SHAP、modality-conditioned conformal 都已有先行工作；首创性需进一步明确新目标/算法/理论，不能凭组合宣称首次。

## 9. 检索与来源核验记录

采用 academic-research-suite 的检索、来源核验及综合阶段；这是定向实现调研，不是 PRISMA 系统综述。检索日期 2026-10-08；关键词含 `CORAL/CORN rank consistent ordinal regression`、`squared Earth Mover loss`、`temperature scaling`、`conformal risk control monotone loss`、`medical missing modality conformal`、`Integrated Gradients`、`ViT GradCAM`、`ProtoPNet mammography`、`InterSHAP`。来源优先原始论文/会议出版社、作者原始代码、官方 API；检索中二手摘要只用于定位，不作结论证据。

已核验 CORAL/CORN 论文身份与作者实现、温度校准官方 ICML 页、CRC 官方实现、ICML 缺失值文章、MC² 全文 §3/§5/§6（作者姓名以PDF首页为准）、解释原论文与工具文档。没有对全部论文做逐页完整复现，没有使用其未核验的数值作为 SYSU 目标依据。AI 辅助完成检索和推导，公式与接口建议需在实际代码/患者协议下核验。
