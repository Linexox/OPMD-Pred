# SYSU 三分类：缺失多模态、小样本与可信解释的文献核验

检索日期：2026-10-08。状态：研究设计，尚无本文件提出的方法在 SYSU 上达标的实验结果。仅使用论文原文、官方会议/出版社页面和作者仓库；不将检索摘要或第三方复述当作原文证据。

范围：口腔 RGB、病史、TCT、DNA 倍体、甲基化。用户已确认保留病史与 TCT；年龄、性别及其缺失指示一并禁用。病史 9 维是当前工程约定，临床含义和字段来源需数据审计确认。本文件不访问医院患者记录、不上传数据、不启动训练、不删除远程文件。

## 1. 先核对用户指定的三篇

### 1.1 M2-OPMDNet：口腔融合和解释的直接参考

You 等，*Explainable Multimodal Deep Learning Integrating Imaging and Clinical Data for Oral Potentially Malignant Disorder Detection*，arXiv:2609.04512 **v1，2026-09-03**，本次核验的是预印本。[完整原文](https://arxiv.org/html/2609.04512v1)

- 数据：667 名受试者，2650 组白光/自体荧光配对影像，23 项问卷；患者分组五折。
- 标签：口腔专科医生的“无增高风险/增高风险”筛查结果。不是 SYSU 正常/轻度、中度、重度的三级病理任务。
- 方法：图像编码器与问卷 MLP 拼接、加权二分类损失、分阶段微调；SHAP 背景取训练数据。
- 报告 AUC 0.952；**不是正确率 95.2%**，也不是我们的三级任务达标依据。
- 适配：病史标准化、图像与病史对照、训练集解释背景、患者隔离可直接借鉴。我们没有荧光影像，不制造该模态；文中人口学变量不能移植进来。
- 限制：筛查标签本身结合病史，问卷贡献可能部分反映医生筛查规则；不能据此宣称 DNA/甲基化具有病理分级预测效益。原文未给出可直接复现的公开作者仓库。

### 1.2 Gu 等：可学习缺失 token + 同时模态 dropout

Gu、Saito、Ma，*Learning Contrastive Multimodal Fusion with Improved Modality Dropout for Disease Detection and Prediction*，**MICCAI 2025**；arXiv:2509.18284 **v1**。[会议官方页](https://papers.miccai.org/miccai-2025/0483-Paper2038.html)、[原文](https://arxiv.org/html/2509.18284v1)、[作者代码](https://github.com/omron-sinicx/medical-modality-dropout)

- 原文数据：PE 1837 次 CT/1794 名患者；NLST 64117 次 CT/12498 名患者。均患者分组四折，折内留 10% 验证。
- 冻结单模态编码器，在融合输入加入可学习缺失 token；同时监督完整、图像单模态和表格单模态视图。对比学习是标签监督的 sigmoid 形式，包含单模态与融合表示的配对。
- PE 多模态 AUROC 0.842；评估单模态推理和模拟缺失。未证明临床检测选择导致的非随机缺失被消除。
- 实施成本低：先缓存 SAM-DINO 表示，训练小融合头。五模态有 31 个非空子集，**不能直接沿用两模态“全枚举”训练而称成本不变**；应采样已观测模态的子集。
- 作者仓库存在 `losses`、`networks`，README 只有论文说明，未提供完整数据/启动流程。本次未固定仓库 SHA；复现时需锁定版本并补齐协议。

本项目附加审计要求：论文用 SHAP 选表格字段，原文未充分交代每一步是否折内执行；我们的选择/标准化/解释背景必须训练折内拟合。患病时间或病理报告不能进入“诊断前”输入。

### 1.3 InterSHAP：交互证据的量化，而非医学因果

Wenderoth 等，*Measuring Cross-Modal Interactions in Multimodal Models*，**AAAI 2025，39(20):21501–21509**；arXiv:2412.15828 **v2，2025-01-28**。[会议页](https://ojs.aaai.org/index.php/AAAI/article/view/35452)、[原文](https://arxiv.org/html/2412.15828v2)、[作者代码](https://github.com/LauraWenderoth/InterSHAP)

- 使用 Shapley interaction index 分离单模态贡献与交互；支持局部/全局及超过两模态，无需标签计算交互分数。
- 验证：受控合成任务；单细胞 RNA/蛋白；MIMIC-III 生理时序/静态变量，后者 80/10/10 分割。
- 枚举 coalition 的复杂度随模态数指数增长；五模态 32 个 coalition 可以缓存编码器后计算。
- 交互强度不等于预测准确性、临床价值或因果作用。不同背景/遮蔽定义可产生不同解释。

本项目适配：优先解释 `P(重度)`，另报每类概率或期望级别，不能混用 logit/probability 标度。对自然缺失样本，只让已观测模态参与主解释；将“检测未做”单独显示，不解释成阴性生物信号。作者的可视化可以复用，SYSU 缺失语义仍须我们明确设计。

## 2. 另外九篇：筛选矩阵

表中“成本”是结合当前项目的工程估计，不是论文声称的 SYSU 运行时间。无患者 ID 的论文协议不能替代我们的患者独立协议。

| 工作、核验版本与原文 | 数据与原协议的直接证据 | SYSU 可实施借鉴 / 成本 / 限制 |
|---|---|---|
| **HeMIS**，Havaei 等，MICCAI 2016 / arXiv:1607.05194。[作者预印本](https://arxiv.org/abs/1607.05194) | 脑肿瘤与多发性硬化 MRI 分割；把不同序列映射到公共表示，用可用序列的统计量融合，并移除序列检验鲁棒性。 | 低成本对照：可用模态表示的 masked mean/variance，不合成缺失检查值。原始像素 MRI 有空间配准，RGB/病史/DNA 不具有该条件；只能借鉴集合融合思想，不能直接搬分割架构或分割结果。 |
| **SMIL**，Ma 等，AAAI 2021，35(3):2302–2310。[原文](https://ojs.aaai.org/index.php/AAAI/article/download/16330/16137)、[作者仓库](https://github.com/deep-real/SMIL) | MM-IMDb、CMU-MOSI、avMNIST；训练时文本/声音仅 10%/20% 可用等人工设定，贝叶斯元学习生成缺失表示。 | 高成本研究对照，训练和推理均缺失的任务定义有用；SYSU 配对样本太少时生成的甲基化/DNA可能仅复制类别先验。保留为备选，不先实现双层元学习，更不把生成值展示成真实检测值。 |
| **MultiBench**，Liang 等，NeurIPS Datasets and Benchmarks 2021。[官方原文](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/37693cfc748049e45d87b8c7d8b9aacd-Paper-round1.pdf)、[作者实现](https://github.com/pliang279/MultiBench) | 15 数据集、10 模态、20 任务；有性能、训练/推理复杂度、噪声/缺失鲁棒性评价。包含医疗任务，但不是 OPMD 专用方法。 | 低成本评价模板：完整、自然缺失、模拟缺失分别画曲线，报告参数/时延。仅借鉴评估接口，避免为小项目整体迁入旧依赖。普通人工破坏鲁棒性不能证明真实临床缺失泛化。 |
| **GRU-D**，Che 等，Scientific Reports 2018，8:6085。[出版社原文](https://www.nature.com/articles/s41598-018-24271-9) | PhysioNet ICU 与 MIMIC-III，5 折；值、观测 mask 和时间间隔共同建模真实医疗时序缺失，另有合成缺失实验。 | 借鉴“缺失模式本身有信息”与 mask-only 对照。SYSU 若只有一次静态检查，不实现时间衰减 GRU，也不虚构检测间隔。mask 的预测效力可能来自检查路径，跨时期/科室外推需检验，不能把相关性称作疾病原因。 |
| **VIME**，Yoon 等，NeurIPS 2020。[原文](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf)、[作者代码](https://github.com/jsyoon0823/VIME) | 表格自监督 mask/值重建及一致性；临床实验约 1 万 UK 有标签前列腺癌患者 + 超过 20 万 US 无标签样本，UK 50% 训练，其余测试。 | 低成本部分思想：对已观测字段再遮蔽，做训练折内重建和分类一致性。若没有大量独立无标签 SYSU，不能宣称复制半监督收益；逐字段随机替换容易造出不合理病史组合，禁止跨患者拼接完整检测记录。 |
| **CORAL**，Cao、Mirjalili、Raschka，Pattern Recognition Letters 140 (2020):325–331。[作者全文](https://sebastianraschka.com/pdf/papers/2020-coral.pdf)、[作者代码](https://github.com/Raschka-research-group/coral-cnn) | MORPH-2/CACD/AFAD 人脸年龄，随机图像 80/20；共享权重的累计二分类有 rank consistency 保证。 | 极低成本三分类有序头：`P(Y>0)` 与 `P(Y>1)`。借鉴级别顺序，**不使用年龄输入或人脸数据**。有序概率不自动体现低估的医学代价，也不保证准确率/重度召回提高；需独立不对称决策消融。 |
| **Sanity Checks for Saliency Maps**，Adebayo 等，NeurIPS 2018。[会议全文](https://proceedings.nips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf) | 图像分类上参数随机化与标签随机化，证明某些视觉上好看的热图不依赖学习到的模型/标签。 | 低成本参数随机化审计；图像遮挡检查与轻微变换稳定性一起评估。通过这些检查只排除部分失真，不证明病灶定位正确；没有病灶标注时不报定位 IoU。标签随机化需要额外训练，只在后续专门验证阶段考虑。 |
| **ProtoPNet / This Looks Like That**，Chen 等，NeurIPS 2019。[会议全文](https://proceedings.neurips.cc/paper/9095-this-looks-like-that-deep-learning-for-interpretable-image-recognition.pdf)、[作者代码](https://github.com/cfchen-duke/ProtoPNet) | CUB 鸟类和 Stanford Cars；原文 CUB 训练使用数据集 bounding box 裁剪。通过原型 patch 相似度分类。 | 中成本图像原型展示方向：只从训练患者取相似参考 patch，显示“与哪些训练病例像”。DINO 全局 token 的感受范围使 patch 不能天然等价于局部组织。原文不是无框口腔医疗证据；医师审阅后才赋临床名称，检索近邻也不能冒称 ProtoPNet。 |
| **Concept Bottleneck Models**，Koh 等，ICML 2020，PMLR 119:5338–5348。[原文](https://proceedings.mlr.press/v119/koh20a/koh20a.pdf)、[补充协议](https://proceedings.mlr.press/v119/koh20a/koh20a-supp.pdf)、[作者代码](https://github.com/yewsiang/ConceptBottleneck) | OAI 膝关节 X 光：4172 患者、36369 观察，患者独立 train/val/test；放射科标注 10 个概念，预测 4 级有序严重度。另有 CUB。 | 医工可解释性较直接类比：图像→临床可检查概念→等级，可编辑概念看模型响应。需导师提供口腔形态概念标注，成本主要在标注；不从病理结论生成输入概念。编辑模型概念是模型干预，不证明改变患者病灶会改变真实病理。 |

可解释性优先级：先把已有 SHAP/InterSHAP 的背景、目标、缺失遮蔽和稳定性做好，再考虑病例原型；概念瓶颈等获得合格概念标注再实施。VIME/SMIL 的完整框架不是当前最小必要工程。

## 3. 对“可信增强”和真实缺失的具体约束

以下是结合 SYSU 场景提出的适配设计，**不是上述论文已证明的结论**。

1. 数据审计应区分“不做检查”“记录不全”“样本无效”“结果尚未出”及“确实阴性”。如果来源无法区分，保留未知原因，不臆造缺失原因标签。先检查 TCT/DNA/甲基化是否早于作为金标准的病理诊断。
2. 自然缺失模式先原样保留；再遮蔽只能作用于已观测模态，训练子集必须非空。不能使用缺失值的占位零来参加重建目标、对比正对或 SHAP 的正常背景。
3. 训练与解释分别使用 mask：自然观测 mask 说明检查可用性，模拟 mask 记录研究干预。可做统一最终 mask 的模型基线，但评估记录必须能区分两者。
4. 数据少时先冻结 SAM-DINO 图像编码器，轻量投影与表格头；较大的微调模型作为独立方向。维持患者隔离，避免同一患者重复图像被当作对比学习负样本。
5. DNA、甲基化以真实检测的有限数值/分类向量为独立模态。检测字段内部缺失需字段 mask，不能“其中一个字段有值”就宣称整模态完整。统计量仅在训练折拟合。
6. 不生成用于展示的假检测值；最早的增强只做医学上轻微、不会改变病灶语义的图像扰动与观测子集遮蔽。强色变、跨患者病史 mixup、合成甲基化阳性均不先采用。
7. 对自然缺失的条件化抽样可减少训练/测试模式差异，但**不能从一份观察性数据识别或消除 MNAR 机制**。如果 mask-only 能预测类别而跨时期效果下降，需报告检测流程偏差，不能将该效果作为生物学创新。

## 4. 三个可证伪假设和对应分支

### H1：观测子集监督与一致性可提高自然缺失条件下的重度召回

分支建议：`codex/sysu-missing-consistency`。沿用冻结图像表征、小模态编码器、显式 mask，加入已观测视图到随机子视图的监督/一致性。先做最小消融，不一开始叠加对比、重建和生成。

最小训练式：`L = Lclass(obs) + a * Lclass(subset) + b * confidence_gate * KL(stopgrad(p_obs) || p_subset)`。

- 子集来自当前患者已观测模态。教师只是该患者较完整的观测视图，不能称“全模态老师”。
- 临床信息删除可能合理改变预测；过强一致性会抹掉 DNA/甲基化增益。`b=0` 为必要对照，门控阈值只用开发集拟定。评估校准而非仅比较硬标签。
- 对照：图像-only、表格-only、简单 late fusion；zero+mask；learned missing token；随机 dropout；子集监督；再加入一致性。另做 mask-only 和打乱 mask 的诊断对照。
- 主要检验：相同患者划分/训练预算下，自然缺失队列重度 recall 与整体 accuracy；报告每模式人数、MAE、跨两级错误及患者 bootstrap 差值区间。
- **反证条件**：自然缺失无增益、增益只在人工随机缺失成立、DNA/甲基化加入后无贡献、或收益完全由 mask-only 复制。不能仅凭完整病例 accuracy 改善认定 H1 成立。

扩展对比学习只在 H1 最小版本有证据后加入：有效观测模态及其融合视图作为对，不能把缺失 token 当检测表征。至少比较同患者对比与标签监督对比；同级别患者不一定有同一病变表型。

### H2：有序概率与不对称决策可减少低估和跨两级错误

分支建议：`codex/sysu-cost-ordinal`。使用两累计概率 `q0=P(Y>0)`、`q1=P(Y>1)`，必须 `q0>=q1`，还原 `p=[1-q0,q0-q1,q1]`。CORAL 只是方法来源；具体有序参数化和损失按实现证据核验。

决策用 `argmin_a sum_y C[y,a]*p[y]`。成本矩阵需导师认可；候选研究矩阵可设行=真实级别、列=判定级别：`[[0,1,4],[2,0,1],[8,3,0]]`，**数值是待定假设，不是医学标准**。重度判轻处罚最大；相邻错误轻于跨两级，低估高于对应过估。

- 先用 CE+argmax、CE+代价决策、ordinal+常规决策、ordinal+代价决策形成 2×2 消融；同时固定相同输入/分割。
- 加权 CE、采样平衡会改变输出概率，不能直接视为自然患病概率；需要训练折/内部验证校准或同时比较未加权模型。
- 主要检验：整体 accuracy≥75%、重度 recall、重度→轻度、跨两级率、MAE/QWK 与过估率共同报告。开发集选择矩阵/阈值，测试固定。
- **反证条件**：仅全判重度提高 recall、accuracy 明显下降、或跨两级错误未下降。不能把“阈值调高/调低”包装成表征变好。

### H3：交互解释在缺失条件和轻微扰动下仍可追溯且忠实

分支建议：`codex/sysu-explainability`。这是假设解释质量可验证，而不是假设 InterSHAP 必然优于 SHAP。

- 自然缺失保持不变，已观测的 K 个模态做 `2^K` coalition。分别实现“移除该观测模态”的 availability 游戏与“训练背景替换内容、保留可用性”的内容游戏；二者含义不同，不混为一个贡献数。
- 非线性融合删除模态后允许概率变化。显示三类概率、严重度、缺失状态、单模态贡献、交互，并标明解释目标与背景。
- 检查加和残差、改变背景后的方向/排序稳定性、去掉高贡献模态后的预测变化。背景从训练患者按自然缺失模式取；模式太稀少需标注背景不可靠。
- 图像同时给遮挡响应与适配 ViT 的梯度解释，随机化头/编码器后检查图是否随模型变化。均值填充遮挡也可能分布外，至少比较不同遮挡基线，不把降分自动等价于病灶证据。
- **反证条件**：热图对随机模型保持不变、贡献加和不符目标、解释随背景反号且无提示、或实际未观测 DNA 被展示为预测阴性原因。医学导师复核不能替代算法忠实性检查。

## 5. 推荐先落地的研究组合

主线是 **患者独立、缺失感知的有序多模态风险分层**，先以简单 late fusion 建立可复现底座，再验证 H1、H2；H3 提供可审计的预测展示。图像采用 Zenodo 阶段导出的 SAM-DINO，表格保留病史/TCT/DNA/甲基化，彻底移除年龄性别。

可提出的创新主张须限定为待验证：把自然检测缺失与子集训练区分；在固定 accuracy 约束下减少重度低估；解释同时区分检测可用性与观测内容，并审计跨模态交互。单纯使用 DINO、CORAL、modality dropout 或 SHAP 都已有前作，不能单项宣称原创。

比赛故事的技术证据应当来自预先定义的患者分割与消融，而非挑选个例：模型在少检查条件下保留安全的风险分层能力、检测补齐后概率变化可以追溯、错误严重程度被明确衡量。75% 是 SYSU 待实现目标，Zenodo 75.40% 不能转移成 SYSU 已达标。

## 6. 检索和可复现性备注

- 已直接读取三篇用户参考的 HTML 原文，以及 SMIL、VIME、CORAL、ProtoPNet、Concept Bottleneck 的全文/补充协议；其他条目读取官方论文页或出版社全文。没有把不确定的评估切分写成患者独立。
- 作者仓库链接已核对；GitHub REST 请求返回 rate limit，未获得可核验 head SHA。因此这里只固定论文版本/年份，仓库不能宣称已锁定版本，也未执行任何仓库代码。
- 研究矩阵未逐项追求最新 SOTA：优先能与当前五模态、小样本、三级有序目标形成清晰对照的工作。大型多模态生成/预训练框架暂不优先，现有私有样本不支撑其数据需求。
- 本轮没有训练、生成医院数据或访问远程输出。后续任何 accuracy/recall 提升须来自实际患者独立实验；本文件所有 H1–H3 都保留可失败的条件。
