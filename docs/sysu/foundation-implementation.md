# SYSU 工程基线：实现说明与使用入口

分支：`codex/sysu-foundation`。工作目录：`C:/Users/aljcn/.codex/worktrees/sysu-foundation/OPMD-Pred`。

状态：代码待总协调者审阅；没有提交，没有真实训练，没有 GPU 操作，没有功能测试。本轮只做源码静态审阅和语法编译（文末记录），不能当作运行结果或模型正确性证明。SYSU 尚无新增指标，不能声称达到 75%。

## 1. 输入与接口

默认仅五模态：`image history tct dna methylation`。年龄/性别可从历史 CSV 读取为原始追溯字段，但 dataset 输出、数值 encoder、缺失 token、modality dropout、模型输入和解释均不包含人口学模态。新的数据准备工具由总协调者另行修改；本轮未重跑准备数据，不重写医院原记录。

模型可选：`--head softmax|ordinal`、`--fusion transformer|late`、`--modalities ...`。late 为各模态独立分类头在**实际观测模态上等权平均概率**；ordinal 平均累计概率再转换为三类概率。未选 image 时不建立图像 encoder。旧类名 `MultimodalOrdinalModel` 为兼容保留，其 `head_kind` 标识当前头。

既有有序头使用共享严重度 score 和 `softplus` 阈值间距，累计概率单调；三类概率为 `[1-q1,q1-q2,q2]`。新 CLI 默认概率 `argmax`。旧 `ordinal_prediction` 保留累计概率阈值 0.5 的**后验中位数**规则，旧探索训练入口仍使用它；两种决策可能不同，不能混合比较结果。未增加 CORN、VIME、SAM、SupCon 或一致性损失。

DNA 输入沿用当前处理字段：**DI≥2.3 细胞数 / 分析上皮细胞数**的比例，不是完整 DNA 倍体分布。甲基化使用当前总数字段，未把十基因缺失补成阴性零。历史变量的固定除数与 methylation/10 是既有固定量纲缩放，不从完整数据拟合参数；不引入新插补。缺失数值保留独立 feature mask，未知 TCT 与缺失区分。

### TCT

仅从有效 train 记录建立词表；`0=missing`，`1=unknown_observed`，训练见过类别从 2 开始。validation/test 新词为 1 且 `tct_present=True`。词表、特殊 ID、模型分类数量保存于完整 checkpoint。`load_sysu_records` 本身不拟合词表，其观测 TCT 初始都为 unknown；训练入口显式 fit/assign。旧探索训练与预训练入口也已改为 train-only fit。

## 2. 图像 encoder

新 CLI 默认 `--image-encoder dinov2_b14`，选择 image 时必须提供 `--image-checkpoint`，不自动下载权重：

- timm 模型 `vit_base_patch14_dinov2.lvd142m`，创建时 `pretrained=False,num_classes=0`。
- 纯 encoder artifact 中 `model`、`feature_dim=1536`、`input_size=518`、padding/norm 元数据核对后，对 `encoder_state_dict` 用 `strict=True` 恢复。
- 实际 descriptor：`forward_features` 的最终 norm CLS 768 + 所有 patch 均值 768，拼接为 1536。不能用 timm 原始 forward 的 CLS 768 替代。
- 518×518，PIL `ImageOps.pad` 保纵横比、BILINEAR、居中 padding RGB `[124,116,104]`，ImageNet mean/std。训练只有 horizontal flip + 轻微 jitter，验证/测试无随机增强。
- 默认 `--unfreeze-image-blocks 0`：参数 frozen，外层 model.train 时 encoder 保持 eval；forward **未用无条件 no_grad**，为未来对图像输入做 IG 保留梯度路径。
- 可选 last-N 解冻为 SYSU 适配：会同时解冻最后 norm。此行为不同于 Zenodo 最佳 SAM 方法冻结 final LN 的原协议，必须按新配置解释，不能声称完全复用该训练策略。
- `--image-encoder vit_b16` 支持旧 torchvision B/16、224 resize 与 `image_encoder` checkpoint。旧有包含人口学模态的融合 checkpoint 不兼容新五模态模型，不以 strict=False 跳过权重。

已知预训练交付位于服务器：

```text
/home/zjkj/24334067/OPMD-Pred-ViT-Pretrain/outputs/zenodo_transfer/dinov2_last4_sam_epoch16_encoder.pt
```

协调者提供的 artifact SHA：`58a3fcc724155c8ecfc7bf0e955b2bf022f0e2aa4bfed06a51a5b99c850a4a65`。本实施代理未远程读取或复制该权重；loader 按协调者实际核验的元数据键 `model`（不是 `model_name`）实现。

总协调者已把 `pyproject.toml` 固定为 `timm==1.0.29`，并执行 `uv lock` 更新 `uv.lock`（退出码 0）；只解析依赖，**未 sync/install、未下载模型、未功能检查**。新 lock 包含 huggingface-hub 2.1.1 / safetensors 0.8；既有个人服务器环境由协调者确认已装 timm 1.0.29 / huggingface-hub 1.32。两种环境不能写成已核验一致；实际训练仍需记录运行版本。

## 3. 患者协议：先修数据，再生成 manifest

当前协调者审计指出旧处理数据 139 记录没有真实 patient ID，且存在同图跨类别与跨旧 train/val。**不能用 sample_id 代替 patient_id**，不得自动修标签、依据测试准确率挑 seed 或反复重拆。

需要外部审核的匿名 `groups.csv`，每条样本正好一行：

```csv
sample_id,patient_id
<真实样本匿名标识>,<审核后的真实患者匿名标识>
```

同患者的多图与复诊用同一 patient_id。这个文件由人工身份/关联审核提供；软件只能检查赋值与 split 一致性，不能证明 patient_id 在医学上真实。代码不从文件名、类别或索引推断患者。

`build_sysu_manifest`：按 patient group 分配；同标签的相同原始图片 SHA 在不同患者组出现时只约束二者在相同 split，**不宣布二者是同一患者**。同图异标签立即停止并报告待核对的匿名 sample ID，所有图片路径存在性/字节 SHA 均核验。默认 group/component 随机划分，**不做标签分层保证**，输出各 split 人数和类计数；训练有效记录若不包含三类则拒绝启动，应审阅协议可行性，不用测试表现反复挑划分。

每行包括 no-input 行都分配 split，manifest 记录全部 IDs、匿名 patient IDs、标签、图片 SHA 和 `excluded` 理由。五模态全缺行不进入 dataset；单模态消融中该模态未观测的行也排除，并保存 `all_selected_modalities_missing`。因此既保存可预测样本指标，也明确报告全分配样本分母和 coverage，不能只挑有数据者展示 75%。

新 `train_sysu` **必须** `--split-manifest`，不调用旧随机样本 split。加载时重新检查 CSV SHA、标签映射、图片字节 SHA、patient/hash 跨 split、ID 完整性和 no-input 排除理由。两种 hash 分开保存：

- `split_manifest_sha256`：统一 sorted-key/compact-JSON canonical 内容摘要，加载 checkpoint 时重新计算。
- `split_manifest_source_sha256`：传入 manifest 文件原始字节摘要，仅作来源记录；不同缩进不冒充内容变化。

manifest 复制保存在 output；没有默默以当前文件覆盖已保存 protocol。

## 4. 最小使用指令

以下是**未来数据审核通过后的入口示例**，本轮没有执行。路径由使用者填入其个人目录；服务器计算资源需启动前即时核对。

```bash
export PYTHONPATH=src
python -m opmd_pred.scripts.build_sysu_manifest \
  --csv /personal/data/reviewed_sysu_samples.csv \
  --images /personal/data/SYSU_processed \
  --groups /personal/data/reviewed_anonymous_groups.csv \
  --seed 42 --val-fraction 0.15 --test-fraction 0.15 \
  --output /personal/protocols/sysu_split_v1.json

python -m opmd_pred.scripts.train_sysu \
  --csv /personal/data/reviewed_sysu_samples.csv \
  --images /personal/data/SYSU_processed \
  --split-manifest /personal/protocols/sysu_split_v1.json \
  --protocol sysu-patient-hash-v1 \
  --image-encoder dinov2_b14 \
  --image-checkpoint /personal/weights/dinov2_last4_sam_epoch16_encoder.pt \
  --head softmax --fusion transformer \
  --modalities image history tct dna methylation \
  --epochs 50 --batch-size 8 --lr 1e-4 --seed 42 \
  --dimension 128 --modality-dropout 0.15 --dropout-mode bernoulli \
  --unfreeze-image-blocks 0 --patience 5 --device cuda \
  --output /personal/outputs/sysu_softmax_five_v1
```

`--modalities image` 为图像单模态；`--modalities history tct dna methylation` 为表格对照（无需图像 checkpoint）；`--fusion late` 为等权概率 late fusion；`--head ordinal` 使用现有有序头。每种配置使用新 output，不覆盖旧输出。

dropout 两种模式均只作用实际观测模态，保留至少一个观测输入。bernoulli 为每个观测模态独立按 p 遮蔽，若全丢则随机保留一个；subset 为每个患者按 p 触发一次从其可用模态抽取非空子集（包含“全部保留”的可能），所以 p 是**触发概率**，不是期望缺失比例。空可用集合安全跳过，但真实 no-input 行应在 protocol/dataset 前已排除。

### 显式研究代价（可选）

默认 argmax。只有传 `--cost-matrix /personal/protocols/research_cost.json` 才启用附加 cost 决策，其 JSON 为 3×3 数组，行真实类别、列判定；有限非负、对角为零。**未提供医学默认值**。代码 `risk=probabilities@C`、action argmin；两套 metrics 同时记录。成本只用于附加决策，不改变训练损失或 checkpoint 选择；样本不足时不能靠重度优先规则冒充新特征学习或召回保证。

## 5. checkpoint、指标与独立评估

仅 train/val 循环。选择规则：validation **全分配样本 argmax accuracy**，平局看有效记录 macro-F1。没有自动 test，也没有用 cost test 表现选择 epoch/矩阵。最佳 `checkpoint.pt` 含完整 model（encoder/fusion/head）、optimizer、epoch、config、TCT vocab、全部训练 history、完整 manifest/canonical SHA、有效 split IDs、所有 exclusions、各 split coverage/class 分母、encoder 源 SHA 和软件版本；`history.json` 每轮完整保留，标记 `test_evaluated=False`。

指标 `accuracy` 是 eligible 记录条件准确率；`accuracy_all_assigned` 把无输入 abstention 算作未正确，以 assigned_count 为分母。同步显示 `coverage`、assigned/eligible 各类 support、全分配重度召回，防止仅报告完整病例抬高结果。保留 macro-F1、balanced accuracy、逐类 precision/recall、MAE、QWK、低估/过估率、跨两级率、重度→轻度数/率、重度 recall。逐类普通 precision/recall 使用 zero-division=0；无重度样本时 severe 指标为 null；QWK 期望分歧分母为零时返回 null，不伪造为 1 或发生 `.item()` 错误。成本平均值仅针对已预测者，缺失者不虚构一个医学代价。

这些指标当前按记录加权；患者分组互斥用于防止 train/val/test 泄漏，不等于已经做患者级概率聚合。若同患者存在多图/复诊，后续需根据临床 endpoint 预定义患者或就诊聚合，以及患者 bootstrap 区间，不能把本基线记录指标叫作已验证的患者聚合准确率。

独立固定 checkpoint 评估：

```bash
python -m opmd_pred.scripts.evaluate_sysu \
  --csv /personal/data/reviewed_sysu_samples.csv \
  --images /personal/data/SYSU_processed \
  --checkpoint /personal/outputs/sysu_softmax_five_v1/checkpoint.pt \
  --split test --batch-size 8 --device cuda \
  --output /personal/outputs/sysu_softmax_five_v1/fixed_test.json
```

先 CPU 读取 checkpoint 并检查 manifest/数据，再构造并迁移模型。使用 checkpoint 内 vocab/modality/head/C 与有效 IDs；没有测试时阈值/温度/矩阵参数。结果包含 checkpoint SHA、canonical manifest SHA、labels/probabilities/匿名 IDs 和 metrics。若 output 已存在则拒绝覆盖；这阻止意外覆盖，不意味着软件能限制使用者另取路径重测，实验协议仍要求固定测试审阅。

## 6. 当前解释入口

`analyze_multimodal` 仅接受 foundation 格式 checkpoint，从 checkpoint 恢复 config、词表、manifest 和 validation IDs；旧含 demo 的融合 checkpoint 明确不兼容。可选目标 `--target severe|at_least_moderate|expected_severity`；默认 severe。只做 validation 解释，不自动使用 test。

coalition 遮蔽保持真实未观测模态缺失，移除观测模态经已有 missing-token 路径；no-input coalition 是解释参考，不能当真实可用患者预测。保存 `f(empty)` 和 `f(full)`，waterfall 从实际 `f(empty)` 起始，不能写成 `E[f(X)]=0`。可视化 pairwise 矩阵用准确名称 Shapley interactions。

基线已补 [InterSHAP 原文公式 4–11](https://arxiv.org/html/2412.15828v2) 的标量目标适配：完整 SII 对角为 `shap_i-Σ_(j≠i)off_ij`；global `Φ=abs(mean_samples(fullSII))`，比例为 `ΣoffΦ/ΣallΦ`；local 每条记录先计算 `Σoff(abs(fullSII))/Σall(abs(fullSII))`，再对 defined 记录求均值。summary 同时保存完整矩阵聚合、两种公式、local 列表、undefined 数量；分母 ≤1e-8 时记 null，不夹 eps 制造比例。mean-absolute 矩阵仅作描述，不能替代 global 的 absolute-mean；不能线性组合已归一化比例得到另一目标的比例。

这是固定 scalar（如 severe 概率）的 **missing-token coalition game**，没有复现作者背景分布积分，也没有声称因果效应。local 指标此处按记录而不是患者聚合；若存在 undefined，均值明确只针对 defined 记录，不冒充原文所有记录均值。pairwise CSV 保持不变，完整对角可从保存的 shap 与 off-diagonal 重建。IG、CAM、conformal 和温度校准仍未实现。

```bash
python -m opmd_pred.scripts.analyze_multimodal \
  --csv /personal/data/reviewed_sysu_samples.csv \
  --images /personal/data/SYSU_processed \
  --checkpoint /personal/outputs/sysu_softmax_five_v1/checkpoint.pt \
  --target severe --device cpu --output /personal/outputs/sysu_val_explanations

python scripts/visualize_explainability.py \
  --summary /personal/outputs/sysu_val_explanations/summary.json \
  --samples /personal/outputs/sysu_val_explanations/samples.csv \
  --output /personal/outputs/sysu_val_explanations/figures
```

旧 `train_multimodal` / `pretrain_multimodal` 仅为明确带 warning 的样本级 exploratory 入口；新临床目标不得用它们的 random split。旧 contrastive pretrain 只走模态 encoder/projection，不经过 fusion transformer/CLS，因此没有复现 Gu 等工作中的融合视图对比预训练。本轮不扩写该研究方法。

## 7. 源码清单和未验证项

新增：`data/sysu_protocol.py`、`scripts/build_sysu_manifest.py`、`scripts/train_sysu.py`、`scripts/evaluate_sysu.py`。修改：`data/sysu.py`、`data/transforms.py`、`model/vit.py`、`model/fusion.py`、`model/ordinal.py`、`analysis/evaluation.py`、`analysis/shapley.py`、`scripts/train_multimodal.py`、`scripts/pretrain_multimodal.py`、`scripts/analyze_multimodal.py`、根 `scripts/visualize_explainability.py`、`pyproject.toml`。总协调者另修改 `scripts/prepare_sysu.py`，不在本代理编辑范围。

尚未执行：真实 protocol 构建、患者映射核对、权重本地 strict 恢复、概率/梯度/指标功能检查、任何训练或评估、依赖安装。lock 已由协调者更新，但不代表运行环境验证。需要总协调者静态审阅；若用户随后授权测试，再做明确的 CPU 小型功能验证。真实数据当前的身份/标签冲突必须先人工解决，不能用代码存在或语法正确当作交付模型达标。

2026-10-08 语法记录：对上述 14 个新增/修改的 Python 源文件执行 `python -m py_compile`，退出码 0、无输出；没有导入模块或执行 forward/数据加载/训练。这只说明 Python 语法可编译。静态 `rg` 检索 `src/` 与 `scripts/` 的 `demo/demo_mask/demo_encoder/demo_token` 活动引用无命中；`sex/age` 仅保留历史 record 追溯读取。

随后小修正增加 `intershap_summary`（约 27 行）：对 `analysis/shapley.py` 与 `scripts/analyze_multimodal.py` 再做语法编译，退出码 0、无输出；仍未运行数值/功能检查。

## 总协调者最终静态复核补充

已显式在指标标记evaluation_unit=record；coverage保存分配/有效/完全拒判患者数。代价动作也报告全部分配重度召回。legacy融合checkpoint需TCT词表相同才可载入。encoder_source补记public revision/SHA、特征定义、预处理与normalization。DINO解冻SYSU最后block时同时解冻finalLN，属于后续SYSU适配，与Zenodo末4块固定finalLN配方不同。

prepare_sysu显式命名工作表、新目录输出且不含年龄性别，图像绑定或精确病损图选择歧义报错；本轮未运行该脚本。现有139条CSV与87张图保持原样。依赖声明增加绘图所需matplotlib并重新uv lock解析成功，未执行uv sync或安装，不能视为运行环境验证。

静态diff与source审阅不证明数据协议/模型前向/梯度/指标/解释可运行。真实训练和功能检查仍未执行，患者匿名映射与同图标签核对是训练前置材料。

总代码最终AST解析27个Python文件、git diff --check和uv lock --check成功；无模型或模块执行。
