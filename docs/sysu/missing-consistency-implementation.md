# SYSU 观测子视图监督与一致性：独立研究实现

分支 `codex/sysu-missing-consistency`，基础 commit `e2e090f`；只在该分支工作树实现。状态：经总协调者静态审阅批准，源码与说明随本提交保存；未训练、未运行模型或功能测试。当前患者关联与同图标签问题未解决，不能启动可信训练。实现依据同目录 `method-proposal-2026-10-08.md` 与 `method-review-2026-10-08.md` 的修订结论；不是已验证新主模型。

## 1. 开关与原基线关系

新增参数：

- `--subset-weight a`，默认 0。
- `--consistency-weight b`，默认 0。
- `--consistency-type cdf_mse|forward_kl`，默认 cdf_mse。

两权重为 0 时走原有训练单 forward、原 `--modality-dropout` 路径，不计算额外 O/S forward、一致性或视图 trace。模型容量、概率头、验证/测试、选择规则与患者 manifest gate 保持基线。默认不会启用绝对 W1、置信门控、熵约束或对比预训练。

只要 a 或 b 为正，训练目标改为

```text
L = Lprob(O,y) + a*Lprob(S,y) + b*Lcons(stopgrad(pO), pS)
```

O 是患者**真实观测**的所选模态集合；没有 teacher 补全未测甲基化或其他模态。S 从 O 按现有 dropout 算法抽取非空子集。两者共用当前 batch 中已经增强的同一 image tensor，调用一次 `modality_embeddings(batch)`，再分别 `forward_from_embeddings(embeddings, observed)` 与 `(..., subset)`；不重载图片、不重新增强、不重新计算 encoder。

子集只通过 modality presence mask 移除完整模态，保留下来的模态仍用原 feature-level mask/值。原 batch 的实际观测 mask 与新 subset batch 的 dropout mask 是独立对象；history 每个 epoch 保存 `view_mask_names` 及每条匿名 `sample_id` 的 `observed_mask/subset_mask`，不能把模拟缺失标成医院未检测。missing-token 路径处理移除模态，不用重新编码被遮蔽的数值作为伪观测。

subset 模式中 p 是抽取动作触发概率，非期望缺失比例；抽取非空子集允许 S=O。只有一个模态可用时，S只能等于O；p=0 时不会移除任何模态。bernoulli 模式逐模态抽取并随机保留最后一个输入。真实无输入记录仍由前置协议排除，并计入 assigned 分母。研究启用后默认训练主指标对应 O，baseline 主指标可能对应 dropout 视图，不能直接把两者的训练 accuracy 当相同条件；需要比较独立的自然观测 validation。

## 2. 一致性定义与梯度

两个头都先转换为合法三类概率。`q=[p1+p2,p2]` 为至少中度/重度的累计概率。

```text
cdf_mse = mean_batch(sum_k (qS[k] - stopgrad(qO[k]))^2)
forward_kl = mean_batch(sum_y pO[y]*log(pO[y]/pS[y]))
```

KL 方向是 teacher O → student S。student 概率日志用 `clamp_min(1e-8)` 避免有序头的数值零导致 log(0)，这是有限数值近似，不是置信门控或温度校准。teacher target 在 helper 内 detach；O 的真实标签监督照常传梯度，S 的监督/一致性向共享 encoder（若解冻）、模态投影和融合头传梯度。没有 detach 共享 embeddings、没有额外优化器步。

损失/权重分别保存于 `config.json` 与 checkpoint `view_recipe` / `training_config`；每轮记录未加权 `observed_supervised`、`subset_supervised`、`consistency` 与加权总 loss。若 b=0，为避免计算不用的匹配项，consistency 日志记0；S supervised 仍用于诊断，即使 a=0也可计算但没有该项训练权重。

## 3. 评价与可否决性

启用训练时 separately 保存 `observed_view` 与 `subset_view` 的 accuracy、class recall/precision、MAE、跨两级错误与重度指标，包含同一 assigned coverage 分母；不能只用 O-S 性能差缩小宣称鲁棒，因为也可能是 O 性能变差。模拟 S 的指标不代表自然缺失人群。

validation/test 仍只对完整的**实际观测 O**单次 forward；没有验证时随机遮蔽、teacher 或一致性 loss。最佳模型依旧由全 assigned validation argmax accuracy、平局 eligible macro-F1 选择；独立 `evaluate_sysu` 固定测试，不用 test 挑 a/b/抽样类型。

融合层 dropout 在 O/S forward 中分别采样，teacher 不是独立模型，也未用 eval/EMA teacher；即使 S=O，一致性可反映 dropout 噪声。这个噪声源尚未解决，需在后续对照里观察，不能把减少一致性 loss 写成自蒸馏保证校准。这里不要求逐患者 `entropy(S)≥entropy(O)`，不把高置信 teacher 视为正确或无偏。

平方匹配/forward KL 的理想条件均值性质不保证有限样本共享模型获得真实后验；139条小样本、MNAR检测路径与不正确teacher都可能使方法失败。原绝对 W1 项有明确多数类塌缩反例，所以未加入此分支；W1 可作为后续分析量，而非本训练默认目标。

## 4. 约定实验指令

下面均是**数据与协议审核完成后**的未来示例，本轮未执行。先使用基础说明中的真实匿名 patient manifest；所有配置采用新输出目录、固定同一患者划分/seed/预算。完整公共参数可见 `foundation-implementation.md`。

```bash
export PYTHONPATH=src
python -m opmd_pred.scripts.train_sysu \
  --csv /personal/data/reviewed_sysu_samples.csv \
  --images /personal/data/SYSU_processed \
  --split-manifest /personal/protocols/sysu_split_v1.json \
  --image-checkpoint /personal/weights/dinov2_last4_sam_epoch16_encoder.pt \
  --head ordinal --fusion transformer --unfreeze-image-blocks 0 \
  --modality-dropout 1 --dropout-mode subset \
  --subset-weight 1 --consistency-weight 0.05 --consistency-type cdf_mse \
  --epochs 50 --batch-size 8 --lr 1e-4 --seed 42 --device cuda \
  --output /personal/outputs/sysu_observed_subset_cdf_v1
```

0.05 仅为研究示例，不是已选最佳权重或导师指定值。必要对照：a=b=0 原基线；a=1,b=0 O+S标签监督；a=1,b>0平方累计；同样 a/b 的 forward KL。只在开发集选择权重，记录 softmax/ordinal、抽样类型与 p。GPU任务未来每次启动前仍需即时资源审核；本实现不自动启动。

## 5. 变更和审阅状态

仅修改 `src/opmd_pred/model/losses.py`、`src/opmd_pred/scripts/train_sysu.py`，新增本说明；没有扩写 legacy contrastive、网页、IG、校准或测试平台。默认推理模型/完整 checkpoint 格式与 foundation 兼容；新增 view_recipe 为训练方法来源，不改变恢复后的单次推理。

未运行：真实数据/模型加载、任何 forward/backward、损失数值例子、unit tests、训练、GPU操作。本轮仅做静态代码审阅、源码语法 AST 与 diff 检查；这些不证明子视图mask、概率梯度或效果正确。总协调者已核对共享 embeddings、target detach、分母与默认路径并批准保存，本提交不表示模型效果验证。

2026-10-08 静态记录：两个修改源文件用 `ast.parse` 读取，成功且未导入任何项目模块；`git diff --check` 无错误（仅 Windows LF/CRLF 提示）。源码增量为 82 行增加、16 行移除，其中提取了原有指标汇总块；本轮没有增加测试文件或执行数据/模型。
