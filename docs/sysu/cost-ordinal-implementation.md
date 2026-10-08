# 有序概率与错误距离研究分支

分支`codex/sysu-cost-ordinal`，基于`e2e090f`工程基线。当前仅代码与静态审阅，未运行模型/训练/功能检查，没有SYSU性能增益证据。

## 修改与来源

保留默认softmax、原有有序阈值头和显式独立cost决策。新增`--head corn`及`--ordinal-distance-weight`（默认0），用于分别比较概率参数化与有序距离。

[CORN原文v5](https://arxiv.org/abs/2111.08851v5)用条件任务：r1=P(Y≥1)，r2=P(Y≥2|Y≥1)，所以q1=r1、q2=r1*r2，p=(1-q1,q1-q2,q2)。两个条件logits各自由同一融合表示的独立线性输出给出，不采用原有共享严重度分数限制。

损失第一项监督全部标签；第二项仅在真实y≥1记录中监督y=2。求和后除以N+N_(y≥1)，空条件子集不产生第二项。实现使用正确选出的单任务`pred`进行BCEWithLogits，不复制2026-10-08作者main源码中对全二维logits调用logsigmoid的疑似形状错误。小batch条件任务的随机计数归一化和有限模型并不保证精确后验，概率校准与真实泛化仍待实际评价。

Transformer是优先CORN对照。late fusion按每模态的**无条件累计概率**先等权平均，再重构条件logits以复用同一损失/概率接口；不能直接平均条件概率后声称等价平均三类概率。这是late-fusion适配，不是CORN论文整体复现。极端概率重构夹到[1e-7,1-1e-7]以保持logit有限，可能影响饱和值。

新增距离项为`mean_i Σ_k(q_ik-1[y_i≥k])²`，k=1,2，与三类预测/目标CDF的平方差等价；**不是绝对W1，也不是W1²**。[Squared EMD参考](https://arxiv.org/abs/1611.05916)。它是两个事件的Brier式目标，有序类别间距按1计，属于对称目标，不能自动实现宁可过估的偏好。

总训练损失是原概率损失+λ乘平方CDF距离。λ默认0，源码记录分开的probability_loss、squared_cumulative_distance、weight；验证/固定评估沿用checkpoint保存的λ，防止loss解释不一致。checkpoint选择依旧验证集全分配argmax准确率/F1平局，不由测试、cost动作或最低loss选模型。

训练诊断为记录加权的minibatch归一化loss。独立静态复核发现CORN条件任务分母依赖batch内标签数，因此验证/评估另按所有任务项总分母汇总`global_conditional_task_mean`，避免仅改变eval batchsize改变其loss；平方距离仍按记录平均。汇总规则保存在指标内，不能把训练minibatch诊断和验证整体任务均值混称相同统计量。此修正仅经静态代数核对，未运行数值验证。

## 独立成本决策

延用工程基线显式`--cost-matrix`，行真实等级、列动作，`risk=p@C`、argmin决策。这里没有新的默认医学成本，没有成本训练正则，没有重采样/类权重叠加。预测概率argmax和成本动作必须各自报告accuracy/重度precision与recall/低估/跨两级率/预测重度比例。成本动作的all-assigned重度召回也按真实全部重度分母报告，资料不足不能悄悄算阴性。

## 运行示例与消融

先完成医院患者ID/图片绑定/标签与检查时间核对，构建冻结manifest；下列只是未来指令，尚未执行。GPU启动前先检查资源。

```bash
PYTHONPATH=src python -m opmd_pred.scripts.train_sysu \
  --csv /personal/data/reviewed_sysu_samples.csv --images /personal/data/SYSU_processed \
  --split-manifest /personal/protocols/sysu_patient_manifest.json \
  --image-checkpoint /personal/models/dinov2_last4_sam_epoch16_encoder.pt \
  --head corn --ordinal-distance-weight 0 \
  --output /personal/outputs/sysu_corn_seed42
```

对照次序：softmax/λ0；ordinal/λ0；CORN/λ0；随后各头分别比较小的λ（例如开发集预设0/.1/1，数值只是研究网格）。同一个已选定概率模型分别报告argmax与冻结成本决策。每个配置用独立output，其他编码器、模态、患者manifest、预算和seed不变，不按test挑头/λ/C。

默认不做温度校准或conformal，不将未经校准概率叫临床患病概率。本分支指标按记录，患者拆分隔离不等于患者/复诊终点已聚合。是否改善75%准确率、重度召回及跨两级错误，只能由之后的冻结协议实测证明。
