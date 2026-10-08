# 训练集类别平衡对照

在同一原始来源姓名/SHA分组、同一basic五模态、同一冻结SAM-DINO/Softmax/优化预算下，仅改变概率损失的类别权重。`--class-weighting none|balanced`默认none；balanced只支持softmax，不与CORN/额外成本/一致性混用。

权重仅由有效训练记录计算：counts45/25/22、N92，w_y=N/(3*n_y)，即约.68148/1.22667/1.39394。未使用验证或测试类别比例拟合权重，也没有预设医学漏判成本。

训练使用PyTorch weighted CE minibatch目标。验证/固定评估按全样本目标权重总和归一化，记录global_weighted_target_mean；训练诊断仍为记录加权minibatch均值。完整checkpoint保存实际counts与weights，评估从checkpoint恢复，不根据测试重算。

加权CE改变概率学习的目标，模型softmax输出不自动等于自然人群的后验概率；没有温度校准或叠加成本宣称临床安全。比较argmax accuracy、逐类召回/precision、MAE、低估与跨两级错判。所有身份/两组标签不确定性保持显式来源协议，标签不改、139条不缩减。
