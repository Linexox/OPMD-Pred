# SYSU研究选项集成入口

`codex/sysu-research`集成三个独立研究方向；工程基线仍保留在`codex/sysu-foundation`，各方向分支分别为`codex/sysu-missing-consistency`、`codex/sysu-cost-ordinal`、`codex/sysu-explainability`。

这里只统一源码兼容性，没有把未验证方法默认堆叠成最终模型。默认softmax、subset_weight=0、consistency_weight=0、ordinal_distance_weight=0、不设成本矩阵；真实训练尚未启动。没有任何SYSU75%或召回增益证据。

- `train_sysu --head softmax|ordinal|corn`统一合法三类概率与概率损失。
- `--ordinal-distance-weight`显式启用平方累计目标损失；值为0时关闭。
- `--subset-weight`及`--consistency-weight`显式启用观测O/非空子集S，CDF平方或teacher→student KL；默认关闭。O/S共享一次已增强图像编码，teacher仅来自真实观测，不伪造检测值。
- 集成后O、S各自监督为L_probability+λL_squaredCDF，再按a/b合成观测、子集与一致性项。`component_losses`中的supervised含各自距离项；各独立分支零权重消融仍可复现，尚未实测组合。
- CORN验证与固定评估loss按全条件任务项分母汇总；训练仍按minibatch诊断，指标明确标记不同聚合方式。默认验证/测试单次实际观测，checkpoint选择不变。
- `predict_sysu`离线HTML与`explain_sysu_image`IG在此可以读取三种概率头以及只改变训练视图的checkpoint。原解释分支自身仅基于softmax/ordinal，不能独自读取CORN；集成入口统一该兼容性。

原患者/hash协议、年龄性别禁用、train-only词表、资料不足拒判与全分配分母保留。输入需要样本归属映射；这不是要求医院新建身份数据库，只需确认现有记录属于谁并用项目编号分组。存在姓名不代表已确认患者身份；同名信息可辅助生成候选分组，不能静默当作经过审核的patient_id。

当前两组同图跨标签尚待核对，任何正式训练/评价仍必须解决归属和金标准。探索性训练和正式患者独立评价可以分别定义协议，不能把前者作为后者成绩。

静态检查：源码AST与差异检查，不运行模型、数据协议、梯度、病例HTML渲染或IG。真实数据身份问题和实际运行正确性不能由merge成功证明。使用各实现说明中的未来指令前，应先完成数据核对与设备资源检查。
