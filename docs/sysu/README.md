# SYSU阶段入口

目标：正常/轻度、中度、重度三分类；输入为口腔图片、吸烟/饮酒/槟榔史、TCT、DNA异常细胞比例、甲基化。年龄/性别以及它们的mask/token全部禁止作为模型输入。

## 当前阶段

预训练已完成：[Zenodo归档和清理状态](D:/.Workspace/医工/OPMD-Pred/docs/pretraining/archive-status-2026-10-08.md)。最佳SAM-DINO编码器供本阶段使用，其Zenodo75.40%不等于SYSU效果。

SYSU处于数据核对、方法研究和工程基线准备阶段，没有新的可信训练指标。139条记录（66/36/37），87张图片，五模态全齐45条、全缺5条。当前没有经医院确认的患者ID；2组同图跨类别需核对，旧随机划分存在重复泄漏。

## 研究交付

- [工作计划](D:/.Workspace/医工/OPMD-Pred/docs/sysu/work-plan-2026-10-08.md)：阶段、分工、评价协议和完成标准。
- [数据与实现审计](D:/.Workspace/医工/OPMD-Pred/docs/sysu/data-audit-2026-10-08.md)：真实缺失、来源语义、重复与现有代码问题；聚合JSON与复现脚本同目录。
- [多模态/小样本文献矩阵](D:/.Workspace/医工/OPMD-Pred/docs/sysu/literature-missing-small-sample-2026-10-08.md)：用户三篇参考与另九篇原始工作，直接证据和适配边界。
- [有序分类、可信预测、解释](D:/.Workspace/医工/OPMD-Pred/docs/sysu/ordinal-trust-explanation-2026-10-08.md)：有序概率、独立成本决策、校准、conformal、IG/ViT解释。
- [方法主线与基础推导](D:/.Workspace/医工/OPMD-Pred/docs/sysu/method-proposal-2026-10-08.md)：缺失导致的有序概率差、动作风险差与解释连接，明确绝对差一致性的反例和修订。
- [独立方法复审](D:/.Workspace/医工/OPMD-Pred/docs/sysu/method-review-2026-10-08.md)：已有近似工作、数学边界与可证伪路线。

文献调研是定向实现研究，不是系统综述。基础恒等式不宣称新定理；框架组合不宣称首创。比赛创新需通过患者级真实缺失/模拟遮蔽对照、低估距离指标及解释忠实性证明。

## 待医院补齐的材料

1. `sample_id,patient_id`的匿名映射，复诊/多图仍用相同patient_id分组；就诊ID和检查/病理时间有则一并提供。姓名哈希、Excel行号不能自行认定为可信患者身份。
2. [本地重复图核对表](D:/.Workspace/医工/OPMD-Pred/tmp/sysu-private/duplicate-image-review-2026-10-08.csv)中4条记录的图片绑定、患者归属、真实标签。该定位表不公开发布；原始数据不自动修正。
3. 明确检测是否在预测时点可用、等级金标准。代价矩阵与重度召回目标目前为开发集研究参数，若导师有具体要求后续记录其版本。

收到可靠映射和重复处理后再冻结患者开发/测试划分。模型资料不足拒判、三分类准确率、可预测覆盖率和转人工比例分别报告，不能静默剔除无输入患者后宣称所有患者75%达标。
