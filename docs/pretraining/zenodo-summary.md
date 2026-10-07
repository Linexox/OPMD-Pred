# Zenodo 图片预训练实验总结

日期：2026-10-07。最终方案为DINOv2-B/14最后4个block微调 + SAM(AdamW)。固定第16轮模型的测试准确率为 **75.40%（236/313）**，macro-F1为 **0.7802**，balanced accuracy为 **0.7972**。已达到>75%的最低目标，未达到80%。训练、评估与编码器导出均已完成，不再追加实验。

## 目标与数据协议

学习能够迁移到 SYSU 下游任务的口腔图像编码器。Zenodo 阶段做 Healthy / Benign / OPMD / OCA 四分类；输入仅完整 RGB 照片，监督仅 Category。患者 ID 只用于分组，不使用病灶标注、Clinical Diagnosis、人口学资料或 SYSU 数据。

固定患者划分 seed=42。移除训练侧5张与验证侧完全重复的源图，保留 train/validation/test 为2366/315/313张、571/71/71名患者；集合间患者与源JPEG SHA均无交集。验证准确率选择完整checkpoint，macro-F1只用于准确率相同的情况。测试集不参与训练、epoch选择、分类阈值或集成权重拟合。

多个候选已在同一测试集合上做探索性比较，因此这些结果是内部研究benchmark，并非新的独立临床验证。测试OCA只有6张。>75%准确率要求313张中至少235张正确。

## 已核验的关键结果

| 方法 | 最佳验证准确率 | 测试准确率 | 测试macro-F1 | 说明 |
|---|---:|---:|---:|---|
| EfficientNet-B0 RAW480 | 66.03% | 71.25%（223/313） | 0.7136 | 小模型基线；全层更新 |
| DINOv2 last4，518 CLS+patch mean | 70.48% | 72.52%（227/313） | 0.7583 | 仅最后4块和分类头更新 |
| 同一DINO，固定训练先验校正 | 71.11% | 73.48%（230/313） | 0.6867 | 仅推理bias改变；编码器完全相同 |
| B0+DINO固定50:50概率集成 | 73.02% | 未评估 | — | 双编码器2816维；按用户要求止于验证 |
| 最后一组：DINO last4 + SAM | 72.70% | **75.40%（236/313）** | **0.7802** | best16；固定rho=.05，其余配方保持 |

完整实验表、分支SHA和输出位置见[结果核对表](D:/.Workspace/医工/OPMD-Pred/docs/pretraining/zenodo-results.md)。其余已尝试方向包括ResNet/DenseNet、更大CNN、ConvNeXt、WS-DAN、Attention MIL、分阶段训练、少数类增强、多seed集成、L2-SP、加权CE、冻结特征LR/SVM；已测试的这些候选均未超过最终SAM方案。未评估test的候选不能声称已知其测试效果。

先验校正的总体准确率提高，但macro-F1由.7583降到.6867、平均类别召回由.7772降到.6598；OCA召回由6/6降到3/6。它没有给编码器增加任何学习，不能把这项准确率增益解释成更好的迁移特征。

## 为什么最后选择SAM

原DINO-last4末轮训练准确率98.27%，最佳验证70.48%，存在明显训练/验证差距。SAM在同一个已增强batch上计算原参数与临时扰动参数两次梯度，恢复原参数后执行一次AdamW；只改变训练优化步骤，没有新增输入或监督。固定rho=.05、末4块+head、518整图letterbox、1536维CLS+patch均值、AdamW encoder1e-5/head1e-3、wd1e-4、batch8、20轮/patience5、seed42。最终LayerNorm和前8块冻结。

root已审阅实现，并在实际公开DINO模型的单张518合成输入上核验：58个参数tensor只在末4块+head变化；冻结参数精确保持；AdamW状态每batch只推进1步；特征[1,1536]和分类[1,4]有限。真实20轮训练结束后，root独立核验最佳epoch16：验证229/315=.726984、macro-F1=.714004、平均类别召回=.718665，完整176个state tensor中的174个encoder tensor与独立导出逐一一致；相对公开权重56个tensor变化且仅在末4块，118个保持精确相同。最佳checkpointSHA为2ea7daf27541571a6c2163c28d861fd3cb53998bc881eaba0288432424b9a838。独立验证重放的全部指标、loss和混淆矩阵与第16轮一致；随后对同一checkpoint运行一次固定raw test，未追加先验校正、TTA或阈值调整。

最终测试比原DINO-last4多正确9张（+2.88个百分点），准确率、macro-F1及平均类别召回均更高。相对此前最高准确率的先验校正方案多正确6张（+1.92个百分点）。这是该固定划分的观察结果，未做多seed重复，不能宣称已证实统计显著提升。

| 测试类别 | 样本数 | 召回率 |
|---|---:|---:|
| Benign | 88 | 52.27%（46/88） |
| Healthy | 74 | 81.08%（60/74） |
| OPMD | 145 | 85.52%（124/145） |
| OCA | 6 | 100%（6/6） |

Benign识别仍弱，34张被预测为OPMD；OCA的6/6不能代表稳定的临床识别能力。测试混淆矩阵按上述类别顺序为`[[46,7,34,1],[7,60,7,0],[9,12,124,0],[0,0,0,6]]`。root独立复算得到accuracy=236/313、macro-F1=.780180995、balanced accuracy=.797177624，并核验测试与验证使用同一模型SHA和manifest SHA。评估进程均已结束，最后GPU检查无compute任务。

SAM的原始依据是[Foret等人的论文](https://arxiv.org/abs/2010.01412)。[2026医学超声迁移学习研究](https://www.nature.com/articles/s41598-026-49376-4)为优化器迁移提供动机，但按图片随机划分，不能预测患者隔离的Zenodo四分类成绩；本实现是适配，未宣称完整复现其CNN配方。

## 论文结论

最接近的同Zenodo四分类论文报告B0增强accuracy .706±.022，采用图像级五折；其输入resize+center-crop480，而本项目RAW缓存全图square stretch480，属于适配。[Cancers原文](https://doi.org/10.3390/cancers18182926)

同数据患者级论文为二分类，全图CNN约.71–.73；ROI/临床资料带来的.75/.81不符合本次输入要求。[IEEE Access原文](https://doi.org/10.1109/ACCESS.2024.3454338) DINO末层微调有患者独立口内照片研究依据，但病种、类别数量和macro-F1指标不同。[MICCAI ODIN原文](https://papers.miccai.org/miccai-2026-sat/ODIN_004.html)

目前未找到同时满足同Zenodo四类、完整图片、仅类别监督、患者隔离且accuracy>.75的直接文献证据。这不代表性能上限。详细可比性见[文献总结](D:/.Workspace/医工/OPMD-Pred/docs/pretraining/zenodo-literature.md)。

## 迁移与复现注意事项

原主分支保存encoder供迁移是合理的；评估最佳Zenodo模型时必须同时恢复其对应分类头，不能把最佳encoder与最后一轮head混用。本次候选都保存完整模型和独立headless encoder。

DINO模型为`timm vit_base_patch14_dinov2.lvd142m`，1536维特征必须使用最终LayerNorm后的CLS拼接全部patch token均值；raw timm forward只有768维CLS。当前SYSU代码使用torchvision ViT-B/16和224输入，不能直接把DINO state_dict载入该结构。后续迁移需要对应DINO结构、518等比例letterbox+ImageNet normalization，并将image_projection输入维度接为1536；本轮未修改SYSU训练流程，也未测量SYSU效果。

分支在本地及个人服务器worktree保存，未推送GitHub。所有远端写入位于`/home/zjkj/24334067`，每次GPU启动前均检查无其他compute任务。最后SAM训练的实际进程2924501已自然结束；发生于训练前的启动参数错误已保留记录，成功训练没有重复启动。

## 最终交付与推荐

推荐将本次SAM第16轮编码器作为下一步SYSU迁移的候选，丢弃Zenodo四分类头。Zenodo准确率提高不等于SYSU特征质量已得到验证。

- 方法分支：`codex/zenodo-dinov2-last4-sam`，最新提交`a3324c8619ae830c66ab3fc87a059b9b941bd0b7`；22个方法/协议分支已同步本地。
- 远端方法worktree：`/home/zjkj/24334067/OPMD-Pred-Zenodo-DINOv2-Last4-SAM`。
- 完整模型与测试JSON：该worktree的`outputs/zenodo_dinov2_b14_518_last4_sam_seed42/{checkpoint.pt,test_evaluation.json}`。完整模型供四分类复现；训练日志中的test_metrics为空，独立测试JSON是最终测试依据。
- 推荐的无分类头文件：`/home/zjkj/24334067/OPMD-Pred-ViT-Pretrain/outputs/zenodo_transfer/dinov2_last4_sam_epoch16_encoder.pt`。
- 编码器SHA-256：`58a3fcc724155c8ecfc7bf0e955b2bf022f0e2aa4bfed06a51a5b99c850a4a65`。
- root重载后确认174个tensor逐一等于完整模型的encoder，不含分类头/完整model state/患者与标签。严格加载对应DINO结构后，单张518合成输入的1536维特征与完整模型encode精确一致。
- [编码器加载说明](D:/.Workspace/医工/OPMD-Pred/docs/pretraining/encoder-loading.md)提供CLS+patch mean实现；[方法与原始训练指令](D:/.Workspace/医工/OPMD-Pred/docs/pretraining/sam-recipe.md)保存具体环境、参数和结果。原命令输出目录已有成果，再训练需换新的output目录并先检查GPU资源。

原DINO best3编码器另保留为`dinov2_last4_epoch3_encoder.pt`供基线复现；先验校正没有改变该编码器。此次工作在Zenodo阶段达到最低目标，80%尚未达到，SYSU迁移效果尚未测量。


## 2026-10-08归档更新

低效实验远端worktree与分支已删除。旧输出路径现为归档内的证据索引，详见[清理与保留状态](D:/.Workspace/医工/OPMD-Pred/docs/pretraining/archive-status-2026-10-08.md)。最佳SAM完整模型与纯编码器仍保留且SHA核验一致。
