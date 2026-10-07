# Oral-image literature summary

日期：2026-10-07。仅汇总已核对原始论文；不查看本次 test 或提出新方法。

| 研究与原文 | 任务、输入与拆分 | 主结果及可比性 |
|---|---|---|
| **Devindi et al., IEEE Access 2024** [原文](https://doi.org/10.1109/ACCESS.2024.3454338) | 同一 Zenodo，2,271 口腔照；Benign vs OPMD+OCA 二分类；全图 CNN，患者级 60/20/20。 | 全图 accuracy：MobileNetV3 .71、ResNet50 .72、DenseNet121 .71、HRNet-W18-C .73。ROI .75、ROI+临床 metadata .81 不符合本项目输入，且不是四分类。 |
| **Ormeño-Arriagada et al., Cancers 2026** [原文](https://doi.org/10.3390/cancers18182926) | 同一 Zenodo，Healthy/Benign/OPMD/OCA；完整照片输入，EfficientNet-B0，resize+center-crop 480，无手工 ROI；分层五折按图像切分，未按患者。 | +DA accuracy .706±.022、macro-F1 .669±.024；RAW .693±.011、.644±.018。最贴近同数据四分类，但不代表患者隔离结果。文中 binary endpoint .799 是 sensitivity，不是 accuracy。 |
| **Redondo et al., BSPC 2026** [原文 PDF](https://oai.e-spacio.uned.es/server/api/core/bitstreams/d4df6f81-30ce-4d69-aa53-f133a8ab3763/content) | 不同队列：1,013 名西班牙患者、3,246 张 RGB 口腔照；Healthy/Benign/OPMD/OC，未标 ROI；按图像 70/20/10（2,272/649/325），未报告 patient grouping。 | ConvNeXt V2-Tiny accuracy：224 .8276、384 .8553、512 .8307。四类整图任务相关，但不是患者隔离；摘要称 354、正文/表称 384、另处称 254，384 数值应带此注。 |
| **Martin et al., MICCAI ODIN 2026** [官方论文](https://papers.miccai.org/miccai-2026-sat/ODIN_004.html) | 931 名患者，每人一张正面口内照，periodontitis 三分类；DINOv2 ViT-B/14 last-four 微调，仅图像和类别标签；五折，每人一张故不会跨折。 | macro-F1 .683±.034、QWK .704±.047（不是 accuracy）。支持患者独立口内照片微调，但病种/标签数/指标不同；MGI 派生的另一任务不作纯类别证据。 |
| **Alhaj et al., Scientific Reports 2026** [原文](https://www.nature.com/articles/s41598-026-49376-4) | 乳腺超声 BUSI 三类、BUS-UCLM 二类；ImageNet 预训练 CNN，用整图；随机 70/30 图像 train/validation，未报告患者分组。 | 17 CNN 上 SAM+Adam 平均 validation accuracy 提升 +2.30pp / +2.27pp。仅是医学影像优化器证据，不能外推为口腔/DINO 性能。 |

## 结论

已核对原论文中，未发现同一 Zenodo 四类、整张口腔照片、仅类别监督、患者隔离评估且 accuracy >.75 的直接证据。Cancers 结果最贴题但按图切分；Devindi 按患者拆分但为二分类；Redondo 的四类高分来自不同队列且患者分组未明；ODIN 有患者独立口内图像但病种与指标不同。文献缺口不是性能上限，也不能预测项目的患者隔离结果。
