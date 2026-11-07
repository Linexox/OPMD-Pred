# OPMD-Pred

口腔黏膜病损辅助分析项目。

当前研究主线：口腔照片编码器＋GSE204943甲基化预训练＋外部脱落细胞图像预训练＋表格网络＋Transformer二分类。

- 研究入口与复现命令：[scripts/research/README.md](scripts/research/README.md)。
- 当前模型配置：[configs/research/pretraining_mainline.json](configs/research/pretraining_mainline.json)。
- APP部署仍使用原ResNet-18＋TabPFN＋Transformer五模型集成；部署方法见[DEPLOYMENT.md](DEPLOYMENT.md)。
- 本地资料位置与完整研究说明见[WORKSPACE.md](WORKSPACE.md)；私有数据和权重不随Git提供。
