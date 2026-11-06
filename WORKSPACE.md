# 当前工作区与运行位置

更新：2026年11月6日。本文件只描述当前状态；旧过程见docs/worklog，不再在这里叠加历史“当前状态”。

## 当前入口

- 唯一开发仓库：D:/.Workspace/医工/OPMD-Pred（实际路径见下表），分支master。模型与APP在同一仓库，分别维护。
- 人可直接使用的产物：仓库上级的交付成果。最新安装包为APP/口腔随访-1.3.9.apk。
- 线上APP使用五个完整ResNet-18＋TabPFN＋Transformer二分类模型，各占20%汇总概率；分类网络训练Dropout为0.1，不使用Router。结果页参考点来自这套结构的六折留出分数。
- Windows本机launch.ps1仍启动旧三分类开发服务；它没有安装线上全部二分类依赖，不应与线上成绩混称。

| 用途 | 当前路径 |
| --- | --- |
| 开发代码 | D:/.Workspace/医工/OPMD-Pred |
| APP前端、Android工程 | apps/mobile |
| API、记录管理与部署启动 | apps/server |
| 固定版本二分类推理实现 | apps/server/binary_runtime；不含模型或患者附件 |
| 模型结构与训练组件 | src/opmd_pred |
| 训练和复现脚本 | scripts/research；保留当前五模型及其图片编码器训练入口 |
| 当前五模型二分类私有模型包 | models/app/binary-mean；manifest.json列出五个成员及固定20%权重 |
| 切换前单模型二分类包 | models/app/binary；保留原位置用于恢复，不是当前主模型 |
| 旧三分类权重 | models/app/classifier.pt、image_encoder.pt；本机models.local.json仍引用它们 |
| 原始数据接收目录 | ../DATASETS；当前data/sysu、data/zenodo及历史数据暂保留，未做删除去重 |
| 整理后的研究数据 | data/processed/sysu；原始病理、图片和就诊关联仍需按研究记录核对 |
| 实验权重、预测、配置及失败结果 | outputs/experiments |
| 工程检查与文件迁移记录 | outputs/checks；本次共享整理为workspace-review-20261105 |
| 可直接阅读的报告与比赛材料 | ../交付成果 |
| AI工作流水和每日尝试 | docs/worklog |
| 论文原文和第三方参考源码 | ../papers/原文、../papers/参考代码 |
| 历史源码与获奖原件 | ../历史项目；未经确认不删除 |
| Windows运行库、附件、口令、固定APK副本 | ../运行数据 |
| Gradle、npm等构建缓存 | ../开发缓存；android/debug.keystore是签名密钥，不能随缓存清理 |

## 启动与安装

项目上一级“启动安卓APP模拟器.cmd”指向apps/mobile/launch-emulator.ps1，默认连接公网模型服务；使用-LocalService才准备Windows开发后端。“启动手机APP服务.cmd”指向apps/server/launch.ps1。

APK构建入口apps/mobile/build-android.ps1；固定分发副本为../运行数据/distribution/app-debug.apk。模拟器优先读取此副本，再读取Android工程构建输出。构建缓存、附件和新运行数据均放D盘，不默认写C盘用户目录。Android SDK仍在D:/.Workspace/android-sdk-opmd，AVD位置本次不变。

## Linux线上服务

ssh lnx连接个人PC，部署根目录/home/nexox/workspace/OPMD_Pred。后台服务opmd-pred.service，入口apps/server/launch-linux.sh，runtime/model-mode.txt为binary时调用binary_linux_service.py。

当前正式模型清单为models/app/binary-mean/manifest.json，包含五份独立融合权重、25份TabPFN参考、共享编码器和基础权重。五个成员分别使用原五分组中的四组训练，覆盖117份开发资料，不按最好测试折选权重。集成实现为apps/server/binary_runtime/mean_ensemble_predictor.py，复用已固定的单模型计算。模型内表格参考包含训练上下文，整个包仅限服务端，不进APK或公开下载。

runtime/binary-model-manifest.txt选择活动清单，当前为models/app/binary-mean/manifest.json；需要恢复时可改为models/app/binary/manifest.json再重启本项目服务。默认启动也采用五模型清单。五模型对照的六折平均准确率75.62%是结构评价，最终部署组合使用更多开发训练资料，不把训练资料上的成绩作为独立验证。

Linux运行数据库、附件、口令、下载包仍在该项目runtime，未迁移用户记录；Python环境和已有隔离依赖目录保留。公网APP入口www.ming56.cn:9020；9018是SSH，9024是NPC连接端口。此次没有改NPS/NPC配置或公网地址。

## 管理约定

组员共享与最小部署见[DEPLOYMENT.md](DEPLOYMENT.md)。它随Git提供，分别说明连接现有服务与独立部署；本机绝对路径仅是当前维护电脑的位置。

源码与非私密配置可由Git管理；docs、data、models、outputs、本地配置和凭据目前被忽略。Git克隆不会带来部署模型或口令；本次整理不等于完成资料备份。

成品只有交付成果中的一个编辑来源；固定APK分发副本因服务需要保留。文件迁移清单位于outputs/checks/workspace-organization-20261103/moves.json。历史实验内部命令和校验记录保留当时路径，必要时对照迁移清单，不批量改写原始实验证据。

此前经用户确认的248个源码文件已删除，执行范围见../交付成果/代码精简审阅清单.md。本次Git对象整理保留分支及提交引用；源码残留缓存和空目录的删除被自动审批拒绝，未执行，具体目录在outputs/checks/workspace-review-20261105/source-cache-cleanup.json。Zenodo、原始数据、部署模型、实验结果和运行数据库保持原位置。
