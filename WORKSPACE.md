# 项目工作区入口

更新：2026-10-22。开发目录为 `D:/.Workspace/医工/OPMD-Pred`，本地主线为 `master`。

## 目录用途

| 目录 | 内容 |
|---|---|
| apps/mobile | Vue手机界面、Android工程、APK打包及模拟器启动脚本 |
| apps/server | 电脑服务接口、记录存储、模型推理和解释；原两套后端已合并 |
| src/opmd_pred | 模型结构、数据处理、训练入口和分析工具 |
| scripts/research | 实验训练、旧方法复现、错误分析及结果汇总 |
| data/sysu | 原始医院资料，保持原位 |
| data/zenodo | 原始公开数据，保持原位 |
| data/processed/sysu/legacy | 较早的SYSU整理数据，原SYSU_processed目录 |
| data/processed/sysu/rebuilt-20261009 | 依据原始资料重建的表格、选定图片、报告核查和固定划分 |
| data/processed/zenodo | Zenodo预处理缓存 |
| models/app | APP实际加载的分类器、图片编码器及来源说明 |
| outputs/experiments/sysu | 医院数据各轮实验的权重、配置、日志、预测和负结果 |
| outputs/experiments/zenodo | 后续Zenodo实验输出入口；原始数据仍在data/zenodo |
| outputs/checks/app | APP工程检查、截图及服务日志 |
| outputs/checks/daily | 以日期组织的私有检查证据 |
| outputs/checks/workspace-structure-20261022 | 本次移动清单、文件SHA、路径对应表及迁移检查结果 |
| outputs/figures | 数据分布图、架构图及解释图 |
| docs/daily | 每日说明、结果及失败记录 |
| docs/meetings | 会议原始材料、转写和待办；2026-10-23目录按收到材料日期存放APP参赛讨论 |
| docs/research | 模型方法、数据核查和研究结论；内含pretraining和sysu |
| docs/references | 文献、调研资料及参考代码；内含zenodo和research |
| docs/product | 产品方案、比赛材料、HTML报告、截图及PDF |
| tmp | 保留的旧过程文件、草稿及Git快照；用户已取消删除 |
| .venv | Python环境；optional下存放已有绘图工具依赖 |
| .agents/skills | 项目专用操作规范 |
| .github/modernize | Java升级工具辅助脚本 |
| .git | Git历史和分支信息 |

## APP运行入口

项目上一级目录的“启动手机APP服务.cmd”和“启动安卓APP模拟器.cmd”继续使用，均已指向新服务路径。Python使用本项目 `.venv`。

服务：`apps/server/server.py`；启动脚本：`apps/server/launch.ps1`；本机模型配置：`apps/server/models.local.json`。模型路径采用相对于配置文件的写法：

- `models/app/classifier.pt`：SYSU seed42基线fold0分类器的部署版本。
- `models/app/image_encoder.pt`：Zenodo-DINOv2图片编码器。

对应实验分类器仍保留在 `outputs/experiments/sysu/raw-rebuild-20261009/completed_evidence/runs/seed42/fold0/baseline/refit/classifier.pt`，用于实验复现。更换部署模型应同时更新来源说明及推理接口内的SHA校验。

手机服务运行数据在仓库外 `D:/.Workspace/医工/OPMDMobileDemo`：records.sqlite3为记录，attachments/uploads为附件，distribution为APK分发副本，pairing-token.txt为口令。旧桌面资料 `%LOCALAPPDATA%/OPMDFollowupDemo` 不作为手机服务数据入口。

Android SDK在 `D:/.Workspace/android-sdk-opmd`。当前发布包固定在 `D:/.Workspace/医工/OPMDMobileDemo/distribution/app-debug.apk`，手机下载安装和模拟器均使用此包。项目上一级目录的日期APK是当时的导出文件。此次整理不改变手机服务端口、配对口令、数据库结构或模型内容。

## 文件管理

源码与非私密配置可提交Git。data、models、outputs、docs、tmp及本机模型配置、依赖和构建产物均已忽略；Git不会自动保存这些本地资料。归档或另存备份须先征得用户同意。

本次为按用途移动文件，未创建归档或备份。用户已删除的三个外部归档不能再作为恢复入口。历史实验原始配置和日志内的旧路径保持为当时记录，新位置可查 `outputs/checks/workspace-structure-20261022/path-map.json`；当前源码和文档链接已更新。

服务器上的项目目录与文件未改动；本次只整理本机，不能据此推断远程文件与本机一致。

## 当前脚本和文件保留决定

“启动手机APP服务.cmd”指向apps/server/launch.ps1；模拟器启动脚本调用apps/server/server.py。模型配置位于apps/server/models.local.json，引用models/app。

打包脚本默认使用公网地址http://www.ming56.cn:9020，也可通过-ServiceUrl指定其他地址。构建成功后自动更新固定发布包及其版本说明。模拟器优先读取固定发布包，按APK实际版本和文件校验值判断是否更新，不再固定引用旧日期安装包。

用户已决定不删除先前列出的65个文件。它们保留原位，没有另存副本；旧apps/mobile_service、apps/patient_demo只有缓存，实际服务源码为apps/server。清理清单的执行许可已撤销，旧清理脚本会停止执行。

## 电脑运行数据迁移（2026-10-22）

当前服务数据固定在 `D:/.Workspace/医工/OPMDMobileDemo`（仓库的同级目录）。服务直接启动、模拟器启动、打包发布及工程检查脚本均使用此位置。`older-local-data/` 是从C盘一并移来的4个较早文件，内容与当前数据库不同，不参与运行；未另存备份。48个文件迁移前后SHA一致。本机和公网连接、31处附件引用及公网APK下载均已核对。手机与模拟器内的安卓私有数据仍由Android管理。

C盘原目录中的文件已移出，只剩空目录；工具的自动审批拒绝删除空目录，返回“blocked by policy”。当前脚本不再向该旧目录写入。证据：`outputs/checks/app/runtime-data-move-20261022/`。
