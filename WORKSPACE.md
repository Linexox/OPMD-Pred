# 项目工作区入口

更新：2026-10-13。当前本地主线为 **master**，统一开发目录为 **D:/.Workspace/医工/OPMD-Pred**。

## 从哪里开始

| 目录 | 内容 |
|---|---|
| apps/mobile | 手机APP界面及Android工程 |
| apps/mobile_service | 电脑上的API服务 |
| apps/patient_demo | 存储、研究推理接口和桌面Demo |
| scripts/research | SYSU研究脚本，包括原始数据重建、增强、BSDA与距离损失 |
| src/opmd_pred | 模型与数据处理库 |

启动手机服务或桌面Demo，使用项目上一级目录已有的对应CMD入口。启动脚本从自身位置确定项目目录，Python使用本项目.venv；模型配置在apps/patient_demo/models.local.json，保持本地忽略。

## 文件管理

- data/：原始/处理数据；原始SYSU资料保持原位。
- outputs/：实验输出及模型，不因分支清理而删除。
- docs/daily/YYYY-MM-DD/：工作结论、来源和失败记录；docs/已取消Git跟踪并忽略。
- tmp/sysu-private/、tmp/product-private/、tmp/daily-private/：病例、模型及私有证据，不能按临时文件批量清空。
- 本次备份：D:/.Workspace/医工/archive/main-integration-2026-10-13/。含Git历史、迁移前文件、冲突版本和SHA清单。

docs/与tmp/通过日期归档备份，不会随Git提交自动保存。只提交源码和非私密配置示例，不提交病例、模型、配对口令、node_modules或构建缓存。

## 分支与服务器

master承接原codex/sysu-proxy-balanced的APP及SYSU代码，并合并两边本地修改。ViT-Pretrain与原balanced分支保留为迁移前参考。

服务器 /home/zjkj/24334067/OPMD-Pred-SYSU-Proxy-Balanced 仍是较早的训练版本；本次未同步远端或推送GitHub。不能根据同名分支推断文件相同。

Zenodo的SAM最终方法、原DINO、B0和协议分支继续用于复现。旧方法的代码和结果可查2026-10-13清理账本及archive标签。

旧Codex工作区在迁移核验后申请归档；若接口仍返回queued，保留其目录并在当日日志记录，不继续在旧目录开发。恢复细节见docs/daily/2026-10-13/main-integration.md。
