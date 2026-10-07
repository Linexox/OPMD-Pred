# 预训练结束与远端清理

2026-10-08，已完成用户授权的归档清理。服务器ubuntu-System-Product-Name，所有目标均在`/home/zjkj/24334067`；GitHub实际认证为Linexox。

## 保留成果

最终SAM-DINOv2-B/14最后4层微调，best16，固定患者划分test236/313=75.40%、macro-F1 .780181、balanced accuracy .797178。仅整图和Category监督，无病灶标注或其他模态。SYSU效果尚未评估。

- 完整模型：`OPMD-Pred-Zenodo-DINOv2-Last4-SAM/outputs/zenodo_dinov2_b14_518_last4_sam_seed42/checkpoint.pt`，SHA256 `2ea7daf27541571a6c2163c28d861fd3cb53998bc881eaba0288432424b9a838`。
- 无头编码器：`OPMD-Pred-ViT-Pretrain/outputs/zenodo_transfer/dinov2_last4_sam_epoch16_encoder.pt`，SHA256 `58a3fcc724155c8ecfc7bf0e955b2bf022f0e2aa4bfed06a51a5b99c850a4a65`。
- 上述路径均相对个人根目录；删除后重新计算两文件SHA，完全一致。原数据、venv、用户主repo/Yanwei、ViT-Pretrain及最佳SAM/协议worktree保留。

## 归档与删除证据

远端统一归档：`/home/zjkj/24334067/archives/zenodo-pretraining-20261008`。

- `experiment-code.bundle`，22个方法/协议ref的代码历史，253508字节，bundle verify成功，SHA256 `a74a0eaa5d7a5bec14d5deec51037c27a0e342c7a73cb3ca03295199da839caf`。
- `metrics-and-method-docs.tar.gz`，280份轻量指标/方法/日志文件，6021575字节，归档成员核验成功，SHA256 `96b3e4451893852c2b656213beb7f2b225028bc51a076be30598058432830801`。
- 删除20个干净的实验worktree、20个对应远程分支及224/384派生缓存，目标绝对路径逐个限定于个人目录，确认无运行进程引用，删除前核对HEAD与归档。通过git worktree remove管理，未直接改.git。
- 释放文件逻辑大小约 **6.97GB（6.50GiB）**；此数由删除清单文件大小汇总，不是文件系统物理占用测量。
- 远端仅保留`codex/zenodo-dinov2-last4-sam`与`codex/zenodo-protocol`研究分支。本地旧方法refs仍保留作为参考；没有推送GitHub或运行GC。
- [删除清单与核验](D:/.Workspace/医工/OPMD-Pred/docs/pretraining/cleanup-result-2026-10-08.json)与[删除前计划](D:/.Workspace/医工/OPMD-Pred/docs/pretraining/cleanup-plan-2026-10-08.json)已下载本地。

## 后续读取

统一入口为[预训练总结](D:/.Workspace/医工/OPMD-Pred/docs/pretraining/zenodo-summary.md)，辅以结果索引、文献结论、SAM配方及encoder加载说明。旧结果索引中的低效worktree输出路径现已删除，证据从归档包按原目录名读取；代码可以从bundle恢复，低效模型的大权重和特征缓存未保留。原临时目录记录是历史文档，当前清理状态以本文件为准。
