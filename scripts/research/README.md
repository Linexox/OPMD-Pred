# 研究脚本入口

更新于2026年11月7日。研究与APP都在本地master维护；当前研究主线已迁入主目录，APP部署模型仍保持原样。

## 当前研究主线

口腔照片使用Zenodo预训练ResNet-18，细胞图像使用外部脱落细胞预训练ResNet-18；10基因甲基化使用GSE204943预训练网络，其他表格资料使用普通神经网络，四组特征经Transformer输出二分类概率。没有使用TabPFN。模型定义和表格处理复用run_methylation_cell_transfer.py中的Fusion、MethylEncoder、Inputs；该历史核心文件未改动。

入口run_pretraining_mainline.py按自身位置查找仓库，统一读取configs/research/pretraining_mainline.json中的相对路径。数据、模型和结果不随Git提供。

在仓库根目录运行，Windows可把python换成.venv\Scripts\python.exe：

```powershell
# 只检查资料、权重SHA和依赖，不训练
python scripts/research/run_pretraining_mainline.py inspect

# 重算已有结果，恢复第一折模型并核对预测，不训练
python scripts/research/run_pretraining_mainline.py verify

# 查看五折训练命令；移除--dry-run才会运行，包含主方案及ImageNet细胞对照
python scripts/research/run_pretraining_mainline.py fivefold --output outputs/experiments/pretraining-mainline/new-fivefold --device cuda --dry-run

# 查看六折五模型平均集成命令；同样保留ImageNet细胞对照
python scripts/research/run_pretraining_mainline.py sixfold --output outputs/experiments/pretraining-mainline/new-sixfold --device cuda --dry-run

# 查看现有43、44、45三种子的重跑命令，仅运行外部细胞预训练主方案
python scripts/research/run_pretraining_mainline.py multiseed --output outputs/experiments/pretraining-mainline/new-multiseed --device cuda --dry-run
```

训练必须指定尚不存在的outputs/experiments子目录，拒绝覆盖旧结果或向数据目录写结果。默认inspect；上面的dry-run不会创建结果目录，也不启动训练。

五折单模型：种子42五折平均80.54%，最高95.24%，汇总112份留出预测为80.36%。六折五模型集成：种子42总体75.89%；新增43、44、45总体平均73.51%。这些是两种评价方式，详细结果及各类召回见本地docs/reports/研究主线迁移说明-20261107.md。

现有.venv能检查资料、恢复模型和核对结果；本次未安装依赖或更改APP环境。重新生成随机划分还需scikit-learn（已完成GPU实验使用1.9.1，补充清单requirements-pretraining.txt）；绘图还需项目pyproject.toml已声明的matplotlib。本机这两项当前未安装。正式训练使用具备相应依赖的研究环境和CUDA GPU；本次没有启动训练或修改远程GPU目录。

五折对照需要的resnet18-f37072fd.pth已放入主目录outputs/experiments/cache/pretraining-mainline/torch/checkpoints并列入配置SHA核对，使用该缓存时不重复下载。它是通用ImageNet基础权重，与已训练好的口腔照片/细胞编码器分别保存。

### 预处理与预训练代码

- prepare_methylation_cell_transfer.py：按固定开发名单整理10基因、病史/TCT/DNA数值、口腔照片特征和细胞视野图，保留缺失标记及来源校验；原22份测试资料排除。
- audit_oral_cytology.py、prepare_oral_cytology.py：外部细胞图像审计、按患者划分及核周局部图整理。
- train_oral_cytology_encoder.py：外部I—IV细胞分类预训练ResNet-18；均衡抽样以类别、患者、局部图为顺序。
- run_methylation_cell_transfer.py：包含甲基化预训练、共同模型及早期四组合对照；不是当前APP服务入口。
- run_sysu_cell_encoder_transfer.py：原五折下外部细胞编码器与ImageNet对照。
- prepare_cell_sixfold_protocol.py、run_cell_sixfold_ensemble.py：APP历史六折名单与五模型平均对照。
- prepare_cell_random_protocols.py、run_cell_multiseed.py：患者组随机划分及43、44、45新增种子。
- summarize_methylation_cell_transfer.py、summarize_cell_encoder_transfer.py、summarize_cell_sixfold.py、summarize_cell_multiseed.py：原实验的核查和报告生成源码；历史SHA与执行记录保留。

旧run_*_remote.sh和wait_oral_cytology_upload.sh记录当时个人GPU目录及上传启动方式，保留作执行证据，不能直接作为当前仓库启动命令。主目录统一使用上面的Python入口，不依赖旧工作树、服务器个人目录或额外vendor路径。

## APP现行模型、其他保留入口

| 工作 | 主要入口 |
| --- | --- |
| 图片模型训练 | train_resnet18_zenodo.py、train_zenodo_classifier.py、run_zenodo_supplement.py |
| APP部署的五个完整模型训练 | run_sysu_full_model_moe.py、run_sysu_binary_fusion.py、build_sysu_table_branch.py |
| 五模型训练及验证入口 | run_full_model_moe_study.sh、check_full_model_moe.py、evaluate_full_model_moe_inference.py、verify_full_model_moe.py |
| 训练Dropout取值对照：复用表格参考，五模型各占20% | run_sysu_dropout_mean.py；配置sysu_moe/dropout_zero_sixfold.json |
| 从患者PDF提取TCT/DNA视野图和细胞核小图 | extract_tct_dna_cells.py；依赖PyMuPDF、Pillow、NumPy；结果写入data/processed/sysu/tct-dna-cell-images |
| 按原始样本文件夹汇总视野图并预选 | prepare_folder_cell_images.py；等级1/2/3为主清单；结果写入data/processed/sysu/cell-image-samples |
| 共用模块 | sysu_binary_common.py、sysu_tabular_common.py、zenodo_supplement_common.py |

部署使用apps/server/binary_runtime中的固定推理实现，不从研究脚本目录动态挑选训练模型。参数和运行证据分别在models与outputs，说明在docs，用户成品在仓库上级交付成果。
