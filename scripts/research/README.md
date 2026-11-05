# 研究脚本入口

2026年11月5日按用户审核结果精简。以下是当前保留的训练与核对入口，模块之间存在直接导入。

| 工作 | 主要入口 |
| --- | --- |
| 图片模型训练 | train_resnet18_zenodo.py、train_zenodo_classifier.py、run_zenodo_supplement.py |
| 当前五个完整模型训练 | run_sysu_full_model_moe.py、run_sysu_binary_fusion.py、build_sysu_table_branch.py |
| 五模型训练及验证入口 | run_full_model_moe_study.sh、check_full_model_moe.py、evaluate_full_model_moe_inference.py、verify_full_model_moe.py |
| 共用模块 | sysu_binary_common.py、sysu_tabular_common.py、zenodo_supplement_common.py |

部署使用apps/server/binary_runtime中的固定推理实现，不从研究脚本目录动态挑选训练模型。参数和运行证据分别在models与outputs，说明在docs，用户成品在仓库上级交付成果。
