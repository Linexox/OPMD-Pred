## 目录

- `src/opmd_pred/data`：SYSU/Zenodo 数据集读取、数值归一化、图像变换和数据划分。
- `src/opmd_pred/model`：ViT 图像编码器、多模态融合、missing token 和序数回归头。
- `src/opmd_pred/scripts`：Zenodo 图像预训练、SYSU 多模态对比预训练和目标训练。
- `scripts`：一次性数据整理脚本。
- `outputs`：运行时生成的 checkpoint 和 `train.log`。

## 环境

```bash
uv sync
```

## 数据整理

当前 SYSU 数据位于 `data/SYSU_processed`。如果需要从原始 XLSX 重新生成：

```bash
uv run python scripts/prepare_sysu.py --output data/SYSU_reviewed_new
```

新准备脚本显式读取mild/moderate/severe工作表，输出12个非人口学字段，只复制文件名主干严格为`病损`的图片。要求新输出目录，发现图像目录绑定歧义会停止；现有医院数据本轮未重写。图片/患者/标签归属仍需医院核对，脚本不能自动确认身份。

## SYSU新阶段

使用`codex/sysu-foundation`的患者审核协议入口。原始数据保留年龄/性别用于追溯，但数据batch、模型和解释不再使用人口学模态。默认五模态为image/history/tct/dna/methylation。

当前139条记录没有可信患者ID，且有2组同图跨类别，必须先核对后训练。详见[阶段入口](docs/sysu/README.md)、[实现与训练指令](docs/sysu/foundation-implementation.md)。

```bash
PYTHONPATH=src uv run python -m opmd_pred.scripts.build_sysu_manifest \
  --csv data/SYSU_reviewed/sysu_samples.csv --images data/SYSU_reviewed \
  --groups data/SYSU_reviewed/anonymous_patient_groups.csv \
  --output outputs/sysu_protocol/split_manifest.json

PYTHONPATH=src uv run python -m opmd_pred.scripts.train_sysu \
  --csv data/SYSU_reviewed/sysu_samples.csv --images data/SYSU_reviewed \
  --split-manifest outputs/sysu_protocol/split_manifest.json \
  --image-encoder dinov2_b14 --image-checkpoint /path/to/dinov2_last4_sam_epoch16_encoder.pt \
  --head softmax --output outputs/sysu_foundation_seed42
```

`--groups`需医院提供`sample_id,patient_id`匿名映射，不以Excel行号或姓名自行生成可信身份。默认冻结SAM-DINO；DINO为B/14、518保比padding、1536维CLS+patch均值，不能直接换进旧B/16结构。新训练只train/val，固定测试使用独立`evaluate_sysu`入口。GPU启动前检查其他用户占用。

## 旧探索性训练

下述样本级拆分入口保留用于历史对照，不能用于新阶段的患者独立结果。含人口学的旧融合checkpoint不兼容本分支；旧对比脚本仅做模态投影对齐，不代表复现融合视图对比方法。

先按患者划分 Zenodo 图像并训练 ViT：

```bash
uv run python -m opmd_pred.scripts.pretrain_vit_zenodo
```

再进行 SYSU 跨模态对比预训练：

```bash
uv run python -m opmd_pred.scripts.pretrain_multimodal \
  --image-checkpoint outputs/zenodo_vit/checkpoint.pt
```

最后进行序数目标训练：

```bash
uv run python -m opmd_pred.scripts.train_multimodal \
  --image-checkpoint outputs/zenodo_vit/checkpoint.pt \
  --fusion-checkpoint outputs/multimodal_contrastive/checkpoint.pt
```

训练脚本默认冻结 ViT，仅训练融合层和序数回归头。使用 `--unfreeze-image-blocks 1` 或 `2` 可解冻最后的 ViT block。

## 缺失值和缺失标志

数值 0 和没有检测不是同一件事：

- `0` 表示检测完成且结果为零，或明确回答“否”；
- 缺失表示没有检测、没有记录或结果无效。

数据集返回字段值和对应 mask。模型只在整个模态缺失时替换为可学习的 missing token；模态内部部分字段缺失则由字段级 mask 传入编码器。对比预训练只对两个真实存在的模态计算正样本，不把 missing token 当成观测结果参与对比。
