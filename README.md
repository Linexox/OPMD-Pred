## 目录

- `src/data`：SYSU/Zenodo 数据集读取、数值归一化、图像变换和数据划分。
- `src/model`：ViT 图像编码器、多模态融合、missing token 和序数回归头。
- `src/scripts`：Zenodo 图像预训练、SYSU 多模态对比预训练和目标训练。
- `scripts`：一次性数据整理脚本。
- `outputs`：运行时生成的 checkpoint 和 `train.log`。

## 环境

```bash
uv sync
```

## 数据整理

当前 SYSU 数据位于 `data/SYSU_processed`。如果需要从原始 XLSX 重新生成：

```bash
uv run python scripts/prepare_sysu.py
```

脚本只写出指定的 14 个字段，并且只复制文件名主干严格为 `病损` 的图片。

## 训练

先按患者划分 Zenodo 图像并训练 ViT：

```bash
uv run python -m src.scripts.pretrain_vit_zenodo
```

再进行 SYSU 跨模态对比预训练：

```bash
uv run python -m src.scripts.pretrain_multimodal \
  --image-checkpoint outputs/zenodo_vit/checkpoint.pt
```

最后进行序数目标训练：

```bash
uv run python -m src.scripts.train_multimodal \
  --image-checkpoint outputs/zenodo_vit/checkpoint.pt \
  --fusion-checkpoint outputs/multimodal_contrastive/checkpoint.pt
```

训练脚本默认冻结 ViT，仅训练融合层和序数回归头。使用 `--unfreeze-image-blocks 1` 或 `2` 可解冻最后的 ViT block。

## 缺失值和缺失标志

数值 0 和没有检测不是同一件事：

- `0` 表示检测完成且结果为零，或明确回答“否”；
- 缺失表示没有检测、没有记录或结果无效。

数据集返回字段值和对应 mask。模型只在整个模态缺失时替换为可学习的 missing token；模态内部部分字段缺失则由字段级 mask 传入编码器。对比预训练只对两个真实存在的模态计算正样本，不把 missing token 当成观测结果参与对比。
