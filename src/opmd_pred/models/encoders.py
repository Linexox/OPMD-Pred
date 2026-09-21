# -*- coding: utf-8 -*-
"""模型组件：冻结 ViT 图像编码器、表格模态 MLP、可学习 missing tokens。

对照 Gu 2025：
- 图像/表格编码器冻结（ViT 天然冻结；表格 MLP 可训练，属于"融合侧"参数）
- 每个模态一个 learnable token：模态缺失时用它替换该模态表征
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import transforms
from transformers import ViTImageProcessor



class ViTEncoder(nn.Module):
    """冻结的 ImageNet 预训练 ViT，输出 [CLS] 表征（默认 768 维）。"""

    def __init__(self, model_id: str = "google/vit-base-patch16-224-in21k", device: str = "cpu"):
        super().__init__()
        from transformers import ViTModel

        self.model = ViTModel.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device_ = device
        self.model.to(device)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values.to(self.device_))
        return out.last_hidden_state[:, 0]  # [CLS]，不做 pooler 的非线性，保留原始表征


def get_image_processor(model_id: str = "google/vit-base-patch16-224-in21k"):
    return ViTImageProcessor.from_pretrained(model_id)


def build_train_augment(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.RandomRotation(degrees=15),
        ]
    )


def build_eval_resize(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
        ]
    )


class TabularEncoder(nn.Module):
    """表格模态 MLP: in_dim -> hidden -> hidden"""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MissingTokens(nn.Module):
    """每个模态一个可学习向量, 模态缺失时替换该模态表征"""

    def __init__(self, modality_dims: dict[str, int]):
        super().__init__()
        self.tokens = nn.ParameterDict(
            {
                m: nn.Parameter(torch.zeros(d))
                for m, d in modality_dims.items()
            }
        )
        for t in self.tokens.values():
            nn.init.normal_(t, std=0.02)

    def get(self, modality: str) -> torch.Tensor:
        return self.tokens[modality]
