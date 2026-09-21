# -*- coding: utf-8 -*-
"""CMF 主模型（Gu 2025 框架的 4 模态实现）。

结构：
  每模态编码 h_m ∈ R^H（图像=冻结ViT嵌入的可训练投影MLP；表格=MLP）
  模态缺失 → h_m 替换为该模态的 learnable missing token
  对比投影 p_m = normalize(proj_m(h_m)) ∈ R^P
  融合 g = fusion_mlp(concat(h_1..h_4)) ∈ R^F，z_f = normalize(proj_f(g))
  任务头：MSE → Linear(F,1)；CORAL → CoralHead(F, K-1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig, N_CLASSES
from .encoders import MissingTokens, TabularEncoder

MODALITY_ORDER = ["image", "exfoliative", "methylation", "basic"]
MODALITY_IDX = {m: i for i, m in enumerate(MODALITY_ORDER)}


class CoralHead(nn.Module):
    """CORAL: 共享权重 w + K-1 个独立阈值 b_k。输出 logits_k = w·g - b_k。"""

    def __init__(self, dim: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(n_classes - 1))

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        # fc(g): (B,1)；biases: (K-1,) → 广播成 (B, K-1)
        return self.fc(g) - self.biases                      # (B, K-1)

    @staticmethod
    def expected_value(logits: torch.Tensor) -> torch.Tensor:
        """E[y] = Σ_{k=0..K-2} P(y > k) = Σ σ(logits)。"""
        return torch.sigmoid(logits).sum(dim=-1)


def _image_projection(in_dim: int, hidden: int, dropout: float) -> nn.Module:
    return TabularEncoder(in_dim, hidden, dropout)


class CMFModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, head: str = "mse",
                 modality_dims: dict[str, int] | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.head = head
        dims = modality_dims or {
            "image": cfg.image.embed_dim,
            "exfoliative": 10,
            "methylation": 1,
            "basic": 1,
        }
        self.modality_dims = dims

        # 模态编码器（图像=768→H 的投影MLP；表格=各自维度→H）
        self.encoders = nn.ModuleDict({
            m: (_image_projection(d, cfg.tabular_hidden, cfg.dropout) if m == "image"
                else TabularEncoder(d, cfg.tabular_hidden, cfg.dropout))
            for m, d in dims.items()
        })
        # 缺失 token（维度=编码器输出 H，替换在融合输入端，Gu Fig.1 的 Ec/Et）
        self.missing = MissingTokens({m: cfg.tabular_hidden for m in dims})
        # 对比投影
        self.proj = nn.ModuleDict({
            m: nn.Linear(cfg.tabular_hidden, cfg.proj_dim) for m in dims
        })
        self.proj_f = nn.Linear(cfg.fusion_dim, cfg.proj_dim)
        # 融合 MLP
        self.fusion = nn.Sequential(
            nn.Linear(cfg.tabular_hidden * len(dims), cfg.fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.ReLU(),
        )
        # 任务头
        if head == "mse":
            self.task_head = nn.Linear(cfg.fusion_dim, 1)
        elif head == "coral":
            self.task_head = CoralHead(cfg.fusion_dim, N_CLASSES)
        else:
            raise ValueError(f"未知 head: {head}")

    def forward(self, x: dict[str, torch.Tensor], missing: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: {modality: (B, in_dim)}；missing: (B, 4) bool，True=缺失(用token)。

        返回 p(各模态对比投影)、z_f(融合对比投影)、g(融合表征)、pred(任务输出)。
        """
        hs: dict[str, torch.Tensor] = {}
        for m in MODALITY_ORDER:
            h = self.encoders[m](x[m])                        # (B, H)
            tok = self.missing.get(m).expand_as(h)
            hs[m] = torch.where(missing[:, MODALITY_IDX[m]].unsqueeze(-1), tok, h)

        u = torch.cat([hs[m] for m in MODALITY_ORDER], dim=-1)
        g = self.fusion(u)                                    # (B, F)
        z_f = F.normalize(self.proj_f(g), dim=-1)
        p = {m: F.normalize(self.proj[m](hs[m]), dim=-1) for m in MODALITY_ORDER}

        out: dict[str, torch.Tensor] = {"p": p, "z_f": z_f, "g": g}
        if self.head == "mse":
            out["pred"] = self.task_head(g).squeeze(-1)       # (B,)
        else:
            out["pred"] = self.task_head(g)                   # (B, K-1)
        return out

    def predict_continuous(self, x: dict[str, torch.Tensor], missing: torch.Tensor) -> torch.Tensor:
        out = self.forward(x, missing)
        if self.head == "mse":
            return out["pred"]
        return CoralHead.expected_value(out["pred"])
