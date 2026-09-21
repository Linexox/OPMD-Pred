# -*- coding: utf-8 -*-
"""全局配置：标签、模态特征定义、类别编码、CV 与模型超参。

约定：
- 标签 normal=0 < mild=1 < moderate=2 < severe=3（序数，回归目标直接用整数）
- 四个模态：image / exfoliative(脱落细胞检测) / methylation(甲基化) / basic(基本信息)
- 任何"只在训练折拟合"的统计量都在 preprocessing.py 里完成，本文件只放静态定义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

LABEL_ORDER = ["normal", "mild", "moderate", "severe"]
LABEL_TO_IDX = {c: i for i, c in enumerate(LABEL_ORDER)}
N_CLASSES = len(LABEL_ORDER)

# ---------------------------------------------------------------------------
# 模态一：脱落细胞检测（10 维 = 3 个类别编码 + 7 个计数）
# ---------------------------------------------------------------------------
EXFOLIATIVE_CAT = ["病原体", "炎细胞", "TCT意见"]
EXFOLIATIVE_NUM = [
    "分析上皮细胞数",
    "二倍体细胞数量",
    "四倍体细胞数量",
    "异倍体细胞数量",
    "高倍体细胞数量",
    "非整倍体细胞数量",
    "DI>=2.3细胞",
]
EXFOLIATIVE_FEATURES = EXFOLIATIVE_CAT + EXFOLIATIVE_NUM

# 模态二：甲基化（1 维）
METHYLATION_FEATURES = ["甲基化总数"]

# 模态三：基本信息（1 维）
BASIC_FEATURES = ["age"]

MODALITIES = ["image", "exfoliative", "methylation", "basic"]

# 表格列名兼容：severe.csv 用的是全角 ≥
COLUMN_ALIASES = {
    "DI≥2.3细胞": "DI>=2.3细胞",
    "DI＞=2.3细胞": "DI>=2.3细胞",
}

# 类别列里的无效值（这些列出现即视为该字段缺失；计数列的 0 是合法值，不在此列）
CAT_INVALID_TOKENS = {"", "nan", "na", "none", "0", "-", "标本不满意"}

# 有序类别映射（序数编码；未知类别报错而不是静默吞掉）
ORDINAL_MAPS: dict[str, dict[str, float]] = {
    "炎细胞": {"未见": 0.0, "少量": 1.0, "中量": 2.0, "大量": 3.0},
    "TCT意见": {
        "未见核异质细胞": 0.0,
        "非典型鳞状上皮细胞": 1.0,
        "可疑异型细胞": 2.0,
        "高度可疑异型细胞": 3.0,
        "癌细胞": 4.0,
    },
}
# 二值类别映射：key=阴性取值；其他非空取值一律记 1（阳性/检出）
BINARY_MAPS: dict[str, str] = {"病原体": "未见"}

# 计数特征做 log1p 再标准化（跨度 54 ~ 86,000，重右偏）
LOG1P_FEATURES = set(EXFOLIATIVE_NUM)

# ---------------------------------------------------------------------------
# 图像模态
# ---------------------------------------------------------------------------
@dataclass
class ImageConfig:
    model_id: str = "google/vit-base-patch16-224-in21k"
    embed_dim: int = 768
    image_size: int = 224
    n_aug: int = 10           # 训练增强倍数
    batch_size: int = 16      # 特征提取 batch


# ---------------------------------------------------------------------------
# 表格编码器 / 融合
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    tabular_hidden: int = 64     # 每个表格模态 MLP 输出维
    fusion_dim: int = 128        # 融合 MLP 中间维
    proj_dim: int = 128          # 对比空间投影维（z_m / z_f 共用）
    dropout: float = 0.1
    image: ImageConfig = field(default_factory=ImageConfig)


# ---------------------------------------------------------------------------
# CV 与训练
# ---------------------------------------------------------------------------
@dataclass
class CVConfig:
    n_splits: int = 5            # 患者级分层 5 折
    val_ratio: float = 0.2       # 折内 4:1 出验证集
    seed: int = 42


@dataclass
class TrainConfig:
    # 阶段B：有监督对比预训练（只用 56 名标注患者）
    pretrain_epochs: int = 100
    pretrain_lr: float = 1e-4
    temperature: float = 0.5     # sigmoid 对比损失的 t 初值
    # 阶段C：目标任务（MSE 回归为主，CORAL 为消融对照）
    train_epochs: int = 150
    train_lr: float = 1e-4
    weight_decay: float = 1e-4
    head: str = "mse"            # "mse" | "coral"


@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    data: Path = field(init=False)
    reports: Path = field(init=False)
    features: Path = field(init=False)
    artifacts: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data = self.root / "data"
        self.reports = self.root / "reports"
        self.features = self.root / "features"
        self.artifacts = self.root / "artifacts"


def normalize_patient_name(name: str) -> str:
    """CSV 和目录两边的姓名统一：去掉 -初诊/-复诊 后缀与空白。"""
    n = (name or "").strip()
    for suffix in ("-初诊", "-复诊", "—初诊", "—复诊"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.strip()
