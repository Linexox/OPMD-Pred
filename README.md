# OPMD-Pred

口腔潜在恶性病变（OPMD）无创序数分级：图像 + 脱落细胞检测 + 甲基化 + 基本信息，四模态缺失感知多模态融合。

框架来源 **Gu et al. 2025, MICCAI**（对比多模态融合 + 全组合同步 modality dropout + 可学习 missing token），
可解释性来源 **Wenderoth et al. 2025, AAAI**（InterSHAP 跨模态交互量化）。

## 目录结构

```
configs/            （超参占位，当前主要由 config.py 的 dataclass 提供）
src/opmd_pred/
├─ config.py              标签序数、四模态字段、CV/训练超参、路径
├─ inventory.py           数据清单扫描（患者级，区分病灶图与单据翻拍）
├─ data/
│  ├─ records.py          按患者组织标注数据，供后续模块消费
│  ├─ split.py            患者级 StratifiedGroupKFold（5 折，折内 4:1 出验证集）
│  ├─ preprocessing.py    表格 Scaling（median 填补 + log1p + StandardScaler），只拟合训练折
│  └─ features.py         冻结 ViT 图像嵌入缓存（阶段A）
├─ models/
│  ├─ encoders.py         冻结 ViT / 表格 MLP / MissingTokens
│  ├─ cmf_model.py        CMF 主模型（Missing Embeddings + 融合 + MSE/CORAL 头）
│  └─ losses.py           有监督 sigmoid 对比损失 + CORAL 损失
├─ training/
│  ├─ dataset.py          折数据组装 → Sample / collate
│  ├─ metrics.py          QWK / MAE / 1-off accuracy / macro-F1
│  └─ trainer.py          阶段B 对比预训练、阶段C 全组合 dropout 任务训练
└─ explain/
   ├─ intershap.py        4 模态精确 SII（含自身贡献 Φ_ii）
   └─ shap_analysis.py    12 维表格精确 Shapley（4096 联盟全枚举）
scripts/
├─ inventory.py           产出 reports/data_inventory.md
├─ extract_features.py    阶段A：全量 ViT 嵌入缓存
├─ train_cv.py            阶段B+C：5 折 CV 训练
├─ explain.py             阶段D：InterSHAP + 表格 SHAP
├─ smoke_preprocessing.py Scaling / 5 折 / 泄漏检查
└─ smoke_pipeline.py      端到端自检（含 SII 权重数学验证，CPU 可跑）
```

## 运行流程

```bash
cd ~/24334067/OPMD-Pred
export PATH=$HOME/.local/bin:$PATH

# 阶段A：图像嵌入缓存（GPU，首次会下载 ViT 权重）
uv run python scripts/extract_features.py --device cuda

# 阶段B+C：5 折 CV（MSE 回归主线）
uv run python scripts/train_cv.py --tag exp1_mse --head mse --device cuda

# 消融：CORAL 头 / 去掉对比预训练
uv run python scripts/train_cv.py --tag exp2_coral --head coral --device cuda
uv run python scripts/train_cv.py --tag exp3_nopre --head mse --skip-pretrain --device cuda

# 阶段D：可解释性
uv run python scripts/explain.py --tag exp1_mse --device cuda
```

结果落在 `artifacts/<tag>/`（模型、scaler、逐折指标、预测）
与 `reports/explain/<tag>/`（InterSHAP 矩阵、特征重要性），
横向对比追加写入 `reports/RESULTS.md`。

## 关键约定

- 标签序数：normal=0 < mild=1 < moderate=2 < severe=3
- 患者级切分：同一姓名（含初诊/复诊）整体进同一集合，杜绝泄漏
- 所有统计量（中位数、均值、标准差）只在训练折拟合，按折持久化
- 缺失模态处理：替换为该模态的 learnable missing embedding（不用零向量、不用别的样本）
- InterSHAP 屏蔽策略同上使用 missing embedding，因此不存在原文 masking 的分布外扰动
