# -*- coding: utf-8 -*-
"""本地自检（不需要 GPU、不需要 ViT 权重）：

1. InterSHAP / SII 的数学性质：用已知的 2-additive 合成博弈反解 Φ_ij 与 φ_i
1b. 两个任务头（MSE / CORAL）的形状自检
2. 端到端 mini 流水线：伪造图像缓存 → 2 折 × 少量 epoch 的对比预训练 + 任务训练（两个头）
3. 解释模块：InterSHAP（含 missing token 屏蔽）+ 表格精确 Shapley 跑通

用法：
    uv run python scripts/smoke_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opmd_pred.config import CVConfig, ModelConfig, Paths, TrainConfig  # noqa: E402
from opmd_pred.data.records import load_labeled_records  # noqa: E402
from opmd_pred.data.split import make_folds  # noqa: E402
from opmd_pred.explain.intershap import global_intershap, intershap, shapley_values, sii_matrix  # noqa: E402
from opmd_pred.explain.shap_analysis import modality_group_shapley, subsample_background, tabular_shapley  # noqa: E402
from opmd_pred.models.cmf_model import MODALITY_ORDER, CMFModel  # noqa: E402
from opmd_pred.training.dataset import build_fold_data, fit_preprocessor  # noqa: E402
from opmd_pred.training.trainer import pretrain_fold, train_fold  # noqa: E402

N_MOD = len(MODALITY_ORDER)


# ---------------------------------------------------------------------------
# 1. SII / Shapley 权重自检
# ---------------------------------------------------------------------------
def test_game_theory_weights(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = N_MOD
    a = rng.normal(size=n)                       # 模态自身贡献
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c = rng.normal()
            C[i, j] = C[j, i] = c                # 成对交互项

    # 2-additive 博弈：v(S) = Σ_{i∈S} a_i + Σ_{i<j∈S} C_ij
    v = np.zeros(2 ** n)
    for s in range(2 ** n):
        members = [i for i in range(n) if (s >> i) & 1]
        v[s] = sum(a[i] for i in members) + sum(
            C[i, j] for k, i in enumerate(members) for j in members[k + 1:]
        )

    phi = shapley_values(v)
    Phi = sii_matrix(v)

    # (i) 成对 SII 必须精确恢复交互项 C_ij
    err_pair = max(abs(Phi[i, j] - C[i, j]) for i, j in
                   [(i, j) for i in range(n) for j in range(n) if i != j])
    # (ii) Shapley 值必须精确、且满足效率性 Σφ = v(N) - v(∅)
    phi_expected = np.array([a[i] + 0.5 * C[i].sum() for i in range(n)])
    err_phi = float(np.abs(phi - phi_expected).max())
    err_eff = abs(phi.sum() - (v[-1] - v[0]))
    # (iii) Φ_ii = φ_i - Σ_{j≠i} Φ_ij 的总量必须守恒
    err_total = abs(Phi.sum() - (v[-1] - v[0]))

    assert err_pair < 1e-10, f"成对交互未恢复: {err_pair}"
    assert err_phi < 1e-10, f"Shapley 值错: {err_phi}"
    assert err_eff < 1e-10, f"Shapley 效率性失败: {err_eff}"
    assert err_total < 1e-10, f"Φ 分解不守恒: {err_total}"
    print(f"[math] 2-additive 博弈反解 OK — 成对误差 {err_pair:.2e}, "
          f"Shapley 误差 {err_phi:.2e}, 守恒误差 {err_total:.2e}")


# ---------------------------------------------------------------------------
# 1b. 两个任务头的形状自检（MSE / CORAL）
# ---------------------------------------------------------------------------
def test_heads() -> None:
    """两个头都跑一次前向 + 损失 + 连续预测，防止只测 MSE 漏掉 CORAL 的形状 bug。"""
    from opmd_pred.config import N_CLASSES
    from opmd_pred.models.losses import coral_loss

    rng = np.random.default_rng(0)
    n = 5
    missing = torch.zeros(n, N_MOD, dtype=torch.bool)
    x = {
        "image": torch.randn(n, ModelConfig().image.embed_dim),
        "exfoliative": torch.randn(n, 10),
        "methylation": torch.randn(n, 1),
        "basic": torch.randn(n, 1),
    }
    y = torch.tensor([0, 1, 2, 3, 2])

    for head in ("mse", "coral"):
        model = CMFModel(ModelConfig(), head=head).eval()
        out = model(x, missing)
        cont = model.predict_continuous(x, missing)
        assert cont.shape == (n,), f"{head}: predict_continuous 形状 {tuple(cont.shape)}"
        if head == "mse":
            assert out["pred"].shape == (n,), f"mse head 形状 {tuple(out['pred'].shape)}"
            loss = torch.nn.functional.mse_loss(out["pred"], y.float())
        else:
            assert out["pred"].shape == (n, N_CLASSES - 1), \
                f"coral head 形状 {tuple(out['pred'].shape)}，应为 {(n, N_CLASSES - 1)}"
            loss = coral_loss(out["pred"], y, N_CLASSES)
            # 序概率应随阈值单调不增，且期望值落在 [0, K-1]
            assert torch.all((cont >= 0) & (cont <= N_CLASSES - 1)), "CORAL 期望值越界"
        assert torch.isfinite(loss), f"{head}: loss 非有限"
        print(f"[heads] {head:5s} ok — pred={tuple(out['pred'].shape)} "
              f"cont={tuple(cont.shape)} loss={loss.item():.4f}")


# ---------------------------------------------------------------------------
# 2. 伪造图像缓存（跳过 ViT，验证流程本身）
# ---------------------------------------------------------------------------
def build_fake_cache(records, out_dir: Path, dim: int = 768, n_aug: int = 3,
                     seed: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arrays, index = {}, {}
    for r in records:
        if not r.lesion_images:
            continue
        entry = {"images": [str(p) for p in r.lesion_images], "keys": []}
        for i in range(len(r.lesion_images)):
            ek = f"{r.patient_key}|{i}|eval"
            arrays[ek] = rng.normal(size=dim).astype(np.float32)
            entry["keys"].append(ek)
            for k in range(n_aug):
                ak = f"{r.patient_key}|{i}|aug{k}"
                arrays[ak] = rng.normal(size=dim).astype(np.float32)
                entry["keys"].append(ak)
        index[r.patient_key] = entry
    np.savez_compressed(out_dir / "image_embeddings.npz", **arrays)
    (out_dir / "image_index.json").write_text(
        json.dumps({"model_id": "FAKE", "n_aug": n_aug, "patients": index},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


# ---------------------------------------------------------------------------
# 3. 端到端 mini 训练 + 解释
# ---------------------------------------------------------------------------
def main() -> None:
    test_game_theory_weights()
    test_heads()

    paths = Paths()
    device = "cpu"
    records = load_labeled_records(paths.data)
    print(f"[records] {len(records)} 条, 有病灶图 {sum(bool(r.lesion_images) for r in records)} 条")

    cache_dir = build_fake_cache(records, paths.features / "_smoke_cache")
    print(f"[cache] fake cache -> {cache_dir}")

    folds = make_folds(records, CVConfig(seed=42))

    # 两个头都走端到端（CORAL 只跑 1 折；解释模块与 head 无关，只跑一次）
    for head, n_folds in (("mse", 2), ("coral", 1)):
        cfg = TrainConfig(head=head)
        cfg.pretrain_epochs, cfg.train_epochs = 2, 3
        for fold in folds[:n_folds]:
            torch.manual_seed(0)
            rng = np.random.default_rng(0)
            prep = fit_preprocessor(records, fold)
            data = build_fold_data(records, fold, prep, cache_dir)
            n_img = sum(s.has_image for s in data.train + data.val + data.test)
            print(f"\n[head={head} fold {fold.fold}] train {len(data.train)} / val {len(data.val)} "
                  f"/ test {len(data.test)}, 有图像样本 {n_img}")

            model = CMFModel(ModelConfig(), head=head)
            pretrain_fold(model, data, cfg, device, rng, log_every=1)
            metric, _ = train_fold(model, data, cfg, device, rng)
            print(f"  [test] " + "  ".join(f"{k}={v:.4f}" for k, v in metric.items()
                                           if isinstance(v, float)))

            if head != "mse":
                continue

            res = intershap(model, data.test[:3], device)
            g, mat = global_intershap(res, complete_case_only=False)
            assert res.phi_matrix.shape == (3, N_MOD, N_MOD)
            assert res.intershap_local.shape == (3,)
            print(f"  [intershap] local={np.round(res.intershap_local, 4).tolist()} global={g:.4f}")

            bg = subsample_background(data.train, max_n=4)
            phi = tabular_shapley(model, data.test[0], bg, device)
            print(f"  [shap] shape={phi.shape} group={modality_group_shapley(phi)}")

    print("\n[OK] smoke pipeline 全部通过")


if __name__ == "__main__":
    main()
