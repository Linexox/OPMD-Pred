# -*- coding: utf-8 -*-
"""冒烟测试：患者记录 -> 5折划分 -> 训练折Scaling -> ViT嵌入形状。

只验证管线和形状，不做训练。ViT 只跑 2 张图。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from opmd_pred.config import CVConfig, ImageConfig, Paths
from opmd_pred.data.records import load_labeled_records
from opmd_pred.data.split import make_folds, summarize_folds
from opmd_pred.data.preprocessing import TabularPreprocessor


def main() -> None:
    paths = Paths()
    records = load_labeled_records(paths.data)
    print(f"[records] 标注患者记录: {len(records)} 条, 独立姓名: {len({r.name for r in records})}")

    folds = make_folds(records, CVConfig())
    for row in summarize_folds(folds, records):
        print(f"[fold {row['fold']}] train={row['train']} val={row['val']} test={row['test']} "
              f"test分布={row['test_dist']}")

    # ---- Scaling：只在 fold0 训练折拟合 ----
    by_key = {r.patient_key: r for r in records}
    f0 = folds[0]
    train_rows = [by_key[k].csv_row for k in f0.train if by_key[k].csv_row]
    prep = TabularPreprocessor().fit(train_rows)
    prep_path = paths.artifacts / "fold0" / "tabular_preprocessor.joblib"
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    prep.save(prep_path)
    prep = TabularPreprocessor.load(prep_path)

    n_present = {"exfoliative": 0, "methylation": 0, "basic": 0}
    sample_printed = False
    for k in f0.train + f0.val:
        r = by_key[k]
        if not r.csv_row:
            continue
        vecs, present = prep.transform(r.csv_row)
        for m, ok in present.items():
            n_present[m] += int(ok)
        if not sample_printed:
            print(f"[scaling] 示例 {r.name}: exfoliative(10d)={np.round(vecs['exfoliative'], 2)}")
            print(f"[scaling] methyl={vecs['methylation']} basic(age)={vecs['basic']} present={present}")
            sample_printed = True
    print(f"[scaling] fold0 train+val 有表格行患者 present 统计: {n_present}")

    # 泄漏检查：train 拟合的均值不应等于全量拟合的均值
    all_rows = [r.csv_row for r in records if r.csv_row]
    prep_all = TabularPreprocessor().fit(all_rows)
    col = "非整倍体细胞数量"
    print(f"[leak-check] {col} train均值={prep.num_mean[col]:.3f} 全量均值={prep_all.num_mean[col]:.3f}"
          f"（应不同，证明只在训练折拟合）")

    # ---- ViT：2 张图验证形状 ----
    from opmd_pred.data.features import extract_image_embeddings
    from opmd_pred.models.encoders import ViTEncoder, get_image_processor, build_eval_resize

    with_img = [r for r in records if r.lesion_images][:2]
    cfg = ImageConfig(n_aug=2)
    processor = get_image_processor(cfg.model_id)
    encoder = ViTEncoder(cfg.model_id, device="cpu")
    from PIL import Image
    resize = build_eval_resize(cfg.image_size)
    for r in with_img:
        img = Image.open(r.lesion_images[0]).convert("RGB")
        pixel = processor(images=[resize(img)], return_tensors="pt")["pixel_values"]
        emb = encoder(pixel)
        print(f"[vit] {r.name} 图像嵌入 shape={tuple(emb.shape)} 前4维={emb[0, :4].tolist()}")
    print("[ok] 冒烟测试通过")


if __name__ == "__main__":
    main()
