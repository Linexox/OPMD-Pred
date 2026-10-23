"""Export anonymous aggregate experiment results and readable comparison figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    audit = json.loads((root / "audit.json").read_text())
    assert audit["passed"]
    multi = json.loads((root / "multiseed-results.json").read_text())
    legacy = json.loads((root / "legacy-reconstruction-results.json").read_text())
    series = dict(multi["per_initialization"], legacy_pipeline=legacy["per_initialization"])
    labels = {"dino_softmax_balanced": "当前对照", "legacy_subset": "改为每次抽资料子集",
              "contrastive30": "增加对比预训练", "legacy_pipeline": "旧流程五模态重建",
              "vit_softmax": "换回旧图片表示", "dna_log_standardized": "改变DNA数值尺度"}
    aggregate = {}
    for name, runs in series.items():
        aggregate[name] = dict(label=labels[name], correct=[m["correct_count"] for m in runs],
           accuracy_percent=[m["correct_count"] / 117 * 100 for m in runs],
           severe_to_mild=[m["severe_to_mild_count"] for m in runs],
           cross_two=[m["two_level_error_count"] for m in runs],
           severe_recall=[m["severe_recall_all_assigned"] for m in runs],
           mean_accuracy_percent=statistics.mean(m["correct_count"] / 117 * 100 for m in runs),
           accuracy_initialization_sd=statistics.stdev(m["correct_count"] / 117 * 100 for m in runs),
           mae=[m["mae"] for m in runs])
    core = json.loads((root / "summary-core.json").read_text())
    details = json.loads((root / "summary-details.json").read_text())
    paired = json.loads((root / "small-sample/summary-paired.json").read_text())
    curves = json.loads((root / "small-sample/summary-curves.json").read_text())
    diagnostic = json.loads((root / "runs/old_data_adaptation/oof_results.json").read_text())["metrics"]
    training_sizes = {}
    for kind in ("linear", "cosine"):
        training_sizes[kind] = {}
        for fraction in (25, 50, 100):
            folders = list((root / f"small-sample/{kind}_none_fraction{fraction}").glob("fold*/refit/done.json"))
            values = [json.loads(p.read_text())["training_records"] for p in folders]
            training_sizes[kind][str(fraction)] = dict(min=min(values), max=max(values), per_fold=values)
    # All records and per-patient inputs remain in the private source tree.
    compact = lambda m: {k: m[k] for k in ("correct_count", "assigned_count", "eligible_count", "abstained_count", "severe_to_mild_count", "two_level_error_count", "mae", "severe_recall_all_assigned")}
    public = dict(date="2026-10-15", baseline_reproduction=json.loads((root / "reference_reproduction.json").read_text())["passed"],
                  multiseed=aggregate, seed42_core={k: compact(v) for k, v in core.items()},
                  seed42_details={k: compact(v) for k, v in details.items()},
                  old_data_diagnostic=compact(diagnostic), image_paired={k: compact(v) for k, v in paired.items()},
                  image_curves={k: compact(v) for k, v in curves.items()}, image_training_sizes=training_sizes,
                  verification={k: audit[k] for k in ("passed", "classifiers_replayed", "predictions_replayed", "max_absolute_probability_error", "old_test_new_inference")})
    (root / "public-summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    assert font.is_file()
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(), "font.size": 12,
                         "axes.unicode_minus": False, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    output = root / "figures"
    output.mkdir(exist_ok=True)
    names = ["dino_softmax_balanced", "legacy_subset", "contrastive30", "legacy_pipeline", "vit_softmax", "dna_log_standardized"]
    colors = ["#596274", "#398979", "#6976b9", "#bd965e", "#9b9ea7", "#9b9ea7"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    y = np.arange(len(names))
    for ax, key, title, limit in ((axes[0], "correct", "117份中平均判对多少份", (0, 117)),
                                  (axes[1], "severe_to_mild", "平均几份重度被判成正常/轻度", (0, 12))):
        values = [statistics.mean(aggregate[n][key]) for n in names]
        errors = [statistics.stdev(aggregate[n][key]) for n in names]
        ax.barh(y, values, xerr=errors, color=colors, capsize=3, height=.6)
        ax.set_xlim(*limit)
        ax.set_title(title, fontsize=13, pad=14)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(value + errors[i] + (1 if key == "correct" else .15), i, f"{value:.2f}", va="center", fontsize=11)
    axes[0].set_yticks(y, [labels[n] for n in names])
    axes[0].invert_yaxis()
    fig.text(.5, .015, "三次初始化；使用同一批117份资料。误差线为初始化间标准差，不代表临床置信区间。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output / ("multiseed-comparison." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for kind, label, color in (("linear", "普通线性分类器", "#398979"), ("cosine", "余弦分类器", "#6976b9")):
        values = [curves[f"{kind}_none_{fraction}"]["correct_count"] for fraction in (0.25, 0.5, 1.0)]
        ax.plot([25, 50, 100], values, marker="o", label=label, color=color, linewidth=2)
        for x, value in zip((25, 50, 100), values):
            ax.annotate(f"{value}/75", (x, value), xytext=(0, 10 if kind == "linear" else -18), textcoords="offset points", ha="center")
    ax.set(xlim=(20, 105), ylim=(0, 75), xlabel="使用每折训练来源组的比例（%）", ylabel="固定75份评价资料中判对的份数", title="真实训练资料增加后，结果怎样变化")
    ax.set_xticks([25, 50, 100]); ax.grid(alpha=.15); ax.legend(loc="lower right")
    fig.text(.5, .01, "固定一种抽样顺序、一次初始化；每档实际训练资料数随折变化，见数据表。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output / ("training-size-curve." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    values = [paired[k]["correct_count"] for k in ("linear_none_1.0", "linear_geometry_1.0", "cosine_none_1.0", "cosine_geometry_1.0")]
    ax.bar(range(4), values, color=["#398979", "#95c2b9", "#6976b9", "#afb5da"], width=.6)
    ax.set_xticks(range(4), ["线性\n不增强", "线性\n空间增强", "余弦\n不增强", "余弦\n空间增强"])
    ax.set(ylim=(0, 75), ylabel="75份中判对的份数", title="小样本分类器与空间增强的首轮比较")
    for i, value in enumerate(values):ax.text(i, value + 1, f"{value}/75", ha="center")
    ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.text(.5, .01, "一次初始化；同75份有照片资料、同五折、相同训练更新数；只在训练时增强。", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output / ("image-augmentation-comparison." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(dict(public_summary_written=True, figures=3, records_exported=False)), flush=True)


if __name__ == "__main__":
    main()
