"""Create SYSU label and modality availability pie charts."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "SYSU_processed" / "sysu_samples.csv"
OUTPUT_DIR = ROOT / "outputs" / "figures" / "dataset_distribution"
MISSING = {"", "NAN", "nan", "NA", "N/A", "null", "NULL", "/", "#VALUE!"}
HISTORY_FIELDS = (
    "是否吸烟",
    "吸烟年限（年）",
    "吸烟量（支/天）",
    "是否饮酒",
    "饮酒年限(年)",
    "饮酒量(ml/天)",
    "是否嚼槟榔",
    "嚼槟榔年限(年)",
    "嚼槟榔量（颗/天）",
)


def present(value):
    return str(value).strip() not in MISSING


def load_rows():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def modality_flags(row):
    return {
        "图像": present(row["image_path"]),
        "基本信息": present(row["sex"]) or present(row["age"]),
        "烟酒槟榔史": any(present(row[field]) for field in HISTORY_FIELDS),
        "TCT": present(row["TCT意见"]),
        "DNA倍体": present(row["DNA倍体分析结果"]),
        "甲基化": present(row["甲基化总数"]),
    }


def configure_font():
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_pie(values, labels, colors, title, filename, legend_labels):
    figure, axis = plt.subplots(figsize=(7.2, 6.0), dpi=180)
    wedges, _, _ = axis.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 5 else "",
        pctdistance=0.68,
        wedgeprops={"linewidth": 2.2, "edgecolor": "white"},
        textprops={"fontsize": 15, "weight": "bold", "color": "#1F2937"},
    )
    axis.set_title(title, fontsize=20, weight="bold", pad=18, color="#1F2937")
    axis.legend(
        wedges,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        fontsize=13,
        ncol=2,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_DIR / filename.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    configure_font()
    rows = load_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_names = {
        "level1_mild_or_normal": "正常与轻度合并",
        "level2_moderate": "中度",
        "level3_severe": "重度",
    }
    label_counts = {}
    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
    ordered_labels = ["level1_mild_or_normal", "level2_moderate", "level3_severe"]
    label_values = [label_counts[label] for label in ordered_labels]
    label_legend = [
        f"{label_names[label]}：{label_counts[label]} 条"
        for label in ordered_labels
    ]
    save_pie(
        label_values,
        [label_names[label] for label in ordered_labels],
        ["#2A9D8F", "#E9C46A", "#E76F51"],
        "SYSU 标签分布（n=139）",
        "sysu_label_distribution_pie.png",
        label_legend,
    )

    modality_colors = {
        "图像": "#2A9D8F",
        "基本信息": "#457B9D",
        "烟酒槟榔史": "#E9C46A",
        "TCT": "#F4A261",
        "DNA倍体": "#7B61A8",
        "甲基化": "#E76F51",
    }
    flags = [modality_flags(row) for row in rows]
    for modality, color in modality_colors.items():
        available = sum(flag[modality] for flag in flags)
        missing = len(rows) - available
        save_pie(
            [available, missing],
            ["可用", "缺失"],
            [color, "#E5E7EB"],
            f"{modality}可用率（n=139）",
            f"sysu_{modality}_availability_pie.png",
            [f"可用：{available} 条（{available / len(rows):.1%}）", f"缺失：{missing} 条（{missing / len(rows):.1%}）"],
        )


if __name__ == "__main__":
    main()
