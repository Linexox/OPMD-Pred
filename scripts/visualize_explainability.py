"""Create per-sample SHAP waterfall and InterSHAP matrix figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon


LABEL_NAMES = {
    0: "normal/mild",
    1: "moderate",
    2: "severe",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/tune_subset_lr1e4/explainability/summary.json",
    )
    parser.add_argument(
        "--samples",
        default="outputs/tune_subset_lr1e4/explainability/samples.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/tune_subset_lr1e4/explainability/figures",
    )
    return parser.parse_args()


def read_samples(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def draw_waterfall(row, names, output):
    values = [float(row[f"shap_{name}"]) for name in names]
    order = sorted(range(len(names)), key=lambda index: abs(values[index]), reverse=True)
    labels = [names[index] for index in order]
    ordered_values = [values[index] for index in order]
    baseline = float(row.get("base_value", row.get("baseline_score", 0.0)))
    cumulative = [baseline]
    for value in ordered_values:
        cumulative.append(cumulative[-1] + value)

    figure, axis = plt.subplots(figsize=(12, 6.8), facecolor="#f4f6ef")
    axis.set_facecolor("#f4f6ef")
    bar_height = 0.62
    for index, value in enumerate(ordered_values):
        start = cumulative[index]
        end = cumulative[index + 1]
        left, right = sorted((start, end))
        width = right - left
        direction = 1 if value >= 0 else -1
        tip = min(width * 0.25, 0.018)
        if width > tip * 1.8:
            if direction > 0:
                points = [
                    (left, index - bar_height / 2),
                    (right - tip, index - bar_height / 2),
                    (right, index),
                    (right - tip, index + bar_height / 2),
                    (left, index + bar_height / 2),
                ]
            else:
                points = [
                    (left + tip, index - bar_height / 2),
                    (right, index - bar_height / 2),
                    (right, index + bar_height / 2),
                    (left + tip, index + bar_height / 2),
                    (left, index),
                ]
        else:
            points = [
                (left, index - bar_height / 2),
                (right, index - bar_height / 2),
                (right, index + bar_height / 2),
                (left, index + bar_height / 2),
            ]
        color = "#f59b45" if value >= 0 else "#344054"
        axis.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor="none"))
        text_color = "white" if width >= 0.055 else color
        text_x = (left + right) / 2 if width >= 0.055 else right + 0.012
        horizontal = "center" if width >= 0.055 else "left"
        axis.text(
            text_x,
            index,
            f"{value:+.2f}",
            ha=horizontal,
            va="center",
            color=text_color,
            fontsize=14,
            fontweight="bold",
        )
        if index < len(ordered_values) - 1:
            axis.plot(
                [end, end],
                [index + bar_height / 2, index + 1 - bar_height / 2],
                color="#b8bdb4",
                linestyle=(0, (1, 3)),
                linewidth=1,
            )

    total = cumulative[-1]
    axis.axvline(baseline, color="#8b9188", linestyle=(0, (1, 3)), linewidth=1)
    axis.axvline(total, color="#8b9188", linestyle=(0, (1, 3)), linewidth=1)
    label_y = len(labels) - 0.15
    baseline_x, total_x = baseline, total
    if abs(total - baseline) < 0.06:
        baseline_x -= 0.025
        total_x += 0.025
    axis.text(baseline_x, label_y, f"f(empty) = {baseline:.2f}", ha="center", va="top", fontsize=13)
    axis.text(total_x, label_y, f"f(x) = {total:.2f}", ha="center", va="top", fontsize=13)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Contribution to model output", fontsize=14)
    sample_title = f"Sample {row['sample_index']} | {row.get('sample_id', '')}"
    if "true_label" in row and "predicted_label" in row:
        true_label = int(row["true_label"])
        predicted_label = int(row["predicted_label"])
        status = "CORRECT" if row.get("correct", "0") == "1" else "WRONG"
        sample_title += (
            f" | true: {LABEL_NAMES[true_label]}"
            f" | pred: {LABEL_NAMES[predicted_label]} | {status}"
        )
    sample_title += "\nmodality SHAP"
    axis.set_title(sample_title, fontsize=14, pad=30, fontweight="bold")
    axis.grid(axis="y", color="#dfe3da", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=14)
    axis.tick_params(axis="x", labelsize=13)
    padding = max(abs(value) for value in cumulative) * 0.16 + 0.02
    axis.set_xlim(min(cumulative) - padding, max(cumulative) + padding)
    axis.set_ylim(len(labels) + 0.15, -0.7)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output / f"shap_waterfall_{int(row['sample_index']):03d}.png", dpi=180)
    return figure


def draw_interactions(row, names, output):
    matrix = [[0.0 for _ in names] for _ in names]
    for first, name in enumerate(names):
        for second in range(first + 1, len(names)):
            value = float(row[f"interaction_{name}_{names[second]}"])
            matrix[first][second] = value
            matrix[second][first] = value

    limit = max(abs(value) for values in matrix for value in values) or 1.0
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_title(f"Sample {row['sample_index']} Shapley interactions")
    for first in range(len(names)):
        for second in range(len(names)):
            axis.text(
                second,
                first,
                f"{matrix[first][second]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Interaction value")
    figure.tight_layout()
    figure.savefig(output / f"intershap_matrix_{int(row['sample_index']):03d}.png", dpi=180)
    return figure


def draw_global_interactions(summary, names, output):
    matrix = summary["global_mean_abs_interaction"]

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="YlOrRd")
    axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_title("Mean absolute Shapley interactions")
    for first in range(len(names)):
        for second in range(len(names)):
            axis.text(
                second,
                first,
                f"{matrix[first][second]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Mean absolute interaction")
    figure.tight_layout()
    figure.savefig(output / "intershap_mean_abs.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.summary).open(encoding="utf-8") as file:
        summary = json.load(file)
    names = summary["modality_names"]
    rows = read_samples(args.samples)
    draw_global_interactions(summary, names, output)

    waterfall_pdf = PdfPages(output / "shap_waterfalls.pdf")
    interaction_pdf = PdfPages(output / "intershap_matrices.pdf")
    try:
        for row in rows:
            waterfall = draw_waterfall(row, names, output)
            interaction = draw_interactions(row, names, output)
            waterfall_pdf.savefig(waterfall)
            interaction_pdf.savefig(interaction)
            plt.close(waterfall)
            plt.close(interaction)
    finally:
        waterfall_pdf.close()
        interaction_pdf.close()

    print(f"wrote {len(rows)} samples to {output}")


if __name__ == "__main__":
    main()
