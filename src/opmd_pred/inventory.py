# -*- coding: utf-8 -*-
"""患者级数据清单。

扫描 data/<类别>/<患者目录>/，按患者（而不是按文件）建立记录，用于回答三个问题：
  1. 每个类别到底有多少独立患者、多少张可用病灶图（病历单/病理单不算输入）
  2. 每个患者哪些模态是缺的（目录名标注 + PDF 是否存在 + CSV 是否为空值）
  3. CSV 表格与患者目录能否按姓名一一对齐（旧管线的已知风险）

输出：
  reports/patients.csv      每个患者一行，供下游切分与建模使用
  reports/data_inventory.md 人读的清单报告
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

# 四个有明确标签的类别；unknown / 缺少数据 单独统计，不参与建模
LABELED_CLASSES = ("normal", "mild", "moderate", "severe")
AUX_CLASSES = ("unknown", "缺少数据")
CLASS_ORDER = ("normal", "mild", "moderate", "severe")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 只有这些是病灶/临床照片，可以作为模型输入
LESION_KEYWORDS = ("病损", "临床")
# 这些是纸质单据的翻拍，绝不能混进图像训练集
DOC_KEYWORDS = ("病历", "病理单", "病理图", "首页", "报告", "检查单", "首页")
NULL_TOKENS = {"", "nan", "na", "n/a", "none", "null", "-", "无"}

MISSING_PATTERNS = (
    ("甲基化", "methylation"),
    ("脱落细胞", "exfoliative"),
    ("临床图片", "clinical_image"),
    ("临床图", "clinical_image"),
)


def is_null(value: str | None) -> bool:
    return value is None or value.strip().lower() in NULL_TOKENS


def split_patient_name(dir_name: str) -> tuple[str, str]:
    """把 '何玉玲-初诊（右舌腹-符合OLP）' 拆成 ('何玉玲', '初诊（右舌腹-符合OLP）')。"""
    head = re.split(r"[（(]", dir_name, maxsplit=1)[0].strip()
    parts = head.split("-")
    name = parts[0].strip()
    rest = "-".join(parts[1:]).strip()
    return name, rest


def parse_missing_notes(dir_name: str) -> list[str]:
    notes: list[str] = []
    for kw, flag in MISSING_PATTERNS:
        if re.search(r"缺[^，,；;）)]{0,6}" + kw, dir_name) or re.search(
            r"缺少[^，,；;）)]{0,6}" + kw, dir_name
        ):
            if flag not in notes:
                notes.append(flag)
    return notes


def read_tag(tag_path: Path) -> str:
    try:
        return tag_path.read_text(encoding="utf-8", errors="replace").strip()[:40]
    except OSError:
        return ""


@dataclass
class Patient:
    name: str
    visit: str
    label: str
    folder: str
    group: str = ""
    lesion_images: list[str] = field(default_factory=list)
    doc_images: list[str] = field(default_factory=list)
    other_images: list[str] = field(default_factory=list)
    tct_pdfs: list[str] = field(default_factory=list)
    methyl_pdfs: list[str] = field(default_factory=list)
    quality_files: list[str] = field(default_factory=list)
    tag: str = ""
    missing_notes: list[str] = field(default_factory=list)
    csv_row: dict[str, str] | None = None

    @property
    def n_lesion(self) -> int:
        return len(self.lesion_images)

    @property
    def has_csv(self) -> bool:
        return self.csv_row is not None

    @property
    def n_tabular_valid(self) -> int:
        if not self.csv_row:
            return 0
        return sum(1 for k, v in self.csv_row.items() if k != "name" and not is_null(v))


def load_class_csv(class_dir: Path, class_name: str) -> dict[str, dict[str, str]]:
    csv_path = class_dir / f"{class_name}.csv"
    if not csv_path.exists():
        cands = list(class_dir.glob("*.csv"))
        if not cands:
            return {}
        csv_path = cands[0]
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("name") or "").strip()
            if name:
                # CSV 里姓名可能带 -初诊/-复诊 后缀，与患者目录匹配时统一去掉
                for suffix in ("-初诊", "-复诊"):
                    if name.endswith(suffix):
                        name = name[: -len(suffix)]
                        break
                rows[name] = row
    return rows


def collect_patient_dirs(class_dir: Path) -> list[tuple[Path, str]]:
    """返回 [(患者目录, 分组名或"")]。

    支持两种布局：
      class/<患者目录>/...              （normal/mild/moderate/severe/unknown）
      class/<缺失分组>/<患者目录>/...   （缺少数据/缺少甲基化/<患者>）
    """
    out: list[tuple[Path, str]] = []
    for entry in sorted(class_dir.iterdir()):
        if not entry.is_dir():
            continue
        has_files = any(f.is_file() for f in entry.iterdir())
        if has_files:
            out.append((entry, ""))
        else:
            for sub in sorted(entry.iterdir()):
                if sub.is_dir():
                    out.append((sub, entry.name))
    return out


def scan_class(class_dir: Path, class_name: str) -> list[Patient]:
    csv_rows = load_class_csv(class_dir, class_name)
    patients: list[Patient] = []
    for entry, group in collect_patient_dirs(class_dir):
        name, visit = split_patient_name(entry.name)
        p = Patient(
            name=name,
            visit=visit,
            label=class_name,
            folder=entry.name,
            group=group,
            missing_notes=parse_missing_notes(entry.name),
        )
        tag_path = entry / "tag.txt"
        if tag_path.exists():
            p.tag = read_tag(tag_path)
        for f in sorted(entry.rglob("*")):
            if not f.is_file():
                continue
            if f.name == ".DS_Store":
                continue
            rel = f.relative_to(entry).as_posix()
            low = f.name.lower()
            if f.suffix.lower() in IMAGE_EXTS:
                if any(k in f.name for k in LESION_KEYWORDS):
                    p.lesion_images.append(rel)
                elif any(k in f.name for k in DOC_KEYWORDS):
                    p.doc_images.append(rel)
                else:
                    p.other_images.append(rel)
            elif f.suffix.lower() == ".pdf":
                if "甲基化" in f.name:
                    p.methyl_pdfs.append(rel)
                else:
                    p.tct_pdfs.append(rel)
            elif f.suffix.lower() == ".txt":
                if "图片质量" in f.name or f.name != "tag.txt":
                    p.quality_files.append(rel)
        # 同名 CSV 行匹配；同一类别下重名按出现顺序消耗
        if name in csv_rows:
            p.csv_row = csv_rows.pop(name)
        patients.append(p)
    return patients


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def build_report(all_patients: dict[str, list[Patient]], leftover_csv: dict[str, dict[str, list[str]]]) -> str:
    lines: list[str] = ["# OPMD-Pred 患者级数据清单", ""]

    total_dir_patients = sum(len(v) for v in all_patients.values())
    labeled = [p for c in LABELED_CLASSES for p in all_patients.get(c, [])]
    lines.append(
        f"- 扫描到患者目录 **{total_dir_patients}** 个，其中四类明确标签 **{len(labeled)}** 个。"
    )
    lines.append("")

    # 1. 每类概览
    rows = []
    for c in CLASS_ORDER:
        ps = all_patients.get(c, [])
        n_lesion_total = sum(p.n_lesion for p in ps)
        n_has_lesion = sum(1 for p in ps if p.n_lesion > 0)
        n_tct = sum(1 for p in ps if p.tct_pdfs)
        n_methyl = sum(1 for p in ps if p.methyl_pdfs)
        n_csv = sum(1 for p in ps if p.has_csv)
        rows.append([c, len(ps), n_has_lesion, n_lesion_total, n_tct, n_methyl, n_csv])
    for c in AUX_CLASSES:
        ps = all_patients.get(c, [])
        rows.append([c, len(ps), sum(1 for p in ps if p.n_lesion > 0), sum(p.n_lesion for p in ps),
                     sum(1 for p in ps if p.tct_pdfs), sum(1 for p in ps if p.methyl_pdfs),
                     sum(1 for p in ps if p.has_csv)])
    lines.append("## 1. 每类概览")
    lines.append("")
    lines.append(markdown_table(
        ["类别", "患者目录数", "有病灶图患者数", "病灶图总数", "有TCT/DNA PDF", "有甲基化PDF", "匹配到CSV"],
        rows,
    ))
    lines.append("")
    lines.append(
        "> 病灶图只统计文件名含「病损/临床」的图片；病历单、病理单、病理图、首页等单据翻拍一律排除。"
    )
    lines.append("")

    # 2. 图片构成
    doc_total = sum(len(p.doc_images) for ps in all_patients.values() for p in ps)
    other_total = sum(len(p.other_images) for ps in all_patients.values() for p in ps)
    lines.append("## 2. 图片构成（关键）")
    lines.append("")
    lines.append(markdown_table(
        ["类型", "数量", "能否作为模型输入"],
        [
            ["病灶/临床照片", sum(p.n_lesion for ps in all_patients.values() for p in ps), "是"],
            ["病历/病理/首页等单据", doc_total, "否"],
            ["未归类图片", other_total, "需人工确认"],
        ],
    ))
    lines.append("")
    lines.append(
        "旧材料中「约 228 张病灶照片」的口径应当以本表为准重新核定：大量 jpg 是纸质单据翻拍，不是病灶照片。"
    )
    lines.append("")

    # 3. 缺失模态
    lines.append("## 3. 缺失模态分布（四类明确标签）")
    lines.append("")
    miss_counter: Counter[str] = Counter()
    for p in labeled:
        flags = set(p.missing_notes)
        if not p.tct_pdfs:
            flags.add("(实测无TCT PDF)")
        if not p.methyl_pdfs:
            flags.add("(实测无甲基化 PDF)")
        if p.n_lesion == 0:
            flags.add("(实测无病灶图)")
        if not p.has_csv:
            flags.add("(无CSV行)")
        elif p.n_tabular_valid == 0:
            flags.add("(CSV全空值)")
        for f in flags:
            miss_counter[f] += 1
        if not flags:
            miss_counter["四模态齐全"] += 1
    lines.append(markdown_table(
        ["缺失情况", "患者数", "占比"],
        [[k, v, f"{v / max(len(labeled), 1):.1%}"] for k, v in miss_counter.most_common()],
    ))
    lines.append("")

    # 4. CSV 对齐
    lines.append("## 4. CSV 与目录的对齐")
    lines.append("")
    align_rows = []
    for c in CLASS_ORDER + AUX_CLASSES:
        leftover = leftover_csv.get(c, {})
        align_rows.append([c, len(all_patients.get(c, [])), len(leftover),
                           "、".join(sorted(leftover)[:8]) or "-"])
    lines.append(markdown_table(["类别", "患者目录数", "CSV中未匹配行数", "未匹配姓名（最多8个）"], align_rows))
    lines.append("")
    lines.append(
        "未匹配行 = CSV 里有这个人、但目录里没有对应患者文件夹（或姓名写法不一致），这些行不能直接喂给模型。"
    )
    lines.append("")

    # 5. 跨类同名（含 unknown / 缺少数据）
    by_name: dict[str, list[str]] = defaultdict(list)
    for c in CLASS_ORDER + AUX_CLASSES:
        for p in all_patients.get(c, []):
            tag = c if not p.group else f"{c}/{p.group}"
            by_name[p.name].append(tag)
    conflicts = {n: cs for n, cs in by_name.items() if len(set(cs)) > 1}
    lines.append("## 5. 跨目录同名患者（数据泄漏风险）")
    lines.append("")
    if conflicts:
        lines.append(markdown_table(
            ["姓名", "出现的类别/分组", "次数"],
            [[n, "、".join(sorted(set(cs))), len(cs)] for n, cs in sorted(conflicts.items())],
        ))
        lines.append("")
        lines.append("同名出现在多个类别/分组 → 切分必须以姓名为 group 单位，unknown/缺少数据 里的同名患者也不能漏。")
    else:
        lines.append("各类别与分组之间没有同名重复。")
    lines.append("")

    # 6. 推断
    usable = [p for p in labeled if p.n_lesion > 0]
    lines.append("## 6. 建模可用样本推断")
    lines.append("")
    lines.append(markdown_table(
        ["口径", "数量"],
        [
            ["四类明确标签患者目录", len(labeled)],
            ["其中有病灶图的患者", len(usable)],
            ["其中同时有可用表格行的患者", sum(1 for p in usable if p.has_csv and p.n_tabular_valid > 0)],
            ["病灶图总数（可用于图像增强）", sum(p.n_lesion for p in usable)],
        ],
    ))
    lines.append("")
    lines.append("结论：按患者数而非图像数来衡量样本量；切分必须 leave-patient-out。")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="生成患者级数据清单")
    ap.add_argument("--data-root", default="data", help="data/ 根目录")
    ap.add_argument("--out-dir", default="reports", help="报告输出目录")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_patients: dict[str, list[Patient]] = {}
    leftover_csv: dict[str, dict[str, list[str]]] = {}
    for c in CLASS_ORDER + AUX_CLASSES:
        class_dir = root / c
        if not class_dir.exists():
            continue
        patients = scan_class(class_dir, c)
        all_patients[c] = patients
        # 重新加载一次 CSV，扣除已匹配的名字，剩下的就是未匹配
        csv_rows = load_class_csv(class_dir, c)
        matched_names = {p.name for p in patients if p.has_csv}
        leftover_csv[c] = {n: r for n, r in csv_rows.items() if n not in matched_names}

    report = build_report(all_patients, leftover_csv)
    (out_dir / "data_inventory.md").write_text(report, encoding="utf-8")

    flat = [p for c in all_patients for p in all_patients[c]]
    with (out_dir / "patients.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_key", "name", "visit", "label", "group", "folder", "n_lesion",
                    "n_doc_image", "n_tct_pdf", "n_methyl_pdf", "has_csv",
                    "n_tabular_valid", "missing_notes", "tag", "lesion_files"])
        for p in flat:
            w.writerow([
                f"{p.label}/{p.group + '/' if p.group else ''}{p.folder}", p.name, p.visit,
                p.label, p.group, p.folder,
                p.n_lesion, len(p.doc_images), len(p.tct_pdfs), len(p.methyl_pdfs),
                int(p.has_csv), p.n_tabular_valid, "|".join(p.missing_notes), p.tag,
                "|".join(p.lesion_images),
            ])

    summary = {
        "patients_by_class": {c: len(all_patients.get(c, [])) for c in all_patients},
        "lesion_images_total": sum(p.n_lesion for p in flat),
        "doc_images_total": sum(len(p.doc_images) for p in flat),
        "labeled_patients": sum(len(all_patients.get(c, [])) for c in LABELED_CLASSES),
    }
    (out_dir / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n报告已写出：{out_dir / 'data_inventory.md'}")
    print(f"患者表已写出：{out_dir / 'patients.csv'}（{len(flat)} 行）")


if __name__ == "__main__":
    main()
