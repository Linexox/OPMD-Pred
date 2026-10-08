"""Private offline report from a fixed SYSU checkpoint's saved validation split."""

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import torch
from PIL import Image

from opmd_pred.analysis.coalitions import coalition_outputs
from opmd_pred.analysis.shapley import intershap_summary, interaction_values, shapley_values
from opmd_pred.data.sysu import MODALITIES, assign_tct_ids, available_modalities
from opmd_pred.data.sysu_protocol import file_sha256, select_records, validate_manifest
from opmd_pred.model.ordinal import model_probabilities, probability_decision
from opmd_pred.scripts.train_multimodal import move_batch
from opmd_pred.scripts.train_sysu import make_loader, read_checkpoint, restore_checkpoint, write_json


CLASS_NAMES = ("正常/轻度", "中度", "重度")
MODALITY_NAMES = {"image": "口腔图片", "history": "病史", "tct": "TCT",
                  "dna": "DNA异常细胞比例", "methylation": "甲基化总数"}
TARGET_NAMES = {"severe": "重度概率", "at_least_moderate": "至少中度概率",
                "expected_severity": "期望严重度（0–2）"}
NOTICE = "私有医疗研究资料，仅限项目内使用；模型尚未验证临床效果，输出不构成诊断。"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New private directory: predictions.json and report.html")
    parser.add_argument("--sample-ids", nargs="+", help="Only IDs assigned to saved val; default all saved val rows")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--explain", action="store_true", help="Exact observed-mask coalitions, SHAP and full SII")
    parser.add_argument("--intershap", action="store_true", help="Add InterSHAP ratios; requires --explain")
    parser.add_argument("--target", choices=TARGET_NAMES, default="severe")
    parser.add_argument("--embed-images", action="store_true", help="Explicitly include private original image bytes")
    parser.add_argument("--include-labels", action="store_true", help="Explicitly include numeric reference labels")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.intershap and not args.explain):
        parser.error("Use positive batch-size; --intershap requires --explain")
    return args


def target_values(probabilities, target):
    if target == "severe":
        return probabilities[..., 2]
    if target == "at_least_moderate":
        return probabilities[..., 1:].sum(dim=-1)
    return probabilities[..., 1] + 2 * probabilities[..., 2]


def validate_cost(cost):
    if cost is None:
        return None
    matrix = torch.tensor(cost, dtype=torch.float32)
    if matrix.shape != (3, 3) or not torch.isfinite(matrix).all() or (matrix < 0).any() or matrix.diag().ne(0).any():
        raise ValueError("Saved cost snapshot must be finite nonnegative 3x3, row=true and column=action, diagonal=0")
    return matrix


def embedded_image(path, expected_sha):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Image changed after manifest validation")
    with Image.open(path) as opened:
        mime = Image.MIME.get(opened.format)
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Only JPEG/PNG/WebP original images can be embedded")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def initial_rows(records, requested_ids, modalities, excluded, include_labels):
    by_id = {record.sample_id: record for record in records}
    reasons = {entry["sample_id"]: entry["reason"] for entry in excluded if entry["split"] == "val"}
    rows = {}
    for sample_id in requested_ids:
        record = by_id[sample_id]
        observed = available_modalities(record)
        row = {"sample_id": sample_id, "split": "val", "evaluation_unit": "record",
               "status": "abstain" if sample_id in reasons else "pending",
               "abstain_reason": reasons.get(sample_id),
               "observed_modalities": [name for name in MODALITIES if observed[name]],
               "missing_modalities": [name for name in MODALITIES if not observed[name]],
               "selected_observed_modalities": [name for name in modalities if observed[name]],
               "unselected_modalities": [name for name in MODALITIES if name not in modalities],
               "probabilities": None, "argmax_action": None, "cost_action": None,
               "action_risks": None, "explanation": None}
        if include_labels:
            row["reference_label"] = record.label
        rows[sample_id] = row
    return rows


@torch.no_grad()
def infer_selected(model, loader, rows, args, cost):
    all_shapley, all_off_diagonal = [], []
    names = list(model.fusion.names)
    for batch in loader:
        batch = move_batch(batch, torch.device(args.device))
        if args.explain:
            logits, _ = coalition_outputs(model, batch, args.device)
            coalition_probabilities = model_probabilities(logits, model.head_kind)
            values = target_values(coalition_probabilities, args.target).cpu()
            phi = shapley_values(values, len(names))
            off_diagonal = interaction_values(values, len(names))
            full_sii, local_summary = intershap_summary(phi, off_diagonal)
            all_shapley.append(phi)
            all_off_diagonal.append(off_diagonal)
            probabilities = coalition_probabilities[:, -1].cpu()
        else:
            probabilities = model_probabilities(model(batch), model.head_kind).cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite probabilities; refusing to render predictions")
        if probabilities.ndim != 2 or probabilities.size(-1) != 3 or (probabilities < 0).any():
            raise ValueError("Predictions must be three nonnegative class probabilities")
        if not torch.allclose(probabilities.sum(dim=-1), torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=0):
            raise ValueError("Class probabilities do not sum to one")
        actions = probability_decision(probabilities)
        risks = probabilities @ cost if cost is not None else None
        cost_actions = probability_decision(probabilities, cost) if cost is not None else None
        for index, sample_id in enumerate(batch["sample_id"]):
            row = rows[sample_id]
            row.update(status="predicted", probabilities=probabilities[index].tolist(),
                       argmax_action=actions[index].item(),
                       cost_action=cost_actions[index].item() if cost_actions is not None else None,
                       action_risks=risks[index].tolist() if risks is not None else None)
            if args.explain:
                row["explanation"] = {"target": args.target, "modalities": names,
                    "baseline_value": values[index, 0].item(), "model_value": values[index, -1].item(),
                    "shapley": phi[index].tolist(), "pairwise_off_diagonal": off_diagonal[index].tolist(),
                    "full_sii": full_sii[index].tolist(),
                    "efficiency_residual": (phi[index].sum() - (values[index, -1] - values[index, 0])).item()}
                if args.intershap:
                    row["explanation"]["local_intershap"] = local_summary["local_intershap"][index]
    if args.intershap and all_shapley:
        _, summary = intershap_summary(torch.cat(all_shapley), torch.cat(all_off_diagonal))
        return {"scope": "selected_eligible_validation_records", **summary}
    return None


def escape(value):
    return html.escape(str(value), quote=True)


def format_number(value):
    return "未定义（贡献分母为零或资料不足）" if value is None else f"{value:.4f}"


def modality_list(names):
    return "、".join(escape(MODALITY_NAMES[name]) for name in names) if names else "无"


def explanation_html(explanation):
    if explanation is None:
        return "<p>未生成解释；无输入病例不进行归因。</p>"
    names = explanation["modalities"]
    rows = "".join(f"<tr><td>{escape(MODALITY_NAMES[name])}</td><td>{value:+.5f}</td></tr>"
                   for name, value in zip(names, explanation["shapley"]))
    header = "<tr><th>完整SII</th>" + "".join(f"<th>{escape(MODALITY_NAMES[n])}</th>" for n in names) + "</tr>"
    matrix = "".join("<tr><th>" + escape(MODALITY_NAMES[name]) + "</th>" +
                     "".join(f"<td class='{'diagonal' if first == second else 'pairwise'}'>{value:+.4f}</td>"
                             for second, value in enumerate(explanation["full_sii"][first])) + "</tr>"
                     for first, name in enumerate(names))
    ratio = (f"<p>局部InterSHAP：{escape(format_number(explanation['local_intershap']))}</p>"
             if "local_intershap" in explanation else "")
    return (f"<h3>解释目标：{escape(TARGET_NAMES[explanation['target']])}</h3>"
            f"<p>全遮蔽基准 {explanation['baseline_value']:.5f} → 观测视图 {explanation['model_value']:.5f}；"
            f"加和残差 {explanation['efficiency_residual']:.3g}</p>"
            f"<table><tr><th>模态</th><th>SHAP贡献</th></tr>{rows}</table>"
            f"<h3>主效应与成对交互</h3><div class='scroll'><table>{header}{matrix}</table></div>{ratio}"
            "<p class='note'>对角线为SII主效应，非对角线为成对交互；交互不代表因果关系。"
            "自然缺失模态的零贡献表示当前没有观察该输入，不表示检查阴性或该检查无用。"
            "此解释使用模态遮蔽游戏，不含背景分布积分，不是病灶定位。</p>")


def card_html(row, index):
    info = (f"<h2>记录 {escape(row['sample_id'])}</h2>"
            f"<p>已观测：{modality_list(row['observed_modalities'])}</p>"
            f"<p>自然缺失：{modality_list(row['missing_modalities'])}（未检测/未记录，不能视为阴性）</p>"
            f"<p>模型选用且已观测：{modality_list(row['selected_observed_modalities'])}；"
            f"未选入模型：{modality_list(row['unselected_modalities'])}</p>")
    label = f"<p>参考标签编号：{row['reference_label']}（仅显式请求时包含）</p>" if "reference_label" in row else ""
    if row["status"] == "abstain":
        content = "<p class='warning'>资料不足：所有选用模态都未观测，拒绝给出个体概率与归因；需补资料或转人工。</p>"
    else:
        bars = "".join(f"<div class='prob'><span>{escape(name)}</span>"
                       f"<meter min='0' max='1' value='{value:.8f}'></meter><b>{value:.2%}</b></div>"
                       for name, value in zip(CLASS_NAMES, row["probabilities"]))
        decision = f"<p>最大概率判定：<b>{escape(CLASS_NAMES[row['argmax_action']])}</b></p>"
        if row["cost_action"] is not None:
            risk = "、".join(f"{escape(name)}={value:.5f}" for name, value in zip(CLASS_NAMES, row["action_risks"]))
            decision += (f"<p>保存代价矩阵下的行动：<b>{escape(CLASS_NAMES[row['cost_action']])}</b>；"
                         f"各行动期望代价：{risk}。</p><p class='note'>成本偏好与概率判定分开；不是医学批准的流程。</p>")
        else:
            decision += "<p>未设定代价矩阵，不另给成本偏好行动。</p>"
        image = (f"<details><summary>显式附带的私有原图</summary><img alt='私有口腔原图' src='{escape(row['image_data_uri'])}'></details>"
                 if "image_data_uri" in row else "")
        content = bars + decision + image + explanation_html(row["explanation"])
        content = "<p class='note'>模型输出概率尚未校准，不代表经过验证的患病概率或个体误诊率。</p>" + content
    return f"<article id='record-{index}'>{info}{label}{content}</article>"


def render_html(report):
    cards = "".join(card_html(row, index) for index, row in enumerate(report["records"]))
    navigation = "".join(f"<a href='#record-{index}'>{escape(row['sample_id'])}</a> "
                         for index, row in enumerate(report["records"]))
    if not cards:
        cards = "<p>资料不足：保存验证集没有选中记录，没有生成个体预测。</p>"
    summary = report["intershap"]
    global_note = (f"<p>选中有效验证记录的全局InterSHAP：{escape(format_number(summary['global_intershap']))}；"
                   f"局部定义数 {summary['local_defined_count']}，未定义数 {summary['local_undefined_count']}。</p>"
                   if summary is not None else "<p>未请求或没有可解释记录，未生成全局InterSHAP。</p>")
    provenance = {key: value for key, value in report.items() if key not in {"records", "intershap"}}
    return """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<meta http-equiv='Content-Security-Policy' content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>SYSU 私有预测研究报告</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f3f5f8;color:#172435;margin:0;padding:24px}
main{max-width:1100px;margin:auto}article,header{background:white;border-radius:12px;padding:24px;margin-bottom:20px}
h1,h2,h3{line-height:1.3}.warning{padding:12px;border-left:4px solid #d39100;background:#fff4d8}
.note{color:#526379;font-size:14px}nav a{display:inline-block;margin:4px;padding:4px 8px;background:#e9eff9;border-radius:5px}
.prob{display:flex;align-items:center;gap:12px;margin:8px 0}.prob span{width:90px}.prob meter{width:60%;height:24px}
table{border-collapse:collapse;font-size:14px;margin:12px 0}th,td{padding:8px 12px;border:1px solid #d9e1ec;text-align:right}
th{background:#eef3fa}.diagonal{background:#f4eee6}.pairwise{background:#f5f8fc}.scroll{overflow-x:auto}
img{max-width:100%;max-height:600px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#235a98}
</style></head><body><main><header><h1>SYSU 验证记录预测研究报告</h1>""" + (
        f"<p class='warning'>{escape(NOTICE)}</p><p>来源分组探索性协议；患者身份及重复标签尚未核对。单位：记录；仅保存val，不访问test。"
        f"选中 {len(report['records'])} 条，预测 {report['predicted_count']} 条，资料不足 {report['abstained_count']} 条。</p>"
        f"{global_note}<nav aria-label='记录导航'>{navigation}</nav></header>{cards}"
        f"<details><summary>模型与数据协议来源</summary><pre>{escape(json.dumps(provenance, ensure_ascii=False, indent=2))}</pre></details>"
        "</main></body></html>")


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new private output directory; reports are not overwritten")
    checkpoint_sha = file_sha256(args.checkpoint)
    checkpoint = read_checkpoint(args.checkpoint)
    manifest, config = checkpoint["split_manifest"], checkpoint["model_config"]
    records = validate_manifest(manifest, args.csv, args.images)
    if not set(config["modalities"]) <= set(MODALITIES):
        raise ValueError("Only the five non-demographic modalities are permitted")
    splits, excluded = select_records(records, manifest, config["modalities"])
    if ({name: [r.sample_id for r in group] for name, group in splits.items()} != checkpoint["effective_split_ids"]
            or excluded != checkpoint["excluded"]):
        raise ValueError("Effective split IDs or modality exclusions differ from checkpoint")
    requested = manifest["split_ids"]["val"] if args.sample_ids is None else args.sample_ids
    if len(set(requested)) != len(requested) or not set(requested) <= set(manifest["split_ids"]["val"]):
        raise ValueError("Request distinct IDs from the saved validation split only; train/test selection is disabled")
    assign_tct_ids(records, checkpoint["tct_vocab"])
    rows = initial_rows(records, requested, config["modalities"], excluded, args.include_labels)
    selected = [record for record in splits["val"] if record.sample_id in rows]
    cost = validate_cost(checkpoint["cost_matrix"])
    intershap = None
    if selected:
        model, _ = restore_checkpoint(checkpoint, torch.device(args.device))
        loader = make_loader(selected, args.images, config, args.batch_size, 0, torch.device(args.device))
        intershap = infer_selected(model, loader, rows, args, cost)
    if args.embed_images:
        by_id = {record.sample_id: record for record in records}
        hashes = {entry["sample_id"]: entry["image_sha256"] for entry in manifest["records"]}
        for sample_id, row in rows.items():
            if row["status"] == "predicted" and "image" in row["selected_observed_modalities"]:
                row["image_data_uri"] = embedded_image(Path(args.images) / by_id[sample_id].image_path, hashes[sample_id])
    if file_sha256(args.checkpoint) != checkpoint_sha:
        raise ValueError("Checkpoint bytes changed during prediction")
    if file_sha256(args.csv) != manifest["csv_sha256"]:
        raise ValueError("Dataset bytes changed during prediction")
    report = {"format": "sysu-private-prediction-v1", "notice": NOTICE,
              "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": checkpoint["epoch"],
              "split_manifest_sha256": checkpoint["split_manifest_sha256"], "csv_sha256": manifest["csv_sha256"],
              "protocol": manifest["protocol"], "identity_verified": manifest["identity_verified"],
              "unresolved_image_label_conflicts": manifest["unresolved_image_label_conflicts"],
              "split": "val", "evaluation_unit": "record", "model_config": config,
              "selection_rule": checkpoint.get("selection_rule"), "encoder_source": checkpoint.get("encoder_source"),
              "cost_matrix": checkpoint["cost_matrix"], "cost_source": "checkpoint_saved_snapshot" if cost is not None else None,
              "cost_original_declared_path_not_revalidated": checkpoint.get("training_config", {}).get("cost_matrix"),
              "target": args.target, "explanation_game": "observed_mask_coalitions; natural_missing_stays_missing; no_causal_claim",
              "images_embedded": args.embed_images, "reference_labels_included": args.include_labels,
              "probability_status": "uncalibrated_model_outputs; not validated clinical risk",
              "predicted_count": sum(row["status"] == "predicted" for row in rows.values()),
              "abstained_count": sum(row["status"] == "abstain" for row in rows.values()),
              "intershap": intershap, "records": [rows[sample_id] for sample_id in requested]}
    output.mkdir(parents=True)
    write_json(output / "predictions.json", report)
    (output / "report.html").write_text(render_html(report), encoding="utf-8")
    print(f"Private validation report saved to {output}; no test inference was performed")


if __name__ == "__main__":
    main()
