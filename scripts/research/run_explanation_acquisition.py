"""Freeze source-based actions and replay five-modality coalitions on existing CPU models."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("acquisition_model_reference", Path(__file__).with_name("audit_september_comparison.py"))
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
from opmd_pred.model.fusion import MultimodalFusion
from opmd_pred.analysis.shapley import shapley_values, interaction_values, intershap_summary

INPUT = ROOT / "data/processed/sysu/rebuilt-20261009"
PREVIOUS = ROOT / "outputs/experiments/sysu/contrastive-ordinal-20261016/attempt1/evidence/run"
CACHE = ROOT / "outputs/experiments/sysu/raw-rebuild-20261009/completed_evidence/paired_views_final/baseline.npz"
QUALITY = ROOT / "outputs/experiments/sysu/moderate-quality-20261016/source-review"
DEFAULT_OUT = ROOT / "outputs/experiments/sysu/explanation-acquisition-20261017/attempt1"
NAMES = A.NAMES
SCALES = [1, 50, 100, 1, 50, 1000, 1, 20, 50]
ACTION_ORDER = ("image", "history", "tct", "dna", "tct_dna", "methylation")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_presence(row, entry):
    return [bool(entry["image_sha256"]), any(A.number(row[k]) is not None for k in A.HISTORY),
            row["TCT意见"].strip() not in A.MISSING, A.number(row["DNA倍体分析结果"]) is not None,
            A.number(row["甲基化总数"]) is not None]


def states(actions):
    """All nonempty initial action sets leaving at least two alternatives."""
    result = []
    for bits in range(1, 1 << len(actions)):
        remaining = [a for i, a in enumerate(actions) if not bits & (1 << i)]
        if len(remaining) < 2:
            continue
        initial = 0
        for i, action in enumerate(actions):
            if bits & (1 << i):
                initial |= action["mask"]
        result.append(dict(initial_mask=initial, candidates=remaining))
    return result


def prepare(out):
    if (out / "protocol.json").exists():
        raise ValueError("Protocol already exists; infer can resume, preparation cannot overwrite")
    cv = read(INPUT / "development_nested_cv_v3.json")
    previous = read(PREVIOUS / "protocol.json")
    if sha(INPUT / "sysu_samples.csv") != previous["csv_sha256"] or sha(INPUT / "development_nested_cv_v3.json") != previous["protocol_sha256"]:
        raise ValueError("Frozen data differs from model protocol")
    if sha(CACHE) != previous["baseline_cache_sha256"]:
        raise ValueError("Feature cache changed")
    entries = {e["sample_id"]: e for e in cv["development_entries"]}
    ids = list(entries)
    locked = set(cv["locked_previous_test_ids"] + cv["additional_related_locked_ids"])
    if len(ids) != 117 or set(ids) & locked:
        raise ValueError("Unexpected development scope")
    with (INPUT / "sysu_samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        raw = {r["sample_id"]: r for r in csv.DictReader(stream) if r["sample_id"] in entries}
    inv = {r["sample_id"]: r for r in read(INPUT / "inventory.private.json")["records"] if r["sample_id"] in entries}
    prov = {r["sample_id"]: r for r in read(INPUT / "record_provenance.private.json") if r["sample_id"] in entries}
    audited = {r["sample_id"]: r for r in read(QUALITY / "case-review.private.json")}
    duplicate_ids = {sid for r in read(QUALITY / "duplicate-review.private.json") if len(set(r["labels"])) > 1 for sid in r["sample_ids"]}
    photo = {r["sample_id"]: r for r in read(QUALITY / "photo-review.joined.private.json")}
    rng = random.Random(20261017)
    cases = []
    for sid in sorted(ids):
        observed = numeric_presence(raw[sid], entries[sid])
        observed_mask = sum((1 << i) for i, yes in enumerate(observed) if yes)
        report_candidates = [r for r in inv[sid]["report_candidates"] if r.get("source_name_present") and r.get("sha256") not in prov[sid].get("wrong_patient_report_shas", [])]
        dna = prov[sid].get("dna") or {}
        chosen = [r for r in report_candidates if r["sha256"] == dna.get("report_sha256")]
        joint = [r for r in chosen if r.get("has_dna_text") and r.get("has_tct_text")
                 and "TCT" in Path(r["path"]).stem.upper() and "DNA" in Path(r["path"]).stem.upper()
                 and dna.get("patient_header_matched")]
        merged = bool(observed[2] and observed[3] and joint)
        action_items = []
        for i, name in enumerate(NAMES):
            if not observed[i] or merged and name in ("tct", "dna"):
                continue
            evidence = []
            if name == "dna" and dna.get("patient_header_matched") and dna.get("report_path"):
                evidence = [dict(path=dna["report_path"], sha256=dna["report_sha256"], basis="selected_DNA_report")]
            elif name == "tct":
                evidence = [dict(path=r["path"], sha256=r["sha256"], basis="matched_report_contains_TCT_text") for r in report_candidates if r.get("has_tct_text") and not r.get("has_dna_text")]
            elif name == "methylation":
                evidence = [dict(path=r["path"], sha256=r["sha256"], basis="matched_methylation_report_filename") for r in report_candidates if "甲基化" in Path(r["path"]).name]
            confirmed = name in ("image", "history") or bool(evidence)
            action_items.append(dict(name=name, mask=1 << i, material_mapping_recorded=confirmed, report_evidence=evidence))
        if merged:
            action_items.append(dict(name="tct_dna", mask=12, material_mapping_recorded=True,
                report_evidence=[dict(path=r["path"], sha256=r["sha256"], basis="selected_matched_combined_TCT_DNA_report") for r in joint]))
        action_items.sort(key=lambda a: ACTION_ORDER.index(a["name"]))
        scenario_options = states(action_items)
        scenario = rng.choice(scenario_options) if scenario_options else None
        flags = []
        q = audited.get(sid)
        if q:
            if q["audit_grade_status"] in {"priority_grade_review", "mild_to_moderate_interval", "different_grades_by_specimen", "no_histology_found_in_bound_folder"}:
                flags.append(q["audit_grade_status"])
            if q["dna_collection_after_selected_pathology_report"]:
                flags.append("dna_collection_after_available_histology")
        if sid in duplicate_ids:
            flags.append("same_photo_different_grade")
        cases.append(dict(sample_id=sid, label=entries[sid]["label"], component=entries[sid]["cv_component"], observed_mask=observed_mask,
            observed=observed, actions=action_items, joined_tct_dna=merged,
            material_mapping_complete=all(a["material_mapping_recorded"] for a in action_items),
            source_inferred_not_hospital_verified=True, quality_flags=flags, photo_flags=photo.get(sid, {}).get("flags", []),
            primary_scenario=scenario, legal_start_count=len(scenario_options), missing_reason="unknown"))
    models = []
    for method in ("ordinal_pretrain", "balanced_reference"):
        for seed in (42, 43, 44):
            for fold in cv["folds"]:
                path = PREVIOUS / "runs" / f"{method}_seed{seed}" / f"fold{fold['fold']}/refit/classifier.pt"
                prediction = path.parent.parent / "outer_predictions.json"
                models.append(dict(method=method, seed=seed, fold=fold["fold"], path=str(path), sha256=sha(path),
                    archived_predictions=str(prediction), archived_predictions_sha256=sha(prediction),
                    train_ids=fold["outer_train"]["sample_ids"], held_ids=fold["outer_holdout"]["sample_ids"]))
    experiment = dict(format="sysu-explanation-acquisition-protocol", prepared_unix=time.time(),
        decision_rule="fixed_cumulative_median_0.5", target="severe_score", training=False, gpu_used=False,
        random_start_seed=20261017, model_methods=["ordinal_pretrain", "balanced_reference"], model_seeds=[42,43,44],
        minimum_historical_support=10, score_tie_tolerance=1e-12, fallback="fixed_order_if_any_candidate_support_missing_or_all_interaction_scores_zero",
        action_type_order=list(ACTION_ORDER), cases=cases, models=models,
        source_sha={"csv":sha(INPUT / "sysu_samples.csv"), "cv":sha(INPUT / "development_nested_cv_v3.json"), "feature_cache":sha(CACHE),
            "inventory":sha(INPUT / "inventory.private.json"), "provenance":sha(INPUT / "record_provenance.private.json"),
            "quality_cases":sha(QUALITY / "case-review.private.json"), "quality_duplicates":sha(QUALITY / "duplicate-review.private.json"),
            "plan":sha(ROOT / "docs/daily/2026-10-17/experiment-plan.md")},
        source_paths={"input":str(INPUT),"cache":str(CACHE)},
        primary_unit="one_added_information_action_with_verified_combined_TCT_DNA_merged",
        secondary_scopes=["all_primary", "material_mapping_recorded", "without_prelisted_quality_flags", "all_legal_starts_record_weighted"],
        limitations=["material_mapping_recorded is not clinical same-visit/site confirmation", "original trained labels remain unchanged", "all missing reasons unknown unless future evidence supports them"])
    write(out / "protocol.json", experiment)
    snapshot = out / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    used = [Path(__file__), Path(A.__file__), ROOT / "src/opmd_pred/model/fusion.py", ROOT / "src/opmd_pred/analysis/shapley.py", ROOT / "docs/daily/2026-10-17/experiment-plan.md"]
    write(out / "source_snapshot.json", {str(f.relative_to(ROOT)):sha(f) for f in used})
    for f in used:
        dest = snapshot / f.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    write(out / "environment.json", dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        preexisting_git_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        protocol_sha256=sha(out / "protocol.json")))
    eligible = [c for c in cases if c["primary_scenario"]]
    summary = dict(records=len(cases), with_inputs=sum(bool(c["observed_mask"]) for c in cases),
        joined_tct_dna=sum(c["joined_tct_dna"] for c in cases), primary_records=len(eligible),
        primary_class_counts=[sum(c["label"]==y for c in eligible) for y in range(3)],
        mapped_records=sum(c["material_mapping_complete"] for c in eligible),
        mapped_class_counts=[sum(c["material_mapping_complete"] and c["label"]==y for c in eligible) for y in range(3)],
        without_quality_flags=sum(not c["quality_flags"] for c in eligible),
        source_status_not_clinically_verified=True, models=len(models), protocol_sha256=sha(out / "protocol.json"), predictions_computed=False)
    write(out / "preparation-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


def assemble_all(cases, raw, fit, features, index):
    batches = []
    for case in cases:
        sid=case["sample_id"]; r=raw[sid]
        history,hmask=A.numeric([A.number(r[k]) for k in A.HISTORY],SCALES)
        ratio,threshold=A.number(r["DNA倍体分析结果"]),A.number(r["dna_threshold"])
        if (ratio is None)!=(threshold is None) or threshold not in (None,2.3,2.5):
            raise ValueError("DNA availability or threshold changed")
        dna,dmask=A.numeric([ratio,None if threshold is None else float(threshold==2.5)],[1,1])
        methyl,mmask=A.numeric([A.number(r["甲基化总数"])],[10])
        text=r["TCT意见"].strip();text="" if text in A.MISSING else text
        batches.append(dict(history=history,history_mask=hmask,dna=dna,dna_mask=dmask,methylation=methyl,methylation_mask=mmask,
            tct=torch.tensor([fit["tct_vocab"].get(text,1) if text else 0]),tct_present=torch.tensor([bool(text)]),
            image_features=torch.from_numpy(features[index[sid],0].copy())[None],image_present=torch.tensor([case["observed"][0]])))
    return {key:torch.cat([b[key] for b in batches]) for key in batches[0]}


@torch.inference_mode()
def all_coalitions(model,batch):
    f=model.fusion
    embeddings={"image":f.image_projection(batch["image_features"]),"tct":f.tct_encoder(batch["tct"])}
    observed={"image":batch["image_present"],"tct":batch["tct_present"]}
    for name in ("history","dna","methylation"):
        embeddings[name]=getattr(f,name+"_encoder")(batch[name],batch[name+"_mask"])
        observed[name]=batch[name+"_mask"].any(1)
    outputs=[]
    for mask in range(32):
        present={name:observed[name]&bool(mask&(1<<i)) for i,name in enumerate(NAMES)}
        logits=model.ordinal_head(MultimodalFusion.fuse_embeddings(f,embeddings,present))
        if model.config["head"]=="ordinal":
            q=logits.sigmoid();p=torch.stack([1-q[:,0],q[:,0]-q[:,1],q[:,1]],dim=-1)
        else:p=logits.softmax(-1)
        outputs.append(p)
    return torch.stack(outputs,dim=1),observed


def infer(out):
    started=time.monotonic();torch.set_num_threads(2)
    protocol=read(out / "protocol.json");receipt=read(out / "environment.json")
    if sha(out / "protocol.json")!=receipt["protocol_sha256"]:raise ValueError("Prospective protocol changed")
    for key,path in [("csv",INPUT/"sysu_samples.csv"),("cv",INPUT/"development_nested_cv_v3.json"),("feature_cache",CACHE)]:
        if sha(path)!=protocol["source_sha"][key]:raise ValueError("Input changed")
    cases=protocol["cases"];ids=[c["sample_id"] for c in cases]
    with (INPUT/"sysu_samples.csv").open(encoding="utf-8-sig",newline="") as stream:
        raw={r["sample_id"]:r for r in csv.DictReader(stream) if r["sample_id"] in ids}
    with np.load(CACHE,allow_pickle=False) as c:
        feature_ids=c["sample_ids"].tolist();features=c["features"].copy()
    if set(feature_ids)!=set(ids):raise ValueError("Features have different records")
    index={sid:i for i,sid in enumerate(feature_ids)}
    logs=[]
    for identity in protocol["models"]:
        target=out/"coalitions"/f"{identity['method']}_seed{identity['seed']}_fold{identity['fold']}.npz"
        checkpath=target.with_suffix(".json")
        if target.exists() and checkpath.exists():
            existing=read(checkpath)
            if existing["checkpoint_sha256"]!=identity["sha256"] or sha(target)!=existing["coalitions_sha256"]:raise ValueError("Resume output differs")
            logs.append(existing);continue
        if sha(identity["path"])!=identity["sha256"] or sha(identity["archived_predictions"])!=identity["archived_predictions_sha256"]:
            raise ValueError("Checkpoint or archived output changed")
        artifact=torch.load(identity["path"],map_location="cpu",weights_only=True)
        if artifact["contract"]["train_ids"]!=identity["train_ids"] or set(artifact["fit"]["fit_ids"])&set(identity["held_ids"]):raise ValueError("Training/holdout changed")
        model=A.FeatureModel(artifact["config"],artifact["fit"]).eval();model.load_state_dict(artifact["state"],strict=True)
        batch=assemble_all(cases,raw,artifact["fit"],features,index)
        probs,observed=all_coalitions(model,batch)
        if not bool(torch.isfinite(probs).all()) or float(probs.min()) < -1e-7 or float((probs.sum(-1)-1).abs().max())>1e-6:raise ValueError("Invalid probabilities")
        shap=shapley_values(probs[:,:,2],5);inter=interaction_values(probs[:,:,2],5)
        residual=float((shap.sum(1)-(probs[:,31,2]-probs[:,0,2])).abs().max())
        symmetric=float((inter-inter.transpose(1,2)).abs().max())
        if residual>1e-5 or symmetric>1e-7:raise ValueError("Attribution identity failed")
        for i,c in enumerate(cases):
            if [bool(observed[n][i]) for n in NAMES]!=c["observed"]:raise ValueError("Input presence changed")
            for j in range(5):
                if not c["observed"][j] and (abs(float(shap[i,j]))>1e-6 or float(inter[i,j].abs().max())>1e-6):raise ValueError("Natural missing attribution is nonzero")
        archived={r["sample_id"]:r for r in read(identity["archived_predictions"])["records"]}
        maximum=0.;replayed=0
        for sid in identity["held_ids"]:
            c=cases[ids.index(sid)];r=archived[sid]
            if c["observed_mask"]:
                maximum=max(maximum,max(abs(a-b) for a,b in zip(probs[ids.index(sid),31].tolist(),r["probabilities"])));replayed+=1
            elif r["probabilities"] is not None:raise ValueError("All missing no longer abstains")
        if maximum>1e-4:raise ValueError(f"Cannot reproduce {identity['method']}: {maximum}")
        decisions=((probs[:,:,1]+probs[:,:,2]>.5).long()+(probs[:,:,2]>.5).long()).numpy()
        target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,sample_ids=np.array(ids),probabilities=probs.numpy(),shap=shap.numpy(),interactions=inter.numpy(),decisions=decisions,
            observed_masks=np.array([c["observed_mask"] for c in cases],dtype=np.int64))
        log=dict(method=identity["method"],seed=identity["seed"],fold=identity["fold"],checkpoint_sha256=identity["sha256"],
            coalitions_sha256=sha(target),protocol_sha256=receipt["protocol_sha256"],replayed_heldout_records=replayed,
            maximum_probability_error=maximum,maximum_shap_efficiency_error=residual,maximum_symmetry_error=symmetric,
            all_records_computed=len(cases),elapsed_seconds=time.monotonic()-started)
        write(checkpath,log);logs.append(log)
        write(out/"inference-progress.json",dict(completed=len(logs),total=len(protocol["models"]),latest=log))
        print(json.dumps(log,ensure_ascii=True),flush=True)
    write(out/"inference-summary.json",dict(completed_models=len(logs),logs=logs,
        maximum_probability_error=max(x["maximum_probability_error"] for x in logs),
        maximum_shap_efficiency_error=max(x["maximum_shap_efficiency_error"] for x in logs),
        heldout_vectors_reproduced=sum(x["replayed_heldout_records"] for x in logs),
        all_coalition_vectors=len(cases)*32*len(logs),no_locked_test_evaluated=True,training=False,gpu_used=False,elapsed_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","infer"]);parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    (prepare if args.stage=="prepare" else infer)(args.output)


if __name__=="__main__":main()
