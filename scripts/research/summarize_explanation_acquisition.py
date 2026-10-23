"""Evaluate prospective acquisition rules using cached frozen-model coalition outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "src"))
SPEC=importlib.util.spec_from_file_location("acquisition_protocol",Path(__file__).with_name("run_explanation_acquisition.py"))
R=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(R)
METHODS=("ordinal_pretrain","balanced_reference")
POLICIES=("fixed","random","historical_shap","historical_interaction")
SEEDS=(42,43,44)


def onehot(index):
    return np.eye(3,dtype=float)[int(index)]


def metric_vector(y,outcomes):
    """Rows are class-decision distributions; random policy is exact expectation."""
    n=len(y)
    if n==0:return {k:None for k in ("accuracy","moderate_recall","severe_recall","severe_to_low","two_level_error","mae")}|{"records":0,"class_counts":[0,0,0]}
    yy=np.asarray(y,dtype=int);o=np.asarray(outcomes,dtype=float)
    eligible=int((o.sum(1)>0).sum())
    counts=[int((yy==j).sum()) for j in range(3)]
    recall=[float(o[yy==j,j].mean()) if counts[j] else None for j in range(3)]
    return dict(records=n,class_counts=counts,eligible_records=eligible,abstained_records=n-eligible,abstention_rate=(n-eligible)/n,accuracy=float(o[np.arange(n),yy].mean()),moderate_recall=recall[1],severe_recall=recall[2],
        severe_to_low=float(o[yy==2,0].mean()) if counts[2] else None,
        two_level_error=float(np.where(yy==2,o[:,0],np.where(yy==0,o[:,2],0)).mean()),
        mae=float((o*np.abs(np.arange(3)[None]-yy[:,None])).sum()/eligible) if eligible else None)


def fit_history(cases,ids,decisions,shap,interactions,min_support):
    """Only training rows and their labels can enter the recommendation history."""
    support=[[] for _ in range(5)];pairs=[[[] for _ in range(5)] for _ in range(5)]
    gain=defaultdict(list);severe_gain=defaultdict(list)
    for i in ids:
        case=cases[i]
        for j in range(5):
            if case["observed_mask"]&(1<<j):support[j].append(float(abs(shap[i,j])))
            for k in range(5):
                if j!=k and case["observed_mask"]&(1<<j) and case["observed_mask"]&(1<<k):pairs[j][k].append(float(abs(interactions[i,j,k])))
        scenario=case["primary_scenario"]
        if scenario is None:continue
        before=int(decisions[i,scenario["initial_mask"]]);label=case["label"]
        for action in scenario["candidates"]:
            after=int(decisions[i,scenario["initial_mask"]|action["mask"]])
            gain[action["name"]].append(float(after==label)-float(before==label))
            if label==2:severe_gain[action["name"]].append(float(before==0)-float(after==0))
    single=[dict(count=len(v),mean_abs=float(np.mean(v)) if v else None) for v in support]
    pair=[[dict(count=len(v),mean_abs=float(np.mean(v)) if v else None) for v in row] for row in pairs]
    fixed={name:dict(count=len(gain[name]),accuracy_gain=float(np.mean(gain[name])) if gain[name] else 0.,
        severe_count=len(severe_gain[name]),severe_error_reduction=float(np.mean(severe_gain[name])) if severe_gain[name] else 0.) for name in R.ACTION_ORDER}
    return dict(training_ids=[cases[i]["sample_id"] for i in ids],single=single,pairs=pair,fixed=fixed,minimum_support=min_support)


def choose(policy,initial_mask,candidates,history):
    """Never receives this record's outcome, hidden values, or pathology label."""
    def fixed_key(action):
        value=history["fixed"][action["name"]]
        return value["accuracy_gain"],value["severe_error_reduction"],-R.ACTION_ORDER.index(action["name"])
    preferred=max(candidates,key=fixed_key)
    if policy=="fixed":return preferred,False,None,{}
    scores={}
    for action in candidates:
        new=[j for j in range(5) if action["mask"]&(1<<j) and not initial_mask&(1<<j)]
        if policy=="historical_shap":
            values=[history["single"][j] for j in new]
        elif policy=="historical_interaction":
            values=[history["pairs"][j][k] for j in new for k in range(5) if initial_mask&(1<<k)]
        else:raise ValueError("Unknown deterministic policy")
        if not values or any(v["count"]<history["minimum_support"] or v["mean_abs"] is None for v in values):
            return preferred,True,"insufficient_training_support",scores
        scores[action["name"]]=float(sum(v["mean_abs"] for v in values))
    best=max(scores.values())
    if best<=1e-12:return preferred,True,"zero_history_score",scores
    tied=[a for a in candidates if abs(scores[a["name"]]-best)<=1e-12]
    if len(tied)>1:return max(tied,key=fixed_key),True,"tie_uses_fixed_order",scores
    return tied[0],False,None,scores


def evaluate_scenario(case,index,scenario,decisions,probabilities,history,retain_details=True):
    initial=scenario["initial_mask"];candidates=scenario["candidates"]
    before=int(decisions[index,initial]);full=int(decisions[index,case["observed_mask"]])
    output=dict(sample_id=case["sample_id"],label=case["label"],component=case["component"],initial_mask=initial,
        before=onehot(before).tolist(),full=onehot(full).tolist(),outcomes={},choices={},fallback={},fallback_reason={},candidate_scores={},
        material_mapping_complete=case["material_mapping_complete"],quality_flags=case["quality_flags"])
    candidate_preds={a["name"]:int(decisions[index,initial|a["mask"]]) for a in candidates}
    candidate_outcomes=np.array([onehot(candidate_preds[a["name"]]) for a in candidates])
    output["outcomes"]["random"]=candidate_outcomes.mean(0).tolist()
    for policy in ("fixed","historical_shap","historical_interaction"):
        action,fallback,reason,scores=choose(policy,initial,candidates,history)
        output["outcomes"][policy]=onehot(candidate_preds[action["name"]]).tolist()
        output["choices"][policy]=action["name"];output["fallback"][policy]=fallback
        output["fallback_reason"][policy]=reason;output["candidate_scores"][policy]=scores
    # Post-hoc labels may describe best possible one-step recovery, never choose().
    oracle=max(range(len(candidates)),key=lambda i:(candidate_preds[candidates[i]["name"]]==case["label"],-abs(candidate_preds[candidates[i]["name"]]-case["label"]),-i))
    output["outcomes"]["posthoc_best_not_a_policy"]=candidate_outcomes[oracle].tolist()
    output["candidate_predictions"]=candidate_preds
    output["any_candidate_can_correct"]=before!=case["label"] and any(p==case["label"] for p in candidate_preds.values())
    output["any_candidate_can_harm"]=before==case["label"] and any(p!=case["label"] for p in candidate_preds.values())
    if retain_details:
        output["before_probabilities"]=probabilities[index,initial].tolist()
        output["full_probabilities"]=probabilities[index,case["observed_mask"]].tolist()
        output["candidate_probabilities"]={a["name"]:probabilities[index,initial|a["mask"]].tolist() for a in candidates}
    return output


def scope_result(rows):
    y=[r["label"] for r in rows]
    result={key:metric_vector(y,[r[key] for r in rows]) for key in ("before","full")}
    for policy in (*POLICIES,"posthoc_best_not_a_policy"):
        result[policy]=metric_vector(y,[r["outcomes"][policy] for r in rows])
        result[policy]["fallback_rate"]=float(np.mean([r.get("fallback",{}).get(policy,False) for r in rows])) if rows and policy in ("historical_shap","historical_interaction") else None
        result[policy]["action_counts"]=dict(Counter(r.get("choices",{}).get(policy) for r in rows if r.get("choices",{}).get(policy)))
        result[policy]["corrected_expected"]=float(sum(r.get("transition_rates",{}).get(policy,{}).get("corrected",(1-r["before"][r["label"]])*r["outcomes"][policy][r["label"]]) for r in rows))
        result[policy]["harmed_expected"]=float(sum(r.get("transition_rates",{}).get(policy,{}).get("harmed",r["before"][r["label"]]*(1-r["outcomes"][policy][r["label"]])) for r in rows))
    result["any_candidate_can_correct_records"]=sum(r.get("any_candidate_can_correct",False) for r in rows)
    result["any_candidate_can_harm_records"]=sum(r.get("any_candidate_can_harm",False) for r in rows)
    return result


def mean_results(seed_results):
    result={}
    for policy in ("before","full",*POLICIES,"posthoc_best_not_a_policy"):
        ref=seed_results[0][policy]
        item={k:ref[k] for k in ("records","class_counts","eligible_records","abstained_records")}
        for key in ("accuracy","moderate_recall","severe_recall","severe_to_low","two_level_error","mae","abstention_rate","fallback_rate","corrected_expected","harmed_expected"):
            values=[r[policy].get(key) for r in seed_results]
            if any(v is not None for v in values):item[key]=float(np.mean([v for v in values if v is not None]))
        result[policy]=item
    for key in ("any_candidate_can_correct_records","any_candidate_can_harm_records"):
        result[key]=float(np.mean([r[key] for r in seed_results]))
    return result


def paired_intervals(seed_rows,n=2000):
    """Cluster bootstrap over existing source groups, not synthetic scenarios."""
    ids=[r["sample_id"] for r in seed_rows[0]]
    if not ids:return {}
    if any([r["sample_id"] for r in rows]!=ids for rows in seed_rows):raise ValueError("Unpaired records")
    groups=defaultdict(list)
    for i,r in enumerate(seed_rows[0]):groups[r["component"]].append(i)
    keys=sorted(groups);rng=np.random.default_rng(20261017)
    samples=[np.concatenate([groups[keys[j]] for j in rng.integers(0,len(keys),len(keys))]) for _ in range(n)]
    y=np.array([r["label"] for r in seed_rows[0]])
    result={}
    for reference in ("fixed","random","historical_shap"):
        diffs={}
        for metric in ("accuracy","moderate_recall","severe_to_low","two_level_error"):
            arrays=[]
            for rows in seed_rows:
                inter=np.array([r["outcomes"]["historical_interaction"] for r in rows]);ref=np.array([r["outcomes"][reference] for r in rows])
                if metric=="accuracy":v=inter[np.arange(len(y)),y]-ref[np.arange(len(y)),y];mask=np.ones(len(y),bool)
                elif metric=="moderate_recall":v=inter[:,1]-ref[:,1];mask=y==1
                elif metric=="severe_to_low":v=inter[:,0]-ref[:,0];mask=y==2
                else:v=np.where(y==2,inter[:,0]-ref[:,0],np.where(y==0,inter[:,2]-ref[:,2],0));mask=np.ones(len(y),bool)
                arrays.append(v)
            v=np.mean(arrays,axis=0)
            bootstrap=[float(v[ix][mask[ix]].mean()) for ix in samples if mask[ix].any()]
            diffs[metric]=dict(difference=float(v[mask].mean()) if mask.any() else None,
                lower=float(np.quantile(bootstrap,.025)) if bootstrap else None,upper=float(np.quantile(bootstrap,.975)) if bootstrap else None,
                bootstrap_defined=len(bootstrap),groups=len(keys),records=len(ids))
        result[reference]=diffs
    return result


def check_gate(means,per_seed):
    inter=means["historical_interaction"];checks={}
    for ref in ("fixed","random"):
        base=means[ref]
        directions=sum(s["historical_interaction"]["severe_to_low"]<s[ref]["severe_to_low"]-1e-12 for s in per_seed)
        checks[ref]=dict(severe_to_low_reduced=inter["severe_to_low"]<base["severe_to_low"]-1e-12,
            accuracy_within_two_points=inter["accuracy"]>=base["accuracy"]-.02-1e-12,
            moderate_not_lower=inter["moderate_recall"]>=base["moderate_recall"]-1e-12,
            two_level_not_higher=inter["two_level_error"]<=base["two_level_error"]+1e-12,
            at_least_two_seed_directions=directions>=2,seeds_with_lower_severe_to_low=directions)
    passed=all(all(v for k,v in c.items() if k!="seeds_with_lower_severe_to_low") for c in checks.values())
    return dict(passed=passed,comparators=checks)


def summarize(out):
    start=time.monotonic();p=R.read(out/"protocol.json");cases=p["cases"];index={c["sample_id"]:i for i,c in enumerate(cases)}
    if R.sha(out/"protocol.json")!=R.read(out/"environment.json")["protocol_sha256"]:raise ValueError("Protocol changed")
    analysis_config=dict(bootstrap_samples=2000,bootstrap_seed=20261017,
        primary_scenario="one prospectively frozen start per record",secondary="all legal starts averaged within record",
        quality_sensitivity="prelisted flags only; no source data edits",source_sha256=R.sha(Path(__file__)))
    R.write(out/"analysis-settings.json",analysis_config)
    tables={};detailed={};explanations={};gates={}
    for method in METHODS:
        primary_seeds=[];mapped_seeds=[];quality_seeds=[];all_start_seeds=[];full_seeds=[]
        oof_shap=[];oof_inter=[];oof_pred=[];oof_p=[]
        rows_by_seed=[]
        for seed in SEEDS:
            rows=[];full=[];all_starts=[]
            shap_all=np.zeros((len(cases),5));inter_all=np.zeros((len(cases),5,5));pred_all=np.zeros(len(cases),dtype=int);prob_all=np.zeros((len(cases),3))
            for identity in [m for m in p["models"] if m["method"]==method and m["seed"]==seed]:
                name=f"{method}_seed{seed}_fold{identity['fold']}"
                file=out/"coalitions"/f"{name}.npz";log=R.read(file.with_suffix(".json"))
                if R.sha(file)!=log["coalitions_sha256"] or R.sha(identity["path"])!=identity["sha256"]:raise ValueError("Inference cache changed")
                with np.load(file,allow_pickle=False) as cache:
                    if cache["sample_ids"].tolist()!=list(index):raise ValueError("Case order changed")
                    probs=cache["probabilities"].copy();decisions=cache["decisions"].copy();shap=cache["shap"].copy();inter=cache["interactions"].copy()
                train=[index[sid] for sid in identity["train_ids"]];held=[index[sid] for sid in identity["held_ids"]]
                if set(train)&set(held):raise ValueError("Train/held overlap")
                history=fit_history(cases,train,decisions,shap,inter,p["minimum_historical_support"])
                R.write(out/"histories"/f"{name}.json",history)
                for i in held:
                    c=cases[i];shap_all[i]=shap[i];inter_all[i]=inter[i];pred_all[i]=decisions[i,c["observed_mask"]];prob_all[i]=probs[i,c["observed_mask"]]
                    full.append(dict(sample_id=c["sample_id"],label=c["label"],outcome=onehot(pred_all[i]).tolist() if c["observed_mask"] else [0.,0.,0.]))
                    if c["primary_scenario"] is None:continue
                    r=evaluate_scenario(c,i,c["primary_scenario"],decisions,probs,history);r.update(seed=seed,fold=identity["fold"]);rows.append(r)
                    starts=[evaluate_scenario(c,i,s,decisions,probs,history,False) for s in R.states(c["actions"])]
                    a=dict(sample_id=c["sample_id"],label=c["label"],component=c["component"],outcomes={},transition_rates={},before=np.mean([z["before"] for z in starts],0).tolist(),
                        full=r["full"],choices={},fallback={},quality_flags=c["quality_flags"],material_mapping_complete=c["material_mapping_complete"])
                    for pol in (*POLICIES,"posthoc_best_not_a_policy"):
                        a["outcomes"][pol]=np.mean([z["outcomes"][pol] for z in starts],0).tolist()
                        a["transition_rates"][pol]=dict(
                            corrected=float(np.mean([(1-z["before"][c["label"]])*z["outcomes"][pol][c["label"]] for z in starts])),
                            harmed=float(np.mean([z["before"][c["label"]]*(1-z["outcomes"][pol][c["label"]]) for z in starts])))
                    a["any_candidate_can_correct"]=float(np.mean([z["any_candidate_can_correct"] for z in starts]))
                    a["any_candidate_can_harm"]=float(np.mean([z["any_candidate_can_harm"] for z in starts]))
                    for pol in ("historical_shap","historical_interaction"):a["fallback"][pol]=float(np.mean([z["fallback"][pol] for z in starts]))
                    all_starts.append(a)
            rows.sort(key=lambda r:r["sample_id"]);full.sort(key=lambda r:r["sample_id"]);all_starts.sort(key=lambda r:r["sample_id"])
            rows_by_seed.append(rows)
            primary_seeds.append(scope_result(rows));mapped_seeds.append(scope_result([r for r in rows if r["material_mapping_complete"]]));quality_seeds.append(scope_result([r for r in rows if not r["quality_flags"]]));all_start_seeds.append(scope_result(all_starts))
            full_seeds.append(metric_vector([r["label"] for r in full],[r["outcome"] for r in full]))
            oof_shap.append(shap_all);oof_inter.append(inter_all);oof_pred.append(pred_all);oof_p.append(prob_all)
            R.write(out/"decisions"/f"{method}_seed{seed}.json",rows)
            detailed[f"{method}_seed{seed}"]=rows
        means=mean_results(primary_seeds)
        tables[method]=dict(primary=dict(per_seed=primary_seeds,mean=means,intervals=paired_intervals(rows_by_seed)),
            material_mapping_recorded=dict(per_seed=mapped_seeds,mean=mean_results(mapped_seeds)),
            without_prelisted_quality_flags=dict(per_seed=quality_seeds,mean=mean_results(quality_seeds)),
            all_legal_starts_record_weighted=dict(per_seed=all_start_seeds,mean=mean_results(all_start_seeds)),
            original_full_117=full_seeds)
        gates[method]=check_gate(means,primary_seeds)
        eligible=np.array([bool(c["observed_mask"]) for c in cases]);ph=np.array(oof_shap)[:,eligible];it=np.array(oof_inter)[:,eligible]
        group_explanations={}
        for group,condition in [("severe_predicted_low",lambda c,z:c["label"]==2 and z==0),("moderate_predicted_low",lambda c,z:c["label"]==1 and z==0),("correct",lambda c,z:c["label"]==z)]:
            vals=[]
            for j in range(3):
                vals.extend(oof_shap[j][i].tolist() for i,c in enumerate(cases) if c["observed_mask"] and condition(c,int(oof_pred[j][i])))
            group_explanations[group]=dict(record_model_observations=len(vals),mean_signed_shap=np.mean(vals,axis=0).tolist() if vals else None)
        consistency=[]
        allphi=np.array(oof_shap)
        for j in range(5):
            present=np.array([bool(c["observed_mask"]&(1<<j)) for c in cases]);v=allphi[:,present,j]
            stable=(v>=0).all(0)|(v<=0).all(0)
            consistency.append(dict(modality=R.NAMES[j],present_records=int(present.sum()),same_sign_fraction=float(stable.mean()),
                small_all_seeds_records=int((np.abs(v).max(0)<1e-4).sum()),mean_abs=float(np.abs(v).mean())))
        explanations[method]=dict(modalities=list(R.NAMES),target="severe_score",mean_signed_shap=ph.mean((0,1)).tolist(),
            mean_abs_shap=np.abs(ph).mean((0,1)).tolist(),mean_abs_pairwise_interaction=np.abs(it).mean((0,1)).tolist(),
            same_record_sign_consistency=consistency,by_full_prediction_group=group_explanations,
            oof_records=112,source_models=15)
        R.write(out/"oof_explanations"/f"{method}.json",[dict(sample_id=c["sample_id"],label=c["label"],observed=c["observed"],quality_flags=c["quality_flags"],
            per_seed=[dict(seed=SEEDS[j],prediction=int(oof_pred[j][i]) if c["observed_mask"] else None,probabilities=oof_p[j][i].tolist() if c["observed_mask"] else None,
                shap=oof_shap[j][i].tolist(),interactions=oof_inter[j][i].tolist()) for j in range(3)]) for i,c in enumerate(cases)])
    # Examples are descriptive, deterministic, and cannot change policies or outcomes.
    rows=detailed["ordinal_pretrain_seed42"];by_id={c["sample_id"]:c for c in cases};full_ex=R.read(out/"oof_explanations/ordinal_pretrain.json")
    predicates=[("severe_predicted_low",lambda c:c["label"]==2 and c["per_seed"][0]["prediction"]==0),
                ("moderate_predicted_low",lambda c:c["label"]==1 and c["per_seed"][0]["prediction"]==0)]
    examples=[]
    for name,fn in predicates:
        candidates=[c for c in full_ex if fn(c)]
        candidates.sort(key=lambda c:(bool(c["quality_flags"]),c["sample_id"]))
        if candidates:examples.append(dict(type=name,record=candidates[0]))
    for name,fn in [("acquisition_corrected",lambda r:not r["before"][r["label"]] and r["outcomes"]["historical_interaction"][r["label"]]==1),
                    ("acquisition_harmed",lambda r:r["before"][r["label"]] and r["outcomes"]["historical_interaction"][r["label"]]==0)]:
        candidates=[r for r in rows if fn(r)]
        candidates.sort(key=lambda r:(bool(r["quality_flags"]),not r["material_mapping_complete"],r["sample_id"]))
        if candidates:examples.append(dict(type=name,record=candidates[0]))
    R.write(out/"examples.private.json",examples)
    R.write(out/"results.json",dict(tables=tables,explanations=explanations,gates=gates,settings=analysis_config,
        protocol_sha256=R.sha(out/"protocol.json"),analysis_seconds=time.monotonic()-start,models_trained=0,new_test_evaluations=0))
    print(json.dumps(dict(primary={k:tables[k]["primary"]["mean"] for k in METHODS},gates=gates),ensure_ascii=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=R.DEFAULT_OUT)
    summarize(parser.parse_args().output)
