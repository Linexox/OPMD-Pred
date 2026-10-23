"""Independent numpy reconstruction of attribution, train-only policies and reported metrics."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/experiments/sysu/explanation-acquisition-20261017/attempt1"


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    protocol=read(OUT/"protocol.json");report=read(OUT/"results.json");cases=protocol["cases"];idx={c["sample_id"]:i for i,c in enumerate(cases)}
    assert sha(OUT/"protocol.json")==read(OUT/"environment.json")["protocol_sha256"]==report["protocol_sha256"]
    inputroot=Path(protocol["source_paths"]["input"])
    assert sha(inputroot/"sysu_samples.csv")==protocol["source_sha"]["csv"]
    assert sha(inputroot/"development_nested_cv_v3.json")==protocol["source_sha"]["cv"]
    cv=read(inputroot/"development_nested_cv_v3.json")
    locked=set(cv["locked_previous_test_ids"]+cv["additional_related_locked_ids"])
    assert len(cases)==117 and not set(idx)&locked
    vectors=0;maximum_shap=0.;maximum_inter=0.;policy_rows=0;model_count=0
    by_seed=defaultdict(list)
    order=protocol["action_type_order"];minimum=protocol["minimum_historical_support"]
    for model in protocol["models"]:
        name=f"{model['method']}_seed{model['seed']}_fold{model['fold']}"
        file=OUT/"coalitions"/f"{name}.npz";receipt=read(file.with_suffix(".json"))
        assert sha(model["path"])==model["sha256"]==receipt["checkpoint_sha256"]
        assert sha(file)==receipt["coalitions_sha256"]
        with np.load(file,allow_pickle=False) as cached:
            probs=cached["probabilities"].copy().astype(float);phi=cached["shap"].copy();interaction=cached["interactions"].copy();dec=cached["decisions"].copy()
            assert cached["sample_ids"].tolist()==list(idx)
        assert np.isfinite(probs).all() and probs.min()>-1e-7 and np.max(np.abs(probs.sum(-1)-1))<1e-6
        calculated=(probs[:,:,1:].sum(-1)>.5).astype(int)+(probs[:,:,2]>.5).astype(int)
        assert np.array_equal(calculated,dec)
        values=probs[:,:,2];ind_phi=np.zeros_like(phi,dtype=float);ind_inter=np.zeros_like(interaction,dtype=float)
        for i in range(5):
            for mask in range(32):
                if mask&(1<<i):continue
                k=mask.bit_count();w=math.factorial(k)*math.factorial(4-k)/math.factorial(5)
                ind_phi[:,i]+=w*(values[:,mask|(1<<i)]-values[:,mask])
            for j in range(i+1,5):
                for mask in range(32):
                    if mask&((1<<i)|(1<<j)):continue
                    k=mask.bit_count();w=math.factorial(k)*math.factorial(3-k)/(2*math.factorial(4))
                    term=values[:,mask|(1<<i)|(1<<j)]-values[:,mask|(1<<i)]-values[:,mask|(1<<j)]+values[:,mask]
                    ind_inter[:,i,j]+=w*term
                ind_inter[:,j,i]=ind_inter[:,i,j]
        err=float(np.max(np.abs(ind_phi-phi)));ierr=float(np.max(np.abs(ind_inter-interaction)))
        assert err<1e-6 and ierr<1e-6;maximum_shap=max(maximum_shap,err);maximum_inter=max(maximum_inter,ierr)
        for i,c in enumerate(cases):
            for mask in range(32):
                assert np.max(np.abs(probs[i,mask]-probs[i,mask&c["observed_mask"]]))<1e-7
        train=[idx[s] for s in model["train_ids"]];held=[idx[s] for s in model["held_ids"]]
        assert not set(train)&set(held)
        history=read(OUT/"histories"/f"{name}.json")
        assert history["training_ids"]==model["train_ids"]
        h_phi=np.zeros(5);h_n=np.zeros(5,int);h_inter=np.zeros((5,5));h_pair_n=np.zeros((5,5),int)
        fixed_gains=defaultdict(list);fixed_severe=defaultdict(list)
        for i in train:
            c=cases[i]
            for j in range(5):
                if not c["observed_mask"]&(1<<j):continue
                h_n[j]+=1;h_phi[j]+=abs(float(phi[i,j]))
                for k in range(5):
                    if j==k or not c["observed_mask"]&(1<<k):continue
                    h_pair_n[j,k]+=1;h_inter[j,k]+=abs(float(interaction[i,j,k]))
            scenario=c["primary_scenario"]
            if scenario is None:continue
            before=int(dec[i,scenario["initial_mask"]]);y=c["label"]
            for action in scenario["candidates"]:
                after=int(dec[i,scenario["initial_mask"]|action["mask"]])
                fixed_gains[action["name"]].append(int(after==y)-int(before==y))
                if y==2:fixed_severe[action["name"]].append(int(before==0)-int(after==0))
        for j in range(5):
            assert int(h_n[j])==history["single"][j]["count"]
            if h_n[j]:assert abs(h_phi[j]/h_n[j]-history["single"][j]["mean_abs"])<1e-12
            for k in range(5):
                assert int(h_pair_n[j,k])==history["pairs"][j][k]["count"]
                if h_pair_n[j,k]:assert abs(h_inter[j,k]/h_pair_n[j,k]-history["pairs"][j][k]["mean_abs"])<1e-12
        def fixed_key(a):
            name=a["name"]
            return float(np.mean(fixed_gains[name])) if fixed_gains[name] else 0.,float(np.mean(fixed_severe[name])) if fixed_severe[name] else 0.,-order.index(name)
        saved={r["sample_id"]:r for r in read(OUT/"decisions"/f"{model['method']}_seed{model['seed']}.json")}
        for i in held:
            c=cases[i];s=c["primary_scenario"]
            if s is None:continue
            r=saved[c["sample_id"]];initial=s["initial_mask"];choices=s["candidates"];preferred=max(choices,key=fixed_key)
            assert r["initial_mask"]==initial and r["label"]==c["label"]
            assert np.array_equal(r["before"],np.eye(3)[dec[i,initial]])
            assert np.array_equal(r["full"],np.eye(3)[dec[i,c["observed_mask"]]])
            for policy in ["fixed","historical_shap","historical_interaction"]:
                chosen=preferred;fallback=False
                if policy!="fixed":
                    scores=[];supported=True
                    for a in choices:
                        new=[j for j in range(5) if a["mask"]&(1<<j) and not initial&(1<<j)]
                        vals=[(h_n[j],h_phi[j]/h_n[j] if h_n[j] else 0.) for j in new] if policy=="historical_shap" else [
                            (h_pair_n[j,k],h_inter[j,k]/h_pair_n[j,k] if h_pair_n[j,k] else 0.) for j in new for k in range(5) if initial&(1<<k)]
                        if not vals or any(n<minimum for n,v in vals):supported=False;break
                        scores.append(sum(v for n,v in vals))
                    if not supported:fallback=True
                    elif max(scores)<=1e-12:fallback=True
                    else:
                        tied=[a for a,v in zip(choices,scores) if abs(v-max(scores))<=1e-12]
                        chosen=max(tied,key=fixed_key);fallback=len(tied)>1
                assert chosen["name"]==r["choices"][policy] and fallback==r["fallback"][policy]
                assert np.array_equal(r["outcomes"][policy],np.eye(3)[dec[i,initial|chosen["mask"]]])
            expected=np.mean([np.eye(3)[dec[i,initial|a["mask"]]] for a in choices],axis=0)
            assert np.max(np.abs(expected-r["outcomes"]["random"]))<1e-12
            by_seed[(model["method"],model["seed"])].append(r);policy_rows+=1
        model_count+=1;vectors+=len(cases)*32
    transition_checks=0
    for method,table in report["tables"].items():
        for scope in ["primary","material_mapping_recorded","without_prelisted_quality_flags","all_legal_starts_record_weighted"]:
            for result in table[scope]["per_seed"]:
                for policy in ["fixed","random","historical_shap","historical_interaction","posthoc_best_not_a_policy"]:
                    delta=(result[policy]["accuracy"]-result["before"]["accuracy"])*result[policy]["records"]
                    assert abs(delta-result[policy]["corrected_expected"]+result[policy]["harmed_expected"])<1e-9
                    transition_checks+=1
        for seed,primary in zip([42,43,44],table["primary"]["per_seed"]):
            rows=by_seed[(method,seed)];assert len(rows)==90
            for policy in ["fixed","random","historical_shap","historical_interaction"]:
                y=np.array([r["label"] for r in rows]);o=np.array([r["outcomes"][policy] for r in rows])
                expected={"accuracy":float(o[np.arange(90),y].mean()),"moderate_recall":float(o[y==1,1].mean()),
                    "severe_to_low":float(o[y==2,0].mean()),"two_level_error":float(np.where(y==2,o[:,0],np.where(y==0,o[:,2],0)).mean())}
                assert all(abs(value-primary[policy][key])<1e-12 for key,value in expected.items())
    initial=read(OUT/"results-initial-before-transition-fix.json")
    assert all(abs(initial["tables"][m]["primary"]["mean"][p][k]-report["tables"][m]["primary"]["mean"][p][k])<1e-12
        for m in report["tables"] for p in ["fixed","random","historical_shap","historical_interaction"] for k in ["accuracy","moderate_recall","severe_to_low","two_level_error"])
    result=dict(passed=True,models_checked=model_count,coalition_vectors_checked=vectors,
        independently_reconstructed_heldout_policy_rows=policy_rows,transition_identities_checked=transition_checks,
        maximum_independent_shap_error=maximum_shap,maximum_independent_interaction_error=maximum_inter,
        main_metrics_unchanged_by_summary_fixes=True,no_locked_test_evaluated=True,training=False,
        protocol_sha256=sha(OUT/"protocol.json"),results_sha256=sha(OUT/"results.json"),audit_source_sha256=sha(Path(__file__)))
    (OUT/"independent-audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result))


if __name__=="__main__":main()
