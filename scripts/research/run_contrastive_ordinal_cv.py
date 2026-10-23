"""Fixed-cohort contrastive pretraining / ordinal-head factorial and nested LR study."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

SPEC = importlib.util.spec_from_file_location("september_reference", Path(__file__).with_name("run_september_comparison.py"))
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)
save = S.REFERENCE.save_torch


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def initialize(data, train_ids, config, folder, seed, role):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "initial.pt"
    identity = dict(train_ids=train_ids, seed=seed, role=role, config=config)
    if path.exists():
        initial = torch.load(path, map_location="cpu", weights_only=True)
        S.require(initial["identity"] == identity, "Shared initialization has different inputs")
        return initial, path
    S.check_time()
    S.check_gpu()
    S.set_seed(seed)
    fit = data.fit(train_ids, "baseline")
    model = S.make_model(config, fit)
    history, pair_totals = [], {}
    if config["pretrain_epochs"]:
        # Match the reference's RNG isolation: adding pretraining does not advance
        # the supervised-stage random sequence.
        with torch.random.fork_rng(devices=[0]):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            ix = data.indices(train_ids, True)
            for epoch in range(1, config["pretrain_epochs"] + 1):
                S.check_time()
                model.train()
                losses_by_batch, updates, pairs_by_epoch = [], 0, {}
                for chunk in ix[torch.randperm(len(ix))].split(32):
                    batch = S.pack(data, chunk, fit, True, config)
                    embeddings, present = S.embedding_inputs(model, batch)
                    losses = []
                    for a, b in itertools.combinations(model.fusion.names, 2):
                        observed = present[a] & present[b]
                        count = int(observed.sum())
                        name = a + "+" + b
                        item = pairs_by_epoch.setdefault(name, dict(eligible_batches=0, paired_records=0, skipped_batches=0))
                        if count >= 2:
                            losses.append(S.info_nce_loss(embeddings[a][observed], embeddings[b][observed], .1))
                            item["eligible_batches"] += 1
                            item["paired_records"] += count
                        else:
                            item["skipped_batches"] += 1
                    if not losses:
                        continue
                    loss = torch.stack(losses).mean()
                    S.require(torch.isfinite(loss), "Nonfinite contrastive loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses_by_batch.append(float(loss.detach()))
                    updates += 1
                for name, values in pairs_by_epoch.items():
                    total = pair_totals.setdefault(name, dict(eligible_batches=0, paired_records=0, skipped_batches=0))
                    for key, value in values.items():
                        total[key] += value
                history.append(dict(epoch=epoch, loss=statistics.mean(losses_by_batch) if losses_by_batch else None,
                                    effective_updates=updates, pairs=pairs_by_epoch))
            del optimizer
    initial = dict(state=cpu_state(model), fit=fit, identity=identity,
        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state().cpu(),
        pretraining_history=history, pair_totals=pair_totals,
        pretraining_updates=sum(item["effective_updates"] for item in history))
    save(path, initial)
    S.write(folder / "pretraining.json", dict(identity=identity, history=history,
        pair_totals=pair_totals, effective_updates=initial["pretraining_updates"],
        initial_sha256=S.digest(path), fusion_transformer_pretrained=False))
    del model
    torch.cuda.empty_cache()
    return initial, path


def stage(data, train_ids, val_ids, config, folder, seed, epochs, initial, initial_path, lr):
    folder.mkdir(parents=True, exist_ok=True)
    contract = dict(train_ids=train_ids, validation_ids=val_ids, config=config, seed=seed,
                    lr=lr, epochs=epochs, initial_sha256=S.digest(initial_path))
    done = folder / "done.json"
    if done.exists():
        result = read(done)
        S.require(result["contract"] == contract, "Finished stage changed")
        S.require(S.digest(folder / "classifier.pt") == result["classifier_sha256"], "Saved classifier changed")
        return result
    S.check_time()
    S.check_gpu()
    fit = initial["fit"]
    S.require(set(fit["fit_ids"]) <= set(train_ids) and not set(fit["fit_ids"]).intersection(val_ids or []), "Fit includes held-out inputs")
    model = S.make_model(config, fit)
    model.load_state_dict(initial["state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    torch.set_rng_state(initial["cpu_rng"])
    torch.cuda.set_rng_state(initial["cuda_rng"].cpu())
    weights = torch.tensor(fit["class_weights"], device="cuda") if config["balanced"] else None
    ix = data.indices(train_ids, True)
    history, best_key, best_epoch, stale, first_epoch = [], (-1., -1.), None, 0, 1
    last = folder / "last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location="cuda", weights_only=True)
        S.require(checkpoint["contract"] == contract, "Resume identity differs")
        model.load_state_dict(checkpoint["state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best_key, best_epoch, stale = checkpoint["history"], tuple(checkpoint["best_key"]), checkpoint["best_epoch"], checkpoint["stale"]
        first_epoch = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
    for epoch in range(first_epoch, epochs + 1):
        S.check_time()
        if val_ids is not None and stale >= config["patience"]:
            break
        model.train()
        total = 0.
        for chunk in ix[torch.randperm(len(ix))].split(8):
            batch = S.dropout(S.pack(data, chunk, fit, True, config), config)
            logits = S.forward(model, batch)
            loss = S.ordinal_loss(logits, batch["label"]) if config["head"] == "ordinal" else F.cross_entropy(logits, batch["label"], weight=weights)
            S.require(torch.isfinite(loss), "Nonfinite supervised loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(chunk)
        item = dict(epoch=epoch, training_loss=total / len(ix))
        improved = val_ids is None
        if val_ids is not None:
            evaluation, _ = S.evaluate(model, data, val_ids, fit, config)
            item["inner_validation"] = evaluation
            primary = evaluation["fixed_median_0.5"]
            key = primary["accuracy_all_assigned"], primary["macro_f1"]
            if key > best_key:
                best_key, best_epoch, stale, improved = key, epoch, 0, True
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(item)
        if improved:
            save(folder / "classifier.pt", dict(state=cpu_state(model), fit=fit, config=config,
                 selected_epoch=best_epoch, contract=contract, identity=dict(train_ids=train_ids, validation_ids=val_ids)))
        save(last, dict(state=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
             best_key=best_key, best_epoch=best_epoch, stale=stale, epoch=epoch,
             cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), contract=contract))
    S.require(best_epoch is not None, "No epoch completed")
    result = dict(contract=contract, selected_epoch=best_epoch, best_key=list(best_key),
                  history=history, fit=fit, config=config, seed=seed, lr=lr,
                  initial_sha256=contract["initial_sha256"], classifier_sha256=S.digest(folder / "classifier.pt"))
    S.write(done, result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def predict(data, ids, config, stage_folder):
    artifact = torch.load(stage_folder / "classifier.pt", map_location="cpu", weights_only=True)
    model = S.make_model(config, artifact["fit"])
    model.load_state_dict(artifact["state"], strict=True)
    metrics, rows = S.evaluate(model, data, ids, artifact["fit"], config)
    del model
    torch.cuda.empty_cache()
    return metrics, rows


def validate_fold(data, protocol, fold):
    train, held = fold["outer_train"]["sample_ids"], fold["outer_holdout"]["sample_ids"]
    inner, val = fold["inner_train"]["sample_ids"], fold["inner_val"]["sample_ids"]
    S.require(not set(train).intersection(held) and set(train) | set(held) == set(data.ids), "Outer split changed")
    S.require(not set(inner).intersection(val) and set(inner) | set(val) == set(train), "Inner split changed")
    entries = {item["sample_id"]: item for item in protocol["development_entries"]}
    for left, right in ((train, held), (inner, val)):
        for field in ("cv_component", "image_sha256", "dna_report_sha256"):
            l = {entries[sid][field] for sid in left if entries[sid][field]}
            r = {entries[sid][field] for sid in right if entries[sid][field]}
            S.require(not l.intersection(r), "Source group crosses a split")


def run_fixed(args, protocol, data, method, seed):
    changes = dict(softmax=dict(), softmax_pretrain=dict(pretrain_epochs=30),
                   ordinal=dict(head="ordinal"), ordinal_pretrain=dict(head="ordinal", pretrain_epochs=30),
                   balanced_reference=dict(balanced=True))
    config = S.configuration(balanced=False, seed_offset=seed - 42, **changes[method]) if method != "balanced_reference" else S.configuration(seed_offset=seed - 42)
    root = args.output / "runs" / f"{method}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in protocol["folds"]:
        validate_fold(data, protocol, fold)
        folder = root / f"fold{fold['fold']}"
        offset = config["seed_offset"] + fold["fold"]
        inner_ids, val_ids = fold["inner_train"]["sample_ids"], fold["inner_val"]["sample_ids"]
        train_ids, held_ids = fold["outer_train"]["sample_ids"], fold["outer_holdout"]["sample_ids"]
        initial, initial_path = initialize(data, inner_ids, config, folder / "inner-initial", 42 + offset, "inner")
        inner = stage(data, inner_ids, val_ids, config, folder / "inner", 42 + offset,
                      50, initial, initial_path, 1e-4)
        initial, initial_path = initialize(data, train_ids, config, folder / "refit-initial", 142 + offset, "refit")
        trained = stage(data, train_ids, None, config, folder / "refit", 142 + offset,
                        inner["selected_epoch"], initial, initial_path, 1e-4)
        m, predicted = predict(data, held_ids, config, folder / "refit")
        S.write(folder / "outer_predictions.json", dict(records=predicted, metrics=m,
            classifier_sha256=trained["classifier_sha256"], config=config, selected_epoch=inner["selected_epoch"]))
        rows.extend(predicted)
        print(json.dumps(dict(event="fold_complete", method=method, seed=seed, fold=fold["fold"],
                              correct=m["fixed_median_0.5"]["correct_count"])), flush=True)
    S.require(len(rows) == len({r["sample_id"] for r in rows}) == 117, "Missing outer predictions")
    result = dict(method=method, seed=seed, config=config, records=rows, metrics=S.summarize(rows))
    S.write(root / "oof_results.json", result)
    print(json.dumps(dict(event="method_complete", method=method, seed=seed, metrics=result["metrics"])), flush=True)
    return result


def tune(args, protocol, data, seed):
    config = S.configuration(balanced=False, head="ordinal", pretrain_epochs=30, seed_offset=seed - 42)
    fixed_root = args.output / "runs" / f"ordinal_pretrain_seed{seed}"
    root = args.output / "runs" / f"ordinal_pretrain_tuned_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    rows, choices = [], []
    for fold in protocol["folds"]:
        fixed = fixed_root / f"fold{fold['fold']}"
        folder = root / f"fold{fold['fold']}"
        offset = seed - 42 + fold["fold"]
        initial_path = fixed / "inner-initial/initial.pt"
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        normal = read(fixed / "inner/done.json")
        low = stage(data, fold["inner_train"]["sample_ids"], fold["inner_val"]["sample_ids"], config,
            folder / "inner-low", 42 + offset, 50, initial, initial_path, 5e-5)
        selected = max((normal, low), key=lambda r: (tuple(r["best_key"]), r["lr"], -r["selected_epoch"]))
        choice = dict(fold=fold["fold"], lr=selected["lr"], epoch=selected["selected_epoch"],
                      normal_key=normal["best_key"], low_key=low["best_key"],
                      shared_inner_initial_sha256=normal["initial_sha256"],
                      both_learning_rates_share_initialization=normal["initial_sha256"] == low["initial_sha256"])
        S.require(choice["both_learning_rates_share_initialization"], "LR candidates did not share initialization")
        if selected["lr"] == 1e-4:
            refit = fixed / "refit"
        else:
            initial_path = fixed / "refit-initial/initial.pt"
            initial = torch.load(initial_path, map_location="cpu", weights_only=True)
            refit = folder / "refit-low"
            stage(data, fold["outer_train"]["sample_ids"], None, config, refit, 142 + offset,
                  selected["selected_epoch"], initial, initial_path, 5e-5)
        trained = read(refit / "done.json")
        metrics, predicted = predict(data, fold["outer_holdout"]["sample_ids"], config, refit)
        S.write(folder / "selection.json", choice)
        S.write(folder / "outer_predictions.json", dict(records=predicted, metrics=metrics,
            config=config, classifier_sha256=trained["classifier_sha256"],
            classifier_path=str(refit / "classifier.pt"), selection=choice))
        rows.extend(predicted)
        choices.append(choice)
    result = dict(method="ordinal_pretrain_tuned", seed=seed, config=config, records=rows,
                  metrics=S.summarize(rows), choices=choices)
    S.write(root / "oof_results.json", result)
    return result


def check_archived(args, results):
    checks = []
    names = dict(balanced_reference="dino_softmax_balanced", softmax="dino_softmax", ordinal="dino_ordinal")
    for result in results:
        method, seed = result["method"], result["seed"]
        if method not in names or (method != "balanced_reference" and seed != 42):
            continue
        name = names[method] + (f"_seed{seed}" if seed != 42 else "")
        previous = read(args.previous / "runs" / name / "oof_results.json")
        expected = {r["sample_id"]: r for r in previous["records"]}
        S.require(previous["config"] == result["config"], "Archived control configuration differs")
        errors = []
        for row in result["records"]:
            old = expected[row["sample_id"]]
            S.require(row["status"] == old["status"] and row["label"] == old["label"], "Control inputs differ")
            if row["probabilities"] is not None:
                errors.extend(abs(a - b) for a, b in zip(row["probabilities"], old["probabilities"]))
        error = max(errors, default=0.)
        S.require(error <= 1e-4, "Fresh control does not reproduce archived probabilities")
        checks.append(dict(method=method, seed=seed, max_absolute_probability_error=error,
                           correct=result["metrics"]["correct_count"], passed=True))
    S.write(args.output / "archived-control-checks.json", checks)


def aggregate(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["method"], []).append(result)
    keys = ("correct_count", "accuracy_all_assigned", "severe_to_mild_count", "two_level_error_count",
            "severe_recall_all_assigned", "mae", "abstained_count")
    return {name: dict(per_seed=[dict(seed=r["seed"], metrics=r["metrics"]) for r in runs],
                       means={key: statistics.mean(r["metrics"][key] for r in runs) for key in keys})
            for name, runs in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "cache", "previous", "encoder", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200)
    args = parser.parse_args()
    S.DEADLINE = time.monotonic() + args.seconds
    torch.set_num_threads(2)
    S.check_gpu()
    S.require(args.output.resolve().is_relative_to(Path("/home/zjkj/24334067")), "Output outside personal directory")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = read(args.input / "development_nested_cv_v3.json")
    S.require(S.digest(args.input / "sysu_samples.csv") == protocol["csv_sha256"], "CSV differs from frozen split")
    locked = set(protocol["locked_previous_test_ids"] + protocol["additional_related_locked_ids"])
    S.require(not {r["sample_id"] for r in protocol["development_entries"]}.intersection(locked), "Locked data reached study")
    previous_receipt = read(args.previous / "command-core.json")["arguments"]
    S.require(Path(previous_receipt["input"]) == args.input and Path(previous_receipt["cache"]) == args.cache, "Previous study inputs differ")
    meta = read(args.cache / "metadata.json")
    S.require(S.digest(args.encoder) == S.REFERENCE.ENCODER_SHA, "Encoder changed")
    for name in ("baseline", "medical_aug"):
        S.require(S.digest(args.cache / f"{name}.npz") == meta["cache_sha256"][name], "Cached input changed")
    command = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256=S.digest(__file__), started_unix=time.time(), host=os.uname().nodename,
        csv_sha256=S.digest(args.input / "sysu_samples.csv"),
        protocol_sha256=S.digest(args.input / "development_nested_cv_v3.json"),
        baseline_cache_sha256=S.digest(args.cache / "baseline.npz"), encoder_sha256=S.digest(args.encoder),
        fixed_lrs=[1e-4], tuning_lrs=[1e-4, 5e-5], seeds=[42, 43, 44],
        pretrain_epochs=30, locked_test_evaluated=False, app_model_replaced=False)
    if (args.output / "protocol.json").exists():
        previous = read(args.output / "protocol.json")
        for key in ("source_sha256", "csv_sha256", "protocol_sha256", "baseline_cache_sha256", "encoder_sha256"):
            S.require(previous[key] == command[key], "Resumed protocol differs")
    else:
        S.write(args.output / "protocol.json", command)
    data = S.REFERENCE.Data(args.input / "sysu_samples.csv", protocol, args.cache)
    results = []
    # Verify the actual balanced control before interpreting new candidate results.
    order = [("balanced_reference", 42), ("ordinal_pretrain", 42)]
    order += [(method, 42) for method in ("softmax", "ordinal", "softmax_pretrain")]
    order += [(method, seed) for seed in (43, 44) for method in
              ("balanced_reference", "ordinal_pretrain", "softmax", "ordinal", "softmax_pretrain")]
    for method, seed in order:
        result = run_fixed(args, protocol, data, method, seed)
        results.append(result)
        check_archived(args, results)
        S.write(args.output / "summary-progress.json", aggregate(results))
        S.write(args.output / "progress.json", dict(stage="fixed_comparison", completed=len(results), total=len(order), method=method, seed=seed))
    for seed in (42, 43, 44):
        results.append(tune(args, protocol, data, seed))
        S.write(args.output / "summary-progress.json", aggregate(results))
        S.write(args.output / "progress.json", dict(stage="nested_lr_selection", completed_seed=seed))
    S.write(args.output / "summary.json", aggregate(results))
    S.write(args.output / "progress.json", dict(stage="training_completed", variants=len(results), ended_unix=time.time()))
    print(json.dumps(dict(event="training_completed", variants=len(results))), flush=True)


if __name__ == "__main__":
    main()
