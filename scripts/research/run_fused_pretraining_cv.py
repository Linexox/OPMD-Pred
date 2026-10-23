"""Fold-isolated contrastive pretraining of fused, observed-modality views."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time
import warnings

import torch
from torch import nn
import torch.nn.functional as F

SPEC = importlib.util.spec_from_file_location("contrastive_reference", Path(__file__).with_name("run_contrastive_ordinal_cv.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
S = C.S
NAMES = ("image", "history", "tct", "dna", "methylation")
MODES = ("fused_full_instance", "fused_full_class", "fused_subset_class")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def projection():
    return nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 64))


def contrastive_loss(first, second, labels, class_positives):
    first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
    logits = first @ second.T / .1
    positive = labels[:, None].eq(labels[None, :]) if class_positives else torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    def directional(scores, mask):
        return -(F.log_softmax(scores, dim=1) * mask).sum(1).div(mask.sum(1)).mean()
    loss = (directional(logits, positive) + directional(logits.T, positive.T)) / 2
    return loss, int(positive.sum()), int(positive.numel() - positive.sum())


def availability(batch):
    return torch.stack([batch[name + "_present"].bool() if name in ("image", "tct")
                        else batch[name + "_mask"].any(1) for name in NAMES], dim=1)


def sample_subset(observed, generator):
    observed = observed.cpu()
    result = torch.zeros_like(observed)
    for row in range(len(observed)):
        available = observed[row].nonzero().flatten()
        S.require(len(available) >= 2, "A pretraining view has fewer than two observed modalities")
        count = int(torch.randint(2, len(available), (), generator=generator)) if len(available) > 2 else 2
        keep = available[torch.randperm(len(available), generator=generator)[:count]]
        result[row, keep] = True
    S.require(bool((result.sum(1) >= 2).all()) and not bool((result & ~observed).any()), "Invalid modality subset")
    return result


def hide_modalities(batch, retained):
    result = dict(batch)
    retained = retained.to(batch["label"].device)
    for column, name in enumerate(NAMES):
        key = name + ("_present" if name in ("image", "tct") else "_mask")
        keep = retained[:, column]
        result[key] = batch[key].bool() & (keep if name in ("image", "tct") else keep[:, None])
    return result


def pack_view(data, indices, fit, views):
    # pack(False) avoids consuming the global training RNG for image-view selection.
    batch = data.pack(indices, fit, False)
    batch["image_features"] = data.images["baseline"][indices, views].to("cuda")
    return batch


def group_gradients(model, projector):
    squares = {}
    for name, parameter in list(model.named_parameters()) + [("projector." + name, p) for name, p in projector.named_parameters()]:
        if parameter.grad is None:
            continue
        S.require(bool(torch.isfinite(parameter.grad).all()), "Nonfinite gradient")
        if name.startswith("fusion.missing_tokens."):
            group = "missing." + name.split(".")[2]
        elif name.startswith("fusion.transformer.") or name.startswith("fusion.norm.") or name == "fusion.cls_token":
            group = "fusion"
        elif name.startswith("projector."):
            group = "projector"
        elif name.startswith("ordinal_head."):
            raise ValueError("The ordinal classifier received a pretraining gradient")
        else:
            group = name.split(".")[1]
        squares[group] = squares.get(group, 0.) + float(parameter.grad.detach().square().sum())
    return {key: math.sqrt(value) for key, value in squares.items()}


def self_check():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(841)
        a, b = torch.randn(4, 64, requires_grad=True), torch.randn(4, 64, requires_grad=True)
        labels = torch.arange(4)
        standard = S.info_nce_loss(a, b, .1)
        actual, positive, negative = contrastive_loss(a, b, labels, False)
        S.require(torch.allclose(standard, actual, atol=1e-6), "Instance loss differs from the existing InfoNCE")
        class_unique, _, _ = contrastive_loss(a, b, labels, True)
        S.require(torch.allclose(actual, class_unique, atol=1e-6), "Unique-label class loss differs from instance loss")
        class_loss, positives, negatives = contrastive_loss(a, b, torch.tensor([0, 0, 1, 2]), True)
        S.require(positives == 6 and negatives == 10, "Class positives include a wrong pair")
        class_loss.backward()
        S.require(bool(torch.isfinite(a.grad).all()) and float(a.grad.norm()) > 0, "Loss has invalid gradients")
        generator = torch.Generator().manual_seed(481)
        checked = 0
        for pattern in range(32):
            mask = torch.tensor([[bool(pattern & (1 << column)) for column in range(5)]])
            if int(mask.sum()) < 2:
                continue
            for _ in range(25):
                subset = sample_subset(mask, generator)
                S.require(int(subset.sum()) < int(mask.sum()) if int(mask.sum()) > 2 else torch.equal(subset, mask), "Incorrect subset cardinality")
                checked += 1
    return dict(passed=True, instance_loss_matches_reference=True,
                class_positive_pairs_checked=True, finite_nonzero_loss_gradients=True,
                legal_subset_draws_checked=checked, minimum_observed_modalities=2)


def common_initial(data, train_ids, folder, seed, role, reference_folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "base.pt"
    identity = dict(train_ids=train_ids, seed=seed, role=role)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        S.require(saved["identity"] == identity, "Shared base identity changed")
        return saved, path
    S.check_time()
    S.check_gpu()
    S.set_seed(seed)
    fit = data.fit(train_ids, "baseline")
    model = S.make_model(S.configuration(head="ordinal", balanced=False), fit)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().cpu().clone()
    with torch.random.fork_rng(devices=[0]):
        projector_state = C.cpu_state(projection())
    old = torch.load(reference_folder / "initial.pt", map_location="cpu", weights_only=True)
    state = C.cpu_state(model)
    S.require(fit == old["fit"] and state.keys() == old["state"].keys(), "Shared base differs from the archived no-pretraining method")
    S.require(all(torch.equal(value, old["state"][key]) for key, value in state.items()), "Common model initialization changed")
    S.require(torch.equal(cpu_rng, old["cpu_rng"]) and torch.equal(cuda_rng, old["cuda_rng"]), "Supervised RNG initialization changed")
    saved = dict(identity=identity, fit=fit, state=state, projector=projector_state,
                 cpu_rng=cpu_rng, cuda_rng=cuda_rng, archived_no_pretrain_initial_equal=True)
    C.save(path, saved)
    del model, old
    torch.cuda.empty_cache()
    return saved, path


def initialize(data, train_ids, config, folder, base, base_path, seed, role):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "initial.pt"
    identity = dict(train_ids=train_ids, seed=seed, role=role, config=config)
    if path.exists():
        initial = torch.load(path, map_location="cpu", weights_only=True)
        S.require(initial["identity"] == identity and initial["base_sha256"] == S.digest(base_path), "Pretraining identity changed")
        return initial, path
    S.check_gpu()
    model = S.make_model(config, base["fit"])
    model.load_state_dict(base["state"], strict=True)
    projector = projection().cuda()
    projector.load_state_dict(base["projector"], strict=True)
    indices = data.indices(train_ids, True)
    observed = torch.stack([data.present,
        data.base["history_mask"].any(1), torch.tensor([bool(data.records[sid].tct) for sid in data.ids]),
        data.base["dna_mask"].any(1), data.base["methylation_mask"].any(1)], dim=1)
    indices = indices[observed[indices].sum(1) >= 2]
    pretrain_ids = [data.ids[int(index)] for index in indices]
    S.require(len(indices) >= 2 and set(pretrain_ids) <= set(train_ids), "Invalid pretraining cohort")
    history, gradients, missing_exposures = [], {}, {name: 0 for name in NAMES}
    order_gen = torch.Generator().manual_seed(seed + 100000)
    view_gen = torch.Generator().manual_seed(seed + 200000)
    subset_gen = torch.Generator().manual_seed(seed + 300000)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(projector.parameters()), lr=1e-4, weight_decay=1e-4)
    last = folder / "pretrain-last.pt"
    with torch.random.fork_rng(devices=[0]):
        torch.set_rng_state(base["cpu_rng"])
        torch.cuda.set_rng_state(base["cuda_rng"])
        if last.exists():
            resumed = torch.load(last, map_location="cpu", weights_only=True)
            S.require(resumed["identity"] == identity, "Pretraining resume changed")
            model.load_state_dict(resumed["state"], strict=True)
            projector.load_state_dict(resumed["projector"], strict=True)
            optimizer.load_state_dict(resumed["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and value.ndim:
                        state[key] = value.cuda()
            history, gradients, missing_exposures = resumed["history"], resumed["gradients"], resumed["missing_exposures"]
            torch.set_rng_state(resumed["cpu_rng"])
            torch.cuda.set_rng_state(resumed["cuda_rng"])
            for generator, key in ((order_gen, "order_rng"), (view_gen, "view_rng"), (subset_gen, "subset_rng")):
                generator.set_state(resumed[key])
        for epoch in range(len(history) + 1, 31):
            S.check_time()
            model.train()
            projector.train()
            batches, loss_values = [], []
            for chunk in indices[torch.randperm(len(indices), generator=order_gen)].split(32):
                S.check_time()
                if len(chunk) < 2:
                    batches.append(dict(sample_ids=[data.ids[int(i)] for i in chunk], skipped_single_record_batch=True))
                    continue
                va = torch.randint(1, 5, (len(chunk),), generator=view_gen)
                vb = (va - 1 + torch.randint(1, 4, (len(chunk),), generator=view_gen)) % 4 + 1
                first, second = pack_view(data, chunk, base["fit"], va), pack_view(data, chunk, base["fit"], vb)
                available = availability(first).cpu()
                retained = sample_subset(available, subset_gen) if config["pretrain_mode"] == "fused_subset_class" else available.clone()
                second = hide_modalities(second, retained)
                S.require(bool((available.sum(1) >= 2).all()) and bool((availability(second).sum(1) >= 2).all()), "A single-modality contrastive input escaped the guard")
                ha = model.fusion.fuse_embeddings(*S.embedding_inputs(model, first))
                hb = model.fusion.fuse_embeddings(*S.embedding_inputs(model, second))
                loss, positives, negatives = contrastive_loss(projector(ha), projector(hb), first["label"],
                                                              config["pretrain_mode"] != "fused_full_instance")
                S.require(bool(torch.isfinite(loss)), "Nonfinite pretraining loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norms = group_gradients(model, projector)
                for group, value in norms.items():
                    gradients[group] = gradients.get(group, 0.) + value
                for column, name in enumerate(NAMES):
                    missing_exposures[name] += int((~available[:, column]).sum()) + int((~retained[:, column]).sum())
                optimizer.step()
                value = float(loss.detach())
                loss_values.append(value)
                batches.append(dict(sample_ids=first["sample_id"], image_views_a=va.tolist(), image_views_b=vb.tolist(),
                    observed=available.tolist(), retained_b=retained.tolist(), labels=first["label"].tolist(),
                    positive_pairs=positives, negative_pairs=negatives, loss=value, gradient_norms=norms))
            S.require(loss_values, "No effective pretraining update")
            history.append(dict(epoch=epoch, loss=statistics.mean(loss_values), effective_updates=len(loss_values), batches=batches))
            C.save(last, dict(identity=identity, state=C.cpu_state(model), projector=C.cpu_state(projector), optimizer=optimizer.state_dict(),
                history=history, gradients=gradients, missing_exposures=missing_exposures,
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state().cpu(),
                order_rng=order_gen.get_state(), view_rng=view_gen.get_state(), subset_rng=subset_gen.get_state()))
    S.require(gradients.get("fusion", 0) > 0 and gradients.get("projector", 0) > 0, "Fused pretraining did not train fusion/projector")
    for name in NAMES:
        if missing_exposures[name]:
            S.require(gradients.get("missing." + name, 0) > 0, "An exposed missing token received no gradient")
    state = C.cpu_state(model)
    S.require(all(torch.equal(state[key], base["state"][key]) for key in state if key.startswith("ordinal_head.")), "Pretraining changed the ordinal classifier")
    delta = {"fusion": math.sqrt(sum(float((value - base["state"][key]).square().sum())
             for key, value in state.items() if key.startswith("fusion.transformer."))),
             "missing_tokens": math.sqrt(sum(float((value - base["state"][key]).square().sum())
             for key, value in state.items() if key.startswith("fusion.missing_tokens.")))}
    S.require(delta["fusion"] > 0 and delta["missing_tokens"] > 0, "Fusion or missing token weights did not change")
    initial = dict(state=state, fit=base["fit"], identity=identity, cpu_rng=base["cpu_rng"], cuda_rng=base["cuda_rng"],
        projector=C.cpu_state(projector), base_sha256=S.digest(base_path), pretrain_ids=pretrain_ids,
        pretraining_history=history, pretraining_updates=sum(row["effective_updates"] for row in history),
        gradients=gradients, missing_exposures=missing_exposures, parameter_delta_norms=delta)
    C.save(path, initial)
    S.write(folder / "pretraining.json", dict(identity=identity, base_sha256=initial["base_sha256"],
        pretrain_ids=pretrain_ids, pretrain_record_count=len(pretrain_ids), history=history,
        gradient_norm_sums=gradients, missing_exposures=missing_exposures, parameter_delta_norms=delta,
        initial_sha256=S.digest(path), fusion_transformer_pretrained=True, minimum_modalities=2,
        projection_removed_for_supervised_stage=True, ordinal_classifier_unchanged=True))
    del model, projector, optimizer
    torch.cuda.empty_cache()
    return initial, path


def verify_controls(args, data, protocol):
    checks = []
    for method in ("ordinal", "ordinal_pretrain"):
        for seed in (42, 43, 44):
            root = args.reference / "runs" / f"{method}_seed{seed}"
            result = read(root / "oof_results.json")
            errors = []
            for fold in protocol["folds"]:
                folder = root / f"fold{fold['fold']}"
                saved = read(folder / "outer_predictions.json")
                S.require(S.digest(folder / "refit/classifier.pt") == saved["classifier_sha256"], "Archived control classifier changed")
                _, rows = C.predict(data, fold["outer_holdout"]["sample_ids"], result["config"], folder / "refit")
                for actual, expected in zip(rows, saved["records"], strict=True):
                    S.require(actual["sample_id"] == expected["sample_id"] and actual["status"] == expected["status"] and actual["label"] == expected["label"], "Archived control cohort differs")
                    if actual["probabilities"] is not None:
                        errors.extend(abs(a - b) for a, b in zip(actual["probabilities"], expected["probabilities"], strict=True))
            maximum = max(errors, default=0.)
            S.require(maximum <= 1e-4, "Archived control forward differs")
            checks.append(dict(method=method, seed=seed, max_probability_error=maximum, metrics=result["metrics"]))
    S.write(args.output / "control-verification.json", dict(passed=True, checks=checks))
    return checks


def run_mode(args, protocol, data, mode, seed):
    config = S.configuration(head="ordinal", balanced=False, pretrain_epochs=30, seed_offset=seed - 42,
                             pretrain_mode=mode, projection_dimensions=[128, 128, 64], minimum_modalities=2)
    root = args.output / "runs" / f"{mode}_seed{seed}"
    rows = []
    for fold in protocol["folds"]:
        C.validate_fold(data, protocol, fold)
        offset = seed - 42 + fold["fold"]
        folder = root / f"fold{fold['fold']}"
        shared = args.output / "shared" / f"seed{seed}" / f"fold{fold['fold']}"
        old = args.reference / "runs" / f"ordinal_seed{seed}" / f"fold{fold['fold']}"
        inner_ids, val_ids = fold["inner_train"]["sample_ids"], fold["inner_val"]["sample_ids"]
        train_ids, held_ids = fold["outer_train"]["sample_ids"], fold["outer_holdout"]["sample_ids"]
        base, base_path = common_initial(data, inner_ids, shared / "inner", 42 + offset, "inner", old / "inner-initial")
        initial, initial_path = initialize(data, inner_ids, config, folder / "inner-initial", base, base_path, 42 + offset, "inner")
        inner = C.stage(data, inner_ids, val_ids, config, folder / "inner", 42 + offset, 50, initial, initial_path, 1e-4)
        base, base_path = common_initial(data, train_ids, shared / "refit", 142 + offset, "refit", old / "refit-initial")
        initial, initial_path = initialize(data, train_ids, config, folder / "refit-initial", base, base_path, 142 + offset, "refit")
        trained = C.stage(data, train_ids, None, config, folder / "refit", 142 + offset, inner["selected_epoch"], initial, initial_path, 1e-4)
        metrics, predicted = C.predict(data, held_ids, config, folder / "refit")
        S.write(folder / "outer_predictions.json", dict(records=predicted, metrics=metrics, config=config,
            classifier_sha256=trained["classifier_sha256"], selected_epoch=inner["selected_epoch"]))
        rows.extend(predicted)
        print(json.dumps(dict(event="fold_complete", mode=mode, seed=seed, fold=fold["fold"] + 1,
                              accuracy=metrics["fixed_median_0.5"]["accuracy_all_assigned"])), flush=True)
    S.require(len(rows) == len({row["sample_id"] for row in rows}) == 117, "Incomplete out-of-fold predictions")
    result = dict(method=mode, seed=seed, config=config, records=rows, metrics=S.summarize(rows))
    S.write(root / "oof_results.json", result)
    return result


def main():
    warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "cache", "reference", "encoder", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--pretrain-mode", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    check = self_check()
    if args.self_check:
        print(json.dumps(check))
        return
    if any(getattr(args, name) is None for name in ("input", "cache", "reference", "encoder", "output")):
        parser.error("input, cache, reference, encoder and output are required")
    started = time.monotonic()
    S.DEADLINE = started + args.seconds
    torch.set_num_threads(2)
    S.check_gpu()
    S.require(args.output.resolve().is_relative_to(Path("/home/zjkj/24334067")), "Output outside the personal workspace")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = read(args.input / "development_nested_cv_v3.json")
    reference = read(args.reference / "protocol.json")
    metadata = read(args.cache / "metadata.json")
    fingerprints = dict(csv_sha256=S.digest(args.input / "sysu_samples.csv"),
        protocol_sha256=S.digest(args.input / "development_nested_cv_v3.json"),
        baseline_cache_sha256=S.digest(args.cache / "baseline.npz"), encoder_sha256=S.digest(args.encoder))
    S.require(fingerprints["csv_sha256"] == protocol["csv_sha256"], "CSV changed")
    S.require(all(reference[key] == value for key, value in fingerprints.items()), "Inputs differ from the archived controls")
    S.require(fingerprints["encoder_sha256"] == S.REFERENCE.ENCODER_SHA, "Encoder changed")
    for recipe in ("baseline", "medical_aug"):
        S.require(S.digest(args.cache / (recipe + ".npz")) == metadata["cache_sha256"][recipe], "Feature cache changed")
    locked = set(protocol["locked_previous_test_ids"] + protocol["additional_related_locked_ids"])
    S.require(not {e["sample_id"] for e in protocol["development_entries"]}.intersection(locked), "Locked test reached this experiment")
    frozen = dict(arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        **fingerprints, source_sha256=S.digest(__file__), reference_runner_sha256=S.digest(Path(C.__file__)),
        seeds=[42, 43, 44], multimodal_pretrain_records=109, pretrain_epochs=30,
        host=os.uname().nodename, locked_test_evaluated=False, app_model_replaced=False)
    if (args.output / "protocol.json").exists():
        S.require(read(args.output / "protocol.json") == frozen, "Run protocol changed")
    else:
        S.write(args.output / "protocol.json", frozen)
    S.write(args.output / "self-check.json", check)
    data = S.REFERENCE.Data(args.input / "sysu_samples.csv", protocol, args.cache)
    observed_counts = [sum((bool(data.present[i]), bool(data.base["history_mask"][i].any()), bool(data.records[sid].tct),
                           bool(data.base["dna_mask"][i].any()), bool(data.base["methylation_mask"][i].any())))
                       for i, sid in enumerate(data.ids)]
    S.require(sum(count >= 2 for count in observed_counts) == 109, "Pretraining availability changed")
    controls = verify_controls(args, data, protocol)
    results = []
    try:
        for seed in (42, 43, 44):
            for mode in args.pretrain_mode:
                results.append(run_mode(args, protocol, data, mode, seed))
                S.write(args.output / "summary-progress.json", C.aggregate(results))
                S.write(args.output / "progress.json", dict(stage="training", completed=len(results), total=3 * len(args.pretrain_mode), seed=seed, mode=mode))
    except TimeoutError:
        S.write(args.output / "progress.json", dict(stage="budget_exhausted", completed=len(results), seconds=time.monotonic() - started))
        raise
    S.write(args.output / "summary.json", C.aggregate(results))
    S.write(args.output / "progress.json", dict(stage="training_completed", variants=len(results), seconds=time.monotonic() - started))
    print(json.dumps(dict(event="training_completed", seconds=time.monotonic() - started,
                         results=C.aggregate(results), controls=controls)), flush=True)


if __name__ == "__main__":
    main()
