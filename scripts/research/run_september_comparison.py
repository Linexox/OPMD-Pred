"""Frozen-protocol September/October comparisons; private outputs stay in the personal workspace."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F

from opmd_pred.data.transforms import make_image_transform
from opmd_pred.model.losses import info_nce_loss, ordinal_loss
from opmd_pred.model.ordinal import model_probabilities
from opmd_pred.model.vit import ImageEncoder
from opmd_pred.scripts.common import set_seed
from opmd_pred.scripts.train_multimodal import apply_modality_dropout

SPEC = importlib.util.spec_from_file_location("paired_reference", Path(__file__).with_name("train_medical_augmentation_cv.py"))
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
DEADLINE = float("inf")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_time():
    if time.monotonic() >= DEADLINE:
        raise TimeoutError("Prespecified wall-clock budget reached; completed stages are retained")


def require(value, message):
    if not value:
        raise ValueError(message)


def check_gpu():
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"
    ], text=True).strip()
    require(all(int(line.split(",")[0]) == os.getpid() for line in processes.splitlines() if line.strip()),
            "Another GPU process is present; stop before taking resources")


def configuration(**changes):
    result = dict(encoder="dino", head="softmax", balanced=True, dimension=128,
                  dropout="bernoulli", dna_scale="raw", pretrain_epochs=0,
                  seed_offset=0, max_epochs=50, patience=5)
    result.update(changes)
    return result


def make_model(config, fit):
    mc = dict(dimension=config["dimension"], tct_categories=max(fit["tct_vocab"].values(), default=1),
              image_encoder="dinov2_b14", modalities=["image", "history", "tct", "dna", "methylation"],
              head=config["head"], fusion="transformer", dna_input_size=2)
    model = REFERENCE.model_from_config(mc)
    model.fusion.freeze_image()
    model.fusion.image_encoder = None
    if config["encoder"] == "vit":
        model.fusion.image_projection = nn.Linear(768, config["dimension"])
    return model.cuda()


def embedding_inputs(model, batch):
    return model.fusion.modality_embeddings(batch)


def forward(model, batch):
    return model.forward_from_embeddings(*embedding_inputs(model, batch))


def pack(data, ix, fit, training, config):
    batch = data.pack(ix, fit, training)
    if config["dna_scale"] == "log_standardized":
        values = batch["dna"][:, 0]
        transformed = (torch.log1p(values * 10000) - fit["dna_log_mean"]) / fit["dna_log_population_std"]
        batch["dna"][:, 0] = torch.where(batch["dna_mask"][:, 0].bool(), transformed, torch.zeros_like(values))
    return batch


def dropout(batch, config):
    if config["dropout"] == "legacy_subset":
        # The historical positive probability was ignored in subset mode.
        # p=1 in today's subset implementation reproduces that actual sampling law.
        return apply_modality_dropout(batch, 1., "subset")
    return apply_modality_dropout(batch, .15, "bernoulli")


@torch.inference_mode()
def evaluate(model, data, ids, fit, config):
    model.eval()
    ix = data.indices(ids, True)
    p = torch.cat([model_probabilities(forward(model, pack(data, chunk, fit, False, config)), config["head"]).cpu()
                   for chunk in ix.split(8)])
    require(torch.isfinite(p).all() and (p >= -1e-7).all(), "Invalid predictions")
    require(torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-6), "Probabilities do not sum to one")
    metrics = REFERENCE.score(p, data.base["label"][ix], [data.records[s].label for s in ids])
    by_id = {data.ids[i]: row for i, row in zip(ix.tolist(), p.tolist())}
    rows = [dict(sample_id=sid, label=data.records[sid].label, probabilities=by_id.get(sid),
                 status="predicted" if sid in by_id else "abstain") for sid in ids]
    return metrics, rows


def verify_reference(args, protocol):
    path = args.output / "reference_reproduction.json"
    if path.exists():
        saved = json.loads(path.read_text())
        require(saved["passed"], "Stored reference verification failed")
        return saved
    data = REFERENCE.Data(args.input / "sysu_samples.csv", protocol, args.cache)
    checks = []
    all_rows = []
    for fold in protocol["folds"]:
        folder = args.reference / f"fold{fold['fold']}" / "baseline"
        artifact = torch.load(folder / "refit/classifier.pt", map_location="cpu", weights_only=True)
        archived = json.loads((folder / "outer_predictions.json").read_text())
        require(digest(folder / "refit/classifier.pt") == archived["classifier_sha256"], "Reference classifier SHA mismatch")
        require(artifact["receipt"]["csv_sha256"] == digest(args.input / "sysu_samples.csv"), "Reference uses different data")
        require(artifact["receipt"]["cv_sha256"] == digest(args.input / "development_nested_cv_v3.json"), "Reference uses different folds")
        held = fold["outer_holdout"]["sample_ids"]
        train = set(artifact["fit"]["fit_ids"])
        require(not train.intersection(held), "Reference holdout participated in training")
        require(set(train) <= set(fold["outer_train"]["sample_ids"]), "Reference training IDs are out of scope")
        model = make_model(configuration(), artifact["fit"])
        model.load_state_dict(artifact["state"], strict=True)
        metrics, rows = evaluate(model, data, held, artifact["fit"], configuration())
        require([r["sample_id"] for r in rows] == [r["sample_id"] for r in archived["records"]], "Reference order differs")
        errors = []
        for actual, expected in zip(rows, archived["records"]):
            require(actual["status"] == expected["status"] and actual["label"] == expected["label"], "Reference coverage differs")
            if actual["probabilities"] is not None:
                errors.append(max(abs(a - b) for a, b in zip(actual["probabilities"], expected["probabilities"])))
        delta = max(errors, default=0.)
        require(delta <= 1e-4, "Fresh forward differs from archived probabilities")
        checks.append(dict(fold=fold["fold"], max_absolute_probability_error=delta,
                           classifier_sha256=archived["classifier_sha256"], metrics=metrics))
        all_rows.extend(rows)
        del model
        torch.cuda.empty_cache()
    scored = summarize(all_rows)
    require(scored["correct_count"] == 65 and scored["assigned_count"] == 117, "Reference count differs")
    result = dict(passed=True, fresh_classifier_forward=True, image_features="frozen archived cache",
                  old_test_new_inference=False, folds=checks, metrics=scored)
    write(path, result)
    print(json.dumps(dict(event="reference_verified", max_error=max(r["max_absolute_probability_error"] for r in checks), metrics=scored)), flush=True)
    return result


def summarize(rows):
    chosen = [r for r in rows if r["status"] == "predicted"]
    metrics = REFERENCE.score(torch.tensor([r["probabilities"] for r in chosen]),
                              torch.tensor([r["label"] for r in chosen]), [r["label"] for r in rows])
    return metrics["fixed_median_0.5"]


@torch.inference_mode()
def prepare_features(args, protocol):
    path = args.output / "features"
    ready = path / "receipt.json"
    if ready.exists():
        receipt = json.loads(ready.read_text())
        require(all(digest(path / k) == v for k, v in receipt["files_sha256"].items()), "Feature cache was modified")
        return receipt
    path.mkdir(exist_ok=True)
    saved = torch.load(args.old_checkpoint, map_location="cpu", weights_only=True)
    original = torch.load(args.old_contrastive, map_location="cpu", weights_only=True)
    state = {k.removeprefix("fusion.image_encoder."): v for k, v in saved["model"].items() if k.startswith("fusion.image_encoder.")}
    require(all(torch.equal(v, original["fusion"]["image_encoder." + k]) for k, v in state.items()), "Old encoder differs between pretrain and supervised checkpoint")
    zenodo = torch.load(args.old_zenodo, map_location="cpu", weights_only=True)
    require("image_encoder" in zenodo, "Old Zenodo checkpoint has no image encoder")
    same_zenodo = set(state) == set(zenodo["image_encoder"]) and all(torch.equal(v, zenodo["image_encoder"][k]) for k, v in state.items())
    require(same_zenodo, "Old image encoder provenance not verified against Zenodo checkpoint")
    ids = [e["sample_id"] for e in protocol["development_entries"]]
    entries = {e["sample_id"]: e for e in protocol["development_entries"]}
    records = {r.sample_id: r for r in REFERENCE.load_sysu_records(args.input / "sysu_samples.csv")}
    vit = ImageEncoder(pretrained=False, kind="vit_b16")
    vit.load_state_dict(state, strict=True)
    vit.freeze()
    vit.cuda()
    dino = ImageEncoder(pretrained=False, kind="dinov2_b14")
    require(digest(args.encoder) == REFERENCE.ENCODER_SHA, "DINO encoder changed")
    dino.load_artifact(args.encoder)
    dino.freeze()
    dino.cuda()
    vit_features = np.zeros((len(ids), 5, 768), dtype=np.float32)
    geometry = np.zeros((len(ids), 5, 1536), dtype=np.float32)
    with np.load(args.cache / "baseline.npz", allow_pickle=False) as cache:
        require(cache["sample_ids"].tolist() == ids, "Input cache order differs")
        geometry[:, 0] = cache["features"][:, 0]
        present = cache["image_present"].copy()
    for index, sid in enumerate(ids):
        check_time()
        if not present[index]:
            continue
        image_path = args.input / records[sid].image_path
        require(digest(image_path) == entries[sid]["image_sha256"], "Image identity changed")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        tensors = []
        geometry_tensors = []
        for view in range(5):
            seed = int(hashlib.sha256(f"september-compare42:{entries[sid]['image_sha256']}:{view}".encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            tensors.append(make_image_transform(view > 0, "vit_b16")(image.copy()))
            if view:
                torch.manual_seed(seed)
                angle = (torch.rand(()).item() * 2 - 1) * 5
                transformed = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(124, 116, 104))
                w, h = transformed.size
                padx, pady = max(1, round(w * .02)), max(1, round(h * .02))
                dx = round((torch.rand(()).item() * 2 - 1) * padx)
                dy = round((torch.rand(()).item() * 2 - 1) * pady)
                canvas = Image.new("RGB", (w + 2 * padx, h + 2 * pady), (124, 116, 104))
                canvas.paste(transformed, (padx + dx, pady + dy))
                geometry_tensors.append(make_image_transform(False, "dinov2_b14")(canvas))
                if index < 8 and view == 1:
                    preview = path / "previews"
                    preview.mkdir(exist_ok=True)
                    image.save(preview / f"record-{index}-original.png")
                    canvas.save(preview / f"record-{index}-geometry.png")
        vit_features[index] = vit(torch.stack(tensors).cuda()).cpu().numpy()
        geometry[index, 1:] = dino(torch.stack(geometry_tensors).cuda()).cpu().numpy()
    require(np.isfinite(vit_features).all() and np.isfinite(geometry).all(), "Nonfinite image features")
    np.savez_compressed(path / "vit.npz", features=vit_features, sample_ids=np.asarray(ids), image_present=present)
    np.savez_compressed(path / "geometry.npz", features=geometry, sample_ids=np.asarray(ids), image_present=present)
    receipt = dict(old_checkpoint_sha256=digest(args.old_checkpoint), old_contrastive_sha256=digest(args.old_contrastive),
                   old_zenodo_sha256=digest(args.old_zenodo), old_encoder_identical_to_zenodo=same_zenodo,
                   old_encoder_identical_to_contrastive=True, old_SYSU_fusion_weights_used_for_new_training=False,
                   dino_encoder_sha256=digest(args.encoder), input_cache_sha256=digest(args.cache / "baseline.npz"),
                   image_records=int(present.sum()), sample_ids=ids, csv_sha256=digest(args.input / "sysu_samples.csv"),
                   protocol_sha256=digest(args.input / "development_nested_cv_v3.json"), old_test_new_inference=False,
                   augmentation_geometry=dict(rotation_degrees=5, translation_fraction=.02, color_changed=False, full_image_preserved=True),
                   files_sha256={name: digest(path / name) for name in ("vit.npz", "geometry.npz")})
    write(ready, receipt)
    del vit, dino, saved, original, zenodo
    torch.cuda.empty_cache()
    print(json.dumps(dict(event="features_ready", image_records=int(present.sum()), old_encoder_identical_to_zenodo=same_zenodo)), flush=True)
    return receipt


def pretrain(model, data, ids, fit, config):
    if not config["pretrain_epochs"]:
        return []
    # Preserve the supervised random sequence when the pretraining branch is added.
    history = []
    with torch.random.fork_rng(devices=[0]):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        ix = data.indices(ids, True)
        for epoch in range(config["pretrain_epochs"]):
            check_time()
            model.train()
            loss_values = []
            for chunk in ix[torch.randperm(len(ix))].split(32):
                batch = pack(data, chunk, fit, True, config)
                embeddings, present = embedding_inputs(model, batch)
                losses = []
                for a, b in itertools.combinations(model.fusion.names, 2):
                    observed = present[a] & present[b]
                    if int(observed.sum()) >= 2:
                        losses.append(info_nce_loss(embeddings[a][observed], embeddings[b][observed], .1))
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                require(torch.isfinite(loss), "Nonfinite contrastive loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_values.append(float(loss.detach()))
            history.append(dict(epoch=epoch + 1, loss=float(np.mean(loss_values)) if loss_values else None))
    return history


def stage(data, train_ids, val_ids, config, directory, seed, epochs, identity):
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / "done.json"
    if done.exists():
        result = json.loads(done.read_text())
        require(result["identity"] == identity and result["config"] == config, "Cannot reuse an unrelated stage")
        if val_ids is None:
            require(digest(directory / "classifier.pt") == result["classifier_sha256"], "Saved classifier changed")
        return result
    check_time()
    check_gpu()
    set_seed(seed)
    fit = data.fit(train_ids, "baseline")
    require(set(fit["fit_ids"]) <= set(train_ids) and not set(fit["fit_ids"]).intersection(val_ids or []), "Fit scope is invalid")
    model = make_model(config, fit)
    contrastive_history = pretrain(model, data, train_ids, fit, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    weights = torch.tensor(fit["class_weights"], device="cuda") if config["balanced"] else None
    ix = data.indices(train_ids, True)
    history = []
    best_key, best_epoch, stale = (-1., -1.), None, 0
    first_epoch = 1
    checkpoint_path = directory / "last.pt"
    if checkpoint_path.exists():
        previous = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
        require(previous["identity"] == identity and previous["config"] == config, "Resume identity differs")
        model.load_state_dict(previous["state"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        history, best_key, best_epoch, stale = previous["history"], tuple(previous["best_key"]), previous["best_epoch"], previous["stale"]
        first_epoch = previous["epoch"] + 1
        torch.set_rng_state(previous["cpu_rng"].cpu())
        torch.cuda.set_rng_state(previous["cuda_rng"].cpu())
    for epoch in range(first_epoch, epochs + 1):
        check_time()
        if val_ids is not None and stale >= config["patience"]:
            break
        model.train()
        total = 0.
        for chunk in ix[torch.randperm(len(ix))].split(8):
            batch = dropout(pack(data, chunk, fit, True, config), config)
            output = forward(model, batch)
            loss = ordinal_loss(output, batch["label"]) if config["head"] == "ordinal" else F.cross_entropy(output, batch["label"], weight=weights)
            require(torch.isfinite(loss), "Nonfinite supervised loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(chunk)
        item = dict(epoch=epoch, training_loss=total / len(ix))
        if val_ids is not None:
            m, _ = evaluate(model, data, val_ids, fit, config)
            item["inner_validation"] = m
            primary = m["fixed_median_0.5"]
            key = primary["accuracy_all_assigned"], primary["macro_f1"]
            if key > best_key:
                best_key, best_epoch, stale = key, epoch, 0
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(item)
        REFERENCE.save_torch(checkpoint_path, dict(state=model.state_dict(), optimizer=optimizer.state_dict(),
                             history=history, best_key=best_key, best_epoch=best_epoch, stale=stale,
                             epoch=epoch, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                             identity=identity, config=config))
    require(best_epoch is not None, "No training epoch was completed")
    result = dict(identity=identity, config=config, fit=fit, selected_epoch=best_epoch, history=history,
                  contrastive_history=contrastive_history, seed=seed)
    if val_ids is None:
        REFERENCE.save_torch(directory / "classifier.pt", dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              fit=fit, config=config, identity=identity, selected_epoch=best_epoch))
        result["classifier_sha256"] = digest(directory / "classifier.pt")
    write(done, result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def make_data(args, protocol, config):
    data = REFERENCE.Data(args.input / "sysu_samples.csv", protocol, args.cache)
    if config["encoder"] == "vit":
        with np.load(args.output / "features/vit.npz", allow_pickle=False) as cached:
            require(cached["sample_ids"].tolist() == data.ids, "ViT feature order mismatch")
            data.images["baseline"] = torch.from_numpy(cached["features"].copy())
    return data


def train_variant(args, protocol, name, config):
    root = args.output / "runs" / name
    root.mkdir(parents=True, exist_ok=True)
    contract = dict(config=config, csv_sha256=digest(args.input / "sysu_samples.csv"),
                    cv_sha256=digest(args.input / "development_nested_cv_v3.json"),
                    feature_receipt_sha256=digest(args.output / "features/receipt.json"), source_sha256=digest(__file__))
    freeze = root / "freeze.json"
    if freeze.exists():
        require(json.loads(freeze.read_text()) == contract, "Started variant changed")
    else:
        write(freeze, contract)
    data = make_data(args, protocol, config)
    oof = []
    for fold in protocol["folds"]:
        check_time()
        folder = root / f"fold{fold['fold']}"
        inner = fold["inner_train"]["sample_ids"]
        validation = fold["inner_val"]["sample_ids"]
        train = fold["outer_train"]["sample_ids"]
        held = fold["outer_holdout"]["sample_ids"]
        require(not set(inner).intersection(validation) and set(inner) | set(validation) == set(train), "Invalid inner split")
        require(not set(train).intersection(held) and set(train) | set(held) == set(data.ids), "Invalid outer split")
        for left, right in ((train, held), (inner, validation)):
            l = [e for e in protocol["development_entries"] if e["sample_id"] in left]
            r = [e for e in protocol["development_entries"] if e["sample_id"] in right]
            for field in ("cv_component", "image_sha256", "dna_report_sha256"):
                require(not {e[field] for e in l if e[field]}.intersection(e[field] for e in r if e[field]), "Group or source crosses folds")
        identity = dict(name=name, fold=fold["fold"], role="inner", train_ids=inner, validation_ids=validation)
        inner_result = stage(data, inner, validation, config, folder / "inner", 42 + config["seed_offset"] + fold["fold"], config["max_epochs"], identity)
        identity = dict(name=name, fold=fold["fold"], role="refit", train_ids=train, validation_ids=[])
        trained = stage(data, train, None, config, folder / "refit", 142 + config["seed_offset"] + fold["fold"], inner_result["selected_epoch"], identity)
        output = folder / "outer_predictions.json"
        if output.exists():
            saved = json.loads(output.read_text())
            require(saved["contract"] == contract and saved["classifier_sha256"] == trained["classifier_sha256"], "Old predictions are unrelated")
            require([r["sample_id"] for r in saved["records"]] == held, "Holdout predictions have wrong IDs")
        else:
            artifact = torch.load(folder / "refit/classifier.pt", map_location="cpu", weights_only=True)
            model = make_model(config, trained["fit"])
            model.load_state_dict(artifact["state"], strict=True)
            metrics, rows = evaluate(model, data, held, trained["fit"], config)
            saved = dict(contract=contract, records=rows, metrics=metrics, classifier_sha256=trained["classifier_sha256"], selected_epoch=inner_result["selected_epoch"])
            write(output, saved)
            del model
            torch.cuda.empty_cache()
        oof.extend(saved["records"])
        print(json.dumps(dict(event="fold_complete", variant=name, fold=fold["fold"], selected_epoch=inner_result["selected_epoch"], correct=saved["metrics"]["fixed_median_0.5"]["correct_count"])), flush=True)
    require(len(oof) == len({r["sample_id"] for r in oof}) == 117, "OOF coverage is incomplete")
    result = dict(config=config, records=oof, metrics=summarize(oof), contract=contract,
                  old_test_new_inference=False, evaluation="exploratory fixed development five-fold OOF")
    write(root / "oof_results.json", result)
    print(json.dumps(dict(event="variant_complete", variant=name, metrics=result["metrics"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "cache", "reference", "old-checkpoint", "old-contrastive", "old-zenodo", "encoder", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("verify", "features", "core", "details"), default="core")
    parser.add_argument("--seconds", type=float, default=7200)
    args = parser.parse_args()
    global DEADLINE
    DEADLINE = time.monotonic() + args.seconds
    require(args.output.resolve().is_relative_to(Path("/home/zjkj/24334067")), "Output outside personal directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    check_gpu()
    protocol = json.loads((args.input / "development_nested_cv_v3.json").read_text())
    require(digest(args.input / "sysu_samples.csv") == protocol["csv_sha256"], "Data changed")
    require(protocol["development_count"] == 117, "Unexpected cohort")
    require(not {e["sample_id"] for e in protocol["development_entries"]}.intersection(protocol["locked_previous_test_ids"] + protocol["additional_related_locked_ids"]), "Locked records in development")
    write(args.output / f"command-{args.phase}.json", dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
          source_sha256=digest(__file__), started_at_unix=time.time(), host=os.uname().nodename))
    verify_reference(args, protocol)
    if args.phase == "verify":
        return
    prepare_features(args, protocol)
    if args.phase == "features":
        return
    variants = {
        "dino_softmax_balanced": configuration(),
        "vit_ordinal": configuration(encoder="vit", head="ordinal", balanced=False),
        "vit_softmax": configuration(encoder="vit", balanced=False),
        "dino_ordinal": configuration(head="ordinal", balanced=False),
        "dino_softmax": configuration(balanced=False),
    }
    if args.phase == "details":
        variants = {"width256": configuration(dimension=256),
                    "legacy_subset": configuration(dropout="legacy_subset"),
                    "contrastive30": configuration(pretrain_epochs=30),
                    "dna_log_standardized": configuration(dna_scale="log_standardized")}
    summary = {}
    for name, config in variants.items():
        result = train_variant(args, protocol, name, config)
        summary[name] = result["metrics"]
        write(args.output / f"summary-{args.phase}.json", summary)
    write(args.output / f"completed-{args.phase}.json", dict(completed=True, variants=list(variants), old_test_new_inference=False, completed_at_unix=time.time()))


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as error:
        print(json.dumps(dict(status="budget_stopped", reason=str(error))), flush=True)
        raise SystemExit(75)
