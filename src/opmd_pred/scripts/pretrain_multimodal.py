import argparse
from pathlib import Path
from loguru import logger
from itertools import combinations

import torch
from torch.utils.data import DataLoader

from opmd_pred.data.sysu import SysuDataset, assign_tct_ids, build_tct_vocab, filter_observed_records, load_sysu_records, split_sysu_records
from opmd_pred.model.fusion import MultimodalOrdinalModel
from opmd_pred.model.losses import info_nce_loss
from opmd_pred.scripts.common import configure_logger, device_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/processed/sysu/legacy/sysu_samples.csv")
    parser.add_argument("--images", default="data/processed/sysu/legacy")
    parser.add_argument("--image-checkpoint", default="")
    parser.add_argument("--output", default="outputs/experiments/sysu/multimodal_contrastive")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_image_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.fusion.image_encoder.load_state_dict(checkpoint["image_encoder"])


def contrastive_loss(model, batch, temperature):
    embeddings, present = model.fusion.modality_embeddings(batch)
    losses = []
    for first_name, second_name in combinations(model.fusion.names, 2):
        available = present[first_name] & present[second_name]
        if available.sum() < 2:
            continue
        losses.append(
            info_nce_loss(
                embeddings[first_name][available],
                embeddings[second_name][available],
                temperature,
            )
        )
    return torch.stack(losses).mean() if losses else embeddings["image"].sum() * 0.0


def main():
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    configure_logger(output)
    logger.warning("Legacy exploratory sample split only; not a patient-independent clinical protocol")
    device = torch.device(device_name())
    records = load_sysu_records(args.csv)
    records, excluded = filter_observed_records(records)
    train_records, _ = split_sysu_records(records, seed=args.seed)
    tct_vocab = build_tct_vocab(train_records)
    assign_tct_ids(records, tct_vocab)
    dataset = SysuDataset(train_records, args.images, train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = MultimodalOrdinalModel(
        dimension=args.dimension,
        pretrained_image=not bool(args.image_checkpoint),
        tct_categories=max(tct_vocab.values(), default=1),
    ).to(device)
    load_image_checkpoint(model, args.image_checkpoint, device)
    model.fusion.freeze_image()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            loss = contrastive_loss(model, batch, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info("epoch={} contrastive_loss={:.4f}", epoch, total_loss / len(loader))
    torch.save({"fusion": model.fusion.state_dict(), "tct_vocab": tct_vocab,
                "protocol": "legacy_exploratory_sample_split", "excluded": excluded}, output / "checkpoint.pt")


if __name__ == "__main__":
    main()
