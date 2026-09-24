import argparse
from itertools import combinations
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from loguru import logger

from src.data.sysu import SysuDataset, load_sysu_records, split_sysu_records
from src.model.fusion import MultimodalOrdinalModel
from src.model.losses import info_nce_loss
from src.scripts.common import configure_logger, device_name, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/SYSU_processed/sysu_samples.csv")
    parser.add_argument("--images", default="data/SYSU_processed")
    parser.add_argument("--image-checkpoint", default="")
    parser.add_argument("--output", default="outputs/multimodal_contrastive")
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
    device = torch.device(device_name())
    records = load_sysu_records(args.csv)
    train_records, _ = split_sysu_records(records, seed=args.seed)
    dataset = SysuDataset(train_records, args.images, train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = MultimodalOrdinalModel(
        dimension=args.dimension,
        pretrained_image=not bool(args.image_checkpoint),
        tct_categories=max(record.tct_id for record in records),
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
    torch.save({"fusion": model.fusion.state_dict()}, output / "checkpoint.pt")


if __name__ == "__main__":
    main()
