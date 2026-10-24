import torch
import torch.nn as nn

from .vit import ImageEncoder
from opmd_pred.data.sysu import MODALITIES


class NumericEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size * 2, output_size),
            nn.GELU(),
            nn.LayerNorm(output_size),
        )

    def forward(self, values, mask):
        if values.shape != mask.shape or values.shape[-1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} numeric values with a matching availability mask")
        return self.network(torch.cat([values, mask.float()], dim=-1))


class MultimodalFusion(nn.Module):
    names = MODALITIES

    def __init__(
        self,
        dimension=256,
        pretrained_image=True,
        tct_categories=8,
        image_encoder="vit_b16",
        modalities=MODALITIES
    ):
        super().__init__()
        self.names = tuple(modalities)
        if not self.names or len(set(self.names)) != len(self.names) or not set(self.names) <= set(MODALITIES):
            raise ValueError(f"Select distinct modalities from {MODALITIES}")
        
        self.image_encoder       = ImageEncoder(pretrained=pretrained_image, kind=image_encoder) if "image" in self.names else None
        self.image_projection    = nn.Linear(self.image_encoder.output_dim, dimension) if self.image_encoder else None
        self.history_encoder     = NumericEncoder(9, dimension)
        self.dna_encoder         = NumericEncoder(1, dimension)
        self.methylation_encoder = NumericEncoder(1, dimension)
        self.tct_encoder         = nn.Embedding(max(2, tct_categories + 1), dimension)
        self.missing_tokens      = nn.ParameterDict({name: nn.Parameter(torch.randn(dimension) * 0.02) for name in self.names})
        self.cls_token           = nn.Parameter(torch.zeros(dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=dimension * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm        = nn.LayerNorm(dimension)

    def modality_embeddings(self, batch):
        embeddings, present = {}, {}
        for name in self.names:
            if name == "image":
                if "image_features" in batch:
                    features = batch["image_features"]
                else:
                    features = self.image_encoder(batch["image"])
                embeddings[name] = self.image_projection(features)
                present[name] = batch["image_present"].bool()
                
            elif name == "tct":
                embeddings[name] = self.tct_encoder(batch["tct"])
                present[name] = batch["tct_present"].bool()
                
            else:
                encoder = getattr(self, f"{name}_encoder")
                embeddings[name] = encoder(batch[name], batch[f"{name}_mask"])
                present[name] = batch[f"{name}_mask"].any(dim=1)
        
        return embeddings, present

    def forward(self, batch):
        embeddings, present = self.modality_embeddings(batch)
        return self.fuse_embeddings(embeddings, present)

    def fuse_embeddings(self, embeddings, present):
        tokens = []
        for name in self.names:
            missing = self.missing_tokens[name].expand_as(embeddings[name])
            tokens.append(torch.where(present[name].unsqueeze(-1), embeddings[name], missing))
        cls = self.cls_token.expand(tokens[0].size(0), -1).unsqueeze(1)
        sequence = torch.cat([cls, torch.stack(tokens, dim=1)], dim=1)
        return self.norm(self.transformer(sequence)[:, 0])

    def freeze_image(self):
        if self.image_encoder is not None:
            self.image_encoder.freeze()

    def unfreeze_image_blocks(self, count):
        if self.image_encoder is not None:
            self.image_encoder.unfreeze_last_blocks(count)


class OrdinalHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.score = nn.Linear(dimension, 1)
        self.first_threshold = nn.Parameter(torch.tensor(0.0))
        self.threshold_gap = nn.Parameter(torch.tensor(0.0))

    def forward(self, features):
        score = self.score(features)
        thresholds = torch.stack(
            [
                self.first_threshold,
                self.first_threshold + torch.nn.functional.softplus(self.threshold_gap),
            ]
        )
        return score - thresholds


class MultimodalOrdinalModel(nn.Module):
    def __init__(
            self,
            dimension=256,
            pretrained_image=True,
            tct_categories=8,
            image_encoder="vit_b16",
            modalities=MODALITIES,
            head="ordinal",
            fusion_mode="transformer",
            dna_input_size=1
        ):
        super().__init__()
        if dna_input_size not in (1, 2):
            raise ValueError("DNA input must be one legacy value or two threshold-aware values")

        self.fusion = MultimodalFusion(
            dimension=dimension,
            pretrained_image=pretrained_image,
            tct_categories=tct_categories,
            image_encoder=image_encoder,
            modalities=modalities,
        )
        if head not in {"ordinal", "softmax", "corn"} or fusion_mode not in {"transformer", "late"}:
            raise ValueError("Unsupported head or fusion mode")
        self.head_kind = head
        self.fusion_mode = fusion_mode
        output_size = 2 if head == "corn" else 3
        self.ordinal_head = OrdinalHead(dimension) if head == "ordinal" else nn.Linear(dimension, output_size)
        self.late_heads = nn.ModuleDict({
            name: OrdinalHead(dimension)
            if head == "ordinal"
            else nn.Linear(dimension, output_size)
            for name in self.fusion.names
        }) if fusion_mode == "late" else None
        # Historical two-input builders replaced this layer after constructing all heads.
        # Keep that initialization order so existing seeded training remains reproducible.
        if dna_input_size != 1:
            self.fusion.dna_encoder = NumericEncoder(dna_input_size, dimension)

    def forward(self, batch):
        embeddings, present = self.fusion.modality_embeddings(batch)
        return self.forward_from_embeddings(embeddings, present)

    def forward_from_embeddings(self, embeddings, present):
        if self.fusion_mode == "late":
            scores = torch.stack([self.late_heads[name](embeddings[name]) for name in self.fusion.names], dim=1)
            available = torch.stack([present[name] for name in self.fusion.names], dim=1).unsqueeze(-1)
            if self.head_kind == "corn":
                conditional = scores.sigmoid()
                probabilities = torch.stack([conditional[..., 0], conditional[..., 0] * conditional[..., 1]], dim=-1)
            else:
                probabilities = scores.sigmoid() if self.head_kind == "ordinal" else scores.softmax(dim=-1)
            averaged = (probabilities * available).sum(dim=1) / available.sum(dim=1).clamp_min(1)
            # Empty coalitions are an explanation reference only; real no-input rows are excluded.
            prior = 0.5 if self.head_kind in {"ordinal", "corn"} else 1.0 / 3
            averaged = torch.where(available.any(dim=1), averaged, torch.full_like(averaged, prior))
            if self.head_kind == "corn":
                # Pool unconditional cumulative probabilities; reconstruct conditional logits.
                averaged = torch.stack([averaged[..., 0], averaged[..., 1] / averaged[..., 0].clamp_min(1e-7)], dim=-1)
                return torch.logit(averaged.clamp(1e-7, 1 - 1e-7))
            return torch.logit(averaged.clamp(1e-7, 1 - 1e-7)) if self.head_kind == "ordinal" else averaged.clamp_min(1e-7).log()
        features = self.fusion.fuse_embeddings(embeddings, present)
        return self.ordinal_head(features)
