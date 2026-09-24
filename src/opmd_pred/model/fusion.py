import torch
import torch.nn as nn

from .vit import ImageEncoder


class NumericEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size * 2, output_size),
            nn.GELU(),
            nn.LayerNorm(output_size),
        )

    def forward(self, values, mask):
        return self.network(torch.cat([values, mask.float()], dim=-1))


class MultimodalFusion(nn.Module):
    names = ("image", "demo", "history", "tct", "dna", "methylation")

    def __init__(self, dimension=256, pretrained_image=True, tct_categories=8):
        super().__init__()
        self.image_encoder       = ImageEncoder(pretrained=pretrained_image)
        self.image_projection    = nn.Linear(self.image_encoder.output_dim, dimension)
        self.demo_encoder        = NumericEncoder(2, dimension)
        self.history_encoder     = NumericEncoder(9, dimension)
        self.dna_encoder         = NumericEncoder(1, dimension)
        self.methylation_encoder = NumericEncoder(1, dimension)
        self.tct_encoder         = nn.Embedding(tct_categories + 1, dimension)
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
        self.norm = nn.LayerNorm(dimension)

    def modality_embeddings(self, batch):
        embeddings = {
            "image": self.image_projection(self.image_encoder(batch["image"])),
            "demo": self.demo_encoder(batch["demo"], batch["demo_mask"]),
            "history": self.history_encoder(batch["history"], batch["history_mask"]),
            "tct": self.tct_encoder(batch["tct"]),
            "dna": self.dna_encoder(batch["dna"], batch["dna_mask"]),
            "methylation": self.methylation_encoder(
                batch["methylation"], batch["methylation_mask"]
            ),
        }
        present = {
            "image": batch["image_present"].bool(),
            "demo": batch["demo_mask"].any(dim=1),
            "history": batch["history_mask"].any(dim=1),
            "tct": batch["tct_present"].bool(),
            "dna": batch["dna_mask"].any(dim=1),
            "methylation": batch["methylation_mask"].any(dim=1),
        }
        return embeddings, present

    def forward(self, batch):
        embeddings, present = self.modality_embeddings(batch)
        tokens = []
        for name in self.names:
            missing = self.missing_tokens[name].expand_as(embeddings[name])
            tokens.append(torch.where(present[name].unsqueeze(-1), embeddings[name], missing))
        cls = self.cls_token.expand(tokens[0].size(0), -1).unsqueeze(1)
        sequence = torch.cat([cls, torch.stack(tokens, dim=1)], dim=1)
        return self.norm(self.transformer(sequence)[:, 0])

    def freeze_image(self):
        self.image_encoder.freeze()

    def unfreeze_image_blocks(self, count):
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
    def __init__(self, dimension=256, pretrained_image=True, tct_categories=8):
        super().__init__()
        self.fusion = MultimodalFusion(
            dimension=dimension,
            pretrained_image=pretrained_image,
            tct_categories=tct_categories,
        )
        self.ordinal_head = OrdinalHead(dimension)

    def forward(self, batch):
        return self.ordinal_head(self.fusion(batch))
