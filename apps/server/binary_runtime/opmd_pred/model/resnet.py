"""Classic 512-dimensional photo encoder, with explicit artifact preprocessing."""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


PREPROCESSING = {
    "input_size": 224,
    "resize": "aspect_preserving_pad",
    "interpolation": "bilinear",
    "padding_rgb": [124, 116, 104],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class ResNet18Encoder(nn.Module):
    output_dim = 512

    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Identity()
        self._frozen = False

    def forward(self, image):
        return self.model(image)

    def freeze(self):
        self._frozen = True
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Frozen BatchNorm running statistics must not learn from SYSU batches.
        return super().train(False if self._frozen else mode)

    def load_artifact(self, path: str | Path):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if (artifact.get("format") != "opmd-resnet18-encoder-v1"
                or artifact.get("feature_dim") != self.output_dim
                or artifact.get("preprocessing") != PREPROCESSING):
            raise ValueError("ResNet-18 artifact architecture or preprocessing mismatch")
        self.load_state_dict(artifact["encoder_state_dict"], strict=True)
        return artifact

    def artifact(self, **metadata):
        return {
            **metadata,
            "format": "opmd-resnet18-encoder-v1",
            "model": "resnet18",
            "feature_dim": self.output_dim,
            "preprocessing": PREPROCESSING,
            "encoder_state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }


class ResNet18Classifier(nn.Module):
    def __init__(self, pretrained=False, class_count=4):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained)
        self.classifier = nn.Linear(self.encoder.output_dim, class_count)

    def forward(self, image):
        return self.classifier(self.encoder(image))
