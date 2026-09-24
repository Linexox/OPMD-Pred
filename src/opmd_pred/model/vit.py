import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = vit_b_16(weights=weights)
        self.model.heads = nn.Identity()
        self.output_dim = self.model.hidden_dim

    def forward(self, image):
        return self.model(image)

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, count):
        self.freeze()
        if count == 0:
            return
        for layer in self.model.encoder.layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in self.model.encoder.ln.parameters():
            parameter.requires_grad = True


class ZenodoViTClassifier(nn.Module):
    def __init__(self, pretrained=True, class_count=4, trainable_blocks=2):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.image_encoder.unfreeze_last_blocks(trainable_blocks)
        self.classifier = nn.Linear(self.image_encoder.output_dim, class_count)

    def forward(self, image):
        return self.classifier(self.image_encoder(image))
