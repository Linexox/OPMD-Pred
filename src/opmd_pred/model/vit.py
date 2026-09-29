import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def interpolate_pos_embedding(pos_embedding, grid_old, grid_new):
    cls, grid = pos_embedding[:, :1], pos_embedding[:, 1:]
    dim = grid.shape[-1]
    grid = grid.transpose(1, 2).reshape(1, dim, grid_old, grid_old)
    grid = torch.nn.functional.interpolate(
        grid, size=(grid_new, grid_new), mode="bicubic", align_corners=False
    )
    grid = grid.reshape(1, dim, -1).transpose(1, 2)
    return torch.cat([cls, grid], dim=1)


def resolve_vit_weights(pretrained, weights_name="v1"):
    if not pretrained:
        return None
    if weights_name == "swag_linear":
        return ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    return ViT_B_16_Weights.IMAGENET1K_V1


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True, weights_name="v1", img_size=224):
        super().__init__()
        self.backbone = weights_name
        if weights_name == "dinov2_vitb14":
            import timm

            self.model = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
                dynamic_img_size=True,
            )
            self.output_dim = self.model.num_features
            return
        self.model = vit_b_16(weights=resolve_vit_weights(pretrained, weights_name))
        if img_size != 224:
            self.model.image_size = img_size
            new_embedding = interpolate_pos_embedding(
                self.model.encoder.pos_embedding.detach(), 14, img_size // 16
            )
            self.model.encoder.pos_embedding = torch.nn.Parameter(new_embedding)
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
        if self.backbone == "dinov2_vitb14":
            for layer in self.model.blocks[-count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.model.norm.parameters():
                parameter.requires_grad = True
            return
        for layer in self.model.encoder.layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in self.model.encoder.ln.parameters():
            parameter.requires_grad = True


class ZenodoViTClassifier(nn.Module):
    def __init__(
        self,
        pretrained=True,
        class_count=4,
        trainable_blocks=2,
        weights_name="v1",
        mlp_head=False,
        img_size=224,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained, weights_name=weights_name, img_size=img_size)
        self.image_encoder.unfreeze_last_blocks(trainable_blocks)
        if mlp_head:
            dim = self.image_encoder.output_dim
            self.classifier = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim, class_count),
            )
        else:
            self.classifier = nn.Linear(self.image_encoder.output_dim, class_count)

    def forward(self, image):
        return self.classifier(self.image_encoder(image))
