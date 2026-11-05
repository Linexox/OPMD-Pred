import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True, kind="vit_b16"):
        super().__init__()
        self.kind = kind
        self._frozen = False
        if kind == "dinov2_b14":
            import timm
            self.model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
            self.output_dim = 1536
        elif kind == "resnet18_fpn":
            from .lesion_detector import FPNPhotoEncoder
            self.model = FPNPhotoEncoder()
            self.output_dim = 256
        elif kind == "resnet18":
            from .resnet import ResNet18Encoder
            self.model = ResNet18Encoder(pretrained=pretrained)
            self.output_dim = self.model.output_dim
        elif kind == "vit_b16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = vit_b_16(weights=weights)
            self.model.heads = nn.Identity()
            self.output_dim = self.model.hidden_dim
        else:
            raise ValueError(f"Unknown image encoder: {kind}")

    def forward(self, image):
        if self.kind == "dinov2_b14":
            features = self.model.forward_features(image)
            return torch.cat([features[:, 0], features[:, 1:].mean(dim=1)], dim=1)
        return self.model(image)

    def load_artifact(self, path):
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if artifact.get('format')=='zenodo-supplement-encoder-v1':
            if artifact['backbone']!=self.kind or artifact['feature_dim']!=self.output_dim:
                raise ValueError('Supplement encoder kind/dimension mismatch')
            if self.kind=='resnet18_fpn':
                expected={'min_size':512,'max_size':1024,'feature_layer':'3','fixed_canvas':True,
                          'mean':[.485,.456,.406],'std':[.229,.224,.225]}
                if any(artifact['preprocessing'].get(k)!=v for k,v in expected.items()):
                    raise ValueError('Unexpected FPN preprocessing')
                self.model.backbone.load_state_dict(artifact['state'],strict=True)
            else:
                from .resnet import PREPROCESSING
                if artifact['preprocessing']!=PREPROCESSING:
                    raise ValueError('Supplement photo preprocessing changed')
                self.model.load_state_dict(artifact['state'],strict=True)
            return artifact
        if self.kind == "resnet18":
            return self.model.load_artifact(path)
        if self.kind == "dinov2_b14":
            if artifact.get("model") != "vit_base_patch14_dinov2.lvd142m":
                raise ValueError("DINO artifact model is incompatible")
            if artifact.get("input_size") != 518 or artifact.get("feature_dim") != 1536:
                raise ValueError("DINO artifact must use 518 pixels and CLS + patch mean (1536)")
            if artifact.get("padding_rgb") != [124, 116, 104] or artifact.get("normalization") != {
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
            }:
                raise ValueError("DINO artifact preprocessing differs from the fixed SYSU transform")
            self.model.load_state_dict(artifact["encoder_state_dict"], strict=True)
        else:
            self.load_state_dict(artifact["image_encoder"], strict=True)
        return artifact

    def train(self, mode=True):
        return super().train(False if self._frozen else mode)

    def freeze(self):
        self._frozen = True
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def unfreeze_last_blocks(self, count):
        if self.kind in {"resnet18","resnet18_fpn"} and count != 0:
            raise ValueError("ResNet-18 does not have Transformer blocks; select its training policy explicitly")
        self.freeze()
        if count == 0:
            return
        layers = self.model.blocks if self.kind == "dinov2_b14" else self.model.encoder.layers
        if count < 0 or count > len(layers):
            raise ValueError(f"Block count must be between 0 and {len(layers)}")
        self._frozen = False
        for layer in layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        norm = self.model.norm if self.kind == "dinov2_b14" else self.model.encoder.ln
        for parameter in norm.parameters():
            parameter.requires_grad = True


class ZenodoViTClassifier(nn.Module):
    def __init__(self, pretrained=True, class_count=4, trainable_blocks=2):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.image_encoder.unfreeze_last_blocks(trainable_blocks)
        self.classifier = nn.Linear(self.image_encoder.output_dim, class_count)

    def forward(self, image):
        return self.classifier(self.image_encoder(image))
