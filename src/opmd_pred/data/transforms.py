import torch
from PIL import Image, ImageOps
from torchvision import transforms


class DINOAspectPad:
    def __call__(self, image):
        return ImageOps.pad(
            image, (518, 518), method=Image.Resampling.BILINEAR,
            color=(124, 116, 104), centering=(0.5, 0.5),
        )


def make_image_transform(train: bool, image_encoder="vit_b16"):
    steps = [DINOAspectPad()] if image_encoder == "dinov2_b14" else [transforms.Resize((224, 224))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(steps)


def make_cached_image_transform(train: bool):
    steps = []
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            ]
        )
    steps.extend(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(steps)
