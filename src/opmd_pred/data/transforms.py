import torch
from torchvision import transforms


def make_image_transform(train: bool):
    steps = [transforms.Resize((224, 224))]
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


def make_cached_image_transform(train: bool, size: int = 224):
    if train:
        steps = [
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.85, 1.18), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ]
    else:
        steps = [
            transforms.ConvertImageDtype(torch.float32),
            transforms.CenterCrop(size),
        ]
    steps.append(
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    )
    return transforms.Compose(steps)
