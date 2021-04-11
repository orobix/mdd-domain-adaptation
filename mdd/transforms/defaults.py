import albumentations
import torch
import torchvision
from albumentations.pytorch import ToTensorV2


def train(resize_size=256, crop_size=224):
    return albumentations.Compose(
        [
            albumentations.Resize(resize_size, resize_size),
            albumentations.RandomResizedCrop(crop_size, crop_size),
            albumentations.HorizontalFlip(),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]
    )


def test(resize_size=256, crop_size=224):
    start_center = int(round((resize_size - crop_size - 1) / 2))
    return albumentations.Compose(
        [
            albumentations.Resize(resize_size, resize_size),
            albumentations.Crop(
                start_center,
                start_center,
                start_center + crop_size,
                start_center + crop_size,
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]
    )


def test_10crop(resize_size=256, crop_size=224):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((resize_size, resize_size)),
            torchvision.transforms.TenCrop((crop_size, crop_size)),
            torchvision.transforms.Lambda(
                lambda crops: torch.stack([crop for crop in crops])
            ),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
