import albumentations as A
from albumentations.pytorch import ToTensorV2


def train(resize_size=256, crop_size=224):
    return A.Compose(
        [
            A.Resize(resize_size, resize_size),
            A.RandomResizedCrop(crop_size, crop_size),
            A.HorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def test(resize_size=256, crop_size=224):
    start_center = int(round((resize_size - crop_size - 1) / 2))
    return A.Compose(
        [
            A.Resize(resize_size, resize_size),
            A.Crop(
                start_center,
                start_center,
                start_center + crop_size,
                start_center + crop_size,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )