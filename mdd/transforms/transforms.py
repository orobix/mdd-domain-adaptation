import numbers
from typing import List, Tuple

import albumentations
import numpy
from albumentations.augmentations import functional as F


def _check_size(size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    return size


class FiveCrop(albumentations.core.transforms_interface.ImageOnlyTransform):
    def __init__(
        self,
        size,
        always_apply=False,
        p=1.0,
    ):
        super(FiveCrop, self).__init__(always_apply, p)
        self.size = _check_size(size)

    def apply(self, image, **params):
        return five_crop(image, self.size)

    def get_transform_init_args_names(self):
        return "size"


def five_crop(
    img: numpy.ndarray, size: List[int]
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Crop the given image into four corners and the central crop.
    Args:
        img (numpy.ndarray): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0]).
    Returns:
       tuple: tuple (tl, tr, bl, br, center)
       Corresponding top left, top right, bottom left, bottom right and center crop.
    """

    image_height, image_width = img.shape[:2]
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = F.crop(img, 0, 0, crop_width, crop_height)
    tr = F.crop(img, image_width - crop_width, 0, image_width, crop_height)
    bl = F.crop(img, 0, image_height - crop_height, crop_width, image_height)
    br = F.crop(
        img,
        image_width - crop_width,
        image_height - crop_height,
        image_width,
        image_height,
    )

    center = F.center_crop(img, crop_height, crop_width)

    return tl, tr, bl, br, center


class TenCrop(albumentations.core.transforms_interface.ImageOnlyTransform):
    def __init__(
        self,
        size,
        always_apply=False,
        p=1.0,
    ):
        super(TenCrop, self).__init__(always_apply, p)
        self.size = _check_size(size)

    def apply(self, image, **params):
        return ten_crop(image, self.size)

    def get_transform_init_args_names(self):
        return "size"


def ten_crop(
    img: numpy.ndarray, size: List[int], vertical_flip: bool = False
) -> List[numpy.ndarray]:
    """Generate ten cropped images from the given image.
    Crop the given image into four corners and the central crop plus the
    flipped version of these (horizontal flipping is used by default).
    Args:
        img (numpy.ndarray): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0]).
        vertical_flip (bool): Use vertical flipping instead of horizontal
    Returns:
        tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
        Corresponding top left, top right, bottom left, bottom right and
        center crop and same for the flipped image.
    """

    first_five = five_crop(img, size)

    if vertical_flip:
        img = F.vflip(img)
    else:
        img = F.hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five
