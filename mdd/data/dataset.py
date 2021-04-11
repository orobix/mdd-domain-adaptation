""" 
Adapted from https://github.com/pytorch/vision/blob/e8dded4c05ee403633529cef2e09bf94b07f6170/torchvision/datasets/folder.py#L142
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import torch.utils.data as data
import torchvision

from . import IMG_EXTENSIONS
from .data_helper import find_classes, make_dataset

_logger = logging.getLogger(__name__)


class ImageListDataset(data.Dataset):
    """A generic data loader where the samples are arranged in a txt file in this way: ::
        img_path_1.ext class_i1
        img_apth_2.ext class_i1
        img_path_3.ext class_i3
        ...
        img_path_n class_im
    Args:
        path (string): path to txt file. The txt file must be contained in the
            dataset root folder and must follow this structure:
                - images/
                    img1.ext
                    img2.ext
                    ...
                    imgN.ext
                - images_and_targets.txt
            where images_and_targets.txt contains:
                - images/img1.ext class_i1
                - images/img2.ext class_i1
                - ...
                - images/imgN.ext class_iM
        root_dir (str, optional): the root dir of the dataset folder
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        test_10crop (bool, optional): whether to transform images to obtain
            a 10 crop used for testing purpose. If 10 crop is used, than
            the transformartion must be a ``torchvision.transforms``
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        path: str,
        root_dir: Optional[str] = "",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_10crop: Optional[bool] = False,
    ):
        super(ImageListDataset, self).__init__()
        images_list = open(path).readlines()
        samples = self.make_dataset(root_dir, images_list)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + path))

        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.classes = list(set(self.targets))

        self.transform = transform
        self.target_transform = target_transform
        self.test_10crop = test_10crop

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            if self.test_10crop:
                img = self.transform(img)
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def make_dataset(root_dir: str, images_list: List[str]) -> List[Tuple[str, int]]:
        samples = [
            (os.path.join(root_dir, img), int(label))
            for img, label in (img_n_label.split() for img_n_label in images_list)
        ]
        return samples


class ImageFolderDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/[...]/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/[...]/asd932_.ext
    Args:
        root (string): Root directory path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
            If ``None`` default image extensions will be used,
            such as: ".jpg",".jpeg",".png",".ppm",".bmp",".pgm",".tif",".tiff",".webp",
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        test_10crop (bool, optional): whether to transform images to obtain
            a 10 crop used for testing purpose. If 10 crop is used, than
            the transformartion must be a ``torchvision.transforms``
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        test_10crop: Optional[bool] = False,
    ) -> None:
        super(ImageFolderDataset, self).__init__()
        classes, class_to_idx = self.find_classes(root)
        if extensions is None:
            extensions = IMG_EXTENSIONS
        samples = self.make_dataset(
            root,
            class_to_idx,
            extensions if is_valid_file is None else None,
            is_valid_file,
        )

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.test_10crop = test_10crop
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return make_dataset(
            directory,
            class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
        )

    def find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """Same as :func:`find_classes`.
        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.
        """
        return find_classes(dir)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            if self.test_10crop:
                img = self.transform(img)
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)


class CyclicDataset(data.Dataset):
    def __init__(self, *datasets, num_iterations=10000):
        self.datasets = datasets
        self.num_iter = num_iterations

    def __getitem__(self, i):
        result = []
        for dataset in self.datasets:
            cycled_i = i % len(dataset)
            result.append(dataset[cycled_i])
        return tuple(result)

    def __len__(self):
        return self.num_iter
