import os
from typing import Callable, Optional

from mdd import PROJECT_ROOT_DIR

from .dataset import ImageFolderDataset, ImageListDataset


class Office31(ImageFolderDataset):
    def __init__(
        self,
        domain: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_10crop: Optional[bool] = False,
    ):
        assert str.lower(domain) in ["dslr", "amazon", "webcam"]
        super(Office31, self).__init__(
            os.path.join(PROJECT_ROOT_DIR, "datasets", "office-31", domain, "images"),
            transform=transform,
            target_transform=target_transform,
            test_10crop=test_10crop,
        )


class OfficeHome(ImageFolderDataset):
    def __init__(
        self,
        domain: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_10crop: Optional[bool] = False,
    ):
        assert str.lower(domain) in ["art", "clipart", "product", "real word"]
        super(OfficeHome, self).__init__(
            os.path.join(PROJECT_ROOT_DIR, "datasets", "office-home", domain),
            transform=transform,
            target_transform=target_transform,
            test_10crop=test_10crop,
        )


class ImageClef(ImageListDataset):
    def __init__(
        self,
        domain: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_10crop: Optional[bool] = False,
    ):
        assert str.lower(domain) in ["b", "c", "i", "p"]
        root_dir = os.path.join(PROJECT_ROOT_DIR, "datasets", "image-clef")
        super(ImageClef, self).__init__(
            os.path.join(root_dir, domain + ".txt"),
            root_dir=root_dir,
            transform=transform,
            target_transform=target_transform,
            test_10crop=test_10crop,
        )