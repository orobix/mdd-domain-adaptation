import cv2
from torch.utils.data import Dataset


def make_dataset(imgs):
    images = [(x[0], int(x[1])) for x in (x.split() for x in imgs)]
    return images


class ImageList(Dataset):
    def __init__(
        self,
        path,
        labels=None,
        transform=None,
        target_transform=None,
        mode="RGB",
    ):
        image_list = open(path).readlines()
        imgs = make_dataset(image_list)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + path))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == "RGB":
            self.cvt = cv2.COLOR_BGR2RGB
        elif mode == "L":
            self.cvt = cv2.COLOR_BGR2GRAY

    def __getitem__(self, idx):
        path, target = self.imgs[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, self.cvt)
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class CyclicDataset(Dataset):
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
