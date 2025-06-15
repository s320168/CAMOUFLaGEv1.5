from pathlib import Path

import torch
from torchvision.datasets import VisionDataset
from PIL import Image
import glob

from torchvision.transforms import transforms


class DatasetImgFolder(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, out_crop=None):
        super(DatasetImgFolder, self).__init__(root, transform=transform, target_transform=target_transform)

        self.samples = glob.glob(root + f'/**/*.png', recursive=True) + \
                       glob.glob(root + f'/**/*.jpg', recursive=True)
        if out_crop is not None:
            self.samples = [s for s in self.samples if not (Path(out_crop) / Path(s).name).exists()]
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        file_name = self.samples[index]
        image = Image.open(file_name)

        if self.transform is not None:
            image = self.transform(image)

        return image, file_name

    def __len__(self):
        return len(self.samples)
