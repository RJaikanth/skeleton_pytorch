import os
import PIL.Image as Image

import torch
from torch.utils import data

LABEL_DICT = {
    "buildings": 0,
    "forest": 1,
    "glacier": 2,
    "mountain": 3,
    "sea": 4,
    "street": 5,
}


class IntelImages(data.Dataset):
    """
    Loads the  Intel Images data

    Args:
        df (pd.DataFrame): DataFrame containg metadata.
        transforms (torchvision.transforms, optional): Transforms to be applied to the image. Defaults to None.
        data_type (str): Indicates dataset type. Defaults to train.
    """

    def __init__(self, df, transforms=None, data_type="train"):
        self.df = df
        self.transforms = transforms
        self.type = data_type.lower()

        self.images = df.images.tolist()
        if self.type == "train":
            self.labels = df.labels.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # images
        img = Image.open(self.images[idx])
        if self.transforms is not None:
            img = self.transforms(img)

        if self.type == "train":
            # label
            label = self.labels[idx]
            label = LABEL_DICT.get(label)

            return img, label
        else:
            return img
