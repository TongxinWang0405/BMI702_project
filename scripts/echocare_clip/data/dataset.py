"""
Dataset classes for EchoCare-CLIP training and evaluation.
"""

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiDatasetUltrasound(Dataset):
    """
    Multi-source ultrasound dataset with free-text captions.

    Returns per sample:
        image   : (3, image_size, image_size) float tensor, normalized
        caption : str — clinical text description
        dataset : str — source dataset name

    Transforms:
        train  — Resize(288) → RandomCrop(256) → RandomHorizontalFlip → ToTensor → Normalize
        val/test — Resize(256) → CenterCrop(256) → ToTensor → Normalize
    """

    def __init__(self, dataframe: pd.DataFrame, split: str = "train", cfg: dict = None):
        if cfg is None:
            raise ValueError("cfg (config dict) is required")
        self.df = dataframe.reset_index(drop=True)
        self.split = split

        sz = cfg["image_size"]
        mean = cfg["img_mean"]
        std = cfg["img_std"]

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize(int(sz * 1.125)),  # 288 for sz=256
                transforms.RandomCrop(sz),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(sz),
                transforms.CenterCrop(sz),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        return image, row["caption"], row["dataset"]


class LabeledUltrasoundDataset(Dataset):
    """
    Ultrasound dataset for evaluation — includes label and image_path in output.

    Returns per sample:
        image      : (3, image_size, image_size) float tensor, normalized
        caption    : str
        label      : str
        dataset    : str
        image_path : str
    """

    def __init__(self, dataframe: pd.DataFrame, split: str = "test", cfg: dict = None):
        if cfg is None:
            raise ValueError("cfg (config dict) is required")
        self.df = dataframe.reset_index(drop=True)

        sz = cfg["image_size"]
        mean = cfg["img_mean"]
        std = cfg["img_std"]

        # Deterministic transforms for evaluation
        self.transform = transforms.Compose([
            transforms.Resize(sz),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _safe_str(x, default=""):
        if pd.isna(x):
            return default
        return str(x)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self._safe_str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = self._safe_str(row["caption"], default="")
        label = self._safe_str(row.get("label", "unknown"), default="unknown")
        dataset = self._safe_str(row["dataset"], default="unknown")

        return image, caption, label, dataset, image_path
