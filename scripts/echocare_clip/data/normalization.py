"""
Compute per-channel mean and std over the training set images.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm


class RawImageDataset(Dataset):
    """Loads images as tensors with NO normalization (just [0, 1])."""

    def __init__(self, paths, image_size):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.tf(Image.open(self.paths[idx]).convert("RGB"))


def compute_normalization_stats(image_paths, image_size=256, batch_size=64, num_workers=2):
    """
    Compute per-channel mean and std over a list of image paths.
    Returns (mean_list, std_list) each of length 3 (for RGB).
    """
    dataset = RawImageDataset(image_paths, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=False, shuffle=False)

    mean = torch.zeros(3)
    var = torch.zeros(3)
    n = 0

    print(f"Computing mean and std over {len(dataset):,} images...")
    for imgs in tqdm(loader, desc="Normalization stats"):
        B, C, H, W = imgs.shape
        pixels = imgs.permute(1, 0, 2, 3).reshape(C, -1)  # (3, B*H*W)
        mean += pixels.mean(dim=1)
        var += pixels.var(dim=1)
        n += 1

    mean /= n
    var /= n
    std = var.sqrt()

    mean_list = [round(v, 6) for v in mean.tolist()]
    std_list = [round(v, 6) for v in std.tolist()]

    print(f"  mean : {mean_list}")
    print(f"  std  : {std_list}")

    return mean_list, std_list
