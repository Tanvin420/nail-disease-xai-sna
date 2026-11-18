"""Datasets, transforms (on-the-fly, training-only augmentation) and the component-balanced sampler."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.io import ImageReadMode, decode_image, read_file
from torchvision.transforms import v2


class RandomRot90(torch.nn.Module):
    def forward(self, x):
        return torch.rot90(x, random.randint(0, 3), dims=(-2, -1))


def build_transform(kind: str, size: int, mean, std, aug: dict | None = None, degrade: int | None = None):
    """kind: 'train' | 'eval' | 'degrade' (downscale to `degrade` px, then back up: resolution control)."""
    tail = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    if kind == "train":
        t = [v2.RandomResizedCrop(size, scale=tuple(aug["rrc_scale"]), ratio=tuple(aug["rrc_ratio"]), antialias=True),
             v2.RandomHorizontalFlip(aug["hflip"]), v2.RandomVerticalFlip(aug["vflip"])]
        if aug.get("rot90"):
            t.append(RandomRot90())
        if aug.get("color_jitter"):
            t.append(v2.ColorJitter(*aug["color_jitter"]))
        return v2.Compose(t + tail)
    t = [v2.Resize((size, size), antialias=True)]
    if kind == "degrade":
        t = [v2.Resize((degrade, degrade), antialias=True), v2.Resize((size, size), antialias=True)]
    return v2.Compose(t + tail)


class ImageDS(Dataset):
    def __init__(self, paths: list[str], labels: np.ndarray, transform):
        self.paths, self.labels, self.t = paths, np.asarray(labels, dtype=np.int64), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = decode_image(read_file(self.paths[i]), mode=ImageReadMode.RGB)
        return self.t(x), self.labels[i]


def component_balanced_sampler(df: pd.DataFrame, num_samples: int, generator: torch.Generator):
    """Each class equally likely; within a class each component equally likely; within a component each file.
    weight(image) = 1 / (|component| * #components in its class)."""
    comp_size = df.comp.map(df.comp.value_counts())
    comps_per_class = df.label.map(df.groupby("label").comp.nunique())
    w = 1.0 / (comp_size * comps_per_class)
    return WeightedRandomSampler(torch.as_tensor(w.to_numpy(), dtype=torch.double), num_samples=num_samples,
                                 replacement=True, generator=generator)


def seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
