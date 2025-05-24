from seiz_eeg.dataset import EEGDataset

from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler

import torch
from hydra.utils import instantiate
import numpy as np
import pandas as pd

from helpers.filters import normalize_signal, make_pipeline


def get_datasets(cfg, split="train"):

    # Load
    path = f"{cfg.data.root}/{split}"
    clips = pd.read_parquet(f"{path}/segments.parquet")
    if cfg.data.subset > 0 and split == "train":
        clips = clips.iloc[: cfg.data.subset]

    # Get signal transform
    list_of_transforms = []
    readable_transforms = []
    if cfg.data.normalize:
        mean = torch.load(f"{cfg.repo_root}/data/trn_mean.pt").type(torch.float64).numpy()
        std = torch.load(f"{cfg.repo_root}/data/trn_std.pt").type(torch.float64).numpy()
        norm = partial(normalize_signal, mean=mean, std=std)
        list_of_transforms.append(norm)
        readable_transforms.append("normalize")
    if cfg.signal_transform.is_enabled:
        signal_transform = instantiate(cfg.signal_transform)
        list_of_transforms.append(signal_transform)
        readable_transforms.append(cfg.signal_transform.name)
    if len(list_of_transforms) > 0:
        signal_transform = make_pipeline(list_of_transforms)
    else:
        signal_transform = None 
    print(f"FYI: using the following signal transform: {' -> '.join(readable_transforms)}")

    dataset = EEGDataset(
        clips,
        signals_root=path,
        signal_transform=signal_transform,
        prefetch=cfg.data.prefetch,
        return_id=split == "test",
    )

    if split == "test":
        return dataset

    # Split
    train_size = int(cfg.data.trn_frac * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator) 

    return train_dataset, val_dataset

def get_weighted_sampler(dataset):
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler

def get_dataloaders(cfg):
    dataset_tr, dataset_val = get_datasets(cfg)
    sampler = get_weighted_sampler(dataset_tr) if cfg.data.use_weighted_sampler else None

    trn_dataloader = DataLoader(
        dataset_tr,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        sampler=sampler,
        shuffle=(sampler is None),
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    return trn_dataloader, val_dataloader
