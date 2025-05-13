from seiz_eeg.dataset import EEGDataset

from functools import partial

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch

import pandas as pd

from helpers.filters import get_filter, normalize_signal, make_pipeline


def get_datasets(cfg):

    # Load
    path = f"{cfg.data.root}/train"
    clips = pd.read_parquet(f"{path}/segments.parquet")
    if cfg.data.subset > 0:
        clips = clips.iloc[: cfg.data.subset]

    list_of_transforms = []
    readable_transforms = []
    if cfg.model.normalize:
        mean = torch.load(f"{cfg.repo_root}/data/trn_mean.pt").type(torch.float64).numpy()
        std = torch.load(f"{cfg.repo_root}/data/trn_std.pt").type(torch.float64).numpy()
        norm = partial(normalize_signal, mean=mean, std=std)
        list_of_transforms.append(norm)
        readable_transforms.append("normalize")
    if cfg.model.signal_transform is not None:
        list_of_transforms.append(get_filter(cfg))
        readable_transforms.append(cfg.model.signal_transform)
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
    )

    # Split
    train_size = int(cfg.data.trn_frac * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator) 

    return train_dataset, val_dataset


def get_dataloaders(cfg):
    dataset_tr, dataset_val = get_datasets(cfg)

    trn_dataloader = DataLoader(
        dataset_tr,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    return trn_dataloader, val_dataloader
