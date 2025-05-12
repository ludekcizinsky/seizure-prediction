import pandas as pd
from seiz_eeg.dataset import EEGDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch

from helpers.filters import get_filter, get_normalization


def get_datasets(cfg):

    # Load
    path = f"{cfg.data.root}/train"
    clips = pd.read_parquet(f"{path}/segments.parquet")
    if cfg.data.subset > 0:
        clips = clips.iloc[: cfg.data.subset]

    if cfg.model.normalize:
        mean = torch.tensor([ 1.9907e-03, -1.1654e-03,  2.0860e-03, -7.4935e-04,  4.5839e-03, 2.8718e-04, -4.0319e-04, -7.7969e-05, -4.2803e-03, -1.6963e-03, 1.2549e-03, -2.8467e-04,  2.5901e-04,  6.5623e-03, -3.3983e-03, 1.2480e-03,  1.3762e-03, -1.7694e-03, -5.8233e-03], dtype=torch.float64)
        std = torch.tensor([158.2415, 159.4878, 152.8925, 150.5167, 150.6428, 148.8908, 153.9944, 153.8446, 152.9122, 153.9926, 152.8411, 151.9071, 151.2794, 153.7461, 151.1686, 150.3015, 151.7241, 152.1091, 156.2211], dtype=torch.float64)
        signal_transform = transforms.Compose([get_filter(cfg),get_normalization(mean, std)])
    else:
        signal_transform = get_filter(cfg)

    dataset = EEGDataset(
        clips,
        signals_root=path,
        signal_transform=signal_transform,
        prefetch=cfg.data.prefetch,
    )

    # Split
    train_size = int(cfg.data.trn_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) 

    return train_dataset, val_dataset


def get_dataloaders(cfg):
    dataset_tr, dataset_val = get_datasets(cfg)

    trn_dataloader = DataLoader(
        dataset_tr,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers // 2 + cfg.data.num_workers % 2,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers // 2,
        shuffle=False,
    )

    return trn_dataloader, val_dataloader
