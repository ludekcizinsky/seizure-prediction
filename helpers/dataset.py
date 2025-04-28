import pandas as pd
from seiz_eeg.dataset import EEGDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from helpers.filters import get_filter


def get_datasets(cfg):

    # Load
    path = f"{cfg.data.root}/train"
    clips = pd.read_parquet(f"{path}/segments.parquet")
    if cfg.data.subset > 0:
        clips = clips.iloc[: cfg.data.subset]

    dataset = EEGDataset(
        clips,
        signals_root=path,
        signal_transform=get_filter(cfg),
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
