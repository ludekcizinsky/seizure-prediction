import pandas as pd
from seiz_eeg.dataset import EEGDataset
from torch.utils.data import DataLoader

from helpers.filters import get_filter


def get_datasets(cfg):

    trn_path = f"{cfg.data.root}/train"
    clips_tr = pd.read_parquet(f"{trn_path}/segments.parquet")
    dataset_tr = EEGDataset(
        clips_tr,
        signals_root=trn_path,
        signal_transform=get_filter(cfg),
        prefetch=cfg.data.prefetch,
    )

    val_path = f"{cfg.data.root}/val"
    clips_val = pd.read_parquet(f"{val_path}/segments.parquet")
    dataset_val = EEGDataset(
        clips_val,
        signals_root=val_path,
        signal_transform=get_filter(cfg),
        prefetch=cfg.data.prefetch,
    )

    return dataset_tr, dataset_val


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
