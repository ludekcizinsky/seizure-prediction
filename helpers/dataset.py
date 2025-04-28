from seiz_eeg.dataset import EEGDataset


def get_dataloaders(cfg):
    dataset_tr = EEGDataset(
        clips_tr,
        signals_root=DATA_ROOT / "train",
        signal_transform=fft_filtering,
        prefetch=True,  # If your compute does not allow it, you can use `prefetch=False`
    )