import numpy as np
from scipy import signal

def normalize_min_max(x: np.ndarray) -> np.ndarray:
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-8)

def normalize_z_score(x: np.ndarray) -> np.ndarray:
    normalized = (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + 1e-8)
    return normalized

def time_filtering(x: np.ndarray, fmin: float = 0.5, fmax: float = 40) -> np.ndarray:
    """Filter signal in the time domain"""
    bp_filter = signal.butter(4, (fmin, fmax), btype="bandpass", output="sos", fs=250)
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w fmin and fmax
    return x[int(fmin * win_len // 250) : fmax * win_len // 250]

def mean_pool_downsampling(x: np.ndarray, factor: int) -> np.ndarray:
    """Downsamples the signal using mean pooling over non-overlapping windows."""
    assert x.shape[0] % factor == 0, "Time dimension must be divisible by factor"
    return np.mean(x.reshape(-1, factor, x.shape[1]), axis=1)