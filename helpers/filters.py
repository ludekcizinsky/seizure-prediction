import numpy as np
from scipy import signal
from typing import Callable, Iterable
from functools import reduce

def normalize_signal(x: np.ndarray,
                     mean: np.ndarray,
                     std: np.ndarray) -> np.ndarray:
    return (x - mean) / std

def get_filter(cfg):

    if cfg.model.signal_transform == "time_filtering":
        return time_filtering
    elif cfg.model.signal_transform == "fft_filtering":
        return fft_filtering
    else:
        raise ValueError(f"Invalid signal transform: {cfg.model.signal_transform}")
    

def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""
    bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

def make_pipeline(steps: Iterable[Callable]) -> Callable:
    """
    Return a function that is the composition:
        funcs[-1](...funcs[1](funcs[0](x))...)
    """
    return lambda x: reduce(lambda v, f: f(v), steps, x)