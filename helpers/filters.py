import numpy as np
from scipy import signal
from typing import Callable, Iterable
from functools import reduce, partial
from scipy.fftpack import dct
import pywt


def normalize_signal(x: np.ndarray,
                     mean: np.ndarray,
                     std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def time_filtering(
    x: np.ndarray,
    order: int,
    low_freq: float,
    high_freq: float,
    fs: float
) -> np.ndarray:
    """
    Band-pass filter in the time domain.

    Args:
        x         : (T, C) raw signal
        order     : filter order
        low_freq  : lower cutoff (Hz)
        high_freq : upper cutoff (Hz)
        fs        : sampling rate (Hz)
    Returns:
        (T, C) filtered signal
    """
    sos = signal.butter(
        N=order,
        Wn=(low_freq, high_freq),
        btype="bandpass",
        output="sos",
        fs=fs
    )
    return signal.sosfiltfilt(sos, x, axis=0).copy()

def make_time_filter(
    order: int,
    low_freq: float,
    high_freq: float,
    fs: float
):
    """
    Hydra factory: returns a callable filter_fn(x)
    with order, low/high cutoffs, and fs baked in.
    """
    return partial(
        time_filtering,
        order=order,
        low_freq=low_freq,
        high_freq=high_freq,
        fs=fs
    )

def fft_filtering(
    x: np.ndarray,
    low_freq: float = 0.5,
    high_freq: float = 30.0,
    fs: float = 250.0,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute the FFT magnitude, log‐scale it, and keep only the
    frequency bins between low_freq and high_freq (in Hz).

    Args:
        x          : (T, C) raw time‐series
        low_freq   : lower cutoff frequency in Hz
        high_freq  : upper cutoff frequency in Hz
        fs         : sampling frequency in Hz
        eps        : floor for magnitude before log

    Returns:
        Array of shape (n_bins, C) where
        n_bins = int(high_freq * T / fs) - int(low_freq * T / fs)
    """
    # 1) FFT magnitude
    X = np.abs(np.fft.fft(x, axis=0))
    # 2) log‐scale with floor
    X = np.log(np.where(X > eps, X, eps))

    # 3) translate freq to bin indices
    T = X.shape[0]
    start = int(low_freq  * T / fs)
    end   = int(high_freq * T / fs)

    return X[start:end, :]

def make_fft_filter(low_freq: float, high_freq: float, fs: float, eps: float, **kwargs):
    """
    Hydra factory: returns a single-arg callable so you only
    need to pass `x` at runtime.
    """
    return partial(fft_filtering,
                   low_freq=low_freq,
                   high_freq=high_freq,
                   fs=fs,
                   eps=eps)

def decimate_signal(
    x: np.ndarray,
    q: int = 10
) -> np.ndarray:
    """
    Anti-alias + downsample by factor q.
    Input:  (3000, 19), Output: (3000/q, 19)
    """
    y = signal.decimate(x, q, axis=0, ftype='iir', zero_phase=True)
    # ensure positive strides / contiguous memory
    return np.ascontiguousarray(y)

def make_decimate_filter(q: int, **kwargs):
    """
    Hydra factory: returns a single-arg callable so you only
    need to pass `x` at runtime.
    """
    return partial(decimate_signal, q=q)

def window_downsample(x: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Non-overlapping mean over each 'window' timesteps.
    (3000, 19) → (3000//window, 19)
    """
    B, C = x.shape
    n = B // window
    x = x[: n*window].reshape(n, window, C)
    return x.mean(axis=1)

def make_window_downsample_filter(window: int, **kwargs):
    """
    Hydra factory: returns a single-arg callable so you only
    need to pass `x` at runtime.
    """
    return partial(window_downsample, window=window)

def dct_downsample(x: np.ndarray, K: int = 500) -> np.ndarray:
    """
    DCT-II along time, keep first K coefficients.
    (3000, 19) → (K, 19)
    """
    X = dct(x, axis=0, norm='ortho')
    return X[:K]

def make_dct_downsample_filter(K: int, **kwargs):
    """
    Hydra factory: returns a single-arg callable so you only
    need to pass `x` at runtime.
    """
    return partial(dct_downsample, K=K)

def wavelet_approx(x: np.ndarray, wavelet='db4', level=4) -> np.ndarray:
    """
    Keep only the level-L approximation coefficients.
    (3000, 19) → (len(cA_L), 19), typically ~3000/2**level
    """
    coeffs = [pywt.wavedec(x[:,ch], wavelet, level=level) 
              for ch in range(x.shape[1])]
    # wavedec returns [cA_L, cD_L, ..., cD1]
    # we only need cA_L for each channel:
    cAs = np.stack([c[0] for c in coeffs], axis=1)
    return cAs

def make_wavelet_approx_filter(wavelet: str, level: int, **kwargs):

    """
    Hydra factory: returns a single-arg callable so you only
    need to pass `x` at runtime.
    """
    return partial(wavelet_approx, wavelet=wavelet, level=level)

def make_pipeline(steps: Iterable[Callable]) -> Callable:
    """
    Return a function that is the composition:
        funcs[-1](...funcs[1](funcs[0](x))...)
    """
    return lambda x: reduce(lambda v, f: f(v), steps, x)