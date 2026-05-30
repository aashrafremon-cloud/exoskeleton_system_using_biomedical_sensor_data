"""Time-domain EMG feature extraction."""
import numpy as np
from typing import List

from config import EMG_CHANNELS


def extract_window_features(window_data: np.ndarray, channel_names: List[str] = None) -> np.ndarray:
    """
    Extract features from envelope window (window_size, n_channels).

    Features per channel: MAV, RMS, WL, VAR, ZC, SSC, IEMG
    """
    channel_names = channel_names or EMG_CHANNELS
    n_ch = window_data.shape[1]
    eps = 1e-8

    mav = np.mean(np.abs(window_data), axis=0)
    rms = np.sqrt(np.mean(window_data ** 2, axis=0) + eps)
    wl = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
    var = np.var(window_data, axis=0)
    iemg = np.sum(np.abs(window_data), axis=0)

    zc = np.zeros(n_ch)
    ssc = np.zeros(n_ch)
    for j in range(n_ch):
        x = window_data[:, j]
        dx = np.diff(x)
        zc[j] = np.sum((x[:-1] * x[1:]) < 0)
        ssc[j] = np.sum((dx[:-1] * dx[1:]) < 0)

    return np.concatenate([mav, rms, wl, var, zc, ssc, iemg])


def get_feature_names(channel_names: List[str] = None) -> List[str]:
    channel_names = channel_names or EMG_CHANNELS
    suffixes = ["mav", "rms", "wl", "var", "zc", "ssc", "iemg"]
    names = []
    for suf in suffixes:
        for ch in channel_names:
            names.append(f"{ch}_{suf}")
    return names
