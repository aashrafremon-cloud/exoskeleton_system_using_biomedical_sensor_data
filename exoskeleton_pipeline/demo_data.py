"""Synthetic gait-like EMG + knee data when Camargo files are unavailable (--demo)."""
import numpy as np

from config import EMG_CHANNELS, EMG_FS, IK_FS


def generate_trial(duration_sec: float = 60.0, seed: int = 0) -> tuple:
    """Return (emg [T,11], knee [T,1]) compatible with pipeline sync."""
    rng = np.random.default_rng(seed)
    n_emg = int(duration_sec * EMG_FS)
    n_ik = int(duration_sec * IK_FS)
    t_emg = np.arange(n_emg) / EMG_FS
    t_ik = np.arange(n_ik) / IK_FS
    freq = 0.85

    knee = (
        15
        + 25 * np.sin(2 * np.pi * freq * t_ik)
        + 15 * np.sin(4 * np.pi * freq * t_ik)
        + rng.normal(0, 0.8, n_ik)
    ).reshape(-1, 1)

    emg = np.zeros((n_emg, len(EMG_CHANNELS)))
    for j, ch in enumerate(EMG_CHANNELS):
        phase = 2 * np.pi * freq * t_emg + j * 0.2
        emg[:, j] = np.maximum(0, np.sin(phase)) * (0.3 + 0.05 * j) + rng.normal(0, 0.01, n_emg)

    return emg.astype(np.float64), knee.astype(np.float64)
