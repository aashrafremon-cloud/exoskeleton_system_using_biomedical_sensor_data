"""
Load Camargo dataset .mat files (MATLAB v7 compressed tables) for exoskeleton_pipeline.

Dataset layout (see Data_repository_for_Camargo/README.txt):
  <subject>/<date>/<mode>/<sensor>/<trial>.mat

  - emg/  : 11 muscles @ 1000 Hz
  - ik/   : joint angles @ 200 Hz (we use knee_angle_r)
"""
from __future__ import annotations

import glob
import os
import struct
import zlib
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

EMG_FS = 1000.0
IK_FS = 200.0

EMG_CHANNELS = [
    "gastrocmed", "tibialisanterior", "soleus", "vastusmedialis",
    "vastuslateralis", "rectusfemoris", "bicepsfemoris",
    "semitendinosus", "gracilis", "gluteusmedius", "rightexternaloblique",
]

MI_DOUBLE = 9
MI_COMPRESSED = 15


def find_camargo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ("../Data_repository_for_Camargo", "Data_repository_for_Camargo"):
        p = os.path.normpath(os.path.join(here, rel))
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Data_repository_for_Camargo not found next to exoskeleton_pipeline."
    )


def _decompress_mat(path: str) -> bytes:
    """Decompress main MCOS payload from a Camargo .mat file."""
    raw = open(path, "rb").read()
    for pos in range(len(raw) - 8):
        mi_type, nbytes = struct.unpack_from("<II", raw, pos)
        if mi_type == MI_COMPRESSED and nbytes > 1_000_000:
            payload = raw[pos + 8 : pos + 8 + nbytes]
            return zlib.decompress(payload)
    raise ValueError(f"No compressed MCOS payload in {path}")


def _extract_double_columns(dec: bytes, min_size: int = 500) -> List[np.ndarray]:
    """Extract aligned miDOUBLE column vectors from decompressed MCOS stream."""
    columns: List[np.ndarray] = []
    for pos in range(0, len(dec) - 8, 8):
        mi_type, nbytes = struct.unpack_from("<II", dec, pos)
        if mi_type != MI_DOUBLE or nbytes < min_size * 8:
            continue
        arr = np.frombuffer(dec[pos + 8 : pos + 8 + nbytes], dtype=np.float64).copy()
        if arr.size >= min_size and np.isfinite(arr).mean() > 0.99:
            columns.append(arr)
    return columns


def _pick_knee_column(columns: List[np.ndarray]) -> np.ndarray:
    """Heuristic: knee flexion during gait has moderate std in degrees."""
    best = None
    best_score = -1.0
    for col in columns[1:]:
        if col.std() < 1e-6:
            continue
        if col.max() > 150 or col.min() < -90:
            continue
        score = float(col.std())
        if 3.0 < score < 30.0 and score > best_score:
            best_score = score
            best = col
    if best is None:
        raise ValueError("Could not identify knee_angle_r column in IK file.")
    return best.reshape(-1, 1)


def load_emg_mat(path: str) -> np.ndarray:
    """EMG (T, 11) at 1000 Hz."""
    cols = _extract_double_columns(_decompress_mat(path), min_size=1000)
    if len(cols) < 12:
        raise ValueError(f"Expected >=12 numeric columns in EMG file, got {len(cols)}: {path}")
    emg = np.stack(cols[1:12], axis=1)
    if emg.shape[1] != len(EMG_CHANNELS):
        raise ValueError(f"EMG channel count mismatch: {emg.shape}")
    return emg.astype(np.float64)


def load_ik_knee_mat(path: str) -> np.ndarray:
    """Right knee angle (T, 1) at 200 Hz."""
    cols = _extract_double_columns(_decompress_mat(path), min_size=200)
    if len(cols) < 2:
        raise ValueError(f"Expected >=2 numeric columns in IK file: {path}")
    return _pick_knee_column(cols).astype(np.float64)


def discover_emg_ik_pairs(
    root: str,
    modes: Optional[List[str]] = None,
    max_files: Optional[int] = None,
) -> List[Tuple[str, str]]:
    modes = modes or ["treadmill", "levelground", "ramp", "stair"]
    pairs: List[Tuple[str, str]] = []
    for mode in modes:
        pattern = os.path.join(root, "AB*", "*", mode, "emg", "*.mat")
        for emg_path in sorted(glob.glob(pattern)):
            ik_path = emg_path.replace(os.sep + "emg" + os.sep, os.sep + "ik" + os.sep)
            if os.path.isfile(ik_path):
                pairs.append((emg_path, ik_path))
    if max_files:
        pairs = pairs[:max_files]
    return pairs


def sync_emg_knee(emg: np.ndarray, knee: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trim to common duration (EMG @ 1 kHz, knee @ 200 Hz)."""
    min_dur = min(emg.shape[0] / EMG_FS, knee.shape[0] / IK_FS)
    return emg[: int(min_dur * EMG_FS), :], knee[: int(min_dur * IK_FS), :]


def describe_dataset(root: str) -> pd.DataFrame:
    rows = []
    for subj in sorted(glob.glob(os.path.join(root, "AB*"))):
        subj_id = os.path.basename(subj)
        for mode in ("treadmill", "levelground", "ramp", "stair"):
            n = len(glob.glob(os.path.join(subj, "*", mode, "emg", "*.mat")))
            if n:
                rows.append({"subject": subj_id, "mode": mode, "emg_trials": n})
    return pd.DataFrame(rows)
