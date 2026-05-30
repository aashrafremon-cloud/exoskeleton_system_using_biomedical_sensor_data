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


def find_camargo_root(explicit: Optional[str] = None) -> str:
    """Resolve Camargo dataset directory (env CAMARGO_ROOT, --data-root, or defaults)."""
    if explicit and os.path.isdir(explicit):
        return os.path.abspath(explicit)
    env = os.environ.get("CAMARGO_ROOT")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "Data_repository_for_Camargo"),
        os.path.join(here, "Data_repository_for_Camargo"),
        os.path.join(here, "..", "..", "Data_repository_for_Camargo"),
        r"d:\Downloads\exoskeleton_system_using_biomedical_sensor_data\Data_repository_for_Camargo",
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Data_repository_for_Camargo not found. Set CAMARGO_ROOT or pass --data-root."
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


# Camargo IK table: column 0 = Header (time). knee_angle_r is typically the
# sagittal knee channel with gait-like oscillation (~0-60 deg), not near-constant ~90 deg.
IK_KNEE_COLUMN_INDEX = 7  # index in full column list (0=Header, 7=knee_angle_r on AB06 treadmill)


def _pick_knee_column(columns: List[np.ndarray]) -> np.ndarray:
    """
    Select knee_angle_r from IK numeric columns (skip Header at index 0).

    Uses fixed index when available; fallback picks channel with gait-like
    range (15-90 deg span, mean 0-50 deg, std > 3).
    """
    if len(columns) > IK_KNEE_COLUMN_INDEX:
        col = columns[IK_KNEE_COLUMN_INDEX]
        if col.std() > 1.0 and (col.max() - col.min()) > 10:
            return col.reshape(-1, 1)

    best, best_score = None, -1.0
    for col in columns[1:]:
        if col.std() < 1e-6:
            continue
        span = float(col.max() - col.min())
        if span < 15 or col.min() < -90 or col.max() > 120:
            continue
        if col.mean() < -25 or col.mean() > 75:
            continue
        score = float(col.std()) * min(span, 80.0)
        if score > best_score:
            best_score, best = score, col
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
