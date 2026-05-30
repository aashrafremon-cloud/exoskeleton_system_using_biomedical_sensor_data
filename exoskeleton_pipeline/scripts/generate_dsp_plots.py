"""Generate Phase 4 DSP validation plots from one Camargo trial."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from camargo_io import find_camargo_root, load_emg_mat, discover_emg_ik_pairs
from config import IMAGES_DIR, EMG_FS, TARGET_FS
from pipeline import EMGPipeline

os.makedirs(IMAGES_DIR, exist_ok=True)
pipe = EMGPipeline()
root = find_camargo_root()
pairs = discover_emg_ik_pairs(root, modes=["treadmill"], max_files=1)
emg = load_emg_mat(pairs[0][0])
ch = 3
name = pipe.channels[ch]

centered = pipe.remove_dc_offset(emg)
hp = pipe.apply_highpass(centered)
rect = pipe.rectify(hp)
env = pipe.extract_envelope(rect)
env_n = pipe.normalize_envelope(env)
ds = pipe.decimate_signal(env_n)

steps = [
    ("emg_raw", emg[:3000, ch], EMG_FS, "Raw EMG"),
    ("emg_highpass", hp[:3000, ch], EMG_FS, "High-pass 20 Hz"),
    ("emg_rectified", rect[:3000, ch], EMG_FS, "Rectified"),
    ("emg_envelope", env[:3000, ch], EMG_FS, "Envelope 6 Hz"),
    ("emg_downsampled", ds[:300, ch], TARGET_FS, f"Downsampled {TARGET_FS:.0f} Hz"),
]
for fname, y, fs, title in steps:
    t = np.arange(len(y)) / fs
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, y, lw=0.8)
    ax.set_title(f"{title} — {name} (Camargo)")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, f"{fname}_{name}.png"), dpi=150)
    plt.close()
print("DSP plots saved to", IMAGES_DIR)
