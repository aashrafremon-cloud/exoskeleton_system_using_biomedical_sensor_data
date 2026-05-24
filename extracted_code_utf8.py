--- EDA&Preprocessing (1).ipynb ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.signal import butter, filtfilt, welch
import os
import glob
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

normal_path = 'N_TXT'
abnormal_path = 'A_TXT'

# Activity mapping
activity_map = {
    'mar': 'Walking (March)',
    'pie': 'Leg Extension',
    'sen': 'Knee Flexion'
}

# Channel names
channels = ['RF', 'BF', 'VM', 'ST', 'FX']
# --- end of cell ---
def get_subject_info(filename):
    """Extract subject number, activity type, and group from filename."""
    # Example: 1Amar.txt -> subject=1, activity=mar, group=A(abnormal)
    basename = os.path.basename(filename).replace('.txt', '')
    
    # Find where the letter starts (subject number)
    subject_num = ''
    idx = 0
    for i, c in enumerate(basename):
        if c.isalpha():
            subject_num = basename[:i]
            idx = i
            break
    
    # Group is next letter (N or A)
    group = basename[idx]
    
    # Activity is from index+1 to end
    activity = basename[idx+1:]
    
    return int(subject_num), group, activity

# Load all data files
data_files = []
for group_type, group_path, group_label in [('N', normal_path, 'Normal'), 
                                             ('A', abnormal_path, 'Abnormal')]:
    files = sorted(glob.glob(os.path.join(group_path, '*.txt')))
    for filepath in files:
        subject_num, _, activity = get_subject_info(filepath)
        print(f"Loading file: {filepath} | Subject: {subject_num} | Group: {group_label} | Activity: {activity_map.get(activity, activity)}")
        
        data_files.append({
            'filepath': filepath,
            'group': group_label,
            'subject': subject_num,
            'activity': activity,
            'activity_name': activity_map.get(activity, activity),
        })

# Create DataFrame for easy access
df_info = pd.DataFrame(data_files)
# --- end of cell ---
df_info
# --- end of cell ---
def load_emg_file(filepath):
    """Load EMG data from txt file."""
    data = np.genfromtxt(filepath,skip_header=8,invalid_raise=False)
    df = pd.DataFrame(data[:, :5], columns=channels)
    return df
# --- end of cell ---
def plot_emg_signals(subject_label, group_label, activity):
    """Plot all 5 channels for a given EMG file."""
    filepath = df_info[(df_info['subject'] == subject_label) & (df_info['group'] == group_label) & (df_info['activity'] == activity)]['filepath'].iloc[0]
    print(filepath)
    df = load_emg_file(filepath)

    title = f"{group_label} | {activity_map.get(activity, activity)}"
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, channel in enumerate(channels):
        plt.subplot(len(channels), 1, i + 1)
        plt.plot(df.index, df[channel], label=channel)
        plt.ylabel(channel, fontsize=12, fontweight='bold', rotation=0, labelpad=20)
        if i == len(channels) - 1:
            plt.xlabel('Time (samples)', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
# --- end of cell ---
plot_emg_signals(11, 'Normal', 'mar')
# --- end of cell ---
plot_emg_signals(1, 'Normal', 'mar')
# --- end of cell ---
plot_emg_signals(11,'Normal', 'pie')
# --- end of cell ---
plot_emg_signals(1,'Normal', 'pie')
# --- end of cell ---
plot_emg_signals(11, 'Normal', 'sen')
# --- end of cell ---
plot_emg_signals(1, 'Normal', 'sen')
# --- end of cell ---
plot_emg_signals(11, 'Abnormal', 'mar')
# --- end of cell ---
plot_emg_signals(1, 'Abnormal', 'mar')
# --- end of cell ---
plot_emg_signals(11, 'Abnormal', 'pie')
# --- end of cell ---
plot_emg_signals(1, 'Abnormal', 'mar')
# --- end of cell ---
plot_emg_signals(11, 'Abnormal', 'sen')
# --- end of cell ---
plot_emg_signals(1, 'Abnormal', 'sen')
# --- end of cell ---
def plot_difference(emg, emg_filtered, channel='RF'):
    # Plot original vs filtered signals for one channel
    channel_index = channels.index(channel)
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(emg.index, emg.iloc[:, channel_index], label=f'Original {channel}', color='blue')
    plt.title(f'Original EMG Signal ({channel} Channel)', fontsize=12)
    plt.ylabel(channel, fontsize=12, rotation=0, labelpad=20)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(emg_filtered.index, emg_filtered.iloc[:, channel_index], label=f'Filtered {channel}', color='orange')
    plt.title(f'Filtered EMG Signal ({channel} Channel)', fontsize=12)
    plt.ylabel(channel, fontsize=12, rotation=0, labelpad=20)
    plt.legend()
# --- end of cell ---
def apply_highpass(data: np.ndarray, fs: float = 1000.0, cutoff: float = 20.0, order: int = 4):
    """Apply a high-pass Butterworth filter to EMG data."""
    
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = signal.butter(order, norm_cutoff, btype="highpass")
    df_filtered = pd.DataFrame(signal.filtfilt(b, a, data, axis=0))
    return df_filtered
# --- end of cell ---
emg = load_emg_file(r"A_TXT\1Asen.txt")
emg_filtered = apply_highpass(emg.values)
# --- end of cell ---
plot_difference(emg, emg_filtered, channel='BF')
# --- end of cell ---
def rectify(data: np.ndarray):
    """Rectify EMG signal by taking the absolute value."""
    return pd.DataFrame(np.abs(data))
# --- end of cell ---
emg = load_emg_file(r"N_TXT\1Nsen.txt")
emg_filtered = rectify(emg.values)
# --- end of cell ---
plot_difference(emg, emg_filtered, channel='BF')
# --- end of cell ---
def apply_lowpass(data: np.ndarray, fs: float = 1000.0, cutoff: float = 4.0, order: int = 2):
    """Apply a low-pass Butterworth filter to EMG data."""
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = signal.butter(order, norm_cutoff, btype="lowpass")
    return pd.DataFrame(signal.filtfilt(b, a, data, axis=0))

# --- end of cell ---
emg = load_emg_file(r"N_TXT\10Nsen.txt")
emg_filtered = apply_lowpass(emg.values)
# --- end of cell ---
plot_difference(emg, emg_filtered, channel='BF')
# --- end of cell ---
def normalize_emg(data: np.ndarray, eps: float = 1e-8):
    """Z-score normalize EMG per channel (mean=0, std=1).

    Normalization is performed along the time axis (axis=0):
    - For shape (n_samples,), computes global mean/std.
    - For shape (n_samples, n_channels), computes mean/std per channel.
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return pd.DataFrame((data - mean) / (std + eps))

# --- end of cell ---
emg = load_emg_file(r"N_TXT\1Nsen.txt")
emg_filtered = normalize_emg(emg.values)
# --- end of cell ---
plot_difference(emg, emg_filtered, channel='BF')
# --- end of cell ---
def downsample_emg(data: np.ndarray, original_fs: float = 1000.0, target_fs: float = 100.0):
    """Downsample EMG data from original_fs to target_fs using decimation (1000 → 100, factor=10)."""

    factor = int(round(original_fs / target_fs))
    if not np.isclose(original_fs / target_fs, factor):
        raise ValueError("original_fs / target_fs must be an integer factor for decimation.")
    downsampled = signal.decimate(data, factor, axis=0, zero_phase=True)
    return pd.DataFrame(downsampled)

# --- end of cell ---
emg = load_emg_file(r"N_TXT\1Nsen.txt")
emg_filtered = downsample_emg(emg.values)
# --- end of cell ---
plot_difference(emg, emg_filtered, channel='RF')
# --- end of cell ---
def window_signal(data: np.ndarray, window_size: int, overlap: float = 0.0) -> np.ndarray:
    """Segment data into overlapping windows."""
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1).")

    step = int(window_size * (1.0 - overlap))
    if step <= 0:
        raise ValueError("Step size must be positive. Reduce overlap or window_size.")

    n_samples = data.shape[0]
    indices = []
    start = 0
    while start + window_size <= n_samples:
        indices.append((start, start + window_size))
        start += step

    if len(indices) == 0:
        raise ValueError("Not enough samples for a single window.")

    windows = []
    for start, end in indices:
        segment = data[start:end]
        windows.append(segment)

    return np.stack(windows, axis=0)
# --- end of cell ---
emg = load_emg_file(r"N_TXT\1Nsen.txt")
emg_windowed = window_signal(emg.values, window_size=200, overlap=0.5)
# --- end of cell ---
plt.figure(figsize=(15, 6))

# Original
plt.subplot(2, 1, 1)
plt.plot(emg.iloc[:, 0], label='Original', color='blue')
plt.title('Original Signal')
plt.legend()

# Windowed (first window)
plt.subplot(2, 1, 2)
plt.plot(emg_windowed[0][:, 0], label='Windowed', color='orange')
plt.title('First Window')
plt.legend()

plt.show()
# --- end of cell ---
def preprocess_emg(
    raw_emg: np.ndarray,
    original_fs: float = 1000.0,
    highpass_cutoff: float = 20.0,
    lowpass_cutoff: float = 4.0,
    lowpass_order: int = 4,
    target_fs: float = 100.0,
    window_size_samples: int = 200,
    window_overlap: float = 0.0,
):
    """Full EMG preprocessing pipeline (clean NumPy version)."""

    # 1. High-pass
    hp = apply_highpass(raw_emg, fs=original_fs, cutoff=highpass_cutoff)

    # 2. Rectify
    rect = np.abs(hp.values)

    # 3. Low-pass
    lp = apply_lowpass(rect, fs=original_fs, cutoff=lowpass_cutoff, order=lowpass_order)

    # 4. Normalize
    mean = np.mean(lp.values, axis=0, keepdims=True)
    std = np.std(lp.values, axis=0, keepdims=True)
    norm = (lp.values - mean) / (std + 1e-8)

    # 5. Downsample
    factor = int(round(original_fs / target_fs))
    downsampled = signal.decimate(norm, factor, axis=0, zero_phase=True)

    # 6. Windowing
    windows = window_signal(downsampled, window_size=window_size_samples, overlap=window_overlap)

    return windows
# --- end of cell ---
emg = load_emg_file(r"N_TXT\1Nsen.txt")
windows = preprocess_emg(emg.values, window_overlap=0.5)

print(windows.shape)
# --- end of cell ---
plt.figure(figsize=(15, 6))

# Original
plt.subplot(2, 1, 1)
plt.plot(emg.iloc[:, 1], label='Original', color='blue')
plt.title('Original Signal')
plt.legend()

# Windowed (first window)
plt.subplot(2, 1, 2)
plt.plot(windows[13][:, 1], label='Windowed', color='orange')
plt.title('Window')
plt.legend()

plt.show()
# --- end of cell ---
def process_dataset(root_path):
    all_X = []
    all_y = []

    for folder in ["N_TXT", "A_TXT"]:
        folder_path = os.path.join(root_path, folder)

        for file in os.listdir(folder_path):
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(folder_path, file)

            df = load_emg_file(file_path)

            emg = df.iloc[:, :4].values
            knee = df.iloc[:, 4].values.reshape(-1, 1)

            X = preprocess_emg(emg, window_overlap=0.0)

            # ===== preprocess Knee =====
            factor = int(1000 / 100)
            knee_down = signal.decimate(knee, factor, axis=0, zero_phase=True)

            # windowing
            y = window_signal(knee_down, window_size=200, overlap=0.0)

            all_X.append(X)
            all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, y
# --- end of cell ---
X, y = process_dataset('.')
print(X.shape, y.shape)
# --- end of cell ---
# save preprocessed data
np.savez("emg_processed_dataset.npz", X=X, y=y)
# --- end of cell ---
--- Preprocessing_georgia.ipynb ---
import h5py

file = h5py.File('georgia_processing/AB06/ik/knee_treadmill_01_01.mat', 'r')

print(list(file.keys()))

shape = file['knee_angle_l'].shape
print(shape)
# --- end of cell ---
import h5py
import numpy as np

file = h5py.File('georgia_processing/AB06/emg/emg_treadmill_01_01.mat', 'r')
emg = np.array(file['emg'])
emg = emg.T
print(emg.shape)
# --- end of cell ---
import os
import h5py
import numpy as np
from scipy.signal import resample

# CONFIG
ROOT = "georgia_processing"
X_all = []
y_all = []

# LOAD
for subject in os.listdir(ROOT):

    subject_path = os.path.join(ROOT, subject)

    emg_path = os.path.join(subject_path, "emg")
    ik_path  = os.path.join(subject_path, "ik")

    if not os.path.exists(emg_path) or not os.path.exists(ik_path):
        continue

    for file_name in os.listdir(emg_path):

        if not file_name.endswith(".mat"):
            continue

        try:
            # EMG
            with h5py.File(os.path.join(emg_path, file_name), 'r') as f:
                emg = np.array(f['emg']).T   # (T, 11)

            # IK
            ik_name = file_name.replace("emg_", "knee_")

            with h5py.File(os.path.join(ik_path, ik_name), 'r') as f:
                knee_r = np.array(f['knee_angle_r']).squeeze().T
                knee_l = np.array(f['knee_angle_l']).squeeze().T

            # FIX SHAPES
            knee_r = knee_r.flatten()
            knee_l = knee_l.flatten()

            # combine (T, 2)
            knee = np.stack([knee_r, knee_l], axis=1)

            # ALIGN LENGTH
            target_len = knee.shape[0]
            emg = resample(emg, target_len, axis=0)

            X_all.append(emg)
            y_all.append(knee)

        except Exception as e:
            print(f"Error in {file_name}: {e}")
            continue


# FINAL ARRAYS
X_all = np.array(X_all, dtype=object)
y_all = np.array(y_all, dtype=object)

print("X shape:", X_all.shape)
print("y shape:", y_all.shape)
# --- end of cell ---
print("X shape:", X_all[0].shape)
print("y shape:", y_all[0].shape)
# --- end of cell ---
emg_columns = [
    "gastrocmed",
    "tibialisanterior",
    "soleus",
    "vastusmedialis",
    "vastuslateralis",
    "rectusfemoris",
    "bicepsfemoris",
    "semitendinosus",
    "gracilis",
    "gluteusmedius",
    "rightexternaloblique"
]
import pandas as pd

df_list = []

for i in range(len(X_all)):
    df_emg = pd.DataFrame(X_all[i], columns=emg_columns)
    df_knee = pd.DataFrame(y_all[i], columns=["knee_angle_r", "knee_angle_l"])

    df = pd.concat([df_emg, df_knee], axis=1)
    df_list.append(df)

df_list[0]
# --- end of cell ---
len(df_list)
# --- end of cell ---
channels = [
    "bicepsfemoris",
    "rectusfemoris",
    "semitendinosus",
    "vastusmedialis"
    ]

ik = [
    "knee_angle_r",
    "knee_angle_l"
]
# --- end of cell ---
import matplotlib.pyplot as plt
import numpy as np

def plot_emg_signals(df, title="EMG Signals"):

    # Create subplots
    fig, axes = plt.subplots(len(channels), 1, figsize=(16, 10), sharex=True)

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, channel in enumerate(channels):
        ax = axes[i]

        ax.plot(df.index, df[channel], color= "#d78946", linewidth=1.2)

        ax.set_ylabel(channel, fontsize=10, rotation=0, labelpad=40)
        ax.grid(True, linestyle="--", alpha=0.4)

        # clean style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (samples)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_ik_signals(df, title="Knee Angles"):
    fig, axes = plt.subplots(len(ik), 1, figsize=(16, 5), sharex=True)

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, channel in enumerate(ik):
        ax = axes[i]

        ax.plot(df.index, df[channel], color= "#d78946", linewidth=1.2)

        ax.set_ylabel(channel, fontsize=10, rotation=0, labelpad=40)
        ax.grid(True, linestyle="--", alpha=0.4)

        # clean style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (samples)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- end of cell ---
plot_emg_signals(df_list[0], title="EMG Signals for Sample 1")
# --- end of cell ---
plot_ik_signals(df_list[0], title="Knee Angles for Sample 1")
# --- end of cell ---
plot_emg_signals(df_list[1], title="EMG Signals for Sample 2")
# --- end of cell ---
plot_ik_signals(df_list[1], title="Knee Angles for Sample 2")
# --- end of cell ---
def plot_emg_difference(emg, emg_filtered, channel):
    # Plot original vs filtered signals for one channel
    channel_index = channels.index(channel)
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(emg.index, emg.iloc[:, channel_index], label=f'Original {channel}', color='blue')
    plt.ylabel(channel, fontsize=12, rotation=0, labelpad=30)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(emg_filtered.index, emg_filtered.iloc[:, channel_index], label=f'Filtered {channel}', color='orange')
    plt.ylabel(channel, fontsize=12, rotation=0, labelpad=30)
    plt.legend()
# --- end of cell ---
from scipy import signal

def apply_highpass(data: np.ndarray, fs: float = 1000.0, cutoff: float = 20.0, order: int = 4):
    """Apply a high-pass Butterworth filter to EMG data."""
    
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = signal.butter(order, norm_cutoff, btype="highpass")
    df_filtered = pd.DataFrame(signal.filtfilt(b, a, data, axis=0))
    return df_filtered
# --- end of cell ---
emg = df_list[0][channels]
emg_filtered = apply_highpass(emg.values)
# --- end of cell ---
plot_emg_difference(emg, emg_filtered, channel="bicepsfemoris")
# --- end of cell ---
def rectify(data: np.ndarray):
    """Rectify EMG signal by taking the absolute value."""
    return pd.DataFrame(np.abs(data))
# --- end of cell ---
emg = df_list[0][channels]
emg_filtered = rectify(emg.values)
# --- end of cell ---
plot_difference(emg, emg_filtered, channel='bicepsfemoris')
# --- end of cell ---
def apply_lowpass(data: np.ndarray, fs: float = 1000.0, low=20, high=450, order: int = 2):
    """Apply a low-pass Butterworth filter to EMG data."""
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype="bandpass")
    return pd.DataFrame(signal.filtfilt(b, a, data, axis=0))

# --- end of cell ---
emg = df_list[0][channels]
emg_filtered = apply_lowpass(emg.values)
# --- end of cell ---
plot_difference(emg, emg_filtered, channel='semitendinosus')
# --- end of cell ---
def normalize_emg(data: np.ndarray, eps: float = 1e-8):
    """Z-score normalize EMG per channel (mean=0, std=1).

    Normalization is performed along the time axis (axis=0):
    - For shape (n_samples,), computes global mean/std.
    - For shape (n_samples, n_channels), computes mean/std per channel.
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    return pd.DataFrame((data - mean) / (std + eps))

# --- end of cell ---
emg = df_list[0][channels]
emg_filtered = normalize_emg(emg.values)
# --- end of cell ---
plot_difference(emg, emg_filtered, channel='bicepsfemoris')
# --- end of cell ---
def window_signal(data: np.ndarray, window_size: int, overlap: float = 0.0) -> np.ndarray:
    """Segment data into overlapping windows."""
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1).")

    step = int(window_size * (1.0 - overlap))
    if step <= 0:
        raise ValueError("Step size must be positive. Reduce overlap or window_size.")

    n_samples = data.shape[0]
    indices = []
    start = 0
    while start + window_size <= n_samples:
        indices.append((start, start + window_size))
        start += step

    if len(indices) == 0:
        raise ValueError("Not enough samples for a single window.")

    windows = []
    for start, end in indices:
        segment = data[start:end]
        windows.append(segment)

    return np.stack(windows, axis=0)
# --- end of cell ---
emg = df_list[0][channels]
emg_windowed = window_signal(emg.values, window_size=200, overlap=0.5)
# --- end of cell ---
plt.figure(figsize=(15, 6))

# Original
plt.subplot(2, 1, 1)
plt.plot(emg.iloc[:, 0], label='Original', color='blue')
plt.title('Original Signal')
plt.legend()

# Windowed (first window)
plt.subplot(2, 1, 2)
plt.plot(emg_windowed[0][:, 0], label='Windowed', color='orange')
plt.title('First Window')
plt.legend()

plt.show()
# --- end of cell ---
def preprocess_emg(
    raw_emg: np.ndarray,
    original_fs: float = 1000.0,
    highpass_cutoff: float = 20.0,
    lowpass_order: int = 4,
    window_size_samples: int = 200,
    window_overlap: float = 0.0,
):
    """Full EMG preprocessing pipeline (clean NumPy version)."""

    # 1. High-pass
    hp = apply_highpass(raw_emg, fs=original_fs, cutoff=highpass_cutoff)

    # 2. Rectify
    rect = np.abs(hp.values)

    # 3. Low-pass
    lp = apply_lowpass(rect, fs=original_fs,  low=20, high=450, order=lowpass_order)

    # 4. Normalize
    mean = np.mean(lp.values, axis=0, keepdims=True)
    std = np.std(lp.values, axis=0, keepdims=True)
    norm = (lp.values - mean) / (std + 1e-8)

    # 5. Windowing
    windows = window_signal(norm, window_size=window_size_samples, overlap=window_overlap)

    return windows
# --- end of cell ---
emg = df_list[0][channels]
windows = preprocess_emg(emg.values, window_overlap=0.5)

print(windows.shape)
# --- end of cell ---
plt.figure(figsize=(15, 6))

# Original
plt.subplot(2, 1, 1)
plt.plot(emg.iloc[:, 1], label='Original', color='blue')
plt.title('Original Signal')
plt.legend()

# Windowed (first window)
plt.subplot(2, 1, 2)
plt.plot(windows[13][:, 1], label='Windowed', color='orange')
plt.title('Window')
plt.legend()

plt.show()
# --- end of cell ---
def process_dataset(df_list):
    all_X = []
    all_y = []

    for df in df_list:

        emg = df[channels].values
        knee = df['knee_angle_r'].values.reshape(-1, 1)

        X = preprocess_emg(emg, window_overlap=0.0)

        # ===== preprocess Knee =====
        factor = int(1000 / 100)
        knee_down = signal.decimate(knee, factor, axis=0, zero_phase=True)

        # windowing
        y = window_signal(knee_down, window_size=200, overlap=0.0)

        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, y
# --- end of cell ---
X, y = process_dataset(df_list)
print(X.shape, y.shape)
# --- end of cell ---
# save preprocessed data
np.savez("georgia_processed_data.npz", X=X, y=y)
# --- end of cell ---
