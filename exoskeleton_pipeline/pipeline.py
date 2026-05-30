import numpy as np
import pandas as pd
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

class EMGPipeline:
    def __init__(self, original_fs: float = 1000.0, target_fs: float = 100.0, window_size: int = 20, overlap: float = 0.5):
        """
        EMG signal processing pipeline for exoskeleton joint angle prediction.
        
        Args:
            original_fs: Sampling frequency of raw EMG (typically 1000 Hz)
            target_fs: Downsampled frequency matching joint angle kinetics (typically 100 Hz)
            window_size: Number of samples in the target frequency domain (e.g. 20 samples = 200 ms)
            overlap: Overlap percentage for windowing (0.0 to 1.0)
        """
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.window_size = window_size
        self.overlap = overlap
        
        # 11 Standard EMG channels
        from config import EMG_CHANNELS
        self.channels = list(EMG_CHANNELS)

    def remove_dc_offset(self, data: np.ndarray) -> np.ndarray:
        """Remove per-channel DC offset (mean)."""
        return data - np.mean(data, axis=0, keepdims=True)

    def apply_highpass(self, data: np.ndarray, cutoff: float = 20.0, order: int = 4) -> np.ndarray:
        """Apply a high-pass Butterworth filter to remove movement artifacts."""
        nyq = 0.5 * self.original_fs
        norm_cutoff = cutoff / nyq
        b, a = signal.butter(order, norm_cutoff, btype="highpass")
        # Apply zero-phase filtering
        return signal.filtfilt(b, a, data, axis=0)

    def rectify(self, data: np.ndarray) -> np.ndarray:
        """Rectify the EMG signal by taking its absolute value."""
        return np.abs(data)

    def extract_envelope(self, data: np.ndarray, cutoff: float = 6.0, order: int = 2) -> np.ndarray:
        """Apply a low-pass Butterworth filter to extract the linear envelope."""
        nyq = 0.5 * self.original_fs
        norm_cutoff = cutoff / nyq
        b, a = signal.butter(order, norm_cutoff, btype="lowpass")
        return signal.filtfilt(b, a, data, axis=0)

    def normalize_envelope(self, envelope: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Per-channel max normalization to [0, 1] scale (per trial)."""
        peak = np.max(np.abs(envelope), axis=0, keepdims=True) + eps
        return envelope / peak
        
    def decimate_signal(self, data: np.ndarray) -> np.ndarray:
        """Downsample the signal from original_fs to target_fs using decimation."""
        factor = int(round(self.original_fs / self.target_fs))
        if factor > 1:
            return signal.decimate(data, factor, axis=0, zero_phase=True)
        return data

    def extract_time_features(self, window_data: np.ndarray) -> np.ndarray:
        from features import extract_window_features
        return extract_window_features(window_data, self.channels)

    def get_feature_names(self) -> list:
        from features import get_feature_names
        return get_feature_names(self.channels)

    def process_raw_emg(self, raw_emg: np.ndarray, return_features: bool = True) -> dict:
        """
        Runs the full digital signal processing (DSP) chain.
        
        Steps: High-pass Filter -> Rectify -> Low-pass Envelope -> Downsample -> Window -> Feature Extraction
        
        Args:
            raw_emg: Raw EMG signal of shape (N_samples, num_channels)
            return_features: If True, computes features for each window
            
        Returns:
            Dictionary containing 'windows' and 'features'
        """
        if raw_emg.shape[1] != len(self.channels):
            raise ValueError(f"EMG signal must have exactly {len(self.channels)} channels. Got shape {raw_emg.shape}")

        # 0. DC offset removal
        centered = self.remove_dc_offset(raw_emg)
        # 1. High-pass filtering to remove motion artifacts (>20Hz)
        hp = self.apply_highpass(centered, cutoff=20.0)
        
        # 2. Rectification
        rect = self.rectify(hp)
        
        # 3. Low-pass filtering to extract smooth linear envelope (<6Hz)
        envelope = self.extract_envelope(rect, cutoff=6.0)
        envelope = self.normalize_envelope(envelope)

        # 4. Decimation (Downsampling from 1000Hz to 100Hz)
        downsampled = self.decimate_signal(envelope)
        
        # 5. Windowing
        step = int(self.window_size * (1.0 - self.overlap))
        if step <= 0:
            step = 1
            
        n_samples = downsampled.shape[0]
        windows = []
        features_list = []
        
        start = 0
        while start + self.window_size <= n_samples:
            segment = downsampled[start:start + self.window_size]
            windows.append(segment)
            if return_features:
                features_list.append(self.extract_time_features(segment))
            start += step
            
        return {
            'windows': np.array(windows),
            'features': np.array(features_list) if return_features else None
        }

def process_kinematics(
    raw_kin: np.ndarray,
    original_fs: float = 100.0,
    target_fs: float = 100.0,
    window_size: int = 20,
    overlap: float = 0.5,
    lag_samples: int = 0,
) -> dict:
    """
    Process joint angle kinematics to align perfectly with EMG window intervals.
    Predicts the knee angle at the END of each window.
    """
    factor = int(round(original_fs / target_fs))
    if factor > 1:
        downsampled = signal.decimate(raw_kin, factor, axis=0, zero_phase=True)
    else:
        downsampled = raw_kin
        
    step = int(window_size * (1.0 - overlap))
    if step <= 0:
        step = 1
        
    n_samples = downsampled.shape[0]
    windows = []
    y_reg_list = []
    
    start = 0
    while start + window_size <= n_samples:
        segment = downsampled[start:start + window_size]
        windows.append(segment)
        # Target: knee angle at end of window + physiological lag (EMG leads motion)
        tgt_idx = min(start + window_size - 1 + lag_samples, n_samples - 1)
        y_reg_list.append(downsampled[tgt_idx, :])
        start += step
        
    return {
        'windows': np.array(windows),
        'y_reg': np.array(y_reg_list)
    }
