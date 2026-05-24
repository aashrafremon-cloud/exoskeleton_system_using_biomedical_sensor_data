import numpy as np
import pandas as pd
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

class EMG_Pipeline:
    def __init__(self, original_fs=1000.0, target_fs=100.0, window_size=20, overlap=0.5):
        """
        EMG processing pipeline for real-time exoskeleton control.
        """
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.window_size = window_size  # in target_fs samples (e.g. 20 samples = 200 ms)
        self.overlap = overlap
        
    def _apply_highpass(self, data: np.ndarray, cutoff: float = 20.0, order: int = 4):
        nyq = 0.5 * self.original_fs
        norm_cutoff = cutoff / nyq
        b, a = signal.butter(order, norm_cutoff, btype="highpass")
        return signal.filtfilt(b, a, data, axis=0)

    def _rectify(self, data: np.ndarray):
        return np.abs(data)

    def _extract_envelope(self, data: np.ndarray, cutoff: float = 6.0, order: int = 2):
        nyq = 0.5 * self.original_fs
        norm_cutoff = cutoff / nyq
        b, a = signal.butter(order, norm_cutoff, btype="lowpass")
        return signal.filtfilt(b, a, data, axis=0)
        
    def extract_time_features(self, window_data: np.ndarray):
        """
        Extract fast time-domain features for ML Classification/Regression
        window_data: shape (window_size, num_channels)
        """
        # MAV (Mean Absolute Value)
        mav = np.mean(np.abs(window_data), axis=0)
        # RMS (Root Mean Square)
        rms = np.sqrt(np.mean(window_data**2, axis=0))
        # WL (Waveform Length)
        wl = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
        
        return np.concatenate([mav, rms, wl])
        
    def process_sequence(self, raw_emg: np.ndarray, return_features=True):
        """
        Full Pipeline: Highpass -> Rectify -> Lowpass 6Hz -> Downsample -> Window -> Features
        raw_emg: shape (N, channels)
        """
        # 1. Highpass Filter
        hp = self._apply_highpass(raw_emg, cutoff=20.0)
        
        # 2. Rectify
        rect = self._rectify(hp)
        
        # 3. Envelope Extraction (True lowpass)
        envelope = self._extract_envelope(rect, cutoff=6.0)
        
        # 4. Downsampling (1000 Hz to target_fs)
        factor = int(round(self.original_fs / self.target_fs))
        if factor > 1:
            downsampled = signal.decimate(envelope, factor, axis=0, zero_phase=True)
        else:
            downsampled = envelope
            
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
            'windows': np.array(windows),           # Shape: (num_windows, window_size, channels) 
            'features': np.array(features_list) if return_features else None # Shape: (num_windows, channels*3)
        }

def process_kinematics(raw_kin, original_fs=100.0, target_fs=100.0, window_size=20, overlap=0.5):
    """
    Process kinematic data (Knee Angle) to align with EMG windows.
    Since Kinematics are typically collected at 100Hz in this dataset, no decimation is needed if target_fs=100Hz.
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
        # For regression target, we predict the state at the END of the window
        y_reg_list.append(segment[-1, :])
        start += step
        
    return {
        'windows': np.array(windows),               # (num_windows, window_size, 2)
        'y_reg': np.array(y_reg_list)               # (num_windows, 2)
    }
