import os
import glob
import h5py
import numpy as np
import pandas as pd
from pipeline import EMG_Pipeline, process_kinematics

ROOT = "Data repository for Camargo, et. al. A comprehensive, open-source dataset of lower limb biomechanics. Part 1 of 3"

# 11 EMG channels exactly as requested
emg_columns = [
    "gastrocmed", "tibialisanterior", "soleus", "vastusmedialis", 
    "vastuslateralis", "rectusfemoris", "bicepsfemoris", 
    "semitendinosus", "gracilis", "gluteusmedius", "rightexternaloblique"
]

def prepare_dataset():
    pipeline = EMG_Pipeline(original_fs=1000.0, target_fs=100.0, window_size=20, overlap=0.5)
    
    X_ml_all = []
    y_reg_all = []
    X_seq_all = []
    y_seq_all = []
    
    if not os.path.exists(ROOT):
        print(f"Error: Database {ROOT} not found!")
        return

    # Find all emg .mat files recursively
    emg_files = glob.glob(os.path.join(ROOT, "AB*", "*", "*", "emg", "*.mat"))
    print(f"Found {len(emg_files)} EMG files.")

    for emg_file in emg_files:
        try:
            ik_file = emg_file.replace(os.sep + "emg" + os.sep, os.sep + "ik" + os.sep)
            filename = os.path.basename(emg_file)
            ik_filename = filename.replace("emg_", "knee_")
            ik_file = os.path.join(os.path.dirname(ik_file), ik_filename)
            
            if not os.path.exists(ik_file):
                print(f"Warning: Corresponding IK file not found for {emg_file}")
                continue
                
            # 1. Load EMG (fs = 1000 Hz)
            with h5py.File(emg_file, 'r') as f:
                emg = np.array(f['emg']).T   # Shape: (T_emg, 11)
                
            # 2. Load IK (fs = 100 Hz in georgia dataset usually, but we must check)
            with h5py.File(ik_file, 'r') as f:
                knee_r = np.array(f['knee_angle_r']).squeeze().flatten()
                knee_l = np.array(f['knee_angle_l']).squeeze().flatten()
                
            knee = np.stack([knee_r, knee_l], axis=1) # Shape: (T_ik, 2)
            
            # Check for sync. if T_emg == T_ik * 10 
            dur_emg = emg.shape[0] / 1000.0
            dur_ik = knee.shape[0] / 100.0
            min_dur = min(dur_emg, dur_ik)
            
            emg_len = int(min_dur * 1000.0)
            ik_len = int(min_dur * 100.0)
            
            emg = emg[:emg_len, :]
            knee = knee[:ik_len, :]
            
            # Process EMG
            emg_res = pipeline.process_sequence(emg, return_features=True)
            # Process Kinematics
            kin_res = process_kinematics(knee, original_fs=100.0, target_fs=100.0, window_size=20, overlap=0.5)
            
            # Ensure same number of windows
            min_windows = min(emg_res['windows'].shape[0], kin_res['windows'].shape[0])
            
            if min_windows > 0:
                X_ml_all.append(emg_res['features'][:min_windows])
                y_reg_all.append(kin_res['y_reg'][:min_windows])
                
                X_seq_all.append(emg_res['windows'][:min_windows])
                y_seq_all.append(kin_res['windows'][:min_windows])
            
        except Exception as e:
            print(f"Error processing {emg_file}: {e}")
                
    if not X_ml_all:
        print("No valid files found or processed.")
        return
        
    X_ml = np.concatenate(X_ml_all, axis=0)
    y_reg = np.concatenate(y_reg_all, axis=0)
    X_seq = np.concatenate(X_seq_all, axis=0)
    y_seq = np.concatenate(y_seq_all, axis=0)
    
    print(f"Total ML Samples: {X_ml.shape[0]}, Features: {X_ml.shape[1]}")
    print(f"Total Seq Samples: {X_seq.shape[0]}, Window: {X_seq.shape[1]}, Channels: {X_seq.shape[2]}")
    
    overall_amp = np.mean(X_ml, axis=1) 
    bins = [0] + [np.percentile(overall_amp, i) for i in [16.6, 33.3, 50, 66.6, 83.3]] + [np.inf]
    y_class = np.digitize(overall_amp, bins) - 1
    y_class = np.clip(y_class, 0, 5) 
    
    print(f"Generated Label distribution: {np.unique(y_class, return_counts=True)}")

    np.savez_compressed("dataset_ml.npz", X=X_ml, y_reg=y_reg, y_class=y_class)
    print("Saved dataset_ml.npz")
    
    np.savez_compressed("dataset_dl.npz", X_seq=X_seq, y_seq=y_seq, y_reg=y_reg, y_class=y_class)
    print("Saved dataset_dl.npz")

if __name__ == "__main__":
    prepare_dataset()
