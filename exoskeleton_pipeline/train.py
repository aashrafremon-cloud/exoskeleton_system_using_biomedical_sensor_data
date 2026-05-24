import os
import glob
import h5py
import numpy as np
import pandas as pd
from pipeline import EMGPipeline, process_kinematics

# Initialize the pipeline
pipeline = EMGPipeline(original_fs=1000.0, target_fs=100.0, window_size=20, overlap=0.5)

def generate_synthetic_data(num_samples: int = 1500) -> pd.DataFrame:
    """
    Generates highly realistic simulated EMG and Knee Angle data for training.
    Simulates walking gait cycles at 100 Hz.
    """
    print("\n--- Generating High-Quality Synthetic Walking EMG & Kinematics Data ---")
    
    # 1. Generate time vector at 100 Hz
    t = np.linspace(0, num_samples / 100.0, num_samples)
    
    # 2. Simulate standard knee angle joint kinetics (gait cycle)
    # A typical knee angle has a primary flexion during swing phase (~60 degrees) and stance phase flexion (~15 degrees)
    gait_cycle_freq = 0.8  # ~1.2 seconds per stride
    knee_angle_r = 15 + 25 * np.sin(2 * np.pi * gait_cycle_freq * t) + 15 * np.sin(4 * np.pi * gait_cycle_freq * t)
    # Add minor noise
    knee_angle_r += np.random.normal(0, 0.5, size=num_samples)
    
    # 3. Simulate 11 channels of EMG features (features extracted at 100 Hz)
    # In real walking, muscles activate at different phases of the gait cycle
    features_list = []
    
    for i in range(num_samples):
        # We need (11 channels * 3 features) = 33 features
        # Generate phase-shifted activations based on the knee angle (simulating physical movement)
        phase = 2 * np.pi * gait_cycle_freq * t[i]
        
        # Phase shifts for different muscle groups
        quad_act = max(0.1, np.sin(phase) + np.random.normal(0, 0.05))       # Quadriceps (VM, VL, RF)
        ham_act = max(0.05, np.sin(phase + np.pi/3) + np.random.normal(0, 0.05)) # Hamstrings (BF, ST)
        calf_act = max(0.05, np.sin(phase - np.pi/3) + np.random.normal(0, 0.05)) # Gastrocnemius (gastrocmed, soleus)
        other_act = max(0.02, 0.2 + np.random.normal(0, 0.03))                 # Other stabilizer muscles
        
        mav_vector = []
        for ch in pipeline.channels:
            if ch in ["vastusmedialis", "vastuslateralis", "rectusfemoris"]:
                val = quad_act
            elif ch in ["bicepsfemoris", "semitendinosus"]:
                val = ham_act
            elif ch in ["gastrocmed", "soleus"]:
                val = calf_act
            else:
                val = other_act
            mav_vector.append(val)
            
        # RMS is proportional to MAV with some scaling
        rms_vector = [m * 1.2 + np.random.normal(0, 0.02) for m in mav_vector]
        # WL (waveform length) represents changes
        wl_vector = [m * 0.5 + np.random.normal(0, 0.01) for m in mav_vector]
        
        sample_features = np.concatenate([mav_vector, rms_vector, wl_vector])
        features_list.append(sample_features)
        
    # Build dataframe
    feature_names = pipeline.get_feature_names()
    df = pd.DataFrame(features_list, columns=feature_names)
    df['knee_angle_r'] = knee_angle_r
    
    print(f"Generated {df.shape[0]} simulated samples with {df.shape[1] - 1} features.")
    return df

def load_real_data() -> pd.DataFrame:
    """
    Attempts to recursively scan the workspace for Camargo dataset .mat files,
    processes them through the DSP pipeline, and aggregates the features.
    """
    ROOT = "../Data_repository_for_Camargo"
    if not os.path.exists(ROOT):
        # Also try workspace root
        ROOT = "Data_repository_for_Camargo"
        if not os.path.exists(ROOT):
            return None
            
    print(f"\n--- Scanning for Real Camargo Dataset Files in '{ROOT}' ---")
    emg_files = glob.glob(os.path.join(ROOT, "AB*", "*", "*", "emg", "*.mat"))
    print(f"Found {len(emg_files)} EMG files.")
    
    if not emg_files:
        return None
        
    X_all = []
    y_all = []
    
    # Process up to 5 files to prevent training from running too long
    files_to_process = emg_files[:5]
    
    for emg_file in files_to_process:
        try:
            # Map EMG file to corresponding Kinematic IK file
            ik_file = emg_file.replace(os.sep + "emg" + os.sep, os.sep + "ik" + os.sep)
            filename = os.path.basename(emg_file)
            ik_filename = filename.replace("emg_", "knee_")
            ik_file = os.path.join(os.path.dirname(ik_file), ik_filename)
            
            if not os.path.exists(ik_file):
                continue
                
            print(f"Processing: {filename}...")
            
            # 1. Load EMG (1000 Hz)
            with h5py.File(emg_file, 'r') as f:
                emg = np.array(f['emg']).T   # Shape: (T_emg, 11)
                
            # 2. Load IK (100 Hz)
            with h5py.File(ik_file, 'r') as f:
                knee_r = np.array(f['knee_angle_r']).squeeze().flatten()
                
            knee = knee_r.reshape(-1, 1) # Shape: (T_ik, 1)
            
            # Sync lengths
            dur_emg = emg.shape[0] / 1000.0
            dur_ik = knee.shape[0] / 100.0
            min_dur = min(dur_emg, dur_ik)
            
            emg_len = int(min_dur * 1000.0)
            ik_len = int(min_dur * 100.0)
            
            emg = emg[:emg_len, :]
            knee = knee[:ik_len, :]
            
            # Process EMG through pipeline
            emg_res = pipeline.process_raw_emg(emg, return_features=True)
            # Process Kinematics
            kin_res = process_kinematics(knee, original_fs=100.0, target_fs=100.0, window_size=20, overlap=0.5)
            
            min_windows = min(emg_res['features'].shape[0], kin_res['y_reg'].shape[0])
            
            if min_windows > 0:
                X_all.append(emg_res['features'][:min_windows])
                y_all.append(kin_res['y_reg'][:min_windows])
                
        except Exception as e:
            print(f"Error reading file {emg_file}: {e}")
            continue
            
    if not X_all:
        return None
        
    X_ml = np.concatenate(X_all, axis=0)
    y_reg = np.concatenate(y_all, axis=0).flatten()
    
    feature_names = pipeline.get_feature_names()
    df = pd.DataFrame(X_ml, columns=feature_names)
    df['knee_angle_r'] = y_reg
    
    print(f"Processed real dataset: {df.shape[0]} samples, {df.shape[1] - 1} features.")
    return df

def main():
    # 1. Load Data
    data = load_real_data()
    if data is None:
        print("Real data directory not found, empty, or unreadable. Falling back to synthetic simulation.")
        data = generate_synthetic_data()
        
    # 2. PyCaret Setup
    print("\n--- Initializing PyCaret Regression Experiment ---")
    from pycaret.regression import setup, compare_models, tune_model, finalize_model, save_model
    
    # We setup PyCaret experiment targeting 'knee_angle_r' prediction
    reg_setup = setup(
        data=data,
        target='knee_angle_r',
        session_id=42,
        normalize=True,           # Z-score normalize features
        transformation=True,      # Power transform features to normalize distributions
        remove_multicollinearity=True, # Remove collinear features (e.g. redundant features across channels)
        multicollinearity_threshold=0.95,
        html=False,               # Disable HTML since running in script
        verbose=True
    )
    
    # 3. Model Comparison
    print("\n--- Comparing Regressors (Fast Run: 3-fold CV) ---")
    # Exclude complex/slow models like SVM or MLP for speed
    best_model = compare_models(
        fold=3,
        exclude=['lar', 'par', 'omp', 'svm', 'mlp'],
        sort='RMSE'
    )
    print(f"Best model selected: {best_model}")
    
    # 4. Model Tuning (optional but good practice)
    print("\n--- Tuning the Best Regressor Model ---")
    tuned_model = tune_model(best_model, fold=3, n_iter=10, choose_better=True)
    
    # 5. Finalize Model (train on entire dataset)
    print("\n--- Finalizing the Model ---")
    final_model = finalize_model(tuned_model)
    
    # 6. Save Model
    model_save_path = 'best_regressor_model'
    save_model(final_model, model_save_path)
    print(f"\nSuccessfully saved the best regressor to '{model_save_path}.pkl'!")

if __name__ == '__main__':
    main()
