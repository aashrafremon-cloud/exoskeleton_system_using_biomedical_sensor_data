import scipy.io
import os

try:
    path = r'Data repository for Camargo, et. al. A comprehensive, open-source dataset of lower limb biomechanics. Part 1 of 3\AB06\10_09_18\treadmill\emg\treadmill_01_01.mat'
    print(f"Loading {path}...")
    mat = scipy.io.loadmat(path)
    print("Keys found:", list(mat.keys()))
    
    if 'emg' in mat:
        print("EMG shape:", mat['emg'].shape)
    elif 'data' in mat: # Camargo dataset sometimes stores under 'data' or similar
        print("data shape:", mat['data'].shape)
        
    ik_path = r'Data repository for Camargo, et. al. A comprehensive, open-source dataset of lower limb biomechanics. Part 1 of 3\AB06\10_09_18\treadmill\ik\knee_treadmill_01_01.mat'
    if os.path.exists(ik_path):
        ik_mat = scipy.io.loadmat(ik_path)
        print("IK Keys found:", list(ik_mat.keys()))
except Exception as e:
    print(f"Error: {e}")
