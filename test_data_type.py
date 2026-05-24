import os
import scipy.io

path = os.path.join(
    "Data repository for Camargo, et. al. A comprehensive, open-source dataset of lower limb biomechanics. Part 1 of 3",
    "AB06", "10_09_18", "treadmill", "emg", "treadmill_01_01.mat"
)

try:
    data = scipy.io.loadmat(path)
    print("Scipy loadmat keys:", list(data.keys()))
except Exception as e:
    print("Scipy loadmat failed:", e)

try:
    import h5py
    with h5py.File(path, 'r') as f:
        print("h5py keys:", list(f.keys()))
except Exception as e:
    print("h5py failed:", e)
