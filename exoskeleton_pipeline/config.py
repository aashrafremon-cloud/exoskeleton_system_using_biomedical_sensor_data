"""Central configuration for the exoskeleton EMG pipeline."""
import os

# Sampling rates (Camargo README)
EMG_FS = 1000.0
IK_FS = 200.0
TARGET_FS = 100.0

# DSP
HP_CUTOFF = 20.0
LP_CUTOFF = 6.0
HP_ORDER = 4
LP_ORDER = 2
WINDOW_MS = 200
WINDOW_SIZE = int(WINDOW_MS / 1000.0 * TARGET_FS)  # 20 samples @ 100 Hz
OVERLAP = 0.5

# EMG channels (Camargo order)
EMG_CHANNELS = [
    "gastrocmed", "tibialisanterior", "soleus", "vastusmedialis",
    "vastuslateralis", "rectusfemoris", "bicepsfemoris",
    "semitendinosus", "gracilis", "gluteusmedius", "rightexternaloblique",
]

TARGET_COL = "knee_angle_r"

# Time-lag search (ms): EMG often leads joint motion
LAG_MS_CANDIDATES = [0, 50, 100, 150, 200, 250, 300]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_regressor_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.pkl")
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

RANDOM_SEED = 42
