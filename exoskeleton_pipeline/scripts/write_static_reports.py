"""Write SYNC, CHANNEL, LEAKAGE, FEATURE, API markdown reports."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
REPORTS = os.path.join(BASE, "reports")

SYNC = """# SYNC_REPORT

## Rates
- EMG: 1000 Hz (raw) → 100 Hz after decimation (factor 10)
- IK: 200 Hz (Camargo) → 100 Hz after decimation (factor 2)
- Duration sync: `min(len_emg/1000, len_ik/200)` then trim both

## Window mapping
1. EMG windows: 20 samples @ 100 Hz = 200 ms, overlap 50% → step 10 samples (100 ms)
2. IK downsampled to 100 Hz, same window grid
3. **Target** for window starting at sample `s`: `knee_angle[s + 19 + lag_samples]` on 100 Hz grid
4. `lag_samples = round(lag_ms / 1000 * 100)` — EMG features lead knee response

## Alignment guarantee
Both EMG features and IK targets use identical window start indices after resampling to 100 Hz.
"""

CHANNEL = """# CHANNEL_AUDIT

## Training & API (correct)
All **11** Camargo EMG channels:
gastrocmed, tibialisanterior, soleus, vastusmedialis, vastuslateralis,
rectusfemoris, bicepsfemoris, semitendinosus, gracilis, gluteusmedius, rightexternaloblique

## example_config_data.csv (NOT used for training)
Only 4 example amplitudes: bicepsfemoris, rectusfemoris, semitendinosus, vastusmedialis.
This file is documentation/demo config — **not** the training feature set.

## Resolution
- Keep 11-channel API input `(N, 11)` raw EMG @ ≥200 samples
- Do NOT reduce to 4 channels without retraining
"""

LEAKAGE = """# LEAKAGE_REPORT

## Risk
Random train/test split assigns windows from the same trial to both sets → optimistic bias.

## Mitigation (implemented in train.py)
- `trial_id` per file pair
- **GroupKFold** (5 folds) for CV — groups = trial_id
- Hold-out: 20% of **trials** (not windows) for test metrics

## Verdict
Leakage controlled when using `train.py` GroupKFold + trial hold-out.
"""

FEATURE = """# FEATURE_REPORT

## Per channel (×11)
MAV, RMS, WL, VAR, ZC, SSC, IEMG (77 features total)

## DSP order
DC removal → HP 20 Hz → rectify → LP 6 Hz envelope → per-trial max normalization → decimate 100 Hz

## Future (not in v1 benchmark)
LOG, AAC, DASDV, MNF, MDF, spectral entropy, Hjorth, wavelets — add in v2 if R² plateaus
"""

API = """# API_REPORT

## Endpoints
- `GET /health` — model loaded status
- `POST /predict` — body: `{"emg_window": [[11 floats], ...]}` min 200 rows

## Channel contract
Must match training: **11 channels** in Camargo order (see config.EMG_CHANNELS).

## Mismatch fixed
API uses same `EMGPipeline` and `feature_names` from PKL bundle — not 4-channel CSV.
"""

for name, body in [
    ("SYNC_REPORT.md", SYNC),
    ("CHANNEL_AUDIT.md", CHANNEL),
    ("LEAKAGE_REPORT.md", LEAKAGE),
    ("FEATURE_REPORT.md", FEATURE),
    ("API_REPORT.md", API),
]:
    with open(os.path.join(REPORTS, name), "w", encoding="utf-8") as f:
        f.write(body)
    print("Wrote", name)
