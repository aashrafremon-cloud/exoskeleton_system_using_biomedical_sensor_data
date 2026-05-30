# Time-Lag Analysis (Camargo EMG -> Knee Angle)

**Data:** 15 treadmill trials, real `.mat` files under `D:\Downloads\exoskeleton_system_using_biomedical_sensor_data\Data_repository_for_Camargo`
**Metric:** GroupKFold Ridge CV RMSE (degrees), 5-fold by `trial_id`
**Target FS:** 100.0 Hz — lag converted to samples as `round(lag_ms/1000 * 100.0)`

## Optimal lag: **50 ms** (CV RMSE = 5.944°)

| Lag (ms) | Lag (samples) | CV RMSE (°) | Samples | Trials |
|----------|---------------|-------------|---------|--------|
| 0 | 0 | 6.034 | 21535 | 15 |
| 50 | 5 | 5.944 | 21535 | 15 |
| 100 | 10 | 6.212 | 21535 | 15 |
| 150 | 15 | 6.666 | 21535 | 15 |
| 200 | 20 | 6.861 | 21535 | 15 |
| 250 | 25 | 6.610 | 21535 | 15 |
| 300 | 30 | 6.266 | 21535 | 15 |

## Interpretation

EMG activation typically precedes visible joint motion (electromechanical delay).
Shifting the knee-angle target forward in time aligns muscle drive with subsequent flexion/extension.
Training dataset (`training_dataset_sample.csv`) was built with **100 ms** lag unless re-exported.
