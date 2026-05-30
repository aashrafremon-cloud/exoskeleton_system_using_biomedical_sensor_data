# Exoskeleton EMG Pipeline — Technical Report

## Dataset

{
  "n_samples": 50190,
  "n_features": 77,
  "n_trials": 35,
  "nan_features": 0,
  "inf_features": 0,
  "nan_target": 0,
  "target_min": -36.75998409892042,
  "target_max": 38.63088602088261,
  "target_mean": 7.61991220914131,
  "target_std": 16.527850469205653
}

## Metrics

{
  "rmse": 2.565907432314965,
  "mae": 1.9110527908827828,
  "r2": 0.9758982143712651,
  "explained_variance": 0.9758982143712651,
  "best_model": "ExtraTrees",
  "optimal_lag_ms": 100.0,
  "data_source": "D:\\Downloads\\exoskeleton_system_using_biomedical_sensor_data\\Data_repository_for_Camargo"
}

## Target

`knee_angle_r` = right knee angle (degrees) at window end + 100.0ms lag.

## Split

GroupKFold by `trial_id` — windows from same trial stay in one fold.

## PKL

{
  "path": "D:\\Downloads\\exoskeleton_system_using_biomedical_sensor_data\\exoskeleton_pipeline\\best_regressor_model.pkl",
  "exists": true,
  "size_bytes": 102943279,
  "load_ok": true,
  "test_prediction": 6.925595496079174
}

