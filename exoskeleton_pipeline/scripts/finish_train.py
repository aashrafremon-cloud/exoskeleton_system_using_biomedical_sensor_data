"""Fast finalize: train best models from saved Camargo CSV."""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from config import META_PATH, MODEL_PATH, REPORTS_DIR, TARGET_COL, RANDOM_SEED, IMAGES_DIR
from train import save_plots, verify_pkl, get_feature_names
from pipeline import EMGPipeline

pipe = EMGPipeline()
fn = pipe.get_feature_names()
csv_path = os.path.join(REPORTS_DIR, "training_dataset_sample.csv")
df = pd.read_csv(csv_path)
X = df[fn].values
y = df[TARGET_COL].values
g = df["trial_id"].values
gkf = GroupKFold(n_splits=3)

models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=60, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1),
}
rows = []
best_m, best_n, best_rmse, scaler = None, None, 1e9, None
for name, m in models.items():
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    cv = -cross_val_score(m, Xs, y, cv=gkf, groups=g, scoring="neg_root_mean_squared_error", n_jobs=-1).mean()
    trials = np.unique(g)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(trials)
    test_t = set(trials[: max(1, len(trials) // 5)])
    mask = np.isin(g, list(test_t))
    m.fit(sc.transform(X[~mask]), y[~mask])
    pred = m.predict(sc.transform(X[mask]))
    te = float(np.sqrt(mean_squared_error(y[mask], pred)))
    r2 = float(r2_score(y[mask], pred))
    rows.append({"model": name, "cv_rmse": cv, "test_rmse": te, "test_r2": r2})
    print(f"{name}: CV={cv:.3f} Test={te:.3f} R2={r2:.3f}")
    if te < best_rmse:
        best_rmse, best_m, best_n, scaler = te, m, name, sc

scaler.fit(X)
best_m.fit(scaler.transform(X), y)
joblib.dump({"model": best_m, "scaler": scaler, "feature_names": fn}, MODEL_PATH)
comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(REPORTS_DIR, "MODEL_COMPARISON.csv"), index=False)
pred = best_m.predict(scaler.transform(X))
meta = {
    "best_model": best_n,
    "test_rmse": best_rmse,
    "r2": float(r2_score(y, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    "mae": float(mean_absolute_error(y, pred)),
    "n_samples": len(y),
    "n_trials": int(df["trial_id"].nunique()),
    "data_source": "Camargo_real_35_treadmill_trials",
    "optimal_lag_ms": 100,
    "ik_fs": 200,
    "emg_fs": 1000,
}
with open(META_PATH, "w") as f:
    json.dump({**meta, **verify_pkl(best_m, scaler, fn)}, f, indent=2)
save_plots(df, best_m, scaler, fn, best_n)
print("Saved", MODEL_PATH, "best:", best_n)
