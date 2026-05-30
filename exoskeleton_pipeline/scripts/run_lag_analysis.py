"""Phase 8: time-lag optimization on real Camargo trials (Ridge GroupCV)."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from config import LAG_MS_CANDIDATES, REPORTS_DIR, TARGET_FS, RANDOM_SEED
from camargo_io import find_camargo_root
from train import build_dataset, lag_ms_to_samples

MAX_FILES = 15
MODES = ["treadmill"]


def main():
    root = find_camargo_root()
    rows = []
    best_lag, best_rmse = 100.0, np.inf

    for lag_ms in LAG_MS_CANDIDATES:
        print(f"Lag {lag_ms} ms ...")
        try:
            df = build_dataset(root, max_files=MAX_FILES, modes=MODES, lag_ms=lag_ms)
            feat = [c for c in df.columns if c not in ("knee_angle_r", "trial_id")]
            X = df[feat].values
            y = df["knee_angle_r"].values
            g = df["trial_id"].values
            n_splits = min(5, df["trial_id"].nunique())
            gkf = GroupKFold(n_splits=n_splits)
            rmse = -cross_val_score(
                Ridge(alpha=1.0),
                StandardScaler().fit_transform(X),
                y,
                cv=gkf,
                groups=g,
                scoring="neg_root_mean_squared_error",
            ).mean()
            rows.append({
                "lag_ms": lag_ms,
                "lag_samples": lag_ms_to_samples(lag_ms),
                "cv_rmse": rmse,
                "n_samples": len(df),
                "n_trials": int(df["trial_id"].nunique()),
            })
            print(f"  CV RMSE = {rmse:.3f} ({len(df)} samples)")
            if rmse < best_rmse:
                best_rmse, best_lag = rmse, lag_ms
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"lag_ms": lag_ms, "lag_samples": lag_ms_to_samples(lag_ms),
                         "cv_rmse": np.nan, "n_samples": 0, "n_trials": 0})

    lag_df = pd.DataFrame(rows)
    lag_df.to_csv(os.path.join(REPORTS_DIR, "lag_comparison.csv"), index=False)

    md = [
        "# Time-Lag Analysis (Camargo EMG -> Knee Angle)",
        "",
        f"**Data:** {MAX_FILES} treadmill trials, real `.mat` files under `{root}`",
        f"**Metric:** GroupKFold Ridge CV RMSE (degrees), {n_splits if rows else 5}-fold by `trial_id`",
        f"**Target FS:** {TARGET_FS} Hz — lag converted to samples as `round(lag_ms/1000 * {TARGET_FS})`",
        "",
        f"## Optimal lag: **{best_lag} ms** (CV RMSE = {best_rmse:.3f}°)",
        "",
        "| Lag (ms) | Lag (samples) | CV RMSE (°) | Samples | Trials |",
        "|----------|---------------|-------------|---------|--------|",
    ]
    for _, r in lag_df.iterrows():
        cv = f"{r['cv_rmse']:.3f}" if pd.notna(r["cv_rmse"]) else "—"
        md.append(
            f"| {int(r['lag_ms'])} | {int(r['lag_samples'])} | {cv} | {int(r['n_samples'])} | {int(r['n_trials'])} |"
        )
    md.extend([
        "",
        "## Interpretation",
        "",
        "EMG activation typically precedes visible joint motion (electromechanical delay).",
        "Shifting the knee-angle target forward in time aligns muscle drive with subsequent flexion/extension.",
        "Training dataset (`training_dataset_sample.csv`) was built with **100 ms** lag unless re-exported.",
        "",
    ])
    out = os.path.join(REPORTS_DIR, "LAG_ANALYSIS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("Wrote", out, "best_lag=", best_lag)


if __name__ == "__main__":
    main()
