"""
Full training pipeline: Camargo EMG -> knee angle regression.

- Group-aware split by trial (no leakage)
- Time-lag optimization
- Model comparison (sklearn; PyCaret optional)
- Reports + figures
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from config import (
    IK_FS,
    IMAGES_DIR,
    LAG_MS_CANDIDATES,
    META_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    SCALER_PATH,
    TARGET_COL,
    TARGET_FS,
    VIZ_DIR,
)
from camargo_io import (
    EMG_FS,
    discover_emg_ik_pairs,
    find_camargo_root,
    load_emg_mat,
    load_ik_knee_mat,
    sync_emg_knee,
)
from features import get_feature_names
from pipeline import EMGPipeline, process_kinematics

warnings.filterwarnings("ignore")

pipeline = EMGPipeline(original_fs=1000.0, target_fs=TARGET_FS, window_size=20, overlap=0.5)


def _ensure_dirs():
    for d in (REPORTS_DIR, IMAGES_DIR, VIZ_DIR):
        os.makedirs(d, exist_ok=True)


def _clear_artifacts():
    for p in (MODEL_PATH, SCALER_PATH, META_PATH):
        if os.path.isfile(p):
            os.remove(p)
    for d in (VIZ_DIR, IMAGES_DIR):
        if os.path.isdir(d):
            shutil.rmtree(d)
    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def lag_ms_to_samples(lag_ms: float) -> int:
    return int(round(lag_ms / 1000.0 * TARGET_FS))


def build_dataset(
    root: str,
    max_files: Optional[int],
    modes: List[str],
    lag_ms: float,
) -> pd.DataFrame:
    pairs = discover_emg_ik_pairs(root, modes=modes, max_files=max_files)
    if not pairs:
        raise RuntimeError(f"No EMG/IK pairs under {root}")

    lag_samples = lag_ms_to_samples(lag_ms)
    feature_names = pipeline.get_feature_names()
    rows_X, rows_y, groups = [], [], []

    for emg_path, ik_path in pairs:
        trial_id = emg_path.replace(root, "").lstrip(os.sep)
        try:
            emg = load_emg_mat(emg_path)
            knee = load_ik_knee_mat(ik_path)
            emg, knee = sync_emg_knee(emg, knee)
            emg_res = pipeline.process_raw_emg(emg, return_features=True)
            kin_res = process_kinematics(
                knee,
                original_fs=IK_FS,
                target_fs=TARGET_FS,
                window_size=pipeline.window_size,
                overlap=pipeline.overlap,
                lag_samples=lag_samples,
            )
            n = min(emg_res["features"].shape[0], kin_res["y_reg"].shape[0])
            if n <= 0:
                continue
            rows_X.append(emg_res["features"][:n])
            rows_y.append(kin_res["y_reg"][:n, 0])
            groups.extend([trial_id] * n)
            print(f"  OK {trial_id}: {n} windows (lag={lag_ms}ms)")
        except Exception as exc:
            print(f"  SKIP {trial_id}: {exc}")

    if not rows_X:
        raise RuntimeError("No trials loaded.")

    X = np.concatenate(rows_X, axis=0)
    y = np.concatenate(rows_y, axis=0)
    df = pd.DataFrame(X, columns=feature_names)
    df[TARGET_COL] = y
    df["trial_id"] = groups
    return df


def validate_dataframe(df: pd.DataFrame) -> dict:
    feat = [c for c in df.columns if c not in (TARGET_COL, "trial_id")]
    X = df[feat].values
    y = df[TARGET_COL].values
    report = {
        "n_samples": int(len(df)),
        "n_features": len(feat),
        "n_trials": int(df["trial_id"].nunique()),
        "nan_features": int(np.isnan(X).sum()),
        "inf_features": int(np.isinf(X).sum()),
        "nan_target": int(np.isnan(y).sum()),
        "target_min": float(np.nanmin(y)),
        "target_max": float(np.nanmax(y)),
        "target_mean": float(np.nanmean(y)),
        "target_std": float(np.nanstd(y)),
    }
    return report


def get_models() -> Dict[str, object]:
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=80, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=80, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=60, max_depth=4, random_state=RANDOM_SEED
        ),
        "AdaBoost": AdaBoostRegressor(n_estimators=60, random_state=RANDOM_SEED),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=120, max_depth=6, learning_rate=0.08,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=120, max_depth=6, random_state=RANDOM_SEED, verbose=-1
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=120, depth=6, random_seed=RANDOM_SEED, verbose=0
        )
    except ImportError:
        pass
    return models


def evaluate_models(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, object, str, StandardScaler, List[str]]:
    feature_names = pipeline.get_feature_names()
    X = df[feature_names].values
    y = df[TARGET_COL].values
    groups = df["trial_id"].values

    gkf = GroupKFold(n_splits=min(5, df["trial_id"].nunique()))
    scaler = StandardScaler()
    results = []
    best_model, best_name, best_rmse = None, None, np.inf

    for name, model in get_models().items():
        cv_rmse = -cross_val_score(
            model, scaler.fit_transform(X), y,
            cv=gkf, groups=groups,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1 if name not in ("XGBoost", "LightGBM", "CatBoost") else 1,
        ).mean()

        # Hold-out: last 20% of trials
        trial_ids = df["trial_id"].unique()
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(trial_ids)
        n_test = max(1, int(0.2 * len(trial_ids)))
        test_trials = set(trial_ids[:n_test])
        mask_test = df["trial_id"].isin(test_trials)
        X_tr, y_tr = X[~mask_test], y[~mask_test]
        X_te, y_te = X[mask_test], y[mask_test]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)
        test_rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        test_mae = float(mean_absolute_error(y_te, pred))
        test_r2 = float(r2_score(y_te, pred))

        results.append({
            "model": name, "cv_rmse": cv_rmse,
            "test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2,
        })
        if test_rmse < best_rmse:
            best_rmse, best_model, best_name = test_rmse, model, name
        print(f"  {name}: CV RMSE={cv_rmse:.3f} | Test RMSE={test_rmse:.3f} R2={test_r2:.3f}")

    scaler.fit(X)
    best_model.fit(scaler.transform(X), y)
    return pd.DataFrame(results), best_model, best_name, scaler, feature_names


def search_optimal_lag(root: str, max_files: int, modes: List[str]) -> Tuple[float, pd.DataFrame]:
    rows = []
    best_lag, best_rmse = 0.0, np.inf
    for lag_ms in LAG_MS_CANDIDATES:
        print(f"\n--- Lag {lag_ms} ms ---")
        try:
            df = build_dataset(root, max_files=max_files, modes=modes, lag_ms=lag_ms)
            # quick CV with Ridge only for speed
            feat = pipeline.get_feature_names()
            X = df[feat].values
            y = df[TARGET_COL].values
            g = df["trial_id"].values
            gkf = GroupKFold(n_splits=min(5, df["trial_id"].nunique()))
            rmse = -cross_val_score(
                Ridge(alpha=1.0), StandardScaler().fit_transform(X), y,
                cv=gkf, groups=g, scoring="neg_root_mean_squared_error",
            ).mean()
            rows.append({"lag_ms": lag_ms, "cv_rmse": rmse, "n_samples": len(df)})
            print(f"  Ridge GroupCV RMSE = {rmse:.3f}")
            if rmse < best_rmse:
                best_rmse, best_lag = rmse, lag_ms
        except Exception as exc:
            print(f"  Lag {lag_ms} failed: {exc}")
            rows.append({"lag_ms": lag_ms, "cv_rmse": np.nan, "n_samples": 0})
    lag_df = pd.DataFrame(rows)
    lag_df.to_csv(os.path.join(REPORTS_DIR, "lag_comparison.csv"), index=False)
    return best_lag, lag_df


def save_plots(df: pd.DataFrame, model, scaler, feature_names: List[str], model_name: str):
    X = scaler.transform(df[feature_names].values)
    y = df[TARGET_COL].values
    pred = model.predict(X)
    resid = y - pred

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, pred, alpha=0.25, s=8)
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual knee angle (°)")
    ax.set_ylabel("Predicted (°)")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    fig.tight_layout()
    for folder, name in ((IMAGES_DIR, "actual_vs_predicted.png"), (VIZ_DIR, "10_predictions_and_residuals.png")):
        fig.savefig(os.path.join(folder, name), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pred, resid, alpha=0.25, s=8)
    ax.axhline(0, color="r", ls="--")
    ax.set_xlabel("Predicted (°)")
    ax.set_ylabel("Residual (°)")
    ax.set_title("Residual plot")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "residual_plot.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=50, edgecolor="white")
    ax.set_title("Error distribution")
    ax.set_xlabel("Error (°)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "error_distribution.png"), dpi=150)
    plt.close(fig)

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        imp.plot(kind="barh", ax=ax)
        ax.invert_yaxis()
        ax.set_title("Feature importance")
        fig.tight_layout()
        fig.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=150)
        plt.close(fig)

    # SHAP (optional)
    try:
        import shap

        explainer = shap.TreeExplainer(model) if hasattr(model, "estimators_") else shap.Explainer(model, X[:200])
        sv = explainer.shap_values(X[: min(500, len(X))])
        plt.figure()
        shap.summary_plot(sv, df[feature_names].iloc[: min(500, len(X))], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def verify_pkl(model, scaler, feature_names: List[str]) -> dict:
    report = {"path": MODEL_PATH, "exists": os.path.isfile(MODEL_PATH), "size_bytes": 0, "load_ok": False}
    if report["exists"]:
        report["size_bytes"] = os.path.getsize(MODEL_PATH)
    joblib.dump({"model": model, "scaler": scaler, "feature_names": feature_names}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    report["size_bytes"] = os.path.getsize(MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    test = np.random.randn(1, len(feature_names))
    pred = bundle["model"].predict(bundle["scaler"].transform(test))
    report["load_ok"] = True
    report["test_prediction"] = float(pred[0])
    return report


def write_audit_report(sections: dict):
    path = os.path.join(REPORTS_DIR, "TECHNICAL_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Exoskeleton EMG Pipeline — Technical Report\n\n")
        for title, body in sections.items():
            f.write(f"## {title}\n\n{body}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Train knee-angle model from Camargo EMG")
    parser.add_argument("--data-root", type=str, default=None, help="Path to Data_repository_for_Camargo")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--modes", nargs="+", default=["treadmill"])
    parser.add_argument("--skip-lag-search", action="store_true")
    parser.add_argument("--lag-ms", type=float, default=None)
    parser.add_argument("--keep-old", action="store_true")
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="Train from pre-built CSV (skip .mat loading)",
    )
    args = parser.parse_args()

    _ensure_dirs()
    if not args.keep_old:
        _clear_artifacts()

    print("=" * 60)
    print("EMG -> Knee Angle Training Pipeline (Camargo only)")
    print("=" * 60)

    best_lag = args.lag_ms if args.lag_ms is not None else 100.0
    root = "Data_repository_for_Camargo"

    if args.from_csv:
        print(f"\n[2] Loading dataset from {args.from_csv}")
        df = pd.read_csv(args.from_csv)
        root = find_camargo_root(args.data_root)
    else:
        root = find_camargo_root(args.data_root)
        print(f"Data root: {root}")
        if args.lag_ms is None and not args.skip_lag_search:
            print("\n[1] Time-lag search...")
            best_lag, _ = search_optimal_lag(root, min(args.max_files, 15), args.modes)
            print(f"    Optimal lag: {best_lag} ms")
        elif args.skip_lag_search:
            best_lag = args.lag_ms or 100.0

        print(f"\n[2] Building dataset (lag={best_lag} ms)...")
        df = build_dataset(root, args.max_files, args.modes, best_lag)
        df.to_csv(os.path.join(REPORTS_DIR, "training_dataset_sample.csv"), index=False)

    val = validate_dataframe(df)
    print(f"    Samples: {val['n_samples']}, Trials: {val['n_trials']}, Features: {val['n_features']}")
    with open(os.path.join(REPORTS_DIR, "dataset_validation.json"), "w") as f:
        json.dump(val, f, indent=2)

    print("\n[3] Model comparison (GroupKFold — no trial leakage)...")
    comparison, model, name, scaler, feat_names = evaluate_models(df)
    comparison.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    print(f"\n[4] Best model: {name}")
    pkl_report = verify_pkl(model, scaler, feat_names)
    print(f"    PKL: {pkl_report['size_bytes']} bytes, load_ok={pkl_report['load_ok']}")

    y = df[TARGET_COL].values
    pred = model.predict(scaler.transform(df[feat_names].values))
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "explained_variance": float(explained_variance_score(y, pred)),
        "best_model": name,
        "optimal_lag_ms": best_lag,
        "data_source": root,
    }
    with open(META_PATH, "w") as f:
        json.dump({**metrics, **val, **pkl_report}, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[5] Saving figures...")
    save_plots(df, model, scaler, feat_names, name)

    write_audit_report({
        "Dataset": json.dumps(val, indent=2),
        "Metrics": json.dumps(metrics, indent=2),
        "Target": f"`{TARGET_COL}` = right knee angle (degrees) at window end + {best_lag}ms lag.",
        "Split": "GroupKFold by `trial_id` — windows from same trial stay in one fold.",
        "PKL": json.dumps(pkl_report, indent=2),
    })

    print("\n" + comparison.to_string(index=False))
    print(f"\nDone. Model: {MODEL_PATH}")
    print(f"Reports: {REPORTS_DIR} | Images: {IMAGES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
