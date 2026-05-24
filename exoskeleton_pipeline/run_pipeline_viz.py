"""
Train knee-angle regressor on Camargo EMG+IK data with step-by-step visualizations.
"""
import argparse
import json
import os
import shutil
import warnings
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from camargo_io import (
    EMG_FS,
    IK_FS,
    describe_dataset,
    discover_emg_ik_pairs,
    find_camargo_root,
    load_emg_mat,
    load_ik_knee_mat,
    sync_emg_knee,
)
from pipeline import EMGPipeline, process_kinematics

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
MODEL_PATH = os.path.join(BASE_DIR, "best_regressor_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.pkl")
META_PATH = os.path.join(BASE_DIR, "model_metadata.json")

pipeline = EMGPipeline(original_fs=1000.0, target_fs=100.0, window_size=20, overlap=0.5)


def clear_previous_outputs():
    """Remove prior model artifacts and visualizations."""
    for path in (MODEL_PATH, SCALER_PATH, META_PATH):
        if os.path.isfile(path):
            os.remove(path)
    if os.path.isdir(VIZ_DIR):
        shutil.rmtree(VIZ_DIR)
    os.makedirs(VIZ_DIR, exist_ok=True)


def run_dsp_with_intermediates(raw_emg: np.ndarray) -> dict:
    hp = pipeline.apply_highpass(raw_emg, cutoff=20.0)
    rect = pipeline.rectify(hp)
    envelope = pipeline.extract_envelope(rect, cutoff=6.0)
    downsampled = pipeline.decimate_signal(envelope)
    result = pipeline.process_raw_emg(raw_emg, return_features=True)
    return {
        "raw": raw_emg,
        "highpass": hp,
        "rectified": rect,
        "envelope": envelope,
        "downsampled": downsampled,
        "windows": result["windows"],
        "features": result["features"],
    }


def plot_pipeline_steps(dsp: dict, channel_idx: int = 3, ch_name: str = "vastusmedialis"):
    fs_raw = pipeline.original_fs
    fs_ds = pipeline.target_fs
    steps = [
        ("01_raw_emg", dsp["raw"], fs_raw, "Raw EMG (Camargo)"),
        ("02_highpass", dsp["highpass"], fs_raw, "High-pass (>20 Hz)"),
        ("03_rectified", dsp["rectified"], fs_raw, "Rectified |EMG|"),
        ("04_envelope", dsp["envelope"], fs_raw, "Low-pass envelope (<6 Hz)"),
        ("05_downsampled", dsp["downsampled"], fs_ds, f"Downsampled ({fs_ds:.0f} Hz)"),
    ]
    n_show_raw = min(3000, dsp["raw"].shape[0])
    n_show_ds = min(300, dsp["downsampled"].shape[0])
    for fname, data, fs, title in steps:
        n_show = n_show_raw if fs == fs_raw else n_show_ds
        t = np.arange(n_show) / fs
        y = data[:n_show, channel_idx]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, y, color="#2563eb", linewidth=0.8)
        ax.set_title(f"{title} — {ch_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(VIZ_DIR, f"{fname}_{ch_name}.png"), dpi=120)
        plt.close(fig)
    if len(dsp["windows"]) > 0:
        w0 = dsp["windows"][0][:, channel_idx]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(w0, marker="o", markersize=3, color="#059669")
        ax.set_title(f"Windowing — first window ({pipeline.window_size} samples)")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Envelope")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(VIZ_DIR, f"06_window_example_{ch_name}.png"), dpi=120)
        plt.close(fig)


def plot_dataset_overview(summary: pd.DataFrame, trial_log: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pivot = summary.pivot_table(index="subject", columns="mode", values="emg_trials", fill_value=0)
    im = axes[0].imshow(pivot.values, aspect="auto", cmap="Blues")
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns, rotation=30, ha="right")
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_title("Camargo dataset — EMG trials per subject/mode")
    plt.colorbar(im, ax=axes[0], label="# trials")
    axes[1].bar(range(len(trial_log)), trial_log["n_windows"], color="#6366f1")
    axes[1].set_title("Windows extracted per processed trial")
    axes[1].set_xlabel("Trial index")
    axes[1].set_ylabel("# windows")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "00_dataset_overview.png"), dpi=120)
    plt.close(fig)


def plot_features_overview(features: np.ndarray):
    n_ch = len(pipeline.channels)
    mav = features[:, :n_ch]
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mav[: min(200, len(mav))].T, aspect="auto", cmap="viridis")
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(pipeline.channels, fontsize=7)
    ax.set_xlabel("Window index")
    ax.set_title("Feature extraction — MAV (Camargo, first 200 windows)")
    plt.colorbar(im, ax=ax, label="MAV")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "07_features_mav_heatmap.png"), dpi=120)
    plt.close(fig)


def plot_eda(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["knee_angle_r"], bins=40, color="#7c3aed", edgecolor="white")
    axes[0].set_title("Target: knee_angle_r (Camargo IK)")
    axes[0].set_xlabel("Degrees")
    corr = df.drop(columns=["knee_angle_r"]).corrwith(df["knee_angle_r"]).sort_values()
    top = pd.concat([corr.head(5), corr.tail(5)])
    axes[1].barh(top.index.str.replace("_", " "), top.values, color="#dc2626")
    axes[1].set_title("Top feature correlations")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "08_eda_target_and_correlations.png"), dpi=120)
    plt.close(fig)


def build_ml_dataset_from_camargo(
    root: str,
    max_files: Optional[int] = 20,
    modes: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load trials, return (df, trial_log, demo_emg, demo_knee)."""
    pairs = discover_emg_ik_pairs(root, modes=modes, max_files=max_files)
    if not pairs:
        raise RuntimeError(f"No EMG/IK pairs under {root}")

    feature_names = pipeline.get_feature_names()
    chunks_X, chunks_y = [], []
    log_rows = []
    demo_emg, demo_knee = None, None

    for i, (emg_path, ik_path) in enumerate(pairs):
        rel = emg_path.replace(root, "").lstrip(os.sep)
        try:
            emg = load_emg_mat(emg_path)
            knee = load_ik_knee_mat(ik_path)
            emg, knee = sync_emg_knee(emg, knee)

            emg_res = pipeline.process_raw_emg(emg, return_features=True)
            kin_res = process_kinematics(
                knee,
                original_fs=IK_FS,
                target_fs=pipeline.target_fs,
                window_size=pipeline.window_size,
                overlap=pipeline.overlap,
            )
            n = min(emg_res["features"].shape[0], kin_res["y_reg"].shape[0])
            if n <= 0:
                continue

            chunks_X.append(emg_res["features"][:n])
            chunks_y.append(kin_res["y_reg"][:n, 0])
            log_rows.append({"trial": rel, "n_windows": n, "emg_samples": emg.shape[0]})

            if demo_emg is None:
                demo_emg, demo_knee = emg, knee
            print(f"  [{i+1}/{len(pairs)}] {rel} -> {n} windows")
        except Exception as exc:
            print(f"  SKIP {rel}: {exc}")

    if not chunks_X:
        raise RuntimeError("No trials processed successfully.")

    X = np.concatenate(chunks_X, axis=0)
    y = np.concatenate(chunks_y, axis=0)
    df = pd.DataFrame(X, columns=feature_names)
    df["knee_angle_r"] = y
    return df, pd.DataFrame(log_rows), demo_emg, demo_knee


def train_and_select_best(df: pd.DataFrame, meta_extra: dict):
    feature_names = pipeline.get_feature_names()
    X = df[feature_names].values
    y = df["knee_angle_r"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }
    results = []
    best_model, best_name, best_rmse = None, None, np.inf

    for name, model in candidates.items():
        scores = cross_val_score(model, X_train_s, y_train, cv=3, scoring="neg_root_mean_squared_error")
        cv_rmse = -scores.mean()
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        test_mae = float(mean_absolute_error(y_test, pred))
        test_r2 = float(r2_score(y_test, pred))
        results.append({"model": name, "cv_rmse": cv_rmse, "test_rmse": test_rmse, "test_mae": test_mae, "test_r2": test_r2})
        if test_rmse < best_rmse:
            best_rmse, best_model, best_name = test_rmse, model, name

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(VIZ_DIR, "model_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(results_df))
    w = 0.35
    ax.bar(x - w / 2, results_df["cv_rmse"], w, label="CV RMSE", color="#3b82f6")
    ax.bar(x + w / 2, results_df["test_rmse"], w, label="Test RMSE", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"])
    ax.set_ylabel("RMSE (degrees)")
    ax.set_title("Model comparison — Camargo data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "09_model_comparison.png"), dpi=120)
    plt.close(fig)

    best_model.fit(scaler.transform(X), y)
    pred_all = best_model.predict(scaler.transform(X))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y, pred_all, alpha=0.2, s=6, c="#2563eb")
    lims = [min(y.min(), pred_all.min()), max(y.max(), pred_all.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("Actual knee angle (°)")
    axes[0].set_ylabel("Predicted (°)")
    axes[0].set_title(f"Predictions — {best_name}")
    axes[1].hist(y - pred_all, bins=40, color="#10b981", edgecolor="white")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Error (°)")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "10_predictions_and_residuals.png"), dpi=120)
    plt.close(fig)

    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot(kind="barh", ax=ax, color="#6366f1")
        ax.set_title(f"Top 15 features — {best_name}")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(VIZ_DIR, "11_feature_importance.png"), dpi=120)
        plt.close(fig)

    joblib.dump({"model": best_model, "scaler": scaler, "feature_names": feature_names}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    meta = {
        "best_model": best_name,
        "test_rmse": float(results_df.loc[results_df["model"] == best_name, "test_rmse"].iloc[0]),
        "test_r2": float(results_df.loc[results_df["model"] == best_name, "test_r2"].iloc[0]),
        "n_samples": int(len(df)),
        "data_source": "Data_repository_for_Camargo",
        **meta_extra,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return best_name, results_df


def main():
    parser = argparse.ArgumentParser(description="Camargo EMG pipeline training + viz")
    parser.add_argument("--max-files", type=int, default=15, help="Max EMG/IK trial pairs to process")
    parser.add_argument("--modes", nargs="+", default=["treadmill"], help="Ambulation modes")
    parser.add_argument("--keep-old", action="store_true", help="Do not delete previous outputs")
    args = parser.parse_args()

    print("=" * 60)
    print("Exoskeleton Pipeline — Camargo Real Data")
    print("=" * 60)

    if not args.keep_old:
        print("\n[0] Clearing previous model & visualizations...")
        clear_previous_outputs()
    else:
        os.makedirs(VIZ_DIR, exist_ok=True)

    root = find_camargo_root()
    summary = describe_dataset(root)
    summary.to_csv(os.path.join(VIZ_DIR, "dataset_summary.csv"), index=False)
    print(f"\n[1] Dataset root: {root}")
    print(summary.to_string(index=False))

    print(f"\n[2] Loading & processing up to {args.max_files} trials ({args.modes})...")
    df, trial_log, demo_emg, demo_knee = build_ml_dataset_from_camargo(
        root, max_files=args.max_files, modes=args.modes
    )
    trial_log.to_csv(os.path.join(VIZ_DIR, "trials_processed.csv"), index=False)
    plot_dataset_overview(summary, trial_log)
    print(f"    ML dataset: {df.shape[0]} samples, {df.shape[1]-1} features")

    print("\n[3] DSP visualization on first successful trial...")
    dsp = run_dsp_with_intermediates(demo_emg)
    plot_pipeline_steps(dsp, channel_idx=3, ch_name="vastusmedialis")
    plot_features_overview(dsp["features"])
    plot_eda(df)

    print("\n[4] Training regressors...")
    best_name, comparison = train_and_select_best(
        df,
        meta_extra={
            "trials_processed": int(len(trial_log)),
            "modes": args.modes,
            "emg_fs": EMG_FS,
            "ik_fs": IK_FS,
        },
    )
    print(comparison.to_string(index=False))
    print(f"\n[5] Best model: {best_name}")
    print(f"    Model  -> {MODEL_PATH}")
    print(f"    Plots  -> {VIZ_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
