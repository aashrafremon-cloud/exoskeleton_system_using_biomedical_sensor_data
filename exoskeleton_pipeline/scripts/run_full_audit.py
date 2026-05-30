"""
Generate Phase 1-2 audit reports (discovery only, no training).
Run from exoskeleton_pipeline: python scripts/run_full_audit.py
"""
import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from camargo_io import (
    EMG_FS,
    IK_FS,
    EMG_CHANNELS,
    find_camargo_root,
    discover_emg_ik_pairs,
    load_emg_mat,
    load_ik_knee_mat,
    sync_emg_knee,
    describe_dataset,
)
from config import REPORTS_DIR, TARGET_FS

os.makedirs(REPORTS_DIR, exist_ok=True)


def write(name: str, content: str):
    path = os.path.join(REPORTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {path}")


def phase1_repo_audit():
    lines = [
        "# REPOSITORY_AUDIT",
        f"\nGenerated: {datetime.utcnow().isoformat()}Z\n",
        "## Python modules (exoskeleton_pipeline)\n",
        "| File | Purpose |",
        "|------|---------|",
        "| `config.py` | Sampling rates, paths, constants |",
        "| `pipeline.py` | EMG DSP + kinematics windowing |",
        "| `features.py` | Time-domain feature extraction |",
        "| `camargo_io.py` | Load Camargo .mat EMG/IK |",
        "| `train.py` | Training, lag search, model benchmark |",
        "| `run_pipeline_viz.py` | Training + DSP visualizations |",
        "| `app.py` | FastAPI inference (11-ch raw EMG) |",
        "| `test_client.py` | API smoke tests |",
        "| `example_config_data.csv` | Example amplitudes (4 ch) — NOT training data |",
        "\n## Root-level legacy files\n",
        "| File | Note |",
        "|------|------|",
        "| `pipeline.py` (root) | Older duplicate of DSP |",
        "| `prepare_data.py` | Legacy Camargo loader |",
        "| `*.ipynb` | Exploratory notebooks |",
        "\n## Data flow\n",
        "```mermaid\nflowchart LR\n  MAT[Camargo .mat] --> IO[camargo_io]\n  IO --> DSP[pipeline.py]\n  DSP --> FEAT[features.py]\n  FEAT --> ML[train.py]\n  ML --> PKL[best_regressor_model.pkl]\n  PKL --> API[app.py]\n```\n",
    ]
    write("REPOSITORY_AUDIT.md", "\n".join(lines))


def phase2_dataset_report(root: str):
    summary = describe_dataset(root)
    pairs = discover_emg_ik_pairs(root, max_files=None)
    lines = [
        "# DATASET_REPORT",
        f"\nRoot: `{root}`\n",
        f"- **EMG/IK pairs:** {len(pairs)}",
        f"- **EMG sampling rate:** {EMG_FS} Hz (Camargo README)",
        f"- **IK sampling rate:** {IK_FS} Hz → decimated to {TARGET_FS} Hz for alignment",
        f"- **EMG channels (11):** {', '.join(EMG_CHANNELS)}",
        f"- **Target:** `knee_angle_r` (right knee angle, degrees)\n",
        "## Trials per subject / mode\n",
        summary.to_markdown(index=False),
        "\n## Layout\n",
        "`<subject>/<date>/<mode>/emg|ik/<trial>.mat`\n",
    ]
    summary.to_csv(os.path.join(REPORTS_DIR, "dataset_summary.csv"), index=False)
    write("DATASET_REPORT.md", "\n".join(lines))


def phase3_quality_sample(root: str, n_sample: int = 30):
    pairs = discover_emg_ik_pairs(root, max_files=n_sample)
    rows = []
    for emg_p, ik_p in pairs:
        tid = emg_p.replace(root, "").lstrip(os.sep)
        rec = {"trial_id": tid, "ok": False}
        try:
            emg = load_emg_mat(emg_p)
            knee = load_ik_knee_mat(ik_p)
            emg, knee = sync_emg_knee(emg, knee)
            rec.update({
                "ok": True,
                "emg_samples": emg.shape[0],
                "ik_samples": knee.shape[0],
                "duration_s": round(emg.shape[0] / EMG_FS, 2),
                "emg_nan": int(np.isnan(emg).sum()),
                "knee_nan": int(np.isnan(knee).sum()),
                "knee_min": float(knee.min()),
                "knee_max": float(knee.max()),
                "knee_std": float(knee.std()),
            })
        except Exception as e:
            rec["error"] = str(e)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPORTS_DIR, "quality_sample.csv"), index=False)
    ok_n = df["ok"].sum()
    lines = [
        "# DATA_QUALITY_REPORT",
        f"\nSampled **{n_sample}** trial pairs.\n",
        f"- Successful loads: **{ok_n}/{n_sample}**",
        f"- Failed: **{n_sample - ok_n}**\n",
    ]
    if ok_n:
        ok = df[df["ok"]]
        lines.append(f"- Mean duration: **{ok['duration_s'].mean():.1f} s**")
        lines.append(f"- Knee angle range (sample): **{ok['knee_min'].min():.1f}° to {ok['knee_max'].max():.1f}°**")
        lines.append(f"- NaN in EMG/knee: **{ok['emg_nan'].sum()} / {ok['knee_nan'].sum()}**")
    lines.append("\nSee `quality_sample.csv` for per-trial details.\n")
    write("DATA_QUALITY_REPORT.md", "\n".join(lines))


def main():
    root = find_camargo_root()
    print(f"Camargo root: {root}")
    phase1_repo_audit()
    phase2_dataset_report(root)
    phase3_quality_sample(root, n_sample=40)
    print("Audit discovery complete (no training).")


if __name__ == "__main__":
    main()
