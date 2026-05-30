"""Generate SHAP plots for the saved RandomForest model."""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from config import IMAGES_DIR, MODEL_PATH, REPORTS_DIR, TARGET_COL
from train import save_plots

N_SHAP = 400


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    fn = bundle["feature_names"]

    csv_path = os.path.join(REPORTS_DIR, "training_dataset_sample.csv")
    df = pd.read_csv(csv_path)
    X = scaler.transform(df[fn].values[:N_SHAP])

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, df[fn].iloc[:N_SHAP], show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()

        mean_abs = np.abs(sv).mean(axis=0)
        rank = pd.Series(mean_abs, index=fn).sort_values(ascending=False)
        rank.head(30).to_csv(os.path.join(REPORTS_DIR, "shap_feature_ranking.csv"))

        # Per-muscle aggregate (channel prefix before _)
        muscle_imp = {}
        for feat, val in rank.items():
            muscle = feat.rsplit("_", 1)[0]
            muscle_imp[muscle] = muscle_imp.get(muscle, 0.0) + val
        muscle_rank = pd.Series(muscle_imp).sort_values(ascending=False)
        muscle_rank.to_csv(os.path.join(REPORTS_DIR, "shap_muscle_ranking.csv"))

        lines = [
            "# SHAP Explainability",
            "",
            f"Model: RandomForest on {len(df)} Camargo windows (subset {N_SHAP} for SHAP).",
            "",
            "## Top muscles (mean |SHAP| sum over features)",
            "",
        ]
        for m, v in muscle_rank.head(11).items():
            lines.append(f"- **{m}**: {v:.4f}")
        lines.extend(["", "## Top features", ""])
        for f, v in rank.head(15).items():
            lines.append(f"- `{f}`: {v:.4f}")

        with open(os.path.join(REPORTS_DIR, "SHAP_REPORT.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("SHAP saved:", os.path.join(IMAGES_DIR, "shap_summary.png"))
    except ImportError:
        print("shap not installed; running save_plots only")
        save_plots(df, model, scaler, fn, "RandomForest")


if __name__ == "__main__":
    main()
