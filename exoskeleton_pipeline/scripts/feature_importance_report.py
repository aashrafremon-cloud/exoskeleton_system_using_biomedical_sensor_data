"""Build explainability report from RandomForest feature_importances_ (no SHAP)."""
import os
import sys

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from config import MODEL_PATH, REPORTS_DIR

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
fn = bundle["feature_names"]
imp = pd.Series(model.feature_importances_, index=fn).sort_values(ascending=False)
imp.to_csv(os.path.join(REPORTS_DIR, "feature_importance_ranking.csv"))

muscle = {}
for feat, v in imp.items():
    m = feat.rsplit("_", 1)[0]
    muscle[m] = muscle.get(m, 0.0) + v
muscle_rank = pd.Series(muscle).sort_values(ascending=False)
muscle_rank.to_csv(os.path.join(REPORTS_DIR, "muscle_importance_ranking.csv"))

lines = [
    "# Model Explainability (RandomForest feature importance)",
    "",
    "SHAP optional (`pip install shap` in a clean venv). This report uses Gini importance.",
    "",
    "## Muscles ranked by total importance",
    "",
]
for m, v in muscle_rank.items():
    lines.append(f"- **{m}**: {v:.4f}")
lines.extend(["", "## Top 20 features", ""])
for f, v in imp.head(20).items():
    lines.append(f"- `{f}`: {v:.4f}")

with open(os.path.join(REPORTS_DIR, "SHAP_REPORT.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Wrote SHAP_REPORT.md (importance-based)")
