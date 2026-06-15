import sys
from pathlib import Path

import pandas as pd

# src/aggregate_scores.py → ROOT is one level up (project root).
SRC_DIR    = Path(__file__).resolve().parent
ROOT       = SRC_DIR.parent
INPUT_CSV  = ROOT / "model_comparison_table.csv"
OUTPUT_CSV = ROOT / "avg_scores_by_model.csv"

sys.path.insert(0, str(SRC_DIR))
from models import MODELS  # noqa: E402

df = pd.read_csv(INPUT_CSV)

metrics = [
    "correct",
    "rouge_l",
    "bertscore_f1",
    "roscoe_sa",
    "roscoe_sa_recall",
    "roscoe_sa_f1",
    "rule_f1",
    "rule_precision",
    "rule_recall",
    "rule_exact_match",
    "rule_extra",
    "rule_missing",
    "rule_wrong",
    "srmr_precision",
    "srmr_recall",
    "srmr_f1",
    "ssca",
]

rows = []
for model in MODELS:
    row = {"model": model}
    for metric in metrics:
        col = f"{model}_{metric}"
        if col in df.columns:
            row[metric] = df[col].mean()
        else:
            row[metric] = float("nan")
    rows.append(row)

result = pd.DataFrame(rows).set_index("model")
print(result.to_string(float_format="{:.4f}".format))
print()

result.to_csv(OUTPUT_CSV, float_format="%.4f")
print(f"Saved to {OUTPUT_CSV}")
