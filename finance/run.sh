#!/usr/bin/env bash
# Fudan reimbursement benchmark — one-shot runner.
#
#   1. Edit the three variables below (API_KEY, BASE_URL, MODEL).
#   2. Run:  bash run.sh
#
# What it does, in order:
#   • Creates a .venv on first run and installs requirements.txt
#   • Runs sample1 + sample2 + sample3 against the chosen model
#   • Builds model_comparison_table.csv
#   • Builds avg_scores_by_model.csv  (BertScore/ROSCOE will download a model
#     the first time, so the first scoring pass takes a few minutes)
#
# Re-running is safe: the runner skips cases whose result is already on disk,
# and the scoring stages just rebuild the CSVs.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# EDIT THESE THREE LINES
# ─────────────────────────────────────────────────────────────────────────────
API_KEY="${API_KEY:-PUT-YOUR-KEY-HERE}"
BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
MODEL="${MODEL:-anthropic/claude-3.5-sonnet}"

# Optional: directory name under results/<NAME>/. Defaults to MODEL with
# slashes replaced by "--", e.g.  anthropic--claude-3.5-sonnet.
NAME="${NAME:-}"

# Optional knobs — lower these if your provider rate-limits you.
BATCH_SIZE="${BATCH_SIZE:-4}"
REQUEST_GAP_SEC="${REQUEST_GAP_SEC:-0}"

# ─────────────────────────────────────────────────────────────────────────────
# Run from the project root regardless of where the script was invoked.
cd "$(dirname "$0")"

if [[ "$API_KEY" == "PUT-YOUR-KEY-HERE" || -z "$API_KEY" ]]; then
    echo "ERROR: API_KEY is not set." >&2
    echo "Edit the API_KEY line near the top of run.sh, or run:" >&2
    echo "    API_KEY=sk-... bash run.sh" >&2
    exit 1
fi

PYTHON="${PYTHON:-python3}"
VENV=".venv"

# ── Step 1: create .venv + install deps (skipped if already up-to-date) ──────
if [[ ! -d "$VENV" ]]; then
    echo "▶ Creating $VENV"
    "$PYTHON" -m venv "$VENV"
fi
if [[ requirements.txt -nt "$VENV/.installed" ]]; then
    echo "▶ Installing dependencies"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r requirements.txt
    touch "$VENV/.installed"
fi
PY="$VENV/bin/python"

# Common args every run_benchmark invocation needs.
BENCH_ARGS=(
    --api-key  "$API_KEY"
    --base-url "$BASE_URL"
    --model    "$MODEL"
    --batch-size   "$BATCH_SIZE"
    --request-gap  "$REQUEST_GAP_SEC"
)
[[ -n "$NAME" ]] && BENCH_ARGS+=(--name "$NAME")

# ── Step 2: run the benchmark across all three datasets ──────────────────────
for ds in sample1 sample2 sample3; do
    echo
    echo "▶ Running $ds"
    "$PY" src/run_benchmark.py "${BENCH_ARGS[@]}" --dataset "$ds"
done

# ── Step 3: scoring ──────────────────────────────────────────────────────────
echo
echo "▶ Building comparison table  → model_comparison_table.csv"
"$PY" src/comparison_table.py

echo
echo "▶ Computing reasoning metrics (first run downloads BertScore/ROSCOE models)"
"$PY" src/reasoning_metrics.py

echo
echo "▶ Aggregating per-model averages → avg_scores_by_model.csv"
"$PY" src/aggregate_scores.py

echo
echo "✓ Done."
echo "  Per-case predictions: model_comparison_table.csv"
echo "  Per-model averages:   avg_scores_by_model.csv"
