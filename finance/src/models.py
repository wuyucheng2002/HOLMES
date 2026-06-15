"""Single source of truth for the model directory names this benchmark scores.

Each entry must match a sub-directory name under ``results/`` — the runner
(``run_benchmark.py``) writes ``results/<model>/sample{1,2,3}_*_results.jsonl``,
and ``comparison_table.py`` / ``reasoning_metrics.py`` / ``aggregate_scores.py``
all iterate this list.

Directory names can use any string. The convention used in the published
results is ``<vendor>--<model-id>`` so they round-trip safely on every OS,
but plain vendor-free names work equally well.
"""

from __future__ import annotations

# Replace these placeholders with the model directory names you have under
# ``results/``. The list may be empty until you produce your first run.
MODELS: list[str] = [
    # "anthropic--claude-3.5-sonnet",
    # "openai--gpt-4o",
    # "deepseek--deepseek-chat",
]
