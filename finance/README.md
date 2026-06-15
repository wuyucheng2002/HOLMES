# Fudan Reimbursement Reasoning Benchmark

Benchmark and evaluation harness for testing whether LLMs can apply Fudan
University's reimbursement rules to natural-language case descriptions and
produce a verifiable reasoning trace that matches an Isabelle/HOL ground truth.

> 中文文档见 [README.zh.md](README.zh.md).

## What's in this repo

| Path                                  | Description                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------- |
| `run.sh`                              | One-shot runner: install → run all 3 datasets → score. Edit the 3 lines at the top.    |
| `data/fudan_reimbursement_rules.json` | Source rules document (sections + clauses).                                            |
| `data/sample1.json`                   | 200 multi-fee compositional cases (`fee_items` outputs).                               |
| `data/sample2.json`                   | 200 multi-object review cases (`review_objects` outputs).                              |
| `data/sample3.json`                   | 679 single-task cases (case ids span 1–719 with gaps) covering 16 sub-tasks (material / invoice / approval / amount). |
| `data/case_index.json`                | 621 Isabelle case templates with per-template difficulty stats (`total_steps`, `avg_difficulty`, `numeric_steps`, …) and `meta_info` linking them back to sample3 case ids. Used by `comparison_table.py`. |
| `data/gold.json`                      | Gold answers for every accuracy task: per-task `gold_labels` / `gold_values` / `gold_pairs`. Loaded by `accuracy.py` at import time and merged into `TASK_CONFIGS`. |
| `prompts/prompts_en.py`               | Prompt templates used by the runner. (Plain Python file — variables prefixed `PROMPT_TEMPLATE`.) |
| `src/run_benchmark.py`                | Main runner. Calls a chat-completions API for each case and writes JSONL results.      |
| `src/accuracy.py`                     | Per-task gold labels + accuracy scorer.                                                |
| `src/sample3_spec.py`                 | Single source of truth for sample3 case-id ranges (shared by runner + comparison table). |
| `src/reasoning_metrics.py`            | Reasoning-trace metrics: ROUGE-L, BertScore-F1, ROSCOE-SA, Rule-F1, SRMR, SSCA.        |
| `src/comparison_table.py`             | Builds `model_comparison_table.csv` (one row per case × model).                        |
| `src/aggregate_scores.py`             | Averages metric columns to produce `avg_scores_by_model.csv`.                          |
| `src/models.py`                       | Single source of truth for the list of model directory names under `results/`.         |
| `results/`                            | Per-model JSONL outputs (one sub-directory per model).                                 |

Each case in `sample{1,2,3}.json` carries:

- `case_ch` — the natural-language scenario (English in this release).
- `isabelle_case_names` / `isabelle_case_conditions_nl` — the Isabelle case template the scenario was instantiated from.
- `natural_language_trace` — the gold reasoning trace ("Step N: Check/Compute/Conclusion — …. Result: ….").
- `rules_used` — gold rule IDs (used by Rule-Consistency metric).
- `reasoning_steps`, `reasoning_depth`, `case_category` — difficulty metadata.

## Quick start

```bash
git clone <this repo>
cd finance
$EDITOR run.sh           # edit the API_KEY / BASE_URL / MODEL lines at the top
bash run.sh              # creates .venv, installs deps, runs all 3 datasets, scores everything
```

That's it. Re-running `bash run.sh` is safe — the runner skips cases whose
results are already on disk, and the scoring stages just rebuild the CSVs.

The first scoring pass downloads ~500 MB of `bert-score` and
`sentence-transformers` models (cached in `~/.cache/huggingface`). If you only
care about pass/fail accuracy and not reasoning-trace metrics, comment out the
`reasoning_metrics.py` line in `run.sh`.

Outputs land at the project root:

| File                          | Produced by                  | Description                                  |
| ----------------------------- | ---------------------------- | -------------------------------------------- |
| `results/<NAME>/*.jsonl`      | `run_benchmark.py`           | One JSON line per case (with reasoning trace). |
| `results/<NAME>/acc.json`     | `run_benchmark.py` (sample3) | Per-task accuracy summary.                   |
| `model_comparison_table.csv`  | `comparison_table.py`        | One row per (dataset, case_id) × model.      |
| `avg_scores_by_model.csv`     | `aggregate_scores.py`        | Per-model means across all metric columns.   |

`<NAME>` defaults to `MODEL` with `/` replaced by `--`, so
`anthropic/claude-3.5-sonnet` → `results/anthropic--claude-3.5-sonnet/`.
Override with the `NAME` variable in `run.sh`.

## Manual / advanced usage

Skip `run.sh` and call the scripts directly when you want finer control.

### 1. Run the benchmark

`run_benchmark.py` calls a chat-completions endpoint for every case and writes one
JSON line per case to `results/<NAME>/<spec>_results.jsonl`. It supports
**OpenRouter** out of the box — any provider exposing the OpenAI-style
`/chat/completions` route works (set `--base-url`).

```bash
python src/run_benchmark.py \
    --api-key  "$API_KEY" \
    --base-url https://openrouter.ai/api/v1 \
    --model    anthropic/claude-3.5-sonnet \
    --dataset  sample3                   # one of sample1 | sample2 | sample3
```

Resumable: each spec re-reads its `*_results.jsonl` and skips cases that already
have a successful record, so killing and restarting is safe.

Useful flags (see `--help` for full list):

| Flag             | Default                          | Notes                                                       |
| ---------------- | -------------------------------- | ----------------------------------------------------------- |
| `--name`         | derived from `--model`           | Directory name under `results/`. Override if you want a custom label. |
| `--base-url`     | `https://openrouter.ai/api/v1`   | OpenAI-compatible chat-completions endpoint.                |
| `--batch-size`   | `4`                              | Concurrent requests per batch (`ThreadPoolExecutor`).        |
| `--request-gap`  | `0`                              | Seconds between requests within a batch (rate-limit valve). |
| `--timeout`      | `300`                            | Per-request timeout (seconds).                              |
| `--max-retries`  | `4`                              | Exponential-backoff retries on failure.                     |

The runner enables explicit `cache_control` on the system-prompt block when
`--model` starts with `anthropic/`, so the constant rules JSON benefits from
prompt caching across cases.

### 2. Score outputs

#### Per-task accuracy

`run_benchmark.py` already writes `results/<NAME>/acc.json` after a `sample3`
run finishes. To score a single result file by hand:

```bash
python src/accuracy.py material results/<NAME>/sample3_3_results.jsonl
python src/accuracy.py amount    results/<NAME>/sample3_amount_results.jsonl
# Available task names: material, invoice, approval{6,4_6,7,8,9,10,11,13..19,20},
# amount, sample1top50, sample1_{51_100,101_150,151_200},
# sample2_{1_50,51_100,101_150,151_200}
```

##### Cross-model comparison table

```bash
python src/comparison_table.py
# Writes model_comparison_table.csv to the project root.
# One row per (dataset, case_id) × {dataset, case_id, task, groundtruth, *_correct, *_pred}.
# Per-case difficulty stats (total_steps, avg_difficulty, …) come from data/case_index.json.
```

##### Reasoning-trace metrics

```bash
python src/reasoning_metrics.py            # all samples × all models
python src/reasoning_metrics.py \
    --sample sample3 \
    --model anthropic/claude-3.5-sonnet
```

Metrics computed per case and merged into `model_comparison_table.csv`:

- **ROUGE-L** — surface n-gram overlap with `natural_language_trace`.
- **BertScore-F1** (`distilbert-base-uncased`) — semantic similarity.
- **ROSCOE-SA** — bidirectional step-alignment via `sentence-transformers`.
- **Rule-F1 / Precision / Recall / Exact-Match** — over the model's `selected_rules`.
- **SRMR** (Step Result Match Rate) — greedy bipartite match between model and GT steps; checks that the *result* of each matched step agrees.
- **SSCA** — sub-scenario conclusion accuracy (sample1/sample2 only).

#### Per-model aggregate

```bash
python src/aggregate_scores.py
# Reads model_comparison_table.csv → writes avg_scores_by_model.csv.
```

## Output schema

Each line in `results/<MODEL_NAME>/<spec>_results.jsonl` is:

```jsonc
{
  "case_id": 1,
  "model": "anthropic/claude-3.5-sonnet",
  "prompt_name": "PROMPT_TEMPLATE_1",
  "case_ch": "...",
  "status": "ok",                  // or "error"
  "model_output": { ... },         // schema depends on the prompt template
  "raw_response": { ... }          // full provider response
}
```

`model_output` shape varies by prompt template:

- **`PROMPT_TEMPLATE_1`** (sample1) → `{ fee_items: [{ fee_item_index, fee_item_name, audit_result, reasoning_trace, ... }] }`
- **`PROMPT_TEMPLATE_2`** (sample2) → `{ review_objects: [{ object_index, object_name, audit_result, reasoning_trace, ... }] }`
- **`PROMPT_TEMPLATE_3*`** (sample3) → `{ audit_result, reasoning_trace, selected_rules, ... }`

## Customising the model list

The list of models lives in **`src/models.py`** as a single `MODELS` constant —
edit that one file when you add a new model directory under `results/`. The
runner does not iterate this list; only the scoring scripts
(`reasoning_metrics.py`, `comparison_table.py`, `aggregate_scores.py`) do.

## License

The code is released under the [MIT License](LICENSE).

## Citation

If you use this benchmark in academic work, please cite the repository:

```bibtex
@software{fudan_reimbursement_benchmark,
  author  = {wuyucheng2002},
  title   = {Fudan Reimbursement Reasoning Benchmark},
  year    = {2026},
  url     = {https://github.com/wuyucheng2002/fudan-reimbursement-benchmark}
}
```
