# Law Evaluation Core

This folder is a runnable handoff bundle for the law evaluation pipeline.

It contains:

- the core law evaluation scripts
- the current `current10` benchmark datasets under `generated/`
- the benchmark bundle list under `configs/current10_bundles.txt`
- usage documentation for running evaluation, aggregation, and result export directly with Python

It does not contain:

- dataset generation code
- Isabelle theory files
- Isabelle installation
- the broader research workflow outside the law evaluation pipeline

The intended use is simple:

1. install Python dependencies
2. configure an API key
3. change a small number of runtime parameters such as model name, API base URL, and output path
4. run the evaluation scripts directly with Python

## Directory Layout

```text
law_eval_core/
  README.md
  requirements.txt
  .env.example
  .gitignore
  configs/
    current10_bundles.txt
  generated/
    <bundle>/dataset.json
    ...
  scripts/
    run_llm_evaluation.py
    run_batch_llm_evaluation.py
    rebuild_batch_reports.py
    export_batch_api_jsonl.py
    plot_evaluation_results.py
    render_evaluation_svgs.py
    render_dataset_difficulty_svgs.py
```

## Included Data

The `generated/` directory contains ready-to-run evaluation datasets. Each subdirectory contains a `dataset.json` file that can be passed directly to the evaluation scripts.

This handoff bundle currently includes only the `current10` benchmark subset used for the main law evaluation workflow.

The included bundles are defined in:

```text
configs/current10_bundles.txt
```

## Environment Setup

Use Python 3.10+.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Dependency notes:

- `python-dotenv` is used to load `.env`
- `matplotlib` is only needed if you want PNG plots from `plot_evaluation_results.py`
- the main evaluation flow and SVG export do not depend on Isabelle

## API Key Configuration

Set an environment variable before running the scripts.

OpenRouter example:

```powershell
$env:OPENROUTER_API_KEY="your_api_key"
```

OpenAI example:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

You can also copy `.env.example` to `.env` and fill in the relevant key.

The scripts use `--api-key-env` to decide which environment variable to read.

## Python Usage

### Single-Dataset Evaluation

```powershell
python .\scripts\run_llm_evaluation.py `
  --dataset-path .\generated\property_crime_with_defenses\dataset.json `
  --model qwen/qwen3-30b-a3b-thinking-2507 `
  --api-base https://openrouter.ai/api/v1 `
  --api-format chat_completions `
  --api-key-env OPENROUTER_API_KEY `
  --chat-prompt-layout split_bundle_cache `
  --openrouter-prompt-cache true `
  --output-path .\eval_results\single_property_crime.json
```

If you only want to verify the pipeline without calling an API, add:

```powershell
--dry-run
```

### Current10 Batch Evaluation

The following command runs directly on the `current10` subset included in this handoff folder:

```powershell
python .\scripts\run_batch_llm_evaluation.py `
  --model qwen/qwen3-30b-a3b-thinking-2507 `
  --api-base https://openrouter.ai/api/v1 `
  --api-format chat_completions `
  --api-key-env OPENROUTER_API_KEY `
  --chat-prompt-layout split_bundle_cache `
  --openrouter-prompt-cache true `
  --dataset-path .\generated\aviation_security_with_authorizations\dataset.json `
  --dataset-path .\generated\cyber_intrusion_with_authorizations\dataset.json `
  --dataset-path .\generated\environmental_pollution_with_permits\dataset.json `
  --dataset-path .\generated\mirror_customs_border_rules\dataset.json `
  --dataset-path .\generated\property_crime_with_defenses\dataset.json `
  --dataset-path .\generated\archive_relic_with_rescue_exceptions\dataset.json `
  --dataset-path .\generated\dream_archive_with_exceptions\dataset.json `
  --dataset-path .\generated\mechanical_city_maintenance_rules\dataset.json `
  --dataset-path .\generated\orbital_habitat_safety_rules\dataset.json `
  --dataset-path .\generated\subterranean_signal_dispatch_rules\dataset.json `
  --output-dir .\eval_results\current10_qwen `
  --parallel-runs 1 `
  --continue-on-error
```

If you only want a quick pipeline check, add:

```powershell
--dry-run --max-cases-per-dataset 1
```

### Arbitrary Multi-Dataset Batch Evaluation

```powershell
python .\scripts\run_batch_llm_evaluation.py `
  --model qwen/qwen3-30b-a3b-thinking-2507 `
  --api-base https://openrouter.ai/api/v1 `
  --api-format chat_completions `
  --api-key-env OPENROUTER_API_KEY `
  --chat-prompt-layout split_bundle_cache `
  --openrouter-prompt-cache true `
  --dataset-path .\generated\property_crime_with_defenses\dataset.json `
  --dataset-path .\generated\cyber_intrusion_with_authorizations\dataset.json `
  --output-dir .\eval_results\demo_batch `
  --parallel-runs 1 `
  --continue-on-error
```

### Rebuild Batch Summaries

If you already have a batch result directory and want to regenerate the aggregate files:

```powershell
python .\scripts\rebuild_batch_reports.py `
  --batch-dir .\eval_results\current10_qwen
```

If you want to rescore existing outputs with the current parser and scoring logic before rebuilding:

```powershell
python .\scripts\rebuild_batch_reports.py `
  --batch-dir .\eval_results\current10_qwen `
  --rescore
```

### Export Tables and SVG Charts

Tables:

```powershell
python .\scripts\plot_evaluation_results.py `
  --batch-dir .\eval_results\current10_qwen `
  --output-dir .\eval_results\current10_qwen\plots `
  --tables-only
```

SVG charts:

```powershell
python .\scripts\render_evaluation_svgs.py `
  --batch-dir .\eval_results\current10_qwen `
  --output-dir .\eval_results\current10_qwen\svg_plots
```

Dataset-difficulty SVG charts:

```powershell
python .\scripts\render_dataset_difficulty_svgs.py `
  --datasets-root .\generated `
  --output-dir .\eval_results\dataset_svgs
```

## Output Structure

A batch evaluation output directory typically looks like this:

```text
eval_results/<run_name>/
  per_dataset/
  case_metrics.csv
  aggregate_metrics.csv
  aggregate_summary.json
  plots/
  svg_plots/
```

The most important files are:

- `per_dataset/*.json`
  - full result JSON for each `model × dataset`
- `case_metrics.csv`
  - one row per `case × model`, useful for downstream analysis
- `aggregate_metrics.csv`
  - aggregated metrics table
- `aggregate_summary.json`
  - JSON summary of the batch run

## Key Metrics

Common fields in `case_metrics.csv` include:

- `answer_correct`
- `reasoning_score`
- `overall_score`
- `key_rule_f1`
- `key_rule_recall`
- `key_rule_precision`
- `order_score`
- `trigger_rule_coverage`
- `conflict_rule_coverage`
- `conclusion_rule_coverage`
- `error_stage`
- `difficulty_level`
- `case_type`
- `prompt_tokens`
- `completion_tokens`
- `cached_tokens`

## Recommended Workflow

If you want to use this bundle to evaluate a model, the recommended order is:

1. run a single dataset first with `run_llm_evaluation.py`
2. run the full `current10` batch with `run_batch_llm_evaluation.py`
3. if parser or scoring logic changes later, rerun `rebuild_batch_reports.py --rescore`

## Integration Note

This folder is meant to be a runnable handoff bundle, not the final public-facing repository structure.

For downstream consolidation, the safest first step is to preserve the current `scripts/`, `generated/`, and `configs/` layout so the workflow remains runnable before any larger refactor.
