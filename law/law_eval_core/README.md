# Law Evaluation Core

这个文件夹是从 `law_data_final_3` 中单独整理出来的 law 评测子集，只保留“模型评测 / 指标计算 / 结果汇总 / 可视化导出”相关代码，不包含造数据代码。

目标是方便后续把这一小块直接并入一个更干净的总 repo，由合作者继续统一整理。

## 包含内容

```text
law_eval_core/
  README.md
  requirements.txt
  .gitignore
  scripts/
    run_llm_evaluation.py
    run_batch_llm_evaluation.py
    rebuild_batch_reports.py
    export_batch_api_jsonl.py
    plot_evaluation_results.py
    render_evaluation_svgs.py
```

## 不包含内容

- 数据生成代码
- `rule_bundles/` 与 Isabelle 理论文件
- 已生成数据集
- 已跑出的评测结果

也就是说，这个文件夹默认假设你已经有外部准备好的 `dataset.json`，然后直接用这些脚本跑模型和算指标。

## 脚本说明

- `scripts/run_llm_evaluation.py`
  - 单个数据集评测主入口。
  - 给定一个 `dataset.json`，调用模型 API，解析输出，计算每个 case 的答案正确性与推理指标，保存完整结果 JSON。

- `scripts/run_batch_llm_evaluation.py`
  - 多模型 / 多数据集批量评测主入口。
  - 会额外输出：
    - `per_dataset/*.json`
    - `case_metrics.csv`
    - `aggregate_metrics.csv`
    - `aggregate_summary.json`

- `scripts/rebuild_batch_reports.py`
  - 从已有的 `per_dataset/*.json` 重新生成汇总表。
  - 适合 parser 或 scoring 逻辑更新后，对旧结果重打分。

- `scripts/export_batch_api_jsonl.py`
  - 可选工具。
  - 将数据集导出成 OpenAI-compatible batch API 的 JSONL 请求文件，便于后续走异步 batch 提交。

- `scripts/plot_evaluation_results.py`
  - 从 `case_metrics.csv` 生成汇总表和 PNG 图。
  - `--tables-only` 模式下只导出中间 CSV，不画图。

- `scripts/render_evaluation_svgs.py`
  - 从 `case_metrics.csv` 生成 SVG 图。
  - 纯标准库实现，不依赖 `matplotlib`。

## 目录约束

这些脚本之间有同目录 import，所以建议保持如下结构不变：

```text
law_eval_core/
  scripts/
    run_llm_evaluation.py
    run_batch_llm_evaluation.py
    ...
```

如果后续把这些文件并入新 repo，最好仍然保留为同一个 `scripts/` 目录。

## 环境

建议 Python 3.10+。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

说明：

- `python-dotenv` 用于自动读取 `.env`
- `matplotlib` 仅 `plot_evaluation_results.py` 画 PNG 时需要
- 只跑评测、导表、导 SVG 时，`matplotlib` 不是硬依赖

## 最小使用方法

### 1. 配置 API Key

最简单的是直接设环境变量：

```powershell
$env:OPENROUTER_API_KEY="your_api_key"
```

或者：

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

脚本通过 `--api-key-env` 指定读取哪个环境变量。

### 2. 单数据集评测

```powershell
python scripts\run_llm_evaluation.py `
  --dataset-path C:\path\to\dataset.json `
  --model qwen/qwen3-30b-a3b-thinking-2507 `
  --api-base https://openrouter.ai/api/v1 `
  --api-format chat_completions `
  --api-key-env OPENROUTER_API_KEY `
  --chat-prompt-layout split_bundle_cache `
  --openrouter-prompt-cache true `
  --output-path C:\path\to\eval_results\single_eval.json
```

常改参数主要就是：

- `--model`
- `--api-base`
- `--api-key-env`
- `--api-format`

如果是 Qwen / OpenRouter 这类 OpenAI-compatible chat 接口，通常用：

- `--api-format chat_completions`

如果是 OpenAI Responses API，通常用：

- `--api-format responses`

### 3. 批量评测

```powershell
python scripts\run_batch_llm_evaluation.py `
  --model qwen/qwen3-30b-a3b-thinking-2507 `
  --api-base https://openrouter.ai/api/v1 `
  --api-format chat_completions `
  --api-key-env OPENROUTER_API_KEY `
  --chat-prompt-layout split_bundle_cache `
  --openrouter-prompt-cache true `
  --dataset-path C:\path\to\bundle_a\dataset.json `
  --dataset-path C:\path\to\bundle_b\dataset.json `
  --output-dir C:\path\to\eval_results\qwen_batch `
  --parallel-runs 1 `
  --continue-on-error
```

输出目录通常长这样：

```text
<output-dir>/
  per_dataset/
  case_metrics.csv
  aggregate_metrics.csv
  aggregate_summary.json
```

### 4. 重建汇总结果

```powershell
python scripts\rebuild_batch_reports.py `
  --batch-dir C:\path\to\eval_results\qwen_batch
```

如果想用当前 parser / scoring 逻辑对旧结果重新打分：

```powershell
python scripts\rebuild_batch_reports.py `
  --batch-dir C:\path\to\eval_results\qwen_batch `
  --rescore
```

### 5. 导出图表

PNG / 表格：

```powershell
python scripts\plot_evaluation_results.py `
  --case-metrics C:\path\to\eval_results\qwen_batch\case_metrics.csv `
  --output-dir C:\path\to\eval_results\qwen_batch\plots
```

只导出表格：

```powershell
python scripts\plot_evaluation_results.py `
  --case-metrics C:\path\to\eval_results\qwen_batch\case_metrics.csv `
  --output-dir C:\path\to\eval_results\qwen_batch\plots `
  --tables-only
```

SVG：

```powershell
python scripts\render_evaluation_svgs.py `
  --batch-dir C:\path\to\eval_results\qwen_batch `
  --output-dir C:\path\to\eval_results\qwen_batch\svg_plots
```

## 输入数据要求

这里假设输入是本项目当前 law 数据集格式的 `dataset.json`。评测脚本依赖其中的这些信息：

- `cases`
- 每个 case 下的 `questions`
- `q_main` / `q_diagnostic`
- gold answer / scoring target / gold trace summary
- `rules_text`
- `case_text`
- `difficulty_profile`

因此，这个整理包不是通用 benchmark runner，而是“面向当前 law/HOL 评测格式”的代码子集。

## 关键指标

批量结果里最重要的是 `case_metrics.csv`，常用字段包括：

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

`aggregate_metrics.csv` 则是在不同维度上做聚合后的结果，例如：

- 按 model
- 按 bundle
- 按 difficulty level
- 按 conflict count bucket
- 按 reasoning step bucket

## 推荐接入方式

后续并入总 repo 时，建议把这部分当成一个“law evaluation module”处理：

1. 保留 `scripts/` 目录结构不变。
2. 将数据集路径改为由外层 repo 统一传参，而不是依赖本目录默认路径。
3. 将 `.env` / API 参数管理统一交给总 repo。
4. 如果后续要追求“一键跑”，优先包装这三个主入口：
   - `run_llm_evaluation.py`
   - `run_batch_llm_evaluation.py`
   - `rebuild_batch_reports.py`

## 备注

- 这个文件夹是“评测子集整理包”，不是最终对外发布版 repo。
- 这里没有再做代码重构，只做了抽取和说明，目的是便于后续合作者继续汇总整理。
