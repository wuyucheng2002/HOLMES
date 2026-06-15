# 金融领域 (Finance)

围绕 *复旦大学财务报销规则* 构建的推理基准与评测工具链。
评估大模型能否：(1) 根据自然语言场景描述判断是否准予报销，
(2) 给出可验证的推理过程，与 Isabelle/HOL 形式化产生的 ground-truth trace 对齐。

> English version: [README.md](README.md)

## 目录结构

| 路径                                  | 说明                                                                                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------- |
| `run.sh`                              | 一键脚本：装依赖 → 跑 3 个 dataset → scoring。改顶部 3 行变量后 `bash run.sh` 即可。            |
| `data/fudan_reimbursement_rules.json` | 规则原文（按章节、条款组织）。                                                                |
| `data/sample1.json`                   | 200 条 *多费项组合* 案例（输出形如 `fee_items: [...]`）。                                     |
| `data/sample2.json`                   | 200 条 *多对象审查* 案例（输出形如 `review_objects: [...]`）。                                 |
| `data/sample3.json`                   | 679 条 *单一子任务* 案例（case id 取值范围 1–719，存在跳号），覆盖 16 个子任务（材料 / 发票 / 审批 / 金额）。 |
| `data/case_index.json`                | 621 个 Isabelle 案例模板，每个模板带有难度统计（`total_steps`、`avg_difficulty`、`numeric_steps` 等）以及 `meta_info` 字段把模板回指到 sample3 的 case id。`comparison_table.py` 用它生成 difficulty 列。 |
| `data/gold.json`                      | 所有准确率任务的 gold 答案（按任务名键的 `gold_labels` / `gold_values` / `gold_pairs`）。`accuracy.py` 在 import 时加载并合并进 `TASK_CONFIGS`。 |
| `prompts/prompts_en.py`               | 推理用提示词模板。**注意：实际是 Python 文件**，变量名以 `PROMPT_TEMPLATE` 开头。              |
| `src/run_benchmark.py`                | 主运行入口：按 OpenAI 兼容协议调用模型，逐条写入 JSONL 结果（断点续跑）。                      |
| `src/accuracy.py`                     | 每个子任务的 gold 标签 + 准确率统计脚本。                                                      |
| `src/sample3_spec.py`                 | sample3 的 case-id 范围与文件分布的单一来源（runner 与对比表共用）。                            |
| `src/reasoning_metrics.py`            | 推理过程指标：ROUGE-L、BertScore-F1、ROSCOE-SA、Rule-F1、SRMR、SSCA。                          |
| `src/comparison_table.py`             | 跨模型对比表生成器，输出 `model_comparison_table.csv`（每行 = case × 模型）。                  |
| `src/aggregate_scores.py`             | 把对比表里每个模型的指标列做平均，写入 `avg_scores_by_model.csv`。                              |
| `src/models.py`                       | 单一来源的模型列表（`MODELS`），其它三个评分脚本统一 import 它。                                |
| `results/`                            | 每个模型一个子目录，存放该模型的 JSONL 输出。                                                  |

每条 case 都包含：

- `case_ch` — 自然语言场景描述（本仓库提供英文版）。
- `isabelle_case_names` / `isabelle_case_conditions_nl` — 该场景所基于的 Isabelle 案例模板及其条件。
- `natural_language_trace` — Isabelle 推理过程的 NL 化表达，作为推理 trace 的 ground truth。
- `rules_used` — 该案例真实命中的规则编号，用于 Rule-Consistency 指标。
- `reasoning_steps`、`reasoning_depth`、`case_category` — 难度元数据。

## 快速开始

```bash
git clone <this repo>
cd finance
$EDITOR run.sh           # 编辑顶部的 API_KEY / BASE_URL / MODEL 三行
bash run.sh              # 自动建 .venv、装依赖、跑全部 3 个 dataset、做 scoring
```

就这三步。再次运行 `bash run.sh` 是安全的——runner 会跳过已有结果的 case，scoring 阶段只是重建 CSV。

首次跑 scoring 会下载 ~500 MB 的 `bert-score` / `sentence-transformers` 模型（缓存在 `~/.cache/huggingface`）。如果你只关心准确率而不需要推理过程指标，把 `run.sh` 里的 `reasoning_metrics.py` 那一行注释掉即可。

输出文件都落在项目根目录：

| 文件                          | 来源                          | 说明                                              |
| ----------------------------- | ----------------------------- | ------------------------------------------------- |
| `results/<NAME>/*.jsonl`      | `run_benchmark.py`            | 每行一个 case 的预测（含推理 trace）。            |
| `results/<NAME>/acc.json`     | `run_benchmark.py`（sample3） | 各子任务准确率汇总。                              |
| `model_comparison_table.csv`  | `comparison_table.py`         | 每行 = (dataset, case_id) × 模型。                |
| `avg_scores_by_model.csv`     | `aggregate_scores.py`         | 每个模型在所有指标列上的平均。                    |

`<NAME>` 默认是 `MODEL` 把 `/` 替换成 `--` 后的字符串，例如
`anthropic/claude-3.5-sonnet` → `results/anthropic--claude-3.5-sonnet/`。
要自定义可以改 `run.sh` 顶部的 `NAME` 变量。

## 手动 / 进阶用法

不想用 `run.sh` 的话直接调脚本：

### 1. 运行基准

`run_benchmark.py` 会逐条调用 `/chat/completions` 接口，并把每条结果以 JSONL 写入
`results/<NAME>/<spec>_results.jsonl`。默认走 **OpenRouter**，但任何
OpenAI 兼容端点都可以用（通过 `--base-url` 切换）。

```bash
python src/run_benchmark.py \
    --api-key  "$API_KEY" \
    --base-url https://openrouter.ai/api/v1 \
    --model    anthropic/claude-3.5-sonnet \
    --dataset  sample3                   # sample1 | sample2 | sample3
```

**断点续跑**：每个 spec 会先读取已有 `*_results.jsonl`，跳过已经成功的 case，
所以中途 kill 掉再启动是安全的。

常用参数（完整列表见 `--help`）：

| 参数             | 默认值                           | 说明                                                            |
| ---------------- | -------------------------------- | --------------------------------------------------------------- |
| `--name`         | 由 `--model` 推导                | `results/` 下的目录名，可自定义。                                |
| `--base-url`     | `https://openrouter.ai/api/v1`   | OpenAI 兼容端点。                                               |
| `--batch-size`   | `4`                              | 每个 batch 的并发请求数（`ThreadPoolExecutor`）。               |
| `--request-gap`  | `0`                              | 同一 batch 内每条请求之间的间隔秒数（限速阀）。                 |
| `--timeout`      | `300`                            | 单条请求超时秒数。                                              |
| `--max-retries`  | `4`                              | 失败时的指数退避重试次数。                                      |

如果 `--model` 以 `anthropic/` 开头，runner 会在 system message 上显式打开
`cache_control: {"type": "ephemeral"}`，使复用率高的规则 JSON 走 prompt 缓存。

### 2. 评分

#### 各子任务准确率

跑完 `sample3` 后，`run_benchmark.py` 会把汇总写到
`results/<NAME>/acc.json`。也可以单独评测某一个文件：

```bash
python src/accuracy.py material results/<NAME>/sample3_3_results.jsonl
python src/accuracy.py amount    results/<NAME>/sample3_amount_results.jsonl
# 可选 task: material, invoice, approval{6,4_6,7,8,9,10,11,13..19,20},
# amount, sample1top50, sample1_{51_100,101_150,151_200},
# sample2_{1_50,51_100,101_150,151_200}
```

#### 跨模型对比表

```bash
python src/comparison_table.py
# 在项目根目录生成 model_comparison_table.csv
# 每行 = (dataset, case_id) × {task, groundtruth, *_correct, *_pred}
# 难度统计列 (total_steps, avg_difficulty, …) 来自 data/case_index.json。
```

#### 推理过程指标

```bash
python src/reasoning_metrics.py            # 所有 sample × 所有模型
python src/reasoning_metrics.py \
    --sample sample3 \
    --model anthropic/claude-3.5-sonnet
```

逐 case 计算并合并进 `model_comparison_table.csv` 的指标：

- **ROUGE-L** — 与 `natural_language_trace` 的表层 n-gram 重叠。
- **BertScore-F1**（`distilbert-base-uncased`）— 语义相似度。
- **ROSCOE-SA** — 双向步骤对齐，使用 `sentence-transformers` 嵌入。
- **Rule-F1 / Precision / Recall / Exact-Match** — 模型 `selected_rules` 与 GT 的差异。
- **SRMR** (Step Result Match Rate) — 模型与 GT 步骤之间的贪心二分匹配，并核对每对匹配步骤的*结果值*是否一致。
- **SSCA** — 子场景结论准确率（仅 sample1 / sample2）。

##### 单模型平均

```bash
python src/aggregate_scores.py
# 读取 model_comparison_table.csv，输出 avg_scores_by_model.csv
```

## 输出 schema

`results/<MODEL_NAME>/<spec>_results.jsonl` 每行格式：

```jsonc
{
  "case_id": 1,
  "model": "anthropic/claude-3.5-sonnet",
  "prompt_name": "PROMPT_TEMPLATE_1",
  "case_ch": "...",
  "status": "ok",                  // 失败时为 "error"
  "model_output": { ... },         // schema 取决于 prompt
  "raw_response": { ... }          // 厂商完整响应（含 usage 等）
}
```

不同 prompt 模板对应的 `model_output` 形态：

- **`PROMPT_TEMPLATE_1`**（sample1）→ `{ fee_items: [{ fee_item_index, fee_item_name, audit_result, reasoning_trace, ... }] }`
- **`PROMPT_TEMPLATE_2`**（sample2）→ `{ review_objects: [{ object_index, object_name, audit_result, reasoning_trace, ... }] }`
- **`PROMPT_TEMPLATE_3*`**（sample3）→ `{ audit_result, reasoning_trace, selected_rules, ... }`

## 修改模型列表

- 模型列表统一在 **`src/models.py`** 中维护（变量名 `MODELS`）。三个评分脚本（`reasoning_metrics.py`、`comparison_table.py`、`aggregate_scores.py`）都从这里 import。
- 加入新模型：在 `results/` 下建立对应目录，然后把目录名加进 `src/models.py` 的 `MODELS` 列表即可。
- `run_benchmark.py` 不依赖这个列表，它只用 `--model` 命令行参数指定单个目标模型。

加入新模型的输出目录 `results/<NEW_MODEL>/` 后，把名字同步加到这三处即可。

## 引用 / 许可证

定稿后补充。
