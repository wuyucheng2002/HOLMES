from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from run_llm_evaluation import (
    ROOT,
    dump_json,
    evaluate_dataset,
    load_dotenv,
    load_json,
    normalize_dataset_document,
    parse_optional_bool,
)


QUESTION_IDS = [
    "q_main",
    "q_diagnostic",
]

# Fixed corpus-level bucket edges chosen from the current 600-case dataset's
# approximate quartiles. They are much more balanced than the old ad hoc bins
# while remaining stable across future experiments.
REASONING_STEP_BUCKETS = [(0, 8, "0-8"), (9, 11, "9-11"), (12, 14, "12-14"), (15, None, "15+")]
CONFLICT_COUNT_BUCKETS = [(0, 1, "0-1"), (2, 3, "2-3"), (4, 5, "4-5"), (6, None, "6+")]
TRIGGERED_ARTICLE_BUCKETS = [(0, 2, "0-2"), (3, 3, "3"), (4, 4, "4"), (5, None, "5+")]
MIDDLE_CONCEPT_BUCKETS = [(0, 6, "0-6"), (7, 8, "7-8"), (9, 10, "9-10"), (11, None, "11+")]
DIFFICULTY_SCORE_BUCKETS = [(0, 20, "0-20"), (21, 31, "21-31"), (32, 42, "32-42"), (43, None, "43+")]


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "unnamed"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def discover_dataset_paths(datasets_root: str, dataset_paths: Optional[List[str]]) -> List[Path]:
    if dataset_paths:
        paths = [resolve_path(path_text) for path_text in dataset_paths]
    else:
        paths = sorted(resolve_path(datasets_root).glob("*/dataset.json"))
    return [path for path in paths if path.exists()]


def bucket_count(value: Any, buckets: List[tuple[int, Optional[int], str]]) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "unknown"
    for lower, upper, label in buckets:
        if number >= lower and (upper is None or number <= upper):
            return label
    return "unknown"


def reasoning_step_bin(value: Any) -> str:
    return bucket_count(value, REASONING_STEP_BUCKETS)


def conflict_count_bin(value: Any) -> str:
    return bucket_count(value, CONFLICT_COUNT_BUCKETS)


def triggered_article_count_bin(value: Any) -> str:
    return bucket_count(value, TRIGGERED_ARTICLE_BUCKETS)


def middle_concept_count_bin(value: Any) -> str:
    return bucket_count(value, MIDDLE_CONCEPT_BUCKETS)


def difficulty_score_bin(value: Any) -> str:
    return bucket_count(value, DIFFICULTY_SCORE_BUCKETS)


def final_result_type(gold_trace_summary: Dict[str, Any]) -> str:
    if gold_trace_summary.get("final_no_liability") is True:
        return "no_liability"
    if gold_trace_summary.get("final_result") == "No_Criminal_Liability":
        return "no_liability"
    return "liability"


def question_results_by_id(case_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        question_result.get("question_id", ""): question_result
        for question_result in case_result.get("question_results", [])
    }


def optional_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def gold_key_rule_count(gold_summary: Dict[str, Any]) -> int:
    explicit = gold_summary.get("key_rule_numbers")
    if isinstance(explicit, list) and explicit:
        return len(explicit)
    gold_trace = gold_summary.get("gold_rule_trace", [])
    if isinstance(gold_trace, list):
        return len(set(gold_trace))
    return 0


def case_metric_rows(eval_result: Dict[str, Any], dataset_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta = eval_result.get("meta", {})
    rel_dataset_path = str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path)

    for case_result in eval_result.get("cases", []):
        difficulty = case_result.get("difficulty_profile", {})
        summary = case_result.get("summary", {})
        gold_summary = case_result.get("gold_trace_summary", {})
        by_id = question_results_by_id(case_result)
        main_question = by_id.get("q_main")
        parsed_main = main_question.get("parsed_answer", {}) if isinstance(main_question, dict) else {}
        main_raw_response = main_question.get("raw_response", {}) if isinstance(main_question, dict) else {}
        token_info = main_raw_response.get("tokens", {}) if isinstance(main_raw_response, dict) else {}
        cache_info = main_raw_response.get("cache", {}) if isinstance(main_raw_response, dict) else {}

        prompt_tokens = token_info.get("prompt_tokens", token_info.get("input_tokens"))
        completion_tokens = token_info.get("completion_tokens", token_info.get("output_tokens"))
        total_tokens = token_info.get("total_tokens")
        cached_tokens = token_info.get("cached_tokens")
        cache_write_tokens = token_info.get("cache_write_tokens")
        cache_utilization_ratio = (
            float(cached_tokens) / float(prompt_tokens)
            if isinstance(cached_tokens, (int, float))
            and isinstance(prompt_tokens, (int, float))
            and float(prompt_tokens) > 0
            else None
        )

        row: Dict[str, Any] = {
            "model": meta.get("model"),
            "api_format": meta.get("api_format"),
            "dataset_path": rel_dataset_path,
            "bundle_name": case_result.get("bundle_name"),
            "case_id": case_result.get("case_id"),
            "difficulty_level": difficulty.get("difficulty_level", "unknown"),
            "legacy_difficulty_level": difficulty.get("legacy_difficulty_level", "unknown"),
            "case_type": difficulty.get("case_type", "unknown"),
            "difficulty_score": difficulty.get("difficulty_score"),
            "difficulty_score_bin": difficulty_score_bin(difficulty.get("difficulty_score")),
            "reasoning_step_count": difficulty.get("reasoning_step_count"),
            "reasoning_step_count_bin": reasoning_step_bin(difficulty.get("reasoning_step_count")),
            "triggered_article_count": difficulty.get("triggered_article_count"),
            "triggered_article_count_bin": triggered_article_count_bin(difficulty.get("triggered_article_count")),
            "triggered_middle_predicate_count": difficulty.get("triggered_middle_predicate_count"),
            "triggered_middle_predicate_count_bin": middle_concept_count_bin(difficulty.get("triggered_middle_predicate_count")),
            "positive_atomic_fact_count": difficulty.get("positive_atomic_fact_count"),
            "negative_atomic_fact_count": difficulty.get("negative_atomic_fact_count"),
            "conflict_count": difficulty.get("conflict_count"),
            "conflict_count_bin": conflict_count_bin(difficulty.get("conflict_count")),
            "stronger_conflict_count": difficulty.get("stronger_conflict_count"),
            "exception_conflict_count": difficulty.get("exception_conflict_count"),
            "applicable_article_count": difficulty.get("applicable_article_count"),
            "final_result_type": final_result_type(gold_summary),
            "gold_final_result": gold_summary.get("final_result"),
            "answer_correct": int(bool(summary.get("answer_correct"))),
            "reasoning_score": float(summary.get("reasoning_score", 0.0)),
            "overall_score": float(summary.get("overall_score", 0.0)),
            "key_rule_f1": float(summary.get("key_rule_f1", 0.0)),
            "error_stage": summary.get("error_stage", "unknown"),
            "predicted_reasoning_step_count": int(parsed_main.get("step_count", 0) or 0),
            "predicted_used_rule_count": len(parsed_main.get("used_rule_numbers", [])) if isinstance(parsed_main, dict) else None,
            "gold_key_rule_count": gold_key_rule_count(gold_summary),
            "gold_rule_trace_length": len(gold_summary.get("gold_rule_trace", [])),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cache_utilization_ratio": cache_utilization_ratio,
            "openrouter_cache_status": cache_info.get("status"),
            "openrouter_cache_age": cache_info.get("age"),
            "openrouter_cache_ttl": cache_info.get("ttl"),
            "key_rule_recall": optional_float(main_question.get("key_rule_recall")) if main_question else None,
            "key_rule_precision": optional_float(main_question.get("key_rule_precision")) if main_question else None,
            "order_score": optional_float(main_question.get("order_score")) if main_question else None,
            "trigger_rule_coverage": optional_float(main_question.get("trigger_rule_coverage")) if main_question else None,
            "conflict_rule_coverage": optional_float(main_question.get("conflict_rule_coverage")) if main_question else None,
            "conclusion_rule_coverage": optional_float(main_question.get("conclusion_rule_coverage")) if main_question else None,
            "stage_coverage_avg": optional_float(main_question.get("stage_coverage_avg")) if main_question else None,
            "extra_rule_count": int(main_question.get("extra_rule_count", 0) or 0) if main_question else 0,
        }

        for question_id in QUESTION_IDS:
            question_result = by_id.get(question_id)
            row[f"{question_id}_correct"] = (
                int(bool(question_result.get("correct"))) if question_result else None
            )
        rows.append(row)
    return rows


def accuracy(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(int(value) for value in values) / len(values)


def mean_metric(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None and str(row.get(key)).strip() != ""]
    if not values:
        return None
    return sum(values) / len(values)


def histogram(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def aggregate_record(dimension: str, group: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "dimension": dimension,
        "group": group,
        "case_count": len(rows),
        "answer_accuracy": accuracy(rows, "answer_correct"),
        "reasoning_score_mean": mean_metric(rows, "reasoning_score"),
        "overall_score_mean": mean_metric(rows, "overall_score"),
        "key_rule_f1_mean": mean_metric(rows, "key_rule_f1"),
        "key_rule_recall_mean": mean_metric(rows, "key_rule_recall"),
        "key_rule_precision_mean": mean_metric(rows, "key_rule_precision"),
        "order_score_mean": mean_metric(rows, "order_score"),
        "trigger_rule_coverage_mean": mean_metric(rows, "trigger_rule_coverage"),
        "conflict_rule_coverage_mean": mean_metric(rows, "conflict_rule_coverage"),
        "conclusion_rule_coverage_mean": mean_metric(rows, "conclusion_rule_coverage"),
        "prompt_tokens_mean": mean_metric(rows, "prompt_tokens"),
        "completion_tokens_mean": mean_metric(rows, "completion_tokens"),
        "total_tokens_mean": mean_metric(rows, "total_tokens"),
        "cached_tokens_mean": mean_metric(rows, "cached_tokens"),
        "cache_write_tokens_mean": mean_metric(rows, "cache_write_tokens"),
        "cache_utilization_ratio_mean": mean_metric(rows, "cache_utilization_ratio"),
        "response_cache_hit_rate": accuracy(
            [
                {
                    **row,
                    "_response_cache_hit": 1
                    if str(row.get("openrouter_cache_status", "")).lower() in {"hit", "stale_hit", "partial_hit"}
                    else (0 if row.get("openrouter_cache_status") not in {None, ""} else None),
                }
                for row in rows
            ],
            "_response_cache_hit",
        ),
        "error_stage_histogram": histogram(rows, "error_stage"),
    }
    for question_id in QUESTION_IDS:
        record[f"{question_id}_accuracy"] = accuracy(rows, f"{question_id}_correct")
    return record


def group_rows(rows: List[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], str]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = key_fn(row)
        groups.setdefault(key, []).append(row)
    return groups


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    dimensions: Dict[str, Callable[[Dict[str, Any]], str]] = {
        "overall": lambda _row: "all",
        "model": lambda row: str(row.get("model", "unknown")),
        "bundle_name": lambda row: str(row.get("bundle_name", "unknown")),
        "difficulty_level": lambda row: str(row.get("difficulty_level", "unknown")),
        "case_type": lambda row: str(row.get("case_type", "unknown")),
        "difficulty_score_bin": lambda row: str(row.get("difficulty_score_bin", "unknown")),
        "reasoning_step_count_bin": lambda row: str(row.get("reasoning_step_count_bin", "unknown")),
        "conflict_count_bin": lambda row: str(row.get("conflict_count_bin", "unknown")),
        "triggered_article_count_bin": lambda row: str(row.get("triggered_article_count_bin", "unknown")),
        "triggered_middle_predicate_count_bin": lambda row: str(row.get("triggered_middle_predicate_count_bin", "unknown")),
        "final_result_type": lambda row: str(row.get("final_result_type", "unknown")),
        "model_and_bundle": lambda row: f"{row.get('model', 'unknown')}::{row.get('bundle_name', 'unknown')}",
        "model_and_difficulty_level": lambda row: f"{row.get('model', 'unknown')}::{row.get('difficulty_level', 'unknown')}",
        "model_and_case_type": lambda row: f"{row.get('model', 'unknown')}::{row.get('case_type', 'unknown')}",
        "model_and_conflict_count_bin": lambda row: f"{row.get('model', 'unknown')}::{row.get('conflict_count_bin', 'unknown')}",
        "model_and_reasoning_step_count_bin": lambda row: f"{row.get('model', 'unknown')}::{row.get('reasoning_step_count_bin', 'unknown')}",
    }

    summary: Dict[str, Any] = {}
    flat_records: List[Dict[str, Any]] = []
    for dimension, key_fn in dimensions.items():
        records = [
            aggregate_record(dimension, group, group_items)
            for group, group_items in sorted(group_rows(rows, key_fn).items())
        ]
        summary[f"by_{dimension}"] = records
        flat_records.extend(records)
    summary["flat_records"] = flat_records
    return summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def parse_models(args: argparse.Namespace) -> List[str]:
    models: List[str] = []
    if args.models:
        models.extend(args.models)
    if args.model:
        models.extend(args.model)
    if not models:
        models.append("gpt-5.4-mini")
    return list(dict.fromkeys(models))


def relative_to_root(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def evaluate_model_dataset(
    *,
    model: str,
    dataset_path: Path,
    api_key: str,
    api_base: str,
    api_format: str,
    timeout_sec: int,
    reasoning_effort: Optional[str],
    enable_thinking: Optional[bool],
    max_cases_per_dataset: Optional[int],
    sleep_sec: float,
    dry_run: bool,
    output_dir: Path,
    per_dataset_dir: Path,
    chat_prompt_layout: str,
    openrouter_prompt_cache: bool,
    openrouter_prompt_cache_ttl: Optional[str],
    temperature: Optional[float],
    max_completion_tokens: Optional[int],
    openrouter_response_cache: bool,
    openrouter_response_cache_ttl: Optional[int],
    openrouter_http_referer: Optional[str],
    openrouter_x_title: Optional[str],
) -> Dict[str, Any]:
    rel_dataset = relative_to_root(dataset_path)
    dataset, dataset_scoring_target = normalize_dataset_document(load_json(dataset_path))

    def progress_callback(progress: Dict[str, Any]) -> None:
        case_index = progress.get("case_index")
        case_count = progress.get("case_count")
        case_id = progress.get("case_id", "<unknown>")
        print(f"  [{model}] {rel_dataset} case {case_index}/{case_count}: {case_id}")

    result = evaluate_dataset(
        dataset=dataset,
        api_key=api_key,
        model=model,
        base_url=api_base,
        api_format=api_format,
        timeout_sec=timeout_sec,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
        max_cases=max_cases_per_dataset,
        sleep_sec=sleep_sec,
        dry_run=dry_run,
        dataset_scoring_target=dataset_scoring_target,
        chat_prompt_layout=chat_prompt_layout,
        openrouter_prompt_cache=openrouter_prompt_cache,
        openrouter_prompt_cache_ttl=openrouter_prompt_cache_ttl,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        openrouter_response_cache=openrouter_response_cache,
        openrouter_response_cache_ttl=openrouter_response_cache_ttl,
        openrouter_http_referer=openrouter_http_referer,
        openrouter_x_title=openrouter_x_title,
        progress_callback=progress_callback,
    )
    result["meta"]["dataset_path"] = rel_dataset
    result["meta"]["batch_output_dir"] = str(output_dir)

    dataset_name = dataset_path.parent.name
    result_path = per_dataset_dir / f"{dataset_name}_{safe_name(model)}.json"
    dump_json(result_path, result)
    rows = case_metric_rows(result, dataset_path)
    return {
        "model": model,
        "dataset_path": rel_dataset,
        "rows": rows,
        "run_record": {
            "model": model,
            "dataset_path": rel_dataset,
            "result_path": relative_to_root(result_path),
            "case_count": len(rows),
            "summary": result.get("summary", {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation over all generated datasets and produce aggregate reports."
    )
    parser.add_argument(
        "--datasets-root",
        default="generated",
        help="Root directory containing <bundle>/dataset.json files. Ignored when --dataset-path is provided.",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=None,
        help="Specific dataset json path. May be repeated.",
    )
    parser.add_argument("--model", action="append", default=None, help="Model name. May be repeated.")
    parser.add_argument("--models", nargs="+", default=None, help="Alternative space-separated model list.")
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="Base URL for the API.")
    parser.add_argument(
        "--api-format",
        default="responses",
        choices=["responses", "chat_completions"],
        help="API protocol. Use chat_completions for OpenAI-compatible third-party model APIs.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing API key.")
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "xhigh"])
    parser.add_argument(
        "--enable-thinking",
        "--enable-reasoning",
        dest="enable_thinking",
        type=parse_optional_bool,
        default=None,
        help="Optional Qwen-style thinking switch for chat_completions APIs: true or false.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature. If omitted, no temperature field is sent.")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Optional upper bound on completion tokens for chat_completions requests.",
    )
    parser.add_argument(
        "--chat-prompt-layout",
        default="split_bundle_cache",
        choices=["single_user", "split_bundle_cache"],
        help="How chat_completions prompts are split into messages. split_bundle_cache isolates shared bundle rules for cache-friendly reuse.",
    )
    parser.add_argument(
        "--openrouter-prompt-cache",
        type=parse_optional_bool,
        default=False,
        help="Whether to mark the shared rules message as cacheable for OpenRouter-compatible prompt caching.",
    )
    parser.add_argument(
        "--openrouter-prompt-cache-ttl",
        default=None,
        help="Optional cache ttl label such as 5m or 1h when supported by the routed provider.",
    )
    parser.add_argument(
        "--openrouter-response-cache",
        type=parse_optional_bool,
        default=False,
        help="Whether to request OpenRouter response caching headers. Useful mainly for repeated identical requests.",
    )
    parser.add_argument(
        "--openrouter-response-cache-ttl",
        type=int,
        default=None,
        help="Optional TTL in seconds for OpenRouter response cache requests.",
    )
    parser.add_argument("--openrouter-http-referer", default=None, help="Optional HTTP-Referer header for OpenRouter.")
    parser.add_argument("--openrouter-x-title", default=None, help="Optional X-Title header for OpenRouter.")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-cases-per-dataset", type=int, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument(
        "--parallel-runs",
        "--max-workers",
        dest="parallel_runs",
        type=int,
        default=1,
        help="Number of model-dataset evaluations to run concurrently. Start small to avoid rate limits.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not call API; use gold answers as mock outputs.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep evaluating other datasets/models if one API call fails.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-dataset results and aggregate reports. Defaults to eval_results/batch_<timestamp>.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = "" if args.dry_run else os.getenv(args.api_key_env, "")
    if not args.dry_run and not api_key:
        raise RuntimeError(f"Missing API key in env var: {args.api_key_env}")

    dataset_paths = discover_dataset_paths(args.datasets_root, args.dataset_path)
    if not dataset_paths:
        raise RuntimeError("No dataset.json files found.")

    models = parse_models(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(args.output_dir) if args.output_dir else ROOT / f"eval_results/batch_{timestamp}"
    per_dataset_dir = output_dir / "per_dataset"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    tasks = [(model, dataset_path) for model in models for dataset_path in dataset_paths]
    parallel_runs = max(1, args.parallel_runs)

    if parallel_runs == 1:
        for model, dataset_path in tasks:
            rel_dataset = relative_to_root(dataset_path)
            print(f"Evaluating model={model} dataset={rel_dataset}")
            try:
                item = evaluate_model_dataset(
                    model=model,
                    dataset_path=dataset_path,
                    api_key=api_key,
                    api_base=args.api_base,
                    api_format=args.api_format,
                    timeout_sec=args.timeout_sec,
                    reasoning_effort=args.reasoning_effort,
                    enable_thinking=args.enable_thinking,
                    max_cases_per_dataset=args.max_cases_per_dataset,
                    sleep_sec=args.sleep_sec,
                    dry_run=args.dry_run,
                    output_dir=output_dir,
                    per_dataset_dir=per_dataset_dir,
                    chat_prompt_layout=args.chat_prompt_layout,
                    openrouter_prompt_cache=args.openrouter_prompt_cache,
                    openrouter_prompt_cache_ttl=args.openrouter_prompt_cache_ttl,
                    temperature=args.temperature,
                    max_completion_tokens=args.max_completion_tokens,
                    openrouter_response_cache=args.openrouter_response_cache,
                    openrouter_response_cache_ttl=args.openrouter_response_cache_ttl,
                    openrouter_http_referer=args.openrouter_http_referer,
                    openrouter_x_title=args.openrouter_x_title,
                )
                print(f"Completed evaluation for model={model} dataset={rel_dataset}")
                all_rows.extend(item["rows"])
                run_records.append(item["run_record"])
            except Exception as exc:
                error_record = {
                    "model": model,
                    "dataset_path": rel_dataset,
                    "error": repr(exc),
                }
                errors.append(error_record)
                if not args.continue_on_error:
                    raise
                print(f"ERROR: {error_record}")
    else:
        print(f"Running {len(tasks)} model-dataset evaluations with parallel_runs={parallel_runs}")
        with ThreadPoolExecutor(max_workers=parallel_runs) as executor:
            future_to_task = {}
            for model, dataset_path in tasks:
                rel_dataset = relative_to_root(dataset_path)
                print(f"Submitting model={model} dataset={rel_dataset}")
                future = executor.submit(
                    evaluate_model_dataset,
                    model=model,
                    dataset_path=dataset_path,
                    api_key=api_key,
                    api_base=args.api_base,
                    api_format=args.api_format,
                    timeout_sec=args.timeout_sec,
                    reasoning_effort=args.reasoning_effort,
                    enable_thinking=args.enable_thinking,
                    max_cases_per_dataset=args.max_cases_per_dataset,
                    sleep_sec=args.sleep_sec,
                    dry_run=args.dry_run,
                    output_dir=output_dir,
                    per_dataset_dir=per_dataset_dir,
                    chat_prompt_layout=args.chat_prompt_layout,
                    openrouter_prompt_cache=args.openrouter_prompt_cache,
                    openrouter_prompt_cache_ttl=args.openrouter_prompt_cache_ttl,
                    temperature=args.temperature,
                    max_completion_tokens=args.max_completion_tokens,
                    openrouter_response_cache=args.openrouter_response_cache,
                    openrouter_response_cache_ttl=args.openrouter_response_cache_ttl,
                    openrouter_http_referer=args.openrouter_http_referer,
                    openrouter_x_title=args.openrouter_x_title,
                )
                future_to_task[future] = (model, rel_dataset)

            for future in as_completed(future_to_task):
                model, rel_dataset = future_to_task[future]
                try:
                    item = future.result()
                    print(f"Completed evaluation for model={model} dataset={rel_dataset}")
                    all_rows.extend(item["rows"])
                    run_records.append(item["run_record"])
                except Exception as exc:
                    error_record = {
                        "model": model,
                        "dataset_path": rel_dataset,
                        "error": repr(exc),
                    }
                    errors.append(error_record)
                    if not args.continue_on_error:
                        for pending in future_to_task:
                            if pending is not future:
                                pending.cancel()
                        raise RuntimeError(f"Parallel evaluation failed: {error_record}") from exc
                    print(f"ERROR: {error_record}")

    all_rows.sort(key=lambda row: (str(row.get("model", "")), str(row.get("dataset_path", "")), str(row.get("case_id", ""))))
    run_records.sort(key=lambda row: (str(row.get("model", "")), str(row.get("dataset_path", ""))))
    aggregate = aggregate_rows(all_rows)
    aggregate_report = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": models,
            "dataset_count": len(dataset_paths),
            "case_metric_row_count": len(all_rows),
            "api_base": args.api_base,
            "api_format": args.api_format,
            "reasoning_effort": args.reasoning_effort,
            "enable_thinking": args.enable_thinking,
            "temperature": args.temperature,
            "chat_prompt_layout": args.chat_prompt_layout,
            "openrouter_prompt_cache": args.openrouter_prompt_cache,
            "openrouter_prompt_cache_ttl": args.openrouter_prompt_cache_ttl,
            "openrouter_response_cache": args.openrouter_response_cache,
            "openrouter_response_cache_ttl": args.openrouter_response_cache_ttl,
            "openrouter_http_referer": args.openrouter_http_referer,
            "openrouter_x_title": args.openrouter_x_title,
            "max_completion_tokens": args.max_completion_tokens,
            "dry_run": args.dry_run,
            "max_cases_per_dataset": args.max_cases_per_dataset,
            "parallel_runs": parallel_runs,
        },
        "runs": run_records,
        "errors": errors,
        "aggregate": aggregate,
    }

    dump_json(output_dir / "aggregate_summary.json", aggregate_report)
    write_csv(output_dir / "case_metrics.csv", all_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate["flat_records"])

    print(f"Saved aggregate summary: {output_dir / 'aggregate_summary.json'}")
    print(f"Saved case metrics CSV: {output_dir / 'case_metrics.csv'}")
    print(f"Saved aggregate metrics CSV: {output_dir / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
