from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from run_batch_llm_evaluation import (
    ROOT,
    aggregate_rows,
    case_metric_rows,
    relative_to_root,
    resolve_path,
    write_csv,
)
from run_llm_evaluation import (
    dump_json,
    load_json,
    normalize_dataset_document,
    rescore_saved_evaluation_result,
)


def load_per_dataset_results(per_dataset_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in sorted(per_dataset_dir.glob("*.json")):
        data = load_json(path)
        if isinstance(data, dict):
            data["_source_path"] = str(path)
            results.append(data)
    return results


def rebuild_batch_reports(batch_dir: Path, rescore: bool = False) -> Dict[str, Any]:
    per_dataset_dir = batch_dir / "per_dataset"
    if not per_dataset_dir.exists():
        raise FileNotFoundError(f"per_dataset directory not found: {per_dataset_dir}")

    results = load_per_dataset_results(per_dataset_dir)
    if not results:
        raise FileNotFoundError(f"no per-dataset result json files found under: {per_dataset_dir}")

    all_rows: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    models = set()
    dataset_paths = set()

    for result in results:
        meta = result.get("meta", {})
        dataset_path_text = str(meta.get("dataset_path", "")).strip()
        dataset_path = resolve_path(dataset_path_text) if dataset_path_text else Path("")
        source_path = Path(result.get("_source_path", "")) if result.get("_source_path") else None
        if rescore and dataset_path_text and dataset_path.exists():
            dataset_data = load_json(dataset_path)
            dataset_cases, dataset_scoring_target = normalize_dataset_document(dataset_data)
            result = rescore_saved_evaluation_result(result, dataset_cases, dataset_scoring_target)
            if source_path:
                dump_json(source_path, result)
        rows = case_metric_rows(result, dataset_path)
        all_rows.extend(rows)

        model = str(meta.get("model", "unknown"))
        dataset_name = dataset_path.parent.name if dataset_path_text else "unknown_dataset"
        result_path = per_dataset_dir / f"{dataset_name}_{model.replace('/', '_')}.json"

        models.add(model)
        if dataset_path_text:
            dataset_paths.add(dataset_path_text)

        run_records.append(
            {
                "model": model,
                "dataset_path": dataset_path_text,
                "result_path": relative_to_root(result_path),
                "case_count": len(result.get("cases", [])),
                "summary": result.get("summary", {}),
            }
        )

    all_rows.sort(key=lambda row: (str(row.get("model", "")), str(row.get("dataset_path", "")), str(row.get("case_id", ""))))
    run_records.sort(key=lambda row: (str(row.get("model", "")), str(row.get("dataset_path", ""))))
    aggregate = aggregate_rows(all_rows)

    aggregate_report = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": sorted(models),
            "dataset_count": len(dataset_paths),
            "case_metric_row_count": len(all_rows),
            "recovered_from_per_dataset": True,
            "batch_dir": str(batch_dir),
        },
        "runs": run_records,
        "errors": [],
        "aggregate": aggregate,
    }

    dump_json(batch_dir / "aggregate_summary.json", aggregate_report)
    write_csv(batch_dir / "case_metrics.csv", all_rows)
    write_csv(batch_dir / "aggregate_metrics.csv", aggregate["flat_records"])
    return aggregate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild batch evaluation aggregate reports from per_dataset/*.json files.")
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Batch directory containing per_dataset/*.json.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-parse and re-score existing raw model responses with the current evaluation logic before rebuilding reports.",
    )
    args = parser.parse_args()

    batch_dir = resolve_path(args.batch_dir)
    report = rebuild_batch_reports(batch_dir, rescore=args.rescore)
    print(f"Rebuilt batch reports in: {batch_dir}")
    print(f"Models: {', '.join(report['meta']['models'])}")
    print(f"Dataset count: {report['meta']['dataset_count']}")
    print(f"Case metric rows: {report['meta']['case_metric_row_count']}")


if __name__ == "__main__":
    main()
