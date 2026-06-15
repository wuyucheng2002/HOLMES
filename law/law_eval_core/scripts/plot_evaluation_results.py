from __future__ import annotations

import argparse
import csv
import math
import re
import site
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METRIC = "overall_score"
METRIC_LABELS = {
    "overall_score": "Overall Score",
    "answer_correct": "Answer Accuracy",
    "reasoning_score": "Reasoning Score",
    "key_rule_f1": "Key Rule F1",
    "key_rule_recall": "Key Rule Recall",
    "key_rule_precision": "Key Rule Precision",
    "order_score": "Rule Order Score",
    "trigger_rule_coverage": "Trigger Rule Coverage",
    "conflict_rule_coverage": "Conflict Rule Coverage",
    "conclusion_rule_coverage": "Conclusion Rule Coverage",
    "q_main_correct": "Main Question Accuracy",
}


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [value for value in (parse_float(item) for item in values) if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def extract_model_size_b(model: str) -> Optional[float]:
    candidates = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z0-9])", model)
    if not candidates:
        candidates = re.findall(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)[-_\s]*[Bb](?![A-Za-z0-9])", model)
    if not candidates:
        return None
    return max(float(candidate) for candidate in candidates)


def load_model_size_map(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    mapping: Dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            model = (row.get("model") or "").strip()
            size = parse_float(row.get("size_b"))
            if model and size is not None:
                mapping[model] = size
    return mapping


def model_size(model: str, manual_map: Dict[str, float]) -> Optional[float]:
    if model in manual_map:
        return manual_map[model]
    return extract_model_size_b(model)


def read_case_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def group_by(rows: List[Dict[str, str]], keys: List[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(item, "unknown") or "unknown" for item in keys)
        groups.setdefault(key, []).append(row)
    return groups


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
        writer.writerows(rows)


def sorted_models(rows: List[Dict[str, str]], manual_map: Dict[str, float]) -> List[str]:
    models = sorted({row.get("model", "unknown") or "unknown" for row in rows})
    return sorted(
        models,
        key=lambda model: (
            model_size(model, manual_map) is None,
            model_size(model, manual_map) if model_size(model, manual_map) is not None else math.inf,
            model,
        ),
    )


def build_model_summary(rows: List[Dict[str, str]], manual_map: Dict[str, float], metrics: List[str]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for (model,), group in sorted(group_by(rows, ["model"]).items()):
        item: Dict[str, Any] = {
            "model": model,
            "model_size_b": model_size(model, manual_map),
            "case_count": len(group),
        }
        for metric in metrics:
            item[metric] = mean(row.get(metric) for row in group)
        summary_rows.append(item)
    return sorted(summary_rows, key=lambda row: (row["model_size_b"] is None, row["model_size_b"] or math.inf, row["model"]))


def build_model_difficulty_summary(
    rows: List[Dict[str, str]], manual_map: Dict[str, float], metric: str
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for (model, difficulty), group in sorted(group_by(rows, ["model", "difficulty_level"]).items()):
        summary_rows.append(
            {
                "model": model,
                "model_size_b": model_size(model, manual_map),
                "difficulty_level": difficulty,
                "case_count": len(group),
                metric: mean(row.get(metric) for row in group),
            }
        )
    return sorted(
        summary_rows,
        key=lambda row: (
            row["model_size_b"] is None,
            row["model_size_b"] or math.inf,
            row["model"],
            row["difficulty_level"],
        ),
    )


def build_model_conflict_summary(rows: List[Dict[str, str]], manual_map: Dict[str, float], metric: str) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for (model, conflict_bin), group in sorted(group_by(rows, ["model", "conflict_count_bin"]).items()):
        summary_rows.append(
            {
                "model": model,
                "model_size_b": model_size(model, manual_map),
                "conflict_count_bin": conflict_bin,
                "case_count": len(group),
                metric: mean(row.get(metric) for row in group),
            }
        )
    return sorted(
        summary_rows,
        key=lambda row: (
            row["model_size_b"] is None,
            row["model_size_b"] or math.inf,
            row["model"],
            row["conflict_count_bin"],
        ),
    )


def require_matplotlib() -> Any:
    user_site = site.USER_SITE
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Run `python -m pip install matplotlib` or use --tables-only."
        ) from exc
    return plt


def plot_model_trend(model_summary: List[Dict[str, Any]], metrics: List[str], output_dir: Path) -> None:
    plt = require_matplotlib()
    rows = [row for row in model_summary if row.get("model_size_b") is not None]
    if not rows:
        print("Skip model-size trend: no model sizes could be parsed. Provide --model-size-map if needed.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = [float(row["model_size_b"]) for row in rows]
    for metric in metrics:
        y = [row.get(metric) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2, label=METRIC_LABELS.get(metric, metric))
        for xi, yi, row in zip(x, y, rows):
            if yi is not None:
                ax.annotate(row["model"], (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    ax.set_xlabel("Model Size (B parameters, parsed from model name)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Accuracy vs Model Size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_size_trend.png", dpi=180)
    plt.close(fig)


def plot_difficulty_grouped_bars(
    model_difficulty_summary: List[Dict[str, Any]], metric: str, output_dir: Path
) -> None:
    plt = require_matplotlib()
    models = sorted({str(row["model"]) for row in model_difficulty_summary})
    difficulties = sorted({str(row["difficulty_level"]) for row in model_difficulty_summary})
    if not models or not difficulties:
        return

    lookup = {
        (str(row["model"]), str(row["difficulty_level"])): row.get(metric)
        for row in model_difficulty_summary
    }
    width = 0.8 / max(len(difficulties), 1)
    x_positions = list(range(len(models)))

    fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.2), 5))
    for index, difficulty in enumerate(difficulties):
        offsets = [x + (index - (len(difficulties) - 1) / 2) * width for x in x_positions]
        values = [lookup.get((model, difficulty), 0.0) for model in models]
        ax.bar(offsets, values, width=width, label=difficulty)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Model and Difficulty")
    ax.legend(title="Difficulty")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "difficulty_by_model.png", dpi=180)
    plt.close(fig)


def plot_conflict_3d(rows: List[Dict[str, str]], manual_map: Dict[str, float], metric: str, output_dir: Path) -> None:
    plt = require_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    points = []
    for row in rows:
        size = model_size(row.get("model", ""), manual_map)
        conflict_count = parse_float(row.get("conflict_count"))
        score = parse_float(row.get(metric))
        if size is None or conflict_count is None or score is None:
            continue
        points.append((size, conflict_count, score, row.get("difficulty_level", "unknown"), row.get("model", "unknown")))
    if not points:
        print("Skip 3D plot: no rows have both model size and conflict_count.")
        return

    difficulty_values = sorted({point[3] for point in points})
    color_lookup = {difficulty: index for index, difficulty in enumerate(difficulty_values)}
    colors = [color_lookup[point[3]] for point in points]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        [point[2] for point in points],
        c=colors,
        cmap="viridis",
        alpha=0.8,
        s=45,
    )
    ax.set_xlabel("Model Size (B)")
    ax.set_ylabel("Conflict Count")
    ax.set_zlabel("Accuracy")
    ax.set_zlim(0, 1.05)
    ax.set_title(f"3D View: Model Size × Conflict Count × {METRIC_LABELS.get(metric, metric)}")

    handles, _labels = scatter.legend_elements()
    ax.legend(handles, difficulty_values, title="Difficulty", loc="upper left", bbox_to_anchor=(1.03, 1.0))
    fig.tight_layout()
    fig.savefig(output_dir / "model_size_conflict_3d.png", dpi=180)
    plt.close(fig)


def plot_conflict_lines(model_conflict_summary: List[Dict[str, Any]], metric: str, output_dir: Path) -> None:
    plt = require_matplotlib()
    models = sorted({str(row["model"]) for row in model_conflict_summary})
    conflict_bins = ["0", "1-2", "3-5", "6+", "unknown"]
    lookup = {
        (str(row["model"]), str(row["conflict_count_bin"])): row.get(metric)
        for row in model_conflict_summary
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(conflict_bins)))
    for model in models:
        y = [lookup.get((model, conflict_bin)) for conflict_bin in conflict_bins]
        if all(value is None for value in y):
            continue
        ax.plot(x, [0.0 if value is None else value for value in y], marker="o", label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(conflict_bins)
    ax.set_xlabel("Conflict Count Bin")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Conflict Complexity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "conflict_complexity_lines.png", dpi=180)
    plt.close(fig)


def find_case_metrics_path(args: argparse.Namespace) -> Path:
    if args.case_metrics:
        return resolve_path(args.case_metrics)
    batch_dir = resolve_path(args.batch_dir)
    return batch_dir / "case_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregate figures from batch LLM evaluation outputs.")
    parser.add_argument(
        "--batch-dir",
        default="eval_results/batch_dry_run_check",
        help="Batch output directory containing case_metrics.csv.",
    )
    parser.add_argument("--case-metrics", default=None, help="Direct path to case_metrics.csv.")
    parser.add_argument("--output-dir", default=None, help="Where to save plots and plot-source CSV files.")
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=sorted(METRIC_LABELS.keys()),
        help="Primary metric used for grouped and 3D plots.",
    )
    parser.add_argument(
        "--trend-metrics",
        nargs="+",
        default=["overall_score", "answer_correct", "reasoning_score", "q_main_correct"],
        choices=sorted(METRIC_LABELS.keys()),
        help="Metrics to draw on the model-size trend line plot.",
    )
    parser.add_argument(
        "--model-size-map",
        default=None,
        help="Optional CSV with columns model,size_b for model names whose size cannot be parsed.",
    )
    parser.add_argument("--tables-only", action="store_true", help="Only write plot-source CSV files, no PNG plots.")
    args = parser.parse_args()

    case_metrics_path = find_case_metrics_path(args)
    if not case_metrics_path.exists():
        raise FileNotFoundError(f"case metrics file not found: {case_metrics_path}")

    output_dir = resolve_path(args.output_dir) if args.output_dir else case_metrics_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_case_metrics(case_metrics_path)
    manual_map = load_model_size_map(resolve_path(args.model_size_map) if args.model_size_map else None)
    model_summary = build_model_summary(rows, manual_map, sorted(set(args.trend_metrics + [args.metric])))
    model_difficulty_summary = build_model_difficulty_summary(rows, manual_map, args.metric)
    model_conflict_summary = build_model_conflict_summary(rows, manual_map, args.metric)

    write_csv(output_dir / "plot_source_model_summary.csv", model_summary)
    write_csv(output_dir / "plot_source_model_difficulty.csv", model_difficulty_summary)
    write_csv(output_dir / "plot_source_model_conflict.csv", model_conflict_summary)

    if not args.tables_only:
        plot_model_trend(model_summary, args.trend_metrics, output_dir)
        plot_difficulty_grouped_bars(model_difficulty_summary, args.metric, output_dir)
        plot_conflict_lines(model_conflict_summary, args.metric, output_dir)
        plot_conflict_3d(rows, manual_map, args.metric, output_dir)

    print(f"Read case metrics: {case_metrics_path}")
    print(f"Saved plot outputs: {output_dir}")


if __name__ == "__main__":
    main()
