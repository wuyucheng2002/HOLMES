from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def ordered_difficulty_levels(rows: List[Dict[str, Any]]) -> List[str]:
    levels = sorted({str(row.get("difficulty_level", "unknown")) for row in rows})
    tier_levels = sorted(
        [level for level in levels if level.startswith("tier_") and level.split("_")[-1].isdigit()],
        key=lambda item: int(item.split("_")[-1]),
    )
    legacy_order = [level for level in ["basic", "medium", "hard", "very_hard"] if level in levels]
    remaining = [level for level in levels if level not in tier_levels and level not in legacy_order]
    return tier_levels + legacy_order + remaining


def read_case_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


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
    numeric = [v for v in (parse_float(item) for item in values) if v is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def sorted_models(rows: List[Dict[str, str]]) -> List[str]:
    models = sorted({row.get("model", "unknown") or "unknown" for row in rows})

    def model_size(model: str) -> float:
        import re
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*[Bb]", model)
        if matches:
            return max(float(x) for x in matches)
        return math.inf

    return sorted(models, key=lambda model: (model_size(model), model))


def bucket_sort_key(label: str) -> tuple[int, int, str]:
    text = (label or "").strip()
    if text == "unknown":
        return (10**9, 10**9, text)
    match = re.match(r"^\s*(\d+)(?:-(\d+)|\+)?\s*$", text)
    if match:
        lower = int(match.group(1))
        upper = int(match.group(2)) if match.group(2) else (10**9 if text.endswith("+") else lower)
        return (lower, upper, text)
    return (10**8, 10**8, text)


def ordered_labels(rows: List[Dict[str, str]], key: str, preferred: Optional[Sequence[str]] = None) -> List[str]:
    values = sorted({(row.get(key, "unknown") or "unknown") for row in rows})
    if preferred:
        preferred_set = set(preferred)
        ordered = [label for label in preferred if label in values]
        rest = [label for label in values if label not in preferred_set]
        return ordered + sorted(rest, key=bucket_sort_key)
    return sorted(values, key=bucket_sort_key)


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #222; }",
        ".title { font-size: 20px; font-weight: bold; }",
        ".axis-label { font-size: 14px; }",
        ".tick { font-size: 12px; }",
        ".small { font-size: 11px; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def svg_footer(lines: List[str], path: Path) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_text(lines: List[str], x: float, y: float, text: str, css_class: str = "tick", anchor: str = "start", rotate: float | None = None) -> None:
    attrs = f'x="{x:.1f}" y="{y:.1f}" class="{css_class}" text-anchor="{anchor}"'
    if rotate is not None:
        attrs += f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"'
    lines.append(f"<text {attrs}>{escape(text)}</text>")


def group_mean(rows: List[Dict[str, str]], group_keys: Sequence[str], metric: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple((row.get(k, "unknown") or "unknown") for k in group_keys)
        groups[key].append(row)

    out: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        record: Dict[str, Any] = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        record["case_count"] = len(items)
        record[metric] = mean(item.get(metric) for item in items)
        out.append(record)
    return out


def macro_group_mean(
    rows: List[Dict[str, str]],
    *,
    model_key: str,
    group_key: str,
    bundle_key: str,
    metric: str,
) -> List[Dict[str, Any]]:
    bundle_level: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        model = row.get(model_key, "unknown") or "unknown"
        group = row.get(group_key, "unknown") or "unknown"
        bundle = row.get(bundle_key, "unknown") or "unknown"
        value = parse_float(row.get(metric))
        if value is not None:
            bundle_level[(model, group, bundle)].append(value)

    pooled_bundle_means: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    bundle_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for (model, group, bundle), values in bundle_level.items():
        if not values:
            continue
        pooled_bundle_means[(model, group)].append(sum(values) / len(values))
        bundle_counts[(model, group)] += 1

    out: List[Dict[str, Any]] = []
    for (model, group), values in sorted(pooled_bundle_means.items()):
        out.append(
            {
                model_key: model,
                group_key: group,
                "bundle_count": bundle_counts[(model, group)],
                metric: (sum(values) / len(values)) if values else None,
            }
        )
    return out


def series_from_grouped_rows(
    grouped_rows: List[Dict[str, Any]],
    *,
    x_key: str,
    value_key: str,
    models: Sequence[str],
    x_labels: Sequence[str],
    colors: Sequence[str],
) -> List[Tuple[str, Sequence[float | None], str]]:
    series: List[Tuple[str, Sequence[float | None], str]] = []
    for idx, model in enumerate(models):
        values: List[float | None] = []
        for x_label in x_labels:
            value = next(
                (
                    grouped_row.get(value_key)
                    for grouped_row in grouped_rows
                    if str(grouped_row.get("model")) == model and str(grouped_row.get(x_key)) == x_label
                ),
                None,
            )
            values.append(None if value is None else float(value))
        series.append((model, values, colors[idx % len(colors)]))
    return series


def draw_grouped_bar_svg(
    category_labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], str]],
    title: str,
    output_path: Path,
    ylabel: str = "Accuracy",
) -> None:
    width, height = 1100, 620
    left, right, top, bottom = 90, 40, 70, 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = 1.0

    lines = svg_header(width, height)
    draw_text(lines, width / 2, 32, title, "title", "middle")
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')

    slot_w = plot_w / max(len(category_labels), 1)
    total_series = max(len(series), 1)
    bar_w = slot_w * 0.7 / total_series

    for idx, label in enumerate(category_labels):
        draw_text(lines, left + idx * slot_w + slot_w / 2, top + plot_h + 20, label, "tick", "middle", rotate=-20)

    for s_idx, (series_name, values, color) in enumerate(series):
        for idx, value in enumerate(values):
            v = 0.0 if value is None else float(value)
            x = left + idx * slot_w + slot_w * 0.15 + s_idx * bar_w
            bar_h = plot_h * (v / max_value)
            y = top + plot_h - bar_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.9"/>')
            draw_text(lines, x + bar_w / 2, y - 6, f"{v:.2f}", "small", "middle")

    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - plot_h * ratio
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        draw_text(lines, left - 10, y + 4, f"{ratio:.2f}", "tick", "end")

    legend_x = left
    legend_y = height - 80
    for idx, (name, _values, color) in enumerate(series):
        x = legend_x + idx * 280
        lines.append(f'<rect x="{x}" y="{legend_y}" width="18" height="18" fill="{color}"/>')
        draw_text(lines, x + 26, legend_y + 14, name, "tick")

    draw_text(lines, 26, height / 2, ylabel, "axis-label", "middle", rotate=-90)
    svg_footer(lines, output_path)


def draw_line_chart_svg(
    x_labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], str]],
    title: str,
    output_path: Path,
    ylabel: str = "Accuracy",
) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 90, 40, 70, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    def scale_x(index: int) -> float:
        if len(x_labels) == 1:
            return left + plot_w / 2
        return left + index * plot_w / (len(x_labels) - 1)

    def scale_y(value: float) -> float:
        return top + plot_h - plot_h * value

    lines = svg_header(width, height)
    draw_text(lines, width / 2, 32, title, "title", "middle")
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')

    for idx, label in enumerate(x_labels):
        x = scale_x(idx)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eee" stroke-width="1"/>')
        draw_text(lines, x, top + plot_h + 22, label, "tick", "middle")

    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = scale_y(ratio)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        draw_text(lines, left - 10, y + 4, f"{ratio:.2f}", "tick", "end")

    for name, values, color in series:
        points = []
        for idx, value in enumerate(values):
            if value is None:
                continue
            points.append((scale_x(idx), scale_y(float(value))))
        if len(points) >= 2:
            points_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points_text}"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')

    legend_x = left + plot_w - 240
    legend_y = top + 10
    for idx, (name, _values, color) in enumerate(series):
        y = legend_y + idx * 24
        lines.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        draw_text(lines, legend_x + 14, y + 4, name, "tick")

    draw_text(lines, 26, height / 2, ylabel, "axis-label", "middle", rotate=-90)
    svg_footer(lines, output_path)


def draw_heatmap_svg(rows: List[Dict[str, Any]], x_key: str, y_key: str, value_key: str, title: str, output_path: Path) -> None:
    x_labels = sorted({str(row[x_key]) for row in rows})
    y_labels = sorted({str(row[y_key]) for row in rows})
    lookup = {(str(row[y_key]), str(row[x_key])): row.get(value_key) for row in rows}

    width = max(900, 220 + len(x_labels) * 170)
    height = max(520, 140 + len(y_labels) * 44)
    left, right, top, bottom = 220, 40, 70, 60
    cell_w, cell_h = 150, 42

    vals = [float(v) for v in (row.get(value_key) for row in rows) if v is not None]
    min_v = min(vals) if vals else 0.0
    max_v = max(vals) if vals else 1.0

    def color(value: Optional[float]) -> str:
        if value is None:
            return "rgb(240,240,240)"
        if max_v == min_v:
            ratio = 0.5
        else:
            ratio = (float(value) - min_v) / (max_v - min_v)
        red = int(245 - 120 * ratio)
        green = int(248 - 40 * ratio)
        blue = int(255 - 165 * ratio)
        return f"rgb({red},{green},{blue})"

    lines = svg_header(width, height)
    draw_text(lines, width / 2, 32, title, "title", "middle")

    for c_idx, x_label in enumerate(x_labels):
        x = left + c_idx * cell_w + cell_w / 2
        draw_text(lines, x, top - 14, x_label, "tick", "middle")

    for r_idx, y_label in enumerate(y_labels):
        y = top + r_idx * cell_h
        draw_text(lines, left - 10, y + 25, y_label, "tick", "end")
        for c_idx, x_label in enumerate(x_labels):
            x = left + c_idx * cell_w
            value = lookup.get((y_label, x_label))
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color(value)}" stroke="white"/>')
            if value is not None:
                draw_text(lines, x + cell_w / 2, y + 25, f"{float(value):.2f}", "small", "middle")

    svg_footer(lines, output_path)


def draw_stacked_bar_svg(
    categories: Sequence[str],
    stacks: Sequence[Tuple[str, Sequence[float], str]],
    title: str,
    output_path: Path,
    ylabel: str = "Case Count",
) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 90, 40, 70, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    totals = [sum(stack[1][idx] for stack in stacks) for idx in range(len(categories))]
    max_total = max(totals) if totals else 1

    lines = svg_header(width, height)
    draw_text(lines, width / 2, 32, title, "title", "middle")
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')

    slot_w = plot_w / max(len(categories), 1)
    bar_w = slot_w * 0.62
    for idx, category in enumerate(categories):
        x = left + idx * slot_w + (slot_w - bar_w) / 2
        running_h = 0.0
        for name, values, color in stacks:
            value = values[idx]
            rect_h = 0 if max_total == 0 else plot_h * (value / max_total)
            y = top + plot_h - running_h - rect_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" fill="{color}" opacity="0.95"/>')
            running_h += rect_h
        draw_text(lines, x + bar_w / 2, top + plot_h + 20, category, "tick", "middle")

    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - plot_h * ratio
        count = round(max_total * ratio)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        draw_text(lines, left - 10, y + 4, str(count), "tick", "end")

    legend_x = left
    legend_y = height - 70
    for idx, (name, _values, color) in enumerate(stacks):
        x = legend_x + idx * 180
        lines.append(f'<rect x="{x}" y="{legend_y}" width="18" height="18" fill="{color}"/>')
        draw_text(lines, x + 26, legend_y + 14, name, "tick")

    draw_text(lines, 26, height / 2, ylabel, "axis-label", "middle", rotate=-90)
    svg_footer(lines, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render evaluation-result plots as SVG without matplotlib.")
    parser.add_argument("--batch-dir", required=True, help="Batch evaluation directory containing case_metrics.csv.")
    parser.add_argument("--output-dir", default=None, help="Output directory for SVG charts and source CSV files.")
    args = parser.parse_args()

    batch_dir = resolve_path(args.batch_dir)
    case_metrics_path = batch_dir / "case_metrics.csv"
    if not case_metrics_path.exists():
        raise FileNotFoundError(f"case_metrics.csv not found: {case_metrics_path}")

    output_dir = resolve_path(args.output_dir) if args.output_dir else batch_dir / "svg_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_case_metrics(case_metrics_path)
    for row in rows:
        answer = parse_float(row.get("answer_correct"))
        reasoning = parse_float(row.get("reasoning_score"))
        if answer is not None and reasoning is not None:
            row["answer_reasoning_gap"] = answer - reasoning
        else:
            row["answer_reasoning_gap"] = ""
    models = sorted_models(rows)
    difficulty_order = ordered_difficulty_levels(rows)
    case_type_order = ordered_labels(rows, "case_type", preferred=["liability_only", "liability_with_exception"])
    conflict_order = ordered_labels(rows, "conflict_count_bin")
    step_order = ordered_labels(rows, "reasoning_step_count_bin")
    article_order = ordered_labels(rows, "triggered_article_count_bin")
    middle_order = ordered_labels(rows, "triggered_middle_predicate_count_bin")
    raw_conflict_order = ordered_labels(rows, "conflict_count")
    raw_step_order = ordered_labels(rows, "reasoning_step_count")
    colors = ["#4C78A8", "#E15759", "#59A14F", "#F28E2B"]

    model_summary = group_mean(rows, ["model"], "overall_score")
    for record in model_summary:
        record["answer_correct"] = mean(row.get("answer_correct") for row in rows if row.get("model") == record["model"])
        record["q_main_correct"] = mean(row.get("q_main_correct") for row in rows if row.get("model") == record["model"])
        record["reasoning_score"] = mean(row.get("reasoning_score") for row in rows if row.get("model") == record["model"])
        record["key_rule_f1"] = mean(row.get("key_rule_f1") for row in rows if row.get("model") == record["model"])
        record["order_score"] = mean(row.get("order_score") for row in rows if row.get("model") == record["model"])
        record["stage_coverage_avg"] = mean(row.get("stage_coverage_avg") for row in rows if row.get("model") == record["model"])
        record["cache_utilization_ratio"] = mean(row.get("cache_utilization_ratio") for row in rows if row.get("model") == record["model"])
        record["response_cache_hit"] = mean(
            1.0 if str(row.get("openrouter_cache_status", "")).lower() in {"hit", "stale_hit", "partial_hit"} else 0.0
            for row in rows
            if row.get("model") == record["model"] and str(row.get("openrouter_cache_status", "")).strip()
        )
    write_csv(output_dir / "plot_source_model_summary.csv", model_summary)

    difficulty_summary = group_mean(rows, ["model", "difficulty_level"], "overall_score")
    write_csv(output_dir / "plot_source_model_difficulty.csv", difficulty_summary)

    difficulty_answer_summary = group_mean(rows, ["model", "difficulty_level"], "answer_correct")
    write_csv(output_dir / "plot_source_model_difficulty_answer.csv", difficulty_answer_summary)

    difficulty_reasoning_summary = group_mean(rows, ["model", "difficulty_level"], "reasoning_score")
    write_csv(output_dir / "plot_source_model_difficulty_reasoning.csv", difficulty_reasoning_summary)

    case_type_summary = group_mean(rows, ["model", "case_type"], "overall_score")
    write_csv(output_dir / "plot_source_model_case_type.csv", case_type_summary)

    case_type_reasoning_summary = group_mean(rows, ["model", "case_type"], "reasoning_score")
    write_csv(output_dir / "plot_source_model_case_type_reasoning.csv", case_type_reasoning_summary)

    conflict_summary = group_mean(rows, ["model", "conflict_count_bin"], "overall_score")
    write_csv(output_dir / "plot_source_model_conflict.csv", conflict_summary)

    conflict_reasoning_summary = group_mean(rows, ["model", "conflict_count_bin"], "reasoning_score")
    write_csv(output_dir / "plot_source_model_conflict_reasoning.csv", conflict_reasoning_summary)

    step_summary = group_mean(rows, ["model", "reasoning_step_count_bin"], "overall_score")
    write_csv(output_dir / "plot_source_model_steps.csv", step_summary)

    step_reasoning_summary = group_mean(rows, ["model", "reasoning_step_count_bin"], "reasoning_score")
    write_csv(output_dir / "plot_source_model_steps_reasoning.csv", step_reasoning_summary)

    bundle_summary = group_mean(rows, ["model", "bundle_name"], "overall_score")
    write_csv(output_dir / "plot_source_model_bundle.csv", bundle_summary)

    bundle_reasoning_summary = group_mean(rows, ["model", "bundle_name"], "reasoning_score")
    write_csv(output_dir / "plot_source_model_bundle_reasoning.csv", bundle_reasoning_summary)

    article_summary = group_mean(rows, ["model", "triggered_article_count_bin"], "overall_score")
    write_csv(output_dir / "plot_source_model_article_count.csv", article_summary)

    middle_summary = group_mean(rows, ["model", "triggered_middle_predicate_count_bin"], "overall_score")
    write_csv(output_dir / "plot_source_model_middle_count.csv", middle_summary)

    complexity_specs = [
        ("conflict_count", raw_conflict_order, "conflict_raw", "Conflict Count"),
        ("reasoning_step_count", raw_step_order, "reasoning_step_raw", "Reasoning Step Count"),
    ]
    metric_specs = [
        ("answer_correct", "answer_accuracy", "Answer Accuracy"),
        ("key_rule_recall", "key_rule_recall", "Key Rule Recall"),
        ("key_rule_f1", "key_rule_f1", "Key Rule F1"),
        ("reasoning_score", "reasoning_score", "Reasoning Score"),
        ("overall_score", "overall_score", "Overall Score"),
        ("answer_reasoning_gap", "answer_reasoning_gap", "Answer - Reasoning Gap"),
    ]

    for group_key, x_order, slug, label in complexity_specs:
        for metric_key, metric_slug, metric_label in metric_specs:
            grouped = group_mean(rows, ["model", group_key], metric_key)
            write_csv(output_dir / f"plot_source_{slug}_{metric_slug}.csv", grouped)
            series = series_from_grouped_rows(
                grouped,
                x_key=group_key,
                value_key=metric_key,
                models=models,
                x_labels=x_order,
                colors=colors,
            )
            draw_line_chart_svg(
                x_order,
                series,
                f"{metric_label} vs {label}",
                output_dir / f"{slug}_{metric_slug}_lines.svg",
                ylabel=metric_label,
            )

    macro_specs = [
        ("answer_correct", "answer_accuracy", "Answer Accuracy"),
        ("key_rule_f1", "key_rule_f1", "Key Rule F1"),
        ("reasoning_score", "reasoning_score", "Reasoning Score"),
        ("overall_score", "overall_score", "Overall Score"),
        ("answer_reasoning_gap", "answer_reasoning_gap", "Answer - Reasoning Gap"),
    ]
    for group_key, x_order, slug, label in complexity_specs:
        for metric_key, metric_slug, metric_label in macro_specs:
            grouped = macro_group_mean(
                rows,
                model_key="model",
                group_key=group_key,
                bundle_key="bundle_name",
                metric=metric_key,
            )
            write_csv(output_dir / f"plot_source_macro_{slug}_{metric_slug}.csv", grouped)
            series = series_from_grouped_rows(
                grouped,
                x_key=group_key,
                value_key=metric_key,
                models=models,
                x_labels=x_order,
                colors=colors,
            )
            draw_line_chart_svg(
                x_order,
                series,
                f"Macro Avg {metric_label} vs {label}",
                output_dir / f"macro_{slug}_{metric_slug}_lines.svg",
                ylabel=metric_label,
            )

    error_stage_rows: List[Dict[str, Any]] = []
    for model in models:
        counter = Counter(row.get("error_stage", "unknown") or "unknown" for row in rows if row.get("model") == model)
        record = {"model": model}
        record.update(counter)
        error_stage_rows.append(record)
    write_csv(output_dir / "plot_source_error_stage.csv", error_stage_rows)

    overall_series = [
        (
            "Answer accuracy",
            [next((r["answer_correct"] for r in model_summary if r["model"] == model), 0.0) for model in models],
            "#E15759",
        ),
        (
            "Overall score",
            [next((r["overall_score"] for r in model_summary if r["model"] == model), 0.0) for model in models],
            "#4C78A8",
        ),
        (
            "Q main accuracy",
            [next((r["q_main_correct"] for r in model_summary if r["model"] == model), 0.0) for model in models],
            "#76B7B2",
        ),
    ]
    draw_grouped_bar_svg(models, overall_series, "Answer/Q-Main/Overall Comparison", output_dir / "overall_metrics_by_model.svg")

    reasoning_quality_series = [
        (
            "Reasoning score",
            [next((r["reasoning_score"] for r in model_summary if r["model"] == model), 0.0) for model in models],
            "#59A14F",
        ),
        (
            "Key rule F1",
            [next((r["key_rule_f1"] for r in model_summary if r["model"] == model), 0.0) for model in models],
            "#F28E2B",
        ),
        (
            "Order score",
            [next((r["order_score"] for r in model_summary if r["model"] == model), 0.0) for model in models],
            "#4C78A8",
        ),
        (
            "Stage coverage",
            [next((r["stage_coverage_avg"] for r in model_summary if r["model"] == model), 0.0) for model in models],
            "#E15759",
        ),
    ]
    draw_grouped_bar_svg(models, reasoning_quality_series, "Reasoning-Trace Quality by Model", output_dir / "reasoning_quality_by_model.svg")

    cache_series = [
        (
            "Prompt cache utilization",
            [next((r["cache_utilization_ratio"] for r in model_summary if r["model"] == model), 0.0) for model in models],
            "#4C78A8",
        ),
        (
            "Response cache hit rate",
            [next((r["response_cache_hit"] for r in model_summary if r["model"] == model), 0.0) or 0.0 for model in models],
            "#59A14F",
        ),
    ]
    draw_grouped_bar_svg(models, cache_series, "OpenRouter Cache Utilization by Model", output_dir / "cache_utilization_by_model.svg")

    difficulty_series = []
    for idx, model in enumerate(models):
        values = []
        for level in difficulty_order:
            value = next(
                (
                    r["overall_score"]
                    for r in difficulty_summary
                    if r["model"] == model and r["difficulty_level"] == level
                ),
                None,
            )
            values.append(0.0 if value is None else float(value))
        difficulty_series.append((model, values, colors[idx % len(colors)]))
    draw_grouped_bar_svg(difficulty_order, difficulty_series, "Overall Score by Difficulty Level", output_dir / "difficulty_overall_score_by_model.svg")

    difficulty_answer_series = []
    for idx, model in enumerate(models):
        values = []
        for level in difficulty_order:
            value = next(
                (
                    r["answer_correct"]
                    for r in difficulty_answer_summary
                    if r["model"] == model and r["difficulty_level"] == level
                ),
                None,
            )
            values.append(0.0 if value is None else float(value))
        difficulty_answer_series.append((model, values, colors[idx % len(colors)]))
    draw_grouped_bar_svg(difficulty_order, difficulty_answer_series, "Answer Accuracy by Difficulty Level", output_dir / "difficulty_answer_accuracy_by_model.svg")

    difficulty_reasoning_series = []
    for idx, model in enumerate(models):
        values = []
        for level in difficulty_order:
            value = next(
                (
                    r["reasoning_score"]
                    for r in difficulty_reasoning_summary
                    if r["model"] == model and r["difficulty_level"] == level
                ),
                None,
            )
            values.append(0.0 if value is None else float(value))
        difficulty_reasoning_series.append((model, values, colors[idx % len(colors)]))
    draw_grouped_bar_svg(difficulty_order, difficulty_reasoning_series, "Reasoning Score by Difficulty Level", output_dir / "difficulty_reasoning_score_by_model.svg")

    case_type_overall_series = []
    for idx, model in enumerate(models):
        values = []
        for case_type in case_type_order:
            value = next(
                (
                    r["overall_score"]
                    for r in case_type_summary
                    if r["model"] == model and r["case_type"] == case_type
                ),
                None,
            )
            values.append(0.0 if value is None else float(value))
        case_type_overall_series.append((model, values, colors[idx % len(colors)]))
    draw_grouped_bar_svg(case_type_order, case_type_overall_series, "Overall Score by Case Type", output_dir / "case_type_overall_score_by_model.svg")

    case_type_reasoning_series = []
    for idx, model in enumerate(models):
        values = []
        for case_type in case_type_order:
            value = next(
                (
                    r["reasoning_score"]
                    for r in case_type_reasoning_summary
                    if r["model"] == model and r["case_type"] == case_type
                ),
                None,
            )
            values.append(0.0 if value is None else float(value))
        case_type_reasoning_series.append((model, values, colors[idx % len(colors)]))
    draw_grouped_bar_svg(case_type_order, case_type_reasoning_series, "Reasoning Score by Case Type", output_dir / "case_type_reasoning_score_by_model.svg")

    conflict_series = []
    for idx, model in enumerate(models):
        values = []
        for key in conflict_order:
            value = next(
                (
                    r["overall_score"]
                    for r in conflict_summary
                    if r["model"] == model and r["conflict_count_bin"] == key
                ),
                None,
            )
            values.append(None if value is None else float(value))
        conflict_series.append((model, values, colors[idx % len(colors)]))
    draw_line_chart_svg(conflict_order, conflict_series, "Overall Score vs Conflict Count", output_dir / "conflict_overall_score_lines.svg")

    conflict_reasoning_series = []
    for idx, model in enumerate(models):
        values = []
        for key in conflict_order:
            value = next(
                (
                    r["reasoning_score"]
                    for r in conflict_reasoning_summary
                    if r["model"] == model and r["conflict_count_bin"] == key
                ),
                None,
            )
            values.append(None if value is None else float(value))
        conflict_reasoning_series.append((model, values, colors[idx % len(colors)]))
    draw_line_chart_svg(conflict_order, conflict_reasoning_series, "Reasoning Score vs Conflict Count", output_dir / "conflict_reasoning_score_lines.svg")

    step_series = []
    for idx, model in enumerate(models):
        values = []
        for key in step_order:
            value = next(
                (
                    r["overall_score"]
                    for r in step_summary
                    if r["model"] == model and r["reasoning_step_count_bin"] == key
                ),
                None,
            )
            values.append(None if value is None else float(value))
        step_series.append((model, values, colors[idx % len(colors)]))
    draw_line_chart_svg(step_order, step_series, "Overall Score vs Gold Reasoning-Step Bin", output_dir / "reasoning_step_overall_score_lines.svg")

    step_reasoning_series = []
    for idx, model in enumerate(models):
        values = []
        for key in step_order:
            value = next(
                (
                    r["reasoning_score"]
                    for r in step_reasoning_summary
                    if r["model"] == model and r["reasoning_step_count_bin"] == key
                ),
                None,
            )
            values.append(None if value is None else float(value))
        step_reasoning_series.append((model, values, colors[idx % len(colors)]))
    draw_line_chart_svg(step_order, step_reasoning_series, "Reasoning Score vs Gold Reasoning-Step Bin", output_dir / "reasoning_step_reasoning_score_lines.svg")

    article_series = []
    for idx, model in enumerate(models):
        values = []
        for key in article_order:
            value = next(
                (
                    r["overall_score"]
                    for r in article_summary
                    if r["model"] == model and r["triggered_article_count_bin"] == key
                ),
                None,
            )
            values.append(None if value is None else float(value))
        article_series.append((model, values, colors[idx % len(colors)]))
    draw_line_chart_svg(article_order, article_series, "Overall Score vs Triggered-Article Count", output_dir / "triggered_article_count_overall_score_lines.svg")

    middle_series = []
    for idx, model in enumerate(models):
        values = []
        for key in middle_order:
            value = next(
                (
                    r["overall_score"]
                    for r in middle_summary
                    if r["model"] == model and r["triggered_middle_predicate_count_bin"] == key
                ),
                None,
            )
            values.append(None if value is None else float(value))
        middle_series.append((model, values, colors[idx % len(colors)]))
    draw_line_chart_svg(middle_order, middle_series, "Overall Score vs Triggered-Middle Count", output_dir / "triggered_middle_count_overall_score_lines.svg")

    draw_heatmap_svg(bundle_summary, "model", "bundle_name", "overall_score", "Bundle Overall Score Heatmap", output_dir / "bundle_overall_score_heatmap.svg")
    draw_heatmap_svg(bundle_reasoning_summary, "model", "bundle_name", "reasoning_score", "Bundle Reasoning Score Heatmap", output_dir / "bundle_reasoning_score_heatmap.svg")

    error_stages = sorted({row.get("error_stage", "unknown") or "unknown" for row in rows})
    stacked = []
    stage_palette = ["#4C78A8", "#F28E2B", "#E15759", "#76B7B2", "#B07AA1", "#9C755F"]
    for idx, stage in enumerate(error_stages):
        values = [sum(1 for row in rows if row.get("model") == model and (row.get("error_stage", "unknown") or "unknown") == stage) for model in models]
        stacked.append((stage, values, stage_palette[idx % len(stage_palette)]))
    draw_stacked_bar_svg(models, stacked, "Error Stage Distribution by Model", output_dir / "error_stage_stacked.svg")

    summary = {
        "batch_dir": str(batch_dir),
        "case_count": len(rows),
        "model_count": len(models),
        "models": models,
        "output_dir": str(output_dir),
    }
    (output_dir / "plot_summary.txt").write_text(
        "\n".join(
            [
                f"batch_dir={summary['batch_dir']}",
                f"case_count={summary['case_count']}",
                f"model_count={summary['model_count']}",
                f"models={', '.join(summary['models'])}",
                f"output_dir={summary['output_dir']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved SVG evaluation plots to: {output_dir}")


if __name__ == "__main__":
    main()
