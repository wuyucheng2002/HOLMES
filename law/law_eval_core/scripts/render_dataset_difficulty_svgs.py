from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def ordered_difficulty_levels(rows: List[Dict[str, Any]]) -> List[str]:
    levels = sorted({str(row["difficulty_level"]) for row in rows})
    tier_levels = sorted(
        [level for level in levels if level.startswith("tier_") and level.split("_")[-1].isdigit()],
        key=lambda item: int(item.split("_")[-1]),
    )
    legacy_order = [level for level in ["basic", "medium", "hard", "very_hard"] if level in levels]
    remaining = [level for level in levels if level not in tier_levels and level not in legacy_order]
    return tier_levels + legacy_order + remaining


def load_dataset_rows(datasets_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset_path in sorted(datasets_root.glob("*/dataset.json")):
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        bundle_name = str(data["bundle_name"])
        for case in data["cases"]:
            difficulty = case["difficulty_profile"]
            rows.append(
                {
                    "bundle_name": bundle_name,
                    "case_id": case["case_id"],
                    "difficulty_level": difficulty["difficulty_level"],
                    "case_type": difficulty.get("case_type", "unknown"),
                    "difficulty_score": float(difficulty.get("difficulty_score", 0)),
                    "reasoning_step_count": float(difficulty.get("reasoning_step_count", 0)),
                    "triggered_article_count": float(difficulty["triggered_article_count"]),
                    "triggered_middle_predicate_count": float(difficulty["triggered_middle_predicate_count"]),
                    "conflict_count": float(difficulty["conflict_count"]),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #222; }',
        '.title { font-size: 20px; font-weight: bold; }',
        '.axis-label { font-size: 14px; }',
        '.tick { font-size: 12px; }',
        '.small { font-size: 11px; }',
        '</style>',
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


def draw_histogram_svg(values: Sequence[float], title: str, xlabel: str, output_path: Path, color: str = "#4C78A8") -> None:
    width, height = 900, 560
    left, right, top, bottom = 80, 40, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_v, max_v = int(min(values)), int(max(values))
    bins = list(range(min_v, max_v + 1))
    counts = [sum(1 for value in values if int(value) == item) for item in bins]
    max_count = max(counts) if counts else 1

    lines = svg_header(width, height)
    draw_text(lines, width / 2, 32, title, "title", "middle")
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')

    for idx, (bin_value, count) in enumerate(zip(bins, counts)):
        bar_w = plot_w / max(len(bins), 1) * 0.8
        gap = plot_w / max(len(bins), 1) * 0.2
        x = left + idx * (bar_w + gap) + gap / 2
        bar_h = 0 if max_count == 0 else plot_h * (count / max_count)
        y = top + plot_h - bar_h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.9"/>')
        draw_text(lines, x + bar_w / 2, top + plot_h + 20, str(bin_value), "tick", "middle")
        draw_text(lines, x + bar_w / 2, y - 6, str(count), "small", "middle")

    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        count = round(max_count * ratio)
        y = top + plot_h - plot_h * ratio
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        draw_text(lines, left - 10, y + 4, str(count), "tick", "end")

    draw_text(lines, width / 2, height - 20, xlabel, "axis-label", "middle")
    draw_text(lines, 20, height / 2, "Case Count", "axis-label", "middle", rotate=-90)
    svg_footer(lines, output_path)


def draw_bar_chart_svg(labels: Sequence[str], values: Sequence[float], title: str, xlabel: str, output_path: Path, color: str = "#59A14F") -> None:
    width, height = 900, 560
    left, right, top, bottom = 80, 40, 70, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_count = max(values) if values else 1

    lines = svg_header(width, height)
    draw_text(lines, width / 2, 32, title, "title", "middle")
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')

    slot_w = plot_w / max(len(labels), 1)
    bar_w = slot_w * 0.72
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * slot_w + (slot_w - bar_w) / 2
        bar_h = 0 if max_count == 0 else plot_h * (value / max_count)
        y = top + plot_h - bar_h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.9"/>')
        draw_text(lines, x + bar_w / 2, top + plot_h + 18, label, "tick", "middle", rotate=-25)
        draw_text(lines, x + bar_w / 2, y - 6, str(int(value)), "small", "middle")

    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        count = round(max_count * ratio)
        y = top + plot_h - plot_h * ratio
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        draw_text(lines, left - 10, y + 4, str(count), "tick", "end")

    draw_text(lines, width / 2, height - 20, xlabel, "axis-label", "middle")
    draw_text(lines, 20, height / 2, "Case Count", "axis-label", "middle", rotate=-90)
    svg_footer(lines, output_path)


def draw_bundle_difficulty_stacked_svg(rows: List[Dict[str, Any]], output_path: Path) -> None:
    levels = ordered_difficulty_levels(rows)
    colors = {
        "basic": "#8CD17D",
        "medium": "#F1CE63",
        "hard": "#E15759",
        "very_hard": "#B07AA1",
        "tier_1": "#8CD17D",
        "tier_2": "#B6D97A",
        "tier_3": "#F1CE63",
        "tier_4": "#F28E2B",
        "tier_5": "#E15759",
    }
    bundles = sorted({row["bundle_name"] for row in rows})
    counts = {bundle: Counter() for bundle in bundles}
    for row in rows:
        counts[row["bundle_name"]][row["difficulty_level"]] += 1

    width, height = 1400, 620
    left, right, top, bottom = 100, 40, 70, 180
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_total = max(sum(counts[bundle][level] for level in levels) for bundle in bundles)
    slot_w = plot_w / max(len(bundles), 1)
    bar_w = slot_w * 0.74

    lines = svg_header(width, height)
    draw_text(lines, width / 2, 32, "Difficulty Distribution by Bundle", "title", "middle")
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')

    for idx, bundle in enumerate(bundles):
        x = left + idx * slot_w + (slot_w - bar_w) / 2
        running_h = 0.0
        for level in levels:
            value = counts[bundle][level]
            rect_h = 0 if max_total == 0 else plot_h * (value / max_total)
            y = top + plot_h - running_h - rect_h
            fill = colors.get(level, "#4C78A8")
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" fill="{fill}"/>')
            running_h += rect_h
        draw_text(lines, x + bar_w / 2, top + plot_h + 16, bundle, "small", "middle", rotate=-35)

    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        count = round(max_total * ratio)
        y = top + plot_h - plot_h * ratio
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        draw_text(lines, left - 10, y + 4, str(count), "tick", "end")

    legend_x = left
    legend_y = height - 120
    for idx, level in enumerate(levels):
        x = legend_x + idx * 160
        fill = colors.get(level, "#4C78A8")
        lines.append(f'<rect x="{x}" y="{legend_y}" width="18" height="18" fill="{fill}"/>')
        draw_text(lines, x + 26, legend_y + 14, level, "tick")

    draw_text(lines, 24, height / 2, "Case Count", "axis-label", "middle", rotate=-90)
    svg_footer(lines, output_path)


def draw_heatmap_svg(rows: List[Dict[str, Any]], output_path: Path) -> None:
    bundles = sorted({row["bundle_name"] for row in rows})
    metrics = [
        ("reasoning_step_count", "Steps"),
        ("triggered_article_count", "Articles"),
        ("triggered_middle_predicate_count", "Middle"),
        ("conflict_count", "Conflicts"),
        ("difficulty_score", "Score"),
    ]
    matrix: List[List[float]] = []
    for bundle in bundles:
        bundle_rows = [row for row in rows if row["bundle_name"] == bundle]
        matrix.append([mean(row[field] for row in bundle_rows) for field, _label in metrics])
    all_values = [value for row in matrix for value in row]
    min_v, max_v = min(all_values), max(all_values)

    width, height = 900, max(700, 120 + len(bundles) * 28)
    left, right, top, bottom = 220, 40, 70, 60
    cell_w = 110
    cell_h = 28

    def heat_color(value: float) -> str:
        if max_v == min_v:
            ratio = 0.5
        else:
            ratio = (value - min_v) / (max_v - min_v)
        red = int(245 - 110 * ratio)
        green = int(250 - 50 * ratio)
        blue = int(255 - 145 * ratio)
        return f"rgb({red},{green},{blue})"

    lines = svg_header(width, height)
    draw_text(lines, width / 2, 32, "Average Difficulty Metrics by Bundle", "title", "middle")
    for col_idx, (_field, label) in enumerate(metrics):
        x = left + col_idx * cell_w + cell_w / 2
        draw_text(lines, x, top - 14, label, "tick", "middle")

    for row_idx, bundle in enumerate(bundles):
        y = top + row_idx * cell_h
        draw_text(lines, left - 10, y + 18, bundle, "small", "end")
        for col_idx, value in enumerate(matrix[row_idx]):
            x = left + col_idx * cell_w
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{heat_color(value)}" stroke="white"/>')
            draw_text(lines, x + cell_w / 2, y + 18, f"{value:.2f}", "small", "middle")
    svg_footer(lines, output_path)


def draw_scatter_svg(rows: List[Dict[str, Any]], output_path: Path) -> None:
    levels = ["basic", "medium", "hard", "very_hard"]
    colors = {
        "basic": "#8CD17D",
        "medium": "#F1CE63",
        "hard": "#E15759",
        "very_hard": "#B07AA1",
    }
    width, height = 900, 620
    left, right, top, bottom = 90, 40, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x, max_x = min(row["conflict_count"] for row in rows), max(row["conflict_count"] for row in rows)
    min_y, max_y = min(row["reasoning_step_count"] for row in rows), max(row["reasoning_step_count"] for row in rows)

    def scale_x(value: float) -> float:
        return left + (0 if max_x == min_x else (value - min_x) / (max_x - min_x) * plot_w)

    def scale_y(value: float) -> float:
        return top + plot_h - (0 if max_y == min_y else (value - min_y) / (max_y - min_y) * plot_h)

    lines = svg_header(width, height)
    draw_text(lines, width / 2, 32, "Conflict Count vs Reasoning Steps", "title", "middle")
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5"/>')

    for value in range(int(min_x), int(max_x) + 1):
        x = scale_x(value)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eee" stroke-width="1"/>')
        draw_text(lines, x, top + plot_h + 20, str(value), "tick", "middle")
    for value in range(int(min_y), int(max_y) + 1):
        y = scale_y(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
        draw_text(lines, left - 10, y + 4, str(value), "tick", "end")

    for row in rows:
        x = scale_x(row["conflict_count"])
        y = scale_y(row["reasoning_step_count"])
        color = colors.get(row["difficulty_level"], "#4C78A8")
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" fill-opacity="0.7"/>')

    legend_x = left + plot_w - 140
    legend_y = top + 10
    for idx, level in enumerate(levels):
        y = legend_y + idx * 24
        lines.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{colors[level]}"/>')
        draw_text(lines, legend_x + 14, y + 4, level, "tick")

    draw_text(lines, width / 2, height - 20, "Conflict Count", "axis-label", "middle")
    draw_text(lines, 24, height / 2, "Reasoning Step Count", "axis-label", "middle", rotate=-90)
    svg_footer(lines, output_path)


def build_bundle_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bundles = sorted({row["bundle_name"] for row in rows})
    difficulty_levels = ordered_difficulty_levels(rows)
    summary_rows: List[Dict[str, Any]] = []
    for bundle in bundles:
        bundle_rows = [row for row in rows if row["bundle_name"] == bundle]
        difficulty_counts = Counter(row["difficulty_level"] for row in bundle_rows)
        case_type_counts = Counter(row["case_type"] for row in bundle_rows)
        row = {
            "bundle_name": bundle,
            "case_count": len(bundle_rows),
            "liability_only": case_type_counts["liability_only"],
            "liability_with_exception": case_type_counts["liability_with_exception"],
            "avg_reasoning_step_count": round(mean(row["reasoning_step_count"] for row in bundle_rows), 2),
            "avg_triggered_article_count": round(mean(row["triggered_article_count"] for row in bundle_rows), 2),
            "avg_triggered_middle_predicate_count": round(mean(row["triggered_middle_predicate_count"] for row in bundle_rows), 2),
            "avg_conflict_count": round(mean(row["conflict_count"] for row in bundle_rows), 2),
            "avg_difficulty_score": round(mean(row["difficulty_score"] for row in bundle_rows), 2),
        }
        for level in difficulty_levels:
            row[f"difficulty_{level}"] = difficulty_counts[level]
        summary_rows.append(row)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render difficulty visualizations as SVG without external plotting libraries.")
    parser.add_argument("--datasets-root", default="generated", help="Root directory containing <bundle>/dataset.json files.")
    parser.add_argument("--output-dir", default="eval_results/dataset_difficulty_svgs", help="Output directory for SVG charts and source CSV files.")
    args = parser.parse_args()

    datasets_root = resolve_path(args.datasets_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dataset_rows(datasets_root)
    if not rows:
        raise FileNotFoundError(f"No dataset.json files found under {datasets_root}")

    write_csv(output_dir / "plot_source_case_rows.csv", rows)
    write_csv(output_dir / "plot_source_bundle_summary.csv", build_bundle_summary_rows(rows))

    draw_histogram_svg([row["conflict_count"] for row in rows], "Conflict Count Distribution", "Conflict Count", output_dir / "conflict_count_histogram.svg")
    draw_histogram_svg([row["reasoning_step_count"] for row in rows], "Reasoning Step Distribution", "Reasoning Step Count", output_dir / "reasoning_step_histogram.svg", color="#F28E2B")
    draw_histogram_svg([row["triggered_article_count"] for row in rows], "Triggered Article Count Distribution", "Triggered Article Count", output_dir / "triggered_article_histogram.svg", color="#E15759")
    draw_histogram_svg([row["triggered_middle_predicate_count"] for row in rows], "Triggered Middle-Concept Count Distribution", "Triggered Middle-Concept Count", output_dir / "triggered_middle_histogram.svg", color="#76B7B2")
    difficulty_counts = Counter(row["difficulty_level"] for row in rows)
    difficulty_order = ordered_difficulty_levels(rows)
    draw_bar_chart_svg(difficulty_order, [difficulty_counts[level] for level in difficulty_order], "Difficulty Level Distribution", "Difficulty Level", output_dir / "difficulty_level_bar.svg", color="#59A14F")
    case_type_counts = Counter(row["case_type"] for row in rows)
    draw_bar_chart_svg(["liability_only", "liability_with_exception"], [case_type_counts[level] for level in ["liability_only", "liability_with_exception"]], "Case Type Distribution", "Case Type", output_dir / "case_type_bar.svg", color="#9C755F")
    draw_bundle_difficulty_stacked_svg(rows, output_dir / "bundle_difficulty_stacked.svg")
    draw_heatmap_svg(rows, output_dir / "bundle_metric_heatmap.svg")
    draw_scatter_svg(rows, output_dir / "conflict_vs_steps_scatter.svg")

    summary = {
        "case_count": len(rows),
        "bundle_count": len({row["bundle_name"] for row in rows}),
        "output_dir": str(output_dir),
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
