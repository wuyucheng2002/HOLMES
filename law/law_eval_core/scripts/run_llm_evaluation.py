from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import error, request

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


ROOT = Path(__file__).resolve().parents[1]

COMPACT_INT_LIST_KEYS = {
    "step_rule_trace",
    "used_rule_numbers",
    "invalid_rule_numbers",
    "gold_rule_trace",
    "trigger_rule_numbers",
    "conflict_rule_numbers",
    "conclusion_rule_numbers",
}


def parse_optional_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_dataset_document(data: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        scoring_target = data.get("scoring_target", {})
        return data["cases"], scoring_target if isinstance(scoring_target, dict) else {}
    raise ValueError("dataset json must be either a list of case objects or an object with a cases list")


def format_compact_json(value: Any, indent: int = 0, current_key: Optional[str] = None) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner_indent = indent + 2
        parts = []
        for key, item in value.items():
            rendered = format_compact_json(item, inner_indent, key)
            parts.append(f'{" " * inner_indent}{json.dumps(key, ensure_ascii=False)}: {rendered}')
        return "{\n" + ",\n".join(parts) + f"\n{' ' * indent}" + "}"

    if isinstance(value, list):
        if not value:
            return "[]"
        if current_key in COMPACT_INT_LIST_KEYS and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return "[" + ", ".join(str(item) for item in value) + "]"
        inner_indent = indent + 2
        parts = [f'{" " * inner_indent}{format_compact_json(item, inner_indent)}' for item in value]
        return "[\n" + ",\n".join(parts) + f"\n{' ' * indent}" + "]"

    return json.dumps(value, ensure_ascii=False)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(format_compact_json(data))
        fh.write("\n")


def extract_json_block(text: str, open_char: str, close_char: str) -> Optional[str]:
    start = text.find(open_char)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    block = extract_json_block(text, "{", "}")
    if not block:
        return None
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_json_array(text: str) -> Optional[List[Any]]:
    block = extract_json_block(text, "[", "]")
    if not block:
        return None
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def alias_variants(alias_map: Optional[Dict[str, List[str]]]) -> List[Tuple[str, str]]:
    variants: List[Tuple[str, str]] = []
    if not alias_map:
        return variants
    for canonical, aliases in alias_map.items():
        seen = {str(canonical).strip()}
        for alias in aliases:
            seen.add(str(alias).strip())
        for alias in seen:
            if alias:
                variants.append((str(canonical), alias))
    variants.sort(key=lambda item: len(item[1]), reverse=True)
    return variants


def find_alias_mentions(text: str, alias_map: Optional[Dict[str, List[str]]]) -> List[str]:
    if not text or not alias_map:
        return []

    lower_text = text.lower()
    normalized_text = normalize_label(text)
    matches: List[Tuple[int, str]] = []
    seen: set[str] = set()

    for canonical, alias in alias_variants(alias_map):
        alias_text = alias.strip()
        alias_lower = alias_text.lower()
        alias_norm = normalize_label(alias_text)
        alias_is_numeric = alias_text.isdigit()
        found_index: Optional[int] = None

        if alias_is_numeric:
            # Bare numbers are too noisy inside long sentences, but allow exact-number answers.
            if lower_text.strip() == alias_lower:
                found_index = 0
        else:
            if alias_lower in lower_text:
                found_index = lower_text.find(alias_lower)
            elif alias_norm and alias_norm in normalized_text:
                found_index = normalized_text.find(alias_norm)

        if found_index is not None and canonical not in seen:
            matches.append((found_index, canonical))
            seen.add(canonical)

    matches.sort(key=lambda item: item[0])
    return [canonical for _index, canonical in matches]


def fallback_structured_parse(question: Dict[str, Any], model_answer: str) -> Optional[Dict[str, Any]]:
    parsed: Dict[str, Any] = {}
    qtype = question.get("question_type")

    strict_fields = question.get("scoring_target", {}).get("strict_fields", list(question.get("answer", {}).keys()))

    if qtype in {"open_structured", "single_choice_structured"}:
        if "triggered_articles" in strict_fields:
            article_hits = find_alias_mentions(model_answer, alias_map_for_field(question, "triggered_articles"))
            if article_hits:
                parsed["triggered_articles"] = article_hits
        if "triggered_middle_concepts" in strict_fields:
            middle_hits = find_alias_mentions(model_answer, alias_map_for_field(question, "triggered_middle_concepts"))
            if middle_hits:
                parsed["triggered_middle_concepts"] = middle_hits
        if "applicable_articles" in strict_fields:
            applicable_hits = find_alias_mentions(model_answer, alias_map_for_field(question, "applicable_articles"))
            if applicable_hits:
                parsed["applicable_articles"] = applicable_hits

    if qtype == "open_structured":
        if "final_result" in strict_fields:
            final_hits = find_alias_mentions(model_answer, alias_map_for_field(question, "final_result"))
            if final_hits:
                parsed["final_result"] = final_hits[0]
        if "conflict_count" in strict_fields:
            conflict_count = parse_int(model_answer)
            if conflict_count is not None:
                parsed["conflict_count"] = conflict_count

    if qtype == "single_choice_structured":
        choice_match = re.search(r"\b([A-Z])\b", model_answer.strip())
        if choice_match:
            parsed["final_result_choice"] = choice_match.group(1)

    return parsed or None


def parse_bool(text: str) -> Optional[bool]:
    lowered = text.lower()
    if re.search(r"\btrue\b", lowered):
        return True
    if re.search(r"\bfalse\b", lowered):
        return False
    if re.search(r"\byes\b", lowered):
        return True
    if re.search(r"\bno\b", lowered):
        return False
    return None


def parse_int(text: str) -> Optional[int]:
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    return int(match.group(0))


def normalize_string_list(items: List[Any]) -> List[str]:
    normalized = [str(item).strip() for item in items if str(item).strip()]
    return sorted(set(normalized))


def normalize_label(value: Any) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    if tokens and tokens[0] == "art":
        tokens = tokens[1:]
    tokens = [token for token in tokens if token not in {"the", "article", "articles"}]
    return "".join(tokens)


def normalize_label_list(items: List[Any]) -> List[str]:
    normalized = [normalize_label(item) for item in items if normalize_label(item)]
    return sorted(set(normalized))


def alias_lookup(alias_map: Optional[Dict[str, List[str]]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not alias_map:
        return lookup
    for canonical, aliases in alias_map.items():
        canonical_norm = normalize_label(canonical)
        lookup[canonical_norm] = canonical_norm
        for alias in aliases:
            alias_norm = normalize_label(alias)
            if alias_norm:
                lookup[alias_norm] = canonical_norm
    return lookup


def canonical_normalize(value: Any, lookup: Optional[Dict[str, str]] = None) -> str:
    normalized = normalize_label(value)
    if lookup and normalized in lookup:
        return lookup[normalized]
    return normalized


def canonical_label_list(items: List[Any], alias_map: Optional[Dict[str, List[str]]] = None) -> List[str]:
    lookup = alias_lookup(alias_map)
    normalized = [canonical_normalize(item, lookup) for item in items if canonical_normalize(item, lookup)]
    return sorted(set(normalized))


def predicted_label_list(items: List[Any], alias_map: Optional[Dict[str, List[str]]] = None) -> List[str]:
    if not alias_map:
        return canonical_label_list(items, alias_map)

    lookup = alias_lookup(alias_map)
    normalized: List[str] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, str):
            direct = canonical_normalize(item, lookup)
            if direct and direct in lookup.values():
                normalized.append(direct)
                continue

            mentions = find_alias_mentions(item, alias_map)
            if mentions:
                normalized.extend(canonical_normalize(mention, lookup) for mention in mentions)
                continue

        direct = canonical_normalize(item, lookup)
        if direct:
            normalized.append(direct)
    return sorted(set(normalized))


def compare_list_field(predicted: Any, gold: List[Any], alias_map: Optional[Dict[str, List[str]]] = None) -> bool:
    gold_labels = canonical_label_list(gold, alias_map)
    if isinstance(predicted, str) and alias_map:
        predicted_labels = sorted(set(canonical_normalize(item, alias_lookup(alias_map)) for item in find_alias_mentions(predicted, alias_map)))
        if predicted_labels:
            return predicted_labels == gold_labels
    return predicted_label_list(coerce_to_list(predicted), alias_map) == gold_labels


def list_field_match_stats(
    predicted: Any,
    gold: List[Any],
    alias_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, int]:
    if isinstance(predicted, str) and alias_map:
        predicted_set = set(canonical_normalize(item, alias_lookup(alias_map)) for item in find_alias_mentions(predicted, alias_map))
    else:
        predicted_set = set(predicted_label_list(coerce_to_list(predicted), alias_map))
    gold_set = set(canonical_label_list(gold, alias_map))
    correct = predicted_set & gold_set
    return {
        "correct_count": len(correct),
        "total_count": len(gold_set),
        "predicted_count": len(predicted_set),
        "missing_count": len(gold_set - predicted_set),
        "extra_count": len(predicted_set - gold_set),
    }


def coerce_to_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [value]


def alias_map_for_field(question: Dict[str, Any], key: str) -> Dict[str, List[str]]:
    scoring_target = question.get("scoring_target", {})
    if key == "triggered_articles":
        return scoring_target.get("article_aliases", {})
    if key == "applicable_articles":
        return scoring_target.get("article_aliases", {})
    if key == "triggered_middle_concepts":
        return scoring_target.get("middle_predicate_aliases", {})
    if key == "final_result":
        return scoring_target.get("final_result_aliases", {})
    return {}


def compare_scalar_field(
    predicted: Any,
    gold: str,
    alias_map: Optional[Dict[str, List[str]]] = None,
) -> bool:
    lookup = alias_lookup(alias_map)
    predicted_norm = canonical_normalize(predicted, lookup)
    gold_norm = canonical_normalize(gold, lookup)
    if predicted_norm == gold_norm:
        return True
    if isinstance(predicted, str):
        predicted_mentions = find_alias_mentions(predicted, alias_map)
        if predicted_mentions:
            return canonical_normalize(predicted_mentions[0], lookup) == gold_norm
        return gold_norm in normalize_label(predicted)
    return False


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_nonempty(values: List[Optional[float]]) -> float:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def lcs_length(left: List[int], right: List[int]) -> int:
    if not left or not right:
        return 0
    dp = [0] * (len(right) + 1)
    for item in left:
        prev = 0
        for index, other in enumerate(right, start=1):
            current = dp[index]
            if item == other:
                dp[index] = prev + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            prev = current
    return dp[-1]


def extract_rule_numbers(text: str) -> List[int]:
    matches = re.findall(r"\[(?:Rule|Rules)\s*:?\s*([0-9]+(?:\s*,\s*[0-9]+)*)\]", text, flags=re.IGNORECASE)
    numbers: List[int] = []
    for match in matches:
        for item in re.split(r"\s*,\s*", match.strip()):
            if item.isdigit():
                numbers.append(int(item))
    return numbers


def parse_reasoned_answer(
    text: str,
    allowed_rule_numbers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    allowed = set(int(item) for item in (allowed_rule_numbers or []))
    answer_match = re.search(r"(?im)^\s*Answer:\s*(.+?)\s*$", text)
    answer_text = answer_match.group(1).strip() if answer_match else ""
    reasoning_match = re.search(r"(?im)^\s*Reasoning:\s*$", text)

    reasoning_text = text
    if answer_match:
        reasoning_text = text[: answer_match.start()]
    if reasoning_match:
        reasoning_text = reasoning_text[reasoning_match.end() :]

    raw_steps: List[List[str]] = []
    current_step: List[str] = []
    parse_mode = "numbered"
    for line in reasoning_text.splitlines():
        step_start = re.match(r"^\s*(\d+)\.\s*(.*)$", line)
        if step_start:
            if current_step:
                raw_steps.append(current_step)
            current_step = [step_start.group(2).rstrip()]
            continue
        if current_step and line.strip():
            current_step.append(line.rstrip())
    if current_step:
        raw_steps.append(current_step)

    if not raw_steps:
        parse_mode = "line_citation"
        for line in reasoning_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.search(r"\[(?:Rule|Rules)\s*:?\s*[0-9]+(?:\s*,\s*[0-9]+)*\]", stripped, flags=re.IGNORECASE):
                raw_steps.append([line.rstrip()])

    step_rule_trace: List[int] = []
    invalid_rule_numbers: List[int] = []
    multi_rule_step_count = 0
    missing_citation_step_count = 0

    for index, lines in enumerate(raw_steps, start=1):
        joined = "\n".join(line for line in lines if line.strip()).strip()
        cited_numbers = extract_rule_numbers(joined)
        cited_unique: List[int] = []
        for number in cited_numbers:
            if number not in cited_unique:
                cited_unique.append(number)
        valid_numbers = [
            number
            for number in cited_unique
            if not allowed or number in allowed
        ]
        invalid_numbers = [
            number
            for number in cited_unique
            if allowed and number not in allowed
        ]
        invalid_rule_numbers.extend(invalid_numbers)

        if len(cited_unique) == 0:
            missing_citation_step_count += 1
        if len(cited_unique) != 1:
            multi_rule_step_count += 1

        if len(valid_numbers) == 1 and len(cited_unique) == 1:
            step_rule_trace.append(valid_numbers[0])

    used_rule_numbers: List[int] = []
    for rule_number in step_rule_trace:
        if rule_number not in used_rule_numbers:
            used_rule_numbers.append(rule_number)

    format_ok = bool(
        answer_text
        and reasoning_match
        and raw_steps
        and missing_citation_step_count == 0
        and multi_rule_step_count == 0
        and not invalid_rule_numbers
    )

    return {
        "answer": answer_text,
        "parse_mode": parse_mode,
        "step_count": len(raw_steps),
        "step_rule_trace": step_rule_trace,
        "used_rule_numbers": used_rule_numbers,
        "invalid_rule_numbers": sorted(set(invalid_rule_numbers)),
        "missing_citation_step_count": missing_citation_step_count,
        "multi_rule_step_count": multi_rule_step_count,
        "format_ok": format_ok,
    }


def coverage_score(predicted_rules: List[int], gold_rules: List[int]) -> Optional[float]:
    gold_set = set(int(item) for item in gold_rules)
    if not gold_set:
        return None
    predicted_set = set(int(item) for item in predicted_rules)
    return len(predicted_set & gold_set) / len(gold_set)


def score_reasoned_answer(question: Dict[str, Any], parsed: Dict[str, Any], gold_final_result: str) -> Dict[str, Any]:
    scoring_target = question.get("scoring_target", {})
    final_result_aliases = alias_map_for_field(question, "final_result")

    predicted_trace = [int(item) for item in parsed.get("step_rule_trace", [])]
    predicted_set = set(predicted_trace)
    gold_trace = [int(item) for item in scoring_target.get("gold_rule_trace", [])]
    key_rules = [int(item) for item in scoring_target.get("key_rule_numbers", [])]
    if not key_rules:
        key_rules = sorted(set(gold_trace))
    trigger_rules = [int(item) for item in scoring_target.get("trigger_rule_numbers", [])]
    conflict_rules = [int(item) for item in scoring_target.get("conflict_rule_numbers", [])]
    conclusion_rules = [int(item) for item in scoring_target.get("conclusion_rule_numbers", [])]

    answer_correct = compare_scalar_field(parsed.get("answer", ""), gold_final_result, final_result_aliases)
    key_rule_hits = len(predicted_set & set(key_rules))
    key_rule_recall = safe_ratio(key_rule_hits, len(set(key_rules)))
    key_rule_precision = safe_ratio(key_rule_hits, len(predicted_set))
    key_rule_f1 = f1_score(key_rule_precision, key_rule_recall)
    order_score = safe_ratio(lcs_length(predicted_trace, gold_trace), len(gold_trace))
    trigger_coverage = coverage_score(predicted_trace, trigger_rules)
    conflict_coverage = coverage_score(predicted_trace, conflict_rules)
    conclusion_coverage = coverage_score(predicted_trace, conclusion_rules)
    stage_coverage_avg = average_nonempty([trigger_coverage, conflict_coverage, conclusion_coverage])
    reasoning_score = 0.4 * key_rule_f1 + 0.3 * order_score + 0.3 * stage_coverage_avg
    overall_score = 0.7 * float(answer_correct) + 0.3 * reasoning_score

    return {
        "answer_correct": answer_correct,
        "field_scores": {"final_result": answer_correct},
        "final_result_correct": answer_correct,
        "key_rule_hits": key_rule_hits,
        "key_rule_recall": key_rule_recall,
        "key_rule_precision": key_rule_precision,
        "key_rule_f1": key_rule_f1,
        "order_score": order_score,
        "trigger_rule_coverage": trigger_coverage,
        "conflict_rule_coverage": conflict_coverage,
        "conclusion_rule_coverage": conclusion_coverage,
        "stage_coverage_avg": stage_coverage_avg,
        "reasoning_score": reasoning_score,
        "overall_score": overall_score,
        "used_rule_count": len(predicted_set),
        "extra_rule_count": len(predicted_set - set(key_rules)),
    }


def apply_dataset_scoring_target(
    question: Dict[str, Any],
    dataset_scoring_target: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not dataset_scoring_target:
        return question
    group = question.get("question_group")
    question_id = question.get("question_id")
    shared_target: Dict[str, Any] = {}
    if group and isinstance(dataset_scoring_target.get(group), dict):
        shared_target.update(dataset_scoring_target[group])
    if question_id and isinstance(dataset_scoring_target.get(question_id), dict):
        shared_target.update(dataset_scoring_target[question_id])
    if not shared_target:
        return question

    merged_question = dict(question)
    merged_scoring_target = dict(shared_target)
    merged_scoring_target.update(question.get("scoring_target", {}))
    merged_question["scoring_target"] = merged_scoring_target
    return merged_question


def parse_list_answer(text: str) -> List[str]:
    as_json = parse_json_array(text)
    if as_json is not None:
        return normalize_string_list(as_json)

    cleaned = text.replace("\n", ",")
    parts = []
    for part in cleaned.split(","):
        candidate = part.strip().strip("-").strip("*").strip()
        if candidate:
            parts.append(candidate)
    return normalize_string_list(parts)


def normalize_choice_value(question: Dict[str, Any], value: Any) -> str:
    raw = str(value).strip()
    if not raw:
        return raw

    raw_upper = raw.upper()
    for option in question.get("options", []):
        if not isinstance(option, dict):
            continue
        option_id = str(option.get("option_id", "")).upper()
        if raw_upper == option_id:
            return option_id

    raw_norm = normalize_label(raw)
    for option in question.get("options", []):
        if not isinstance(option, dict):
            continue
        option_id = str(option.get("option_id", "")).upper()
        candidates = [
            option.get("result", ""),
            option.get("text", ""),
        ]
        if any(raw_norm == normalize_label(candidate) for candidate in candidates):
            return option_id
    return raw_upper


def response_instruction(question: Dict[str, Any]) -> str:
    qtype = question["question_type"]
    if qtype == "reasoned_answer":
        return (
            "First write `Reasoning:`. Then provide the reasoning line by line. "
            "Each reasoning line that uses a rule must contain exactly one rule citation in the format `[Rule n]`. "
            "Do not cite more than one rule in the same line; if multiple rules are needed, split them into multiple lines. "
            "Do not use bracketed labels such as `[Case fact]` or `[Fact]`. "
            "Finally write one separate line beginning with `Answer:`. "
            "After `Answer:`, output only the final legal conclusion, not a full sentence. "
        )
    if qtype == "open_structured":
        answer_keys = list(question.get("answer", {}).keys())
        if answer_keys == ["triggered_articles", "final_result"] or set(answer_keys) == {"triggered_articles", "final_result"}:
            keys = [
                "reasoning",
                "triggered_articles",
                "final_result",
            ]
            return (
                "Return JSON only. Keep exactly these keys: "
                + ", ".join(keys)
                + ". triggered_articles must list all legal articles triggered before conflict resolution. "
                + "final_result must state the final Legal consequences (such as penalties) after conflict resolution. "
                + "Use the shortest exact labels available for articles and final_result, not full sentences. "
                + "In triggered_articles, include only articles that are actually triggered; do not include articles merely discussed, negated, or ruled out in the reasoning. "
                + "Keep reasoning concise: list only the key logical steps needed to justify the answer, preferably within 3 to 6 short steps. "
                + "Strict logical reasoning shall be conducted in strict accordance with the provided statutory texts and case facts, without arbitrary assumptions."
            )
        keys = [
            "reasoning",
            "triggered_middle_concepts",
            "triggered_articles",
            "conflict_count",
            "applicable_articles",
        ]
        return (
            "Return JSON only. Keep exactly these keys: "
            + ", ".join(keys)
            + ". triggered_middle_concepts must list the triggered intermediate concepts. "
            + "triggered_articles must list all legal articles triggered before conflict resolution. "
            + "conflict_count must be the number of stronger/exception conflict relations activated in this case. "
            + "applicable_articles must list the articles that remain applicable after conflict resolution. "
            + "For every list field, choose only from the provided field options and copy the option strings exactly. "
            + "Do not paraphrase, rename, summarize, or output full sentences inside the list fields. "
            + "Return conflict_count as one integer. "
            + "Keep reasoning concise: list only the key logical steps needed to justify the answer, preferably within 3 to 6 short steps. "
            + "Strict logical reasoning shall be conducted in strict accordance with the provided statutory texts and case facts, without arbitrary assumptions."
        )
    if qtype == "single_choice_structured":
        keys = [
            "triggered_middle_concepts",
            "triggered_articles",
            "defeated_articles",
            "applicable_articles",
            "final_result_choice",
            "reasoning",
        ]
        return (
            "Return JSON only. Keep exactly these keys: "
            + ", ".join(keys)
            + ". Here, triggered_middle_concepts refer to concepts that are one level more abstract than atomic actions, and from these intermediate concepts we can derive the triggered articles."
            + " Keep reasoning concise: list only the key logical steps needed to justify the answer, preferably within 3 to 6 short steps."
            + " Strict logical reasoning shall be conducted in strict accordance with statutory texts, without arbitrary assumptions. Use exact predicate/article labels. For final_result_choice, return only the option ID."
        )
    if qtype == "judgment":
        return "Return only True or False."
    if qtype == "multi_select":
        return "Return only a JSON array of selected options."
    if qtype == "integer":
        return "Return only one integer."
    if qtype == "open":
        return "Return a concise explanation."
    return "Return a concise answer."


def build_option_block(question: Dict[str, Any]) -> str:
    option_block = ""
    if question.get("options"):
        options = question["options"]
        if isinstance(options, dict):
            option_lines = []
            for field_name, field_options in options.items():
                rendered = ", ".join(str(item) for item in field_options)
                option_lines.append(f"{field_name}: [{rendered}]")
            option_block = f"\nField options:\n" + "\n".join(option_lines) + "\n"
        else:
            option_lines = "\n".join(format_option(opt) for opt in options)
            option_block = f"\nOptions:\n{option_lines}\n"
    return option_block


def build_prompt(case_item: Dict[str, Any], question: Dict[str, Any]) -> str:
    option_block = build_option_block(question)

    return (
        "You are solving a legal-style logic reasoning task.\n"
        "You must reason strictly from the provided rules and case facts.\n\n"
        "Closed-world assumption: only atomic facts explicitly stated in the case are true; atomic facts not stated in the case should be treated as false unless the case expressly says otherwise.\n\n"
        f"Rules:\n{case_item['rules_text']}\n\n"
        f"Case:\n{case_item['case_text']}\n\n"
        f"Question:\n{question['question']}\n"
        f"Output format requirement:\n{response_instruction(question)}\n"
    )


def build_chat_messages(
    case_item: Dict[str, Any],
    question: Dict[str, Any],
    *,
    prompt_layout: str = "split_bundle_cache",
    openrouter_prompt_cache: bool = False,
    openrouter_prompt_cache_ttl: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if prompt_layout == "single_user":
        return [
            {
                "role": "user",
                "content": build_prompt(case_item, question),
            }
        ]

    if prompt_layout != "split_bundle_cache":
        raise ValueError(f"Unsupported prompt layout: {prompt_layout}")

    option_block = build_option_block(question)
    system_text = (
        "You are solving a legal-style logic reasoning task.\n"
        "You must reason strictly from the provided rules and case facts.\n"
        "Closed-world assumption: only atomic facts explicitly stated in the case are true; "
        "atomic facts not stated in the case should be treated as false unless the case expressly says otherwise.\n\n"
        "Output format requirement:\n"
        f"{response_instruction(question)}\n"
    )

    rules_text = f"Rules:\n{case_item['rules_text']}"
    dynamic_text = f"Case:\n{case_item['case_text']}\n\nQuestion:\n{question['question']}\n{option_block}".rstrip()

    rules_message: Dict[str, Any] = {
        "role": "user",
        "content": rules_text,
    }

    if openrouter_prompt_cache:
        cache_control: Dict[str, Any] = {"type": "ephemeral"}
        if openrouter_prompt_cache_ttl:
            cache_control["ttl"] = openrouter_prompt_cache_ttl
        rules_message["content"] = [
            {
                "type": "text",
                "text": rules_text,
                "cache_control": cache_control,
            }
        ]

    return [
        {
            "role": "system",
            "content": system_text,
        },
        rules_message,
        {
            "role": "user",
            "content": dynamic_text,
        },
    ]


def format_option(option: Any) -> str:
    if isinstance(option, dict):
        option_id = option.get("option_id")
        text = option.get("text", option.get("result"))
        result = option.get("result")
        if option_id and text and result:
            return f"{option_id}. {text} ({result})"
        if option_id and text:
            return f"{option_id}. {text}"
    return f"- {option}"


def dry_run_answer_text(question: Dict[str, Any]) -> str:
    if question.get("question_type") == "reasoned_answer":
        gold_trace = [int(item) for item in question.get("scoring_target", {}).get("gold_rule_trace", [])]
        lines = ["Reasoning:"]
        step_numbers = gold_trace or [1]
        for index, rule_number in enumerate(step_numbers, start=1):
            lines.append(f"{index}. This step follows from the cited rule.")
            lines.append(f"   [Rule {rule_number}]")
        lines.append("")
        lines.append(f"Answer: {question['answer']}")
        return "\n".join(lines)
    return json.dumps(question["answer"], ensure_ascii=False)


def parse_output_text(response_data: Dict[str, Any]) -> str:
    if isinstance(response_data.get("output_text"), str) and response_data["output_text"].strip():
        return response_data["output_text"]

    chunks: List[str] = []
    for item in response_data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
    return "\n".join(chunks).strip()


def parse_chat_completion_text(response_data: Dict[str, Any]) -> str:
    choices = response_data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""

    chunks: List[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            chunks.append(text)
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)
    return "\n".join(chunks).strip()


def flatten_usage_ints(value: Any, prefix: str = "") -> Dict[str, int]:
    flattened: Dict[str, int] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            flattened.update(flatten_usage_ints(item, child_prefix))
    elif isinstance(value, int) and not isinstance(value, bool):
        flattened[prefix] = int(value)
    return flattened


def extract_token_counts(response_data: Dict[str, Any]) -> Dict[str, Any]:
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        return {}

    token_keys = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    ]
    token_counts: Dict[str, Any] = {
        key: usage[key]
        for key in token_keys
        if isinstance(usage.get(key), int)
    }

    for key, value in usage.items():
        if key.endswith("_tokens") and isinstance(value, int):
            token_counts.setdefault(key, value)

    flattened = flatten_usage_ints(usage)
    alias_candidates = {
        "cached_tokens": [
            "cached_tokens",
            "prompt_tokens_details_cached_tokens",
            "input_tokens_details_cached_tokens",
        ],
        "cache_write_tokens": [
            "cache_write_tokens",
            "prompt_tokens_details_cache_write_tokens",
            "input_tokens_details_cache_write_tokens",
        ],
        "reasoning_tokens": [
            "reasoning_tokens",
            "completion_tokens_details_reasoning_tokens",
            "output_tokens_details_reasoning_tokens",
        ],
    }
    for alias, candidates in alias_candidates.items():
        for candidate in candidates:
            if candidate in flattened:
                token_counts.setdefault(alias, flattened[candidate])
                break

    for key, value in flattened.items():
        token_counts.setdefault(key, value)
    return token_counts


def extract_cache_metadata(response_data: Dict[str, Any]) -> Dict[str, Any]:
    headers = response_data.get("_response_headers")
    if not isinstance(headers, dict):
        return {}

    normalized = {str(key).lower(): value for key, value in headers.items()}
    cache_data: Dict[str, Any] = {}

    status = normalized.get("x-openrouter-cache-status")
    if status:
        cache_data["status"] = str(status)

    age = normalized.get("x-openrouter-cache-age", normalized.get("age"))
    if age is not None:
        try:
            cache_data["age"] = int(str(age))
        except ValueError:
            pass

    ttl = normalized.get("x-openrouter-cache-ttl")
    if ttl is not None:
        try:
            cache_data["ttl"] = int(str(ttl))
        except ValueError:
            pass

    generation_id = normalized.get("x-generation-id", normalized.get("x-openrouter-generation-id"))
    if generation_id:
        cache_data["generation_id"] = str(generation_id)

    return cache_data


def compact_raw_response(raw_text: str, response_data: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "raw_text": raw_text,
        "tokens": extract_token_counts(response_data),
    }
    cache_data = extract_cache_metadata(response_data)
    if cache_data:
        compact["cache"] = cache_data
    return compact


def call_responses_api(
    api_key: str,
    model: str,
    prompt: str,
    base_url: str,
    timeout_sec: int,
    reasoning_effort: Optional[str],
    max_retries: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    url = base_url.rstrip("/") + "/responses"
    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    raw = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        req = request.Request(url=url, data=raw, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                data["_response_headers"] = dict(resp.headers.items())
                return parse_output_text(data), data
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if attempt == max_retries - 1 or not should_retry:
                raise RuntimeError(f"API call failed ({exc.code}): {body}") from exc
            time.sleep(2 ** attempt)
        except error.URLError as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"API request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        except http.client.IncompleteRead as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"API request interrupted with incomplete read: {exc}") from exc
            time.sleep(2 ** attempt)
        except (TimeoutError, socket.timeout) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"API request timed out after {timeout_sec}s: {exc}") from exc
            time.sleep(2 ** attempt)

    raise RuntimeError("Unreachable API retry state")


def call_chat_completions_api(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    base_url: str,
    timeout_sec: int,
    enable_thinking: Optional[bool],
    temperature: Optional[float],
    max_completion_tokens: Optional[int],
    openrouter_response_cache: bool,
    openrouter_response_cache_ttl: Optional[int],
    openrouter_http_referer: Optional[str],
    openrouter_x_title: Optional[str],
    max_retries: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_completion_tokens is not None:
        payload["max_tokens"] = max_completion_tokens
    if enable_thinking is not None:
        # Qwen OpenAI-compatible endpoints expose this through SDK extra_body.
        payload["enable_thinking"] = enable_thinking

    raw = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if openrouter_response_cache:
        headers["X-OpenRouter-Cache"] = "true"
    if openrouter_response_cache_ttl is not None:
        headers["X-OpenRouter-Cache-TTL"] = str(openrouter_response_cache_ttl)
    if openrouter_http_referer:
        headers["HTTP-Referer"] = openrouter_http_referer
    if openrouter_x_title:
        headers["X-OpenRouter-Title"] = openrouter_x_title

    for attempt in range(max_retries):
        req = request.Request(url=url, data=raw, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                data["_response_headers"] = dict(resp.headers.items())
                return parse_chat_completion_text(data), data
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if attempt == max_retries - 1 or not should_retry:
                raise RuntimeError(f"API call failed ({exc.code}): {body}") from exc
            time.sleep(2 ** attempt)
        except error.URLError as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"API request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        except http.client.IncompleteRead as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"API request interrupted with incomplete read: {exc}") from exc
            time.sleep(2 ** attempt)
        except (TimeoutError, socket.timeout) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"API request timed out after {timeout_sec}s: {exc}") from exc
            time.sleep(2 ** attempt)

    raise RuntimeError("Unreachable API retry state")


def call_model_api(
    api_format: str,
    api_key: str,
    model: str,
    prompt: str,
    base_url: str,
    timeout_sec: int,
    reasoning_effort: Optional[str],
    enable_thinking: Optional[bool],
    chat_messages: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    max_completion_tokens: Optional[int] = None,
    openrouter_response_cache: bool = False,
    openrouter_response_cache_ttl: Optional[int] = None,
    openrouter_http_referer: Optional[str] = None,
    openrouter_x_title: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    if api_format == "responses":
        return call_responses_api(
            api_key=api_key,
            model=model,
            prompt=prompt,
            base_url=base_url,
            timeout_sec=timeout_sec,
            reasoning_effort=reasoning_effort,
        )
    if api_format == "chat_completions":
        return call_chat_completions_api(
            api_key=api_key,
            model=model,
            messages=chat_messages or [{"role": "user", "content": prompt}],
            base_url=base_url,
            timeout_sec=timeout_sec,
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            openrouter_response_cache=openrouter_response_cache,
            openrouter_response_cache_ttl=openrouter_response_cache_ttl,
            openrouter_http_referer=openrouter_http_referer,
            openrouter_x_title=openrouter_x_title,
        )
    raise ValueError(f"Unsupported api_format: {api_format}")


def evaluate_question(question: Dict[str, Any], model_answer: str) -> Dict[str, Any]:
    qtype = question["question_type"]
    gold = question["answer"]
    result: Dict[str, Any] = {
        "question_id": question["question_id"],
        "question_group": question.get("question_group"),
        "question_type": qtype,
        "parsed_answer": None,
        "correct": False,
        "auto_scored": True,
    }

    if qtype == "reasoned_answer":
        parsed = parse_reasoned_answer(
            model_answer,
            allowed_rule_numbers=question.get("scoring_target", {}).get("allowed_rule_numbers", []),
        )
        result["parsed_answer"] = parsed
        scored = score_reasoned_answer(question, parsed, str(gold))
        result.update(scored)
        result["correct"] = bool(result.get("answer_correct", False) and parsed.get("format_ok", False))
        return result

    if qtype in {"open_structured", "single_choice_structured"}:
        parsed = parse_json_object(model_answer)
        if parsed is None and question.get("question_group") != "diagnostic":
            parsed = fallback_structured_parse(question, model_answer)
        result["parsed_answer"] = parsed
        if parsed is None:
            result["correct"] = False
            result["field_scores"] = {}
            result["strict_correct"] = False
            result["final_result_correct"] = False
            return result

        strict_fields = question.get("scoring_target", {}).get("strict_fields", list(gold.keys()))
        field_scores: Dict[str, bool] = {}
        reasoning_step_stats: Dict[str, Dict[str, int]] = {}
        for key in strict_fields:
            gold_value = gold.get(key)
            pred_value = parsed.get(key)
            aliases = alias_map_for_field(question, key)
            if isinstance(gold_value, list):
                field_scores[key] = compare_list_field(pred_value, gold_value, aliases)
                reasoning_step_stats[key] = list_field_match_stats(pred_value, gold_value, aliases)
            elif key == "final_result_choice":
                field_scores[key] = normalize_choice_value(question, pred_value) == normalize_choice_value(question, gold_value)
            elif isinstance(gold_value, str):
                field_scores[key] = compare_scalar_field(pred_value, gold_value, aliases)
            else:
                field_scores[key] = pred_value == gold_value
        result["field_scores"] = field_scores
        if question.get("question_group") == "main" or question.get("question_id") in {"q_main", "q_main_choice", "q_main_open"}:
            result["reasoning_step_stats"] = reasoning_step_stats
        result["final_result_correct"] = field_scores.get("final_result", field_scores.get("final_result_choice", False))
        result["strict_correct"] = all(field_scores.values())
        result["correct"] = result["strict_correct"]
        return result

    if qtype == "judgment":
        parsed_bool = parse_bool(model_answer)
        result["parsed_answer"] = parsed_bool
        result["correct"] = parsed_bool is not None and parsed_bool == bool(gold)
        return result

    if qtype == "multi_select":
        parsed_list = parse_list_answer(model_answer)
        result["parsed_answer"] = parsed_list
        result["correct"] = normalize_label_list(parsed_list) == normalize_label_list(gold)
        return result

    if qtype == "integer":
        parsed_int = parse_int(model_answer)
        result["parsed_answer"] = parsed_int
        result["correct"] = parsed_int is not None and parsed_int == int(gold)
        return result

    result["auto_scored"] = False
    result["parsed_answer"] = model_answer.strip()
    result["correct"] = False
    return result


def localize_main_error(main_eval: Dict[str, Any]) -> str:
    if not main_eval.get("parsed_answer"):
        return "format_error"
    if "reasoning_score" in main_eval:
        parsed = main_eval.get("parsed_answer", {})
        if not isinstance(parsed, dict) or not parsed.get("format_ok", False):
            return "format_error"
        if main_eval.get("answer_correct", False) and float(main_eval.get("reasoning_score", 0.0)) < 0.35:
            return "potential_shortcut"
        if not main_eval.get("answer_correct", False):
            trigger_coverage = main_eval.get("trigger_rule_coverage")
            conflict_coverage = main_eval.get("conflict_rule_coverage")
            conclusion_coverage = main_eval.get("conclusion_rule_coverage")
            if trigger_coverage is not None and float(trigger_coverage) < 0.5:
                return "rule_triggering"
            if conflict_coverage is not None and float(conflict_coverage) < 0.5:
                return "conflict_resolution"
            if conclusion_coverage is not None and float(conclusion_coverage) < 0.5:
                return "final_rule_application"
            return "final_result"
        return "none"
    if main_eval.get("strict_correct"):
        return "none"
    if main_eval.get("final_result_correct"):
        return "potential_shortcut"

    field_scores = main_eval.get("field_scores", {})
    if not field_scores.get("triggered_articles", True):
        return "article_trigger"
    if not field_scores.get("triggered_middle_concepts", True):
        return "middle_concept"
    if (not field_scores.get("defeated_articles", True)) or (not field_scores.get("applicable_articles", True)):
        return "conflict_resolution"
    return "final_result"


def find_main_evaluation(question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    for item in question_results:
        if item.get("question_group") == "main":
            return item
    for item in question_results:
        if item.get("question_id") in {"q_main", "q_main_choice", "q_main_open"}:
            return item
    return {}


def summarize_question_accuracy(case_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, int]] = {}
    for case_result in case_results:
        for question_result in case_result.get("question_results", []):
            if "correct" not in question_result:
                continue
            question_id = question_result.get("question_id", "unknown")
            item = stats.setdefault(question_id, {"correct": 0, "total": 0})
            item["total"] += 1
            if question_result.get("correct", False):
                item["correct"] += 1
    return {
        question_id: {
            "correct": item["correct"],
            "total": item["total"],
            "accuracy": (item["correct"] / item["total"]) if item["total"] else 0.0,
        }
        for question_id, item in sorted(stats.items())
    }


def summarize_difficulty_levels(case_results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case_result in case_results:
        profile = case_result.get("difficulty_profile", {})
        level = str(profile.get("difficulty_level", "unknown"))
        counts[level] = counts.get(level, 0) + 1
    return dict(sorted(counts.items()))


def build_case_result_entry(case: Dict[str, Any], question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    main_eval = find_main_evaluation(question_results)
    gold_trace = case.get("gold_trace", {})
    return {
        "case_id": case["case_id"],
        "bundle_name": case["bundle_name"],
        "difficulty_profile": case.get("difficulty_profile", {}),
        "gold_trace_summary": {
            "final_result": gold_trace.get("final_result"),
            "final_no_liability": gold_trace.get("final_no_liability"),
            "gold_rule_trace": gold_trace.get("gold_rule_trace", []),
            "trigger_rule_numbers": gold_trace.get("trigger_rule_numbers", []),
            "conflict_rule_numbers": gold_trace.get("conflict_rule_numbers", []),
            "conclusion_rule_numbers": gold_trace.get("conclusion_rule_numbers", []),
        },
        "question_results": [compact_question_result_for_output(item) for item in question_results],
        "summary": {
            "answer_correct": bool(main_eval.get("answer_correct", main_eval.get("final_result_correct", False))),
            "reasoning_score": float(main_eval.get("reasoning_score", 0.0)),
            "overall_score": float(main_eval.get("overall_score", 0.0)),
            "key_rule_f1": float(main_eval.get("key_rule_f1", 0.0)),
            "error_stage": localize_main_error(main_eval),
        },
    }


def build_dataset_summary(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_cases = len(case_results)
    answer_correct_count = sum(1 for item in case_results if item["summary"]["answer_correct"])

    stage_hist: Dict[str, int] = {}
    for item in case_results:
        stage = item["summary"]["error_stage"]
        stage_hist[stage] = stage_hist.get(stage, 0) + 1

    reasoning_score_mean = (
        sum(float(item["summary"]["reasoning_score"]) for item in case_results) / total_cases if total_cases else 0.0
    )
    overall_score_mean = (
        sum(float(item["summary"]["overall_score"]) for item in case_results) / total_cases if total_cases else 0.0
    )
    key_rule_f1_mean = (
        sum(float(item["summary"]["key_rule_f1"]) for item in case_results) / total_cases if total_cases else 0.0
    )
    return {
        "answer_accuracy": (answer_correct_count / total_cases) if total_cases else 0.0,
        "reasoning_score_mean": reasoning_score_mean,
        "overall_score_mean": overall_score_mean,
        "key_rule_f1_mean": key_rule_f1_mean,
        "question_accuracy_by_id": summarize_question_accuracy(case_results),
        "difficulty_level_histogram": summarize_difficulty_levels(case_results),
        "error_stage_histogram": dict(sorted(stage_hist.items())),
    }


def compact_question_result_for_output(result: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "question_id": result.get("question_id"),
        "question_group": result.get("question_group"),
        "question_type": result.get("question_type"),
        "parsed_answer": result.get("parsed_answer"),
        "correct": result.get("correct"),
        "answer_correct": result.get("answer_correct"),
        "key_rule_recall": result.get("key_rule_recall"),
        "key_rule_precision": result.get("key_rule_precision"),
        "key_rule_f1": result.get("key_rule_f1"),
        "order_score": result.get("order_score"),
        "trigger_rule_coverage": result.get("trigger_rule_coverage"),
        "conflict_rule_coverage": result.get("conflict_rule_coverage"),
        "conclusion_rule_coverage": result.get("conclusion_rule_coverage"),
        "stage_coverage_avg": result.get("stage_coverage_avg"),
        "reasoning_score": result.get("reasoning_score"),
        "overall_score": result.get("overall_score"),
        "extra_rule_count": result.get("extra_rule_count"),
        "raw_response": result.get("raw_response"),
    }
    parsed = compact.get("parsed_answer")
    if isinstance(parsed, dict):
        compact["parsed_answer"] = {
            "answer": parsed.get("answer"),
            "parse_mode": parsed.get("parse_mode"),
            "step_count": parsed.get("step_count"),
            "step_rule_trace": parsed.get("step_rule_trace", []),
            "used_rule_numbers": parsed.get("used_rule_numbers", []),
        }
    return compact


def rescore_saved_evaluation_result(
    saved_result: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    dataset_scoring_target: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case_index = {case["case_id"]: case for case in dataset}
    rescored_cases: List[Dict[str, Any]] = []

    for saved_case in saved_result.get("cases", []):
        case_id = saved_case.get("case_id")
        if case_id not in case_index:
            continue
        case = case_index[case_id]
        saved_questions = {
            question_result.get("question_id"): question_result
            for question_result in saved_case.get("question_results", [])
            if isinstance(question_result, dict)
        }
        question_results: List[Dict[str, Any]] = []
        for question in case["questions"]:
            question_for_eval = apply_dataset_scoring_target(question, dataset_scoring_target)
            saved_question = saved_questions.get(question["question_id"], {})
            raw_response = saved_question.get("raw_response", {}) if isinstance(saved_question, dict) else {}
            answer_text = raw_response.get("raw_text", "") if isinstance(raw_response, dict) else ""
            eval_result = evaluate_question(question_for_eval, answer_text)
            eval_result["raw_response"] = raw_response
            if "prompt" in saved_question:
                eval_result["prompt"] = saved_question["prompt"]
            question_results.append(eval_result)
        rescored_cases.append(build_case_result_entry(case, question_results))

    updated = dict(saved_result)
    meta = dict(updated.get("meta", {}))
    meta["rescored_with_current_parser"] = True
    updated["meta"] = meta
    updated["cases"] = rescored_cases
    updated["summary"] = build_dataset_summary(rescored_cases)
    return updated


def evaluate_dataset(
    dataset: List[Dict[str, Any]],
    api_key: str,
    model: str,
    base_url: str,
    api_format: str,
    timeout_sec: int,
    reasoning_effort: Optional[str],
    enable_thinking: Optional[bool],
    max_cases: Optional[int],
    sleep_sec: float,
    dry_run: bool,
    dataset_scoring_target: Optional[Dict[str, Any]] = None,
    include_prompts: bool = False,
    chat_prompt_layout: str = "split_bundle_cache",
    openrouter_prompt_cache: bool = False,
    openrouter_prompt_cache_ttl: Optional[str] = None,
    temperature: Optional[float] = None,
    max_completion_tokens: Optional[int] = None,
    openrouter_response_cache: bool = False,
    openrouter_response_cache_ttl: Optional[int] = None,
    openrouter_http_referer: Optional[str] = None,
    openrouter_x_title: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    selected = dataset[:max_cases] if max_cases else dataset
    case_results: List[Dict[str, Any]] = []

    total_cases = len(selected)
    for case_index, case in enumerate(selected, start=1):
        question_results: List[Dict[str, Any]] = []
        for question in case["questions"]:
            question_for_eval = apply_dataset_scoring_target(question, dataset_scoring_target)
            prompt = build_prompt(case, question_for_eval)
            chat_messages = (
                build_chat_messages(
                    case,
                    question_for_eval,
                    prompt_layout=chat_prompt_layout,
                    openrouter_prompt_cache=openrouter_prompt_cache,
                    openrouter_prompt_cache_ttl=openrouter_prompt_cache_ttl,
                )
                if api_format == "chat_completions"
                else None
            )
            if dry_run:
                answer_text = dry_run_answer_text(question_for_eval)
                response_data = {"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
            else:
                answer_text, response_data = call_model_api(
                    api_format=api_format,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    base_url=base_url,
                    timeout_sec=timeout_sec,
                    reasoning_effort=reasoning_effort,
                    enable_thinking=enable_thinking,
                    chat_messages=chat_messages,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    openrouter_response_cache=openrouter_response_cache,
                    openrouter_response_cache_ttl=openrouter_response_cache_ttl,
                    openrouter_http_referer=openrouter_http_referer,
                    openrouter_x_title=openrouter_x_title,
                )
            eval_result = evaluate_question(question_for_eval, answer_text)
            if include_prompts:
                eval_result["prompt"] = prompt
            eval_result["raw_response"] = compact_raw_response(answer_text, response_data)
            question_results.append(eval_result)
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        case_results.append(build_case_result_entry(case, question_results))
        if progress_callback is not None:
            progress_callback(
                {
                    "case_index": case_index,
                    "case_count": total_cases,
                    "case_id": case.get("case_id"),
                    "bundle_name": case.get("bundle_name"),
                }
            )

    return {
        "meta": {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "base_url": base_url,
            "api_format": api_format,
            "reasoning_effort": reasoning_effort,
            "enable_thinking": enable_thinking,
            "chat_prompt_layout": chat_prompt_layout,
            "openrouter_prompt_cache": openrouter_prompt_cache,
            "openrouter_prompt_cache_ttl": openrouter_prompt_cache_ttl,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "openrouter_response_cache": openrouter_response_cache,
            "openrouter_response_cache_ttl": openrouter_response_cache_ttl,
            "openrouter_http_referer": openrouter_http_referer,
            "openrouter_x_title": openrouter_x_title,
            "case_count": len(case_results),
        },
        "summary": build_dataset_summary(case_results),
        "cases": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM evaluation over generated legal reasoning dataset.")
    parser.add_argument(
        "--dataset-path",
        default="generated/intentional_harm_with_exceptions/dataset.json",
        help="Path to dataset json.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save evaluation results json. Defaults to eval_results/<dataset-name>_<model>.json.",
    )
    parser.add_argument("--model", default="gpt-5.4-mini", help="Model name for Responses API.")
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="Base URL for Responses API.")
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. If omitted, no temperature field is sent.",
    )
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
        help="Optional cache ttl label for OpenRouter prompt caching, such as 5m or 1h when supported by the routed provider.",
    )
    parser.add_argument(
        "--openrouter-response-cache",
        type=parse_optional_bool,
        default=False,
        help="Whether to request OpenRouter response caching headers. Useful mainly for repeated identical requests, not per-case prompt reuse.",
    )
    parser.add_argument(
        "--openrouter-response-cache-ttl",
        type=int,
        default=None,
        help="Optional TTL in seconds for OpenRouter response cache requests.",
    )
    parser.add_argument(
        "--openrouter-http-referer",
        default=None,
        help="Optional HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--openrouter-x-title",
        default=None,
        help="Optional X-Title header for OpenRouter.",
    )
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Do not call API; use gold answers as mock outputs.")
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Store full prompts in the output json for debugging. Off by default to keep results compact.",
    )
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset_path
    dataset, dataset_scoring_target = normalize_dataset_document(load_json(dataset_path))
    load_dotenv()
    api_key = os.getenv(args.api_key_env, "")
    if not args.dry_run and not api_key:
        raise RuntimeError(f"Missing API key in env var: {args.api_key_env}")

    results = evaluate_dataset(
        dataset=dataset,
        api_key=api_key,
        model=args.model,
        base_url=args.api_base,
        api_format=args.api_format,
        timeout_sec=args.timeout_sec,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=args.enable_thinking,
        max_cases=args.max_cases,
        sleep_sec=args.sleep_sec,
        dry_run=args.dry_run,
        dataset_scoring_target=dataset_scoring_target,
        include_prompts=args.include_prompts,
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

    if args.output_path:
        output_path = ROOT / args.output_path
    else:
        dataset_name = dataset_path.parent.name
        safe_model = args.model.replace("/", "_")
        output_path = ROOT / f"eval_results/{dataset_name}_{safe_model}.json"
    dump_json(output_path, results)
    print(f"Saved evaluation report: {output_path}")


if __name__ == "__main__":
    main()
