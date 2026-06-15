from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from run_llm_evaluation import build_chat_messages, build_prompt, load_json, normalize_dataset_document


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "generated"
DEFAULT_OUTPUT_DIR = ROOT / "batch_requests"
CURRENT10_FILE = ROOT / "configs" / "current10_bundles.txt"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_bundle_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"bundle list file not found: {path}")
    bundles: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            bundles.append(text)
    if not bundles:
        raise ValueError(f"bundle list file is empty: {path}")
    return bundles


def sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def parse_bool(text: str) -> bool:
    raw = text.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {text}")


def iter_bundle_dataset_paths(dataset_root: Path, bundle_names: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for bundle_name in bundle_names:
        dataset_path = dataset_root / bundle_name / "dataset.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset not found for bundle '{bundle_name}': {dataset_path}")
        paths.append(dataset_path)
    return paths


def find_q_main(case_item: Dict[str, Any]) -> Dict[str, Any]:
    for question in case_item.get("questions", []):
        if question.get("question_id") == "q_main":
            return question
    raise ValueError(f"case {case_item.get('case_id', '<unknown>')} does not contain q_main")


def build_request_body(
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    temperature: Optional[float],
    enable_thinking: Optional[bool],
    messages_override: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if messages_override is not None:
        messages: List[Dict[str, Any]] = messages_override
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    return body


def make_custom_id(index: int, bundle_name: str, case_id: str, question_id: str) -> str:
    return f"{index:06d}__{sanitize_tag(bundle_name)}__{sanitize_tag(case_id)}__{sanitize_tag(question_id)}"


def collect_requests(
    dataset_paths: Sequence[Path],
    model: str,
    system_prompt: Optional[str],
    temperature: Optional[float],
    enable_thinking: Optional[bool],
    method: str,
    url: str,
    chat_prompt_layout: str,
    openrouter_prompt_cache: bool,
    openrouter_prompt_cache_ttl: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requests: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    counter = 0

    for dataset_path in dataset_paths:
        dataset_data = load_json(dataset_path)
        cases, _scoring_target = normalize_dataset_document(dataset_data)
        bundle_name = str(dataset_data.get("bundle_name") or dataset_path.parent.name)

        for case_item in cases:
            question = find_q_main(case_item)
            prompt = build_prompt(case_item, question)
            messages_override = build_chat_messages(
                case_item,
                question,
                prompt_layout=chat_prompt_layout,
                openrouter_prompt_cache=openrouter_prompt_cache,
                openrouter_prompt_cache_ttl=openrouter_prompt_cache_ttl,
            )
            if system_prompt:
                if messages_override and messages_override[0].get("role") == "system":
                    merged = dict(messages_override[0])
                    merged["content"] = f"{system_prompt}\n\n{messages_override[0].get('content', '')}".strip()
                    messages_override = [merged] + messages_override[1:]
                else:
                    messages_override = [{"role": "system", "content": system_prompt}] + messages_override
            custom_id = make_custom_id(counter + 1, bundle_name, str(case_item.get("case_id", "")), str(question.get("question_id", "q_main")))
            body = build_request_body(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                enable_thinking=enable_thinking,
                messages_override=messages_override,
            )
            request_obj = {
                "custom_id": custom_id,
                "method": method,
                "url": url,
                "body": body,
            }
            requests.append(request_obj)
            index_rows.append(
                {
                    "custom_id": custom_id,
                    "bundle_name": bundle_name,
                    "dataset_path": str(dataset_path.relative_to(ROOT)),
                    "case_id": str(case_item.get("case_id", "")),
                    "question_id": str(question.get("question_id", "q_main")),
                    "difficulty_level": str(case_item.get("difficulty_profile", {}).get("difficulty_level", "")),
                    "case_type": str(case_item.get("difficulty_profile", {}).get("case_type", "")),
                    "gold_final_result": str(question.get("answer", "")),
                }
            )
            counter += 1

    return requests, index_rows


def chunk_requests_by_limits(
    requests: Sequence[Dict[str, Any]],
    max_requests_per_file: int,
    max_bytes_per_file: int,
) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_bytes = 0

    for item in requests:
        line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > 6 * 1024 * 1024:
            raise ValueError(f"single request exceeds 6 MB limit: {item.get('custom_id')}")

        would_exceed_count = current_chunk and len(current_chunk) >= max_requests_per_file
        would_exceed_bytes = current_chunk and (current_bytes + line_bytes > max_bytes_per_file)
        if would_exceed_count or would_exceed_bytes:
            chunks.append(current_chunk)
            current_chunk = []
            current_bytes = 0

        current_chunk.append(item)
        current_bytes += line_bytes

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")


def write_index_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_stem(model: str, preset_name: Optional[str], bundle_names: Sequence[str]) -> str:
    model_tag = sanitize_tag(model)
    if preset_name:
        return f"{model_tag}__{preset_name}"
    if len(bundle_names) == 1:
        return f"{model_tag}__{sanitize_tag(bundle_names[0])}"
    return f"{model_tag}__{len(bundle_names)}bundles"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export current dataset cases into JSONL files for OpenAI-compatible batch API submission."
    )
    parser.add_argument("--model", required=True, help="Model name to place inside every request body.")
    parser.add_argument(
        "--dataset-root",
        default=str(GENERATED_DIR),
        help="Root directory containing generated/<bundle>/dataset.json files.",
    )
    parser.add_argument(
        "--preset",
        choices=["current10"],
        default="current10",
        help="Preset bundle list. Default uses the 10 bundles in the current evaluation setup.",
    )
    parser.add_argument(
        "--bundles",
        nargs="+",
        default=None,
        help="Optional explicit bundle names. If provided, this overrides --preset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where JSONL and index files will be written.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional file stem. If omitted, a name is generated from model and preset/bundle count.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt. If omitted, only the user prompt is included, matching the current evaluation code.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature to include in every request body. If omitted, the request body will not contain temperature.",
    )
    parser.add_argument(
        "--enable-thinking",
        type=parse_bool,
        default=None,
        help="Optional boolean for OpenAI-compatible endpoints that support enable_thinking.",
    )
    parser.add_argument(
        "--chat-prompt-layout",
        default="split_bundle_cache",
        choices=["single_user", "split_bundle_cache"],
        help="How to split chat prompts into messages. split_bundle_cache isolates shared bundle rules for cache-friendly reuse.",
    )
    parser.add_argument(
        "--openrouter-prompt-cache",
        type=parse_bool,
        default=False,
        help="Whether to mark the shared rules message as cacheable for OpenRouter-compatible prompt caching.",
    )
    parser.add_argument(
        "--openrouter-prompt-cache-ttl",
        default=None,
        help="Optional cache ttl label such as 5m or 1h when supported by the routed provider.",
    )
    parser.add_argument(
        "--method",
        default="POST",
        help="HTTP method field for each JSONL row. Default: POST",
    )
    parser.add_argument(
        "--url",
        default="/v1/chat/completions",
        help="Relative API URL for each JSONL row. Default: /v1/chat/completions",
    )
    parser.add_argument(
        "--max-requests-per-file",
        type=int,
        default=50000,
        help="Safety split limit. Default: 50000",
    )
    parser.add_argument(
        "--max-bytes-per-file",
        type=int,
        default=500 * 1024 * 1024,
        help="Safety split limit in bytes. Default: 500 MB",
    )
    args = parser.parse_args()

    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)
    bundle_names = list(args.bundles) if args.bundles else load_bundle_list(CURRENT10_FILE)
    dataset_paths = iter_bundle_dataset_paths(dataset_root, bundle_names)

    requests, index_rows = collect_requests(
        dataset_paths=dataset_paths,
        model=args.model,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        method=args.method,
        url=args.url,
        chat_prompt_layout=args.chat_prompt_layout,
        openrouter_prompt_cache=args.openrouter_prompt_cache,
        openrouter_prompt_cache_ttl=args.openrouter_prompt_cache_ttl,
    )

    chunks = chunk_requests_by_limits(
        requests=requests,
        max_requests_per_file=args.max_requests_per_file,
        max_bytes_per_file=args.max_bytes_per_file,
    )

    output_stem = args.output_stem or default_output_stem(args.model, None if args.bundles else args.preset, bundle_names)

    start = 0
    written_files: List[Path] = []
    for idx, chunk in enumerate(chunks, start=1):
        part_suffix = f"_part{idx:02d}" if len(chunks) > 1 else ""
        jsonl_path = output_dir / f"{output_stem}{part_suffix}.jsonl"
        write_jsonl(jsonl_path, chunk)
        end = start + len(chunk)
        index_path = output_dir / f"{output_stem}{part_suffix}_index.csv"
        write_index_csv(index_path, index_rows[start:end])
        written_files.append(jsonl_path)
        start = end

    print(f"Model: {args.model}")
    print(f"Bundle count: {len(bundle_names)}")
    print(f"Case/request count: {len(requests)}")
    print(f"Output dir: {output_dir}")
    for path in written_files:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
