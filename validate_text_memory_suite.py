#!/usr/bin/env python3
"""Validate a TextPy/SoA hidden-memory prompt suite artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return payload


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def as_int(value: Any, label: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer.")
        return None
    return int(value)


def as_number(
    value: Any,
    label: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if not is_finite_number(value):
        errors.append(f"{label} must be a finite number.")
        return None
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        errors.append(f"{label} is below minimum {minimum}: {numeric}")
    if maximum is not None and numeric > maximum:
        errors.append(f"{label} is above maximum {maximum}: {numeric}")
    return numeric


def resolve_artifact_path(base: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    local_by_name = base / path.name
    if local_by_name.exists():
        return local_by_name
    return path


def csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def check_threshold(
    errors: list[str],
    label: str,
    value: float | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if value is None:
        errors.append(f"{label} is missing, cannot apply threshold.")
        return
    if minimum is not None and value < minimum:
        errors.append(f"{label} failed minimum {minimum}: {value}")
    if maximum is not None and value > maximum:
        errors.append(f"{label} failed maximum {maximum}: {value}")


def validate_prompt_artifacts(
    suite_dir: Path,
    prompt_rows: list[dict[str, Any]],
    *,
    require_state_vectors: bool,
    errors: list[str],
) -> int:
    checked = 0
    for row in prompt_rows:
        prompt = row.get("prompt")
        step_count = as_int(row.get("step_count"), f"prompt {row.get('index')} step_count", errors)
        json_path = resolve_artifact_path(suite_dir, row.get("json"))
        csv_path = resolve_artifact_path(suite_dir, row.get("csv"))
        if json_path is None:
            errors.append(f"Prompt {row.get('index')} is missing a JSON artifact path.")
        elif not json_path.exists():
            errors.append(f"Missing prompt JSON artifact: {json_path}")
        else:
            probe = load_json(json_path)
            summary = probe.get("summary") if isinstance(probe.get("summary"), dict) else {}
            if probe.get("prompt") != prompt:
                errors.append(f"Prompt JSON does not match suite prompt: {json_path}")
            if step_count is not None:
                steps = probe.get("steps")
                if not isinstance(steps, list) or len(steps) != step_count:
                    errors.append(f"Prompt JSON step count mismatch: {json_path}")
                if summary.get("step_count") != step_count:
                    errors.append(f"Prompt JSON summary step count mismatch: {json_path}")
            if require_state_vectors and summary.get("has_state_vectors") is not True:
                errors.append(f"Prompt JSON lacks state vectors: {json_path}")
            checked += 1

        if csv_path is None:
            errors.append(f"Prompt {row.get('index')} is missing a CSV artifact path.")
        elif not csv_path.exists():
            errors.append(f"Missing prompt CSV artifact: {csv_path}")
        elif step_count is not None:
            rows = csv_data_rows(csv_path)
            if rows != step_count:
                errors.append(f"Prompt CSV row count mismatch for {csv_path}: {rows} != {step_count}")
    return checked


def validate(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = Path(args.suite)
    suite_dir = suite_path.parent
    payload = load_json(suite_path)
    errors: list[str] = []

    prompts_raw = payload.get("prompts")
    pairs_raw = payload.get("pairs")
    summary_raw = payload.get("summary")
    prompts = prompts_raw if isinstance(prompts_raw, list) else []
    pairs = pairs_raw if isinstance(pairs_raw, list) else []
    summary = summary_raw if isinstance(summary_raw, dict) else {}
    if not isinstance(prompts_raw, list):
        errors.append("Top-level prompts must be a list.")
    if not isinstance(pairs_raw, list):
        errors.append("Top-level pairs must be a list.")
    if not isinstance(summary_raw, dict):
        errors.append("Top-level summary must be an object.")
    prompt_rows = [row for row in prompts if isinstance(row, dict)]
    pair_rows = [row for row in pairs if isinstance(row, dict)]
    if len(prompt_rows) != len(prompts):
        errors.append("Every prompt row must be an object.")
    if len(pair_rows) != len(pairs):
        errors.append("Every pair row must be an object.")

    prompt_count = len(prompt_rows)
    pair_count = len(pair_rows)
    expected_pairs = args.expected_pairs if args.expected_pairs is not None else prompt_count * (prompt_count - 1) // 2

    recorded_prompt_count = as_int(payload.get("prompt_count"), "payload.prompt_count", errors)
    recorded_pair_count = as_int(payload.get("pair_count"), "payload.pair_count", errors)
    summary_prompt_count = as_int(summary.get("prompt_count"), "summary.prompt_count", errors)
    summary_pair_count = as_int(summary.get("pair_count"), "summary.pair_count", errors)
    top_k = as_int(payload.get("top_k"), "payload.top_k", errors)
    max_tokens = as_int(payload.get("max_tokens"), "payload.max_tokens", errors)

    if recorded_prompt_count is not None and recorded_prompt_count != prompt_count:
        errors.append(f"payload.prompt_count mismatch: {recorded_prompt_count} != {prompt_count}")
    if recorded_pair_count is not None and recorded_pair_count != pair_count:
        errors.append(f"payload.pair_count mismatch: {recorded_pair_count} != {pair_count}")
    if summary_prompt_count is not None and summary_prompt_count != prompt_count:
        errors.append(f"summary.prompt_count mismatch: {summary_prompt_count} != {prompt_count}")
    if summary_pair_count is not None and summary_pair_count != pair_count:
        errors.append(f"summary.pair_count mismatch: {summary_pair_count} != {pair_count}")
    if prompt_count < args.min_prompts:
        errors.append(f"Need at least {args.min_prompts} prompts: {prompt_count}")
    if pair_count < args.min_pairs:
        errors.append(f"Need at least {args.min_pairs} pairs: {pair_count}")
    if args.expected_prompts is not None and prompt_count != args.expected_prompts:
        errors.append(f"Prompt count mismatch: {prompt_count} != {args.expected_prompts}")
    if pair_count != expected_pairs:
        errors.append(f"Pair count mismatch: {pair_count} != {expected_pairs}")
    if top_k is not None and top_k <= 0:
        errors.append(f"payload.top_k must be positive: {top_k}")
    if max_tokens is not None and max_tokens <= 0:
        errors.append(f"payload.max_tokens must be positive: {max_tokens}")

    prompt_by_index: dict[int, dict[str, Any]] = {}
    prompt_names: set[str] = set()
    for offset, row in enumerate(prompt_rows):
        index = as_int(row.get("index"), f"prompts[{offset}].index", errors)
        if index is None:
            continue
        if index in prompt_by_index:
            errors.append(f"Duplicate prompt index: {index}")
        prompt_by_index[index] = row
        if index != offset:
            errors.append(f"Prompt index should match row order: {index} != {offset}")
        name = row.get("name")
        prompt = row.get("prompt")
        if not isinstance(name, str) or not name:
            errors.append(f"Prompt {index} has no stable name.")
        elif name in prompt_names:
            errors.append(f"Duplicate prompt name: {name}")
        else:
            prompt_names.add(name)
        if not isinstance(prompt, str) or not prompt:
            errors.append(f"Prompt {index} has empty text.")
        step_count = as_int(row.get("step_count"), f"prompts[{offset}].step_count", errors)
        if step_count is not None and step_count <= 0:
            errors.append(f"Prompt {index} step_count must be positive: {step_count}")
        as_number(row.get("final_state_norm"), f"prompts[{offset}].final_state_norm", errors, minimum=0.0)
        as_number(
            row.get("final_top_probability"),
            f"prompts[{offset}].final_top_probability",
            errors,
            minimum=0.0,
            maximum=1.0,
        )
        if not isinstance(row.get("final_top_prediction"), str) or not row.get("final_top_prediction"):
            errors.append(f"Prompt {index} has no final_top_prediction.")

    seen_pairs: set[tuple[int, int]] = set()
    pair_vector_flags: list[bool] = []
    for offset, row in enumerate(pair_rows):
        left_index = as_int(row.get("baseline_index"), f"pairs[{offset}].baseline_index", errors)
        right_index = as_int(row.get("candidate_index"), f"pairs[{offset}].candidate_index", errors)
        if left_index is None or right_index is None:
            continue
        if left_index >= right_index:
            errors.append(f"Pair indices should be ordered i<j: {left_index}, {right_index}")
        pair_key = (left_index, right_index)
        if pair_key in seen_pairs:
            errors.append(f"Duplicate pair: {pair_key}")
        seen_pairs.add(pair_key)
        left = prompt_by_index.get(left_index)
        right = prompt_by_index.get(right_index)
        if left is None or right is None:
            errors.append(f"Pair references unknown prompt index: {pair_key}")
            continue
        if row.get("baseline_name") != left.get("name"):
            errors.append(f"Pair {pair_key} baseline_name does not match prompt table.")
        if row.get("candidate_name") != right.get("name"):
            errors.append(f"Pair {pair_key} candidate_name does not match prompt table.")
        if row.get("baseline_prompt") != left.get("prompt"):
            errors.append(f"Pair {pair_key} baseline_prompt does not match prompt table.")
        if row.get("candidate_prompt") != right.get("prompt"):
            errors.append(f"Pair {pair_key} candidate_prompt does not match prompt table.")

        left_steps = as_int(row.get("baseline_step_count"), f"pairs[{offset}].baseline_step_count", errors)
        right_steps = as_int(row.get("candidate_step_count"), f"pairs[{offset}].candidate_step_count", errors)
        shared_steps = as_int(row.get("shared_step_count"), f"pairs[{offset}].shared_step_count", errors)
        if shared_steps is not None and shared_steps <= 0:
            errors.append(f"Pair {pair_key} shared_step_count must be positive: {shared_steps}")
        if left_steps is not None and shared_steps is not None and shared_steps > left_steps:
            errors.append(f"Pair {pair_key} shared_step_count exceeds baseline steps.")
        if right_steps is not None and shared_steps is not None and shared_steps > right_steps:
            errors.append(f"Pair {pair_key} shared_step_count exceeds candidate steps.")

        as_number(row.get("token_match_rate"), f"pairs[{offset}].token_match_rate", errors, minimum=0.0, maximum=1.0)
        as_number(row.get("top_match_rate"), f"pairs[{offset}].top_match_rate", errors, minimum=0.0, maximum=1.0)
        as_number(
            row.get("mean_top_prediction_overlap"),
            f"pairs[{offset}].mean_top_prediction_overlap",
            errors,
            minimum=0.0,
            maximum=1.0,
        )
        as_number(
            row.get("min_top_prediction_overlap"),
            f"pairs[{offset}].min_top_prediction_overlap",
            errors,
            minimum=0.0,
            maximum=1.0,
        )
        as_number(row.get("max_abs_state_norm_delta"), f"pairs[{offset}].max_abs_state_norm_delta", errors, minimum=0.0)
        as_number(row.get("mean_abs_state_norm_delta"), f"pairs[{offset}].mean_abs_state_norm_delta", errors, minimum=0.0)
        as_number(
            row.get("max_abs_state_delta_norm_delta"),
            f"pairs[{offset}].max_abs_state_delta_norm_delta",
            errors,
            minimum=0.0,
        )
        as_number(
            row.get("mean_abs_state_delta_norm_delta"),
            f"pairs[{offset}].mean_abs_state_delta_norm_delta",
            errors,
            minimum=0.0,
        )
        as_number(
            row.get("max_abs_top_probability_delta"),
            f"pairs[{offset}].max_abs_top_probability_delta",
            errors,
            minimum=0.0,
            maximum=1.0,
        )
        as_number(
            row.get("mean_abs_top_probability_delta"),
            f"pairs[{offset}].mean_abs_top_probability_delta",
            errors,
            minimum=0.0,
            maximum=1.0,
        )
        as_number(row.get("baseline_final_state_norm"), f"pairs[{offset}].baseline_final_state_norm", errors, minimum=0.0)
        as_number(row.get("candidate_final_state_norm"), f"pairs[{offset}].candidate_final_state_norm", errors, minimum=0.0)
        has_vectors = row.get("has_state_vectors") is True
        pair_vector_flags.append(has_vectors)
        if has_vectors:
            as_number(row.get("mean_state_l2_distance"), f"pairs[{offset}].mean_state_l2_distance", errors, minimum=0.0)
            as_number(row.get("max_state_l2_distance"), f"pairs[{offset}].max_state_l2_distance", errors, minimum=0.0)
            as_number(
                row.get("mean_state_cosine_similarity"),
                f"pairs[{offset}].mean_state_cosine_similarity",
                errors,
                minimum=-1.0,
                maximum=1.0,
            )
            as_number(
                row.get("min_state_cosine_similarity"),
                f"pairs[{offset}].min_state_cosine_similarity",
                errors,
                minimum=-1.0,
                maximum=1.0,
            )
            as_number(
                row.get("final_state_l2_distance"),
                f"pairs[{offset}].final_state_l2_distance",
                errors,
                minimum=0.0,
                allow_none=True,
            )
            as_number(
                row.get("final_state_cosine_similarity"),
                f"pairs[{offset}].final_state_cosine_similarity",
                errors,
                minimum=-1.0,
                maximum=1.0,
                allow_none=True,
            )

    has_state_vectors = summary.get("has_state_vectors") is True
    if args.require_state_vectors and not has_state_vectors:
        errors.append("summary.has_state_vectors is false, but state vectors are required.")
    if has_state_vectors and not all(pair_vector_flags):
        errors.append("summary.has_state_vectors is true, but at least one pair lacks vector geometry.")

    mean_pair_overlap = as_number(
        summary.get("mean_pair_overlap"),
        "summary.mean_pair_overlap",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    min_pair_overlap = as_number(
        summary.get("min_pair_overlap"),
        "summary.min_pair_overlap",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    max_pair_state_norm_delta = as_number(
        summary.get("max_pair_state_norm_delta"),
        "summary.max_pair_state_norm_delta",
        errors,
        minimum=0.0,
    )
    max_pair_state_l2_distance = as_number(
        summary.get("max_pair_state_l2_distance"),
        "summary.max_pair_state_l2_distance",
        errors,
        minimum=0.0,
        allow_none=not has_state_vectors,
    )
    min_pair_state_cosine_similarity = as_number(
        summary.get("min_pair_state_cosine_similarity"),
        "summary.min_pair_state_cosine_similarity",
        errors,
        minimum=-1.0,
        maximum=1.0,
        allow_none=not has_state_vectors,
    )

    check_threshold(errors, "summary.mean_pair_overlap", mean_pair_overlap, minimum=args.min_mean_pair_overlap)
    check_threshold(errors, "summary.min_pair_overlap", min_pair_overlap, minimum=args.min_pair_overlap)
    check_threshold(
        errors,
        "summary.max_pair_state_norm_delta",
        max_pair_state_norm_delta,
        maximum=args.max_pair_state_norm_delta,
    )
    check_threshold(
        errors,
        "summary.max_pair_state_l2_distance",
        max_pair_state_l2_distance,
        maximum=args.max_pair_state_l2_distance,
    )
    check_threshold(
        errors,
        "summary.min_pair_state_cosine_similarity",
        min_pair_state_cosine_similarity,
        minimum=args.min_pair_state_cosine_similarity,
    )

    prompt_artifacts_checked = 0
    if args.require_prompt_artifacts:
        prompt_artifacts_checked = validate_prompt_artifacts(
            suite_dir,
            prompt_rows,
            require_state_vectors=args.require_state_vectors,
            errors=errors,
        )

    prompts_csv_rows: int | None = None
    pairs_csv_rows: int | None = None
    if args.prompts_csv:
        prompts_csv = Path(args.prompts_csv)
        if not prompts_csv.exists():
            errors.append(f"Missing prompts CSV: {prompts_csv}")
        else:
            prompts_csv_rows = csv_data_rows(prompts_csv)
            if prompts_csv_rows != prompt_count:
                errors.append(f"prompts CSV row count mismatch: {prompts_csv_rows} != {prompt_count}")
    if args.pairs_csv:
        pairs_csv = Path(args.pairs_csv)
        if not pairs_csv.exists():
            errors.append(f"Missing pairs CSV: {pairs_csv}")
        else:
            pairs_csv_rows = csv_data_rows(pairs_csv)
            if pairs_csv_rows != pair_count:
                errors.append(f"pairs CSV row count mismatch: {pairs_csv_rows} != {pair_count}")

    return {
        "valid": not errors,
        "errors": errors,
        "suite": str(suite_path),
        "release": payload.get("release"),
        "checkpoint": payload.get("checkpoint"),
        "manifest": payload.get("manifest"),
        "top_k": top_k,
        "max_tokens": max_tokens,
        "prompt_count": prompt_count,
        "pair_count": pair_count,
        "expected_pair_count": expected_pairs,
        "has_state_vectors": has_state_vectors,
        "mean_pair_overlap": mean_pair_overlap,
        "min_pair_overlap": min_pair_overlap,
        "max_pair_state_norm_delta": max_pair_state_norm_delta,
        "max_pair_state_l2_distance": max_pair_state_l2_distance,
        "min_pair_state_cosine_similarity": min_pair_state_cosine_similarity,
        "prompt_artifacts_checked": prompt_artifacts_checked,
        "prompts_csv_rows": prompts_csv_rows,
        "pairs_csv_rows": pairs_csv_rows,
    }


def run(args: argparse.Namespace) -> None:
    result = validate(args)
    print("TextPy/SoA text memory suite validator")
    print(f"suite: {result['suite']}")
    print(f"valid: {result['valid']}")
    print(f"prompts: {result['prompt_count']}")
    print(f"pairs: {result['pair_count']}")
    print(f"has_state_vectors: {result['has_state_vectors']}")
    if result["mean_pair_overlap"] is not None:
        print(f"mean_pair_overlap: {result['mean_pair_overlap']:.4f}")
    if result["min_pair_overlap"] is not None:
        print(f"min_pair_overlap: {result['min_pair_overlap']:.4f}")
    if result["max_pair_state_norm_delta"] is not None:
        print(f"max_pair_state_norm_delta: {result['max_pair_state_norm_delta']:.4f}")
    if result["max_pair_state_l2_distance"] is not None:
        print(f"max_pair_state_l2_distance: {result['max_pair_state_l2_distance']:.4f}")
    if result["min_pair_state_cosine_similarity"] is not None:
        print(f"min_pair_state_cosine_similarity: {result['min_pair_state_cosine_similarity']:.4f}")
    if result["prompt_artifacts_checked"]:
        print(f"prompt_artifacts_checked: {result['prompt_artifacts_checked']}")
    if result["errors"]:
        print("Errors")
        for error in result["errors"]:
            print(f"  {error}")
    else:
        print("MEMORY SUITE VALIDATION OK")
    if args.output_json:
        save_json(args.output_json, result)
        print("")
        print(f"Saved JSON: {args.output_json}")
    if not result["valid"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a text memory prompt-suite artifact.")
    parser.add_argument("--suite", required=True, help="Path to memory_suite.json.")
    parser.add_argument("--prompts-csv", default=None)
    parser.add_argument("--pairs-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--expected-prompts", type=int, default=None)
    parser.add_argument("--expected-pairs", type=int, default=None)
    parser.add_argument("--min-prompts", type=int, default=2)
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument("--require-state-vectors", action="store_true")
    parser.add_argument("--require-prompt-artifacts", action="store_true")
    parser.add_argument("--min-mean-pair-overlap", type=float, default=None)
    parser.add_argument("--min-pair-overlap", type=float, default=None)
    parser.add_argument("--max-pair-state-norm-delta", type=float, default=None)
    parser.add_argument("--max-pair-state-l2-distance", type=float, default=None)
    parser.add_argument("--min-pair-state-cosine-similarity", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
