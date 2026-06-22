#!/usr/bin/env python3
"""Validate a TextPy/SoA release dashboard history JSON artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DELTA_KEYS = [
    "benchmark_loss",
    "benchmark_accuracy",
    "benchmark_best_throughput_tokens_s",
    "memory_mean_pair_overlap",
    "memory_min_pair_state_cosine_similarity",
    "memory_max_pair_state_l2_distance",
    "audit_failed_count",
]


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


def as_number(
    value: Any,
    label: str,
    errors: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if not is_finite_number(value):
        errors.append(f"{label} must be a finite number.")
        return None
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        errors.append(f"{label} is below minimum {minimum}: {numeric}")
    if maximum is not None and numeric > maximum:
        errors.append(f"{label} is above maximum {maximum}: {numeric}")
    return numeric


def as_int(value: Any, label: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer.")
        return None
    return int(value)


def close_enough(left: Any, right: Any, tolerance: float = 1.0e-9) -> bool:
    if is_finite_number(left) and is_finite_number(right):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


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


def best_by(rows: list[dict[str, Any]], key: str, *, higher_is_better: bool) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row[key])) if higher_is_better else min(rows, key=lambda row: float(row[key]))


def validate_row(row: Any, index: int, errors: list[str]) -> dict[str, Any]:
    if not isinstance(row, dict):
        errors.append(f"dashboards[{index}] must be an object.")
        return {}
    prefix = f"dashboards[{index}]"
    if not isinstance(row.get("name"), str) or not row.get("name"):
        errors.append(f"{prefix}.name must be a non-empty string.")
    if not isinstance(row.get("path"), str) or not row.get("path"):
        errors.append(f"{prefix}.path must be a non-empty string.")
    if not isinstance(row.get("created_at"), str) or not row.get("created_at"):
        errors.append(f"{prefix}.created_at must be a non-empty string.")
    history_index = as_int(row.get("history_index"), f"{prefix}.history_index", errors)
    if history_index is not None and history_index != index + 1:
        errors.append(f"{prefix}.history_index must be {index + 1}: {history_index}")
    as_int(row.get("rank"), f"{prefix}.rank", errors)
    as_int(row.get("release_count"), f"{prefix}.release_count", errors)
    as_number(row.get("benchmark_loss"), f"{prefix}.benchmark_loss", errors, minimum=0.0)
    as_number(
        row.get("benchmark_accuracy"),
        f"{prefix}.benchmark_accuracy",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    as_number(
        row.get("benchmark_best_throughput_tokens_s"),
        f"{prefix}.benchmark_best_throughput_tokens_s",
        errors,
        minimum=0.0,
    )
    as_number(
        row.get("memory_mean_pair_overlap"),
        f"{prefix}.memory_mean_pair_overlap",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    as_number(
        row.get("memory_min_pair_overlap"),
        f"{prefix}.memory_min_pair_overlap",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    as_number(
        row.get("memory_max_pair_state_norm_delta"),
        f"{prefix}.memory_max_pair_state_norm_delta",
        errors,
        minimum=0.0,
    )
    as_number(
        row.get("memory_max_pair_state_l2_distance"),
        f"{prefix}.memory_max_pair_state_l2_distance",
        errors,
        minimum=0.0,
    )
    as_number(
        row.get("memory_min_pair_state_cosine_similarity"),
        f"{prefix}.memory_min_pair_state_cosine_similarity",
        errors,
        minimum=-1.0,
        maximum=1.0,
    )
    as_int(row.get("memory_prompt_count"), f"{prefix}.memory_prompt_count", errors)
    as_int(row.get("memory_pair_count"), f"{prefix}.memory_pair_count", errors)
    as_int(row.get("audit_check_count"), f"{prefix}.audit_check_count", errors)
    as_int(row.get("audit_failed_count"), f"{prefix}.audit_failed_count", errors)
    if row.get("audit_valid") is not True:
        errors.append(f"{prefix}.audit_valid must be true.")
    if row.get("has_state_vectors") is not True:
        errors.append(f"{prefix}.has_state_vectors must be true.")
    validation = row.get("validation")
    if not isinstance(validation, dict):
        errors.append(f"{prefix}.validation must be an object.")
    else:
        if validation.get("valid") is not True:
            errors.append(f"{prefix}.validation.valid must be true.")
        as_int(validation.get("check_count"), f"{prefix}.validation.check_count", errors)
    delta = row.get("delta")
    if not isinstance(delta, dict):
        errors.append(f"{prefix}.delta must be an object.")
    else:
        for key in DELTA_KEYS:
            label = f"{prefix}.delta.{key}"
            if key == "audit_failed_count":
                as_int(delta.get(key), label, errors)
            else:
                as_number(delta.get(key), label, errors)
    return row


def validate_deltas(rows: list[dict[str, Any]], errors: list[str]) -> None:
    previous: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        delta = row.get("delta") if isinstance(row.get("delta"), dict) else {}
        if previous is None:
            for key in DELTA_KEYS:
                if not close_enough(delta.get(key), 0):
                    errors.append(f"dashboards[{index}].delta.{key} must be 0 for the first release.")
        else:
            for key in DELTA_KEYS:
                expected = row[key] - previous[key]
                if not close_enough(delta.get(key), expected):
                    errors.append(
                        f"dashboards[{index}].delta.{key} mismatch: {delta.get(key)} != {expected}"
                    )
        previous = row


def validate_summary(summary: dict[str, Any], rows: list[dict[str, Any]], errors: list[str]) -> None:
    release_count = as_int(summary.get("release_count"), "summary.release_count", errors)
    valid_count = as_int(summary.get("valid_count"), "summary.valid_count", errors)
    if release_count is not None and release_count != len(rows):
        errors.append(f"summary.release_count must match dashboards length: {release_count} != {len(rows)}")
    expected_valid = sum(1 for row in rows if row.get("audit_valid") is True and row.get("validation", {}).get("valid") is True)
    if valid_count is not None and valid_count != expected_valid:
        errors.append(f"summary.valid_count mismatch: {valid_count} != {expected_valid}")
    if not rows:
        return
    latest = rows[-1]
    if summary.get("latest_name") != latest.get("name"):
        errors.append("summary.latest_name must match the last dashboard row.")
    if summary.get("latest_created_at") != latest.get("created_at"):
        errors.append("summary.latest_created_at must match the last dashboard row.")
    best_loss = best_by(rows, "benchmark_loss", higher_is_better=False)
    best_speed = best_by(rows, "benchmark_best_throughput_tokens_s", higher_is_better=True)
    best_memory = best_by(rows, "memory_mean_pair_overlap", higher_is_better=True)
    if summary.get("best_loss_name") != best_loss.get("name"):
        errors.append("summary.best_loss_name does not match the best loss row.")
    if not close_enough(summary.get("best_loss"), best_loss.get("benchmark_loss")):
        errors.append("summary.best_loss does not match the best loss row.")
    if summary.get("best_throughput_name") != best_speed.get("name"):
        errors.append("summary.best_throughput_name does not match the best throughput row.")
    if not close_enough(summary.get("best_throughput_tokens_s"), best_speed.get("benchmark_best_throughput_tokens_s")):
        errors.append("summary.best_throughput_tokens_s does not match the best throughput row.")
    if summary.get("best_memory_name") != best_memory.get("name"):
        errors.append("summary.best_memory_name does not match the best memory row.")
    if not close_enough(summary.get("best_memory_mean_pair_overlap"), best_memory.get("memory_mean_pair_overlap")):
        errors.append("summary.best_memory_mean_pair_overlap does not match the best memory row.")
    expected_mean_loss = sum(float(row["benchmark_loss"]) for row in rows) / len(rows)
    expected_mean_memory = sum(float(row["memory_mean_pair_overlap"]) for row in rows) / len(rows)
    if not close_enough(summary.get("mean_loss"), expected_mean_loss):
        errors.append("summary.mean_loss does not match dashboard rows.")
    if not close_enough(summary.get("mean_memory_overlap"), expected_mean_memory):
        errors.append("summary.mean_memory_overlap does not match dashboard rows.")


def validate_artifacts(base: Path, rows: list[dict[str, Any]], errors: list[str]) -> int:
    checked = 0
    for index, row in enumerate(rows):
        path = resolve_artifact_path(base, row.get("path"))
        if path is None:
            errors.append(f"dashboards[{index}].path is missing.")
            continue
        if not path.exists():
            errors.append(f"Missing dashboard artifact: {path}")
            continue
        try:
            payload = load_json(path)
        except SystemExit as exc:
            errors.append(str(exc))
            continue
        if payload.get("kind") != "text_release_dashboard":
            errors.append(f"Referenced dashboard has unexpected kind: {path}")
        dashboard = payload.get("dashboard") if isinstance(payload.get("dashboard"), dict) else {}
        if dashboard.get("checkpoint_sha256") != row.get("checkpoint_sha256"):
            errors.append(f"Referenced dashboard checkpoint SHA mismatch: {path}")
        checked += 1
    return checked


def validate(args: argparse.Namespace) -> dict[str, Any]:
    history_path = Path(args.history)
    payload = load_json(history_path)
    errors: list[str] = []

    if payload.get("kind") != "text_release_history":
        errors.append("kind must be text_release_history.")
    source_count = as_int(payload.get("source_count"), "source_count", errors)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object.")
        summary = {}
    raw_rows = payload.get("dashboards")
    if not isinstance(raw_rows, list):
        errors.append("dashboards must be a list.")
        raw_rows = []
    if args.expected_releases is not None and len(raw_rows) != args.expected_releases:
        errors.append(f"Expected {args.expected_releases} releases, found {len(raw_rows)}.")
    if source_count is not None and source_count != len(raw_rows):
        errors.append(f"source_count must match dashboards length: {source_count} != {len(raw_rows)}")
    if not raw_rows:
        errors.append("dashboards must contain at least one release.")

    rows = [validate_row(row, index, errors) for index, row in enumerate(raw_rows)]
    if len({row.get("path") for row in rows}) != len(rows):
        errors.append("dashboard paths must be unique.")
    if len({row.get("checkpoint_sha256") for row in rows}) != len(rows) and args.require_unique_checkpoints:
        errors.append("checkpoint SHA values must be unique.")
    if rows != sorted(rows, key=lambda row: (str(row.get("created_at") or ""), str(row.get("name") or ""), str(row.get("path") or ""))):
        errors.append("dashboards must be sorted by created_at, name, and path.")
    validate_deltas(rows, errors)
    validate_summary(summary, rows, errors)
    artifact_checks = validate_artifacts(history_path.parent, rows, errors) if args.require_artifacts else 0

    result = {
        "history": str(history_path),
        "valid": not errors,
        "errors": errors,
        "release_count": len(rows),
        "valid_count": summary.get("valid_count"),
        "best_loss": summary.get("best_loss"),
        "best_throughput_tokens_s": summary.get("best_throughput_tokens_s"),
        "best_memory_mean_pair_overlap": summary.get("best_memory_mean_pair_overlap"),
        "artifact_count": artifact_checks,
    }
    if args.output_json:
        save_json(args.output_json, result)
    return result


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    return f"{numeric:.{digits}f}"


def run(args: argparse.Namespace) -> int:
    result = validate(args)
    print("TextPy/SoA release history validator")
    print(f"history: {result['history']}")
    print(f"valid: {result['valid']}")
    print(f"releases: {result['release_count']}")
    print(f"valid_count: {result['valid_count']}")
    print(f"best_loss: {fmt(result['best_loss'])}")
    print(f"best_throughput_tokens_s: {fmt(result['best_throughput_tokens_s'], 0)}")
    print(f"best_memory_overlap: {fmt(result['best_memory_mean_pair_overlap'])}")
    print(f"artifact_checks: {result['artifact_count']}")
    if args.output_json:
        print("")
        print(f"Saved JSON: {args.output_json}")
    if result["errors"]:
        print("")
        print("FAIL")
        for error in result["errors"]:
            print(f"  {error}")
        return 1
    print("")
    print("RELEASE HISTORY VALIDATION OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a text release dashboard history artifact.")
    parser.add_argument("--history", default="artifacts/releases/text_release_history.json")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--expected-releases", type=int, default=None)
    parser.add_argument("--require-unique-checkpoints", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
