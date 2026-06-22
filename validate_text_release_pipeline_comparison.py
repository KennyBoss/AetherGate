#!/usr/bin/env python3
"""Validate a TextPy/SoA release pipeline comparison JSON artifact."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing pipeline comparison JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid pipeline comparison JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return payload


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def as_number(value: Any, label: str, errors: list[str], *, required: bool = True) -> float | None:
    if value is None and not required:
        return None
    if not is_number(value):
        errors.append(f"{label} must be a finite number.")
        return None
    return float(value)


def as_int(value: Any, label: str, errors: list[str], *, required: bool = True) -> int | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer.")
        return None
    return int(value)


def as_bool(value: Any, label: str, errors: list[str], *, required: bool = True) -> bool | None:
    if value is None and not required:
        return None
    if not isinstance(value, bool):
        errors.append(f"{label} must be boolean.")
        return None
    return bool(value)


def as_list(value: Any, label: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{label} must be a list.")
        return []
    return value


def close_enough(left: Any, right: Any, tolerance: float = 1.0e-9) -> bool:
    if left is None and right is None:
        return True
    if is_number(left) and is_number(right):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def delta(candidate: Any, baseline: Any) -> float | int | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def regression_rate(base: float | None, candidate: float | None, *, higher_is_better: bool) -> float | None:
    if base is None or candidate is None:
        return None
    if math.isclose(base, 0.0, abs_tol=1.0e-12):
        return base - candidate if higher_is_better else candidate - base
    if higher_is_better:
        return (base - candidate) / abs(base)
    return (candidate - base) / abs(base)


def positive_regression(base: float | int | None, candidate: float | int | None) -> float | int | None:
    if base is None or candidate is None:
        return None
    return max(0, base - candidate)


def valid_regressed(base: Any, candidate: Any) -> bool:
    return base is True and candidate is not True


def check_sha(value: Any, label: str, errors: list[str], *, required: bool = True) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        errors.append(f"{label} must be a 64-character lowercase SHA256 string.")


def validate_pipeline_row(row: Any, label: str, errors: list[str], *, require_evidence_audit: bool) -> dict[str, Any]:
    if not isinstance(row, dict):
        errors.append(f"{label} must be an object.")
        return {}
    if not isinstance(row.get("path"), str) or not row.get("path"):
        errors.append(f"{label}.path must be a non-empty string.")
    if row.get("name") is not None and not isinstance(row.get("name"), str):
        errors.append(f"{label}.name must be a string or null.")
    stage_count = as_int(row.get("stage_count"), f"{label}.stage_count", errors)
    stage_names = as_list(row.get("stage_names"), f"{label}.stage_names", errors)
    if any(not isinstance(name, str) for name in stage_names):
        errors.append(f"{label}.stage_names entries must be strings.")
    if stage_count is not None and stage_count != len(stage_names):
        errors.append(f"{label}.stage_count must match len({label}.stage_names).")
    shape_errors = as_list(row.get("shape_errors"), f"{label}.shape_errors", errors)
    if any(not isinstance(item, str) for item in shape_errors):
        errors.append(f"{label}.shape_errors entries must be strings.")
    for key in ["release_count", "selected_rank", "memory_prompt_count", "memory_pair_count"]:
        as_int(row.get(key), f"{label}.{key}", errors)
    for key in [
        "loss",
        "accuracy",
        "best_throughput_tokens_s",
        "memory_mean_pair_overlap",
        "memory_min_pair_overlap",
        "memory_max_pair_state_norm_delta",
        "memory_max_pair_state_l2_distance",
        "memory_min_pair_state_cosine_similarity",
    ]:
        as_number(row.get(key), f"{label}.{key}", errors)
    check_sha(row.get("checkpoint_sha256"), f"{label}.checkpoint_sha256", errors)
    check_sha(row.get("archive_sha256"), f"{label}.archive_sha256", errors)
    as_bool(row.get("release_dashboard_audit_valid"), f"{label}.release_dashboard_audit_valid", errors)
    as_int(row.get("release_dashboard_audit_checks"), f"{label}.release_dashboard_audit_checks", errors)
    has_evidence = as_bool(row.get("has_evidence_audit"), f"{label}.has_evidence_audit", errors)
    if require_evidence_audit:
        if has_evidence is not True:
            errors.append(f"{label}.has_evidence_audit must be true.")
        check_sha(row.get("release_evidence_archive_sha256"), f"{label}.release_evidence_archive_sha256", errors)
        check_sha(row.get("release_evidence_ledger_chain_head"), f"{label}.release_evidence_ledger_chain_head", errors)
        as_bool(row.get("release_evidence_audit_valid"), f"{label}.release_evidence_audit_valid", errors)
        as_bool(
            row.get("release_evidence_audit_validation_valid"),
            f"{label}.release_evidence_audit_validation_valid",
            errors,
        )
        as_int(row.get("release_evidence_audit_checks"), f"{label}.release_evidence_audit_checks", errors)
    return row


def compare_expected(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_stages = set(baseline.get("stage_names") or [])
    candidate_stages = set(candidate.get("stage_names") or [])
    return {
        "same_name": baseline.get("name") == candidate.get("name"),
        "same_stage_set": baseline_stages == candidate_stages,
        "missing_stages": sorted(baseline_stages - candidate_stages),
        "added_stages": sorted(candidate_stages - baseline_stages),
        "stage_count_delta": delta(candidate.get("stage_count"), baseline.get("stage_count")),
        "stage_elapsed_s_delta": delta(candidate.get("stage_elapsed_s"), baseline.get("stage_elapsed_s")),
        "release_count_delta": delta(candidate.get("release_count"), baseline.get("release_count")),
        "selected_rank_delta": delta(candidate.get("selected_rank"), baseline.get("selected_rank")),
        "loss_delta": delta(candidate.get("loss"), baseline.get("loss")),
        "accuracy_delta": delta(candidate.get("accuracy"), baseline.get("accuracy")),
        "best_throughput_tokens_s_delta": delta(
            candidate.get("best_throughput_tokens_s"),
            baseline.get("best_throughput_tokens_s"),
        ),
        "loss_regression_rate": regression_rate(baseline.get("loss"), candidate.get("loss"), higher_is_better=False),
        "accuracy_regression": positive_regression(baseline.get("accuracy"), candidate.get("accuracy")),
        "best_throughput_regression_rate": regression_rate(
            baseline.get("best_throughput_tokens_s"),
            candidate.get("best_throughput_tokens_s"),
            higher_is_better=True,
        ),
        "same_checkpoint_sha256": baseline.get("checkpoint_sha256") == candidate.get("checkpoint_sha256"),
        "same_archive_sha256": baseline.get("archive_sha256") == candidate.get("archive_sha256"),
        "dashboard_audit_valid_regressed": valid_regressed(
            baseline.get("release_dashboard_audit_valid"),
            candidate.get("release_dashboard_audit_valid"),
        ),
        "dashboard_audit_checks_delta": delta(
            candidate.get("release_dashboard_audit_checks"),
            baseline.get("release_dashboard_audit_checks"),
        ),
        "dashboard_audit_check_regression": positive_regression(
            baseline.get("release_dashboard_audit_checks"),
            candidate.get("release_dashboard_audit_checks"),
        ),
        "memory_prompt_count_delta": delta(candidate.get("memory_prompt_count"), baseline.get("memory_prompt_count")),
        "memory_pair_count_delta": delta(candidate.get("memory_pair_count"), baseline.get("memory_pair_count")),
        "memory_mean_pair_overlap_delta": delta(
            candidate.get("memory_mean_pair_overlap"),
            baseline.get("memory_mean_pair_overlap"),
        ),
        "memory_min_pair_overlap_delta": delta(
            candidate.get("memory_min_pair_overlap"),
            baseline.get("memory_min_pair_overlap"),
        ),
        "memory_min_pair_state_cosine_similarity_delta": delta(
            candidate.get("memory_min_pair_state_cosine_similarity"),
            baseline.get("memory_min_pair_state_cosine_similarity"),
        ),
        "memory_max_pair_state_l2_distance_delta": delta(
            candidate.get("memory_max_pair_state_l2_distance"),
            baseline.get("memory_max_pair_state_l2_distance"),
        ),
        "memory_max_pair_state_norm_delta_delta": delta(
            candidate.get("memory_max_pair_state_norm_delta"),
            baseline.get("memory_max_pair_state_norm_delta"),
        ),
        "same_has_evidence_audit": baseline.get("has_evidence_audit") == candidate.get("has_evidence_audit"),
        "same_evidence_archive_sha256": (
            baseline.get("release_evidence_archive_sha256") == candidate.get("release_evidence_archive_sha256")
        ),
        "same_evidence_ledger_chain_head": (
            baseline.get("release_evidence_ledger_chain_head") == candidate.get("release_evidence_ledger_chain_head")
        ),
        "evidence_audit_valid_regressed": valid_regressed(
            baseline.get("release_evidence_audit_valid"),
            candidate.get("release_evidence_audit_valid"),
        ),
        "evidence_audit_validation_valid_regressed": valid_regressed(
            baseline.get("release_evidence_audit_validation_valid"),
            candidate.get("release_evidence_audit_validation_valid"),
        ),
        "evidence_audit_checks_delta": delta(
            candidate.get("release_evidence_audit_checks"),
            baseline.get("release_evidence_audit_checks"),
        ),
        "evidence_audit_check_regression": positive_regression(
            baseline.get("release_evidence_audit_checks"),
            candidate.get("release_evidence_audit_checks"),
        ),
        "baseline_shape_errors": baseline.get("shape_errors"),
        "candidate_shape_errors": candidate.get("shape_errors"),
    }


def validate_comparison_values(comparison: Any, expected: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    if not isinstance(comparison, dict):
        errors.append("comparison must be an object.")
        return {}
    for key, expected_value in expected.items():
        if key not in comparison:
            errors.append(f"comparison.{key} is missing.")
            continue
        if not close_enough(comparison.get(key), expected_value):
            errors.append(f"comparison.{key} must equal recomputed value {expected_value!r}.")
    return comparison


def validate(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.comparison)
    payload = load_json(path)
    errors: list[str] = []
    baseline = validate_pipeline_row(
        payload.get("baseline"),
        "baseline",
        errors,
        require_evidence_audit=args.require_evidence_audit,
    )
    candidate = validate_pipeline_row(
        payload.get("candidate"),
        "candidate",
        errors,
        require_evidence_audit=args.require_evidence_audit,
    )
    expected = compare_expected(baseline, candidate)
    comparison = validate_comparison_values(payload.get("comparison"), expected, errors)
    failures = as_list(payload.get("failures"), "failures", errors)
    if any(not isinstance(item, str) for item in failures):
        errors.append("failures entries must be strings.")
    if failures and not args.allow_failures:
        errors.append(f"pipeline comparison has threshold failures: {failures}")
    if not args.allow_shape_errors:
        for label, row in [("baseline", baseline), ("candidate", candidate)]:
            shape_errors = row.get("shape_errors") or []
            if shape_errors:
                errors.append(f"{label}.shape_errors must be empty: {shape_errors}")
        for key in ["baseline_shape_errors", "candidate_shape_errors"]:
            shape_errors = comparison.get(key) if isinstance(comparison, dict) else []
            if shape_errors:
                errors.append(f"comparison.{key} must be empty: {shape_errors}")
    if args.expected_stages is not None:
        for label, row in [("baseline", baseline), ("candidate", candidate)]:
            if row.get("stage_count") != args.expected_stages:
                errors.append(f"{label}.stage_count mismatch: {row.get('stage_count')} != {args.expected_stages}")
    return {
        "valid": not errors,
        "comparison": str(path),
        "baseline_pipeline": baseline.get("path"),
        "candidate_pipeline": candidate.get("path"),
        "baseline_stage_count": baseline.get("stage_count"),
        "candidate_stage_count": candidate.get("stage_count"),
        "same_stage_set": comparison.get("same_stage_set") if isinstance(comparison, dict) else None,
        "stage_count_delta": comparison.get("stage_count_delta") if isinstance(comparison, dict) else None,
        "loss_delta": comparison.get("loss_delta") if isinstance(comparison, dict) else None,
        "memory_mean_pair_overlap_delta": (
            comparison.get("memory_mean_pair_overlap_delta") if isinstance(comparison, dict) else None
        ),
        "same_evidence_archive_sha256": (
            comparison.get("same_evidence_archive_sha256") if isinstance(comparison, dict) else None
        ),
        "same_evidence_ledger_chain_head": (
            comparison.get("same_evidence_ledger_chain_head") if isinstance(comparison, dict) else None
        ),
        "failure_count": len(failures),
        "errors": errors,
    }


def run(args: argparse.Namespace) -> int:
    result = validate(args)
    if args.output_json:
        save_json(args.output_json, result)
    print("TextPy/SoA release pipeline comparison validator")
    print(f"comparison: {result['comparison']}")
    print(f"valid: {result['valid']}")
    print(f"baseline_pipeline: {result['baseline_pipeline']}")
    print(f"candidate_pipeline: {result['candidate_pipeline']}")
    print(f"stages: {result['baseline_stage_count']} -> {result['candidate_stage_count']}")
    print(f"same_stage_set: {result['same_stage_set']}")
    print(f"stage_count_delta: {result['stage_count_delta']}")
    print(f"loss_delta: {result['loss_delta']}")
    print(f"memory_mean_pair_overlap_delta: {result['memory_mean_pair_overlap_delta']}")
    print(f"same_evidence_archive_sha256: {result['same_evidence_archive_sha256']}")
    print(f"same_evidence_ledger_chain_head: {result['same_evidence_ledger_chain_head']}")
    print(f"failures: {result['failure_count']}")
    if args.output_json:
        print(f"Saved JSON: {args.output_json}")
    if not result["valid"]:
        print("")
        print("FAIL")
        for error in result["errors"]:
            print(f"  {error}")
        return 1
    print("")
    print("RELEASE PIPELINE COMPARISON VALIDATION OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a release pipeline comparison JSON artifact.")
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-evidence-audit", action="store_true")
    parser.add_argument("--expected-stages", type=int, default=None)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--allow-shape-errors", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
