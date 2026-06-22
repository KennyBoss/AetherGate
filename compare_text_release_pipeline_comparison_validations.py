#!/usr/bin/env python3
"""Compare two TextPy/SoA release pipeline comparison validation JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class ReleasePipelineComparisonValidationComparisonError(RuntimeError):
    pass


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleasePipelineComparisonValidationComparisonError(
            f"Missing pipeline comparison validation JSON: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ReleasePipelineComparisonValidationComparisonError(
            f"Invalid pipeline comparison validation JSON {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ReleasePipelineComparisonValidationComparisonError(f"Expected JSON object: {path}")
    return payload


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_int(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer.")
        return 0
    return int(value)


def as_number(value: Any, label: str, errors: list[str]) -> float:
    if not is_number(value):
        errors.append(f"{label} must be a finite number.")
        return 0.0
    return float(value)


def validate_payload(payload: dict[str, Any], path: Path, *, allow_failed_validation: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload.get("valid"), bool):
        errors.append("valid must be boolean.")
    for key in ["comparison", "baseline_pipeline", "candidate_pipeline"]:
        if not isinstance(payload.get(key), str) or not payload.get(key):
            errors.append(f"{key} must be a non-empty string.")
    for key in ["baseline_stage_count", "candidate_stage_count", "stage_count_delta", "failure_count"]:
        as_int(payload.get(key), key, errors)
    for key in ["loss_delta", "memory_mean_pair_overlap_delta"]:
        as_number(payload.get(key), key, errors)
    for key in ["same_stage_set", "same_evidence_archive_sha256", "same_evidence_ledger_chain_head"]:
        if not isinstance(payload.get(key), bool):
            errors.append(f"{key} must be boolean.")
    validation_errors = payload.get("errors")
    if not isinstance(validation_errors, list):
        errors.append("errors must be a list.")
        validation_errors = []
    elif any(not isinstance(item, str) for item in validation_errors):
        errors.append("errors entries must be strings.")
    expected_valid = bool(payload.get("valid")) is (int(payload.get("failure_count") or 0) == 0 and not validation_errors)
    if isinstance(payload.get("valid"), bool) and not expected_valid:
        errors.append("valid must match failure_count and errors.")
    if not allow_failed_validation and payload.get("valid") is not True:
        errors.append(f"pipeline comparison validation is not valid: {path}")
    return errors


def load_valid_validation(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(path)
    errors = validate_payload(payload, path, allow_failed_validation=args.allow_failed_validation)
    if errors:
        raise ReleasePipelineComparisonValidationComparisonError(
            f"Pipeline comparison validation failed for {path}: {errors}"
        )
    return {
        "path": str(path),
        "payload": payload,
        "valid": payload.get("valid") is True,
        "comparison": payload.get("comparison"),
        "baseline_pipeline": payload.get("baseline_pipeline"),
        "candidate_pipeline": payload.get("candidate_pipeline"),
        "baseline_stage_count": int(payload.get("baseline_stage_count") or 0),
        "candidate_stage_count": int(payload.get("candidate_stage_count") or 0),
        "same_stage_set": payload.get("same_stage_set") is True,
        "stage_count_delta": int(payload.get("stage_count_delta") or 0),
        "loss_delta": float(payload.get("loss_delta") or 0.0),
        "memory_mean_pair_overlap_delta": float(payload.get("memory_mean_pair_overlap_delta") or 0.0),
        "same_evidence_archive_sha256": payload.get("same_evidence_archive_sha256") is True,
        "same_evidence_ledger_chain_head": payload.get("same_evidence_ledger_chain_head") is True,
        "failure_count": int(payload.get("failure_count") or 0),
        "errors": as_list(payload.get("errors")),
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "valid": row["valid"],
        "comparison": row["comparison"],
        "baseline_pipeline": row["baseline_pipeline"],
        "candidate_pipeline": row["candidate_pipeline"],
        "baseline_stage_count": row["baseline_stage_count"],
        "candidate_stage_count": row["candidate_stage_count"],
        "same_stage_set": row["same_stage_set"],
        "stage_count_delta": row["stage_count_delta"],
        "loss_delta": row["loss_delta"],
        "memory_mean_pair_overlap_delta": row["memory_mean_pair_overlap_delta"],
        "same_evidence_archive_sha256": row["same_evidence_archive_sha256"],
        "same_evidence_ledger_chain_head": row["same_evidence_ledger_chain_head"],
        "failure_count": row["failure_count"],
        "errors": row["errors"],
    }


def compare_validations(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_errors = set(baseline["errors"])
    candidate_errors = set(candidate["errors"])
    return {
        "same_valid": baseline["valid"] == candidate["valid"],
        "same_comparison": baseline["comparison"] == candidate["comparison"],
        "same_baseline_pipeline": baseline["baseline_pipeline"] == candidate["baseline_pipeline"],
        "same_candidate_pipeline": baseline["candidate_pipeline"] == candidate["candidate_pipeline"],
        "baseline_stage_count_delta": candidate["baseline_stage_count"] - baseline["baseline_stage_count"],
        "candidate_stage_count_delta": candidate["candidate_stage_count"] - baseline["candidate_stage_count"],
        "same_stage_set_changed": baseline["same_stage_set"] != candidate["same_stage_set"],
        "stage_count_delta_delta": candidate["stage_count_delta"] - baseline["stage_count_delta"],
        "loss_delta_delta": candidate["loss_delta"] - baseline["loss_delta"],
        "memory_mean_pair_overlap_delta_delta": (
            candidate["memory_mean_pair_overlap_delta"] - baseline["memory_mean_pair_overlap_delta"]
        ),
        "same_evidence_archive_sha256_changed": (
            baseline["same_evidence_archive_sha256"] != candidate["same_evidence_archive_sha256"]
        ),
        "same_evidence_ledger_chain_head_changed": (
            baseline["same_evidence_ledger_chain_head"] != candidate["same_evidence_ledger_chain_head"]
        ),
        "failure_count_delta": candidate["failure_count"] - baseline["failure_count"],
        "error_count_delta": len(candidate["errors"]) - len(baseline["errors"]),
        "same_errors": baseline_errors == candidate_errors,
        "common_errors": sorted(baseline_errors & candidate_errors),
        "missing_errors": sorted(baseline_errors - candidate_errors),
        "added_errors": sorted(candidate_errors - baseline_errors),
    }


def threshold_failures(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.fail_on_validity_change and not comparison["same_valid"]:
        failures.append("pipeline comparison validation validity changed")
    if args.fail_on_comparison_path_change and not comparison["same_comparison"]:
        failures.append("comparison path changed")
    if args.fail_on_pipeline_path_change:
        if not comparison["same_baseline_pipeline"]:
            failures.append("baseline pipeline path changed")
        if not comparison["same_candidate_pipeline"]:
            failures.append("candidate pipeline path changed")
    if args.fail_on_stage_count_change:
        if comparison["baseline_stage_count_delta"] != 0:
            failures.append(f"baseline stage count changed by {comparison['baseline_stage_count_delta']}")
        if comparison["candidate_stage_count_delta"] != 0:
            failures.append(f"candidate stage count changed by {comparison['candidate_stage_count_delta']}")
        if comparison["stage_count_delta_delta"] != 0:
            failures.append(f"stage_count_delta changed by {comparison['stage_count_delta_delta']}")
    if args.fail_on_stage_set_change and comparison["same_stage_set_changed"]:
        failures.append("same_stage_set verdict changed")
    if args.fail_on_loss_delta_change and abs(comparison["loss_delta_delta"]) > args.max_loss_delta_change:
        failures.append(
            f"loss_delta changed by {comparison['loss_delta_delta']:.6f} > {args.max_loss_delta_change:.6f}"
        )
    if (
        args.fail_on_memory_overlap_delta_change
        and abs(comparison["memory_mean_pair_overlap_delta_delta"]) > args.max_memory_overlap_delta_change
    ):
        failures.append(
            "memory overlap delta changed by "
            f"{comparison['memory_mean_pair_overlap_delta_delta']:.6f} > {args.max_memory_overlap_delta_change:.6f}"
        )
    if args.fail_on_proof_flag_change:
        if comparison["same_evidence_archive_sha256_changed"]:
            failures.append("same_evidence_archive_sha256 verdict changed")
        if comparison["same_evidence_ledger_chain_head_changed"]:
            failures.append("same_evidence_ledger_chain_head verdict changed")
    if args.fail_on_failure_regression and comparison["failure_count_delta"] > args.max_failure_regression:
        failures.append(f"failure regression {comparison['failure_count_delta']} > {args.max_failure_regression}")
    if args.fail_on_error_regression and comparison["error_count_delta"] > args.max_error_regression:
        failures.append(f"error regression {comparison['error_count_delta']} > {args.max_error_regression}")
    if args.fail_on_error_set_change and not comparison["same_errors"]:
        if comparison["missing_errors"]:
            failures.append(f"missing validation errors: {comparison['missing_errors']}")
        if comparison["added_errors"]:
            failures.append(f"added validation errors: {comparison['added_errors']}")
    return failures


def print_validation(label: str, row: dict[str, Any]) -> None:
    print(label)
    print(f"  path: {row['path']}")
    print(f"  comparison: {row['comparison']}")
    print(f"  valid: {row['valid']}")
    print(f"  stages: {row['baseline_stage_count']} -> {row['candidate_stage_count']}")
    print(f"  same_stage_set: {row['same_stage_set']}")
    print(f"  stage_count_delta: {row['stage_count_delta']}")
    print(f"  loss_delta: {row['loss_delta']:.6f}")
    print(f"  memory_mean_pair_overlap_delta: {row['memory_mean_pair_overlap_delta']:.6f}")
    print(f"  proof: archive={row['same_evidence_archive_sha256']} ledger={row['same_evidence_ledger_chain_head']}")
    print(f"  failures: {row['failure_count']}")
    print(f"  errors: {len(row['errors'])}")


def run(args: argparse.Namespace) -> int:
    baseline = load_valid_validation(Path(args.baseline_validation), args)
    candidate = load_valid_validation(Path(args.candidate_validation), args)
    comparison = compare_validations(baseline, candidate)
    failures = threshold_failures(comparison, args)
    payload = {
        "baseline": compact(baseline),
        "candidate": compact(candidate),
        "comparison": comparison,
        "failures": failures,
    }

    print("TextPy/SoA release pipeline comparison validation comparison")
    print("")
    print_validation("baseline", baseline)
    print("")
    print_validation("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  same_valid: {comparison['same_valid']}")
    print(f"  same_comparison: {comparison['same_comparison']}")
    print(f"  same_baseline_pipeline: {comparison['same_baseline_pipeline']}")
    print(f"  same_candidate_pipeline: {comparison['same_candidate_pipeline']}")
    print(f"  baseline_stage_count_delta: {comparison['baseline_stage_count_delta']}")
    print(f"  candidate_stage_count_delta: {comparison['candidate_stage_count_delta']}")
    print(f"  stage_count_delta_delta: {comparison['stage_count_delta_delta']}")
    print(f"  loss_delta_delta: {comparison['loss_delta_delta']:.6f}")
    print(f"  memory_mean_pair_overlap_delta_delta: {comparison['memory_mean_pair_overlap_delta_delta']:.6f}")
    print(f"  failure_count_delta: {comparison['failure_count_delta']}")
    print(f"  error_count_delta: {comparison['error_count_delta']}")

    if args.output_json:
        save_json(args.output_json, payload)
        print("")
        print(f"Saved JSON: {args.output_json}")

    if failures:
        print("")
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("")
    print("RELEASE PIPELINE COMPARISON VALIDATION COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two release pipeline comparison validation JSON artifacts.")
    parser.add_argument("--baseline-validation", required=True)
    parser.add_argument("--candidate-validation", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--allow-failed-validation", action="store_true")
    parser.add_argument("--fail-on-validity-change", action="store_true")
    parser.add_argument("--fail-on-comparison-path-change", action="store_true")
    parser.add_argument("--fail-on-pipeline-path-change", action="store_true")
    parser.add_argument("--fail-on-stage-count-change", action="store_true")
    parser.add_argument("--fail-on-stage-set-change", action="store_true")
    parser.add_argument("--fail-on-loss-delta-change", action="store_true")
    parser.add_argument("--max-loss-delta-change", type=float, default=0.0)
    parser.add_argument("--fail-on-memory-overlap-delta-change", action="store_true")
    parser.add_argument("--max-memory-overlap-delta-change", type=float, default=0.0)
    parser.add_argument("--fail-on-proof-flag-change", action="store_true")
    parser.add_argument("--fail-on-failure-regression", action="store_true")
    parser.add_argument("--max-failure-regression", type=int, default=0)
    parser.add_argument("--fail-on-error-regression", action="store_true")
    parser.add_argument("--max-error-regression", type=int, default=0)
    parser.add_argument("--fail-on-error-set-change", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleasePipelineComparisonValidationComparisonError as exc:
        raise SystemExit(str(exc)) from exc
