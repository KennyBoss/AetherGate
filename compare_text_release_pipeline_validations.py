#!/usr/bin/env python3
"""Compare two TextPy/SoA release pipeline validation JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ReleasePipelineValidationComparisonError(RuntimeError):
    pass


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleasePipelineValidationComparisonError(f"Missing pipeline validation JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleasePipelineValidationComparisonError(f"Invalid pipeline validation JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleasePipelineValidationComparisonError(f"Expected JSON object: {path}")
    return payload


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_int(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer.")
        return 0
    return int(value)


def validate_payload(payload: dict[str, Any], path: Path, *, allow_failed_validation: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload.get("valid"), bool):
        errors.append("valid must be boolean.")
    if not isinstance(payload.get("pipeline"), str) or not payload.get("pipeline"):
        errors.append("pipeline must be a non-empty string.")
    if payload.get("name") is not None and not isinstance(payload.get("name"), str):
        errors.append("name must be a string or null.")
    for key in [
        "release_count",
        "selected_rank",
        "stage_count",
        "memory_prompt_count",
        "memory_pair_count",
        "dashboard_audit_checks",
        "artifact_checks",
    ]:
        as_int(payload.get(key), key, errors)
    if not isinstance(payload.get("has_evidence_audit"), bool):
        errors.append("has_evidence_audit must be boolean.")
    if payload.get("evidence_audit_checks") is not None:
        as_int(payload.get("evidence_audit_checks"), "evidence_audit_checks", errors)
    validation_errors = payload.get("errors")
    if not isinstance(validation_errors, list):
        errors.append("errors must be a list.")
        validation_errors = []
    elif any(not isinstance(item, str) for item in validation_errors):
        errors.append("errors entries must be strings.")
    if isinstance(payload.get("valid"), bool) and payload.get("valid") is not (not validation_errors):
        errors.append("valid must match whether errors is empty.")
    if not allow_failed_validation and payload.get("valid") is not True:
        errors.append(f"pipeline validation is not valid: {path}")
    return errors


def load_valid_validation(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(path)
    errors = validate_payload(payload, path, allow_failed_validation=args.allow_failed_validation)
    if errors:
        raise ReleasePipelineValidationComparisonError(f"Pipeline validation failed for {path}: {errors}")
    return {
        "path": str(path),
        "payload": payload,
        "valid": payload.get("valid") is True,
        "pipeline": payload.get("pipeline"),
        "name": payload.get("name"),
        "release_count": int(payload.get("release_count") or 0),
        "selected_rank": int(payload.get("selected_rank") or 0),
        "stage_count": int(payload.get("stage_count") or 0),
        "memory_prompt_count": int(payload.get("memory_prompt_count") or 0),
        "memory_pair_count": int(payload.get("memory_pair_count") or 0),
        "dashboard_audit_checks": int(payload.get("dashboard_audit_checks") or 0),
        "has_evidence_audit": payload.get("has_evidence_audit") is True,
        "evidence_audit_checks": int(payload.get("evidence_audit_checks") or 0),
        "artifact_checks": int(payload.get("artifact_checks") or 0),
        "errors": as_list(payload.get("errors")),
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "valid": row["valid"],
        "pipeline": row["pipeline"],
        "name": row["name"],
        "release_count": row["release_count"],
        "selected_rank": row["selected_rank"],
        "stage_count": row["stage_count"],
        "memory_prompt_count": row["memory_prompt_count"],
        "memory_pair_count": row["memory_pair_count"],
        "dashboard_audit_checks": row["dashboard_audit_checks"],
        "has_evidence_audit": row["has_evidence_audit"],
        "evidence_audit_checks": row["evidence_audit_checks"],
        "artifact_checks": row["artifact_checks"],
        "errors": row["errors"],
    }


def compare_validations(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_errors = set(baseline["errors"])
    candidate_errors = set(candidate["errors"])
    return {
        "same_valid": baseline["valid"] == candidate["valid"],
        "same_pipeline": baseline["pipeline"] == candidate["pipeline"],
        "same_name": baseline["name"] == candidate["name"],
        "same_has_evidence_audit": baseline["has_evidence_audit"] == candidate["has_evidence_audit"],
        "release_count_delta": candidate["release_count"] - baseline["release_count"],
        "selected_rank_delta": candidate["selected_rank"] - baseline["selected_rank"],
        "stage_count_delta": candidate["stage_count"] - baseline["stage_count"],
        "memory_prompt_count_delta": candidate["memory_prompt_count"] - baseline["memory_prompt_count"],
        "memory_pair_count_delta": candidate["memory_pair_count"] - baseline["memory_pair_count"],
        "dashboard_audit_checks_delta": candidate["dashboard_audit_checks"] - baseline["dashboard_audit_checks"],
        "evidence_audit_checks_delta": candidate["evidence_audit_checks"] - baseline["evidence_audit_checks"],
        "artifact_checks_delta": candidate["artifact_checks"] - baseline["artifact_checks"],
        "error_count_delta": len(candidate["errors"]) - len(baseline["errors"]),
        "same_errors": baseline_errors == candidate_errors,
        "common_errors": sorted(baseline_errors & candidate_errors),
        "missing_errors": sorted(baseline_errors - candidate_errors),
        "added_errors": sorted(candidate_errors - baseline_errors),
    }


def threshold_failures(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.fail_on_validity_change and not comparison["same_valid"]:
        failures.append("pipeline validation validity changed")
    if args.fail_on_pipeline_path_change and not comparison["same_pipeline"]:
        failures.append("pipeline path changed")
    if args.fail_on_name_change and not comparison["same_name"]:
        failures.append("pipeline name changed")
    if args.fail_on_evidence_presence_change and not comparison["same_has_evidence_audit"]:
        failures.append("evidence audit presence changed")
    if args.fail_on_stage_count_change and comparison["stage_count_delta"] != 0:
        failures.append(f"stage_count changed by {comparison['stage_count_delta']}")
    if args.fail_on_memory_prompt_count_change and comparison["memory_prompt_count_delta"] != 0:
        failures.append(f"memory_prompt_count changed by {comparison['memory_prompt_count_delta']}")
    if args.fail_on_memory_pair_count_change and comparison["memory_pair_count_delta"] != 0:
        failures.append(f"memory_pair_count changed by {comparison['memory_pair_count_delta']}")
    if args.fail_on_dashboard_audit_check_regression:
        regression = -comparison["dashboard_audit_checks_delta"]
        if regression > args.max_dashboard_audit_check_regression:
            failures.append(
                f"dashboard audit check regression {regression} > {args.max_dashboard_audit_check_regression}"
            )
    if args.fail_on_evidence_audit_check_regression:
        regression = -comparison["evidence_audit_checks_delta"]
        if regression > args.max_evidence_audit_check_regression:
            failures.append(
                f"evidence audit check regression {regression} > {args.max_evidence_audit_check_regression}"
            )
    if args.fail_on_artifact_check_regression:
        regression = -comparison["artifact_checks_delta"]
        if regression > args.max_artifact_check_regression:
            failures.append(f"artifact check regression {regression} > {args.max_artifact_check_regression}")
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
    print(f"  pipeline: {row['pipeline']}")
    print(f"  name: {row['name']}")
    print(f"  valid: {row['valid']}")
    print(f"  stages: {row['stage_count']}")
    print(f"  memory: prompts={row['memory_prompt_count']} pairs={row['memory_pair_count']}")
    print(f"  dashboard_audit_checks: {row['dashboard_audit_checks']}")
    print(f"  evidence_audit: {row['has_evidence_audit']} checks={row['evidence_audit_checks']}")
    print(f"  artifact_checks: {row['artifact_checks']}")
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

    print("TextPy/SoA release pipeline validation comparison")
    print("")
    print_validation("baseline", baseline)
    print("")
    print_validation("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  same_valid: {comparison['same_valid']}")
    print(f"  same_pipeline: {comparison['same_pipeline']}")
    print(f"  same_name: {comparison['same_name']}")
    print(f"  same_has_evidence_audit: {comparison['same_has_evidence_audit']}")
    print(f"  stage_count_delta: {comparison['stage_count_delta']}")
    print(f"  dashboard_audit_checks_delta: {comparison['dashboard_audit_checks_delta']}")
    print(f"  evidence_audit_checks_delta: {comparison['evidence_audit_checks_delta']}")
    print(f"  artifact_checks_delta: {comparison['artifact_checks_delta']}")
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
    print("RELEASE PIPELINE VALIDATION COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two release pipeline validation JSON artifacts.")
    parser.add_argument("--baseline-validation", required=True)
    parser.add_argument("--candidate-validation", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--allow-failed-validation", action="store_true")
    parser.add_argument("--fail-on-validity-change", action="store_true")
    parser.add_argument("--fail-on-pipeline-path-change", action="store_true")
    parser.add_argument("--fail-on-name-change", action="store_true")
    parser.add_argument("--fail-on-evidence-presence-change", action="store_true")
    parser.add_argument("--fail-on-stage-count-change", action="store_true")
    parser.add_argument("--fail-on-memory-prompt-count-change", action="store_true")
    parser.add_argument("--fail-on-memory-pair-count-change", action="store_true")
    parser.add_argument("--fail-on-dashboard-audit-check-regression", action="store_true")
    parser.add_argument("--max-dashboard-audit-check-regression", type=int, default=0)
    parser.add_argument("--fail-on-evidence-audit-check-regression", action="store_true")
    parser.add_argument("--max-evidence-audit-check-regression", type=int, default=0)
    parser.add_argument("--fail-on-artifact-check-regression", action="store_true")
    parser.add_argument("--max-artifact-check-regression", type=int, default=0)
    parser.add_argument("--fail-on-error-regression", action="store_true")
    parser.add_argument("--max-error-regression", type=int, default=0)
    parser.add_argument("--fail-on-error-set-change", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleasePipelineValidationComparisonError as exc:
        raise SystemExit(str(exc)) from exc
