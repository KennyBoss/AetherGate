#!/usr/bin/env python3
"""Compare two TextPy/SoA release evidence audit validation JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ReleaseEvidenceAuditValidationComparisonError(RuntimeError):
    pass


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseEvidenceAuditValidationComparisonError(f"Missing audit validation JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceAuditValidationComparisonError(f"Invalid audit validation JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseEvidenceAuditValidationComparisonError(f"Expected JSON object: {path}")
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
    if not isinstance(payload.get("audit"), str) or not payload.get("audit"):
        errors.append("audit must be a non-empty string.")
    if payload.get("release_name") is not None and not isinstance(payload.get("release_name"), str):
        errors.append("release_name must be a string or null.")
    failed_check_count = as_int(payload.get("failed_check_count"), "failed_check_count", errors)
    failure_count = as_int(payload.get("failure_count"), "failure_count", errors)
    for key in ["check_count", "artifact_check_count", "source_artifact_checks", "source_count"]:
        as_int(payload.get(key), key, errors)
    validation_errors = payload.get("errors")
    if not isinstance(validation_errors, list):
        errors.append("errors must be a list.")
        validation_errors = []
    elif any(not isinstance(item, str) for item in validation_errors):
        errors.append("errors entries must be strings.")
    expected_valid = failed_check_count == 0 and failure_count == 0 and not validation_errors
    if isinstance(payload.get("valid"), bool) and payload.get("valid") is not expected_valid:
        errors.append("valid must match failed checks, failures, and errors.")
    if not allow_failed_validation and payload.get("valid") is not True:
        errors.append(f"audit validation is not valid: {path}")
    return errors


def load_valid_validation(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(path)
    errors = validate_payload(payload, path, allow_failed_validation=args.allow_failed_validation)
    if errors:
        raise ReleaseEvidenceAuditValidationComparisonError(f"Audit validation failed for {path}: {errors}")
    return {
        "path": str(path),
        "payload": payload,
        "valid": payload.get("valid") is True,
        "audit": payload.get("audit"),
        "release_name": payload.get("release_name"),
        "check_count": int(payload.get("check_count") or 0),
        "failed_check_count": int(payload.get("failed_check_count") or 0),
        "failure_count": int(payload.get("failure_count") or 0),
        "artifact_check_count": int(payload.get("artifact_check_count") or 0),
        "source_artifact_checks": int(payload.get("source_artifact_checks") or 0),
        "source_count": int(payload.get("source_count") or 0),
        "errors": as_list(payload.get("errors")),
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "valid": row["valid"],
        "audit": row["audit"],
        "release_name": row["release_name"],
        "check_count": row["check_count"],
        "failed_check_count": row["failed_check_count"],
        "failure_count": row["failure_count"],
        "artifact_check_count": row["artifact_check_count"],
        "source_artifact_checks": row["source_artifact_checks"],
        "source_count": row["source_count"],
        "errors": row["errors"],
    }


def compare_validations(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_errors = set(baseline["errors"])
    candidate_errors = set(candidate["errors"])
    return {
        "same_valid": baseline["valid"] == candidate["valid"],
        "same_audit": baseline["audit"] == candidate["audit"],
        "same_release_name": baseline["release_name"] == candidate["release_name"],
        "check_count_delta": candidate["check_count"] - baseline["check_count"],
        "failed_check_count_delta": candidate["failed_check_count"] - baseline["failed_check_count"],
        "failure_count_delta": candidate["failure_count"] - baseline["failure_count"],
        "artifact_check_count_delta": candidate["artifact_check_count"] - baseline["artifact_check_count"],
        "source_artifact_checks_delta": candidate["source_artifact_checks"] - baseline["source_artifact_checks"],
        "source_count_delta": candidate["source_count"] - baseline["source_count"],
        "error_count_delta": len(candidate["errors"]) - len(baseline["errors"]),
        "same_errors": baseline_errors == candidate_errors,
        "common_errors": sorted(baseline_errors & candidate_errors),
        "missing_errors": sorted(baseline_errors - candidate_errors),
        "added_errors": sorted(candidate_errors - baseline_errors),
    }


def threshold_failures(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.fail_on_validity_change and not comparison["same_valid"]:
        failures.append("audit validation validity changed")
    if args.fail_on_audit_path_change and not comparison["same_audit"]:
        failures.append("audit path changed")
    if args.fail_on_release_name_change and not comparison["same_release_name"]:
        failures.append("release name changed")
    if args.fail_on_check_count_change and comparison["check_count_delta"] != 0:
        failures.append(f"check_count changed by {comparison['check_count_delta']}")
    if args.fail_on_source_count_change and comparison["source_count_delta"] != 0:
        failures.append(f"source_count changed by {comparison['source_count_delta']}")
    if args.fail_on_failed_check_regression and comparison["failed_check_count_delta"] > args.max_failed_check_regression:
        failures.append(
            f"failed check regression {comparison['failed_check_count_delta']} > {args.max_failed_check_regression}"
        )
    if args.fail_on_failure_regression and comparison["failure_count_delta"] > args.max_failure_regression:
        failures.append(f"failure regression {comparison['failure_count_delta']} > {args.max_failure_regression}")
    if args.fail_on_error_regression and comparison["error_count_delta"] > args.max_error_regression:
        failures.append(f"error regression {comparison['error_count_delta']} > {args.max_error_regression}")
    if args.fail_on_artifact_check_regression:
        regression = -comparison["artifact_check_count_delta"]
        if regression > args.max_artifact_check_regression:
            failures.append(f"artifact check regression {regression} > {args.max_artifact_check_regression}")
    if args.fail_on_source_artifact_check_regression:
        regression = -comparison["source_artifact_checks_delta"]
        if regression > args.max_source_artifact_check_regression:
            failures.append(f"source artifact check regression {regression} > {args.max_source_artifact_check_regression}")
    if args.fail_on_error_set_change and not comparison["same_errors"]:
        if comparison["missing_errors"]:
            failures.append(f"missing validation errors: {comparison['missing_errors']}")
        if comparison["added_errors"]:
            failures.append(f"added validation errors: {comparison['added_errors']}")
    return failures


def print_validation(label: str, row: dict[str, Any]) -> None:
    print(label)
    print(f"  path: {row['path']}")
    print(f"  audit: {row['audit']}")
    print(f"  release_name: {row['release_name']}")
    print(f"  valid: {row['valid']}")
    print(f"  checks: {row['check_count']} failed={row['failed_check_count']}")
    print(f"  failures: {row['failure_count']}")
    print(f"  source_artifact_checks: {row['source_artifact_checks']}/{row['source_count']}")
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

    print("TextPy/SoA release evidence audit validation comparison")
    print("")
    print_validation("baseline", baseline)
    print("")
    print_validation("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  same_valid: {comparison['same_valid']}")
    print(f"  same_audit: {comparison['same_audit']}")
    print(f"  same_release_name: {comparison['same_release_name']}")
    print(f"  check_count_delta: {comparison['check_count_delta']}")
    print(f"  failed_check_count_delta: {comparison['failed_check_count_delta']}")
    print(f"  failure_count_delta: {comparison['failure_count_delta']}")
    print(f"  artifact_check_count_delta: {comparison['artifact_check_count_delta']}")
    print(f"  source_artifact_checks_delta: {comparison['source_artifact_checks_delta']}")
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
    print("RELEASE EVIDENCE AUDIT VALIDATION COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two release evidence audit validation JSON artifacts.")
    parser.add_argument("--baseline-validation", required=True)
    parser.add_argument("--candidate-validation", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--allow-failed-validation", action="store_true")
    parser.add_argument("--fail-on-validity-change", action="store_true")
    parser.add_argument("--fail-on-audit-path-change", action="store_true")
    parser.add_argument("--fail-on-release-name-change", action="store_true")
    parser.add_argument("--fail-on-check-count-change", action="store_true")
    parser.add_argument("--fail-on-source-count-change", action="store_true")
    parser.add_argument("--fail-on-failed-check-regression", action="store_true")
    parser.add_argument("--max-failed-check-regression", type=int, default=0)
    parser.add_argument("--fail-on-failure-regression", action="store_true")
    parser.add_argument("--max-failure-regression", type=int, default=0)
    parser.add_argument("--fail-on-error-regression", action="store_true")
    parser.add_argument("--max-error-regression", type=int, default=0)
    parser.add_argument("--fail-on-artifact-check-regression", action="store_true")
    parser.add_argument("--max-artifact-check-regression", type=int, default=0)
    parser.add_argument("--fail-on-source-artifact-check-regression", action="store_true")
    parser.add_argument("--max-source-artifact-check-regression", type=int, default=0)
    parser.add_argument("--fail-on-error-set-change", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceAuditValidationComparisonError as exc:
        raise SystemExit(str(exc)) from exc
