#!/usr/bin/env python3
"""Compare two TextPy/SoA release evidence audit rollup JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ReleaseEvidenceAuditComparisonError(RuntimeError):
    pass


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseEvidenceAuditComparisonError(f"Missing release evidence audit JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceAuditComparisonError(f"Invalid release evidence audit JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseEvidenceAuditComparisonError(f"Expected JSON object: {path}")
    return payload


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def optional_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate_audit(payload: dict[str, Any], path: Path, *, allow_failed_audit: bool) -> list[str]:
    errors: list[str] = []
    if payload.get("kind") != "text_release_evidence_audit":
        errors.append("kind must be text_release_evidence_audit.")
    summary = payload.get("summary")
    checks = payload.get("checks")
    failures = payload.get("failures")
    sources = payload.get("sources")
    if not isinstance(summary, dict):
        errors.append("summary must be an object.")
        summary = {}
    if not isinstance(checks, list):
        errors.append("checks must be a list.")
        checks = []
    if not isinstance(failures, list):
        errors.append("failures must be a list.")
        failures = []
    if not isinstance(sources, dict):
        errors.append("sources must be an object.")
        sources = {}

    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"checks[{index}] must be an object.")
            continue
        if not isinstance(check.get("name"), str) or not check.get("name"):
            errors.append(f"checks[{index}].name must be a non-empty string.")
        if not isinstance(check.get("path"), str) or not check.get("path"):
            errors.append(f"checks[{index}].path must be a non-empty string.")
        if not isinstance(check.get("valid"), bool):
            errors.append(f"checks[{index}].valid must be boolean.")
        if not isinstance(check.get("message"), str):
            errors.append(f"checks[{index}].message must be a string.")
        if "metrics" in check and not isinstance(check.get("metrics"), dict):
            errors.append(f"checks[{index}].metrics must be an object when present.")

    names = [check.get("name") for check in checks if isinstance(check, dict)]
    if len(names) != len(set(names)):
        errors.append("check names must be unique.")
    failed_checks = [check for check in checks if isinstance(check, dict) and check.get("valid") is not True]
    if optional_int(summary.get("check_count"), default=len(checks)) != len(checks):
        errors.append("summary.check_count must match checks length.")
    if optional_int(summary.get("failed_check_count"), default=len(failed_checks)) != len(failed_checks):
        errors.append("summary.failed_check_count must match failed checks.")
    expected_valid = not failed_checks and not failures
    if payload.get("valid") is not expected_valid:
        errors.append("valid must match failed checks and failures.")
    if not allow_failed_audit and payload.get("valid") is not True:
        errors.append(f"audit is not valid: {path}")
    return errors


def check_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for check in row["checks"]:
        if isinstance(check, dict) and isinstance(check.get("name"), str):
            mapped[check["name"]] = check
    return mapped


def compact_check(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": check.get("name"),
        "path": check.get("path"),
        "valid": check.get("valid"),
        "message": check.get("message"),
        "metrics": check.get("metrics") if isinstance(check.get("metrics"), dict) else {},
    }


def load_valid_audit(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(path)
    errors = validate_audit(payload, path, allow_failed_audit=args.allow_failed_audit)
    if errors:
        raise ReleaseEvidenceAuditComparisonError(f"Release evidence audit validation failed for {path}: {errors}")
    summary = as_dict(payload.get("summary"))
    checks = as_list(payload.get("checks"))
    failures = as_list(payload.get("failures"))
    sources = as_dict(payload.get("sources"))
    return {
        "path": str(path),
        "payload": payload,
        "valid": payload.get("valid") is True,
        "release_name": summary.get("release_name"),
        "archive_sha256": summary.get("archive_sha256"),
        "source_chain_head": summary.get("source_chain_head"),
        "replay_chain_head": summary.get("replay_chain_head"),
        "source_entry_count": optional_int(summary.get("source_entry_count")),
        "replay_entry_count": optional_int(summary.get("replay_entry_count")),
        "growth_entry_count_delta": optional_int(summary.get("growth_entry_count_delta")),
        "replay_actions": summary.get("replay_actions") if isinstance(summary.get("replay_actions"), list) else [],
        "tamper_detected": summary.get("tamper_detected") is True,
        "artifact_check_count": optional_int(summary.get("artifact_check_count")),
        "check_count": len(checks),
        "failed_check_count": len([check for check in checks if isinstance(check, dict) and check.get("valid") is not True]),
        "failure_count": len(failures),
        "checks": checks,
        "sources": sources,
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "valid": row["valid"],
        "release_name": row["release_name"],
        "archive_sha256": row["archive_sha256"],
        "source_chain_head": row["source_chain_head"],
        "replay_chain_head": row["replay_chain_head"],
        "source_entry_count": row["source_entry_count"],
        "replay_entry_count": row["replay_entry_count"],
        "growth_entry_count_delta": row["growth_entry_count_delta"],
        "replay_actions": row["replay_actions"],
        "tamper_detected": row["tamper_detected"],
        "artifact_check_count": row["artifact_check_count"],
        "check_count": row["check_count"],
        "failed_check_count": row["failed_check_count"],
        "failure_count": row["failure_count"],
        "checks": [compact_check(check) for check in row["checks"] if isinstance(check, dict)],
        "sources": row["sources"],
    }


def changed_common_checks(
    baseline_checks: dict[str, dict[str, Any]],
    candidate_checks: dict[str, dict[str, Any]],
    field: str,
) -> list[str]:
    changed: list[str] = []
    for name in sorted(set(baseline_checks) & set(candidate_checks)):
        left = baseline_checks[name].get(field)
        right = candidate_checks[name].get(field)
        if left != right:
            changed.append(name)
    return changed


def changed_source_paths(baseline_sources: dict[str, Any], candidate_sources: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for name in sorted(set(baseline_sources) & set(candidate_sources)):
        if baseline_sources.get(name) != candidate_sources.get(name):
            changed.append(name)
    return changed


def compare_audits(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_checks = check_map(baseline)
    candidate_checks = check_map(candidate)
    baseline_names = set(baseline_checks)
    candidate_names = set(candidate_checks)
    baseline_sources = baseline["sources"]
    candidate_sources = candidate["sources"]
    baseline_source_names = set(baseline_sources)
    candidate_source_names = set(candidate_sources)
    changed_validity = changed_common_checks(baseline_checks, candidate_checks, "valid")
    changed_messages = changed_common_checks(baseline_checks, candidate_checks, "message")
    changed_metrics = changed_common_checks(baseline_checks, candidate_checks, "metrics")
    changed_paths = changed_common_checks(baseline_checks, candidate_checks, "path")
    return {
        "same_valid": baseline["valid"] == candidate["valid"],
        "same_release_name": baseline["release_name"] == candidate["release_name"],
        "same_archive_sha256": baseline["archive_sha256"] == candidate["archive_sha256"],
        "same_source_chain_head": baseline["source_chain_head"] == candidate["source_chain_head"],
        "same_replay_chain_head": baseline["replay_chain_head"] == candidate["replay_chain_head"],
        "same_replay_actions": baseline["replay_actions"] == candidate["replay_actions"],
        "same_tamper_detected": baseline["tamper_detected"] == candidate["tamper_detected"],
        "source_entry_count_delta": candidate["source_entry_count"] - baseline["source_entry_count"],
        "replay_entry_count_delta": candidate["replay_entry_count"] - baseline["replay_entry_count"],
        "growth_entry_count_delta_change": candidate["growth_entry_count_delta"] - baseline["growth_entry_count_delta"],
        "artifact_check_count_delta": candidate["artifact_check_count"] - baseline["artifact_check_count"],
        "check_count_delta": candidate["check_count"] - baseline["check_count"],
        "failed_check_count_delta": candidate["failed_check_count"] - baseline["failed_check_count"],
        "failure_count_delta": candidate["failure_count"] - baseline["failure_count"],
        "same_check_set": baseline_names == candidate_names,
        "common_checks": sorted(baseline_names & candidate_names),
        "missing_checks": sorted(baseline_names - candidate_names),
        "added_checks": sorted(candidate_names - baseline_names),
        "same_check_validity": not changed_validity,
        "changed_check_validity": changed_validity,
        "same_check_messages": not changed_messages,
        "changed_check_messages": changed_messages,
        "same_check_metrics": not changed_metrics,
        "changed_check_metrics": changed_metrics,
        "same_check_paths": not changed_paths,
        "changed_check_paths": changed_paths,
        "same_source_set": baseline_source_names == candidate_source_names,
        "missing_sources": sorted(baseline_source_names - candidate_source_names),
        "added_sources": sorted(candidate_source_names - baseline_source_names),
        "same_source_paths": not changed_source_paths(baseline_sources, candidate_sources),
        "changed_source_paths": changed_source_paths(baseline_sources, candidate_sources),
    }


def threshold_failures(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.fail_on_validity_change and not comparison["same_valid"]:
        failures.append("audit validity changed")
    if args.fail_on_release_name_change and not comparison["same_release_name"]:
        failures.append("release name changed")
    if args.fail_on_archive_sha_change and not comparison["same_archive_sha256"]:
        failures.append("archive sha256 changed")
    if args.fail_on_chain_head_change:
        if not comparison["same_source_chain_head"]:
            failures.append("source chain head changed")
        if not comparison["same_replay_chain_head"]:
            failures.append("replay chain head changed")
    if args.fail_on_replay_action_change and not comparison["same_replay_actions"]:
        failures.append("replay actions changed")
    if args.fail_on_tamper_change and not comparison["same_tamper_detected"]:
        failures.append("tamper detection polarity changed")
    if args.fail_on_entry_count_change:
        for key in ["source_entry_count_delta", "replay_entry_count_delta", "growth_entry_count_delta_change"]:
            if comparison[key] != 0:
                failures.append(f"{key} changed by {comparison[key]}")
    if args.fail_on_check_set_change and not comparison["same_check_set"]:
        if comparison["missing_checks"]:
            failures.append(f"missing audit checks: {comparison['missing_checks']}")
        if comparison["added_checks"]:
            failures.append(f"added audit checks: {comparison['added_checks']}")
    if args.fail_on_check_validity_change and not comparison["same_check_validity"]:
        failures.append(f"changed audit check validity: {comparison['changed_check_validity']}")
    if args.fail_on_check_message_change and not comparison["same_check_messages"]:
        failures.append(f"changed audit check messages: {comparison['changed_check_messages']}")
    if args.fail_on_check_metric_change and not comparison["same_check_metrics"]:
        failures.append(f"changed audit check metrics: {comparison['changed_check_metrics']}")
    if args.fail_on_check_path_change and not comparison["same_check_paths"]:
        failures.append(f"changed audit check paths: {comparison['changed_check_paths']}")
    if args.fail_on_source_set_change and not comparison["same_source_set"]:
        if comparison["missing_sources"]:
            failures.append(f"missing audit sources: {comparison['missing_sources']}")
        if comparison["added_sources"]:
            failures.append(f"added audit sources: {comparison['added_sources']}")
    if args.fail_on_source_path_change and not comparison["same_source_paths"]:
        failures.append(f"changed audit source paths: {comparison['changed_source_paths']}")
    if args.fail_on_failed_check_regression and comparison["failed_check_count_delta"] > args.max_failed_check_regression:
        failures.append(
            f"failed audit check regression {comparison['failed_check_count_delta']} > {args.max_failed_check_regression}"
        )
    if args.fail_on_failure_regression and comparison["failure_count_delta"] > args.max_failure_regression:
        failures.append(f"audit failure regression {comparison['failure_count_delta']} > {args.max_failure_regression}")
    if args.fail_on_artifact_check_regression:
        regression = -comparison["artifact_check_count_delta"]
        if regression > args.max_artifact_check_regression:
            failures.append(f"artifact check regression {regression} > {args.max_artifact_check_regression}")
    return failures


def short(value: Any) -> str:
    return str(value)[:12] if value else "none"


def print_audit(label: str, row: dict[str, Any]) -> None:
    print(label)
    print(f"  path: {row['path']}")
    print(f"  release_name: {row['release_name']}")
    print(f"  valid: {row['valid']}")
    print(f"  archive_sha256: {short(row['archive_sha256'])}")
    print(f"  source_chain_head: {short(row['source_chain_head'])}")
    print(f"  replay_chain_head: {short(row['replay_chain_head'])}")
    print(f"  entries: {row['source_entry_count']} -> {row['replay_entry_count']}")
    print(f"  checks: {row['check_count']} failed={row['failed_check_count']}")
    print(f"  artifact_checks: {row['artifact_check_count']}")


def run(args: argparse.Namespace) -> int:
    baseline = load_valid_audit(Path(args.baseline_audit), args)
    candidate = load_valid_audit(Path(args.candidate_audit), args)
    comparison = compare_audits(baseline, candidate)
    failures = threshold_failures(comparison, args)
    payload = {
        "baseline": compact(baseline),
        "candidate": compact(candidate),
        "comparison": comparison,
        "failures": failures,
    }

    print("TextPy/SoA release evidence audit comparison")
    print("")
    print_audit("baseline", baseline)
    print("")
    print_audit("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  same_valid: {comparison['same_valid']}")
    print(f"  same_release_name: {comparison['same_release_name']}")
    print(f"  same_archive_sha256: {comparison['same_archive_sha256']}")
    print(f"  same_source_chain_head: {comparison['same_source_chain_head']}")
    print(f"  same_replay_chain_head: {comparison['same_replay_chain_head']}")
    print(f"  same_check_set: {comparison['same_check_set']}")
    print(f"  same_check_validity: {comparison['same_check_validity']}")
    print(f"  same_check_metrics: {comparison['same_check_metrics']}")
    print(f"  failed_check_count_delta: {comparison['failed_check_count_delta']}")
    print(f"  artifact_check_count_delta: {comparison['artifact_check_count_delta']}")

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
    print("RELEASE EVIDENCE AUDIT COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two release evidence audit JSON artifacts.")
    parser.add_argument("--baseline-audit", required=True)
    parser.add_argument("--candidate-audit", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--allow-failed-audit", action="store_true")
    parser.add_argument("--fail-on-validity-change", action="store_true")
    parser.add_argument("--fail-on-release-name-change", action="store_true")
    parser.add_argument("--fail-on-archive-sha-change", action="store_true")
    parser.add_argument("--fail-on-chain-head-change", action="store_true")
    parser.add_argument("--fail-on-replay-action-change", action="store_true")
    parser.add_argument("--fail-on-tamper-change", action="store_true")
    parser.add_argument("--fail-on-entry-count-change", action="store_true")
    parser.add_argument("--fail-on-check-set-change", action="store_true")
    parser.add_argument("--fail-on-check-validity-change", action="store_true")
    parser.add_argument("--fail-on-check-message-change", action="store_true")
    parser.add_argument("--fail-on-check-metric-change", action="store_true")
    parser.add_argument("--fail-on-check-path-change", action="store_true")
    parser.add_argument("--fail-on-source-set-change", action="store_true")
    parser.add_argument("--fail-on-source-path-change", action="store_true")
    parser.add_argument("--fail-on-failed-check-regression", action="store_true")
    parser.add_argument("--max-failed-check-regression", type=int, default=0)
    parser.add_argument("--fail-on-failure-regression", action="store_true")
    parser.add_argument("--max-failure-regression", type=int, default=0)
    parser.add_argument("--fail-on-artifact-check-regression", action="store_true")
    parser.add_argument("--max-artifact-check-regression", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceAuditComparisonError as exc:
        raise SystemExit(str(exc)) from exc
