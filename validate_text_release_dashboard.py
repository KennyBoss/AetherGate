#!/usr/bin/env python3
"""Validate a TextPy/SoA release dashboard JSON artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_AUDIT_CHECKS = {
    "active_matches_leaderboard",
    "active_rank",
    "package_validation",
    "loss_matches_leaderboard",
    "throughput_matches_leaderboard",
    "memory_suite_counts",
    "memory_overlap_matches_suite",
    "memory_validation",
    "memory_validation_overlap",
    "run_report_release",
    "run_report_sample",
}


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


def close_enough(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
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


def require_object(payload: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object.")
        return {}
    return value


def validate(args: argparse.Namespace) -> dict[str, Any]:
    dashboard_path = Path(args.dashboard)
    base = dashboard_path.parent
    payload = load_json(dashboard_path)
    errors: list[str] = []

    if payload.get("kind") != "text_release_dashboard":
        errors.append("kind must be text_release_dashboard.")

    artifacts = require_object(payload, "artifacts", errors)
    dashboard = require_object(payload, "dashboard", errors)
    audit = require_object(payload, "audit", errors)
    leaderboard = require_object(payload, "leaderboard", errors)
    current = require_object(payload, "current", errors)
    suite = require_object(payload, "suite", errors)
    memory_validation = require_object(payload, "memory_validation", errors)
    run_report = require_object(payload, "run", errors)
    pipeline = payload.get("pipeline")
    if pipeline is not None and not isinstance(pipeline, dict):
        errors.append("pipeline must be an object or null.")
        pipeline = None

    release_count = as_int(dashboard.get("release_count"), "dashboard.release_count", errors)
    rank = as_int(dashboard.get("rank"), "dashboard.rank", errors)
    as_number(dashboard.get("benchmark_loss"), "dashboard.benchmark_loss", errors, minimum=0.0)
    as_number(dashboard.get("benchmark_accuracy"), "dashboard.benchmark_accuracy", errors, minimum=0.0, maximum=1.0)
    as_number(
        dashboard.get("benchmark_best_throughput_tokens_s"),
        "dashboard.benchmark_best_throughput_tokens_s",
        errors,
        minimum=0.0,
    )
    if not isinstance(dashboard.get("name"), str) or not dashboard.get("name"):
        errors.append("dashboard.name must be a non-empty string.")
    for key in ["checkpoint_sha256", "archive_sha256"]:
        if not isinstance(dashboard.get(key), str) or len(dashboard.get(key, "")) < 12:
            errors.append(f"dashboard.{key} must be a SHA-like string.")

    memory = dashboard.get("memory") if isinstance(dashboard.get("memory"), dict) else {}
    if not memory:
        errors.append("dashboard.memory must be an object.")
    prompt_count = as_int(memory.get("prompt_count"), "dashboard.memory.prompt_count", errors)
    pair_count = as_int(memory.get("pair_count"), "dashboard.memory.pair_count", errors)
    mean_overlap = as_number(
        memory.get("mean_pair_overlap"),
        "dashboard.memory.mean_pair_overlap",
        errors,
        minimum=0.0,
        maximum=1.0,
    )
    as_number(memory.get("min_pair_overlap"), "dashboard.memory.min_pair_overlap", errors, minimum=0.0, maximum=1.0)
    as_number(
        memory.get("max_pair_state_norm_delta"),
        "dashboard.memory.max_pair_state_norm_delta",
        errors,
        minimum=0.0,
    )
    as_number(
        memory.get("max_pair_state_l2_distance"),
        "dashboard.memory.max_pair_state_l2_distance",
        errors,
        minimum=0.0,
    )
    as_number(
        memory.get("min_pair_state_cosine_similarity"),
        "dashboard.memory.min_pair_state_cosine_similarity",
        errors,
        minimum=-1.0,
        maximum=1.0,
    )
    if memory.get("has_state_vectors") is not True:
        errors.append("dashboard.memory.has_state_vectors must be true.")

    checks_raw = audit.get("checks")
    checks = checks_raw if isinstance(checks_raw, list) else []
    if not isinstance(checks_raw, list):
        errors.append("audit.checks must be a list.")
    failed_checks = [row for row in checks if isinstance(row, dict) and row.get("ok") is not True]
    check_names = {row.get("name") for row in checks if isinstance(row, dict)}
    missing_checks = sorted(REQUIRED_AUDIT_CHECKS - check_names)
    if missing_checks:
        errors.append(f"Missing audit checks: {missing_checks}")
    if audit.get("valid") is not True:
        errors.append("audit.valid must be true.")
    if int(audit.get("failed_count", -1)) != 0:
        errors.append("audit.failed_count must be 0.")
    if int(audit.get("check_count", -1)) != len(checks):
        errors.append("audit.check_count must match the number of checks.")
    if failed_checks:
        errors.append(f"Audit contains failed checks: {[row.get('name') for row in failed_checks]}")
    if dashboard.get("audit_valid") is not True:
        errors.append("dashboard.audit_valid must be true.")
    if int(dashboard.get("audit_failed_count", -1)) != 0:
        errors.append("dashboard.audit_failed_count must be 0.")

    releases = leaderboard.get("releases")
    release_rows = releases if isinstance(releases, list) else []
    if not isinstance(releases, list) or not release_rows:
        errors.append("leaderboard.releases must contain at least one release.")
    top_release = release_rows[0] if release_rows and isinstance(release_rows[0], dict) else {}
    if release_count is not None and int(leaderboard.get("release_count", -1)) != release_count:
        errors.append("leaderboard.release_count must match dashboard.release_count.")
    if rank is not None and int(current.get("rank", -1)) != rank:
        errors.append("current.rank must match dashboard.rank.")
    if rank is not None and top_release and int(top_release.get("rank", -1)) != rank:
        errors.append("leaderboard top rank must match dashboard.rank.")
    for key in ["name", "checkpoint_sha256", "archive_sha256", "benchmark_loss", "benchmark_accuracy"]:
        if not close_enough(dashboard.get(key), current.get(key)):
            errors.append(f"dashboard.{key} must match current.{key}.")
    if not close_enough(
        dashboard.get("benchmark_best_throughput_tokens_s"),
        current.get("benchmark_best_throughput_tokens_s"),
    ):
        errors.append("dashboard benchmark throughput must match current release throughput.")
    if top_release:
        if current.get("checkpoint_sha256") != top_release.get("checkpoint_sha256"):
            errors.append("current checkpoint SHA must match leaderboard rank-1 checkpoint SHA.")
        if not close_enough(current.get("benchmark_loss"), top_release.get("benchmark_loss")):
            errors.append("current loss must match leaderboard rank-1 loss.")

    suite_summary = suite.get("summary") if isinstance(suite.get("summary"), dict) else {}
    validation_overlap = memory_validation.get("mean_pair_overlap")
    if prompt_count is not None and int(suite.get("prompt_count", -1)) != prompt_count:
        errors.append("suite.prompt_count must match dashboard memory prompt_count.")
    if pair_count is not None and int(suite.get("pair_count", -1)) != pair_count:
        errors.append("suite.pair_count must match dashboard memory pair_count.")
    if mean_overlap is not None and not close_enough(mean_overlap, suite_summary.get("mean_pair_overlap")):
        errors.append("suite summary mean_pair_overlap must match dashboard memory.")
    if mean_overlap is not None and not close_enough(mean_overlap, validation_overlap):
        errors.append("memory validation mean_pair_overlap must match dashboard memory.")
    if memory_validation.get("valid") is not True:
        errors.append("memory_validation.valid must be true.")
    if run_report.get("release") != artifacts.get("current_release"):
        errors.append("run.release must match artifacts.current_release.")
    if not run_report.get("sample"):
        errors.append("run.sample must be present.")

    pipeline_artifact = resolve_artifact_path(base, artifacts.get("pipeline_json"))
    pipeline_for_summary = pipeline
    if pipeline_artifact is not None and pipeline_artifact.exists():
        try:
            pipeline_for_summary = load_json(pipeline_artifact)
        except SystemExit as exc:
            errors.append(str(exc))
    pipeline_summary = (
        pipeline_for_summary.get("summary")
        if isinstance(pipeline_for_summary, dict) and isinstance(pipeline_for_summary.get("summary"), dict)
        else {}
    )
    if pipeline_summary:
        recorded_dashboard_path = resolve_artifact_path(base, pipeline_summary.get("release_dashboard"))
        if recorded_dashboard_path is None:
            errors.append("pipeline.summary.release_dashboard must point to a dashboard file.")
        elif recorded_dashboard_path.resolve() != dashboard_path.resolve() and not recorded_dashboard_path.exists():
            errors.append(f"pipeline.summary.release_dashboard points to a missing dashboard: {recorded_dashboard_path}")
        if pipeline_summary.get("release_dashboard_audit_valid") is not True:
            errors.append("pipeline.summary.release_dashboard_audit_valid must be true.")
        if int(pipeline_summary.get("release_dashboard_audit_checks", -1)) != len(checks):
            errors.append("pipeline.summary.release_dashboard_audit_checks must match audit.check_count.")

    artifact_checks = 0
    if args.require_artifacts:
        for key in ["leaderboard", "current_release", "memory_suite", "memory_validation", "run_json"]:
            path = resolve_artifact_path(base, artifacts.get(key))
            if path is None:
                errors.append(f"artifacts.{key} is missing.")
            elif not path.exists():
                errors.append(f"Missing referenced artifact {key}: {path}")
            else:
                artifact_checks += 1
        if artifacts.get("pipeline_json") and pipeline_artifact is not None:
            if not pipeline_artifact.exists():
                errors.append(f"Missing referenced pipeline artifact: {pipeline_artifact}")
            else:
                artifact_checks += 1

    if args.expected_checks is not None and len(checks) != args.expected_checks:
        errors.append(f"Audit check count mismatch: {len(checks)} != {args.expected_checks}")
    if args.expected_prompts is not None and prompt_count != args.expected_prompts:
        errors.append(f"Prompt count mismatch: {prompt_count} != {args.expected_prompts}")
    if args.expected_pairs is not None and pair_count != args.expected_pairs:
        errors.append(f"Pair count mismatch: {pair_count} != {args.expected_pairs}")
    if args.min_mean_pair_overlap is not None and mean_overlap is not None and mean_overlap < args.min_mean_pair_overlap:
        errors.append(f"mean_pair_overlap failed minimum {args.min_mean_pair_overlap}: {mean_overlap}")

    return {
        "valid": not errors,
        "errors": errors,
        "dashboard": str(dashboard_path),
        "name": dashboard.get("name"),
        "rank": rank,
        "release_count": release_count,
        "benchmark_loss": dashboard.get("benchmark_loss"),
        "benchmark_best_throughput_tokens_s": dashboard.get("benchmark_best_throughput_tokens_s"),
        "prompt_count": prompt_count,
        "pair_count": pair_count,
        "mean_pair_overlap": mean_overlap,
        "audit_check_count": len(checks),
        "artifact_checks": artifact_checks,
        "has_pipeline": isinstance(pipeline, dict),
    }


def run(args: argparse.Namespace) -> None:
    result = validate(args)
    print("TextPy/SoA release dashboard validator")
    print(f"dashboard: {result['dashboard']}")
    print(f"valid: {result['valid']}")
    print(f"name: {result['name']}")
    print(f"rank: {result['rank']}")
    print(f"release_count: {result['release_count']}")
    print(f"benchmark_loss: {float(result['benchmark_loss']):.4f}")
    print(f"benchmark_best_throughput_tokens_s: {float(result['benchmark_best_throughput_tokens_s']):,.0f}")
    print(f"prompts: {result['prompt_count']}")
    print(f"pairs: {result['pair_count']}")
    print(f"mean_pair_overlap: {float(result['mean_pair_overlap']):.4f}")
    print(f"audit_checks: {result['audit_check_count']}")
    print(f"artifact_checks: {result['artifact_checks']}")
    if result["errors"]:
        print("Errors")
        for error in result["errors"]:
            print(f"  {error}")
    if args.output_json:
        save_json(args.output_json, result)
        print("")
        print(f"Saved JSON: {args.output_json}")
    if not result["valid"]:
        raise SystemExit(1)
    print("RELEASE DASHBOARD VALIDATION OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a text release dashboard JSON artifact.")
    parser.add_argument("--dashboard", default="artifacts/releases/text_release_dashboard.json")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--expected-checks", type=int, default=len(REQUIRED_AUDIT_CHECKS))
    parser.add_argument("--expected-prompts", type=int, default=None)
    parser.add_argument("--expected-pairs", type=int, default=None)
    parser.add_argument("--min-mean-pair-overlap", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
