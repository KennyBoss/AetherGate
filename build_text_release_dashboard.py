#!/usr/bin/env python3
"""Build a single TextPy/SoA release dashboard JSON artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise SystemExit(f"Missing required JSON artifact: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def close_enough(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    left_num = finite_number(left)
    right_num = finite_number(right)
    if left_num is not None and right_num is not None:
        return abs(left_num - right_num) <= tolerance
    return left == right


def first_release(leaderboard: dict[str, Any]) -> dict[str, Any]:
    releases = leaderboard.get("releases")
    if not isinstance(releases, list) or not releases:
        raise SystemExit("Leaderboard JSON must contain at least one release.")
    first = releases[0]
    if not isinstance(first, dict):
        raise SystemExit("Leaderboard release rows must be JSON objects.")
    return first


def memory_summary(current: dict[str, Any], suite: dict[str, Any] | None) -> dict[str, Any]:
    current_memory = current.get("memory") if isinstance(current.get("memory"), dict) else {}
    suite_summary = suite.get("summary") if suite and isinstance(suite.get("summary"), dict) else {}
    return {
        "prompt_count": current_memory.get("prompt_count", suite_summary.get("prompt_count")),
        "pair_count": current_memory.get("pair_count", suite_summary.get("pair_count")),
        "mean_pair_overlap": current_memory.get("mean_pair_overlap", suite_summary.get("mean_pair_overlap")),
        "min_pair_overlap": current_memory.get("min_pair_overlap", suite_summary.get("min_pair_overlap")),
        "max_pair_state_norm_delta": current_memory.get(
            "max_pair_state_norm_delta", suite_summary.get("max_pair_state_norm_delta")
        ),
        "max_pair_state_l2_distance": current_memory.get(
            "max_pair_state_l2_distance", suite_summary.get("max_pair_state_l2_distance")
        ),
        "min_pair_state_cosine_similarity": current_memory.get(
            "min_pair_state_cosine_similarity", suite_summary.get("min_pair_state_cosine_similarity")
        ),
        "has_state_vectors": current_memory.get("has_state_vectors", suite_summary.get("has_state_vectors")),
        "suite": current_memory.get("suite"),
        "validation": current_memory.get("validation"),
    }


def audit_dashboard(
    leaderboard: dict[str, Any],
    current: dict[str, Any],
    suite: dict[str, Any] | None,
    validation: dict[str, Any] | None,
    run_report: dict[str, Any] | None,
    current_path: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, **extra: Any) -> None:
        row = {"name": name, "ok": bool(ok), "detail": detail}
        row.update(extra)
        checks.append(row)

    top = first_release(leaderboard)
    add(
        "active_matches_leaderboard",
        current.get("checkpoint_sha256") == top.get("checkpoint_sha256"),
        "active checkpoint matches rank-1 leaderboard checkpoint",
        active=current.get("checkpoint_sha256"),
        leaderboard=top.get("checkpoint_sha256"),
    )
    add(
        "active_rank",
        int(current.get("rank") or 0) == int(top.get("rank") or 0) == 1,
        "active release and leaderboard top row are rank 1",
        active_rank=current.get("rank"),
        leaderboard_rank=top.get("rank"),
    )
    add(
        "package_validation",
        current.get("validation", {}).get("valid") is True,
        "current release package validation is true",
    )
    add(
        "loss_matches_leaderboard",
        close_enough(current.get("benchmark_loss"), top.get("benchmark_loss")),
        "active release loss matches leaderboard row",
        active=current.get("benchmark_loss"),
        leaderboard=top.get("benchmark_loss"),
    )
    add(
        "throughput_matches_leaderboard",
        close_enough(
            current.get("benchmark_best_throughput_tokens_s"),
            top.get("benchmark_best_throughput_tokens_s"),
        ),
        "active release throughput matches leaderboard row",
        active=current.get("benchmark_best_throughput_tokens_s"),
        leaderboard=top.get("benchmark_best_throughput_tokens_s"),
    )

    memory = memory_summary(current, suite)
    if suite:
        suite_summary = suite.get("summary") if isinstance(suite.get("summary"), dict) else {}
        add(
            "memory_suite_counts",
            int(memory.get("prompt_count") or 0) == int(suite.get("prompt_count") or 0)
            and int(memory.get("pair_count") or 0) == int(suite.get("pair_count") or 0),
            "active release memory counts match loaded suite",
            active_prompts=memory.get("prompt_count"),
            suite_prompts=suite.get("prompt_count"),
            active_pairs=memory.get("pair_count"),
            suite_pairs=suite.get("pair_count"),
        )
        add(
            "memory_overlap_matches_suite",
            close_enough(memory.get("mean_pair_overlap"), suite_summary.get("mean_pair_overlap")),
            "active release memory overlap matches suite summary",
            active=memory.get("mean_pair_overlap"),
            suite=suite_summary.get("mean_pair_overlap"),
        )

    if validation:
        add(
            "memory_validation",
            validation.get("valid") is True,
            "memory suite validation is true",
        )
        add(
            "memory_validation_overlap",
            close_enough(memory.get("mean_pair_overlap"), validation.get("mean_pair_overlap")),
            "active release memory overlap matches validation report",
            active=memory.get("mean_pair_overlap"),
            validation=validation.get("mean_pair_overlap"),
        )

    if run_report:
        add(
            "run_report_release",
            run_report.get("release") in {str(current_path), current.get("_source_path")},
            "run report records the active release pointer",
            run_release=run_report.get("release"),
            current_release=str(current_path),
        )
        add(
            "run_report_sample",
            bool(run_report.get("sample")),
            "active release run report contains generated sample text",
        )

    failed = [row for row in checks if not row["ok"]]
    return {
        "valid": not failed,
        "check_count": len(checks),
        "failed_count": len(failed),
        "checks": checks,
    }


def normalize_current(current: dict[str, Any], source_path: Path) -> dict[str, Any]:
    normalized = dict(current)
    normalized["_source_path"] = str(source_path)
    return normalized


def normalize_pipeline(
    pipeline: dict[str, Any] | None,
    output_path: Path,
    audit: dict[str, Any],
) -> dict[str, Any] | None:
    if pipeline is None:
        return None
    normalized = dict(pipeline)
    summary = dict(normalized.get("summary") or {})
    summary["release_dashboard"] = str(output_path)
    summary["release_dashboard_audit_valid"] = audit["valid"]
    summary["release_dashboard_audit_checks"] = audit["check_count"]
    normalized["summary"] = summary
    return normalized


def build(args: argparse.Namespace) -> dict[str, Any]:
    leaderboard_path = Path(args.leaderboard)
    current_path = Path(args.current_release)
    suite_path = Path(args.memory_suite) if args.memory_suite else None
    validation_path = Path(args.memory_validation) if args.memory_validation else None
    run_path = Path(args.run_json) if args.run_json else None
    pipeline_path = Path(args.pipeline_json) if args.pipeline_json else None

    leaderboard = load_json(leaderboard_path)
    current_raw = load_json(current_path)
    assert leaderboard is not None and current_raw is not None
    current = normalize_current(current_raw, current_path)
    suite = load_json(suite_path, required=False) if suite_path else None
    validation = load_json(validation_path, required=False) if validation_path else None
    run_report = load_json(run_path, required=False) if run_path else None
    pipeline = load_json(pipeline_path, required=False) if pipeline_path else None
    audit = audit_dashboard(leaderboard, current, suite, validation, run_report, current_path)
    if args.require_valid and not audit["valid"]:
        failed = ", ".join(row["name"] for row in audit["checks"] if not row["ok"])
        raise SystemExit(f"Release dashboard audit failed: {failed}")

    memory = memory_summary(current, suite)
    output_path = Path(args.output_json)
    pipeline = normalize_pipeline(pipeline, output_path, audit)
    return {
        "created_at": timestamp(),
        "kind": "text_release_dashboard",
        "artifacts": {
            "leaderboard": str(leaderboard_path),
            "current_release": str(current_path),
            "memory_suite": str(suite_path) if suite_path else None,
            "memory_validation": str(validation_path) if validation_path else None,
            "run_json": str(run_path) if run_path else None,
            "pipeline_json": str(pipeline_path) if pipeline_path else None,
        },
        "dashboard": {
            "name": current.get("name"),
            "rank": current.get("rank"),
            "release_count": leaderboard.get("release_count"),
            "benchmark_loss": current.get("benchmark_loss"),
            "benchmark_accuracy": current.get("benchmark_accuracy"),
            "benchmark_best_throughput_tokens_s": current.get("benchmark_best_throughput_tokens_s"),
            "checkpoint_sha256": current.get("checkpoint_sha256"),
            "archive_sha256": current.get("archive_sha256"),
            "memory": memory,
            "audit_valid": audit["valid"],
            "audit_failed_count": audit["failed_count"],
        },
        "leaderboard": leaderboard,
        "current": current,
        "suite": suite,
        "memory_validation": validation,
        "run": run_report,
        "pipeline": pipeline,
        "audit": audit,
    }


def run(args: argparse.Namespace) -> None:
    payload = build(args)
    write_json(Path(args.output_json), payload)
    dashboard = payload["dashboard"]
    memory = dashboard["memory"]

    print("TextPy/SoA release dashboard builder")
    print(f"name: {dashboard.get('name')}")
    print(f"rank: {dashboard.get('rank')}")
    print(f"release_count: {dashboard.get('release_count')}")
    print(f"loss: {float(dashboard['benchmark_loss']):.4f}")
    print(f"best_throughput_tokens_s: {float(dashboard['benchmark_best_throughput_tokens_s']):,.0f}")
    print(f"memory_mean_pair_overlap: {float(memory['mean_pair_overlap']):.4f}")
    print(f"audit_valid: {payload['audit']['valid']}")
    print(f"Saved dashboard JSON: {args.output_json}")
    print("RELEASE DASHBOARD OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one JSON artifact for the static release dashboard.")
    parser.add_argument("--leaderboard", default="artifacts/releases/text_release_leaderboard.json")
    parser.add_argument("--current-release", default="artifacts/releases/current_text_release.json")
    parser.add_argument("--memory-suite", default="artifacts/releases/text_release_memory_suite/memory_suite.json")
    parser.add_argument("--memory-validation", default="artifacts/releases/text_release_memory_suite/validation.json")
    parser.add_argument("--run-json", default="artifacts/releases/run_current_text_release.json")
    parser.add_argument("--pipeline-json", default=None)
    parser.add_argument("--output-json", default="artifacts/releases/text_release_dashboard.json")
    parser.add_argument("--require-valid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
