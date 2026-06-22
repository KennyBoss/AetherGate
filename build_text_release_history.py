#!/usr/bin/env python3
"""Build a compact history index from TextPy/SoA release dashboard JSON artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_text_release_dashboard import validate


class ReleaseHistoryError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ReleaseHistoryError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_float(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    numeric = float(value)
    return numeric if math.isfinite(numeric) else fallback


def finite_int(value: Any, fallback: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return int(value)


def resolve_dashboard_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.dashboard:
        path = Path(raw)
        if not path.exists():
            raise ReleaseHistoryError(f"Missing dashboard JSON: {path}")
        paths.append(path)
    for pattern in args.dashboard_glob:
        matches = [Path(match) for match in glob.glob(pattern)]
        if not matches and args.require_glob_match:
            raise ReleaseHistoryError(f"Dashboard glob did not match files: {pattern}")
        paths.extend(matches)

    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return list(unique.values())


def validation_args(path: Path, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dashboard=str(path),
        output_json=None,
        require_artifacts=args.require_artifacts,
        expected_checks=args.expected_checks,
        expected_prompts=args.expected_prompts,
        expected_pairs=args.expected_pairs,
        min_mean_pair_overlap=args.min_mean_pair_overlap,
    )


def compact_entry(path: Path, payload: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    dashboard = payload["dashboard"]
    memory = dashboard.get("memory") or {}
    audit = payload.get("audit") or {}
    artifacts = payload.get("artifacts") or {}
    pipeline = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else None
    pipeline_summary = pipeline.get("summary") if pipeline and isinstance(pipeline.get("summary"), dict) else {}
    return {
        "path": str(path),
        "created_at": payload.get("created_at"),
        "name": dashboard.get("name"),
        "rank": finite_int(dashboard.get("rank")),
        "release_count": finite_int(dashboard.get("release_count")),
        "benchmark_loss": finite_float(dashboard.get("benchmark_loss")),
        "benchmark_accuracy": finite_float(dashboard.get("benchmark_accuracy")),
        "benchmark_best_throughput_tokens_s": finite_float(
            dashboard.get("benchmark_best_throughput_tokens_s")
        ),
        "checkpoint_sha256": dashboard.get("checkpoint_sha256"),
        "archive_sha256": dashboard.get("archive_sha256"),
        "audit_valid": bool(audit.get("valid")),
        "audit_check_count": finite_int(audit.get("check_count")),
        "audit_failed_count": finite_int(audit.get("failed_count")),
        "memory_prompt_count": finite_int(memory.get("prompt_count")),
        "memory_pair_count": finite_int(memory.get("pair_count")),
        "memory_mean_pair_overlap": finite_float(memory.get("mean_pair_overlap")),
        "memory_min_pair_overlap": finite_float(memory.get("min_pair_overlap")),
        "memory_max_pair_state_norm_delta": finite_float(memory.get("max_pair_state_norm_delta")),
        "memory_max_pair_state_l2_distance": finite_float(memory.get("max_pair_state_l2_distance")),
        "memory_min_pair_state_cosine_similarity": finite_float(
            memory.get("min_pair_state_cosine_similarity")
        ),
        "has_state_vectors": memory.get("has_state_vectors") is True,
        "leaderboard": artifacts.get("leaderboard"),
        "current_release": artifacts.get("current_release"),
        "memory_suite": artifacts.get("memory_suite"),
        "pipeline_json": artifacts.get("pipeline_json"),
        "pipeline_release_dashboard": pipeline_summary.get("release_dashboard"),
        "validation": {
            "valid": bool(validation.get("valid")),
            "artifact_count": finite_int(validation.get("artifact_count")),
            "check_count": finite_int(validation.get("audit_check_count")),
        },
    }


def sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("created_at") or ""), str(row.get("name") or ""), str(row.get("path") or ""))


def add_deltas(rows: list[dict[str, Any]]) -> None:
    previous: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        row["history_index"] = index + 1
        if previous is None:
            row["delta"] = {
                "benchmark_loss": 0.0,
                "benchmark_accuracy": 0.0,
                "benchmark_best_throughput_tokens_s": 0.0,
                "memory_mean_pair_overlap": 0.0,
                "memory_min_pair_state_cosine_similarity": 0.0,
                "memory_max_pair_state_l2_distance": 0.0,
                "audit_failed_count": 0,
            }
        else:
            row["delta"] = {
                "benchmark_loss": row["benchmark_loss"] - previous["benchmark_loss"],
                "benchmark_accuracy": row["benchmark_accuracy"] - previous["benchmark_accuracy"],
                "benchmark_best_throughput_tokens_s": (
                    row["benchmark_best_throughput_tokens_s"]
                    - previous["benchmark_best_throughput_tokens_s"]
                ),
                "memory_mean_pair_overlap": (
                    row["memory_mean_pair_overlap"] - previous["memory_mean_pair_overlap"]
                ),
                "memory_min_pair_state_cosine_similarity": (
                    row["memory_min_pair_state_cosine_similarity"]
                    - previous["memory_min_pair_state_cosine_similarity"]
                ),
                "memory_max_pair_state_l2_distance": (
                    row["memory_max_pair_state_l2_distance"]
                    - previous["memory_max_pair_state_l2_distance"]
                ),
                "audit_failed_count": row["audit_failed_count"] - previous["audit_failed_count"],
            }
        previous = row


def best_by(rows: list[dict[str, Any]], key: str, *, higher_is_better: bool) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row[key]) if higher_is_better else min(rows, key=lambda row: row[key])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_loss = best_by(rows, "benchmark_loss", higher_is_better=False)
    best_speed = best_by(rows, "benchmark_best_throughput_tokens_s", higher_is_better=True)
    best_memory = best_by(rows, "memory_mean_pair_overlap", higher_is_better=True)
    latest = rows[-1] if rows else None
    return {
        "release_count": len(rows),
        "valid_count": sum(1 for row in rows if row["audit_valid"] and row["validation"]["valid"]),
        "latest_name": latest.get("name") if latest else None,
        "latest_created_at": latest.get("created_at") if latest else None,
        "best_loss_name": best_loss.get("name") if best_loss else None,
        "best_loss": best_loss.get("benchmark_loss") if best_loss else None,
        "best_throughput_name": best_speed.get("name") if best_speed else None,
        "best_throughput_tokens_s": (
            best_speed.get("benchmark_best_throughput_tokens_s") if best_speed else None
        ),
        "best_memory_name": best_memory.get("name") if best_memory else None,
        "best_memory_mean_pair_overlap": (
            best_memory.get("memory_mean_pair_overlap") if best_memory else None
        ),
        "mean_loss": (
            sum(row["benchmark_loss"] for row in rows) / len(rows) if rows else None
        ),
        "mean_memory_overlap": (
            sum(row["memory_mean_pair_overlap"] for row in rows) / len(rows) if rows else None
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_dashboard_paths(args)
    if not paths:
        raise ReleaseHistoryError("No dashboard JSON files were provided.")

    rows: list[dict[str, Any]] = []
    for path in paths:
        validation = validate(validation_args(path, args))
        if not validation["valid"]:
            raise ReleaseHistoryError(f"Dashboard validation failed for {path}: {validation['errors']}")
        payload = load_json(path)
        if payload.get("kind") != "text_release_dashboard":
            raise ReleaseHistoryError(f"Unexpected dashboard kind in {path}: {payload.get('kind')}")
        rows.append(compact_entry(path, payload, validation))

    rows.sort(key=sort_key)
    add_deltas(rows)
    return {
        "created_at": timestamp(),
        "kind": "text_release_history",
        "source_count": len(paths),
        "summary": summarize(rows),
        "dashboards": rows,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    return f"{numeric:.{digits}f}"


def run(args: argparse.Namespace) -> int:
    payload = build(args)
    write_json(Path(args.output_json), payload)
    summary = payload["summary"]
    print("TextPy/SoA release history builder")
    print(f"releases: {summary['release_count']}")
    print(f"valid: {summary['valid_count']}")
    print(f"latest: {summary['latest_name']}")
    print(f"best_loss: {summary['best_loss_name']} {fmt(summary['best_loss'])}")
    print(f"best_throughput_tokens_s: {summary['best_throughput_tokens_s'] and fmt(summary['best_throughput_tokens_s'], 0)}")
    print(f"best_memory_overlap: {summary['best_memory_name']} {fmt(summary['best_memory_mean_pair_overlap'])}")
    print(f"Saved history JSON: {args.output_json}")
    print("")
    print("RELEASE HISTORY OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a release dashboard history JSON index.")
    parser.add_argument("--dashboard", action="append", default=[], help="Dashboard JSON artifact. Can be repeated.")
    parser.add_argument("--dashboard-glob", action="append", default=[], help="Glob for dashboard JSON artifacts.")
    parser.add_argument("--output-json", default="artifacts/releases/text_release_history.json")
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--require-glob-match", action="store_true")
    parser.add_argument("--expected-checks", type=int, default=11)
    parser.add_argument("--expected-prompts", type=int, default=None)
    parser.add_argument("--expected-pairs", type=int, default=None)
    parser.add_argument("--min-mean-pair-overlap", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseHistoryError as exc:
        raise SystemExit(str(exc)) from exc
