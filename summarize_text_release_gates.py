#!/usr/bin/env python3
"""Summarize TextPy/SoA release gate artifacts into one operator verdict."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReleaseGateSummaryError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseGateSummaryError(f"Missing gate artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseGateSummaryError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseGateSummaryError(f"Expected a JSON object: {path}")
    return payload


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def gate_record(
    name: str,
    path: Path,
    *,
    valid: bool,
    failures: list[Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "valid": bool(valid),
        "failure_count": len(failures),
        "failures": [str(item) for item in failures],
        "metrics": metrics,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    dashboard_validation_path = Path(args.dashboard_validation)
    dashboard_comparison_path = Path(args.dashboard_comparison)
    history_validation_path = Path(args.history_validation)
    history_analysis_path = Path(args.history_analysis)

    dashboard_validation = load_json(dashboard_validation_path)
    dashboard_comparison = load_json(dashboard_comparison_path)
    history_validation = load_json(history_validation_path)
    history_analysis = load_json(history_analysis_path)

    dashboard_compare_failures = as_list(dashboard_comparison.get("failures"))
    history_analysis_failures = as_list(history_analysis.get("failures"))
    gates = [
        gate_record(
            "dashboard_validation",
            dashboard_validation_path,
            valid=dashboard_validation.get("valid") is True,
            failures=as_list(dashboard_validation.get("errors")),
            metrics={
                "name": dashboard_validation.get("name"),
                "rank": dashboard_validation.get("rank"),
                "release_count": dashboard_validation.get("release_count"),
                "benchmark_loss": dashboard_validation.get("benchmark_loss"),
                "best_throughput_tokens_s": dashboard_validation.get("benchmark_best_throughput_tokens_s"),
                "mean_pair_overlap": dashboard_validation.get("mean_pair_overlap"),
                "audit_check_count": dashboard_validation.get("audit_check_count"),
                "artifact_checks": dashboard_validation.get("artifact_checks"),
            },
        ),
        gate_record(
            "dashboard_comparison",
            dashboard_comparison_path,
            valid=not dashboard_compare_failures,
            failures=dashboard_compare_failures,
            metrics={
                "winner_by_loss": (dashboard_comparison.get("comparison") or {}).get("winner_by_loss"),
                "loss_delta": (dashboard_comparison.get("comparison") or {}).get("loss_delta"),
                "best_throughput_tokens_s_delta": (
                    dashboard_comparison.get("comparison") or {}
                ).get("best_throughput_tokens_s_delta"),
                "memory_mean_pair_overlap_delta": (
                    dashboard_comparison.get("comparison") or {}
                ).get("memory_mean_pair_overlap_delta"),
                "audit_failed_count_delta": (
                    dashboard_comparison.get("comparison") or {}
                ).get("audit_failed_count_delta"),
            },
        ),
        gate_record(
            "history_validation",
            history_validation_path,
            valid=history_validation.get("valid") is True,
            failures=as_list(history_validation.get("errors")),
            metrics={
                "release_count": history_validation.get("release_count"),
                "valid_count": history_validation.get("valid_count"),
                "best_loss": history_validation.get("best_loss"),
                "best_throughput_tokens_s": history_validation.get("best_throughput_tokens_s"),
                "best_memory_mean_pair_overlap": history_validation.get("best_memory_mean_pair_overlap"),
                "artifact_count": history_validation.get("artifact_count"),
            },
        ),
        gate_record(
            "history_analysis",
            history_analysis_path,
            valid=history_analysis.get("valid") is True and not history_analysis_failures,
            failures=history_analysis_failures,
            metrics={
                "baseline_mode": history_analysis.get("baseline_mode"),
                "release_count": history_analysis.get("release_count"),
                "latest_name": (history_analysis.get("latest") or {}).get("name"),
                "baseline_name": (history_analysis.get("baseline") or {}).get("name"),
                "loss_delta": (history_analysis.get("comparison") or {}).get("loss_delta"),
                "best_throughput_tokens_s_delta": (
                    history_analysis.get("comparison") or {}
                ).get("best_throughput_tokens_s_delta"),
                "memory_mean_pair_overlap_delta": (
                    history_analysis.get("comparison") or {}
                ).get("memory_mean_pair_overlap_delta"),
                "audit_failed_count_delta": (
                    history_analysis.get("comparison") or {}
                ).get("audit_failed_count_delta"),
            },
        ),
    ]

    failed_gates = [gate for gate in gates if not gate["valid"]]
    release_name = dashboard_validation.get("name") or (history_analysis.get("latest") or {}).get("name")
    summary = {
        "release_name": release_name,
        "release_count": history_validation.get("release_count"),
        "benchmark_loss": dashboard_validation.get("benchmark_loss"),
        "best_throughput_tokens_s": dashboard_validation.get("benchmark_best_throughput_tokens_s"),
        "memory_mean_pair_overlap": dashboard_validation.get("mean_pair_overlap"),
        "history_baseline_mode": history_analysis.get("baseline_mode"),
        "history_loss_delta": (history_analysis.get("comparison") or {}).get("loss_delta"),
        "history_memory_overlap_delta": (
            history_analysis.get("comparison") or {}
        ).get("memory_mean_pair_overlap_delta"),
        "failed_gate_count": len(failed_gates),
    }
    return {
        "created_at": timestamp(),
        "kind": "text_release_gate_summary",
        "valid": not failed_gates,
        "summary": summary,
        "gates": gates,
        "failures": [
            {"gate": gate["name"], "failures": gate["failures"]}
            for gate in failed_gates
        ],
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    return f"{numeric:.{digits}f}"


def run(args: argparse.Namespace) -> int:
    payload = summarize(args)
    if args.output_json:
        save_json(args.output_json, payload)
    summary = payload["summary"]
    print("TextPy/SoA release gate summary")
    print(f"release_name: {summary['release_name']}")
    print(f"valid: {payload['valid']}")
    print(f"gates: {len(payload['gates'])}")
    print(f"failed_gates: {summary['failed_gate_count']}")
    print(f"benchmark_loss: {fmt(summary['benchmark_loss'])}")
    print(f"best_throughput_tokens_s: {fmt(summary['best_throughput_tokens_s'], 0)}")
    print(f"memory_mean_pair_overlap: {fmt(summary['memory_mean_pair_overlap'])}")
    print(f"history_baseline_mode: {summary['history_baseline_mode']}")
    print(f"history_loss_delta: {fmt(summary['history_loss_delta'])}")
    if args.output_json:
        print("")
        print(f"Saved JSON: {args.output_json}")
    if not payload["valid"]:
        print("")
        print("FAIL")
        for failure in payload["failures"]:
            print(f"  {failure['gate']}: {', '.join(failure['failures'])}")
        return 1
    print("")
    print("RELEASE GATES OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize release gate JSON artifacts.")
    parser.add_argument("--dashboard-validation", default="artifacts/releases/text_release_dashboard_validation.json")
    parser.add_argument("--dashboard-comparison", default="artifacts/releases/text_release_dashboard_comparison.json")
    parser.add_argument("--history-validation", default="artifacts/releases/text_release_history_validation.json")
    parser.add_argument("--history-analysis", default="artifacts/releases/text_release_history_analysis.json")
    parser.add_argument("--output-json", default="artifacts/releases/text_release_gates.json")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseGateSummaryError as exc:
        raise SystemExit(str(exc)) from exc
