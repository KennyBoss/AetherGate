#!/usr/bin/env python3
"""Compare two benchmark JSON reports and flag regressions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple


class MetricSpec(NamedTuple):
    section: str
    path: tuple[str, ...]
    higher_is_better: bool
    label: str


METRICS = (
    MetricSpec("agent", ("throughput_agent_steps_s",), True, "agent throughput"),
    MetricSpec("agent", ("mean_goal_distance", "final"), False, "agent goal distance"),
    MetricSpec("agent", ("mean_threat_distance", "final"), True, "agent threat distance"),
    MetricSpec("text", ("throughput_tokens_s",), True, "text throughput"),
    MetricSpec("text", ("final_loss",), False, "text final loss"),
    MetricSpec("text", ("final_accuracy",), True, "text accuracy"),
    MetricSpec("evolution", ("throughput_agent_steps_s",), True, "evolution throughput"),
    MetricSpec("evolution", ("best_reward_seen",), True, "evolution reward"),
    MetricSpec("evolution", ("best_goal_distance",), False, "evolution goal distance"),
    MetricSpec("replay", ("throughput_agent_steps_s",), True, "replay throughput"),
    MetricSpec("replay", ("reward",), True, "replay reward"),
    MetricSpec("replay", ("mean_goal_distance", "final"), False, "replay goal distance"),
)


class ComparisonFailure(RuntimeError):
    pass


def load_report(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ComparisonFailure(f"Report does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ComparisonFailure(f"Report is not valid JSON: {path}: {exc}") from exc


def get_metric(report: dict[str, Any], spec: MetricSpec) -> float | None:
    value: Any = report.get("results", {}).get(spec.section, {}).get("metrics", {})
    for key in spec.path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    return float(value)


def relative_change(base: float, candidate: float, higher_is_better: bool) -> float:
    if math.isclose(base, 0.0, abs_tol=1.0e-12):
        raw = candidate - base
    else:
        raw = (candidate - base) / abs(base)
    return raw if higher_is_better else -raw


def format_num(value: float | None) -> str:
    if value is None:
        return "missing"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.4f}"


def compare(args: argparse.Namespace) -> int:
    baseline = load_report(Path(args.baseline))
    candidate = load_report(Path(args.candidate))

    print(f"Baseline:  {args.baseline}")
    print(f"Candidate: {args.candidate}")
    print(f"Threshold: regression worse than {args.max_regression * 100:.1f}% fails")
    print("")
    print(f"{'metric':28} {'baseline':>14} {'candidate':>14} {'change':>10} status")

    failures: list[str] = []
    for spec in METRICS:
        base = get_metric(baseline, spec)
        cand = get_metric(candidate, spec)
        if base is None or cand is None:
            if args.require_all:
                failures.append(f"{spec.label}: missing metric")
                status = "FAIL missing"
            else:
                status = "SKIP missing"
            print(f"{spec.label:28} {format_num(base):>14} {format_num(cand):>14} {'n/a':>10} {status}")
            continue

        improvement = relative_change(base, cand, spec.higher_is_better)
        status = "OK"
        if improvement < -args.max_regression:
            status = "FAIL"
            failures.append(
                f"{spec.label}: {improvement * 100:.2f}% regression "
                f"({format_num(base)} -> {format_num(cand)})"
            )
        print(
            f"{spec.label:28} {format_num(base):>14} {format_num(cand):>14} "
            f"{improvement * 100:>9.2f}% {status}"
        )

    if failures:
        print("")
        print("Regressions:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("")
    print("BENCHMARK COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two benchmark JSON reports.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.25,
        help="Allowed relative regression before failing. 0.25 means 25%%.",
    )
    parser.add_argument("--require-all", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(compare(parse_args()))
