#!/usr/bin/env python3
"""Analyze TextPy/SoA release history trends and gate the latest release."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from validate_text_release_history import validate


class ReleaseHistoryAnalysisError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ReleaseHistoryAnalysisError(f"Expected a JSON object: {path}")
    return payload


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def regression_rate(base: float, candidate: float, *, higher_is_better: bool) -> float:
    if math.isclose(base, 0.0, abs_tol=1.0e-12):
        return base - candidate if higher_is_better else candidate - base
    if higher_is_better:
        return (base - candidate) / abs(base)
    return (candidate - base) / abs(base)


def best_by(rows: list[dict[str, Any]], key: str, *, higher_is_better: bool) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row[key])) if higher_is_better else min(rows, key=lambda row: float(row[key]))


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "history_index": row.get("history_index"),
        "path": row.get("path"),
        "name": row.get("name"),
        "created_at": row.get("created_at"),
        "benchmark_loss": float(row["benchmark_loss"]),
        "benchmark_accuracy": float(row["benchmark_accuracy"]),
        "benchmark_best_throughput_tokens_s": float(row["benchmark_best_throughput_tokens_s"]),
        "checkpoint_sha256": row.get("checkpoint_sha256"),
        "audit_failed_count": int(row["audit_failed_count"]),
        "memory_mean_pair_overlap": float(row["memory_mean_pair_overlap"]),
        "memory_min_pair_overlap": float(row["memory_min_pair_overlap"]),
        "memory_max_pair_state_norm_delta": float(row["memory_max_pair_state_norm_delta"]),
        "memory_max_pair_state_l2_distance": float(row["memory_max_pair_state_l2_distance"]),
        "memory_min_pair_state_cosine_similarity": float(row["memory_min_pair_state_cosine_similarity"]),
    }


def compare_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    loss_delta = float(candidate["benchmark_loss"]) - float(baseline["benchmark_loss"])
    accuracy_delta = float(candidate["benchmark_accuracy"]) - float(baseline["benchmark_accuracy"])
    throughput_delta = (
        float(candidate["benchmark_best_throughput_tokens_s"])
        - float(baseline["benchmark_best_throughput_tokens_s"])
    )
    overlap_delta = float(candidate["memory_mean_pair_overlap"]) - float(baseline["memory_mean_pair_overlap"])
    min_overlap_delta = float(candidate["memory_min_pair_overlap"]) - float(baseline["memory_min_pair_overlap"])
    cosine_delta = (
        float(candidate["memory_min_pair_state_cosine_similarity"])
        - float(baseline["memory_min_pair_state_cosine_similarity"])
    )
    l2_delta = (
        float(candidate["memory_max_pair_state_l2_distance"])
        - float(baseline["memory_max_pair_state_l2_distance"])
    )
    norm_delta = (
        float(candidate["memory_max_pair_state_norm_delta"])
        - float(baseline["memory_max_pair_state_norm_delta"])
    )
    audit_failed_delta = int(candidate["audit_failed_count"]) - int(baseline["audit_failed_count"])
    return {
        "winner_by_loss": (
            "candidate"
            if float(candidate["benchmark_loss"]) < float(baseline["benchmark_loss"])
            else "baseline"
            if float(candidate["benchmark_loss"]) > float(baseline["benchmark_loss"])
            else "tie"
        ),
        "loss_delta": loss_delta,
        "accuracy_delta": accuracy_delta,
        "best_throughput_tokens_s_delta": throughput_delta,
        "loss_regression_rate": regression_rate(
            float(baseline["benchmark_loss"]),
            float(candidate["benchmark_loss"]),
            higher_is_better=False,
        ),
        "accuracy_regression": max(0.0, -accuracy_delta),
        "best_throughput_regression_rate": regression_rate(
            float(baseline["benchmark_best_throughput_tokens_s"]),
            float(candidate["benchmark_best_throughput_tokens_s"]),
            higher_is_better=True,
        ),
        "same_checkpoint_sha256": baseline.get("checkpoint_sha256") == candidate.get("checkpoint_sha256"),
        "audit_failed_count_delta": audit_failed_delta,
        "memory_mean_pair_overlap_delta": overlap_delta,
        "memory_min_pair_overlap_delta": min_overlap_delta,
        "memory_min_pair_state_cosine_similarity_delta": cosine_delta,
        "memory_max_pair_state_l2_distance_delta": l2_delta,
        "memory_max_pair_state_norm_delta_delta": norm_delta,
    }


def baseline_for(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    latest = rows[-1]
    if mode == "previous":
        return rows[-2] if len(rows) > 1 else latest
    if mode == "first":
        return rows[0]
    if mode == "best-loss":
        return best_by(rows, "benchmark_loss", higher_is_better=False)
    if mode == "best-throughput":
        return best_by(rows, "benchmark_best_throughput_tokens_s", higher_is_better=True)
    if mode == "best-memory":
        return best_by(rows, "memory_mean_pair_overlap", higher_is_better=True)
    raise ReleaseHistoryAnalysisError(f"Unknown baseline mode: {mode}")


def threshold_failures(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.fail_on_loss_regression and comparison["loss_regression_rate"] > args.max_loss_regression:
        failures.append(
            f"loss regression {comparison['loss_regression_rate']:.4f} > {args.max_loss_regression:.4f}"
        )
    if (
        args.fail_on_throughput_regression
        and comparison["best_throughput_regression_rate"] > args.max_throughput_regression
    ):
        failures.append(
            "best throughput regression "
            f"{comparison['best_throughput_regression_rate']:.4f} > {args.max_throughput_regression:.4f}"
        )
    if args.fail_on_accuracy_regression and comparison["accuracy_regression"] > args.max_accuracy_regression:
        failures.append(
            f"accuracy regression {comparison['accuracy_regression']:.4f} > {args.max_accuracy_regression:.4f}"
        )
    if args.fail_on_audit_regression and comparison["audit_failed_count_delta"] > args.max_audit_failed_regression:
        failures.append(
            "audit failed-count regression "
            f"{comparison['audit_failed_count_delta']} > {args.max_audit_failed_regression}"
        )
    if args.fail_on_memory_overlap_regression:
        overlap_regression = -comparison["memory_mean_pair_overlap_delta"]
        if overlap_regression > args.max_memory_overlap_regression:
            failures.append(
                "memory mean pair overlap regression "
                f"{overlap_regression:.4f} > {args.max_memory_overlap_regression:.4f}"
            )
    if args.fail_on_memory_cosine_regression:
        cosine_regression = -comparison["memory_min_pair_state_cosine_similarity_delta"]
        if cosine_regression > args.max_memory_cosine_regression:
            failures.append(
                "memory min cosine regression "
                f"{cosine_regression:.4f} > {args.max_memory_cosine_regression:.4f}"
            )
    if args.fail_on_memory_l2_regression:
        l2_regression = comparison["memory_max_pair_state_l2_distance_delta"]
        if l2_regression > args.max_memory_l2_regression:
            failures.append(
                f"memory max L2 regression {l2_regression:.4f} > {args.max_memory_l2_regression:.4f}"
            )
    if args.fail_on_memory_norm_regression:
        norm_regression = comparison["memory_max_pair_state_norm_delta_delta"]
        if norm_regression > args.max_memory_norm_regression:
            failures.append(
                "memory max norm-delta regression "
                f"{norm_regression:.4f} > {args.max_memory_norm_regression:.4f}"
            )
    return failures


def validation_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        history=args.history,
        output_json=None,
        require_artifacts=args.require_artifacts,
        expected_releases=args.expected_releases,
        require_unique_checkpoints=args.require_unique_checkpoints,
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    validation = validate(validation_args(args))
    if not validation["valid"]:
        raise ReleaseHistoryAnalysisError(f"Release history validation failed: {validation['errors']}")
    payload = load_json(Path(args.history))
    rows = payload["dashboards"]
    if not rows:
        raise ReleaseHistoryAnalysisError("Release history has no dashboards.")
    latest = rows[-1]
    baseline = baseline_for(rows, args.baseline)
    comparison = compare_rows(baseline, latest)
    failures = threshold_failures(comparison, args)
    summary = payload.get("summary") or {}
    result = {
        "history": str(args.history),
        "baseline_mode": args.baseline,
        "release_count": len(rows),
        "summary": summary,
        "baseline": compact(baseline),
        "latest": compact(latest),
        "comparison": comparison,
        "failures": failures,
        "valid": not failures,
    }
    return result


def save_if_requested(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.output_json:
        save_json(args.output_json, payload)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if abs(value_f) >= 1000:
        return f"{value_f:,.0f}"
    return f"{value_f:.{digits}f}"


def print_row(label: str, row: dict[str, Any]) -> None:
    print(label)
    print(f"  name: {row['name']}")
    print(f"  history_index: {row['history_index']}")
    print(f"  checkpoint_sha256: {str(row.get('checkpoint_sha256') or '')[:12]}")
    print(f"  loss: {fmt(row['benchmark_loss'])}")
    print(f"  accuracy: {fmt(row['benchmark_accuracy'])}")
    print(f"  best_throughput_tokens_s: {fmt(row['benchmark_best_throughput_tokens_s'], 0)}")
    print(f"  memory_mean_pair_overlap: {fmt(row['memory_mean_pair_overlap'])}")
    print(f"  memory_min_pair_state_cosine_similarity: {fmt(row['memory_min_pair_state_cosine_similarity'])}")


def run(args: argparse.Namespace) -> int:
    payload = analyze(args)
    save_if_requested(args, payload)
    comparison = payload["comparison"]
    print("TextPy/SoA release history analysis")
    print(f"history: {payload['history']}")
    print(f"baseline_mode: {payload['baseline_mode']}")
    print(f"releases: {payload['release_count']}")
    print("")
    print_row("baseline", payload["baseline"])
    print("")
    print_row("latest", payload["latest"])
    print("")
    print("Comparison")
    print(f"  winner_by_loss: {comparison['winner_by_loss']}")
    print(f"  loss_delta: {comparison['loss_delta']:.6f}")
    print(f"  accuracy_delta: {comparison['accuracy_delta']:.6f}")
    print(f"  best_throughput_tokens_s_delta: {comparison['best_throughput_tokens_s_delta']:,.0f}")
    print(f"  loss_regression_rate: {comparison['loss_regression_rate']:.4f}")
    print(f"  best_throughput_regression_rate: {comparison['best_throughput_regression_rate']:.4f}")
    print(f"  audit_failed_count_delta: {comparison['audit_failed_count_delta']}")
    print(f"  memory_mean_pair_overlap_delta: {comparison['memory_mean_pair_overlap_delta']:.6f}")
    print(f"  memory_min_pair_state_cosine_similarity_delta: {comparison['memory_min_pair_state_cosine_similarity_delta']:.6f}")
    print(f"  memory_max_pair_state_l2_distance_delta: {comparison['memory_max_pair_state_l2_distance_delta']:.6f}")
    print(f"  same_checkpoint_sha256: {comparison['same_checkpoint_sha256']}")
    if args.output_json:
        print("")
        print(f"Saved JSON: {args.output_json}")
    if payload["failures"]:
        print("")
        print("FAIL")
        for failure in payload["failures"]:
            print(f"  {failure}")
        return 1
    print("")
    print("RELEASE HISTORY ANALYSIS OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze release history trends and gate latest release.")
    parser.add_argument("--history", default="artifacts/releases/text_release_history.json")
    parser.add_argument("--baseline", default="previous", choices=["previous", "first", "best-loss", "best-throughput", "best-memory"])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--expected-releases", type=int, default=None)
    parser.add_argument("--require-unique-checkpoints", action="store_true")
    parser.add_argument("--fail-on-loss-regression", action="store_true")
    parser.add_argument("--max-loss-regression", type=float, default=0.01)
    parser.add_argument("--fail-on-throughput-regression", action="store_true")
    parser.add_argument("--max-throughput-regression", type=float, default=0.05)
    parser.add_argument("--fail-on-accuracy-regression", action="store_true")
    parser.add_argument("--max-accuracy-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-audit-regression", action="store_true")
    parser.add_argument("--max-audit-failed-regression", type=int, default=0)
    parser.add_argument("--fail-on-memory-overlap-regression", action="store_true")
    parser.add_argument("--max-memory-overlap-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-memory-cosine-regression", action="store_true")
    parser.add_argument("--max-memory-cosine-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-memory-l2-regression", action="store_true")
    parser.add_argument("--max-memory-l2-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-memory-norm-regression", action="store_true")
    parser.add_argument("--max-memory-norm-regression", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseHistoryAnalysisError as exc:
        raise SystemExit(str(exc)) from exc
