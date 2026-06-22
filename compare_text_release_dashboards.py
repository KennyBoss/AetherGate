#!/usr/bin/env python3
"""Compare two TextPy/SoA release dashboard JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from validate_text_release_dashboard import validate


class DashboardComparisonError(RuntimeError):
    pass


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def regression_rate(base: float, candidate: float, *, higher_is_better: bool) -> float:
    if math.isclose(base, 0.0, abs_tol=1.0e-12):
        raw = base - candidate if higher_is_better else candidate - base
    elif higher_is_better:
        raw = (base - candidate) / abs(base)
    else:
        raw = (candidate - base) / abs(base)
    return raw


def load_valid_dashboard(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    validation_args = argparse.Namespace(
        dashboard=str(path),
        output_json=None,
        require_artifacts=args.require_artifacts,
        expected_checks=args.expected_checks,
        expected_prompts=args.expected_prompts,
        expected_pairs=args.expected_pairs,
        min_mean_pair_overlap=args.min_mean_pair_overlap,
    )
    validation = validate(validation_args)
    if not validation["valid"]:
        raise DashboardComparisonError(f"Dashboard validation failed for {path}: {validation['errors']}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    dashboard = payload["dashboard"]
    memory = dashboard["memory"]
    audit = payload["audit"]
    return {
        "path": str(path),
        "payload": payload,
        "validation": validation,
        "name": dashboard.get("name"),
        "rank": int(dashboard["rank"]),
        "release_count": int(dashboard["release_count"]),
        "benchmark_loss": float(dashboard["benchmark_loss"]),
        "benchmark_accuracy": float(dashboard["benchmark_accuracy"]),
        "benchmark_best_throughput_tokens_s": float(dashboard["benchmark_best_throughput_tokens_s"]),
        "checkpoint_sha256": dashboard.get("checkpoint_sha256"),
        "archive_sha256": dashboard.get("archive_sha256"),
        "audit_valid": bool(audit["valid"]),
        "audit_check_count": int(audit["check_count"]),
        "audit_failed_count": int(audit["failed_count"]),
        "memory_prompt_count": int(memory["prompt_count"]),
        "memory_pair_count": int(memory["pair_count"]),
        "memory_mean_pair_overlap": float(memory["mean_pair_overlap"]),
        "memory_min_pair_overlap": float(memory["min_pair_overlap"]),
        "memory_max_pair_state_norm_delta": float(memory["max_pair_state_norm_delta"]),
        "memory_max_pair_state_l2_distance": float(memory["max_pair_state_l2_distance"]),
        "memory_min_pair_state_cosine_similarity": float(memory["min_pair_state_cosine_similarity"]),
    }


def compact_dashboard(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "name": row["name"],
        "rank": row["rank"],
        "release_count": row["release_count"],
        "benchmark_loss": row["benchmark_loss"],
        "benchmark_accuracy": row["benchmark_accuracy"],
        "benchmark_best_throughput_tokens_s": row["benchmark_best_throughput_tokens_s"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "archive_sha256": row["archive_sha256"],
        "audit_valid": row["audit_valid"],
        "audit_check_count": row["audit_check_count"],
        "audit_failed_count": row["audit_failed_count"],
        "memory_prompt_count": row["memory_prompt_count"],
        "memory_pair_count": row["memory_pair_count"],
        "memory_mean_pair_overlap": row["memory_mean_pair_overlap"],
        "memory_min_pair_overlap": row["memory_min_pair_overlap"],
        "memory_max_pair_state_norm_delta": row["memory_max_pair_state_norm_delta"],
        "memory_max_pair_state_l2_distance": row["memory_max_pair_state_l2_distance"],
        "memory_min_pair_state_cosine_similarity": row["memory_min_pair_state_cosine_similarity"],
    }


def compare_dashboards(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    loss_delta = candidate["benchmark_loss"] - baseline["benchmark_loss"]
    accuracy_delta = candidate["benchmark_accuracy"] - baseline["benchmark_accuracy"]
    throughput_delta = (
        candidate["benchmark_best_throughput_tokens_s"] - baseline["benchmark_best_throughput_tokens_s"]
    )
    overlap_delta = candidate["memory_mean_pair_overlap"] - baseline["memory_mean_pair_overlap"]
    min_overlap_delta = candidate["memory_min_pair_overlap"] - baseline["memory_min_pair_overlap"]
    cosine_delta = (
        candidate["memory_min_pair_state_cosine_similarity"]
        - baseline["memory_min_pair_state_cosine_similarity"]
    )
    l2_delta = candidate["memory_max_pair_state_l2_distance"] - baseline["memory_max_pair_state_l2_distance"]
    norm_delta = candidate["memory_max_pair_state_norm_delta"] - baseline["memory_max_pair_state_norm_delta"]
    audit_failed_delta = candidate["audit_failed_count"] - baseline["audit_failed_count"]

    return {
        "winner_by_loss": (
            "candidate"
            if candidate["benchmark_loss"] < baseline["benchmark_loss"]
            else "baseline"
            if candidate["benchmark_loss"] > baseline["benchmark_loss"]
            else "tie"
        ),
        "loss_delta": loss_delta,
        "accuracy_delta": accuracy_delta,
        "best_throughput_tokens_s_delta": throughput_delta,
        "loss_regression_rate": regression_rate(
            baseline["benchmark_loss"],
            candidate["benchmark_loss"],
            higher_is_better=False,
        ),
        "accuracy_regression": max(0.0, -accuracy_delta),
        "best_throughput_regression_rate": regression_rate(
            baseline["benchmark_best_throughput_tokens_s"],
            candidate["benchmark_best_throughput_tokens_s"],
            higher_is_better=True,
        ),
        "same_checkpoint_sha256": baseline["checkpoint_sha256"] == candidate["checkpoint_sha256"],
        "same_archive_sha256": baseline["archive_sha256"] == candidate["archive_sha256"],
        "audit_check_count_delta": candidate["audit_check_count"] - baseline["audit_check_count"],
        "audit_failed_count_delta": audit_failed_delta,
        "memory_prompt_count_delta": candidate["memory_prompt_count"] - baseline["memory_prompt_count"],
        "memory_pair_count_delta": candidate["memory_pair_count"] - baseline["memory_pair_count"],
        "memory_mean_pair_overlap_delta": overlap_delta,
        "memory_min_pair_overlap_delta": min_overlap_delta,
        "memory_min_pair_state_cosine_similarity_delta": cosine_delta,
        "memory_max_pair_state_l2_distance_delta": l2_delta,
        "memory_max_pair_state_norm_delta_delta": norm_delta,
    }


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
    if args.fail_on_memory_l2_regression and comparison["memory_max_pair_state_l2_distance_delta"] > args.max_memory_l2_regression:
        failures.append(
            "memory max L2 regression "
            f"{comparison['memory_max_pair_state_l2_distance_delta']:.4f} > {args.max_memory_l2_regression:.4f}"
        )
    if (
        args.fail_on_memory_norm_regression
        and comparison["memory_max_pair_state_norm_delta_delta"] > args.max_memory_norm_regression
    ):
        failures.append(
            "memory max norm regression "
            f"{comparison['memory_max_pair_state_norm_delta_delta']:.4f} > {args.max_memory_norm_regression:.4f}"
        )
    return failures


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if abs(value_f) >= 1000:
        return f"{value_f:,.0f}"
    return f"{value_f:.{digits}f}"


def print_dashboard(label: str, row: dict[str, Any]) -> None:
    print(label)
    print(f"  path: {row['path']}")
    print(f"  name: {row['name']}")
    print(f"  checkpoint_sha256: {str(row['checkpoint_sha256'] or '')[:12]}")
    print(f"  loss: {fmt(row['benchmark_loss'])}")
    print(f"  accuracy: {fmt(row['benchmark_accuracy'])}")
    print(f"  best_throughput_tokens_s: {fmt(row['benchmark_best_throughput_tokens_s'], 0)}")
    print(f"  audit_checks: {row['audit_check_count']}")
    print(f"  memory_mean_pair_overlap: {fmt(row['memory_mean_pair_overlap'])}")
    print(f"  memory_min_pair_state_cosine_similarity: {fmt(row['memory_min_pair_state_cosine_similarity'])}")


def run(args: argparse.Namespace) -> int:
    baseline = load_valid_dashboard(Path(args.baseline_dashboard), args)
    candidate = load_valid_dashboard(Path(args.candidate_dashboard), args)
    comparison = compare_dashboards(baseline, candidate)
    failures = threshold_failures(comparison, args)
    payload = {
        "baseline": compact_dashboard(baseline),
        "candidate": compact_dashboard(candidate),
        "comparison": comparison,
        "failures": failures,
    }

    print("TextPy/SoA release dashboard comparison")
    print("")
    print_dashboard("baseline", baseline)
    print("")
    print_dashboard("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  winner_by_loss: {comparison['winner_by_loss']}")
    print(f"  loss_delta: {comparison['loss_delta']:.6f}")
    print(f"  accuracy_delta: {comparison['accuracy_delta']:.6f}")
    print(f"  best_throughput_tokens_s_delta: {comparison['best_throughput_tokens_s_delta']:,.0f}")
    print(f"  audit_failed_count_delta: {comparison['audit_failed_count_delta']}")
    print(f"  memory_mean_pair_overlap_delta: {comparison['memory_mean_pair_overlap_delta']:.6f}")
    print(f"  memory_min_pair_state_cosine_similarity_delta: {comparison['memory_min_pair_state_cosine_similarity_delta']:.6f}")
    print(f"  memory_max_pair_state_l2_distance_delta: {comparison['memory_max_pair_state_l2_distance_delta']:.6f}")
    print(f"  same_checkpoint_sha256: {comparison['same_checkpoint_sha256']}")

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
    print("RELEASE DASHBOARD COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two text release dashboard JSON artifacts.")
    parser.add_argument("--baseline-dashboard", required=True)
    parser.add_argument("--candidate-dashboard", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--expected-checks", type=int, default=11)
    parser.add_argument("--expected-prompts", type=int, default=None)
    parser.add_argument("--expected-pairs", type=int, default=None)
    parser.add_argument("--min-mean-pair-overlap", type=float, default=None)
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
    raise SystemExit(run(parse_args()))
