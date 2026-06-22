#!/usr/bin/env python3
"""Compare two promoted TextPy/SoA text packages as release artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from validate_text_package import validate


class PackageComparisonError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PackageComparisonError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PackageComparisonError(f"Invalid JSON file: {path}: {exc}") from exc


def validate_package(manifest: Path, args: argparse.Namespace) -> dict[str, Any]:
    validation_args = argparse.Namespace(
        manifest=str(manifest),
        output_json=None,
        model_card_json=args.model_card_json,
        model_card_md=args.model_card_md,
        smoke_json=args.smoke_json,
        benchmark_json=args.benchmark_json,
        require_model_card=True,
        require_smoke=True,
        require_benchmark=True,
    )
    validation = validate(validation_args)
    if not validation["valid"]:
        raise PackageComparisonError(f"Package validation failed for {manifest}: {validation['errors']}")

    package_dir = manifest.parent
    benchmark_path = package_dir / args.benchmark_json
    benchmark = load_json(benchmark_path)
    manifest_payload = load_json(manifest)

    return {
        "manifest": str(manifest),
        "manifest_payload": manifest_payload,
        "checkpoint": validation["checkpoint"],
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "checkpoint_bytes": validation["checkpoint_bytes"],
        "param_count": validation["param_count"],
        "validation": validation,
        "benchmark_json": str(benchmark_path),
        "benchmark": benchmark,
        "loss": float(benchmark["loss"]),
        "accuracy": float(benchmark["accuracy"]),
        "best_throughput_tokens_s": float(benchmark["best_throughput_tokens_s"]),
        "mean_throughput_tokens_s": float(benchmark["mean_throughput_tokens_s"]),
        "aggregate_throughput_tokens_s": float(benchmark["aggregate_throughput_tokens_s"]),
        "source_tokens": int(benchmark["source_tokens"]),
        "vocab_size": int(benchmark["vocab_size"]),
        "streams": int(benchmark["streams"]),
        "tokens_per_stream": int(benchmark["tokens_per_stream"]),
        "seq_len": int(benchmark["seq_len"]),
        "repeat": int(benchmark["repeat"]),
        "sample": benchmark.get("sample"),
    }


def regression_rate(base: float, candidate: float, *, higher_is_better: bool) -> float:
    if math.isclose(base, 0.0, abs_tol=1.0e-12):
        raw = base - candidate if higher_is_better else candidate - base
    elif higher_is_better:
        raw = (base - candidate) / abs(base)
    else:
        raw = (candidate - base) / abs(base)
    return raw


def compare_packages(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    loss_delta = candidate["loss"] - baseline["loss"]
    accuracy_delta = candidate["accuracy"] - baseline["accuracy"]
    best_throughput_delta = candidate["best_throughput_tokens_s"] - baseline["best_throughput_tokens_s"]
    mean_throughput_delta = candidate["mean_throughput_tokens_s"] - baseline["mean_throughput_tokens_s"]
    aggregate_throughput_delta = (
        candidate["aggregate_throughput_tokens_s"] - baseline["aggregate_throughput_tokens_s"]
    )
    checkpoint_bytes_delta = int(candidate["checkpoint_bytes"] or 0) - int(baseline["checkpoint_bytes"] or 0)
    param_count_delta = int(candidate["param_count"] or 0) - int(baseline["param_count"] or 0)

    if candidate["loss"] < baseline["loss"]:
        winner = "candidate"
    elif candidate["loss"] > baseline["loss"]:
        winner = "baseline"
    elif candidate["accuracy"] > baseline["accuracy"]:
        winner = "candidate"
    elif candidate["accuracy"] < baseline["accuracy"]:
        winner = "baseline"
    else:
        winner = "tie"

    return {
        "winner_by_loss": winner,
        "loss_delta": loss_delta,
        "accuracy_delta": accuracy_delta,
        "best_throughput_tokens_s_delta": best_throughput_delta,
        "mean_throughput_tokens_s_delta": mean_throughput_delta,
        "aggregate_throughput_tokens_s_delta": aggregate_throughput_delta,
        "checkpoint_bytes_delta": checkpoint_bytes_delta,
        "param_count_delta": param_count_delta,
        "loss_regression_rate": regression_rate(baseline["loss"], candidate["loss"], higher_is_better=False),
        "accuracy_regression": max(0.0, -accuracy_delta),
        "best_throughput_regression_rate": regression_rate(
            baseline["best_throughput_tokens_s"],
            candidate["best_throughput_tokens_s"],
            higher_is_better=True,
        ),
        "mean_throughput_regression_rate": regression_rate(
            baseline["mean_throughput_tokens_s"],
            candidate["mean_throughput_tokens_s"],
            higher_is_better=True,
        ),
        "same_checkpoint_sha256": baseline["checkpoint_sha256"] == candidate["checkpoint_sha256"],
        "same_eval_shape": all(
            baseline[name] == candidate[name]
            for name in ["streams", "tokens_per_stream", "seq_len", "source_tokens", "vocab_size"]
        ),
    }


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_package(label: str, package: dict[str, Any]) -> None:
    print(label)
    print(f"  manifest: {package['manifest']}")
    print(f"  checkpoint: {package['checkpoint']}")
    print(f"  checkpoint_sha256: {package['checkpoint_sha256']}")
    print(f"  checkpoint_bytes: {package['checkpoint_bytes']}")
    print(f"  param_count: {package['param_count']:,}")
    print(f"  loss: {package['loss']:.4f}")
    print(f"  accuracy: {package['accuracy']:.4f}")
    print(f"  best_throughput_tokens_s: {package['best_throughput_tokens_s']:,.0f}")
    print(f"  mean_throughput_tokens_s: {package['mean_throughput_tokens_s']:,.0f}")
    print(f"  eval_shape: {package['streams']}x{package['seq_len']} of {package['tokens_per_stream']}")
    print(f"  sample: {package['sample']}")


def run(args: argparse.Namespace) -> int:
    baseline = validate_package(Path(args.baseline_manifest), args)
    candidate = validate_package(Path(args.candidate_manifest), args)
    comparison = compare_packages(baseline, candidate)
    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }

    print("TextPy/SoA text package comparison")
    print("")
    print_package("baseline", baseline)
    print("")
    print_package("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  winner_by_loss: {comparison['winner_by_loss']}")
    print(f"  loss_delta: {comparison['loss_delta']:.6f}")
    print(f"  accuracy_delta: {comparison['accuracy_delta']:.6f}")
    print(f"  best_throughput_tokens_s_delta: {comparison['best_throughput_tokens_s_delta']:,.0f}")
    print(f"  mean_throughput_tokens_s_delta: {comparison['mean_throughput_tokens_s_delta']:,.0f}")
    print(f"  loss_regression_rate: {comparison['loss_regression_rate']:.4f}")
    print(f"  best_throughput_regression_rate: {comparison['best_throughput_regression_rate']:.4f}")
    print(f"  mean_throughput_regression_rate: {comparison['mean_throughput_regression_rate']:.4f}")
    print(f"  same_checkpoint_sha256: {comparison['same_checkpoint_sha256']}")
    print(f"  same_eval_shape: {comparison['same_eval_shape']}")

    if args.output_json:
        save_json(args.output_json, payload)
        print("")
        print(f"Saved JSON: {args.output_json}")

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
    if (
        args.fail_on_mean_throughput_regression
        and comparison["mean_throughput_regression_rate"] > args.max_throughput_regression
    ):
        failures.append(
            "mean throughput regression "
            f"{comparison['mean_throughput_regression_rate']:.4f} > {args.max_throughput_regression:.4f}"
        )
    if args.fail_on_accuracy_regression and comparison["accuracy_regression"] > args.max_accuracy_regression:
        failures.append(
            f"accuracy regression {comparison['accuracy_regression']:.4f} > {args.max_accuracy_regression:.4f}"
        )

    if failures:
        print("")
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("")
    print("PACKAGE COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two promoted text packages.")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--model-card-json", default="model_card.json")
    parser.add_argument("--model-card-md", default="MODEL_CARD.md")
    parser.add_argument("--smoke-json", default="smoke_run.json")
    parser.add_argument("--benchmark-json", default="benchmark.json")
    parser.add_argument("--fail-on-loss-regression", action="store_true")
    parser.add_argument("--max-loss-regression", type=float, default=0.01)
    parser.add_argument("--fail-on-throughput-regression", action="store_true")
    parser.add_argument("--fail-on-mean-throughput-regression", action="store_true")
    parser.add_argument("--max-throughput-regression", type=float, default=0.25)
    parser.add_argument("--fail-on-accuracy-regression", action="store_true")
    parser.add_argument("--max-accuracy-regression", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except PackageComparisonError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
