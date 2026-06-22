#!/usr/bin/env python3
"""Compare two registered TextPy/SoA text releases including memory diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from list_text_releases import sort_releases


class ReleaseComparisonError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseComparisonError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseComparisonError(f"Invalid JSON file {path}: {exc}") from exc


def pick_registry_release(path: Path, rank: int, sort: str, name: str | None, sha256: str | None) -> dict[str, Any]:
    registry = load_json(path)
    releases = sort_releases(list(registry.get("releases", [])), sort)
    if not releases:
        raise ReleaseComparisonError(f"Registry has no releases: {path}")
    if name:
        for release in releases:
            if release.get("name") == name:
                return release
        raise ReleaseComparisonError(f"Registry {path} has no release named {name!r}")
    if sha256:
        for release in releases:
            if str(release.get("checkpoint_sha256", "")).startswith(sha256):
                return release
        raise ReleaseComparisonError(f"Registry {path} has no checkpoint SHA prefix {sha256!r}")
    if rank < 1 or rank > len(releases):
        raise ReleaseComparisonError(f"Rank {rank} is outside registry range 1..{len(releases)}")
    return releases[rank - 1]


def release_from_pointer(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    release = {
        "name": payload.get("name"),
        "manifest": payload.get("manifest"),
        "archive": payload.get("archive"),
        "checkpoint": payload.get("checkpoint"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "archive_sha256": payload.get("archive_sha256"),
        "benchmark_loss": payload.get("benchmark_loss"),
        "benchmark_accuracy": payload.get("benchmark_accuracy"),
        "benchmark_best_throughput_tokens_s": payload.get("benchmark_best_throughput_tokens_s"),
        "benchmark_mean_throughput_tokens_s": payload.get("benchmark_mean_throughput_tokens_s"),
        "param_count": payload.get("param_count"),
        "comparison": payload.get("comparison"),
        "memory": payload.get("memory"),
        "_source_pointer": str(path),
    }
    validation = payload.get("validation") or {}
    if release["checkpoint"] is None:
        release["checkpoint"] = validation.get("checkpoint")
    return release


def resolve_release(args: argparse.Namespace, side: str) -> dict[str, Any]:
    pointer = getattr(args, f"{side}_release")
    registry = getattr(args, f"{side}_registry")
    if pointer:
        return release_from_pointer(Path(pointer))
    if registry:
        return pick_registry_release(
            Path(registry),
            getattr(args, f"{side}_rank"),
            getattr(args, f"{side}_sort"),
            getattr(args, f"{side}_name"),
            getattr(args, f"{side}_checkpoint_sha256"),
        )
    raise ReleaseComparisonError(f"Provide --{side}-release or --{side}-registry.")


def number(value: Any, label: str, *, required: bool = True) -> float | None:
    if value is None:
        if required:
            raise ReleaseComparisonError(f"Release is missing {label}.")
        return None
    return float(value)


def memory_number(memory: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(memory, dict):
        return None
    value = memory.get(key)
    return None if value is None else float(value)


def regression_rate(base: float, candidate: float, *, higher_is_better: bool) -> float:
    if math.isclose(base, 0.0, abs_tol=1.0e-12):
        raw = base - candidate if higher_is_better else candidate - base
    elif higher_is_better:
        raw = (base - candidate) / abs(base)
    else:
        raw = (candidate - base) / abs(base)
    return raw


def compact_release(release: dict[str, Any]) -> dict[str, Any]:
    memory = release.get("memory") or {}
    return {
        "name": release.get("name"),
        "manifest": release.get("manifest"),
        "archive": release.get("archive"),
        "checkpoint": release.get("checkpoint"),
        "checkpoint_sha256": release.get("checkpoint_sha256"),
        "archive_sha256": release.get("archive_sha256"),
        "benchmark_loss": release.get("benchmark_loss"),
        "benchmark_accuracy": release.get("benchmark_accuracy"),
        "benchmark_best_throughput_tokens_s": release.get("benchmark_best_throughput_tokens_s"),
        "benchmark_mean_throughput_tokens_s": release.get("benchmark_mean_throughput_tokens_s"),
        "param_count": release.get("param_count"),
        "memory": memory if memory else None,
        "source_pointer": release.get("_source_pointer"),
    }


def compare_releases(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_loss = number(baseline.get("benchmark_loss"), "benchmark_loss")
    candidate_loss = number(candidate.get("benchmark_loss"), "benchmark_loss")
    baseline_accuracy = number(baseline.get("benchmark_accuracy"), "benchmark_accuracy")
    candidate_accuracy = number(candidate.get("benchmark_accuracy"), "benchmark_accuracy")
    baseline_best = number(baseline.get("benchmark_best_throughput_tokens_s"), "best throughput")
    candidate_best = number(candidate.get("benchmark_best_throughput_tokens_s"), "best throughput")
    baseline_mean = number(baseline.get("benchmark_mean_throughput_tokens_s"), "mean throughput")
    candidate_mean = number(candidate.get("benchmark_mean_throughput_tokens_s"), "mean throughput")
    baseline_memory = baseline.get("memory") or {}
    candidate_memory = candidate.get("memory") or {}
    baseline_overlap = memory_number(baseline_memory, "mean_pair_overlap")
    candidate_overlap = memory_number(candidate_memory, "mean_pair_overlap")
    baseline_min_overlap = memory_number(baseline_memory, "min_pair_overlap")
    candidate_min_overlap = memory_number(candidate_memory, "min_pair_overlap")
    baseline_cosine = memory_number(baseline_memory, "min_pair_state_cosine_similarity")
    candidate_cosine = memory_number(candidate_memory, "min_pair_state_cosine_similarity")
    baseline_l2 = memory_number(baseline_memory, "max_pair_state_l2_distance")
    candidate_l2 = memory_number(candidate_memory, "max_pair_state_l2_distance")
    baseline_norm = memory_number(baseline_memory, "max_pair_state_norm_delta")
    candidate_norm = memory_number(candidate_memory, "max_pair_state_norm_delta")

    return {
        "winner_by_loss": (
            "candidate"
            if candidate_loss < baseline_loss
            else "baseline"
            if candidate_loss > baseline_loss
            else "tie"
        ),
        "loss_delta": candidate_loss - baseline_loss,
        "accuracy_delta": candidate_accuracy - baseline_accuracy,
        "best_throughput_tokens_s_delta": candidate_best - baseline_best,
        "mean_throughput_tokens_s_delta": candidate_mean - baseline_mean,
        "loss_regression_rate": regression_rate(baseline_loss, candidate_loss, higher_is_better=False),
        "accuracy_regression": max(0.0, baseline_accuracy - candidate_accuracy),
        "best_throughput_regression_rate": regression_rate(baseline_best, candidate_best, higher_is_better=True),
        "mean_throughput_regression_rate": regression_rate(baseline_mean, candidate_mean, higher_is_better=True),
        "same_checkpoint_sha256": baseline.get("checkpoint_sha256") == candidate.get("checkpoint_sha256"),
        "same_archive_sha256": baseline.get("archive_sha256") == candidate.get("archive_sha256"),
        "has_memory_diagnostics": bool(baseline_memory) and bool(candidate_memory),
        "memory_mean_pair_overlap_delta": (
            candidate_overlap - baseline_overlap
            if baseline_overlap is not None and candidate_overlap is not None
            else None
        ),
        "memory_min_pair_overlap_delta": (
            candidate_min_overlap - baseline_min_overlap
            if baseline_min_overlap is not None and candidate_min_overlap is not None
            else None
        ),
        "memory_min_pair_state_cosine_similarity_delta": (
            candidate_cosine - baseline_cosine
            if baseline_cosine is not None and candidate_cosine is not None
            else None
        ),
        "memory_max_pair_state_l2_distance_delta": (
            candidate_l2 - baseline_l2 if baseline_l2 is not None and candidate_l2 is not None else None
        ),
        "memory_max_pair_state_norm_delta_delta": (
            candidate_norm - baseline_norm if baseline_norm is not None and candidate_norm is not None else None
        ),
        "baseline_memory_mean_pair_overlap": baseline_overlap,
        "candidate_memory_mean_pair_overlap": candidate_overlap,
        "baseline_memory_min_pair_state_cosine_similarity": baseline_cosine,
        "candidate_memory_min_pair_state_cosine_similarity": candidate_cosine,
    }


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if abs(value_f) >= 1000:
        return f"{value_f:,.0f}"
    return f"{value_f:.{digits}f}"


def print_release(label: str, release: dict[str, Any]) -> None:
    memory = release.get("memory") or {}
    print(label)
    print(f"  name: {release.get('name')}")
    print(f"  manifest: {release.get('manifest')}")
    print(f"  checkpoint_sha256: {str(release.get('checkpoint_sha256') or '')[:12]}")
    print(f"  archive_sha256: {str(release.get('archive_sha256') or '')[:12]}")
    print(f"  loss: {fmt(release.get('benchmark_loss'))}")
    print(f"  accuracy: {fmt(release.get('benchmark_accuracy'))}")
    print(f"  best_throughput_tokens_s: {fmt(release.get('benchmark_best_throughput_tokens_s'), 0)}")
    if memory:
        print(f"  memory_mean_pair_overlap: {fmt(memory.get('mean_pair_overlap'))}")
        print(f"  memory_min_pair_state_cosine_similarity: {fmt(memory.get('min_pair_state_cosine_similarity'))}")


def check_thresholds(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.require_memory and not comparison["has_memory_diagnostics"]:
        failures.append("memory diagnostics are required but missing on at least one release")
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
    if comparison["has_memory_diagnostics"]:
        overlap_delta = comparison["memory_mean_pair_overlap_delta"]
        cosine_delta = comparison["memory_min_pair_state_cosine_similarity_delta"]
        l2_delta = comparison["memory_max_pair_state_l2_distance_delta"]
        norm_delta = comparison["memory_max_pair_state_norm_delta_delta"]
        if args.fail_on_memory_overlap_regression and overlap_delta is not None:
            if -overlap_delta > args.max_memory_overlap_regression:
                failures.append(
                    "memory mean pair overlap regression "
                    f"{-overlap_delta:.4f} > {args.max_memory_overlap_regression:.4f}"
                )
        if args.fail_on_memory_cosine_regression and cosine_delta is not None:
            if -cosine_delta > args.max_memory_cosine_regression:
                failures.append(
                    "memory min cosine regression "
                    f"{-cosine_delta:.4f} > {args.max_memory_cosine_regression:.4f}"
                )
        if args.fail_on_memory_l2_regression and l2_delta is not None:
            if l2_delta > args.max_memory_l2_regression:
                failures.append(f"memory max L2 regression {l2_delta:.4f} > {args.max_memory_l2_regression:.4f}")
        if args.fail_on_memory_norm_regression and norm_delta is not None:
            if norm_delta > args.max_memory_norm_regression:
                failures.append(
                    f"memory max norm-delta regression {norm_delta:.4f} > {args.max_memory_norm_regression:.4f}"
                )
    return failures


def run(args: argparse.Namespace) -> int:
    baseline = resolve_release(args, "baseline")
    candidate = resolve_release(args, "candidate")
    comparison = compare_releases(baseline, candidate)
    payload = {
        "baseline": compact_release(baseline),
        "candidate": compact_release(candidate),
        "comparison": comparison,
    }

    print("TextPy/SoA text release comparison")
    print("")
    print_release("baseline", baseline)
    print("")
    print_release("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  winner_by_loss: {comparison['winner_by_loss']}")
    print(f"  loss_delta: {comparison['loss_delta']:.6f}")
    print(f"  accuracy_delta: {comparison['accuracy_delta']:.6f}")
    print(f"  best_throughput_tokens_s_delta: {comparison['best_throughput_tokens_s_delta']:,.0f}")
    print(f"  loss_regression_rate: {comparison['loss_regression_rate']:.4f}")
    print(f"  best_throughput_regression_rate: {comparison['best_throughput_regression_rate']:.4f}")
    print(f"  has_memory_diagnostics: {comparison['has_memory_diagnostics']}")
    print(f"  memory_mean_pair_overlap_delta: {fmt(comparison['memory_mean_pair_overlap_delta'])}")
    print(
        "  memory_min_pair_state_cosine_similarity_delta: "
        f"{fmt(comparison['memory_min_pair_state_cosine_similarity_delta'])}"
    )
    print(f"  same_checkpoint_sha256: {comparison['same_checkpoint_sha256']}")

    if args.output_json:
        save_json(args.output_json, payload)
        print("")
        print(f"Saved JSON: {args.output_json}")

    failures = check_thresholds(comparison, args)
    if failures:
        print("")
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("")
    print("RELEASE COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two registered text releases.")
    parser.add_argument("--baseline-release", default=None)
    parser.add_argument("--candidate-release", default=None)
    parser.add_argument("--baseline-registry", default=None)
    parser.add_argument("--candidate-registry", default=None)
    parser.add_argument("--baseline-rank", type=int, default=1)
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--baseline-sort", default="benchmark_loss")
    parser.add_argument("--candidate-sort", default="benchmark_loss")
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--baseline-checkpoint-sha256", default=None)
    parser.add_argument("--candidate-checkpoint-sha256", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-memory", action="store_true")
    parser.add_argument("--fail-on-loss-regression", action="store_true")
    parser.add_argument("--max-loss-regression", type=float, default=0.01)
    parser.add_argument("--fail-on-throughput-regression", action="store_true")
    parser.add_argument("--fail-on-mean-throughput-regression", action="store_true")
    parser.add_argument("--max-throughput-regression", type=float, default=0.25)
    parser.add_argument("--fail-on-accuracy-regression", action="store_true")
    parser.add_argument("--max-accuracy-regression", type=float, default=0.05)
    parser.add_argument("--fail-on-memory-overlap-regression", action="store_true")
    parser.add_argument("--max-memory-overlap-regression", type=float, default=0.05)
    parser.add_argument("--fail-on-memory-cosine-regression", action="store_true")
    parser.add_argument("--max-memory-cosine-regression", type=float, default=0.05)
    parser.add_argument("--fail-on-memory-l2-regression", action="store_true")
    parser.add_argument("--max-memory-l2-regression", type=float, default=0.25)
    parser.add_argument("--fail-on-memory-norm-regression", action="store_true")
    parser.add_argument("--max-memory-norm-regression", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseComparisonError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
