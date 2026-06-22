#!/usr/bin/env python3
"""Compare two TextPy/SoA hidden-memory probe reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


class MemoryComparisonError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MemoryComparisonError(f"Probe JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MemoryComparisonError(f"Invalid probe JSON {path}: {exc}") from exc


def require_probe(payload: dict[str, Any], label: str) -> None:
    if not isinstance(payload.get("steps"), list) or not payload["steps"]:
        raise MemoryComparisonError(f"{label} probe has no steps.")
    if not isinstance(payload.get("summary"), dict):
        raise MemoryComparisonError(f"{label} probe has no summary.")
    for row in payload["steps"]:
        for key in ["token", "state_norm", "state_delta_norm", "top_prediction", "top_probability"]:
            if key not in row:
                raise MemoryComparisonError(f"{label} probe step missing {key!r}: {row}")


def prediction_set(row: dict[str, Any]) -> set[str]:
    return {str(item.get("token")) for item in row.get("top_predictions", [])}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def vector(row: dict[str, Any]) -> list[float] | None:
    raw = row.get("state_vector")
    if not isinstance(raw, list):
        return None
    return [float(value) for value in raw]


def l2_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise MemoryComparisonError(f"State vector lengths differ: {len(left)} != {len(right)}")
    return sum((b - a) ** 2 for a, b in zip(left, right)) ** 0.5


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise MemoryComparisonError(f"State vector lengths differ: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def compare_rows(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_top = prediction_set(base)
    candidate_top = prediction_set(candidate)
    row = {
        "step": base.get("step"),
        "baseline_token": base["token"],
        "candidate_token": candidate["token"],
        "same_token": base["token"] == candidate["token"],
        "baseline_state_norm": base["state_norm"],
        "candidate_state_norm": candidate["state_norm"],
        "state_norm_delta": candidate["state_norm"] - base["state_norm"],
        "abs_state_norm_delta": abs(candidate["state_norm"] - base["state_norm"]),
        "baseline_state_delta_norm": base["state_delta_norm"],
        "candidate_state_delta_norm": candidate["state_delta_norm"],
        "state_delta_norm_delta": candidate["state_delta_norm"] - base["state_delta_norm"],
        "abs_state_delta_norm_delta": abs(candidate["state_delta_norm"] - base["state_delta_norm"]),
        "baseline_top_prediction": base["top_prediction"],
        "candidate_top_prediction": candidate["top_prediction"],
        "same_top_prediction": base["top_prediction"] == candidate["top_prediction"],
        "baseline_top_probability": base["top_probability"],
        "candidate_top_probability": candidate["top_probability"],
        "top_probability_delta": candidate["top_probability"] - base["top_probability"],
        "abs_top_probability_delta": abs(candidate["top_probability"] - base["top_probability"]),
        "top_prediction_overlap": jaccard(base_top, candidate_top),
    }
    base_vector = vector(base)
    candidate_vector = vector(candidate)
    if base_vector is not None and candidate_vector is not None:
        row["state_l2_distance"] = l2_distance(base_vector, candidate_vector)
        row["state_cosine_similarity"] = cosine_similarity(base_vector, candidate_vector)
    return row


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def compare_probes(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    require_probe(baseline, "baseline")
    require_probe(candidate, "candidate")
    if len(baseline["steps"]) != len(candidate["steps"]):
        raise MemoryComparisonError(
            f"Probe step counts differ: {len(baseline['steps'])} != {len(candidate['steps'])}"
        )

    rows = [
        compare_rows(base, cand)
        for base, cand in zip(baseline["steps"], candidate["steps"])
    ]
    token_match_rate = mean([1.0 if row["same_token"] else 0.0 for row in rows])
    top_match_rate = mean([1.0 if row["same_top_prediction"] else 0.0 for row in rows])
    overlaps = [float(row["top_prediction_overlap"]) for row in rows]
    norm_deltas = [float(row["abs_state_norm_delta"]) for row in rows]
    state_delta_deltas = [float(row["abs_state_delta_norm_delta"]) for row in rows]
    prob_deltas = [float(row["abs_top_probability_delta"]) for row in rows]
    vector_rows = [row for row in rows if "state_l2_distance" in row and "state_cosine_similarity" in row]
    has_state_vectors = len(vector_rows) == len(rows)
    l2_distances = [float(row["state_l2_distance"]) for row in vector_rows]
    cosine_similarities = [float(row["state_cosine_similarity"]) for row in vector_rows]

    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    summary = {
        "step_count": len(rows),
        "same_prompt": baseline.get("prompt") == candidate.get("prompt"),
        "same_tokens": token_match_rate == 1.0,
        "token_match_rate": token_match_rate,
        "top_match_rate": top_match_rate,
        "mean_top_prediction_overlap": mean(overlaps),
        "min_top_prediction_overlap": min(overlaps),
        "max_abs_state_norm_delta": max(norm_deltas),
        "mean_abs_state_norm_delta": mean(norm_deltas),
        "max_abs_state_delta_norm_delta": max(state_delta_deltas),
        "mean_abs_state_delta_norm_delta": mean(state_delta_deltas),
        "max_abs_top_probability_delta": max(prob_deltas),
        "mean_abs_top_probability_delta": mean(prob_deltas),
        "baseline_final_state_norm": baseline_summary["final_state_norm"],
        "candidate_final_state_norm": candidate_summary["final_state_norm"],
        "final_state_norm_delta": candidate_summary["final_state_norm"] - baseline_summary["final_state_norm"],
        "has_state_vectors": has_state_vectors,
        "mean_state_l2_distance": mean(l2_distances) if has_state_vectors else None,
        "max_state_l2_distance": max(l2_distances) if has_state_vectors else None,
        "mean_state_cosine_similarity": mean(cosine_similarities) if has_state_vectors else None,
        "min_state_cosine_similarity": min(cosine_similarities) if has_state_vectors else None,
        "final_state_l2_distance": None,
        "final_state_cosine_similarity": None,
        "baseline_final_top_prediction": baseline_summary["final_top_prediction"],
        "candidate_final_top_prediction": candidate_summary["final_top_prediction"],
        "same_final_top_prediction": (
            baseline_summary["final_top_prediction"] == candidate_summary["final_top_prediction"]
        ),
    }
    base_final_vector = baseline_summary.get("final_state_vector")
    candidate_final_vector = candidate_summary.get("final_state_vector")
    if isinstance(base_final_vector, list) and isinstance(candidate_final_vector, list):
        summary["final_state_l2_distance"] = l2_distance(
            [float(value) for value in base_final_vector],
            [float(value) for value in candidate_final_vector],
        )
        summary["final_state_cosine_similarity"] = cosine_similarity(
            [float(value) for value in base_final_vector],
            [float(value) for value in candidate_final_vector],
        )
    return {
        "baseline": {
            "prompt": baseline.get("prompt"),
            "checkpoint": baseline.get("checkpoint"),
            "manifest": baseline.get("manifest"),
            "release": baseline.get("release"),
            "summary": baseline_summary,
        },
        "candidate": {
            "prompt": candidate.get("prompt"),
            "checkpoint": candidate.get("checkpoint"),
            "manifest": candidate.get("manifest"),
            "release": candidate.get("release"),
            "summary": candidate_summary,
        },
        "summary": summary,
        "steps": rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "baseline_token",
        "candidate_token",
        "same_token",
        "baseline_state_norm",
        "candidate_state_norm",
        "state_norm_delta",
        "abs_state_norm_delta",
        "baseline_state_delta_norm",
        "candidate_state_delta_norm",
        "state_delta_norm_delta",
        "abs_state_delta_norm_delta",
        "baseline_top_prediction",
        "candidate_top_prediction",
        "same_top_prediction",
        "baseline_top_probability",
        "candidate_top_probability",
        "top_probability_delta",
        "abs_top_probability_delta",
        "top_prediction_overlap",
        "state_l2_distance",
        "state_cosine_similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("TextPy/SoA text memory comparison")
    print(f"baseline_prompt: {payload['baseline']['prompt']}")
    print(f"candidate_prompt: {payload['candidate']['prompt']}")
    print("")
    print("Summary")
    print(f"  steps: {summary['step_count']}")
    print(f"  same_prompt: {summary['same_prompt']}")
    print(f"  token_match_rate: {summary['token_match_rate']:.4f}")
    print(f"  top_match_rate: {summary['top_match_rate']:.4f}")
    print(f"  mean_top_prediction_overlap: {summary['mean_top_prediction_overlap']:.4f}")
    print(f"  min_top_prediction_overlap: {summary['min_top_prediction_overlap']:.4f}")
    print(f"  max_abs_state_norm_delta: {summary['max_abs_state_norm_delta']:.6f}")
    print(f"  max_abs_state_delta_norm_delta: {summary['max_abs_state_delta_norm_delta']:.6f}")
    print(f"  max_abs_top_probability_delta: {summary['max_abs_top_probability_delta']:.6f}")
    print(f"  final_state_norm_delta: {summary['final_state_norm_delta']:.6f}")
    if summary["has_state_vectors"]:
        print(f"  mean_state_l2_distance: {summary['mean_state_l2_distance']:.6f}")
        print(f"  min_state_cosine_similarity: {summary['min_state_cosine_similarity']:.6f}")
        print(f"  final_state_l2_distance: {summary['final_state_l2_distance']:.6f}")
        print(f"  final_state_cosine_similarity: {summary['final_state_cosine_similarity']:.6f}")
    print(f"  same_final_top_prediction: {summary['same_final_top_prediction']}")
    print("")
    print("Timeline")
    for row in payload["steps"]:
        print(
            f"  {row['step']:02d} "
            f"{row['baseline_token']!r}->{row['candidate_token']!r} "
            f"norm_delta={row['state_norm_delta']:.6f} "
            f"top={row['baseline_top_prediction']!r}->{row['candidate_top_prediction']!r} "
            f"overlap={row['top_prediction_overlap']:.4f}"
            + (
                f" cosine={row['state_cosine_similarity']:.4f}"
                if "state_cosine_similarity" in row
                else ""
            )
        )


def check_thresholds(payload: dict[str, Any], args: argparse.Namespace) -> list[str]:
    summary = payload["summary"]
    failures: list[str] = []
    if args.require_same_tokens and not summary["same_tokens"]:
        failures.append(f"token match rate {summary['token_match_rate']:.4f} < 1.0000")
    if summary["max_abs_state_norm_delta"] > args.max_state_norm_delta:
        failures.append(
            "state norm drift "
            f"{summary['max_abs_state_norm_delta']:.6f} > {args.max_state_norm_delta:.6f}"
        )
    if summary["max_abs_state_delta_norm_delta"] > args.max_state_delta_norm_delta:
        failures.append(
            "state delta drift "
            f"{summary['max_abs_state_delta_norm_delta']:.6f} > {args.max_state_delta_norm_delta:.6f}"
        )
    if summary["max_abs_top_probability_delta"] > args.max_top_probability_delta:
        failures.append(
            "top probability drift "
            f"{summary['max_abs_top_probability_delta']:.6f} > {args.max_top_probability_delta:.6f}"
        )
    if summary["mean_top_prediction_overlap"] < args.min_mean_top_overlap:
        failures.append(
            "mean top prediction overlap "
            f"{summary['mean_top_prediction_overlap']:.4f} < {args.min_mean_top_overlap:.4f}"
        )
    if summary["has_state_vectors"]:
        if summary["max_state_l2_distance"] > args.max_state_l2_distance:
            failures.append(
                "state L2 distance "
                f"{summary['max_state_l2_distance']:.6f} > {args.max_state_l2_distance:.6f}"
            )
        if summary["min_state_cosine_similarity"] < args.min_state_cosine_similarity:
            failures.append(
                "state cosine similarity "
                f"{summary['min_state_cosine_similarity']:.6f} < {args.min_state_cosine_similarity:.6f}"
            )
    return failures


def run(args: argparse.Namespace) -> int:
    baseline = load_json(Path(args.baseline))
    candidate = load_json(Path(args.candidate))
    payload = compare_probes(baseline, candidate)
    print_report(payload)
    if args.output_json:
        write_json(Path(args.output_json), payload)
        print("")
        print(f"Saved JSON: {args.output_json}")
    if args.output_csv:
        write_csv(Path(args.output_csv), payload["steps"])
        print(f"Saved CSV: {args.output_csv}")

    failures = check_thresholds(payload, args)
    if failures:
        print("")
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("")
    print("MEMORY COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two TextPy/SoA text memory probe JSON reports.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--require-same-tokens", action="store_true")
    parser.add_argument("--max-state-norm-delta", type=float, default=0.05)
    parser.add_argument("--max-state-delta-norm-delta", type=float, default=0.05)
    parser.add_argument("--max-top-probability-delta", type=float, default=0.01)
    parser.add_argument("--min-mean-top-overlap", type=float, default=0.5)
    parser.add_argument("--max-state-l2-distance", type=float, default=1.0e9)
    parser.add_argument("--min-state-cosine-similarity", type=float, default=-1.0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except MemoryComparisonError as exc:
        print(f"FAIL {exc}")
        raise SystemExit(1)
