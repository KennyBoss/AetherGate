#!/usr/bin/env python3
"""List registered TextPy/SoA text package releases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def metric_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("inf")
    return float(value)


def sort_releases(releases: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    reverse = sort.startswith("-")
    key = sort[1:] if reverse else sort
    return sorted(
        releases,
        key=lambda row: (
            metric_value(row, key),
            -float(row.get("benchmark_best_throughput_tokens_s") or 0.0),
            row.get("registered_at", ""),
        ),
        reverse=reverse,
    )


def flatten_release(row: dict[str, Any], rank: int) -> dict[str, Any]:
    comparison = row.get("comparison") or {}
    memory = row.get("memory") or {}
    return {
        "rank": rank,
        "name": row.get("name"),
        "registered_at": row.get("registered_at"),
        "benchmark_loss": row.get("benchmark_loss"),
        "benchmark_accuracy": row.get("benchmark_accuracy"),
        "benchmark_best_throughput_tokens_s": row.get("benchmark_best_throughput_tokens_s"),
        "benchmark_mean_throughput_tokens_s": row.get("benchmark_mean_throughput_tokens_s"),
        "checkpoint_sha256": row.get("checkpoint_sha256"),
        "archive_sha256": row.get("archive_sha256"),
        "checkpoint_bytes": row.get("checkpoint_bytes"),
        "archive_bytes": row.get("archive_bytes"),
        "param_count": row.get("param_count"),
        "streams": row.get("streams"),
        "seq_len": row.get("seq_len"),
        "tokens_per_stream": row.get("tokens_per_stream"),
        "source_tokens": row.get("source_tokens"),
        "vocab_size": row.get("vocab_size"),
        "winner_by_loss": row.get("winner_by_loss"),
        "comparison_winner_by_loss": comparison.get("winner_by_loss"),
        "comparison_loss_delta": comparison.get("loss_delta"),
        "comparison_same_checkpoint_sha256": comparison.get("same_checkpoint_sha256"),
        "memory_prompt_count": memory.get("prompt_count"),
        "memory_pair_count": memory.get("pair_count"),
        "memory_mean_pair_overlap": memory.get("mean_pair_overlap"),
        "memory_min_pair_overlap": memory.get("min_pair_overlap"),
        "memory_max_pair_state_norm_delta": memory.get("max_pair_state_norm_delta"),
        "memory_max_pair_state_l2_distance": memory.get("max_pair_state_l2_distance"),
        "memory_min_pair_state_cosine_similarity": memory.get("min_pair_state_cosine_similarity"),
        "memory_validation": memory.get("validation"),
        "manifest": row.get("manifest"),
        "archive": row.get("archive"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "name",
        "registered_at",
        "benchmark_loss",
        "benchmark_accuracy",
        "benchmark_best_throughput_tokens_s",
        "benchmark_mean_throughput_tokens_s",
        "checkpoint_sha256",
        "archive_sha256",
        "checkpoint_bytes",
        "archive_bytes",
        "param_count",
        "streams",
        "seq_len",
        "tokens_per_stream",
        "source_tokens",
        "vocab_size",
        "winner_by_loss",
        "comparison_winner_by_loss",
        "comparison_loss_delta",
        "comparison_same_checkpoint_sha256",
        "memory_prompt_count",
        "memory_pair_count",
        "memory_mean_pair_overlap",
        "memory_min_pair_overlap",
        "memory_max_pair_state_norm_delta",
        "memory_max_pair_state_l2_distance",
        "memory_min_pair_state_cosine_similarity",
        "memory_validation",
        "manifest",
        "archive",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if abs(value_f) >= 1000:
        return f"{value_f:,.0f}"
    return f"{value_f:.{digits}f}"


def run(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    registry = load_json(registry_path)
    releases = sort_releases(list(registry.get("releases", [])), args.sort)
    if args.limit:
        releases = releases[: args.limit]
    rows = [flatten_release(row, rank + 1) for rank, row in enumerate(releases)]
    payload = {
        "registry": str(registry_path),
        "sort": args.sort,
        "release_count": len(registry.get("releases", [])),
        "shown_count": len(rows),
        "releases": rows,
    }

    print("TextPy/SoA text release leaderboard")
    print(f"registry: {registry_path}")
    print(f"releases: {payload['release_count']}")
    print(f"shown: {payload['shown_count']}")
    print("")
    print(
        f"{'rank':>4} {'name':20} {'loss':>9} {'best tok/s':>12} "
        f"{'mem ovl':>8} {'mem cos':>8} {'sha256':12} archive"
    )
    for row in rows:
        checkpoint_sha = str(row.get("checkpoint_sha256") or "")[:12]
        archive_sha = str(row.get("archive_sha256") or "")[:12]
        print(
            f"{row['rank']:>4} "
            f"{str(row.get('name') or '')[:20]:20} "
            f"{format_num(row.get('benchmark_loss')):>9} "
            f"{format_num(row.get('benchmark_best_throughput_tokens_s'), 0):>12} "
            f"{format_num(row.get('memory_mean_pair_overlap')):>8} "
            f"{format_num(row.get('memory_min_pair_state_cosine_similarity')):>8} "
            f"{checkpoint_sha:12} "
            f"{archive_sha}"
        )

    if args.output_json:
        write_json(Path(args.output_json), payload)
        print("")
        print(f"Saved JSON: {args.output_json}")
    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
        print(f"Saved CSV: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List registered text package releases.")
    parser.add_argument("--registry", default="artifacts/releases/text_packages.json")
    parser.add_argument("--sort", default="benchmark_loss")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
