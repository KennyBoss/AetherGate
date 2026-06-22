#!/usr/bin/env python3
"""List and rank saved text SSM experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SORT_KEYS = {
    "candidate_loss",
    "loss_delta",
    "candidate_accuracy",
    "candidate_throughput_tokens_s",
    "created_at",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_experiment(path: Path) -> dict[str, Any] | None:
    try:
        experiment = load_json(path)
        comparison_path = Path(experiment["comparison_json"])
        if not comparison_path.is_absolute():
            comparison_path = path.parent / comparison_path.name
        comparison_report = load_json(comparison_path)
    except (OSError, KeyError, json.JSONDecodeError):
        return None

    baseline = comparison_report["baseline"]
    candidate = comparison_report["candidate"]
    comparison = comparison_report["comparison"]
    shape = experiment.get("shape", {})
    split = experiment.get("split", {})
    tokenizer_config = experiment.get("tokenizer_config") or candidate.get("checkpoint_config", {}).get(
        "tokenizer_config",
        {"type": "word"},
    )
    return {
        "run_dir": str(path.parent),
        "created_at": experiment.get("created_at", ""),
        "text_file": experiment.get("text_file") or "builtin_default",
        "tokenizer": tokenizer_config.get("type", experiment.get("tokenizer", "word")),
        "bpe_merges": len(tokenizer_config.get("merges", [])),
        "train_tokens": int(split.get("train_tokens", 0)),
        "eval_tokens": int(split.get("eval_tokens", candidate.get("source_tokens", 0))),
        "streams": int(shape.get("streams", candidate.get("streams", 0))),
        "tokens_per_stream": int(shape.get("tokens_per_stream", candidate.get("tokens_per_stream", 0))),
        "seq_len": int(shape.get("seq_len", candidate.get("seq_len", 0))),
        "source_tokens": int(candidate.get("source_tokens", 0)),
        "baseline_checkpoint": baseline["checkpoint"],
        "candidate_checkpoint": candidate["checkpoint"],
        "baseline_loss": float(baseline["loss"]),
        "candidate_loss": float(candidate["loss"]),
        "loss_delta": float(comparison["loss_delta"]),
        "baseline_accuracy": float(baseline["accuracy"]),
        "candidate_accuracy": float(candidate["accuracy"]),
        "accuracy_delta": float(comparison["accuracy_delta"]),
        "baseline_throughput_tokens_s": float(baseline["throughput_tokens_s"]),
        "candidate_throughput_tokens_s": float(candidate["throughput_tokens_s"]),
        "throughput_tokens_s_delta": float(comparison["throughput_tokens_s_delta"]),
        "candidate_oov_token_rate": float(candidate["oov"]["oov_token_rate"]),
        "winner_by_loss": comparison["winner_by_loss"],
    }


def discover_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/experiment.json")):
        row = read_experiment(path)
        if row is not None:
            rows.append(row)
    return rows


def sort_rows(rows: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    if sort_key not in SORT_KEYS:
        available = ", ".join(sorted(SORT_KEYS))
        raise SystemExit(f"Unknown --sort key {sort_key!r}. Available: {available}")
    reverse = sort_key in {"candidate_accuracy", "candidate_throughput_tokens_s", "created_at"}
    return sorted(rows, key=lambda row: row.get(sort_key, 0), reverse=reverse)


def save_json(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump({"experiments": rows}, f, indent=2, ensure_ascii=False)


def save_csv(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "created_at",
        "text_file",
        "tokenizer",
        "bpe_merges",
        "streams",
        "tokens_per_stream",
        "seq_len",
        "source_tokens",
        "train_tokens",
        "eval_tokens",
        "baseline_loss",
        "candidate_loss",
        "loss_delta",
        "baseline_accuracy",
        "candidate_accuracy",
        "accuracy_delta",
        "candidate_throughput_tokens_s",
        "throughput_tokens_s_delta",
        "candidate_oov_token_rate",
        "winner_by_loss",
        "baseline_checkpoint",
        "candidate_checkpoint",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def print_table(rows: list[dict[str, Any]], limit: int) -> None:
    print("TextPy/SoA text experiment leaderboard")
    print(f"runs: {len(rows)}")
    print("")
    if not rows:
        print("No experiment.json files found.")
        return

    header = (
        f"{'#':>2}  {'candidate_loss':>14}  {'loss_delta':>10}  "
        f"{'acc':>7}  {'eval':>6}  {'tok/s':>10}  {'shape':>11}  {'toknz':>5}  {'winner':>9}  run_dir"
    )
    print(header)
    print("-" * len(header))
    for index, row in enumerate(rows[:limit], start=1):
        shape = f"{row['streams']}x{row['tokens_per_stream']}/{row['seq_len']}"
        print(
            f"{index:>2}  "
            f"{row['candidate_loss']:>14.4f}  "
            f"{row['loss_delta']:>10.4f}  "
            f"{row['candidate_accuracy']:>7.4f}  "
            f"{row['eval_tokens']:>6}  "
            f"{row['candidate_throughput_tokens_s']:>10.0f}  "
            f"{shape:>11}  "
            f"{row['tokenizer'][:5]:>5}  "
            f"{row['winner_by_loss']:>9}  "
            f"{row['run_dir']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List saved text experiment runs.")
    parser.add_argument("--root", default="artifacts/text_experiments")
    parser.add_argument("--sort", default="candidate_loss")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    rows = sort_rows(discover_runs(root), args.sort)
    print_table(rows, args.limit)
    if args.output_json:
        save_json(args.output_json, rows)
        print("")
        print(f"Saved JSON: {args.output_json}")
    if args.output_csv:
        save_csv(args.output_csv, rows)
        print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    run(parse_args())
