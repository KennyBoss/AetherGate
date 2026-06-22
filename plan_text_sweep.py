#!/usr/bin/env python3
"""Plan large resumable text sweeps without launching them."""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Any


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_dims = parse_csv_ints(args.input_dims)
    state_dims = parse_csv_ints(args.state_dims)
    learning_rates = parse_csv_floats(args.learning_rates)
    baseline_epochs = parse_csv_ints(args.baseline_epochs)
    candidate_epochs = parse_csv_ints(args.candidate_epochs)
    configs: list[dict[str, Any]] = []
    for input_dim, state_dim, learning_rate, baseline_epoch, candidate_epoch in product(
        input_dims,
        state_dims,
        learning_rates,
        baseline_epochs,
        candidate_epochs,
    ):
        configs.append(
            {
                "input_dim": input_dim,
                "state_dim": state_dim,
                "learning_rate": learning_rate,
                "baseline_epochs": baseline_epoch,
                "candidate_epochs": candidate_epoch,
            }
        )
    return configs


def batch_commands(args: argparse.Namespace, run_count: int) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    for start in range(1, run_count + 1, batch_size):
        size = min(batch_size, run_count - start + 1)
        cmd = [
            "python3",
            "sweep_text_experiments.py",
            "--text-file",
            args.text_file,
            "--tokenizer",
            args.tokenizer,
            "--output-dir",
            args.output_dir,
            "--resume",
            "--start-index",
            str(start),
            "--max-runs",
            str(size),
            "--streams",
            str(args.streams),
            "--tokens-per-stream",
            str(args.tokens_per_stream),
            "--seq-len",
            str(args.seq_len),
            "--input-dims",
            args.input_dims,
            "--state-dims",
            args.state_dims,
            "--learning-rates",
            args.learning_rates,
            "--baseline-epochs",
            args.baseline_epochs,
            "--candidate-epochs",
            args.candidate_epochs,
            "--compare-repeat",
            str(args.compare_repeat),
            "--timeout",
            str(args.timeout),
        ]
        if args.tokenizer_config:
            cmd.extend(["--tokenizer-config", args.tokenizer_config])
        batches.append({"start_index": start, "max_runs": size, "command": cmd})
    return batches


def plan(args: argparse.Namespace) -> dict[str, Any]:
    configs = build_configs(args)
    corpus_manifest = load_json(args.corpus_manifest)
    tokenizer_config = load_json(args.tokenizer_config)
    batches = batch_commands(args, len(configs))
    report = {
        "text_file": args.text_file,
        "corpus_manifest": args.corpus_manifest,
        "corpus_stats": corpus_manifest.get("stats") if corpus_manifest else None,
        "tokenizer": args.tokenizer,
        "tokenizer_config": args.tokenizer_config,
        "tokenizer_merges": len(tokenizer_config.get("merges", [])) if tokenizer_config else None,
        "output_dir": args.output_dir,
        "grid": {
            "input_dims": parse_csv_ints(args.input_dims),
            "state_dims": parse_csv_ints(args.state_dims),
            "learning_rates": parse_csv_floats(args.learning_rates),
            "baseline_epochs": parse_csv_ints(args.baseline_epochs),
            "candidate_epochs": parse_csv_ints(args.candidate_epochs),
            "run_count": len(configs),
        },
        "shape": {
            "streams": args.streams,
            "tokens_per_stream": args.tokens_per_stream,
            "seq_len": args.seq_len,
            "positions_per_epoch": args.streams * ((args.tokens_per_stream - 1) // args.seq_len) * args.seq_len,
        },
        "batch_size": args.batch_size,
        "batch_count": len(batches),
        "batches": batches,
        "estimated_train_positions": sum(
            args.streams
            * ((args.tokens_per_stream - 1) // args.seq_len)
            * args.seq_len
            * (int(config["baseline_epochs"]) + int(config["candidate_epochs"]))
            for config in configs
        ),
        "hundred_run_batches_remaining": max(0, math.ceil((100 - len(configs)) / max(1, args.batch_size))),
    }
    return report


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_report(report: dict[str, Any]) -> None:
    print("TextPy/SoA sweep planner")
    print(f"text_file: {report['text_file']}")
    if report["corpus_stats"]:
        stats = report["corpus_stats"]
        print(f"corpus_bytes: {stats['bytes_utf8']:,}")
        print(f"word_tokens: {stats['word_tokens']:,}")
    print(f"tokenizer: {report['tokenizer']}")
    if report["tokenizer_merges"] is not None:
        print(f"tokenizer_merges: {report['tokenizer_merges']:,}")
    print(f"grid_runs: {report['grid']['run_count']}")
    print(f"batch_count: {report['batch_count']}")
    print(f"positions_per_epoch: {report['shape']['positions_per_epoch']:,}")
    print(f"estimated_train_positions: {report['estimated_train_positions']:,}")
    print("")
    print("Next batch command")
    if report["batches"]:
        print("  " + " ".join(report["batches"][0]["command"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a large resumable text sweep.")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--corpus-manifest", default=None)
    parser.add_argument("--tokenizer", choices=("word", "bpe"), default="bpe")
    parser.add_argument("--tokenizer-config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dims", default="48,64,96")
    parser.add_argument("--state-dims", default="64,96,128,160")
    parser.add_argument("--learning-rates", default="0.0015,0.003,0.006,0.009")
    parser.add_argument("--baseline-epochs", default="3,5")
    parser.add_argument("--candidate-epochs", default="5,8")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--tokens-per-stream", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--compare-repeat", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    report = plan(args)
    print_report(report)
    if args.output_json:
        save_json(args.output_json, report)
        print("")
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    run(parse_args())
