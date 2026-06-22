#!/usr/bin/env python3
"""Run a small grid of held-out text SSM experiments and build a leaderboard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run_cmd(args: list[str], timeout: int) -> str:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {proc.returncode}: {' '.join(args)}\n{proc.stdout}"
        )
    return proc.stdout


def slug_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def experiment_name(config: dict[str, object], index: int) -> str:
    return (
        f"run_{index:03d}_"
        f"in{config['input_dim']}_"
        f"h{config['state_dim']}_"
        f"lr{slug_float(float(config['learning_rate']))}_"
        f"b{config['baseline_epochs']}_"
        f"c{config['candidate_epochs']}"
    )


def build_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    input_dims = parse_csv_ints(args.input_dims)
    state_dims = parse_csv_ints(args.state_dims)
    learning_rates = parse_csv_floats(args.learning_rates)
    candidate_epochs = parse_csv_ints(args.candidate_epochs)
    baseline_epochs = parse_csv_ints(args.baseline_epochs)
    configs: list[dict[str, object]] = []
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
    start_index = max(1, args.start_index)
    configs = configs[start_index - 1 :]
    if args.max_runs is not None:
        configs = configs[: args.max_runs]
    if not configs:
        raise SystemExit("Sweep grid is empty.")
    return configs


def grid_index(args: argparse.Namespace, local_index: int) -> int:
    return max(1, args.start_index) + local_index - 1


def run_experiment(config: dict[str, object], index: int, sweep_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    run_dir = sweep_dir / experiment_name(config, index)
    experiment_json = run_dir / "experiment.json"
    if args.resume and experiment_json.exists():
        experiment = json.loads(experiment_json.read_text(encoding="utf-8"))
        return {
            "run_dir": str(run_dir),
            "config": config,
            "elapsed_s": 0.0,
            "command": ["resume", str(experiment_json)],
            "stdout_tail": ["resumed existing experiment"],
            "winner_by_loss": experiment["comparison_run"]["comparison"]["winner_by_loss"],
            "loss_delta": experiment["comparison_run"]["comparison"]["loss_delta"],
            "resumed": True,
        }
    cmd = [
        sys.executable,
        "run_text_experiment.py",
        "--output-dir",
        str(run_dir),
        "--input-dim",
        str(config["input_dim"]),
        "--state-dim",
        str(config["state_dim"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--baseline-epochs",
        str(config["baseline_epochs"]),
        "--candidate-epochs",
        str(config["candidate_epochs"]),
        "--tokenizer",
        args.tokenizer,
        "--eval-fraction",
        str(args.eval_fraction),
        "--min-train-tokens",
        str(args.min_train_tokens),
        "--min-eval-tokens",
        str(args.min_eval_tokens),
        "--compare-repeat",
        str(args.compare_repeat),
        "--timeout",
        str(args.timeout),
    ]
    if args.text_file:
        cmd.extend(["--text-file", args.text_file])
    if args.tokenizer_config:
        cmd.extend(["--tokenizer-config", args.tokenizer_config])
    if args.tokenizer == "bpe":
        cmd.extend(
            [
                "--bpe-vocab-size",
                str(args.bpe_vocab_size),
                "--bpe-min-frequency",
                str(args.bpe_min_frequency),
                "--bpe-train-chars",
                str(args.bpe_train_chars),
            ]
        )
    if args.streams is not None:
        cmd.extend(["--streams", str(args.streams)])
    if args.tokens_per_stream is not None:
        cmd.extend(["--tokens-per-stream", str(args.tokens_per_stream)])
    if args.seq_len is not None:
        cmd.extend(["--seq-len", str(args.seq_len)])

    start = time.perf_counter()
    stdout = run_cmd(cmd, args.timeout)
    elapsed = time.perf_counter() - start
    experiment = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "config": config,
        "elapsed_s": elapsed,
        "command": cmd,
        "stdout_tail": stdout.splitlines()[-14:],
        "winner_by_loss": experiment["comparison_run"]["comparison"]["winner_by_loss"],
        "loss_delta": experiment["comparison_run"]["comparison"]["loss_delta"],
        "resumed": False,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace) -> None:
    sweep_dir = Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "text_sweeps" / timestamp()
    if args.clean and sweep_dir.exists():
        import shutil

        shutil.rmtree(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    configs = build_grid(args)

    print("TextPy/SoA text experiment sweep")
    print(f"sweep_dir: {sweep_dir}")
    print(f"runs: {len(configs)}")
    print(f"start_index: {max(1, args.start_index)}")
    print(f"resume: {args.resume}")
    print("")

    runs: list[dict[str, object]] = []
    for local_index, config in enumerate(configs, start=1):
        index = grid_index(args, local_index)
        print(
            f"run {local_index}/{len(configs)} (grid {index}): "
            f"input_dim={config['input_dim']} "
            f"state_dim={config['state_dim']} "
            f"lr={config['learning_rate']} "
            f"epochs={config['baseline_epochs']}->{config['candidate_epochs']}"
        )
        result = run_experiment(config, index, sweep_dir, args)
        runs.append(result)
        print(f"  winner_by_loss: {result['winner_by_loss']}")
        print(f"  loss_delta: {result['loss_delta']:.6f}")
        if result.get("resumed"):
            print("  resumed: True")

    leaderboard_json = sweep_dir / "leaderboard.json"
    leaderboard_csv = sweep_dir / "leaderboard.csv"
    run_cmd(
        [
            sys.executable,
            "list_text_experiments.py",
            "--root",
            str(sweep_dir),
            "--output-json",
            str(leaderboard_json),
            "--output-csv",
            str(leaderboard_csv),
        ],
        args.timeout,
    )
    leaderboard = json.loads(leaderboard_json.read_text(encoding="utf-8"))
    payload = {
        "created_at": timestamp(),
        "sweep_dir": str(sweep_dir),
        "text_file": args.text_file,
        "tokenizer": args.tokenizer,
        "tokenizer_config": args.tokenizer_config,
        "start_index": max(1, args.start_index),
        "resume": args.resume,
        "batch_run_count": len(runs),
        "run_count": len(runs),
        "discovered_run_count": len(leaderboard.get("experiments", [])),
        "runs": runs,
        "leaderboard_json": str(leaderboard_json),
        "leaderboard_csv": str(leaderboard_csv),
    }
    sweep_json = sweep_dir / "sweep.json"
    write_json(sweep_json, payload)
    print("")
    print(f"Saved sweep JSON: {sweep_json}")
    print(f"Saved leaderboard JSON: {leaderboard_json}")
    print(f"Saved leaderboard CSV: {leaderboard_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid of text SSM experiments.")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--tokenizer", choices=("word", "bpe"), default="word")
    parser.add_argument("--tokenizer-config", default=None)
    parser.add_argument("--bpe-vocab-size", type=int, default=512)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--bpe-train-chars", type=int, default=250_000)
    parser.add_argument("--input-dims", default="48")
    parser.add_argument("--state-dims", default="32,64")
    parser.add_argument("--learning-rates", default="0.012")
    parser.add_argument("--baseline-epochs", default="1")
    parser.add_argument("--candidate-epochs", default="2")
    parser.add_argument("--start-index", type=int, default=1, help="1-based index into the full grid.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip completed run directories with experiment.json.")
    parser.add_argument("--streams", type=int, default=None)
    parser.add_argument("--tokens-per-stream", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-tokens", type=int, default=64)
    parser.add_argument("--min-eval-tokens", type=int, default=16)
    parser.add_argument("--compare-repeat", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
