#!/usr/bin/env python3
"""Run a profile -> train baseline/candidate -> compare text SSM experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from profile_text_corpus import profile as profile_corpus
from profile_text_corpus import save_json as save_profile_json
from train_ssm_text import detokenize, load_text, tokenize


ROOT = Path(__file__).resolve().parent


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


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def choose_shape(args: argparse.Namespace, profile_report: dict[str, object]) -> dict[str, int]:
    recommended = profile_report["recommended_training_args"]
    streams = args.streams or int(recommended["streams"])
    tokens_per_stream = args.tokens_per_stream or int(recommended["tokens_per_stream"])
    seq_len = args.seq_len or int(recommended["seq_len"])
    if seq_len + 1 > tokens_per_stream:
        raise SystemExit("--seq-len must be smaller than --tokens-per-stream.")
    return {
        "streams": streams,
        "tokens_per_stream": tokens_per_stream,
        "seq_len": seq_len,
    }


def write_split_files(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = load_text(args.text_file)
    tokens = tokenize(text)
    if len(tokens) < args.min_train_tokens + args.min_eval_tokens:
        raise SystemExit(
            "Not enough tokens for train/eval split: "
            f"{len(tokens)} < {args.min_train_tokens + args.min_eval_tokens}"
        )

    eval_tokens = max(args.min_eval_tokens, int(round(len(tokens) * args.eval_fraction)))
    eval_tokens = min(eval_tokens, len(tokens) - args.min_train_tokens)
    train_tokens = len(tokens) - eval_tokens
    if train_tokens < args.min_train_tokens or eval_tokens < args.min_eval_tokens:
        raise SystemExit(
            f"Invalid train/eval split: train={train_tokens}, eval={eval_tokens}. "
            "Adjust --eval-fraction, --min-train-tokens, or --min-eval-tokens."
        )

    train_path = output_dir / "train.txt"
    eval_path = output_dir / "eval.txt"
    train_path.write_text(detokenize(tokens[:train_tokens]), encoding="utf-8")
    eval_path.write_text(detokenize(tokens[train_tokens:]), encoding="utf-8")
    for path, label in [(train_path, "train"), (eval_path, "eval")]:
        if not path.exists() or path.stat().st_size <= 0:
            raise SystemExit(f"Failed to materialize {label} split file: {path}")
    return {
        "source_text_file": args.text_file or "builtin_default",
        "train_text_file": str(train_path),
        "eval_text_file": str(eval_path),
        "total_tokens": len(tokens),
        "train_tokens": train_tokens,
        "eval_tokens": eval_tokens,
        "eval_fraction": eval_tokens / len(tokens),
    }


def train_checkpoint(
    *,
    label: str,
    checkpoint: Path,
    train_text_file: str,
    shape: dict[str, int],
    epochs: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        "train_ssm_text.py",
        "--streams",
        str(shape["streams"]),
        "--tokens-per-stream",
        str(shape["tokens_per_stream"]),
        "--seq-len",
        str(shape["seq_len"]),
        "--epochs",
        str(epochs),
        "--input-dim",
        str(args.input_dim),
        "--state-dim",
        str(args.state_dim),
        "--learning-rate",
        str(args.learning_rate),
        "--grad-clip",
        str(args.grad_clip),
        "--sample-steps",
        str(args.train_sample_steps),
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--tokenizer",
        args.tokenizer,
        "--text-file",
        train_text_file,
        "--save-checkpoint",
        str(checkpoint),
    ]
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
    if args.tokenizer_config:
        cmd.extend(["--tokenizer-config", args.tokenizer_config])

    start = time.perf_counter()
    stdout = run_cmd(cmd, args.timeout)
    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "epochs": epochs,
        "elapsed_s": elapsed,
        "command": cmd,
        "stdout_tail": stdout.splitlines()[-12:],
    }


def compare_checkpoints(
    *,
    baseline: Path,
    candidate: Path,
    output_json: Path,
    eval_text_file: str,
    shape: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        "compare_text_checkpoints.py",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--streams",
        str(shape["streams"]),
        "--tokens-per-stream",
        str(shape["tokens_per_stream"]),
        "--seq-len",
        str(shape["seq_len"]),
        "--repeat",
        str(args.compare_repeat),
        "--sample-steps",
        str(args.compare_sample_steps),
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--text-file",
        eval_text_file,
        "--output-json",
        str(output_json),
    ]
    if args.fail_on_eval_loss_regression:
        cmd.extend(["--fail-on-loss-regression", "--max-loss-regression", str(args.max_loss_regression)])
    if args.fail_on_throughput_regression:
        cmd.extend(
            [
                "--fail-on-throughput-regression",
                "--max-throughput-regression",
                str(args.max_throughput_regression),
            ]
        )

    start = time.perf_counter()
    stdout = run_cmd(cmd, args.timeout)
    elapsed = time.perf_counter() - start
    comparison = json.loads(output_json.read_text(encoding="utf-8"))
    return {
        "elapsed_s": elapsed,
        "command": cmd,
        "stdout_tail": stdout.splitlines()[-16:],
        "comparison_json": str(output_json),
        "comparison": comparison["comparison"],
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "text_experiments" / timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    split = write_split_files(args, output_dir)

    profile_args = argparse.Namespace(
        text_file=split["train_text_file"],
        checkpoint=None,
        output_json=None,
        top_k=args.top_k,
        tokenizer=args.tokenizer,
        tokenizer_config=args.tokenizer_config,
        output_tokenizer_json=None,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_min_frequency=args.bpe_min_frequency,
        bpe_train_chars=args.bpe_train_chars,
        target_streams=args.target_streams,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        input_dim=args.input_dim,
        state_dim=args.state_dim,
    )
    profile_report = profile_corpus(profile_args)
    profile_json = output_dir / "profile.json"
    save_profile_json(str(profile_json), profile_report)
    shape = choose_shape(args, profile_report)

    baseline_checkpoint = output_dir / "baseline.npz"
    candidate_checkpoint = output_dir / "candidate.npz"
    comparison_json = output_dir / "comparison.json"
    experiment_json = output_dir / "experiment.json"

    print("TextPy/SoA text experiment pipeline")
    print(f"output_dir: {output_dir}")
    print("")
    print("Profile")
    print(f"  tokenizer: {profile_report['tokenizer']}")
    print(f"  tokens: {profile_report['tokens']:,}")
    print(f"  train_tokens: {split['train_tokens']:,}")
    print(f"  eval_tokens: {split['eval_tokens']:,}")
    print(f"  vocab_size_with_specials: {profile_report['vocab_size_with_specials']:,}")
    print(f"  selected_streams: {shape['streams']}")
    print(f"  selected_tokens_per_stream: {shape['tokens_per_stream']}")
    print(f"  selected_seq_len: {shape['seq_len']}")
    print("")

    print(f"Training baseline ({args.baseline_epochs} epochs)")
    baseline_train = train_checkpoint(
        label="baseline",
        checkpoint=baseline_checkpoint,
        train_text_file=split["train_text_file"],
        shape=shape,
        epochs=args.baseline_epochs,
        args=args,
    )
    print(f"  checkpoint: {baseline_checkpoint}")

    print(f"Training candidate ({args.candidate_epochs} epochs)")
    candidate_train = train_checkpoint(
        label="candidate",
        checkpoint=candidate_checkpoint,
        train_text_file=split["train_text_file"],
        shape=shape,
        epochs=args.candidate_epochs,
        args=args,
    )
    print(f"  checkpoint: {candidate_checkpoint}")

    print("Comparing checkpoints")
    comparison_run = compare_checkpoints(
        baseline=baseline_checkpoint,
        candidate=candidate_checkpoint,
        output_json=comparison_json,
        eval_text_file=split["eval_text_file"],
        shape=shape,
        args=args,
    )
    comparison = comparison_run["comparison"]
    print(f"  winner_by_loss: {comparison['winner_by_loss']}")
    print(f"  loss_delta: {comparison['loss_delta']:.6f}")
    print(f"  accuracy_delta: {comparison['accuracy_delta']:.6f}")
    print(f"  throughput_tokens_s_delta: {comparison['throughput_tokens_s_delta']:,.0f}")

    payload = {
        "created_at": timestamp(),
        "output_dir": str(output_dir),
        "text_file": args.text_file,
        "split": split,
        "profile_json": str(profile_json),
        "comparison_json": str(comparison_json),
        "shape": shape,
        "tokenizer": profile_report["tokenizer"],
        "tokenizer_config": profile_report["tokenizer_config"],
        "baseline_train": baseline_train,
        "candidate_train": candidate_train,
        "comparison_run": comparison_run,
    }
    with experiment_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("")
    print(f"Saved experiment JSON: {experiment_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a complete text SSM experiment.")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tokenizer", choices=("word", "bpe"), default="word")
    parser.add_argument("--tokenizer-config", default=None)
    parser.add_argument("--bpe-vocab-size", type=int, default=512)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--bpe-train-chars", type=int, default=250_000)
    parser.add_argument("--baseline-epochs", type=int, default=1)
    parser.add_argument("--candidate-epochs", type=int, default=2)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-tokens", type=int, default=64)
    parser.add_argument("--min-eval-tokens", type=int, default=16)
    parser.add_argument("--streams", type=int, default=None)
    parser.add_argument("--tokens-per-stream", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--target-streams", type=int, default=64)
    parser.add_argument("--min-seq-len", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=48)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.012)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-sample-steps", type=int, default=4)
    parser.add_argument("--compare-sample-steps", type=int, default=8)
    parser.add_argument("--compare-repeat", type=int, default=3)
    parser.add_argument("--prompt", default="memory is")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--fail-on-throughput-regression", action="store_true")
    parser.add_argument("--fail-on-eval-loss-regression", action="store_true")
    parser.add_argument("--max-loss-regression", type=float, default=0.01)
    parser.add_argument("--max-throughput-regression", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
