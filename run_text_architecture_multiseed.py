#!/usr/bin/env python3
"""Run repeated-seed SSM-vs-Transformer text architecture comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SKIP_RANK = 32
SSM_VARIANTS = (
    "static",
    "selective",
    "selective-lite",
    "write-gated",
    "skip",
    "skip-write-gated",
    "skip-selective-lite",
    "tied-skip",
    "lowrank-skip",
    "conv-skip",
    "state-mix",
    "skip-state-mix",
    "conv-state-mix",
    "token-memory",
    "conv-token-memory",
    "conv-token-memory-closed",
    "conv-token-memory-sparse",
    "kv-memory",
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv_ints(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer.")
    return seeds


def run_cmd(args: list[str], timeout: int) -> tuple[str, float]:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {proc.returncode}: {' '.join(args)}\n{proc.stdout}"
        )
    return proc.stdout, elapsed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "stdev": stdev(values),
        "min": min(values),
        "max": max(values),
    }


def architecture_summary(runs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return {
        "loss": summarize([float(run[name]["eval"]["loss"]) for run in runs]),
        "accuracy": summarize([float(run[name]["eval"]["accuracy"]) for run in runs]),
        "throughput_tokens_s": summarize(
            [float(run[name]["eval"]["throughput_tokens_s"]) for run in runs]
        ),
        "train_throughput_tokens_s": summarize(
            [float(run[name]["train"]["train_throughput_tokens_s"]) for run in runs]
        ),
        "parameter_count": summarize([float(run[name]["parameter_count"]) for run in runs]),
        "parameter_bytes": summarize([float(run[name]["parameter_bytes"]) for run in runs]),
    }


def comparison_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    winners: dict[str, int] = {"ssm": 0, "transformer": 0, "tie": 0}
    for run in runs:
        winner = str(run["comparison"]["winner_by_loss"])
        winners[winner] = winners.get(winner, 0) + 1
    return {
        "winner_counts": winners,
        "ssm_minus_transformer_loss": summarize(
            [float(run["comparison"]["ssm_minus_transformer_loss"]) for run in runs]
        ),
        "ssm_minus_transformer_accuracy": summarize(
            [float(run["comparison"]["ssm_minus_transformer_accuracy"]) for run in runs]
        ),
        "ssm_minus_transformer_throughput_tokens_s": summarize(
            [
                float(run["comparison"]["ssm_minus_transformer_throughput_tokens_s"])
                for run in runs
            ]
        ),
    }


def memory_context_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ssm_state_bytes = [float(run["ssm"]["state_memory_bytes"]) for run in runs]
    transformer_attention_bytes = [
        float(run["transformer"]["attention_score_memory_bytes"]) for run in runs
    ]
    ssm_context = [float(run["ssm"]["effective_context_tokens"]) for run in runs]
    transformer_context = [float(run["transformer"]["effective_context_tokens"]) for run in runs]
    return {
        "ssm_state_memory_bytes": summarize(ssm_state_bytes),
        "transformer_attention_score_memory_bytes": summarize(transformer_attention_bytes),
        "transformer_attention_to_ssm_state_memory_ratio": summarize(
            [
                transformer / max(1.0, ssm)
                for transformer, ssm in zip(transformer_attention_bytes, ssm_state_bytes)
            ]
        ),
        "ssm_effective_context_tokens": summarize(ssm_context),
        "transformer_effective_context_tokens": summarize(transformer_context),
        "ssm_to_transformer_context_ratio": summarize(
            [
                ssm / max(1.0, transformer)
                for ssm, transformer in zip(ssm_context, transformer_context)
            ]
        ),
    }


def build_inner_command(args: argparse.Namespace, seed: int, output_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        "compare_text_architectures.py",
        "--output-json",
        str(output_json),
        "--tokenizer",
        args.tokenizer,
        "--eval-fraction",
        str(args.eval_fraction),
        "--min-train-tokens",
        str(args.min_train_tokens),
        "--min-eval-tokens",
        str(args.min_eval_tokens),
        "--streams",
        str(args.streams),
        "--tokens-per-stream",
        str(args.tokens_per_stream),
        "--seq-len",
        str(args.seq_len),
        "--ssm-input-dim",
        str(args.ssm_input_dim),
        "--ssm-state-dim",
        str(args.ssm_state_dim),
        "--ssm-variant",
        args.ssm_variant,
        "--ssm-skip-rank",
        str(args.ssm_skip_rank),
        "--transformer-max-d-model",
        str(args.transformer_max_d_model),
        "--transformer-layers",
        str(args.transformer_layers),
        "--transformer-heads",
        str(args.transformer_heads),
        "--transformer-ff-mult",
        str(args.transformer_ff_mult),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--grad-clip",
        str(args.grad_clip),
        "--memory-gate-l1",
        str(args.memory_gate_l1),
        "--ssm-aux-recall-every",
        str(args.ssm_aux_recall_every),
        "--ssm-aux-recall-task",
        args.ssm_aux_recall_task,
        "--ssm-aux-recall-delay",
        str(args.ssm_aux_recall_delay),
        "--ssm-aux-recall-key-count",
        str(args.ssm_aux_recall_key_count),
        "--ssm-aux-recall-lr-scale",
        str(args.ssm_aux_recall_lr_scale),
        "--ssm-aux-recall-write-logit",
        str(args.ssm_aux_recall_write_logit),
        "--ssm-aux-recall-read-logit",
        str(args.ssm_aux_recall_read_logit),
        "--ssm-aux-recall-closed-logit",
        str(args.ssm_aux_recall_closed_logit),
        "--ssm-aux-recall-logit-scale",
        str(args.ssm_aux_recall_logit_scale),
        "--ssm-aux-recall-query-id",
        str(args.ssm_aux_recall_query_id),
        "--ssm-aux-recall-filler-id",
        str(args.ssm_aux_recall_filler_id),
        "--ssm-aux-recall-assign-id",
        str(args.ssm_aux_recall_assign_id),
        "--ssm-aux-recall-separator-id",
        str(args.ssm_aux_recall_separator_id),
        "--repeat",
        str(args.repeat),
        "--sample-steps",
        str(args.sample_steps),
        "--temperature",
        str(args.temperature),
        "--prompt",
        args.prompt,
        "--seed",
        str(seed),
        "--backend",
        args.backend,
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
    if args.transformer_d_model is not None:
        cmd.extend(["--transformer-d-model", str(args.transformer_d_model)])
    if args.ssm_aux_recall_init_gates:
        cmd.append("--ssm-aux-recall-init-gates")
    return cmd


def validate_protocol(args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    train_positions_per_epoch = args.streams * args.seq_len * (
        (args.tokens_per_stream - 1) // args.seq_len
    )
    eval_positions_per_repeat = train_positions_per_epoch
    repeated_seed_count = len(runs)
    corpus_train_tokens = min(int(run["protocol"]["train_tokens"]) for run in runs)
    corpus_eval_tokens = min(int(run["protocol"]["eval_tokens"]) for run in runs)
    evidence = {
        "repeated_seed_count": repeated_seed_count,
        "seed_requirement_met": repeated_seed_count >= args.min_seeds,
        "train_positions_per_epoch": train_positions_per_epoch,
        "eval_positions_per_repeat": eval_positions_per_repeat,
        "million_train_positions_met": train_positions_per_epoch >= args.min_train_positions,
        "million_eval_positions_met": eval_positions_per_repeat >= args.min_eval_positions,
        "min_corpus_train_tokens": corpus_train_tokens,
        "min_corpus_eval_tokens": corpus_eval_tokens,
        "corpus_train_token_requirement_met": corpus_train_tokens >= args.min_corpus_train_tokens,
        "corpus_eval_token_requirement_met": corpus_eval_tokens >= args.min_corpus_eval_tokens,
    }
    evidence["valid"] = all(
        [
            evidence["seed_requirement_met"],
            evidence["million_train_positions_met"],
            evidence["million_eval_positions_met"],
            evidence["corpus_train_token_requirement_met"],
            evidence["corpus_eval_token_requirement_met"],
        ]
    )
    return evidence


def run(args: argparse.Namespace) -> None:
    seeds = parse_csv_ints(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("TextPy/SoA multi-seed architecture comparison")
    print(f"output_dir: {output_dir}")
    print(f"seeds: {', '.join(str(seed) for seed in seeds)}")
    print(
        "million_positions_per_epoch: "
        f"{args.streams * args.seq_len * ((args.tokens_per_stream - 1) // args.seq_len):,}"
    )
    print("")

    runs: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        seed_json = output_dir / f"seed_{seed}.json"
        cmd = build_inner_command(args, seed, seed_json)
        print(f"seed {index}/{len(seeds)}: {seed}")
        print("  " + " ".join(cmd))
        stdout, elapsed = run_cmd(cmd, args.timeout)
        payload = json.loads(seed_json.read_text(encoding="utf-8"))
        runs.append(payload)
        executions.append(
            {
                "seed": seed,
                "output_json": str(seed_json),
                "elapsed_s": elapsed,
                "command": cmd,
                "stdout_tail": stdout.splitlines()[-24:],
                "winner_by_loss": payload["comparison"]["winner_by_loss"],
                "ssm_loss": payload["ssm"]["eval"]["loss"],
                "transformer_loss": payload["transformer"]["eval"]["loss"],
            }
        )
        print(
            "  winner={winner} ssm_loss={ssm:.4f} transformer_loss={tr:.4f}".format(
                winner=payload["comparison"]["winner_by_loss"],
                ssm=float(payload["ssm"]["eval"]["loss"]),
                tr=float(payload["transformer"]["eval"]["loss"]),
            )
        )

    protocol_evidence = validate_protocol(args, runs)
    if args.require_valid_protocol and not protocol_evidence["valid"]:
        raise SystemExit(f"Multi-seed architecture protocol did not meet requirements: {protocol_evidence}")

    payload = {
        "created_at": timestamp(),
        "kind": "text_architecture_multiseed",
        "ssm_variant": args.ssm_variant,
        "ssm_skip_rank": args.ssm_skip_rank,
        "runs": executions,
        "seed_count": len(seeds),
        "seeds": seeds,
        "protocol_evidence": protocol_evidence,
        "summary": {
            "ssm": architecture_summary(runs, "ssm"),
            "transformer": architecture_summary(runs, "transformer"),
            "comparison": comparison_summary(runs),
            "memory_context": memory_context_summary(runs),
        },
        "claim_guardrail": (
            "This repeated-seed million-token run is stronger evidence than a smoke test, "
            "but it is still a bounded benchmark, not a universal proof of architecture superiority."
        ),
    }
    output_json = Path(args.output_json) if args.output_json else output_dir / "multiseed_summary.json"
    write_json(output_json, payload)

    summary = payload["summary"]
    comparison = summary["comparison"]
    print("")
    print("Summary")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  winner_counts: {comparison['winner_counts']}")
    print(f"  ssm_loss_mean: {summary['ssm']['loss']['mean']:.4f}")
    print(f"  transformer_loss_mean: {summary['transformer']['loss']['mean']:.4f}")
    print(f"  ssm_throughput_mean: {summary['ssm']['throughput_tokens_s']['mean']:,.0f}")
    print(f"  transformer_throughput_mean: {summary['transformer']['throughput_tokens_s']['mean']:,.0f}")
    print(
        "  attention_to_state_memory_ratio_mean: "
        f"{summary['memory_context']['transformer_attention_to_ssm_state_memory_ratio']['mean']:.1f}"
    )
    print(f"Saved JSON: {output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compare_text_architectures.py across repeated seeds and aggregate results."
    )
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--output-dir", default="artifacts/text_architecture_multiseed")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seeds", default="11,17,23")
    parser.add_argument("--tokenizer", choices=("word", "bpe"), default="word")
    parser.add_argument("--tokenizer-config", default=None)
    parser.add_argument("--bpe-vocab-size", type=int, default=512)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--bpe-train-chars", type=int, default=250_000)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-tokens", type=int, default=64)
    parser.add_argument("--min-eval-tokens", type=int, default=16)
    parser.add_argument("--streams", type=int, default=128)
    parser.add_argument("--tokens-per-stream", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--ssm-input-dim", type=int, default=64)
    parser.add_argument("--ssm-state-dim", type=int, default=96)
    parser.add_argument(
        "--ssm-variant",
        choices=SSM_VARIANTS,
        default="static",
    )
    parser.add_argument("--ssm-skip-rank", type=int, default=DEFAULT_SKIP_RANK)
    parser.add_argument("--transformer-d-model", type=int, default=None)
    parser.add_argument("--transformer-max-d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=2)
    parser.add_argument("--transformer-ff-mult", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--memory-gate-l1", type=float, default=0.0)
    parser.add_argument(
        "--ssm-aux-recall-every",
        type=int,
        default=0,
        help="Pass-through for compare_text_architectures.py SSM-only delayed-recall curriculum.",
    )
    parser.add_argument("--ssm-aux-recall-task", choices=("delayed-key", "assignment"), default="delayed-key")
    parser.add_argument("--ssm-aux-recall-delay", type=int, default=32)
    parser.add_argument("--ssm-aux-recall-key-count", type=int, default=64)
    parser.add_argument("--ssm-aux-recall-lr-scale", type=float, default=0.25)
    parser.add_argument("--ssm-aux-recall-init-gates", action="store_true")
    parser.add_argument("--ssm-aux-recall-write-logit", type=float, default=8.0)
    parser.add_argument("--ssm-aux-recall-read-logit", type=float, default=8.0)
    parser.add_argument("--ssm-aux-recall-closed-logit", type=float, default=-8.0)
    parser.add_argument("--ssm-aux-recall-logit-scale", type=float, default=32.0)
    parser.add_argument("--ssm-aux-recall-query-id", type=int, default=1)
    parser.add_argument("--ssm-aux-recall-filler-id", type=int, default=0)
    parser.add_argument("--ssm-aux-recall-assign-id", type=int, default=2)
    parser.add_argument("--ssm-aux-recall-separator-id", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default="memory is")
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--min-train-positions", type=int, default=1_000_000)
    parser.add_argument("--min-eval-positions", type=int, default=1_000_000)
    parser.add_argument("--min-corpus-train-tokens", type=int, default=1_000_000)
    parser.add_argument("--min-corpus-eval-tokens", type=int, default=1_000_000)
    parser.add_argument("--require-valid-protocol", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default="cpu",
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
