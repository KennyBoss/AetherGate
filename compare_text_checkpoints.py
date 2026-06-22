#!/usr/bin/env python3
"""Compare two text SSM checkpoints on the same corpus and eval shape."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter


def requested_backend(argv: list[str]) -> str:
    backend = os.environ.get("JAX_PLATFORM_NAME", "cpu")
    for i, arg in enumerate(argv):
        if arg == "--backend" and i + 1 < len(argv):
            backend = argv[i + 1]
        elif arg.startswith("--backend="):
            backend = arg.split("=", 1)[1]
    return backend


REQUESTED_BACKEND = requested_backend(sys.argv[1:])
if REQUESTED_BACKEND == "auto":
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ.pop("JAX_PLATFORMS", None)
else:
    os.environ["JAX_PLATFORM_NAME"] = REQUESTED_BACKEND
    os.environ["JAX_PLATFORMS"] = REQUESTED_BACKEND

import jax
import jax.numpy as jnp
import numpy as np

from train_ssm_text import (
    encode,
    eval_loss,
    load_checkpoint,
    load_text,
    make_streams,
    sample_text,
)
from text_tokenization import default_word_tokenizer_config, tokenize_with_config


def oov_report(tokens: list[str], vocab: list[str]) -> dict[str, object]:
    counts = Counter(tokens)
    vocab_set = set(vocab)
    oov_counts = {
        token: count
        for token, count in counts.items()
        if token not in vocab_set
    }
    oov_tokens = sum(oov_counts.values())
    return {
        "known_unique_tokens": len(counts) - len(oov_counts),
        "oov_unique_tokens": len(oov_counts),
        "oov_tokens": oov_tokens,
        "oov_token_rate": oov_tokens / max(1, len(tokens)),
    }


def evaluate_one(
    *,
    label: str,
    checkpoint: str,
    text: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    params, _, vocab, config = load_checkpoint(checkpoint)
    token_to_id = {token: i for i, token in enumerate(vocab)}
    tokenizer_config = config.get("tokenizer_config") or default_word_tokenizer_config()
    tokens = tokenize_with_config(text, tokenizer_config)
    if len(tokens) < 16:
        raise SystemExit("Need at least 16 tokens to compare text checkpoints.")
    token_ids = encode(tokens, token_to_id)
    streams_np = make_streams(token_ids, args.streams, args.tokens_per_stream)
    streams = jnp.asarray(streams_np)

    if args.seq_len + 1 > args.tokens_per_stream:
        raise SystemExit("--seq-len must be smaller than --tokens-per-stream.")

    warmup = eval_loss(params, streams, seq_len=args.seq_len)
    jax.block_until_ready(warmup.loss)

    eval_times: list[float] = []
    metrics = warmup
    for _ in range(args.repeat):
        start = time.perf_counter()
        metrics = eval_loss(params, streams, seq_len=args.seq_len)
        jax.block_until_ready(metrics.loss)
        eval_times.append(time.perf_counter() - start)

    eval_s = min(eval_times)
    evaluated_tokens = args.streams * args.seq_len
    throughput = evaluated_tokens / max(eval_s, 1.0e-9)
    sample = sample_text(
        params,
        token_to_id,
        vocab,
        args.prompt,
        args.sample_steps,
        args.temperature,
        args.seed + 1,
        tokenizer_config=tokenizer_config,
    )
    return {
        "label": label,
        "checkpoint": checkpoint,
        "checkpoint_config": config,
        "tokenizer": tokenizer_config.get("type", "word"),
        "tokenizer_config": tokenizer_config,
        "source_tokens": len(tokens),
        "vocab_size": len(vocab),
        "streams": args.streams,
        "tokens_per_stream": args.tokens_per_stream,
        "seq_len": args.seq_len,
        "repeat": args.repeat,
        "evaluated_tokens_per_repeat": evaluated_tokens,
        "best_eval_s": eval_s,
        "mean_eval_s": float(np.mean(eval_times)),
        "throughput_tokens_s": throughput,
        "loss": float(metrics.loss),
        "accuracy": float(metrics.accuracy),
        "mean_decay_A": float(metrics.mean_decay),
        "oov": oov_report(tokens, vocab),
        "sample": sample,
    }


def compare_results(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    loss_delta = candidate["loss"] - baseline["loss"]
    accuracy_delta = candidate["accuracy"] - baseline["accuracy"]
    throughput_delta = candidate["throughput_tokens_s"] - baseline["throughput_tokens_s"]
    loss_regression_rate = loss_delta / max(1.0e-9, baseline["loss"])
    throughput_regression_rate = -throughput_delta / max(1.0e-9, baseline["throughput_tokens_s"])

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
        "throughput_tokens_s_delta": throughput_delta,
        "loss_regression_rate": loss_regression_rate,
        "throughput_regression_rate": throughput_regression_rate,
    }


def save_json(path: str, payload: dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_result(result: dict[str, object]) -> None:
    print(f"{result['label']}")
    print(f"  checkpoint: {result['checkpoint']}")
    print(f"  loss: {result['loss']:.4f}")
    print(f"  accuracy: {result['accuracy']:.4f}")
    print(f"  throughput_tokens_s: {result['throughput_tokens_s']:,.0f}")
    print(f"  vocab_size: {result['vocab_size']:,}")
    print(f"  oov_token_rate: {result['oov']['oov_token_rate']:.4f}")
    print(f"  sample: {result['sample']}")


def run(args: argparse.Namespace) -> None:
    text = load_text(args.text_file)
    baseline = evaluate_one(
        label="baseline",
        checkpoint=args.baseline,
        text=text,
        args=args,
    )
    candidate = evaluate_one(
        label="candidate",
        checkpoint=args.candidate,
        text=text,
        args=args,
    )
    comparison = compare_results(baseline, candidate)
    payload = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "text_file": args.text_file,
        "source_tokens": candidate["source_tokens"],
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }

    print("TextPy/SoA text checkpoint comparison")
    print(f"backend: {payload['backend']}")
    print("")
    print_result(baseline)
    print("")
    print_result(candidate)
    print("")
    print("Comparison")
    print(f"  winner_by_loss: {comparison['winner_by_loss']}")
    print(f"  loss_delta: {comparison['loss_delta']:.6f}")
    print(f"  accuracy_delta: {comparison['accuracy_delta']:.6f}")
    print(f"  throughput_tokens_s_delta: {comparison['throughput_tokens_s_delta']:,.0f}")
    print(f"  loss_regression_rate: {comparison['loss_regression_rate']:.4f}")
    print(f"  throughput_regression_rate: {comparison['throughput_regression_rate']:.4f}")

    if args.output_json:
        save_json(args.output_json, payload)
        print("")
        print(f"Saved JSON: {args.output_json}")

    failed = False
    if args.fail_on_loss_regression and comparison["loss_regression_rate"] > args.max_loss_regression:
        print(
            f"FAIL loss regression {comparison['loss_regression_rate']:.4f} "
            f"> {args.max_loss_regression:.4f}",
            file=sys.stderr,
        )
        failed = True
    if (
        args.fail_on_throughput_regression
        and comparison["throughput_regression_rate"] > args.max_throughput_regression
    ):
        print(
            f"FAIL throughput regression {comparison['throughput_regression_rate']:.4f} "
            f"> {args.max_throughput_regression:.4f}",
            file=sys.stderr,
        )
        failed = True
    if failed:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two saved text SSM checkpoints.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--tokens-per-stream", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--prompt", default="memory is")
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-on-loss-regression", action="store_true")
    parser.add_argument("--max-loss-regression", type=float, default=0.01)
    parser.add_argument("--fail-on-throughput-regression", action="store_true")
    parser.add_argument("--max-throughput-regression", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
