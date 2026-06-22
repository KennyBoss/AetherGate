#!/usr/bin/env python3
"""Benchmark a saved or promoted TextPy/SoA text checkpoint package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


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

from train_ssm_text import (
    encode,
    eval_loss,
    load_checkpoint,
    load_text,
    make_streams,
    sample_text,
)
from text_tokenization import default_word_tokenizer_config, tokenize_with_config


def load_manifest(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint_path(manifest_path: str | None, raw_checkpoint: str) -> Path:
    checkpoint = Path(raw_checkpoint)
    if manifest_path is not None:
        package_dir = Path(manifest_path).parent
        if not checkpoint.is_absolute():
            local_exact = package_dir / checkpoint
            if local_exact.exists():
                return local_exact
        local_by_name = package_dir / checkpoint.name
        if local_by_name.exists():
            return local_by_name
    if checkpoint.is_absolute() or checkpoint.exists():
        return checkpoint
    if manifest_path is not None:
        return Path(manifest_path).parent / checkpoint.name
    return checkpoint


def resolve_args(args: argparse.Namespace) -> tuple[argparse.Namespace, dict[str, Any] | None]:
    manifest = load_manifest(args.manifest)
    if manifest is not None:
        if args.checkpoint is None:
            args.checkpoint = str(resolve_checkpoint_path(args.manifest, str(manifest["promoted_checkpoint"])))
        selected = manifest.get("selected_run", {})
        if args.streams is None:
            args.streams = int(selected.get("streams", 8))
        if args.tokens_per_stream is None:
            args.tokens_per_stream = int(selected.get("tokens_per_stream", 128))
        if args.seq_len is None:
            args.seq_len = int(selected.get("seq_len", 16))
    if args.checkpoint is None:
        raise SystemExit("Provide --checkpoint or --manifest.")
    args.streams = 8 if args.streams is None else args.streams
    args.tokens_per_stream = 128 if args.tokens_per_stream is None else args.tokens_per_stream
    args.seq_len = 16 if args.seq_len is None else args.seq_len
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative.")
    return args, manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_integrity(checkpoint: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    actual_bytes = checkpoint.stat().st_size
    actual_sha256 = sha256_file(checkpoint)
    report = {
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "manifest_bytes": None,
        "manifest_sha256": None,
        "matches_manifest": None,
    }
    if manifest is None:
        return report

    expected_bytes = manifest.get("promoted_checkpoint_bytes")
    expected_sha256 = manifest.get("promoted_checkpoint_sha256")
    report["manifest_bytes"] = expected_bytes
    report["manifest_sha256"] = expected_sha256
    bytes_match = expected_bytes is None or int(expected_bytes) == actual_bytes
    sha_match = expected_sha256 is None or str(expected_sha256) == actual_sha256
    report["matches_manifest"] = bytes_match and sha_match
    if not report["matches_manifest"]:
        raise SystemExit(
            "Checkpoint integrity mismatch: "
            f"bytes {actual_bytes} != {expected_bytes}, sha256 {actual_sha256} != {expected_sha256}"
        )
    return report


def oov_report(tokens: list[str], vocab: list[str]) -> dict[str, Any]:
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


def benchmark(args: argparse.Namespace, manifest: dict[str, Any] | None) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    integrity = None if args.skip_integrity_check else verify_checkpoint_integrity(checkpoint, manifest)
    params, _, vocab, config = load_checkpoint(str(checkpoint))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    tokenizer_config = config.get("tokenizer_config") or default_word_tokenizer_config()

    text = load_text(args.text_file)
    tokens = tokenize_with_config(text, tokenizer_config)
    if len(tokens) < 16:
        raise SystemExit("Need at least 16 tokens to benchmark the text checkpoint.")
    if args.seq_len + 1 > args.tokens_per_stream:
        raise SystemExit("--seq-len must be smaller than --tokens-per-stream.")

    token_ids = encode(tokens, token_to_id)
    streams_np = make_streams(token_ids, args.streams, args.tokens_per_stream)
    streams = jnp.asarray(streams_np)

    warmup_times: list[float] = []
    metrics = None
    for _ in range(args.warmup):
        start = time.perf_counter()
        metrics = eval_loss(params, streams, seq_len=args.seq_len)
        jax.block_until_ready(metrics.loss)
        warmup_times.append(time.perf_counter() - start)

    eval_times: list[float] = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        metrics = eval_loss(params, streams, seq_len=args.seq_len)
        jax.block_until_ready(metrics.loss)
        eval_times.append(time.perf_counter() - start)

    assert metrics is not None
    evaluated_tokens = args.streams * args.seq_len
    best_eval_s = min(eval_times)
    mean_eval_s = statistics.fmean(eval_times)
    total_eval_s = sum(eval_times)
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

    selected_run = manifest.get("selected_run") if manifest else None
    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "checkpoint": str(checkpoint),
        "manifest": args.manifest,
        "manifest_promoted_checkpoint": manifest.get("promoted_checkpoint") if manifest else None,
        "checkpoint_integrity": integrity,
        "checkpoint_config": config,
        "tokenizer": tokenizer_config.get("type", "word"),
        "tokenizer_config": tokenizer_config,
        "selected_run": selected_run,
        "text_file": args.text_file,
        "source_tokens": len(tokens),
        "vocab_size": len(vocab),
        "streams": args.streams,
        "tokens_per_stream": args.tokens_per_stream,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "warmup_times_s": warmup_times,
        "eval_times_s": eval_times,
        "evaluated_tokens_per_repeat": evaluated_tokens,
        "total_timed_tokens": evaluated_tokens * args.repeat,
        "best_eval_s": best_eval_s,
        "mean_eval_s": mean_eval_s,
        "median_eval_s": statistics.median(eval_times),
        "total_eval_s": total_eval_s,
        "best_throughput_tokens_s": evaluated_tokens / max(best_eval_s, 1.0e-9),
        "mean_throughput_tokens_s": evaluated_tokens / max(mean_eval_s, 1.0e-9),
        "aggregate_throughput_tokens_s": (evaluated_tokens * args.repeat) / max(total_eval_s, 1.0e-9),
        "loss": float(metrics.loss),
        "accuracy": float(metrics.accuracy),
        "mean_decay_A": float(metrics.mean_decay),
        "oov": oov_report(tokens, vocab),
        "prompt": args.prompt,
        "sample": sample,
    }


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_report(result: dict[str, Any]) -> None:
    print("TextPy/SoA text package benchmark")
    print(f"backend: {result['backend']}")
    print(f"devices: {result['devices']}")
    print("")
    print("Artifact")
    print(f"  checkpoint: {result['checkpoint']}")
    if result["manifest"]:
        print(f"  manifest: {result['manifest']}")
    if result["checkpoint_integrity"]:
        integrity = result["checkpoint_integrity"]
        print(f"  checkpoint_bytes: {integrity['bytes']}")
        print(f"  checkpoint_sha256: {integrity['sha256']}")
        if integrity["matches_manifest"] is not None:
            print(f"  matches_manifest: {integrity['matches_manifest']}")
    print(f"  checkpoint_config: {result['checkpoint_config']}")
    print(f"  tokenizer: {result['tokenizer']}")
    print("")
    print("Eval Shape")
    print(f"  source_tokens: {result['source_tokens']:,}")
    print(f"  vocab_size: {result['vocab_size']:,}")
    print(f"  streams: {result['streams']:,}")
    print(f"  tokens_per_stream: {result['tokens_per_stream']:,}")
    print(f"  seq_len: {result['seq_len']:,}")
    print(f"  evaluated_tokens_per_repeat: {result['evaluated_tokens_per_repeat']:,}")
    print("")
    print("Timing")
    print(f"  warmup: {result['warmup']:,}")
    print(f"  repeat: {result['repeat']:,}")
    print(f"  best_eval_s: {result['best_eval_s']:.6f}")
    print(f"  mean_eval_s: {result['mean_eval_s']:.6f}")
    print(f"  median_eval_s: {result['median_eval_s']:.6f}")
    print(f"  best_throughput_tokens_s: {result['best_throughput_tokens_s']:,.0f}")
    print(f"  mean_throughput_tokens_s: {result['mean_throughput_tokens_s']:,.0f}")
    print(f"  aggregate_throughput_tokens_s: {result['aggregate_throughput_tokens_s']:,.0f}")
    print("")
    print("Quality")
    print(f"  loss: {result['loss']:.4f}")
    print(f"  accuracy: {result['accuracy']:.4f}")
    print(f"  mean_decay_A: {result['mean_decay_A']:.4f}")
    print(f"  oov_token_rate: {result['oov']['oov_token_rate']:.4f}")
    print("")
    print("Sample")
    print("  " + str(result["sample"]))


def run(args: argparse.Namespace) -> None:
    args, manifest = resolve_args(args)
    result = benchmark(args, manifest)
    print_report(result)
    if args.output_json:
        save_json(args.output_json, result)
        print("")
        print(f"Saved JSON: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a saved or promoted text SSM package.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--streams", type=int, default=None)
    parser.add_argument("--tokens-per-stream", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--prompt", default="memory is")
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--skip-integrity-check", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
