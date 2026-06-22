#!/usr/bin/env python3
"""Evaluate and sample from a saved train_ssm_text.py checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


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


def load_manifest(path: str | None) -> dict[str, object] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_package_path(base: Path, value: str) -> Path:
    path = Path(value)
    local_by_name = base / path.name
    if local_by_name.exists():
        return local_by_name
    if path.is_absolute() or path.exists():
        return path
    return local_by_name


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    release = load_manifest(args.release)
    if release is not None:
        release_dir = Path(args.release).parent
        if args.manifest is None:
            args.manifest = str(resolve_package_path(release_dir, str(release["manifest"])))
        if args.checkpoint is None:
            args.checkpoint = str(resolve_package_path(release_dir, str(release["checkpoint"])))
        if args.streams is None and release.get("validation", {}).get("has_benchmark_json"):
            benchmark_path = Path(args.manifest).parent / "benchmark.json"
            if benchmark_path.exists():
                benchmark = load_manifest(str(benchmark_path)) or {}
                args.streams = int(benchmark.get("streams", 8))
                args.tokens_per_stream = int(benchmark.get("tokens_per_stream", 128))
                args.seq_len = int(benchmark.get("seq_len", 16))

    manifest = load_manifest(args.manifest)
    if manifest is not None:
        manifest_dir = Path(args.manifest).parent
        if args.checkpoint is None:
            args.checkpoint = str(resolve_package_path(manifest_dir, str(manifest["promoted_checkpoint"])))
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
    return args


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    params, _, vocab, config = load_checkpoint(args.checkpoint)
    token_to_id = {token: i for i, token in enumerate(vocab)}
    tokenizer_config = config.get("tokenizer_config") or default_word_tokenizer_config()

    text = load_text(args.text_file)
    tokens = tokenize_with_config(text, tokenizer_config)
    if len(tokens) < 16:
        raise SystemExit("Need at least 16 tokens to evaluate the text checkpoint.")

    token_ids = encode(tokens, token_to_id)
    streams_np = make_streams(token_ids, args.streams, args.tokens_per_stream)
    streams = jnp.asarray(streams_np)
    if args.seq_len + 1 > args.tokens_per_stream:
        raise SystemExit("--seq-len must be smaller than --tokens-per-stream.")

    start = time.perf_counter()
    metrics = eval_loss(params, streams, seq_len=args.seq_len)
    jax.block_until_ready(metrics.loss)
    eval_s = time.perf_counter() - start
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
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "release": args.release,
        "checkpoint_config": config,
        "tokenizer": tokenizer_config.get("type", "word"),
        "tokenizer_config": tokenizer_config,
        "source_tokens": len(tokens),
        "vocab_size": len(vocab),
        "streams": args.streams,
        "tokens_per_stream": args.tokens_per_stream,
        "seq_len": args.seq_len,
        "evaluated_tokens": evaluated_tokens,
        "eval_s": eval_s,
        "throughput_tokens_s": throughput,
        "loss": float(metrics.loss),
        "accuracy": float(metrics.accuracy),
        "mean_decay_A": float(metrics.mean_decay),
        "prompt": args.prompt,
        "sample": sample,
    }


def save_json(path: str, payload: dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace) -> None:
    args = resolve_args(args)
    result = evaluate_checkpoint(args)
    print("TextPy/SoA text checkpoint runner")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("Artifact")
    print(f"  checkpoint: {result['checkpoint']}")
    if result["manifest"]:
        print(f"  manifest: {result['manifest']}")
    if result["release"]:
        print(f"  release: {result['release']}")
    print(f"  checkpoint_config: {result['checkpoint_config']}")
    print(f"  tokenizer: {result['tokenizer']}")
    print("")
    print("Eval")
    print(f"  source_tokens: {result['source_tokens']:,}")
    print(f"  vocab_size: {result['vocab_size']:,}")
    print(f"  streams: {result['streams']:,}")
    print(f"  seq_len: {result['seq_len']:,}")
    print(f"  evaluated_tokens: {result['evaluated_tokens']:,}")
    print(f"  eval_s: {result['eval_s']:.4f}")
    print(f"  throughput_tokens_s: {result['throughput_tokens_s']:,.0f}")
    print(f"  loss: {result['loss']:.4f}")
    print(f"  accuracy: {result['accuracy']:.4f}")
    print(f"  mean_decay_A: {result['mean_decay_A']:.4f}")
    print("")
    print("Sample")
    print("  " + str(result["sample"]))
    if args.output_json:
        save_json(args.output_json, result)
        print("")
        print(f"Saved JSON: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a saved text SSM checkpoint.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--release", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--streams", type=int, default=None)
    parser.add_argument("--tokens-per-stream", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--prompt", default="memory is")
    parser.add_argument("--sample-steps", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
