#!/usr/bin/env python3
"""Profile a text corpus before feeding it into the SoA/SSM text runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter

import numpy as np

from train_ssm_text import DEFAULT_TEXT
from text_tokenization import (
    build_tokenizer_config,
    default_word_tokenizer_config,
    tokenize_with_config,
    tokenizer_summary,
)


def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_checkpoint_vocab(path: str) -> list[str]:
    data = np.load(path, allow_pickle=False)
    if "vocab_json" not in data:
        raise SystemExit(f"Checkpoint does not contain vocab_json: {path}")
    return json.loads(str(data["vocab_json"].item()))


def load_checkpoint_tokenizer_config(path: str) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    if "tokenizer_config_json" in data:
        return json.loads(str(data["tokenizer_config_json"].item()))
    if "config_json" in data:
        config = json.loads(str(data["config_json"].item()))
        if isinstance(config.get("tokenizer_config"), dict):
            return config["tokenizer_config"]
    return default_word_tokenizer_config()


def count_token_classes(tokens: list[str]) -> dict[str, int]:
    words = 0
    numbers = 0
    punctuation = 0
    other = 0
    for token in tokens:
        if re.fullmatch(r"\w+", token, flags=re.UNICODE):
            if any(ch.isdigit() for ch in token):
                numbers += 1
            else:
                words += 1
        elif re.fullmatch(r"[^\w\s]", token, flags=re.UNICODE):
            punctuation += 1
        else:
            other += 1
    return {
        "words": words,
        "numbers": numbers,
        "punctuation": punctuation,
        "other": other,
    }


def recommend_shape(
    token_count: int,
    *,
    target_streams: int,
    min_seq_len: int,
    max_seq_len: int,
) -> dict[str, int]:
    usable_tokens = max(token_count + 1, 16)
    streams = max(1, min(target_streams, usable_tokens // 16))
    tokens_per_stream = max(16, math.ceil(usable_tokens / streams))

    # Prefer a window that gives enough chunks per stream for truncated BPTT
    # while staying friendly to XLA compilation and CPU cache.
    preferred = max(min_seq_len, min(max_seq_len, tokens_per_stream // 8))
    seq_len = 1 << max(0, int(math.floor(math.log2(max(1, preferred)))))
    seq_len = max(min_seq_len, min(max_seq_len, seq_len))
    if seq_len + 1 > tokens_per_stream:
        seq_len = max(1, tokens_per_stream - 1)

    chunks_per_epoch = max(1, (tokens_per_stream - 1) // seq_len)
    train_positions_per_epoch = streams * chunks_per_epoch * seq_len
    return {
        "streams": streams,
        "tokens_per_stream": tokens_per_stream,
        "seq_len": seq_len,
        "chunks_per_epoch": chunks_per_epoch,
        "train_positions_per_epoch": train_positions_per_epoch,
    }


def estimate_memory_bytes(
    *,
    vocab_size: int,
    streams: int,
    tokens_per_stream: int,
    input_dim: int,
    state_dim: int,
) -> dict[str, int]:
    token_streams = streams * tokens_per_stream * np.dtype(np.int32).itemsize
    states_h = streams * state_dim * np.dtype(np.float32).itemsize
    embed_x = vocab_size * input_dim * np.dtype(np.float32).itemsize
    weights_a = state_dim * np.dtype(np.float32).itemsize
    weights_b = input_dim * state_dim * np.dtype(np.float32).itemsize
    weights_c = state_dim * vocab_size * np.dtype(np.float32).itemsize
    params = embed_x + weights_a + weights_b + weights_c
    total = token_streams + states_h + params
    return {
        "token_streams": token_streams,
        "states_h": states_h,
        "embed_x": embed_x,
        "weights_a": weights_a,
        "weights_b": weights_b,
        "weights_c": weights_c,
        "params": params,
        "estimated_total": total,
    }


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} {unit}"
        amount /= 1024.0
    return f"{value} B"


def profile(args: argparse.Namespace) -> dict[str, object]:
    text = load_text(args.text_file)
    if args.checkpoint:
        tokenizer_config = load_checkpoint_tokenizer_config(args.checkpoint)
    else:
        loaded_tokenizer_config = load_tokenizer_config(args.tokenizer_config)
        tokenizer_config = build_tokenizer_config(
            text,
            tokenizer=args.tokenizer,
            bpe_vocab_size=args.bpe_vocab_size,
            bpe_min_frequency=args.bpe_min_frequency,
            bpe_train_chars=args.bpe_train_chars,
            tokenizer_config=loaded_tokenizer_config,
        )
    tokens = tokenize_with_config(text, tokenizer_config)
    if not tokens:
        raise SystemExit("Corpus has no tokens after tokenization.")

    counts = Counter(tokens)
    vocab_size = len(counts) + 2
    shape = recommend_shape(
        len(tokens),
        target_streams=args.target_streams,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
    )
    memory = estimate_memory_bytes(
        vocab_size=vocab_size,
        streams=shape["streams"],
        tokens_per_stream=shape["tokens_per_stream"],
        input_dim=args.input_dim,
        state_dim=args.state_dim,
    )

    checkpoint_report: dict[str, object] | None = None
    if args.checkpoint:
        checkpoint_vocab = load_checkpoint_vocab(args.checkpoint)
        checkpoint_tokens = set(checkpoint_vocab)
        oov_counts = {
            token: count
            for token, count in counts.items()
            if token not in checkpoint_tokens
        }
        oov_total = sum(oov_counts.values())
        checkpoint_report = {
            "path": args.checkpoint,
            "vocab_size": len(checkpoint_vocab),
            "known_tokens": len(counts) - len(oov_counts),
            "oov_unique": len(oov_counts),
            "oov_tokens": oov_total,
            "oov_token_rate": oov_total / len(tokens),
            "top_oov": [
                {"token": token, "count": count}
                for token, count in sorted(oov_counts.items(), key=lambda item: (-item[1], item[0]))[: args.top_k]
            ],
        }

    return {
        "source": args.text_file or "builtin_default",
        "tokenizer": tokenizer_config.get("type", "word"),
        "tokenizer_config": tokenizer_config,
        "tokenizer_summary": tokenizer_summary(text, tokenizer_config, tokens),
        "characters": len(text),
        "tokens": len(tokens),
        "unique_tokens": len(counts),
        "vocab_size_with_specials": vocab_size,
        "token_classes": count_token_classes(tokens),
        "top_tokens": [
            {"token": token, "count": count}
            for token, count in counts.most_common(args.top_k)
        ],
        "recommended_training_args": shape,
        "estimated_memory_bytes": memory,
        "estimated_memory_human": {
            key: human_bytes(value)
            for key, value in memory.items()
        },
        "checkpoint_oov": checkpoint_report,
    }


def save_json(path: str, payload: dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_tokenizer_config(path: str | None) -> dict[str, object] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_report(result: dict[str, object]) -> None:
    args = result["recommended_training_args"]
    memory = result["estimated_memory_human"]
    tokenizer = result["tokenizer_summary"]
    print("TextPy/SoA corpus profiler")
    print("")
    print("Corpus")
    print(f"  source: {result['source']}")
    print(f"  tokenizer: {result['tokenizer']}")
    print(f"  characters: {result['characters']:,}")
    print(f"  word_tokens: {tokenizer['word_tokens']:,}")
    print(f"  tokens: {result['tokens']:,}")
    print(f"  tokens_per_word: {tokenizer['tokens_per_word']:.3f}")
    if result["tokenizer"] == "bpe":
        print(f"  bpe_merges: {tokenizer['merges']:,}")
    print(f"  unique_tokens: {result['unique_tokens']:,}")
    print(f"  vocab_size_with_specials: {result['vocab_size_with_specials']:,}")
    print(f"  token_classes: {result['token_classes']}")
    print("")
    print("Recommended training args")
    print(f"  --streams {args['streams']}")
    print(f"  --tokens-per-stream {args['tokens_per_stream']}")
    print(f"  --seq-len {args['seq_len']}")
    print(f"  chunks_per_epoch: {args['chunks_per_epoch']:,}")
    print(f"  train_positions_per_epoch: {args['train_positions_per_epoch']:,}")
    print("")
    print("Estimated SoA memory")
    print(f"  token_streams: {memory['token_streams']}")
    print(f"  states_h: {memory['states_h']}")
    print(f"  params: {memory['params']}")
    print(f"  estimated_total: {memory['estimated_total']}")

    checkpoint = result["checkpoint_oov"]
    if checkpoint is not None:
        print("")
        print("Checkpoint OOV")
        print(f"  checkpoint: {checkpoint['path']}")
        print(f"  checkpoint_vocab_size: {checkpoint['vocab_size']:,}")
        print(f"  known_unique_tokens: {checkpoint['known_tokens']:,}")
        print(f"  oov_unique: {checkpoint['oov_unique']:,}")
        print(f"  oov_tokens: {checkpoint['oov_tokens']:,}")
        print(f"  oov_token_rate: {checkpoint['oov_token_rate']:.4f}")

    print("")
    print("Top tokens")
    for item in result["top_tokens"]:
        print(f"  {item['token']!r}: {item['count']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile text before SoA/SSM training.")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--checkpoint", default=None, help="Optional text checkpoint for OOV analysis.")
    parser.add_argument("--tokenizer", choices=("word", "bpe"), default="word")
    parser.add_argument("--tokenizer-config", default=None, help="Reuse a saved tokenizer_config JSON.")
    parser.add_argument("--output-tokenizer-json", default=None, help="Write tokenizer_config JSON.")
    parser.add_argument("--bpe-vocab-size", type=int, default=512)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--bpe-train-chars", type=int, default=250_000)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--target-streams", type=int, default=64)
    parser.add_argument("--min-seq-len", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=48)
    parser.add_argument("--state-dim", type=int, default=64)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    result = profile(args)
    print_report(result)
    if args.output_tokenizer_json:
        save_json(args.output_tokenizer_json, result["tokenizer_config"])
        print("")
        print(f"Saved tokenizer JSON: {args.output_tokenizer_json}")
    if args.output_json:
        save_json(args.output_json, result)
        print("")
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    run(parse_args())
