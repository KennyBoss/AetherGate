#!/usr/bin/env python3
"""Probe hidden-state memory dynamics of a saved TextPy/SoA text checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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
import numpy as np

from train_ssm_text import BOS, UNK, decay_from_raw, load_checkpoint
from text_tokenization import default_word_tokenizer_config, detokenize_with_config, tokenize_with_config


def load_json(path: str | None) -> dict[str, Any] | None:
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
    release = load_json(args.release)
    if release is not None:
        release_dir = Path(args.release).parent
        if args.manifest is None:
            args.manifest = str(resolve_package_path(release_dir, str(release["manifest"])))
        if args.checkpoint is None:
            args.checkpoint = str(resolve_package_path(release_dir, str(release["checkpoint"])))

    manifest = load_json(args.manifest)
    if manifest is not None and args.checkpoint is None:
        manifest_dir = Path(args.manifest).parent
        args.checkpoint = str(resolve_package_path(manifest_dir, str(manifest["promoted_checkpoint"])))

    if args.checkpoint is None:
        raise SystemExit("Provide --checkpoint, --manifest, or --release.")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be at least 1.")
    return args


@jax.jit
def state_trace(params: dict[str, jax.Array], input_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    input_x = params["token_embed"][input_ids]
    decay_a = decay_from_raw(params["decay_raw"])
    initial_h = jnp.zeros((params["decay_raw"].shape[0],), dtype=jnp.float32)

    def step(state_h: jax.Array, x_t: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        state_h = jnp.tanh(state_h * decay_a + x_t @ params["input_b"] + params["hidden_bias"])
        logits = state_h @ params["output_c"] + params["output_bias"]
        return state_h, (state_h, logits)

    _, (states_h, logits) = jax.lax.scan(step, initial_h, input_x)
    return states_h, logits


def top_predictions(logits: np.ndarray, vocab: list[str], top_k: int) -> list[dict[str, float | str]]:
    shifted = logits - np.max(logits)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs)
    indexes = np.argsort(-probs)[:top_k]
    return [
        {
            "token": vocab[int(index)],
            "probability": float(probs[int(index)]),
        }
        for index in indexes
    ]


def probe(args: argparse.Namespace) -> dict[str, Any]:
    params, _, vocab, config = load_checkpoint(args.checkpoint)
    token_to_id = {token: i for i, token in enumerate(vocab)}
    unk_id = token_to_id[UNK]
    tokenizer_config = config.get("tokenizer_config") or default_word_tokenizer_config()
    tokens = tokenize_with_config(args.prompt, tokenizer_config)
    if args.include_bos:
        tokens = [BOS, *tokens]
    if not tokens:
        tokens = [BOS]
    tokens = tokens[: args.max_tokens]
    input_ids = np.asarray([token_to_id.get(token, unk_id) for token in tokens], dtype=np.int32)

    states_h, logits = state_trace(params, jnp.asarray(input_ids))
    jax.block_until_ready(states_h)
    states_np = np.asarray(states_h)
    logits_np = np.asarray(logits)
    state_norms = np.linalg.norm(states_np, axis=1)
    state_delta_norms = np.zeros_like(state_norms)
    if len(state_norms) > 1:
        state_delta_norms[1:] = np.linalg.norm(states_np[1:] - states_np[:-1], axis=1)

    rows: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        top = top_predictions(logits_np[index], vocab, args.top_k)
        row = {
            "step": index,
            "token": token,
            "token_id": int(input_ids[index]),
            "state_norm": float(state_norms[index]),
            "state_delta_norm": float(state_delta_norms[index]),
            "state_mean": float(np.mean(states_np[index])),
            "state_std": float(np.std(states_np[index])),
            "top_predictions": top,
            "top_prediction": top[0]["token"],
            "top_probability": top[0]["probability"],
        }
        if args.include_state_vectors:
            row["state_vector"] = states_np[index].astype(float).tolist()
        rows.append(row)

    summary = {
        "step_count": len(rows),
        "final_state_norm": float(state_norms[-1]),
        "mean_state_norm": float(np.mean(state_norms)),
        "max_state_delta_norm": float(np.max(state_delta_norms)),
        "final_top_prediction": rows[-1]["top_prediction"],
        "final_top_probability": rows[-1]["top_probability"],
        "has_state_vectors": bool(args.include_state_vectors),
    }
    if args.include_state_vectors:
        summary["final_state_vector"] = states_np[-1].astype(float).tolist()

    return {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "release": args.release,
        "checkpoint_config": config,
        "tokenizer": tokenizer_config.get("type", "word"),
        "tokenizer_config": tokenizer_config,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "prompt": args.prompt,
        "tokens": tokens,
        "detokenized_tokens": detokenize_with_config(tokens, tokenizer_config),
        "vocab_size": len(vocab),
        "state_dim": int(params["decay_raw"].shape[0]),
        "top_k": args.top_k,
        "steps": rows,
        "summary": summary,
    }


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_csv(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "token",
        "token_id",
        "state_norm",
        "state_delta_norm",
        "state_mean",
        "state_std",
        "top_prediction",
        "top_probability",
        "top_predictions",
        "state_vector",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["steps"]:
            csv_row = dict(row)
            csv_row["top_predictions"] = json.dumps(row["top_predictions"], ensure_ascii=False)
            if "state_vector" in csv_row:
                csv_row["state_vector"] = json.dumps(csv_row["state_vector"])
            else:
                csv_row["state_vector"] = ""
            writer.writerow(csv_row)


def print_report(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("TextPy/SoA text memory probe")
    print(f"backend: {payload['backend']}")
    print(f"checkpoint: {payload['checkpoint']}")
    if payload["manifest"]:
        print(f"manifest: {payload['manifest']}")
    if payload["release"]:
        print(f"release: {payload['release']}")
    print(f"prompt: {payload['prompt']}")
    print(f"tokens: {payload['detokenized_tokens']}")
    print("")
    print("Memory")
    print(f"  state_dim: {payload['state_dim']}")
    print(f"  steps: {summary['step_count']}")
    print(f"  final_state_norm: {summary['final_state_norm']:.4f}")
    print(f"  mean_state_norm: {summary['mean_state_norm']:.4f}")
    print(f"  max_state_delta_norm: {summary['max_state_delta_norm']:.4f}")
    print(f"  final_top_prediction: {summary['final_top_prediction']}")
    print(f"  final_top_probability: {summary['final_top_probability']:.4f}")
    print(f"  state_vectors: {summary['has_state_vectors']}")
    print("")
    print("Timeline")
    for row in payload["steps"]:
        print(
            f"  {row['step']:02d} {row['token']!r:<14} "
            f"| norm={row['state_norm']:.4f} "
            f"delta={row['state_delta_norm']:.4f} "
            f"top={row['top_prediction']!r} "
            f"p={row['top_probability']:.4f}"
        )


def run(args: argparse.Namespace) -> None:
    args = resolve_args(args)
    payload = probe(args)
    print_report(payload)
    if args.output_json:
        save_json(args.output_json, payload)
        print("")
        print(f"Saved JSON: {args.output_json}")
    if args.output_csv:
        save_csv(args.output_csv, payload)
        print(f"Saved CSV: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe hidden-state memory dynamics for a text SSM checkpoint.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--release", default=None)
    parser.add_argument("--prompt", default="memory is a river")
    parser.add_argument("--include-bos", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-state-vectors", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
