#!/usr/bin/env python3
"""Compare the text SSM against a tiny causal Transformer on one protocol.

This is a research harness, not a victory claim. It trains both models from
scratch on the same train split, evaluates on the same held-out split, and
reports the budget match before comparing loss, perplexity, accuracy, warmed-up
throughput, parameter memory, and effective context length.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple


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

from text_tokenization import (
    BOS,
    UNK,
    build_tokenizer_config,
    build_vocab_for_tokenizer,
    detokenize_with_config,
    encode_tokens,
    tokenize_with_config,
    tokenizer_summary,
    word_detokenize,
    word_tokenize,
)
from train_ssm_text import (
    DEFAULT_SKIP_RANK,
    SSM_VARIANTS,
    adam_update,
    clip_grads,
    forward as ssm_forward,
    init_adam,
    init_params as init_ssm_params,
    infer_ssm_variant,
    load_text,
    make_streams,
    memory_gate_penalty,
    sample_text as sample_ssm_text,
    train_step as ssm_train_step,
)


class EvalMetrics(NamedTuple):
    loss: jax.Array
    accuracy: jax.Array


class TransformerTrainMetrics(NamedTuple):
    loss: jax.Array
    accuracy: jax.Array
    grad_norm: jax.Array


class AuxTrainMetrics(NamedTuple):
    loss: jax.Array
    accuracy: jax.Array
    grad_norm: jax.Array


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_tokenizer_config(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_params(tree: object) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(tree)))


def count_bytes(tree: object) -> int:
    return int(sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(tree)))


def format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{n} B"


def ssm_parameter_diagnostics(params: dict[str, jax.Array]) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    if "memory_write_token" in params:
        write_gate = jax.nn.sigmoid(params["memory_write_token"])
        diagnostics["memory_write_gate_mean"] = float(jnp.mean(write_gate))
        diagnostics["memory_write_gate_min"] = float(jnp.min(write_gate))
        diagnostics["memory_write_gate_max"] = float(jnp.max(write_gate))
    if "memory_read_token" in params:
        read_gate = jax.nn.sigmoid(params["memory_read_token"])
        diagnostics["memory_read_gate_mean"] = float(jnp.mean(read_gate))
        diagnostics["memory_read_gate_min"] = float(jnp.min(read_gate))
        diagnostics["memory_read_gate_max"] = float(jnp.max(read_gate))
    if "memory_logit_scale" in params:
        diagnostics["memory_logit_scale"] = float(params["memory_logit_scale"])
    if "memory_decay_raw" in params:
        diagnostics["memory_decay"] = float(0.90 + 0.099 * jax.nn.sigmoid(params["memory_decay_raw"]))
    if "memory_closed_marker" in params:
        diagnostics["memory_closed_marker"] = float(params["memory_closed_marker"])
    if "memory_sparse_marker" in params:
        diagnostics["memory_sparse_marker"] = float(params["memory_sparse_marker"])
    kv_gate_names = (
        "kv_key_token",
        "kv_value_token",
        "kv_erase_token",
        "kv_write_context_token",
        "kv_deref_token",
        "kv_query_token",
    )
    for name in kv_gate_names:
        if name in params:
            gate = jax.nn.sigmoid(params[name])
            diagnostics[f"{name}_gate_mean"] = float(jnp.mean(gate))
            diagnostics[f"{name}_gate_min"] = float(jnp.min(gate))
            diagnostics[f"{name}_gate_max"] = float(jnp.max(gate))
    if "kv_logit_scale" in params:
        diagnostics["kv_logit_scale"] = float(params["kv_logit_scale"])
    if "kv_decay_raw" in params:
        diagnostics["kv_decay"] = float(0.90 + 0.099 * jax.nn.sigmoid(params["kv_decay_raw"]))
    if "kv_dynamic_read_w_h" in params:
        diagnostics["kv_dynamic_read_w_h_norm"] = float(jnp.linalg.norm(params["kv_dynamic_read_w_h"]))
    if "kv_dynamic_read_w_x" in params:
        diagnostics["kv_dynamic_read_w_x_norm"] = float(jnp.linalg.norm(params["kv_dynamic_read_w_x"]))
    if "kv_dynamic_read_b" in params:
        diagnostics["kv_dynamic_read_bias"] = float(params["kv_dynamic_read_b"][0])
    if "kv_sparse_marker" in params:
        diagnostics["kv_sparse_marker"] = float(params["kv_sparse_marker"])
    return diagnostics


def oov_report(tokens: list[str], vocab: list[str]) -> dict[str, object]:
    counts = Counter(tokens)
    vocab_set = set(vocab)
    oov_counts = {token: count for token, count in counts.items() if token not in vocab_set}
    oov_tokens = sum(oov_counts.values())
    return {
        "known_unique_tokens": len(counts) - len(oov_counts),
        "oov_unique_tokens": len(oov_counts),
        "oov_tokens": oov_tokens,
        "oov_token_rate": oov_tokens / max(1, len(tokens)),
    }


def split_raw_text(args: argparse.Namespace) -> dict[str, object]:
    text = load_text(args.text_file)
    raw_tokens = word_tokenize(text)
    if len(raw_tokens) < args.min_train_tokens + args.min_eval_tokens:
        raise SystemExit(
            "Not enough word tokens for train/eval split: "
            f"{len(raw_tokens)} < {args.min_train_tokens + args.min_eval_tokens}"
        )

    eval_tokens = max(args.min_eval_tokens, int(round(len(raw_tokens) * args.eval_fraction)))
    eval_tokens = min(eval_tokens, len(raw_tokens) - args.min_train_tokens)
    train_tokens = len(raw_tokens) - eval_tokens
    if train_tokens < args.min_train_tokens or eval_tokens < args.min_eval_tokens:
        raise SystemExit(
            f"Invalid train/eval split: train={train_tokens}, eval={eval_tokens}. "
            "Adjust --eval-fraction, --min-train-tokens, or --min-eval-tokens."
        )

    train_text = word_detokenize(raw_tokens[:train_tokens])
    eval_text = word_detokenize(raw_tokens[train_tokens:])
    return {
        "source_text_file": args.text_file or "builtin_default",
        "raw_word_tokens": len(raw_tokens),
        "train_word_tokens": train_tokens,
        "eval_word_tokens": eval_tokens,
        "eval_fraction": eval_tokens / len(raw_tokens),
        "train_text": train_text,
        "eval_text": eval_text,
    }


def prepare_data(args: argparse.Namespace) -> dict[str, object]:
    split = split_raw_text(args)
    loaded_config = load_tokenizer_config(args.tokenizer_config)
    tokenizer_config = build_tokenizer_config(
        str(split["train_text"]),
        tokenizer=args.tokenizer,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_min_frequency=args.bpe_min_frequency,
        bpe_train_chars=args.bpe_train_chars,
        tokenizer_config=loaded_config,
    )
    train_tokens = tokenize_with_config(str(split["train_text"]), tokenizer_config)
    eval_tokens = tokenize_with_config(str(split["eval_text"]), tokenizer_config)
    if len(train_tokens) < 16 or len(eval_tokens) < 8:
        raise SystemExit(
            "Tokenized split is too small for architecture comparison: "
            f"train={len(train_tokens)}, eval={len(eval_tokens)}"
        )

    token_to_id, vocab = build_vocab_for_tokenizer(train_tokens, tokenizer_config)
    train_ids = encode_tokens(train_tokens, token_to_id)
    eval_ids = encode_tokens(eval_tokens, token_to_id)
    train_streams_np = make_streams(train_ids, args.streams, args.tokens_per_stream)
    eval_streams_np = make_streams(eval_ids, args.streams, args.tokens_per_stream)
    if args.seq_len + 1 > args.tokens_per_stream:
        raise SystemExit("--seq-len must be smaller than --tokens-per-stream.")

    return {
        "split": {k: v for k, v in split.items() if k not in {"train_text", "eval_text"}},
        "tokenizer_config": tokenizer_config,
        "tokenizer_summary": tokenizer_summary(str(split["train_text"]), tokenizer_config, train_tokens),
        "train_tokens": train_tokens,
        "eval_tokens": eval_tokens,
        "token_to_id": token_to_id,
        "vocab": vocab,
        "train_streams": jnp.asarray(train_streams_np),
        "eval_streams": jnp.asarray(eval_streams_np),
        "oov": oov_report(eval_tokens, vocab),
    }


def nll_and_accuracy(logits: jax.Array, target_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1)[..., 0]
    predictions = jnp.argmax(logits, axis=-1).astype(target_ids.dtype)
    accuracy = jnp.mean(predictions == target_ids)
    return jnp.mean(nll), accuracy


def masked_nll_and_accuracy(
    logits: jax.Array,
    target_ids: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mask = mask.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1)[..., 0]
    denom = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=jnp.float32))
    loss = jnp.sum(nll * mask) / denom
    predictions = jnp.argmax(logits, axis=-1).astype(target_ids.dtype)
    accuracy = jnp.sum((predictions == target_ids).astype(jnp.float32) * mask) / denom
    return loss, accuracy


@partial(jax.jit, static_argnames=("learning_rate", "grad_clip", "memory_gate_l1"))
def ssm_masked_train_step(
    params: dict[str, jax.Array],
    opt_state: dict[str, object],
    input_ids: jax.Array,
    target_ids: jax.Array,
    mask: jax.Array,
    states_h: jax.Array,
    *,
    learning_rate: float,
    grad_clip: float,
    memory_gate_l1: float = 0.0,
) -> tuple[dict[str, jax.Array], dict[str, object], jax.Array, "AuxTrainMetrics"]:
    def train_loss_fn(train_params: dict[str, jax.Array]) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        logits, final_h = ssm_forward(train_params, input_ids, states_h)
        nll_loss, accuracy = masked_nll_and_accuracy(logits, target_ids, mask)
        penalty = memory_gate_penalty(train_params)
        total_loss = nll_loss + memory_gate_l1 * penalty
        return total_loss, (accuracy, final_h, nll_loss)

    (loss, (accuracy, final_h, nll_loss)), grads = jax.value_and_grad(train_loss_fn, has_aux=True)(params)
    grads, grad_norm = clip_grads(grads, grad_clip)
    params, opt_state = adam_update(params, grads, opt_state, learning_rate)
    final_h = jax.lax.stop_gradient(final_h)
    metrics = AuxTrainMetrics(
        loss=nll_loss,
        accuracy=accuracy,
        grad_norm=grad_norm,
    )
    return params, opt_state, final_h, metrics


@partial(jax.jit, static_argnames=("seq_len",))
def ssm_eval_full(
    params: dict[str, jax.Array],
    streams: jax.Array,
    *,
    seq_len: int,
) -> EvalMetrics:
    chunks = (streams.shape[1] - 1) // seq_len
    usable = chunks * seq_len
    input_chunks = streams[:, :usable].reshape(streams.shape[0], chunks, seq_len)
    target_chunks = streams[:, 1 : usable + 1].reshape(streams.shape[0], chunks, seq_len)
    input_chunks = jnp.swapaxes(input_chunks, 0, 1)
    target_chunks = jnp.swapaxes(target_chunks, 0, 1)
    initial_h = jnp.zeros((streams.shape[0], params["decay_raw"].shape[0]), dtype=jnp.float32)

    def step(states_h: jax.Array, pair: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        input_ids, target_ids = pair
        logits, final_h = ssm_forward(params, input_ids, states_h)
        loss, accuracy = nll_and_accuracy(logits, target_ids)
        return final_h, (loss, accuracy)

    _, (losses, accuracies) = jax.lax.scan(step, initial_h, (input_chunks, target_chunks))
    return EvalMetrics(loss=jnp.mean(losses), accuracy=jnp.mean(accuracies))


def select_aux_recall_key_ids(
    train_streams: jax.Array,
    *,
    vocab_size: int,
    key_count: int,
    skip_count: int = 0,
    exclude_ids: set[int] | None = None,
) -> np.ndarray:
    if key_count <= 0:
        raise SystemExit("--ssm-aux-recall-key-count must be positive when auxiliary recall is enabled.")
    if vocab_size <= 2:
        raise SystemExit("Auxiliary recall needs at least one non-special vocabulary token.")
    flat = np.asarray(train_streams).reshape(-1)
    counts = np.bincount(flat, minlength=vocab_size)
    excluded = exclude_ids or set()
    candidates = [
        int(token_id)
        for token_id in np.argsort(-counts)
        if int(token_id) >= 2 and int(token_id) not in excluded and counts[int(token_id)] > 0
    ]
    if not candidates:
        candidates = [token_id for token_id in range(2, vocab_size) if token_id not in excluded]
    candidates = candidates[skip_count:]
    if not candidates:
        raise SystemExit("Not enough vocabulary tokens for auxiliary recall key/value pools.")
    return np.asarray(candidates[: min(key_count, len(candidates))], dtype=np.int32)


def init_aux_recall_memory_params(
    params: dict[str, jax.Array],
    *,
    task: str,
    key_ids: np.ndarray,
    value_ids: np.ndarray,
    assign_id: int,
    query_id: int,
    write_logit: float,
    read_logit: float,
    closed_logit: float,
    memory_logit_scale: float,
) -> dict[str, jax.Array]:
    params = dict(params)
    if "memory_write_token" in params and "memory_read_token" in params:
        write = jnp.full_like(params["memory_write_token"], closed_logit)
        read = jnp.full_like(params["memory_read_token"], closed_logit)
        write = write.at[jnp.asarray(key_ids)].set(write_logit)
        read = read.at[query_id].set(read_logit)
        params["memory_write_token"] = write
        params["memory_read_token"] = read
        if "memory_logit_scale" in params:
            params["memory_logit_scale"] = jnp.asarray(memory_logit_scale, dtype=jnp.float32)
    if "kv_key_token" in params and "kv_value_token" in params and "kv_query_token" in params:
        key = jnp.full_like(params["kv_key_token"], closed_logit)
        value = jnp.full_like(params["kv_value_token"], closed_logit)
        erase = jnp.full_like(params["kv_erase_token"], closed_logit) if "kv_erase_token" in params else None
        write_context = (
            jnp.full_like(params["kv_write_context_token"], closed_logit)
            if "kv_write_context_token" in params
            else None
        )
        deref = jnp.full_like(params["kv_deref_token"], closed_logit) if "kv_deref_token" in params else None
        query = jnp.full_like(params["kv_query_token"], closed_logit)
        key = key.at[jnp.asarray(key_ids)].set(write_logit)
        query = query.at[query_id].set(read_logit)
        if task == "assignment":
            value = value.at[jnp.asarray(value_ids)].set(write_logit)
            if erase is not None:
                erase = erase.at[jnp.asarray(value_ids)].set(write_logit)
            if write_context is not None:
                write_context = write_context.at[assign_id].set(write_logit)
            if deref is not None:
                deref = deref.at[jnp.asarray(key_ids)].set(write_logit)
        else:
            value = value.at[jnp.asarray(key_ids)].set(write_logit)
            if erase is not None:
                erase = erase.at[jnp.asarray(key_ids)].set(write_logit)
        params["kv_key_token"] = key
        params["kv_value_token"] = value
        if erase is not None:
            params["kv_erase_token"] = erase
        if write_context is not None:
            params["kv_write_context_token"] = write_context
        if deref is not None:
            params["kv_deref_token"] = deref
        params["kv_query_token"] = query
        if "kv_logit_scale" in params:
            params["kv_logit_scale"] = jnp.asarray(memory_logit_scale, dtype=jnp.float32)
    return params


def make_aux_recall_batch(
    key_ids: np.ndarray,
    *,
    task: str,
    streams: int,
    seq_len: int,
    delay: int,
    seed: int,
    update_index: int,
    value_ids: np.ndarray | None = None,
    assign_id: int = 0,
    separator_id: int = 0,
    query_id: int = 1,
    filler_id: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    if delay <= 0:
        raise SystemExit("--ssm-aux-recall-delay must be positive when auxiliary recall is enabled.")
    effective_delay = min(delay, max(1, seq_len - 1))
    sequence = np.full((streams, seq_len + 1), filler_id, dtype=np.int32)
    mask = np.zeros((streams, seq_len), dtype=np.float32)
    rng = np.random.default_rng(seed + 1_000_003 * update_index)
    keys = rng.choice(key_ids, size=streams, replace=True)
    values = rng.choice(value_ids if value_ids is not None and len(value_ids) else key_ids, size=streams, replace=True)
    recall_events = 0
    if task == "delayed-key":
        key_pos = 0
        while key_pos + effective_delay + 1 <= seq_len:
            query_pos = key_pos + effective_delay
            sequence[:, key_pos] = keys
            sequence[:, query_pos] = query_id
            sequence[:, query_pos + 1] = keys
            mask[:, query_pos] = 1.0
            recall_events += 1
            key_pos = query_pos + 1
    elif task == "assignment":
        if effective_delay < 4:
            raise SystemExit("--ssm-aux-recall-delay must be at least 4 for assignment auxiliary recall.")
        sequence[:, 0] = keys
        sequence[:, 1] = assign_id
        sequence[:, 2] = values
        sequence[:, 3] = separator_id
        query_name_pos = effective_delay - 1
        query_pos = effective_delay
        if query_pos + 1 > seq_len:
            raise SystemExit("--seq-len is too small for the requested assignment auxiliary recall delay.")
        sequence[:, query_name_pos] = keys
        sequence[:, query_pos] = query_id
        sequence[:, query_pos + 1] = values
        mask[:, query_pos] = 1.0
        recall_events += 1
    else:
        raise SystemExit(f"Unknown --ssm-aux-recall-task: {task!r}")
    if recall_events == 0:
        raise SystemExit("--seq-len is too small for the requested auxiliary recall delay.")
    return jnp.asarray(sequence[:, :-1]), jnp.asarray(sequence[:, 1:]), jnp.asarray(mask), recall_events


def train_ssm(
    params: dict[str, jax.Array],
    train_streams: jax.Array,
    args: argparse.Namespace,
) -> tuple[dict[str, jax.Array], dict[str, object]]:
    opt_state = init_adam(params)
    chunks = (args.tokens_per_stream - 1) // args.seq_len
    aux_enabled = args.ssm_aux_recall_every > 0
    aux_key_ids = (
        select_aux_recall_key_ids(
            train_streams,
            vocab_size=params["token_embed"].shape[0],
            key_count=args.ssm_aux_recall_key_count,
            exclude_ids={
                args.ssm_aux_recall_query_id,
                args.ssm_aux_recall_filler_id,
                args.ssm_aux_recall_assign_id,
                args.ssm_aux_recall_separator_id,
            },
        )
        if aux_enabled
        else np.asarray([], dtype=np.int32)
    )
    aux_value_ids = (
        select_aux_recall_key_ids(
            train_streams,
            vocab_size=params["token_embed"].shape[0],
            key_count=args.ssm_aux_recall_key_count,
            skip_count=len(aux_key_ids),
            exclude_ids={
                args.ssm_aux_recall_query_id,
                args.ssm_aux_recall_filler_id,
                args.ssm_aux_recall_assign_id,
                args.ssm_aux_recall_separator_id,
            },
        )
        if aux_enabled and args.ssm_aux_recall_task == "assignment"
        else aux_key_ids
    )

    warm_states = jnp.zeros((args.streams, args.ssm_state_dim), dtype=jnp.float32)
    warm_input = train_streams[:, : args.seq_len]
    warm_target = train_streams[:, 1 : args.seq_len + 1]
    warm_params, warm_opt, warm_states, warm_metrics = ssm_train_step(
        params,
        opt_state,
        warm_input,
        warm_target,
        warm_states,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        memory_gate_l1=args.memory_gate_l1,
    )
    jax.block_until_ready(warm_metrics.loss)
    del warm_params, warm_opt, warm_states

    states_h = jnp.zeros((args.streams, args.ssm_state_dim), dtype=jnp.float32)
    start = time.perf_counter()
    update_count = 0
    aux_update_count = 0
    aux_recall_events = 0
    aux_metrics: Any = None
    metrics = warm_metrics
    for _ in range(args.epochs):
        states_h = jnp.zeros_like(states_h)
        for chunk in range(chunks):
            cursor = chunk * args.seq_len
            input_ids = train_streams[:, cursor : cursor + args.seq_len]
            target_ids = train_streams[:, cursor + 1 : cursor + args.seq_len + 1]
            params, opt_state, states_h, metrics = ssm_train_step(
                params,
                opt_state,
                input_ids,
                target_ids,
                states_h,
                learning_rate=args.learning_rate,
                grad_clip=args.grad_clip,
                memory_gate_l1=args.memory_gate_l1,
            )
            update_count += 1
            if aux_enabled and update_count % args.ssm_aux_recall_every == 0:
                aux_input, aux_target, aux_mask, recall_events = make_aux_recall_batch(
                    aux_key_ids,
                    task=args.ssm_aux_recall_task,
                    streams=args.streams,
                    seq_len=args.seq_len,
                    delay=args.ssm_aux_recall_delay,
                    seed=args.seed,
                    update_index=aux_update_count,
                    value_ids=aux_value_ids,
                    assign_id=args.ssm_aux_recall_assign_id,
                    separator_id=args.ssm_aux_recall_separator_id,
                    query_id=args.ssm_aux_recall_query_id,
                    filler_id=args.ssm_aux_recall_filler_id,
                )
                aux_states = jnp.zeros_like(states_h)
                params, opt_state, _, aux_metrics = ssm_masked_train_step(
                    params,
                    opt_state,
                    aux_input,
                    aux_target,
                    aux_mask,
                    aux_states,
                    learning_rate=args.learning_rate * args.ssm_aux_recall_lr_scale,
                    grad_clip=args.grad_clip,
                    memory_gate_l1=args.memory_gate_l1,
                )
                aux_update_count += 1
                aux_recall_events += recall_events
    jax.block_until_ready(metrics.loss)
    if aux_metrics is not None:
        jax.block_until_ready(aux_metrics.loss)
    elapsed = time.perf_counter() - start
    trained_tokens = update_count * args.streams * args.seq_len
    aux_trained_tokens = aux_update_count * args.streams * args.seq_len
    effective_aux_delay = min(args.ssm_aux_recall_delay, max(1, args.seq_len - 1)) if aux_enabled else None
    return params, {
        "updates": update_count,
        "trained_tokens": trained_tokens,
        "train_s": elapsed,
        "train_throughput_tokens_s": trained_tokens / max(elapsed, 1.0e-9),
        "final_train_loss": float(metrics.loss),
        "final_train_accuracy": float(metrics.accuracy),
        "mean_decay_A": float(metrics.mean_decay),
        "aux_recall": {
            "enabled": aux_enabled,
            "updates": aux_update_count,
            "trained_tokens": aux_trained_tokens,
            "recall_events": aux_recall_events,
            "effective_delay": effective_aux_delay,
            "key_count": int(len(aux_key_ids)),
            "value_count": int(len(aux_value_ids)),
            "learning_rate_scale": args.ssm_aux_recall_lr_scale,
            "final_loss": float(aux_metrics.loss) if aux_metrics is not None else None,
            "final_accuracy": float(aux_metrics.accuracy) if aux_metrics is not None else None,
        },
    }


def estimate_transformer_param_count(
    *,
    vocab_size: int,
    seq_len: int,
    d_model: int,
    layers: int,
    heads: int,
    ff_mult: int,
) -> int:
    if d_model % heads != 0:
        return 10**18
    ff_dim = d_model * ff_mult
    total = vocab_size * d_model
    total += seq_len * d_model
    for _ in range(layers):
        total += 2 * d_model
        total += d_model * (3 * d_model) + (3 * d_model)
        total += d_model * d_model + d_model
        total += 2 * d_model
        total += d_model * ff_dim + ff_dim
        total += ff_dim * d_model + d_model
    total += 2 * d_model
    total += d_model * vocab_size + vocab_size
    return int(total)


def choose_transformer_d_model(
    *,
    target_params: int,
    vocab_size: int,
    seq_len: int,
    layers: int,
    heads: int,
    ff_mult: int,
    max_d_model: int,
) -> int:
    candidates = [d for d in range(heads, max_d_model + 1) if d % heads == 0]
    if not candidates:
        raise SystemExit("--transformer-max-d-model must be at least --transformer-heads.")
    return min(
        candidates,
        key=lambda d: abs(
            math.log(
                estimate_transformer_param_count(
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    d_model=d,
                    layers=layers,
                    heads=heads,
                    ff_mult=ff_mult,
                )
                / max(1, target_params)
            )
        ),
    )


def init_transformer_params(
    key: jax.Array,
    *,
    vocab_size: int,
    seq_len: int,
    d_model: int,
    layers: int,
    ff_mult: int,
) -> dict[str, jax.Array]:
    params: dict[str, jax.Array] = {}
    keys = iter(jax.random.split(key, 4 + 8 * layers))
    params["token_embed"] = 0.04 * jax.random.normal(next(keys), (vocab_size, d_model), dtype=jnp.float32)
    params["pos_embed"] = 0.02 * jax.random.normal(next(keys), (seq_len, d_model), dtype=jnp.float32)
    for layer in range(layers):
        prefix = f"layer_{layer}"
        ff_dim = d_model * ff_mult
        params[f"{prefix}_ln1_scale"] = jnp.ones((d_model,), dtype=jnp.float32)
        params[f"{prefix}_ln1_bias"] = jnp.zeros((d_model,), dtype=jnp.float32)
        params[f"{prefix}_qkv_w"] = jax.random.normal(next(keys), (d_model, 3 * d_model), dtype=jnp.float32)
        params[f"{prefix}_qkv_w"] *= jnp.sqrt(jnp.asarray(1.0 / d_model, dtype=jnp.float32))
        params[f"{prefix}_qkv_b"] = jnp.zeros((3 * d_model,), dtype=jnp.float32)
        params[f"{prefix}_out_w"] = jax.random.normal(next(keys), (d_model, d_model), dtype=jnp.float32)
        params[f"{prefix}_out_w"] *= jnp.sqrt(jnp.asarray(1.0 / d_model, dtype=jnp.float32))
        params[f"{prefix}_out_b"] = jnp.zeros((d_model,), dtype=jnp.float32)
        params[f"{prefix}_ln2_scale"] = jnp.ones((d_model,), dtype=jnp.float32)
        params[f"{prefix}_ln2_bias"] = jnp.zeros((d_model,), dtype=jnp.float32)
        params[f"{prefix}_ff1_w"] = jax.random.normal(next(keys), (d_model, ff_dim), dtype=jnp.float32)
        params[f"{prefix}_ff1_w"] *= jnp.sqrt(jnp.asarray(1.0 / d_model, dtype=jnp.float32))
        params[f"{prefix}_ff1_b"] = jnp.zeros((ff_dim,), dtype=jnp.float32)
        params[f"{prefix}_ff2_w"] = jax.random.normal(next(keys), (ff_dim, d_model), dtype=jnp.float32)
        params[f"{prefix}_ff2_w"] *= jnp.sqrt(jnp.asarray(1.0 / ff_dim, dtype=jnp.float32))
        params[f"{prefix}_ff2_b"] = jnp.zeros((d_model,), dtype=jnp.float32)
    params["final_ln_scale"] = jnp.ones((d_model,), dtype=jnp.float32)
    params["final_ln_bias"] = jnp.zeros((d_model,), dtype=jnp.float32)
    params["lm_head_w"] = jax.random.normal(next(keys), (d_model, vocab_size), dtype=jnp.float32)
    params["lm_head_w"] *= jnp.sqrt(jnp.asarray(1.0 / d_model, dtype=jnp.float32))
    params["lm_head_b"] = jnp.zeros((vocab_size,), dtype=jnp.float32)
    return params


def layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + 1.0e-5) * scale + bias


def transformer_forward(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    *,
    layers: int,
    heads: int,
) -> jax.Array:
    batch, seq_len = input_ids.shape
    d_model = params["token_embed"].shape[1]
    head_dim = d_model // heads
    x = params["token_embed"][input_ids] + params["pos_embed"][:seq_len][None, :, :]
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None, None, :, :]

    for layer in range(layers):
        prefix = f"layer_{layer}"
        y = layer_norm(x, params[f"{prefix}_ln1_scale"], params[f"{prefix}_ln1_bias"])
        qkv = y @ params[f"{prefix}_qkv_w"] + params[f"{prefix}_qkv_b"]
        qkv = qkv.reshape(batch, seq_len, 3, heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
        scores = jnp.where(causal_mask, scores, jnp.asarray(-1.0e9, dtype=jnp.float32))
        attn = jax.nn.softmax(scores, axis=-1)
        attn_out = attn @ v
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(batch, seq_len, d_model)
        x = x + attn_out @ params[f"{prefix}_out_w"] + params[f"{prefix}_out_b"]

        y = layer_norm(x, params[f"{prefix}_ln2_scale"], params[f"{prefix}_ln2_bias"])
        ff = jax.nn.gelu(y @ params[f"{prefix}_ff1_w"] + params[f"{prefix}_ff1_b"])
        x = x + ff @ params[f"{prefix}_ff2_w"] + params[f"{prefix}_ff2_b"]

    x = layer_norm(x, params["final_ln_scale"], params["final_ln_bias"])
    return x @ params["lm_head_w"] + params["lm_head_b"]


def transformer_loss_fn(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    *,
    layers: int,
    heads: int,
) -> tuple[jax.Array, jax.Array]:
    logits = transformer_forward(params, input_ids, layers=layers, heads=heads)
    return nll_and_accuracy(logits, target_ids)


@partial(jax.jit, static_argnames=("learning_rate", "grad_clip", "layers", "heads"))
def transformer_train_step(
    params: dict[str, jax.Array],
    opt_state: dict[str, object],
    input_ids: jax.Array,
    target_ids: jax.Array,
    *,
    learning_rate: float,
    grad_clip: float,
    layers: int,
    heads: int,
) -> tuple[dict[str, jax.Array], dict[str, object], TransformerTrainMetrics]:
    (loss, accuracy), grads = jax.value_and_grad(transformer_loss_fn, has_aux=True)(
        params,
        input_ids,
        target_ids,
        layers=layers,
        heads=heads,
    )
    grads, grad_norm = clip_grads(grads, grad_clip)
    params, opt_state = adam_update(params, grads, opt_state, learning_rate)
    return params, opt_state, TransformerTrainMetrics(loss=loss, accuracy=accuracy, grad_norm=grad_norm)


@partial(jax.jit, static_argnames=("seq_len", "layers", "heads"))
def transformer_eval_full(
    params: dict[str, jax.Array],
    streams: jax.Array,
    *,
    seq_len: int,
    layers: int,
    heads: int,
) -> EvalMetrics:
    chunks = (streams.shape[1] - 1) // seq_len
    usable = chunks * seq_len
    input_chunks = streams[:, :usable].reshape(streams.shape[0], chunks, seq_len)
    target_chunks = streams[:, 1 : usable + 1].reshape(streams.shape[0], chunks, seq_len)
    input_chunks = jnp.swapaxes(input_chunks, 0, 1)
    target_chunks = jnp.swapaxes(target_chunks, 0, 1)

    def step(_: None, pair: tuple[jax.Array, jax.Array]) -> tuple[None, tuple[jax.Array, jax.Array]]:
        input_ids, target_ids = pair
        logits = transformer_forward(params, input_ids, layers=layers, heads=heads)
        loss, accuracy = nll_and_accuracy(logits, target_ids)
        return None, (loss, accuracy)

    _, (losses, accuracies) = jax.lax.scan(step, None, (input_chunks, target_chunks))
    return EvalMetrics(loss=jnp.mean(losses), accuracy=jnp.mean(accuracies))


def train_transformer(
    params: dict[str, jax.Array],
    train_streams: jax.Array,
    args: argparse.Namespace,
) -> tuple[dict[str, jax.Array], dict[str, object]]:
    opt_state = init_adam(params)
    chunks = (args.tokens_per_stream - 1) // args.seq_len

    warm_input = train_streams[:, : args.seq_len]
    warm_target = train_streams[:, 1 : args.seq_len + 1]
    warm_params, warm_opt, warm_metrics = transformer_train_step(
        params,
        opt_state,
        warm_input,
        warm_target,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        layers=args.transformer_layers,
        heads=args.transformer_heads,
    )
    jax.block_until_ready(warm_metrics.loss)
    del warm_params, warm_opt

    start = time.perf_counter()
    update_count = 0
    metrics = warm_metrics
    for _ in range(args.epochs):
        for chunk in range(chunks):
            cursor = chunk * args.seq_len
            input_ids = train_streams[:, cursor : cursor + args.seq_len]
            target_ids = train_streams[:, cursor + 1 : cursor + args.seq_len + 1]
            params, opt_state, metrics = transformer_train_step(
                params,
                opt_state,
                input_ids,
                target_ids,
                learning_rate=args.learning_rate,
                grad_clip=args.grad_clip,
                layers=args.transformer_layers,
                heads=args.transformer_heads,
            )
            update_count += 1
    jax.block_until_ready(metrics.loss)
    elapsed = time.perf_counter() - start
    trained_tokens = update_count * args.streams * args.seq_len
    return params, {
        "updates": update_count,
        "trained_tokens": trained_tokens,
        "train_s": elapsed,
        "train_throughput_tokens_s": trained_tokens / max(elapsed, 1.0e-9),
        "final_train_loss": float(metrics.loss),
        "final_train_accuracy": float(metrics.accuracy),
    }


def timed_eval_ssm(
    params: dict[str, jax.Array],
    streams: jax.Array,
    args: argparse.Namespace,
) -> dict[str, object]:
    warmup = ssm_eval_full(params, streams, seq_len=args.seq_len)
    jax.block_until_ready(warmup.loss)
    times: list[float] = []
    metrics = warmup
    for _ in range(args.repeat):
        start = time.perf_counter()
        metrics = ssm_eval_full(params, streams, seq_len=args.seq_len)
        jax.block_until_ready(metrics.loss)
        times.append(time.perf_counter() - start)
    evaluated_tokens = args.streams * args.seq_len * ((args.tokens_per_stream - 1) // args.seq_len)
    best_s = min(times)
    return {
        "loss": float(metrics.loss),
        "perplexity": float(jnp.exp(metrics.loss)),
        "accuracy": float(metrics.accuracy),
        "best_eval_s": best_s,
        "mean_eval_s": float(np.mean(times)),
        "throughput_tokens_s": evaluated_tokens / max(best_s, 1.0e-9),
        "evaluated_tokens_per_repeat": evaluated_tokens,
    }


def timed_eval_transformer(
    params: dict[str, jax.Array],
    streams: jax.Array,
    args: argparse.Namespace,
) -> dict[str, object]:
    warmup = transformer_eval_full(
        params,
        streams,
        seq_len=args.seq_len,
        layers=args.transformer_layers,
        heads=args.transformer_heads,
    )
    jax.block_until_ready(warmup.loss)
    times: list[float] = []
    metrics = warmup
    for _ in range(args.repeat):
        start = time.perf_counter()
        metrics = transformer_eval_full(
            params,
            streams,
            seq_len=args.seq_len,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
        )
        jax.block_until_ready(metrics.loss)
        times.append(time.perf_counter() - start)
    evaluated_tokens = args.streams * args.seq_len * ((args.tokens_per_stream - 1) // args.seq_len)
    best_s = min(times)
    return {
        "loss": float(metrics.loss),
        "perplexity": float(jnp.exp(metrics.loss)),
        "accuracy": float(metrics.accuracy),
        "best_eval_s": best_s,
        "mean_eval_s": float(np.mean(times)),
        "throughput_tokens_s": evaluated_tokens / max(best_s, 1.0e-9),
        "evaluated_tokens_per_repeat": evaluated_tokens,
    }


def sample_transformer_text(
    params: dict[str, jax.Array],
    token_to_id: dict[str, int],
    vocab: list[str],
    prompt: str,
    *,
    steps: int,
    temperature: float,
    seed: int,
    tokenizer_config: dict[str, Any],
    layers: int,
    heads: int,
    seq_len: int,
) -> str:
    key = jax.random.PRNGKey(seed)
    prompt_tokens = tokenize_with_config(prompt, tokenizer_config)
    unk_id = token_to_id[UNK]
    ids = [token_to_id.get(token, unk_id) for token in prompt_tokens] or [token_to_id[BOS]]
    generated = list(prompt_tokens)
    for _ in range(steps):
        window = ids[-seq_len:]
        input_ids = jnp.asarray([window], dtype=jnp.int32)
        logits = transformer_forward(params, input_ids, layers=layers, heads=heads)[0, -1]
        logits = logits / max(temperature, 1.0e-6)
        if temperature <= 0.0:
            next_id = int(jnp.argmax(logits))
        else:
            key, sample_key = jax.random.split(key)
            next_id = int(jax.random.categorical(sample_key, logits))
        ids.append(next_id)
        generated.append(vocab[next_id])
    return detokenize_with_config(generated, tokenizer_config)


def compare_results(ssm: dict[str, object], transformer: dict[str, object]) -> dict[str, object]:
    loss_delta = float(ssm["loss"]) - float(transformer["loss"])
    accuracy_delta = float(ssm["accuracy"]) - float(transformer["accuracy"])
    throughput_delta = float(ssm["throughput_tokens_s"]) - float(transformer["throughput_tokens_s"])
    if float(ssm["loss"]) < float(transformer["loss"]):
        winner = "ssm"
    elif float(ssm["loss"]) > float(transformer["loss"]):
        winner = "transformer"
    elif float(ssm["accuracy"]) > float(transformer["accuracy"]):
        winner = "ssm"
    elif float(ssm["accuracy"]) < float(transformer["accuracy"]):
        winner = "transformer"
    else:
        winner = "tie"
    return {
        "winner_by_loss": winner,
        "ssm_minus_transformer_loss": loss_delta,
        "ssm_minus_transformer_accuracy": accuracy_delta,
        "ssm_minus_transformer_throughput_tokens_s": throughput_delta,
        "ssm_loss_regression_rate_vs_transformer": loss_delta / max(1.0e-9, float(transformer["loss"])),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    data = prepare_data(args)
    vocab = data["vocab"]
    token_to_id = data["token_to_id"]
    tokenizer_config = data["tokenizer_config"]
    train_streams = data["train_streams"]
    eval_streams = data["eval_streams"]

    key = jax.random.PRNGKey(args.seed)
    ssm_key, transformer_key = jax.random.split(key)
    ssm_params = init_ssm_params(
        ssm_key,
        len(vocab),
        args.ssm_input_dim,
        args.ssm_state_dim,
        variant=args.ssm_variant,
        skip_rank=args.ssm_skip_rank,
    )
    if args.ssm_aux_recall_every > 0 and args.ssm_aux_recall_init_gates:
        aux_init_key_ids = select_aux_recall_key_ids(
            train_streams,
            vocab_size=len(vocab),
            key_count=args.ssm_aux_recall_key_count,
            exclude_ids={
                args.ssm_aux_recall_query_id,
                args.ssm_aux_recall_filler_id,
                args.ssm_aux_recall_assign_id,
                args.ssm_aux_recall_separator_id,
            },
        )
        aux_init_value_ids = (
            select_aux_recall_key_ids(
                train_streams,
                vocab_size=len(vocab),
                key_count=args.ssm_aux_recall_key_count,
                skip_count=len(aux_init_key_ids),
                exclude_ids={
                    args.ssm_aux_recall_query_id,
                    args.ssm_aux_recall_filler_id,
                    args.ssm_aux_recall_assign_id,
                    args.ssm_aux_recall_separator_id,
                },
            )
            if args.ssm_aux_recall_task == "assignment"
            else aux_init_key_ids
        )
        ssm_params = init_aux_recall_memory_params(
            ssm_params,
            task=args.ssm_aux_recall_task,
            key_ids=aux_init_key_ids,
            value_ids=aux_init_value_ids,
            assign_id=args.ssm_aux_recall_assign_id,
            query_id=args.ssm_aux_recall_query_id,
            write_logit=args.ssm_aux_recall_write_logit,
            read_logit=args.ssm_aux_recall_read_logit,
            closed_logit=args.ssm_aux_recall_closed_logit,
            memory_logit_scale=args.ssm_aux_recall_logit_scale,
        )
    ssm_param_count = count_params(ssm_params)

    if args.transformer_d_model is None:
        transformer_d_model = choose_transformer_d_model(
            target_params=ssm_param_count,
            vocab_size=len(vocab),
            seq_len=args.seq_len,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            ff_mult=args.transformer_ff_mult,
            max_d_model=args.transformer_max_d_model,
        )
    else:
        transformer_d_model = args.transformer_d_model
    if transformer_d_model % args.transformer_heads != 0:
        raise SystemExit("--transformer-d-model must be divisible by --transformer-heads.")

    transformer_params = init_transformer_params(
        transformer_key,
        vocab_size=len(vocab),
        seq_len=args.seq_len,
        d_model=transformer_d_model,
        layers=args.transformer_layers,
        ff_mult=args.transformer_ff_mult,
    )
    transformer_param_count = count_params(transformer_params)
    budget_ratio = transformer_param_count / max(1, ssm_param_count)
    budget_match = 0.8 <= budget_ratio <= 1.25

    print("TextPy/SoA architecture comparison")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("Protocol")
    print(f"  source: {data['split']['source_text_file']}")
    print(f"  train_tokens: {len(data['train_tokens']):,}")
    print(f"  eval_tokens: {len(data['eval_tokens']):,}")
    print(f"  vocab_size: {len(vocab):,}")
    print(f"  tokenizer: {tokenizer_config.get('type', 'word')}")
    print(f"  streams: {args.streams:,}")
    print(f"  tokens_per_stream: {args.tokens_per_stream:,}")
    print(f"  seq_len: {args.seq_len:,}")
    print(f"  epochs: {args.epochs:,}")
    if args.ssm_aux_recall_every > 0:
        print(
            "  ssm_aux_recall: "
            f"every={args.ssm_aux_recall_every} "
            f"delay={args.ssm_aux_recall_delay} "
            f"keys={args.ssm_aux_recall_key_count} "
            f"task={args.ssm_aux_recall_task} "
            f"lr_scale={args.ssm_aux_recall_lr_scale:g} "
            f"init_gates={args.ssm_aux_recall_init_gates}"
        )
    print("")
    print("Parameter budget")
    print(f"  ssm_variant: {infer_ssm_variant(ssm_params)}")
    print(f"  ssm_params: {ssm_param_count:,}")
    print(f"  transformer_params: {transformer_param_count:,}")
    print(f"  transformer/ssm: {budget_ratio:.3f}")
    print(f"  budget_match_80_125pct: {budget_match}")
    if not budget_match:
        print("  warning: parameter budgets are not closely matched; treat metrics as diagnostic only.")
    print("")

    print("Training SSM")
    ssm_params, ssm_train = train_ssm(ssm_params, train_streams, args)
    print(f"  final_train_loss: {ssm_train['final_train_loss']:.4f}")
    print(f"  train_throughput_tokens_s: {ssm_train['train_throughput_tokens_s']:,.0f}")
    if ssm_train["aux_recall"]["enabled"]:
        aux_train = ssm_train["aux_recall"]
        if aux_train["updates"] > 0:
            print(
                "  aux_recall: "
                f"updates={aux_train['updates']:,} "
                f"final_loss={aux_train['final_loss']:.4f} "
                f"final_accuracy={aux_train['final_accuracy']:.4f}"
            )
        else:
            print("  aux_recall: enabled but no updates ran")

    print("Training Transformer")
    transformer_params, transformer_train = train_transformer(transformer_params, train_streams, args)
    print(f"  d_model: {transformer_d_model}")
    print(f"  final_train_loss: {transformer_train['final_train_loss']:.4f}")
    print(f"  train_throughput_tokens_s: {transformer_train['train_throughput_tokens_s']:,.0f}")
    print("")

    ssm_eval = timed_eval_ssm(ssm_params, eval_streams, args)
    transformer_eval = timed_eval_transformer(transformer_params, eval_streams, args)

    ssm_runtime_bytes = args.streams * args.ssm_state_dim * 4
    transformer_attention_bytes = (
        args.transformer_layers * args.streams * args.transformer_heads * args.seq_len * args.seq_len * 4
    )
    ssm_arch = {
        "name": "ssm",
        "variant": infer_ssm_variant(ssm_params),
        "input_dim": args.ssm_input_dim,
        "state_dim": args.ssm_state_dim,
        "parameter_count": ssm_param_count,
        "parameter_bytes": count_bytes(ssm_params),
        "optimizer_bytes_estimate": 2 * count_bytes(ssm_params),
        "state_memory_bytes": ssm_runtime_bytes,
        "effective_context_tokens": args.tokens_per_stream,
        "diagnostics": ssm_parameter_diagnostics(ssm_params),
        "train": ssm_train,
        "eval": ssm_eval,
        "sample": sample_ssm_text(
            ssm_params,
            token_to_id,
            vocab,
            args.prompt,
            args.sample_steps,
            args.temperature,
            args.seed + 1,
            tokenizer_config=tokenizer_config,
        ),
    }
    transformer_arch = {
        "name": "transformer",
        "d_model": transformer_d_model,
        "layers": args.transformer_layers,
        "heads": args.transformer_heads,
        "ff_mult": args.transformer_ff_mult,
        "parameter_count": transformer_param_count,
        "parameter_bytes": count_bytes(transformer_params),
        "optimizer_bytes_estimate": 2 * count_bytes(transformer_params),
        "attention_score_memory_bytes": transformer_attention_bytes,
        "effective_context_tokens": args.seq_len,
        "train": transformer_train,
        "eval": transformer_eval,
        "sample": sample_transformer_text(
            transformer_params,
            token_to_id,
            vocab,
            args.prompt,
            steps=args.sample_steps,
            temperature=args.temperature,
            seed=args.seed + 2,
            tokenizer_config=tokenizer_config,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            seq_len=args.seq_len,
        ),
    }
    comparison = compare_results(ssm_eval, transformer_eval)

    print("Held-out eval")
    print(
        "  ssm: "
        f"loss={ssm_eval['loss']:.4f} "
        f"ppl={ssm_eval['perplexity']:.2f} "
        f"acc={ssm_eval['accuracy']:.4f} "
        f"throughput={ssm_eval['throughput_tokens_s']:,.0f} tok/s"
    )
    print(
        "  transformer: "
        f"loss={transformer_eval['loss']:.4f} "
        f"ppl={transformer_eval['perplexity']:.2f} "
        f"acc={transformer_eval['accuracy']:.4f} "
        f"throughput={transformer_eval['throughput_tokens_s']:,.0f} tok/s"
    )
    print("")
    print("Memory/context")
    print(f"  ssm_parameter_bytes: {format_bytes(ssm_arch['parameter_bytes'])}")
    print(f"  transformer_parameter_bytes: {format_bytes(transformer_arch['parameter_bytes'])}")
    print(f"  ssm_state_memory_bytes: {format_bytes(ssm_runtime_bytes)}")
    print(f"  transformer_attention_score_bytes: {format_bytes(transformer_attention_bytes)}")
    print(f"  ssm_effective_context_tokens: {ssm_arch['effective_context_tokens']:,}")
    print(f"  transformer_effective_context_tokens: {transformer_arch['effective_context_tokens']:,}")
    print("")
    print("Comparison")
    print(f"  winner_by_loss: {comparison['winner_by_loss']}")
    print(f"  ssm_minus_transformer_loss: {comparison['ssm_minus_transformer_loss']:.6f}")
    print(f"  ssm_minus_transformer_accuracy: {comparison['ssm_minus_transformer_accuracy']:.6f}")
    print(
        "  ssm_minus_transformer_throughput_tokens_s: "
        f"{comparison['ssm_minus_transformer_throughput_tokens_s']:,.0f}"
    )
    print("")
    print("Interpretation")
    print("  This is a controlled benchmark report, not proof that either architecture is generally better.")
    print("  Increase --text-file size, --epochs, and sweep both architectures before making research claims.")

    output_json = Path(args.output_json) if args.output_json else Path("artifacts/text_architecture_comparison") / f"{timestamp()}.json"
    payload = {
        "created_at": timestamp(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "protocol": {
            "text_file": args.text_file,
            "split": data["split"],
            "tokenizer": tokenizer_config.get("type", "word"),
            "tokenizer_config": tokenizer_config,
            "tokenizer_summary": data["tokenizer_summary"],
            "train_tokens": len(data["train_tokens"]),
            "eval_tokens": len(data["eval_tokens"]),
            "vocab_size": len(vocab),
            "eval_oov": data["oov"],
            "streams": args.streams,
            "tokens_per_stream": args.tokens_per_stream,
            "seq_len": args.seq_len,
            "ssm_variant": infer_ssm_variant(ssm_params),
            "ssm_skip_rank": args.ssm_skip_rank,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "memory_gate_l1": args.memory_gate_l1,
            "ssm_aux_recall": {
                "enabled": args.ssm_aux_recall_every > 0,
                "task": args.ssm_aux_recall_task,
                "every": args.ssm_aux_recall_every,
                "delay": args.ssm_aux_recall_delay,
                "key_count": args.ssm_aux_recall_key_count,
                "learning_rate_scale": args.ssm_aux_recall_lr_scale,
                "init_gates": args.ssm_aux_recall_init_gates,
                "write_logit": args.ssm_aux_recall_write_logit,
                "read_logit": args.ssm_aux_recall_read_logit,
                "closed_logit": args.ssm_aux_recall_closed_logit,
                "memory_logit_scale": args.ssm_aux_recall_logit_scale,
                "query_id": args.ssm_aux_recall_query_id,
                "filler_id": args.ssm_aux_recall_filler_id,
                "assign_id": args.ssm_aux_recall_assign_id,
                "separator_id": args.ssm_aux_recall_separator_id,
            },
            "repeat": args.repeat,
            "budget_match_80_125pct": budget_match,
            "transformer_to_ssm_param_ratio": budget_ratio,
        },
        "ssm": ssm_arch,
        "transformer": transformer_arch,
        "comparison": comparison,
        "claim_guardrail": (
            "A smoke or tiny-corpus run is evidence that the harness works. "
            "It is not evidence that the SSM is better than Transformers."
        ),
    }
    write_json(output_json, payload)
    print("")
    print(f"Saved JSON: {output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare text SSM and tiny Transformer under one protocol.")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--tokenizer", choices=("word", "bpe"), default="word")
    parser.add_argument("--tokenizer-config", default=None)
    parser.add_argument("--bpe-vocab-size", type=int, default=512)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--bpe-train-chars", type=int, default=250_000)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-tokens", type=int, default=64)
    parser.add_argument("--min-eval-tokens", type=int, default=16)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--tokens-per-stream", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--ssm-input-dim", type=int, default=48)
    parser.add_argument("--ssm-state-dim", type=int, default=64)
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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--memory-gate-l1", type=float, default=0.0)
    parser.add_argument(
        "--ssm-aux-recall-every",
        type=int,
        default=0,
        help="Run one SSM-only delayed-recall curriculum update every N LM updates; 0 disables it.",
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
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default="memory is")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
