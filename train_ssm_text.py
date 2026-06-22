#!/usr/bin/env python3
"""Train a tiny word-level SSM language model with JAX autodiff.

This is the second prototype layer: the SoA runtime now learns its rules. Text
tokens become TENSORS_X through an embedding table, STATES_H carries compressed
memory, and A/B/C are updated by gradient descent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
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
    build_vocab as build_token_vocab,
    default_word_tokenizer_config,
    detokenize_with_config,
    encode_tokens,
    tokenize_with_config,
    tokenizer_summary,
)

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
    "sparse-kv-memory",
    "dynamic-kv-memory",
)

BASE_PARAM_NAMES = ("decay_raw", "hidden_bias", "output_bias", "token_embed", "input_b", "output_c")
SELECTIVE_PARAM_NAMES = (
    "select_decay_w",
    "select_write_w",
    "select_write_b",
    "select_write_scalar_w",
    "select_write_scalar_b",
    "write_scalar_w",
    "write_scalar_b",
    "skip_d",
    "prev_skip_d",
    "tied_skip_scale",
    "skip_u",
    "skip_v",
    "state_mix_w",
    "memory_write_token",
    "memory_read_token",
    "memory_decay_raw",
    "memory_logit_scale",
    "memory_closed_marker",
    "memory_sparse_marker",
    "kv_key_token",
    "kv_value_token",
    "kv_erase_token",
    "kv_write_context_token",
    "kv_deref_token",
    "kv_query_token",
    "kv_decay_raw",
    "kv_logit_scale",
    "kv_dynamic_read_w_h",
    "kv_dynamic_read_w_x",
    "kv_dynamic_read_b",
    "kv_sparse_marker",
    "kv_dynamic_key_w_h",
    "kv_dynamic_key_w_x",
    "kv_dynamic_key_b",
    "kv_dynamic_value_w_h",
    "kv_dynamic_value_w_x",
    "kv_dynamic_value_b",
    "kv_dynamic_marker",
)
PARAM_NAMES = BASE_PARAM_NAMES
ALL_PARAM_NAMES = BASE_PARAM_NAMES + SELECTIVE_PARAM_NAMES
DEFAULT_SKIP_RANK = 32

DEFAULT_TEXT = """
memory is a river of state . present signals enter the stream .
the system keeps compressed experience and predicts the next action .
an agent sees food , danger , fatigue , and space .
it moves toward useful signals and away from harmful signals .
text is another world with tokens instead of positions .
each word becomes a vector . each vector updates hidden memory .
the past is not copied . the past is folded into state .
fast state makes long context possible .
small rules can become large behavior when they run in parallel .
the engine learns by comparing prediction with the next token .
gradient descent changes the rules until prediction becomes easier .
state space models trade repeated attention for continuous memory .
the future is guessed from compressed experience .
"""


class TrainMetrics(NamedTuple):
    loss: jax.Array
    accuracy: jax.Array
    grad_norm: jax.Array
    mean_decay: jax.Array


def tokenize(text: str) -> list[str]:
    return tokenize_with_config(text, default_word_tokenizer_config())


def detokenize(tokens: list[str]) -> str:
    return detokenize_with_config(tokens, default_word_tokenizer_config())


def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_tokenizer_config(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vocab(tokens: list[str]) -> tuple[dict[str, int], list[str]]:
    return build_token_vocab(tokens)


def encode(tokens: list[str], token_to_id: dict[str, int]) -> np.ndarray:
    return encode_tokens(tokens, token_to_id)


def make_streams(token_ids: np.ndarray, streams: int, tokens_per_stream: int) -> np.ndarray:
    needed = streams * tokens_per_stream + 1
    repeats = int(np.ceil(needed / len(token_ids)))
    tiled = np.tile(token_ids, repeats)[: streams * tokens_per_stream]
    return tiled.reshape(streams, tokens_per_stream)


def init_params(
    key: jax.Array,
    vocab_size: int,
    input_dim: int,
    state_dim: int,
    *,
    variant: str = "static",
    skip_rank: int = DEFAULT_SKIP_RANK,
) -> dict[str, jax.Array]:
    if variant not in SSM_VARIANTS:
        raise ValueError(f"Unknown SSM variant: {variant!r}")
    if skip_rank <= 0:
        raise ValueError("--ssm-skip-rank must be positive.")
    embed_key, b_key, c_key, mix_key = jax.random.split(key, 4)
    token_embed = 0.06 * jax.random.normal(embed_key, (vocab_size, input_dim), dtype=jnp.float32)
    input_b = jax.random.normal(b_key, (input_dim, state_dim), dtype=jnp.float32)
    input_b = input_b * jnp.sqrt(jnp.asarray(1.0 / input_dim, dtype=jnp.float32))
    output_c = jax.random.normal(c_key, (state_dim, vocab_size), dtype=jnp.float32)
    output_c = output_c * jnp.sqrt(jnp.asarray(1.0 / state_dim, dtype=jnp.float32))

    params = {
        "decay_raw": jnp.zeros((state_dim,), dtype=jnp.float32),
        "hidden_bias": jnp.zeros((state_dim,), dtype=jnp.float32),
        "output_bias": jnp.zeros((vocab_size,), dtype=jnp.float32),
        "token_embed": token_embed,
        "input_b": input_b,
        "output_c": output_c,
    }
    def init_skip_d(fold: int) -> jax.Array:
        skip_key = jax.random.fold_in(c_key, fold)
        skip_d = jax.random.normal(skip_key, (input_dim, vocab_size), dtype=jnp.float32)
        return 0.01 * skip_d * jnp.sqrt(jnp.asarray(1.0 / input_dim, dtype=jnp.float32))

    if variant == "selective":
        params.update(
            {
                "select_decay_w": jnp.zeros((input_dim, state_dim), dtype=jnp.float32),
                "select_write_w": jnp.zeros((input_dim, state_dim), dtype=jnp.float32),
                "select_write_b": jnp.full((state_dim,), 2.0, dtype=jnp.float32),
            }
        )
    elif variant in {"selective-lite", "skip-selective-lite"}:
        params.update(
            {
                "select_decay_w": jnp.zeros((input_dim, state_dim), dtype=jnp.float32),
                "select_write_scalar_w": jnp.zeros((input_dim, 1), dtype=jnp.float32),
                "select_write_scalar_b": jnp.asarray([2.0], dtype=jnp.float32),
            }
        )
    elif variant in {"write-gated", "skip-write-gated"}:
        params.update(
            {
                "write_scalar_w": jnp.zeros((input_dim, 1), dtype=jnp.float32),
                "write_scalar_b": jnp.asarray([2.0], dtype=jnp.float32),
            }
        )
    if variant in {
        "skip",
        "skip-write-gated",
        "skip-selective-lite",
        "conv-skip",
        "conv-token-memory",
        "conv-token-memory-closed",
        "conv-token-memory-sparse",
    }:
        params["skip_d"] = init_skip_d(97)
    if variant in {"conv-skip", "conv-token-memory", "conv-token-memory-closed", "conv-token-memory-sparse"}:
        params["prev_skip_d"] = init_skip_d(193)
    if variant == "tied-skip":
        params["tied_skip_scale"] = jnp.asarray(1.0, dtype=jnp.float32)
    if variant == "lowrank-skip":
        rank = int(skip_rank)
        u_key = jax.random.fold_in(c_key, 211)
        v_key = jax.random.fold_in(c_key, 223)
        skip_u = jax.random.normal(u_key, (input_dim, rank), dtype=jnp.float32)
        skip_u = skip_u * jnp.sqrt(jnp.asarray(1.0 / input_dim, dtype=jnp.float32))
        skip_v = jax.random.normal(v_key, (rank, vocab_size), dtype=jnp.float32)
        skip_v = 0.01 * skip_v * jnp.sqrt(jnp.asarray(1.0 / rank, dtype=jnp.float32))
        params["skip_u"] = skip_u
        params["skip_v"] = skip_v
    if variant in {"state-mix", "skip-state-mix", "conv-state-mix"}:
        state_mix_w = jax.random.normal(mix_key, (state_dim, state_dim), dtype=jnp.float32)
        state_mix_w = 0.01 * state_mix_w * jnp.sqrt(jnp.asarray(1.0 / state_dim, dtype=jnp.float32))
        params["state_mix_w"] = state_mix_w
    if variant in {"skip-state-mix", "conv-state-mix"}:
        params["skip_d"] = init_skip_d(97)
    if variant == "conv-state-mix":
        params["prev_skip_d"] = init_skip_d(193)
    if variant in {"token-memory", "conv-token-memory", "conv-token-memory-closed", "conv-token-memory-sparse"}:
        memory_gate_init = -4.0 if variant in {"conv-token-memory-closed", "conv-token-memory-sparse"} else 0.0
        memory_logit_scale = 0.25 if variant in {"conv-token-memory-closed", "conv-token-memory-sparse"} else 1.0
        params.update(
            {
                "memory_write_token": jnp.full((vocab_size,), memory_gate_init, dtype=jnp.float32),
                "memory_read_token": jnp.full((vocab_size,), memory_gate_init, dtype=jnp.float32),
                "memory_decay_raw": jnp.asarray(5.0, dtype=jnp.float32),
                "memory_logit_scale": jnp.asarray(memory_logit_scale, dtype=jnp.float32),
            }
        )
        if variant == "conv-token-memory-closed":
            params["memory_closed_marker"] = jnp.asarray(1.0, dtype=jnp.float32)
        if variant == "conv-token-memory-sparse":
            params["memory_sparse_marker"] = jnp.asarray(1.0, dtype=jnp.float32)
    if variant in {"kv-memory", "sparse-kv-memory", "dynamic-kv-memory"}:
        params.update(
            {
                "kv_key_token": jnp.zeros((vocab_size,), dtype=jnp.float32),
                "kv_value_token": jnp.zeros((vocab_size,), dtype=jnp.float32),
                "kv_erase_token": jnp.zeros((vocab_size,), dtype=jnp.float32),
                "kv_write_context_token": jnp.zeros((vocab_size,), dtype=jnp.float32),
                "kv_deref_token": jnp.zeros((vocab_size,), dtype=jnp.float32),
                "kv_query_token": jnp.zeros((vocab_size,), dtype=jnp.float32),
                "kv_decay_raw": jnp.asarray(5.0, dtype=jnp.float32),
                "kv_logit_scale": jnp.asarray(1.0, dtype=jnp.float32),
            }
        )
        if variant in {"sparse-kv-memory", "dynamic-kv-memory"}:
            params.update(
                {
                    "kv_dynamic_read_w_h": jnp.zeros((state_dim, 1), dtype=jnp.float32),
                    "kv_dynamic_read_w_x": jnp.zeros((input_dim, 1), dtype=jnp.float32),
                    "kv_dynamic_read_b": jnp.asarray([-2.0], dtype=jnp.float32),
                }
            )
        if variant == "sparse-kv-memory":
            params["kv_sparse_marker"] = jnp.asarray(1.0, dtype=jnp.float32)
        if variant == "dynamic-kv-memory":
            # Context-conditioned write gates: the key/value/query ROLE of a token is
            # inferred from recurrent state + token embedding, not from token id. This
            # is what canonical (shared-vocab, marker-free) MQAR requires, where the
            # same token is a key in one pair and a value/query elsewhere.
            key_h, key_x, val_h, val_x = jax.random.split(jax.random.fold_in(mix_key, 7), 4)
            scale_h = 0.02 * jnp.sqrt(jnp.asarray(1.0 / state_dim, dtype=jnp.float32))
            scale_x = 0.02 * jnp.sqrt(jnp.asarray(1.0 / input_dim, dtype=jnp.float32))
            params.update(
                {
                    "kv_dynamic_key_w_h": scale_h * jax.random.normal(key_h, (state_dim, 1), dtype=jnp.float32),
                    "kv_dynamic_key_w_x": scale_x * jax.random.normal(key_x, (input_dim, 1), dtype=jnp.float32),
                    "kv_dynamic_key_b": jnp.asarray([0.0], dtype=jnp.float32),
                    "kv_dynamic_value_w_h": scale_h * jax.random.normal(val_h, (state_dim, 1), dtype=jnp.float32),
                    "kv_dynamic_value_w_x": scale_x * jax.random.normal(val_x, (input_dim, 1), dtype=jnp.float32),
                    "kv_dynamic_value_b": jnp.asarray([0.0], dtype=jnp.float32),
                    "kv_dynamic_marker": jnp.asarray(1.0, dtype=jnp.float32),
                }
            )
    return params


def param_names_for_params(params: dict[str, jax.Array]) -> tuple[str, ...]:
    known = [name for name in ALL_PARAM_NAMES if name in params]
    extra = sorted(name for name in params if name not in ALL_PARAM_NAMES)
    return tuple(known + extra)


def param_names_from_checkpoint(data: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    names = sorted(key[len("param_") :] for key in data.files if key.startswith("param_"))
    ordered = [name for name in ALL_PARAM_NAMES if name in names]
    extra = [name for name in names if name not in ALL_PARAM_NAMES]
    return tuple(ordered + extra)


def infer_ssm_variant(params: dict[str, jax.Array]) -> str:
    if "kv_dynamic_marker" in params:
        return "dynamic-kv-memory"
    if "kv_sparse_marker" in params:
        return "sparse-kv-memory"
    if "kv_key_token" in params and "kv_value_token" in params and "kv_query_token" in params:
        return "kv-memory"
    if "memory_closed_marker" in params:
        return "conv-token-memory-closed"
    if "memory_sparse_marker" in params:
        return "conv-token-memory-sparse"
    if "memory_write_token" in params and "memory_read_token" in params and "prev_skip_d" in params:
        return "conv-token-memory"
    if "memory_write_token" in params and "memory_read_token" in params:
        return "token-memory"
    if "state_mix_w" in params and "prev_skip_d" in params:
        return "conv-state-mix"
    if "state_mix_w" in params and "skip_d" in params:
        return "skip-state-mix"
    if "state_mix_w" in params:
        return "state-mix"
    if "prev_skip_d" in params:
        return "conv-skip"
    if "skip_u" in params and "skip_v" in params:
        return "lowrank-skip"
    if "tied_skip_scale" in params:
        return "tied-skip"
    if "skip_d" in params and "select_write_scalar_w" in params:
        return "skip-selective-lite"
    if "skip_d" in params and "write_scalar_w" in params:
        return "skip-write-gated"
    if "select_write_w" in params:
        return "selective"
    if "select_write_scalar_w" in params:
        return "selective-lite"
    if "write_scalar_w" in params:
        return "write-gated"
    if "skip_d" in params:
        return "skip"
    return "static"


def init_adam(params: dict[str, jax.Array]) -> dict[str, object]:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.asarray(0, dtype=jnp.int32), "m": zeros, "v": zeros}


def save_checkpoint(
    path: str,
    params: dict[str, jax.Array],
    opt_state: dict[str, object],
    vocab: list[str],
    args: argparse.Namespace,
    metrics: TrainMetrics,
    update_count: int,
    tokenizer_config: dict[str, Any] | None = None,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tokenizer_config = tokenizer_config or default_word_tokenizer_config()
    config = {
        "input_dim": int(params["token_embed"].shape[1]),
        "state_dim": int(params["decay_raw"].shape[0]),
        "vocab_size": int(params["token_embed"].shape[0]),
        "seq_len": int(args.seq_len),
        "streams": int(args.streams),
        "tokens_per_stream": int(args.tokens_per_stream),
        "ssm_variant": infer_ssm_variant(params),
        "ssm_skip_rank": int(getattr(args, "ssm_skip_rank", DEFAULT_SKIP_RANK)),
        "tokenizer": tokenizer_config.get("type", "word"),
        "tokenizer_config": tokenizer_config,
    }
    metric_payload = {
        "loss": float(metrics.loss),
        "accuracy": float(metrics.accuracy),
        "mean_decay": float(metrics.mean_decay),
        "updates": int(update_count),
    }

    payload: dict[str, object] = {
        "format_version": np.asarray(1, dtype=np.int32),
        "vocab_json": np.asarray(json.dumps(vocab, ensure_ascii=False)),
        "config_json": np.asarray(json.dumps(config, ensure_ascii=False)),
        "tokenizer_config_json": np.asarray(json.dumps(tokenizer_config, ensure_ascii=False)),
        "metrics_json": np.asarray(json.dumps(metric_payload, ensure_ascii=False)),
        "opt_step": np.asarray(int(jax.device_get(opt_state["step"])), dtype=np.int32),
    }
    param_names = param_names_for_params(params)
    for name in param_names:
        payload[f"param_{name}"] = np.asarray(jax.device_get(params[name]))
        payload[f"opt_m_{name}"] = np.asarray(jax.device_get(opt_state["m"][name]))
        payload[f"opt_v_{name}"] = np.asarray(jax.device_get(opt_state["v"][name]))

    np.savez_compressed(path, **payload)


def load_checkpoint(path: str) -> tuple[dict[str, jax.Array], dict[str, object], list[str], dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    vocab = json.loads(str(data["vocab_json"].item()))
    config = json.loads(str(data["config_json"].item())) if "config_json" in data else {}
    if "tokenizer_config_json" in data:
        tokenizer_config = json.loads(str(data["tokenizer_config_json"].item()))
    else:
        tokenizer_config = config.get("tokenizer_config") or default_word_tokenizer_config()
    config.setdefault("tokenizer", tokenizer_config.get("type", "word"))
    config.setdefault("tokenizer_config", tokenizer_config)

    param_names = param_names_from_checkpoint(data)
    params = {name: jnp.asarray(data[f"param_{name}"]) for name in param_names}
    config.setdefault("ssm_variant", "selective" if "select_decay_w" in params else "static")
    if all(f"opt_m_{name}" in data and f"opt_v_{name}" in data for name in param_names):
        opt_state = {
            "step": jnp.asarray(data["opt_step"], dtype=jnp.int32),
            "m": {name: jnp.asarray(data[f"opt_m_{name}"]) for name in param_names},
            "v": {name: jnp.asarray(data[f"opt_v_{name}"]) for name in param_names},
        }
    else:
        opt_state = init_adam(params)

    if len(vocab) != int(params["token_embed"].shape[0]):
        raise ValueError("Checkpoint vocab size does not match token embedding shape.")
    return params, opt_state, vocab, config


def decay_from_raw(raw: jax.Array) -> jax.Array:
    return 0.02 + 0.97 * jax.nn.sigmoid(raw)


def slow_decay_from_raw(raw: jax.Array) -> jax.Array:
    return 0.90 + 0.099 * jax.nn.sigmoid(raw)


def forward(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    initial_h: jax.Array,
    *,
    kv_read_gate_hard_threshold: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    tensors_x = params["token_embed"][input_ids]
    tensors_x_t = jnp.swapaxes(tensors_x, 0, 1)
    decay_a = decay_from_raw(params["decay_raw"])
    selective = "select_decay_w" in params and "select_write_w" in params
    selective_lite = "select_decay_w" in params and "select_write_scalar_w" in params
    write_gated = "write_scalar_w" in params
    state_mix = "state_mix_w" in params
    tied_skip = "tied_skip_scale" in params
    lowrank_skip = "skip_u" in params and "skip_v" in params
    prev_skip = "prev_skip_d" in params
    token_memory = "memory_write_token" in params and "memory_read_token" in params
    memory_decay = slow_decay_from_raw(params["memory_decay_raw"]) if token_memory else None
    kv_memory = "kv_key_token" in params and "kv_value_token" in params and "kv_query_token" in params
    kv_erase = "kv_erase_token" in params
    kv_write_context = "kv_write_context_token" in params
    kv_deref = "kv_deref_token" in params
    kv_decay = slow_decay_from_raw(params["kv_decay_raw"]) if kv_memory else None
    kv_dynamic_read = "kv_dynamic_read_w_h" in params
    kv_dynamic_write = "kv_dynamic_key_w_h" in params
    input_ids_t = jnp.swapaxes(input_ids, 0, 1)

    def update_state(states_h: jax.Array, x_t: jax.Array) -> jax.Array:
        input_update = x_t @ params["input_b"]
        state_update = states_h * decay_a
        if state_mix:
            state_update = state_update + states_h @ params["state_mix_w"]
        if selective:
            dynamic_decay = decay_from_raw(params["decay_raw"] + x_t @ params["select_decay_w"])
            state_update = states_h * dynamic_decay
            if state_mix:
                state_update = state_update + states_h @ params["state_mix_w"]
            write_gate = jax.nn.sigmoid(x_t @ params["select_write_w"] + params["select_write_b"])
            states_h = jnp.tanh(state_update + write_gate * input_update + params["hidden_bias"])
        elif selective_lite:
            dynamic_decay = decay_from_raw(params["decay_raw"] + x_t @ params["select_decay_w"])
            state_update = states_h * dynamic_decay
            if state_mix:
                state_update = state_update + states_h @ params["state_mix_w"]
            write_gate = jax.nn.sigmoid(x_t @ params["select_write_scalar_w"] + params["select_write_scalar_b"])
            states_h = jnp.tanh(state_update + write_gate * input_update + params["hidden_bias"])
        elif write_gated:
            write_gate = jax.nn.sigmoid(x_t @ params["write_scalar_w"] + params["write_scalar_b"])
            states_h = jnp.tanh(state_update + write_gate * input_update + params["hidden_bias"])
        else:
            states_h = jnp.tanh(state_update + input_update + params["hidden_bias"])
        return states_h

    def output_logits(states_h: jax.Array, x_t: jax.Array, prev_x: jax.Array | None = None) -> jax.Array:
        logits_y = states_h @ params["output_c"] + params["output_bias"]
        if "skip_d" in params:
            logits_y = logits_y + x_t @ params["skip_d"]
        if tied_skip:
            logits_y = logits_y + params["tied_skip_scale"] * (x_t @ jnp.swapaxes(params["token_embed"], 0, 1))
        if lowrank_skip:
            logits_y = logits_y + (x_t @ params["skip_u"]) @ params["skip_v"]
        if prev_skip and prev_x is not None:
            logits_y = logits_y + prev_x @ params["prev_skip_d"]
        return logits_y

    def memory_logits(memory_h: jax.Array, ids_t: jax.Array) -> jax.Array:
        read_gate = jax.nn.sigmoid(params["memory_read_token"][ids_t])[:, None]
        tied_logits = memory_h @ jnp.swapaxes(params["token_embed"], 0, 1)
        return read_gate * params["memory_logit_scale"] * tied_logits

    def update_memory(memory_h: jax.Array, x_t: jax.Array, ids_t: jax.Array) -> jax.Array:
        write_gate = jax.nn.sigmoid(params["memory_write_token"][ids_t])[:, None]
        return memory_h * memory_decay + write_gate * x_t

    def kv_read_gate(states_h: jax.Array, x_t: jax.Array, ids_t: jax.Array) -> jax.Array:
        if kv_dynamic_read:
            gate = jax.nn.sigmoid(
                states_h @ params["kv_dynamic_read_w_h"]
                + x_t @ params["kv_dynamic_read_w_x"]
                + params["kv_dynamic_read_b"]
            )
        else:
            gate = jax.nn.sigmoid(params["kv_query_token"][ids_t])[:, None]
        if kv_read_gate_hard_threshold is not None:
            gate = (gate > kv_read_gate_hard_threshold).astype(gate.dtype)
        return gate

    def kv_logits(
        kv_matrix: jax.Array,
        key_trace: jax.Array,
        states_h: jax.Array,
        x_t: jax.Array,
        ids_t: jax.Array,
    ) -> jax.Array:
        query_gate = kv_read_gate(states_h, x_t, ids_t)
        # Canonical MQAR scores at the query-key position itself, so the dynamic
        # variant reads the memory with the CURRENT token as the query vector.
        # Legacy variants keep reading with the lagged key_trace (their query is a
        # separate marker token one step after the key).
        query = x_t if kv_dynamic_write else key_trace
        value_read = jnp.einsum("bi,bij->bj", query, kv_matrix)
        tied_logits = value_read @ jnp.swapaxes(params["token_embed"], 0, 1)
        return query_gate * params["kv_logit_scale"] * tied_logits

    def update_kv(
        kv_matrix: jax.Array,
        key_trace: jax.Array,
        states_h: jax.Array,
        x_t: jax.Array,
        ids_t: jax.Array,
        prev_ids_t: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if kv_dynamic_write:
            key_gate = jax.nn.sigmoid(
                states_h @ params["kv_dynamic_key_w_h"]
                + x_t @ params["kv_dynamic_key_w_x"]
                + params["kv_dynamic_key_b"]
            )
            value_gate = jax.nn.sigmoid(
                states_h @ params["kv_dynamic_value_w_h"]
                + x_t @ params["kv_dynamic_value_w_x"]
                + params["kv_dynamic_value_b"]
            )
        else:
            key_gate = jax.nn.sigmoid(params["kv_key_token"][ids_t])[:, None]
            value_gate = jax.nn.sigmoid(params["kv_value_token"][ids_t])[:, None]
        if kv_write_context and prev_ids_t is not None:
            write_context_gate = jax.nn.sigmoid(params["kv_write_context_token"][prev_ids_t])[:, None]
            value_gate = value_gate * write_context_gate
            key_gate = key_gate * (1.0 - write_context_gate)
        else:
            write_context_gate = jnp.ones_like(value_gate)
        if kv_deref:
            deref_gate = jax.nn.sigmoid(params["kv_deref_token"][ids_t])[:, None] * write_context_gate
        else:
            deref_gate = jnp.zeros_like(value_gate)
        if kv_erase:
            erase_gate = jax.nn.sigmoid(params["kv_erase_token"][ids_t])[:, None]
        else:
            erase_gate = jnp.zeros_like(value_gate)
        deref_value = jnp.einsum("bi,bij->bj", x_t, kv_matrix)
        write_value = (1.0 - deref_gate) * x_t + deref_gate * deref_value
        value_gate = jnp.maximum(value_gate, deref_gate)
        erase_gate = jnp.maximum(erase_gate, deref_gate)
        key_trace = (1.0 - key_gate) * key_trace + key_gate * x_t
        erase = jnp.einsum("bi,bj->bij", key_trace, key_trace)
        write = jnp.einsum("bi,bj->bij", key_trace, write_value)
        kv_matrix = kv_matrix * kv_decay
        kv_matrix = kv_matrix - value_gate[:, None, :] * erase_gate[:, None, :] * (erase @ kv_matrix)
        kv_matrix = kv_matrix + value_gate[:, None, :] * write
        return kv_matrix, key_trace

    if kv_memory and prev_skip:
        prev_x0 = jnp.zeros((initial_h.shape[0], params["token_embed"].shape[1]), dtype=jnp.float32)
        prev_ids0 = jnp.zeros((initial_h.shape[0],), dtype=input_ids.dtype)
        key_trace0 = jnp.zeros((initial_h.shape[0], params["token_embed"].shape[1]), dtype=jnp.float32)
        kv0 = jnp.zeros(
            (initial_h.shape[0], params["token_embed"].shape[1], params["token_embed"].shape[1]),
            dtype=jnp.float32,
        )

        def step(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], pair: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
            states_h, prev_x, prev_ids_t, kv_matrix, key_trace = carry
            x_t, ids_t = pair
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t, prev_x) + kv_logits(kv_matrix, key_trace, states_h, x_t, ids_t)
            kv_matrix, key_trace = update_kv(kv_matrix, key_trace, states_h, x_t, ids_t, prev_ids_t)
            return (states_h, x_t, ids_t, kv_matrix, key_trace), logits_y

        (final_h, _, _, _, _), logits_t = jax.lax.scan(
            step,
            (initial_h, prev_x0, prev_ids0, kv0, key_trace0),
            (tensors_x_t, input_ids_t),
        )
    elif kv_memory:
        prev_ids0 = jnp.zeros((initial_h.shape[0],), dtype=input_ids.dtype)
        key_trace0 = jnp.zeros((initial_h.shape[0], params["token_embed"].shape[1]), dtype=jnp.float32)
        kv0 = jnp.zeros(
            (initial_h.shape[0], params["token_embed"].shape[1], params["token_embed"].shape[1]),
            dtype=jnp.float32,
        )

        def step(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], pair: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
            states_h, prev_ids_t, kv_matrix, key_trace = carry
            x_t, ids_t = pair
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t) + kv_logits(kv_matrix, key_trace, states_h, x_t, ids_t)
            kv_matrix, key_trace = update_kv(kv_matrix, key_trace, states_h, x_t, ids_t, prev_ids_t)
            return (states_h, ids_t, kv_matrix, key_trace), logits_y

        (final_h, _, _, _), logits_t = jax.lax.scan(
            step,
            (initial_h, prev_ids0, kv0, key_trace0),
            (tensors_x_t, input_ids_t),
        )
    elif token_memory and prev_skip:
        prev_x0 = jnp.zeros((initial_h.shape[0], params["token_embed"].shape[1]), dtype=jnp.float32)
        memory_h0 = jnp.zeros((initial_h.shape[0], params["token_embed"].shape[1]), dtype=jnp.float32)

        def step(
            carry: tuple[jax.Array, jax.Array, jax.Array], pair: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
            states_h, prev_x, memory_h = carry
            x_t, ids_t = pair
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t, prev_x) + memory_logits(memory_h, ids_t)
            memory_h = update_memory(memory_h, x_t, ids_t)
            return (states_h, x_t, memory_h), logits_y

        (final_h, _, _), logits_t = jax.lax.scan(step, (initial_h, prev_x0, memory_h0), (tensors_x_t, input_ids_t))
    elif token_memory:
        memory_h0 = jnp.zeros((initial_h.shape[0], params["token_embed"].shape[1]), dtype=jnp.float32)

        def step(
            carry: tuple[jax.Array, jax.Array], pair: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            states_h, memory_h = carry
            x_t, ids_t = pair
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t) + memory_logits(memory_h, ids_t)
            memory_h = update_memory(memory_h, x_t, ids_t)
            return (states_h, memory_h), logits_y

        (final_h, _), logits_t = jax.lax.scan(step, (initial_h, memory_h0), (tensors_x_t, input_ids_t))
    elif prev_skip:
        prev_x0 = jnp.zeros((initial_h.shape[0], params["token_embed"].shape[1]), dtype=jnp.float32)

        def step(
            carry: tuple[jax.Array, jax.Array], x_t: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            states_h, prev_x = carry
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t, prev_x)
            return (states_h, x_t), logits_y

        (final_h, _), logits_t = jax.lax.scan(step, (initial_h, prev_x0), tensors_x_t)
    else:
        def step(states_h: jax.Array, x_t: jax.Array) -> tuple[jax.Array, jax.Array]:
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t)
            return states_h, logits_y

        final_h, logits_t = jax.lax.scan(step, initial_h, tensors_x_t)
    logits = jnp.swapaxes(logits_t, 0, 1)
    return logits, final_h


def kv_read_gates(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    initial_h: jax.Array,
    *,
    kv_read_gate_hard_threshold: float | None = None,
) -> jax.Array:
    if "kv_key_token" not in params or "kv_value_token" not in params or "kv_query_token" not in params:
        return jnp.zeros(input_ids.shape, dtype=jnp.float32)

    tensors_x = params["token_embed"][input_ids]
    tensors_x_t = jnp.swapaxes(tensors_x, 0, 1)
    input_ids_t = jnp.swapaxes(input_ids, 0, 1)
    decay_a = decay_from_raw(params["decay_raw"])
    state_mix = "state_mix_w" in params
    dynamic_read = "kv_dynamic_read_w_h" in params

    def update_state(states_h: jax.Array, x_t: jax.Array) -> jax.Array:
        input_update = x_t @ params["input_b"]
        state_update = states_h * decay_a
        if state_mix:
            state_update = state_update + states_h @ params["state_mix_w"]
        return jnp.tanh(state_update + input_update + params["hidden_bias"])

    def step(states_h: jax.Array, pair: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x_t, ids_t = pair
        states_h = update_state(states_h, x_t)
        if dynamic_read:
            gate = jax.nn.sigmoid(
                states_h @ params["kv_dynamic_read_w_h"]
                + x_t @ params["kv_dynamic_read_w_x"]
                + params["kv_dynamic_read_b"]
            )
        else:
            gate = jax.nn.sigmoid(params["kv_query_token"][ids_t])[:, None]
        if kv_read_gate_hard_threshold is not None:
            gate = (gate > kv_read_gate_hard_threshold).astype(gate.dtype)
        return states_h, gate[:, 0]

    _, gates_t = jax.lax.scan(step, initial_h, (tensors_x_t, input_ids_t))
    return jnp.swapaxes(gates_t, 0, 1)


def loss_fn(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    initial_h: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    logits, final_h = forward(params, input_ids, initial_h)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1)[..., 0]
    loss = jnp.mean(nll)
    predictions = jnp.argmax(logits, axis=-1).astype(target_ids.dtype)
    accuracy = jnp.mean(predictions == target_ids)
    return loss, (accuracy, final_h)


def memory_gate_penalty(params: dict[str, jax.Array]) -> jax.Array:
    if "memory_write_token" not in params or "memory_read_token" not in params:
        return jnp.asarray(0.0, dtype=jnp.float32)
    write_gate = jax.nn.sigmoid(params["memory_write_token"])
    read_gate = jax.nn.sigmoid(params["memory_read_token"])
    return 0.5 * (jnp.mean(write_gate) + jnp.mean(read_gate))


def global_norm(tree: object) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def clip_grads(grads: dict[str, jax.Array], max_norm: float) -> tuple[dict[str, jax.Array], jax.Array]:
    norm = global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1.0e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads), norm


def adam_update(
    params: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    opt_state: dict[str, object],
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
) -> tuple[dict[str, jax.Array], dict[str, object]]:
    step = opt_state["step"] + 1
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, opt_state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g), opt_state["v"], grads)

    m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2**step), v)
    params = jax.tree_util.tree_map(
        lambda p, m_h, v_h: p - learning_rate * m_h / (jnp.sqrt(v_h) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, {"step": step, "m": m, "v": v}


@partial(jax.jit, static_argnames=("learning_rate", "grad_clip", "memory_gate_l1"))
def train_step(
    params: dict[str, jax.Array],
    opt_state: dict[str, object],
    input_ids: jax.Array,
    target_ids: jax.Array,
    states_h: jax.Array,
    *,
    learning_rate: float,
    grad_clip: float,
    memory_gate_l1: float = 0.0,
) -> tuple[dict[str, jax.Array], dict[str, object], jax.Array, TrainMetrics]:
    def train_loss_fn(train_params: dict[str, jax.Array]) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        nll_loss, (accuracy, final_h) = loss_fn(train_params, input_ids, target_ids, states_h)
        penalty = memory_gate_penalty(train_params)
        total_loss = nll_loss + memory_gate_l1 * penalty
        return total_loss, (accuracy, final_h, nll_loss)

    (loss, (accuracy, final_h, nll_loss)), grads = jax.value_and_grad(train_loss_fn, has_aux=True)(params)
    grads, grad_norm = clip_grads(grads, grad_clip)
    params, opt_state = adam_update(params, grads, opt_state, learning_rate)
    final_h = jax.lax.stop_gradient(final_h)
    metrics = TrainMetrics(
        loss=nll_loss,
        accuracy=accuracy,
        grad_norm=grad_norm,
        mean_decay=jnp.mean(decay_from_raw(params["decay_raw"])),
    )
    return params, opt_state, final_h, metrics


@partial(jax.jit, static_argnames=("seq_len",))
def eval_loss(
    params: dict[str, jax.Array],
    streams: jax.Array,
    *,
    seq_len: int,
) -> TrainMetrics:
    input_ids = streams[:, :seq_len]
    target_ids = streams[:, 1 : seq_len + 1]
    states_h = jnp.zeros((streams.shape[0], params["decay_raw"].shape[0]), dtype=jnp.float32)
    loss, (accuracy, _) = loss_fn(params, input_ids, target_ids, states_h)
    return TrainMetrics(
        loss=loss,
        accuracy=accuracy,
        grad_norm=jnp.asarray(0.0, dtype=jnp.float32),
        mean_decay=jnp.mean(decay_from_raw(params["decay_raw"])),
    )


def sample_text(
    params: dict[str, jax.Array],
    token_to_id: dict[str, int],
    vocab: list[str],
    prompt: str,
    steps: int,
    temperature: float,
    seed: int,
    tokenizer_config: dict[str, Any] | None = None,
) -> str:
    key = jax.random.PRNGKey(seed)
    unk_id = token_to_id[UNK]
    tokenizer_config = tokenizer_config or default_word_tokenizer_config()
    prompt_tokens = tokenize_with_config(prompt, tokenizer_config)
    prompt_ids = [token_to_id.get(token, unk_id) for token in prompt_tokens] or [token_to_id[BOS]]
    state_h = jnp.zeros((1, params["decay_raw"].shape[0]), dtype=jnp.float32)

    generated = list(prompt_tokens)
    current_id = prompt_ids[0]
    for token_id in prompt_ids:
        logits, state_h = forward(
            params,
            jnp.asarray([[token_id]], dtype=jnp.int32),
            state_h,
        )
        current_id = token_id

    for _ in range(steps):
        logits, state_h = forward(
            params,
            jnp.asarray([[current_id]], dtype=jnp.int32),
            state_h,
        )
        next_logits = logits[0, -1] / max(temperature, 1.0e-6)
        if temperature <= 0.0:
            current_id = int(jnp.argmax(next_logits))
        else:
            key, sample_key = jax.random.split(key)
            current_id = int(jax.random.categorical(sample_key, next_logits))
        generated.append(vocab[current_id])

    return detokenize_with_config(generated, tokenizer_config)


def run(args: argparse.Namespace) -> None:
    text = load_text(args.text_file)
    key = jax.random.PRNGKey(args.seed)
    checkpoint_config: dict[str, object] = {}
    if args.load_checkpoint:
        params, opt_state, vocab, checkpoint_config = load_checkpoint(args.load_checkpoint)
        token_to_id = {token: i for i, token in enumerate(vocab)}
        args.input_dim = int(params["token_embed"].shape[1])
        args.state_dim = int(params["decay_raw"].shape[0])
        tokenizer_config = checkpoint_config.get("tokenizer_config") or default_word_tokenizer_config()
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
    if len(tokens) < 16:
        raise SystemExit("Need at least 16 tokens to train the text SSM prototype.")

    if not args.load_checkpoint:
        token_to_id, vocab = build_vocab_for_tokenizer(tokens, tokenizer_config)
        params = init_params(
            key,
            len(vocab),
            args.input_dim,
            args.state_dim,
            variant=args.ssm_variant,
            skip_rank=args.ssm_skip_rank,
        )
        opt_state = init_adam(params)

    token_ids = encode(tokens, token_to_id)
    streams_np = make_streams(token_ids, args.streams, args.tokens_per_stream)
    streams = jnp.asarray(streams_np)

    if args.seq_len + 1 > args.tokens_per_stream:
        raise SystemExit("--seq-len must be smaller than --tokens-per-stream.")

    states_h = jnp.zeros((args.streams, args.state_dim), dtype=jnp.float32)

    baseline = eval_loss(params, streams, seq_len=args.seq_len)
    jax.block_until_ready(baseline.loss)

    print("TextPy/SoA trainable SSM text prototype")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("SoA rails")
    print(f"  TOKEN_STREAMS: {tuple(streams.shape)} int32")
    print(f"  STATES_H:      {tuple(states_h.shape)} float32")
    print(f"  EMBED_X:       {(len(vocab), args.input_dim)} float32")
    print(f"  WEIGHTS_A:     {(args.state_dim,)} trainable decay")
    print(f"  WEIGHTS_B:     {(args.input_dim, args.state_dim)}")
    print(f"  WEIGHTS_C:     {(args.state_dim, len(vocab))}")
    print(f"  SSM_VARIANT:   {infer_ssm_variant(params)}")
    print("")
    print("Corpus")
    print(f"  source_tokens: {len(tokens):,}")
    print(f"  vocab_size: {len(vocab):,}")
    print(f"  tokenizer: {tokenizer_config.get('type', 'word')}")
    if tokenizer_config.get("type") == "bpe":
        summary = tokenizer_summary(text, tokenizer_config, tokens)
        print(f"  word_tokens: {summary['word_tokens']:,}")
        print(f"  tokens_per_word: {summary['tokens_per_word']:.3f}")
        print(f"  bpe_merges: {summary['merges']:,}")
    print(f"  streams: {args.streams:,}")
    print(f"  seq_len: {args.seq_len:,}")
    print(f"  epochs: {args.epochs:,}")
    if args.load_checkpoint:
        print(f"  loaded_checkpoint: {args.load_checkpoint}")
        if checkpoint_config:
            print(f"  checkpoint_config: {checkpoint_config}")
    print(f"  initial_loss: {float(baseline.loss):.4f}")
    print(f"  initial_accuracy: {float(baseline.accuracy):.4f}")
    print("")

    start = time.perf_counter()
    update_count = 0
    chunks_per_epoch = (args.tokens_per_stream - 1) // args.seq_len

    epochs_to_run = 0 if args.eval_only else args.epochs
    metrics = baseline
    for epoch in range(1, epochs_to_run + 1):
        states_h = jnp.zeros_like(states_h)
        for chunk in range(chunks_per_epoch):
            cursor = chunk * args.seq_len
            input_ids = streams[:, cursor : cursor + args.seq_len]
            target_ids = streams[:, cursor + 1 : cursor + args.seq_len + 1]
            params, opt_state, states_h, metrics = train_step(
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

        jax.block_until_ready(metrics.loss)
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={float(metrics.loss):.4f} "
                f"ppl={float(jnp.exp(metrics.loss)):.2f} "
                f"acc={float(metrics.accuracy):.4f} "
                f"grad_norm={float(metrics.grad_norm):.3f} "
                f"mean_A={float(metrics.mean_decay):.3f}"
            )
    if args.eval_only:
        print("eval_only: skipped training updates")

    elapsed = time.perf_counter() - start
    final_eval = eval_loss(params, streams, seq_len=args.seq_len)
    jax.block_until_ready(final_eval.loss)
    trained_tokens = update_count * args.streams * args.seq_len
    throughput = trained_tokens / max(elapsed, 1.0e-9)

    print("")
    print("Run")
    print(f"  updates: {update_count:,}")
    print(f"  trained_tokens: {trained_tokens:,}")
    print(f"  train_s: {elapsed:.4f}")
    print(f"  throughput_tokens_s: {throughput:,.0f}")
    print(f"  final_loss: {float(final_eval.loss):.4f}")
    print(f"  final_accuracy: {float(final_eval.accuracy):.4f}")
    print(f"  mean_decay_A: {float(final_eval.mean_decay):.4f}")
    if args.save_checkpoint:
        save_checkpoint(
            args.save_checkpoint,
            params,
            opt_state,
            vocab,
            args,
            final_eval,
            update_count,
            tokenizer_config=tokenizer_config,
        )
        print(f"  saved_checkpoint: {args.save_checkpoint}")
    print("")
    print("Sample")
    print(
        "  "
        + sample_text(
            params,
            token_to_id,
            vocab,
            args.prompt,
            args.sample_steps,
            args.temperature,
            args.seed + 1,
            tokenizer_config=tokenizer_config,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny JAX SSM next-token model with SoA hidden-state streams."
    )
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--tokenizer", choices=("word", "bpe"), default="word")
    parser.add_argument("--tokenizer-config", default=None, help="Reuse a saved tokenizer_config JSON.")
    parser.add_argument("--bpe-vocab-size", type=int, default=512)
    parser.add_argument("--bpe-min-frequency", type=int, default=2)
    parser.add_argument("--bpe-train-chars", type=int, default=250_000)
    parser.add_argument("--streams", type=int, default=64)
    parser.add_argument("--tokens-per-stream", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=48)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument(
        "--ssm-variant",
        choices=SSM_VARIANTS,
        default="static",
    )
    parser.add_argument("--ssm-skip-rank", type=int, default=DEFAULT_SKIP_RANK)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--learning-rate", type=float, default=0.012)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--memory-gate-l1", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=3)
    parser.add_argument("--sample-steps", type=int, default=34)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default="memory is")
    parser.add_argument("--save-checkpoint", default=None)
    parser.add_argument("--load-checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
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
