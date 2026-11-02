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

from jax_backend import configure_environment


def requested_backend(argv: list[str]) -> str:
    backend = os.environ.get("JAX_PLATFORM_NAME", "cpu")
    for i, arg in enumerate(argv):
        if arg == "--backend" and i + 1 < len(argv):
            backend = argv[i + 1]
        elif arg.startswith("--backend="):
            backend = arg.split("=", 1)[1]
    return backend


REQUESTED_BACKEND = requested_backend(sys.argv[1:])
configure_environment(REQUESTED_BACKEND)

import jax
import jax.numpy as jnp
import numpy as np

from causal_frequency import causal_frequency_logits, initial_frequency_state, frequency_kwargs
from rare_transition import rare_transition_logits, initial_rare_transition_state
from rare_occurrence import rare_occurrence_logits, initial_rare_occurrence_state
from unseen_occurrence import unseen_occurrence_logits, initial_unseen_occurrence_state

from text_tokenization import (
    BOS,
    UNK,
    build_tokenizer_config,
    build_vocab_for_tokenizer,
    build_vocab as build_token_vocab,
    default_word_tokenizer_config,
    detokenize_with_config,
    encode_tokens,
    fit_bpe_training_data,
    make_token_streams,
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
    "conv-rms-skip",
    "conv-rms-lexical-skip",
    "conv-local-attention",
    "state-mix",
    "skip-state-mix",
    "conv-state-mix",
    "token-memory",
    "conv-token-memory",
    "conv-token-memory-closed",
    "conv-token-memory-sparse",
    "kv-memory",
    "context-kv-memory",
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
    "kv_context_w",
    "kv_context_b",
    "kv_key_token",
    "kv_value_token",
    "kv_erase_token",
    "kv_write_context_token",
    "kv_deref_token",
    "kv_query_token",
    "kv_decay_raw",
    "kv_durable_marker",
    "kv_commit_w",
    "kv_commit_b",
    "kv_commit_embed",
    "kv_commit_window_w",
    "kv_commit_window_b",
    "kv_commit_out",
    "kv_window_roles_marker",
    "kv_cosine_read_marker",
    "kv_commit_threshold",
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
    return make_token_streams(token_ids, streams, tokens_per_stream)


def training_target_counts(training_streams, *, seq_len: int, vocab_size: int):
    """Count only targets of complete training chunks."""
    streams = np.asarray(training_streams)
    if streams.ndim != 2 or not np.issubdtype(streams.dtype, np.integer) or seq_len < 1:
        raise ValueError("Need integer training streams and a positive seq_len")
    used = ((streams.shape[1] - 1) // seq_len) * seq_len
    if streams.shape[0] < 1 or used < 1:
        raise ValueError("Need at least one complete training target chunk")
    targets = streams[:, 1 : used + 1].reshape(-1)
    if np.any(targets < 0) or np.any(targets >= vocab_size):
        raise ValueError("Training target outside classifier vocabulary")
    return np.bincount(targets, minlength=vocab_size).astype(np.float64)


def initialize_output_bias(params, training_streams, *, seq_len: int, label_smoothing: float = 0.0):
    """Initialize an existing classifier bias using only targets actually trained on."""
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be finite and in [0, 1)")
    key = "output_bias" if "output_bias" in params else "lm_head_b"
    vocab_size = int(params[key].shape[0])
    counts = training_target_counts(training_streams, seq_len=seq_len, vocab_size=vocab_size)
    if label_smoothing == 0.0:
        counts += 0.5
    prior = (1.0 - label_smoothing) * counts / counts.sum() + label_smoothing / vocab_size
    bias = np.log(prior)
    bias -= bias.mean()
    return {**params, key: jnp.asarray(bias, dtype=params[key].dtype)}


def init_params(
    key: jax.Array,
    vocab_size: int,
    input_dim: int,
    state_dim: int,
    *,
    variant: str = "static",
    skip_rank: int = DEFAULT_SKIP_RANK,
    attention_window: int = 128,
    attention_dim: int = 32,
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

    if variant in {"conv-rms-skip", "conv-rms-lexical-skip"}:
        params["readout_rms_scale"] = jnp.ones((state_dim,), dtype=jnp.float32)
    if variant == "conv-rms-lexical-skip":
        params["lexical_rms_scale"] = jnp.ones((input_dim,), dtype=jnp.float32)

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
        "conv-rms-skip",
        "conv-rms-lexical-skip",
        "conv-local-attention",
        "conv-token-memory",
        "conv-token-memory-closed",
        "conv-token-memory-sparse",
    }:
        params["skip_d"] = init_skip_d(97)
    if variant in {"conv-skip", "conv-rms-skip", "conv-rms-lexical-skip", "conv-local-attention", "conv-token-memory", "conv-token-memory-closed", "conv-token-memory-sparse"}:
        params["prev_skip_d"] = init_skip_d(193)
    if variant == "conv-local-attention":
        if attention_window < 1 or attention_dim < 1:
            raise ValueError("attention_window and attention_dim must be positive")
        for offset, name in enumerate(("local_q", "local_k", "local_v")):
            params[name] = jax.random.normal(jax.random.fold_in(mix_key, 300 + offset),
                                            (input_dim, attention_dim)) / jnp.sqrt(float(input_dim))
        params["local_o"] = 0.01 * jax.random.normal(jax.random.fold_in(mix_key, 303),
                                                   (attention_dim, state_dim)) / jnp.sqrt(float(attention_dim))
        params["local_position_bias"] = jnp.zeros((attention_window,), dtype=jnp.float32)
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
    if variant in {"kv-memory", "context-kv-memory", "sparse-kv-memory", "dynamic-kv-memory"}:
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
        if variant == "context-kv-memory":
            params["kv_context_w"] = jnp.zeros((state_dim, 5), dtype=jnp.float32)
            params["kv_context_b"] = jnp.zeros((5,), dtype=jnp.float32)
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
    if "lexical_rms_scale" in params:
        return "conv-rms-lexical-skip"
    if "readout_rms_scale" in params:
        return "conv-rms-skip"
    if "kv_context_w" in params:
        return "context-kv-memory"
    if "local_q" in params:
        return "conv-local-attention"
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
    stream_state: SSMState | None = None,
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
        "label_smoothing": float(getattr(args, "label_smoothing", 0.0)),
        "output_bias_init": getattr(args, "output_bias_init", "zero"),
        "tokenizer": tokenizer_config.get("type", "word"),
        "tokenizer_config": tokenizer_config,
    }
    if getattr(args, "causal_frequency_config", None) is not None:
        config["causal_frequency"] = args.causal_frequency_config
    if getattr(args, "rare_transition_config", None) is not None:
        config["rare_transition"] = args.rare_transition_config
    if getattr(args, "rare_occurrence_config", None) is not None:
        config["rare_occurrence"] = args.rare_occurrence_config
    if getattr(args, "unseen_occurrence_config", None) is not None:
        config["unseen_occurrence"] = args.unseen_occurrence_config
    if getattr(args, "vocab_coverage", None) is not None:
        config["vocab_coverage"] = args.vocab_coverage
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

    if stream_state is not None:
        payload["stream_state_version"] = np.asarray(1, dtype=np.int32)
        for name, value in zip(SSMState._fields, stream_state):
            if value is not None:
                payload[f"stream_{name}"] = np.asarray(jax.device_get(value))
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


class SSMState(NamedTuple):
    """Complete recurrent carry; each batch row is an independent stream."""

    h: jax.Array
    prev_x: jax.Array | None
    prev_ids: jax.Array | None
    memory: jax.Array | None
    kv: jax.Array | None
    key_trace: jax.Array | None
    local_keys: jax.Array | None = None
    local_values: jax.Array | None = None
    local_count: jax.Array | None = None
    commit_context: jax.Array | None = None
    fact_memory: jax.Array | None = None
    fact_context: jax.Array | None = None
    fact_write_key: jax.Array | None = None
    fact_read_key: jax.Array | None = None
    fact_write_pos: jax.Array | None = None
    fact_read_pos: jax.Array | None = None
    fact_write_active: jax.Array | None = None
    fact_read_active: jax.Array | None = None
    fact_entity_trace: jax.Array | None = None
    fact_relation_trace: jax.Array | None = None
    open_values: jax.Array | None = None
    open_entities: jax.Array | None = None
    open_relations: jax.Array | None = None
    open_used: jax.Array | None = None
    open_clock: jax.Array | None = None
    open_write_slot: jax.Array | None = None
    open_read_slot: jax.Array | None = None
    open_write_pos: jax.Array | None = None
    open_read_pos: jax.Array | None = None
    open_write_active: jax.Array | None = None
    open_read_active: jax.Array | None = None
    open_context: jax.Array | None = None
    open_sem_entities: jax.Array | None = None
    open_sem_relations: jax.Array | None = None
    open_external_relations: jax.Array | None = None


def init_stream_state(params: dict[str, jax.Array], batch_size: int) -> SSMState:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    d = params["token_embed"].shape[1]
    kv = "kv_key_token" in params
    local = "local_q" in params
    cache_shape = (batch_size, params["local_position_bias"].shape[0] - 1,
                   params["local_q"].shape[1]) if local else None
    fact_state = (None,) * 10
    if "fact_embed" in params:
        from neural_fact_memory import initial as initial_facts
        fact_state = initial_facts(params, batch_size)
    open_state = (None,) * 12
    if "open_embed" in params:
        from open_neural_memory import initial as initial_open
        open_state = initial_open(params, batch_size)
    sem_state = (None, None)
    if "sem_relation_embed" in params:
        from semantic_keys import initial as initial_semantic
        sem_state = initial_semantic(params, batch_size)
    return SSMState(
        jnp.zeros((batch_size, params["decay_raw"].shape[0]), dtype=jnp.float32),
        jnp.zeros((batch_size, d), dtype=jnp.float32) if "prev_skip_d" in params else None,
        jnp.zeros((batch_size,), dtype=jnp.int32) if kv else None,
        jnp.zeros((batch_size, d), dtype=jnp.float32) if "memory_write_token" in params else None,
        jnp.zeros((batch_size, d, d), dtype=jnp.float32) if kv else None,
        jnp.zeros((batch_size, d), dtype=jnp.float32) if kv else None,
        jnp.zeros(cache_shape, dtype=jnp.float32) if local else None,
        jnp.zeros(cache_shape, dtype=jnp.float32) if local else None,
        jnp.zeros((batch_size,), dtype=jnp.int32) if local else None,
        jnp.zeros((batch_size, 11, params["kv_commit_embed"].shape[1]), dtype=jnp.float32) if "kv_commit_embed" in params else None,
        *fact_state,
        *open_state,
        *sem_state,
        jnp.zeros((batch_size, params["open_slot_bias"].shape[0], 64), dtype=jnp.float32) if "v6_projection" in params else None,
    )


def commit_logits(params, input_ids, context):
    """Learned causal convolution; its fixed carry survives chunk boundaries."""
    embedded = params["kv_commit_embed"][input_ids]
    joined = jnp.concatenate([context, embedded], axis=1)
    windows = jax.vmap(lambda t: jax.lax.dynamic_slice_in_dim(joined, t, 12, axis=1))(
        jnp.arange(input_ids.shape[1]))
    windows = jnp.swapaxes(windows, 0, 1).reshape(input_ids.shape[0], input_ids.shape[1], -1)
    hidden = jnp.tanh(windows @ params["kv_commit_window_w"] + params["kv_commit_window_b"])
    return hidden @ params["kv_commit_out"] + params["kv_commit_b"], joined[:, -11:]


def load_stream_checkpoint(path: str) -> tuple[dict, dict, list[str], dict, SSMState]:
    """Restore weights, Adam and the complete stream carry from one NPZ."""
    params, opt, vocab, config = load_checkpoint(path)
    with np.load(path, allow_pickle=False) as data:
        if "stream_state_version" not in data or int(data["stream_state_version"]) != 1:
            raise ValueError("Checkpoint has no supported stream state; start a fresh stream explicitly")
        if "stream_h" not in data or data["stream_h"].ndim != 2:
            raise ValueError("Invalid stream hidden state")
        expected = init_stream_state(params, data["stream_h"].shape[0])
        values = []
        for name, reference in zip(SSMState._fields, expected):
            key = f"stream_{name}"
            if reference is None:
                if key in data:
                    raise ValueError(f"Unexpected stream state: {name}")
                values.append(None)
            else:
                if key not in data:
                    raise ValueError(f"Missing stream state: {name}")
                value = data[key]
                if value.shape != reference.shape or value.dtype != reference.dtype or not np.isfinite(value).all():
                    raise ValueError(f"Invalid stream state: {name}")
                if name in {"prev_ids", "open_context"} and ((value < 0).any() or (value >= len(vocab)).any()):
                    raise ValueError("Invalid previous token IDs")
                values.append(jnp.asarray(value))
    return params, opt, vocab, config, SSMState(*values)


def reset_stream_state(state: SSMState, reset_rows: jax.Array) -> SSMState:
    """Reset selected rows at document boundaries, never the other streams."""
    if reset_rows.shape != (state.h.shape[0],) or reset_rows.dtype != jnp.bool_:
        raise ValueError("reset_rows must be a boolean vector with one entry per stream")
    return jax.tree_util.tree_map(
        lambda x: jnp.where(reset_rows.reshape((-1,) + (1,) * (x.ndim - 1)), 0, x), state
    )


def forward(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    initial_h: jax.Array | SSMState,
    *,
    kv_read_gate_hard_threshold: float | None = None,
    memory_write_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | SSMState]:
    """Pass SSMState to carry every memory rail; arrays retain the legacy API."""
    write_mask = jnp.ones((input_ids.shape[0],), dtype=jnp.float32) if memory_write_mask is None else jnp.asarray(memory_write_mask)
    if write_mask.shape != (input_ids.shape[0],):
        raise ValueError("memory_write_mask must have one entry per stream")
    complete = isinstance(initial_h, SSMState)
    state = initial_h if complete else init_stream_state(params, input_ids.shape[0])._replace(h=initial_h)
    expected = init_stream_state(params, input_ids.shape[0])
    for name, value, reference in zip(SSMState._fields, state, expected):
        if (value is None) != (reference is None) or (
            value is not None and (value.shape != reference.shape or value.dtype != reference.dtype)
        ):
            raise ValueError(f"Invalid stream state field: {name}")
    initial_h = state.h
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
    if "kv_commit_embed" in params:
        commit_z, commit_context = commit_logits(params, input_ids, state.commit_context)
        state = state._replace(commit_context=commit_context)
        soft_commit = jax.nn.sigmoid(commit_z)
        thresholds = jnp.full((soft_commit.shape[-1],), 0.5).at[0].set(params.get("kv_commit_threshold", 0.5))
        commit_gates = soft_commit + jax.lax.stop_gradient((soft_commit >= thresholds).astype(soft_commit.dtype) - soft_commit)
    else:
        commit_gates = jnp.ones((*input_ids.shape, 1), dtype=jnp.float32)
    commit_gates_t = jnp.swapaxes(commit_gates, 0, 1)

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
        readout_h = states_h
        if "readout_rms_scale" in params:
            readout_h = states_h * jax.lax.rsqrt(jnp.mean(jnp.square(states_h), axis=-1, keepdims=True) + 1e-5)
            readout_h = readout_h * params["readout_rms_scale"]
        logits_y = readout_h @ params["output_c"] + params["output_bias"]
        if "lexical_rms_scale" in params:
            # Match lexical feature scale to the normalized recurrent readout.
            x_t = x_t * jax.lax.rsqrt(jnp.mean(jnp.square(x_t), axis=-1, keepdims=True) + 1e-5)
            x_t = x_t * params["lexical_rms_scale"]
            if prev_x is not None:
                prev_x = prev_x * jax.lax.rsqrt(jnp.mean(jnp.square(prev_x), axis=-1, keepdims=True) + 1e-5)
                prev_x = prev_x * params["lexical_rms_scale"]
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
        updated = memory_h * memory_decay + write_gate * x_t
        return updated if memory_write_mask is None else jnp.where(write_mask[:,None] > 0, updated, memory_h)

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
        readout = params["token_embed"]
        if "kv_cosine_read_marker" in params:
            readout = readout / jnp.maximum(jnp.linalg.norm(readout, axis=-1, keepdims=True), 1e-6)
        tied_logits = value_read @ jnp.swapaxes(readout, 0, 1)
        return query_gate * params["kv_logit_scale"] * tied_logits

    def update_kv(
        kv_matrix: jax.Array,
        key_trace: jax.Array,
        states_h: jax.Array,
        x_t: jax.Array,
        ids_t: jax.Array,
        prev_ids_t: jax.Array | None = None,
        commit_t: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        prior_matrix = kv_matrix
        # Shared learned context roles let an ordinary word become a key/value.
        roles = (states_h @ params["kv_context_w"] + params["kv_context_b"]
                 if "kv_context_w" in params else jnp.zeros((states_h.shape[0], 5), dtype=jnp.float32))
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
            key_gate = jax.nn.sigmoid(params["kv_key_token"][ids_t, None] + roles[:, :1])
            value_gate = jax.nn.sigmoid(params["kv_value_token"][ids_t, None] + roles[:, 1:2])
        if kv_write_context and prev_ids_t is not None:
            write_context_gate = jax.nn.sigmoid(params["kv_write_context_token"][prev_ids_t, None] + roles[:, 3:4])
            value_gate = value_gate * write_context_gate
            key_gate = key_gate * (1.0 - write_context_gate)
        else:
            write_context_gate = jnp.ones_like(value_gate)
        if kv_deref:
            deref_gate = jax.nn.sigmoid(params["kv_deref_token"][ids_t, None] + roles[:, 4:5]) * write_context_gate
        else:
            deref_gate = jnp.zeros_like(value_gate)
        if kv_erase:
            erase_gate = jax.nn.sigmoid(params["kv_erase_token"][ids_t, None] + roles[:, 2:3])
        else:
            erase_gate = jnp.zeros_like(value_gate)
        deref_value = jnp.einsum("bi,bij->bj", x_t, kv_matrix)
        write_value = (1.0 - deref_gate) * x_t + deref_gate * deref_value
        value_gate = jnp.maximum(value_gate, deref_gate)
        erase_gate = jnp.maximum(erase_gate, deref_gate)
        if "kv_window_roles_marker" in params:
            key_gate = commit_t[:, 1:2]
        key_trace = (1.0 - key_gate) * key_trace + key_gate * x_t
        write = jnp.einsum("bi,bj->bij", key_trace, write_value)
        if "kv_commit_embed" in params:
            value_gate = commit_t[:, :1] if "kv_window_roles_marker" in params else value_gate * commit_t
        elif "kv_commit_w" in params:
            # A learned commit decision prevents tiny sigmoid writes accumulating on unrelated text.
            soft = jax.nn.sigmoid(states_h @ params["kv_commit_w"] + params["kv_commit_b"])
            commit = soft + jax.lax.stop_gradient((soft >= 0.5).astype(soft.dtype) - soft)
            value_gate = value_gate * commit
        if "kv_durable_marker" not in params:
            kv_matrix = kv_matrix * kv_decay
        # (k k^T) M = k (k^T M): avoid a cubic dense matrix product per token.
        erased_values = jnp.einsum("bi,bij->bj", key_trace, kv_matrix)
        erase = jnp.einsum("bi,bj->bij", key_trace, erased_values)
        if "kv_window_roles_marker" in params:
            # Normalize the delta update so a correction replaces its key projection at any key norm.
            norm2 = jnp.sum(key_trace * key_trace, axis=-1, keepdims=True)
            delta = jnp.einsum("bi,bj->bij", key_trace, (write_value - erased_values) / jnp.maximum(norm2, 1e-6))
            kv_matrix = kv_matrix + value_gate[:, None, :] * delta
        else:
            kv_matrix = kv_matrix - value_gate[:, None, :] * erase_gate[:, None, :] * erase
            kv_matrix = kv_matrix + value_gate[:, None, :] * write
        if memory_write_mask is not None:
            kv_matrix = jnp.where(write_mask[:,None,None] > 0, kv_matrix, prior_matrix)
        return kv_matrix, key_trace

    if kv_memory and prev_skip:
        def step(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], pair: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
            states_h, prev_x, prev_ids_t, kv_matrix, key_trace = carry
            x_t, ids_t, commit_t = pair
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t, prev_x) + kv_logits(kv_matrix, key_trace, states_h, x_t, ids_t)
            kv_matrix, key_trace = update_kv(kv_matrix, key_trace, states_h, x_t, ids_t, prev_ids_t, commit_t)
            return (states_h, x_t, ids_t, kv_matrix, key_trace), logits_y

        (final_h, final_x, final_ids, final_kv, final_key), logits_t = jax.lax.scan(
            step,
            (initial_h, state.prev_x, state.prev_ids, state.kv, state.key_trace),
            (tensors_x_t, input_ids_t, commit_gates_t),
        )
        state = state._replace(prev_x=final_x, prev_ids=final_ids, kv=final_kv, key_trace=final_key)
    elif kv_memory:
        def step(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], pair: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
            states_h, prev_ids_t, kv_matrix, key_trace = carry
            x_t, ids_t, commit_t = pair
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t) + kv_logits(kv_matrix, key_trace, states_h, x_t, ids_t)
            kv_matrix, key_trace = update_kv(kv_matrix, key_trace, states_h, x_t, ids_t, prev_ids_t, commit_t)
            return (states_h, ids_t, kv_matrix, key_trace), logits_y

        (final_h, final_ids, final_kv, final_key), logits_t = jax.lax.scan(
            step,
            (initial_h, state.prev_ids, state.kv, state.key_trace),
            (tensors_x_t, input_ids_t, commit_gates_t),
        )
        state = state._replace(prev_ids=final_ids, kv=final_kv, key_trace=final_key)
    elif token_memory and prev_skip:
        def step(
            carry: tuple[jax.Array, jax.Array, jax.Array], pair: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
            states_h, prev_x, memory_h = carry
            x_t, ids_t = pair
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t, prev_x) + memory_logits(memory_h, ids_t)
            memory_h = update_memory(memory_h, x_t, ids_t)
            return (states_h, x_t, memory_h), logits_y

        (final_h, final_x, final_memory), logits_t = jax.lax.scan(step, (initial_h, state.prev_x, state.memory), (tensors_x_t, input_ids_t))
        state = state._replace(prev_x=final_x, memory=final_memory)
    elif token_memory:
        def step(
            carry: tuple[jax.Array, jax.Array], pair: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            states_h, memory_h = carry
            x_t, ids_t = pair
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t) + memory_logits(memory_h, ids_t)
            memory_h = update_memory(memory_h, x_t, ids_t)
            return (states_h, memory_h), logits_y

        (final_h, final_memory), logits_t = jax.lax.scan(step, (initial_h, state.memory), (tensors_x_t, input_ids_t))
        state = state._replace(memory=final_memory)
    elif prev_skip:
        def step(
            carry: tuple[jax.Array, jax.Array], x_t: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            states_h, prev_x = carry
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t, prev_x)
            return (states_h, x_t), logits_y

        (final_h, final_x), logits_t = jax.lax.scan(step, (initial_h, state.prev_x), tensors_x_t)
        state = state._replace(prev_x=final_x)
    else:
        def step(states_h: jax.Array, x_t: jax.Array) -> tuple[jax.Array, jax.Array]:
            states_h = update_state(states_h, x_t)
            logits_y = output_logits(states_h, x_t)
            return states_h, logits_y

        final_h, logits_t = jax.lax.scan(step, initial_h, tensors_x_t)
    logits = jnp.swapaxes(logits_t, 0, 1)
    if "local_q" in params:
        from local_attention_ssm import local_attention
        residual, keys, values, count = local_attention(
            params, tensors_x, state.local_keys, state.local_values, state.local_count
        )
        logits = logits + (residual @ params["local_o"]) @ params["output_c"]
        state = state._replace(local_keys=keys, local_values=values, local_count=count)
    if "fact_embed" in params:
        from neural_fact_memory import apply as apply_facts
        logits, state = apply_facts(params, input_ids, logits, state, write_mask)
    if "open_embed" in params:
        from open_neural_memory import apply as apply_open
        logits, state = apply_open(params, input_ids, logits, state, write_mask)
    return logits, state._replace(h=final_h) if complete else final_h


def forward_chunks(
    params: dict[str, jax.Array], input_ids: jax.Array, state: SSMState, *, chunk_size: int, memory_write_mask: jax.Array | None = None
) -> tuple[jax.Array, SSMState]:
    """Differentiable chunked execution: no detach until the optimizer boundary."""
    if not isinstance(state, SSMState):
        raise ValueError("Chunked execution requires a complete SSMState")
    if chunk_size < 1 or input_ids.shape[1] < 1:
        raise ValueError("chunk_size and sequence length must be positive")
    outputs = []
    for start in range(0, input_ids.shape[1], chunk_size):
        logits, state = forward(params, input_ids[:, start:start + chunk_size], state, memory_write_mask=memory_write_mask)
        outputs.append(logits)
    return jnp.concatenate(outputs, axis=1), state


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


def language_cross_entropy(logits, target_ids, *, label_smoothing: float = 0.0):
    """Training-only target smoothing; zero is ordinary evaluation NLL."""
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be finite and in [0, 1)")
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.mean(jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1))
    if label_smoothing == 0.0:
        return nll
    return (1.0 - label_smoothing) * nll - label_smoothing * jnp.mean(log_probs)


def loss_fn(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    initial_h: jax.Array | SSMState,
    *,
    chunk_size: int = 0,
    label_smoothing: float = 0.0,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array | SSMState]]:
    if chunk_size:
        if not isinstance(initial_h, SSMState):
            raise ValueError("Chunked training requires a complete SSMState")
        logits, final_h = forward_chunks(params, input_ids, initial_h, chunk_size=chunk_size)
    else:
        logits, final_h = forward(params, input_ids, initial_h)
    loss = language_cross_entropy(logits, target_ids, label_smoothing=label_smoothing)
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


@partial(jax.jit, static_argnames=("learning_rate", "grad_clip", "memory_gate_l1", "chunk_size", "label_smoothing"))
def train_step(
    params: dict[str, jax.Array],
    opt_state: dict[str, object],
    input_ids: jax.Array,
    target_ids: jax.Array,
    states_h: jax.Array | SSMState,
    *,
    learning_rate: float,
    grad_clip: float,
    memory_gate_l1: float = 0.0,
    chunk_size: int = 0,
    label_smoothing: float = 0.0,
) -> tuple[dict[str, jax.Array], dict[str, object], jax.Array | SSMState, TrainMetrics]:
    def train_loss_fn(train_params: dict[str, jax.Array]) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        objective_loss, (accuracy, final_h) = loss_fn(train_params, input_ids, target_ids, states_h, chunk_size=chunk_size, label_smoothing=label_smoothing)
        penalty = memory_gate_penalty(train_params)
        total_loss = objective_loss + memory_gate_l1 * penalty
        return total_loss, (accuracy, final_h, objective_loss)

    (loss, (accuracy, final_h, objective_loss)), grads = jax.value_and_grad(train_loss_fn, has_aux=True)(params)
    grads, grad_norm = clip_grads(grads, grad_clip)
    params, opt_state = adam_update(params, grads, opt_state, learning_rate)
    final_h = jax.tree_util.tree_map(jax.lax.stop_gradient, final_h)
    metrics = TrainMetrics(
        loss=objective_loss,
        accuracy=accuracy,
        grad_norm=grad_norm,
        mean_decay=jnp.mean(decay_from_raw(params["decay_raw"])),
    )
    return params, opt_state, final_h, metrics


@partial(jax.jit, static_argnames=("seq_len", "transition_capacity", "occurrence_capacity", "unseen_capacity"))
def eval_loss(
    params: dict[str, jax.Array],
    streams: jax.Array,
    *,
    seq_len: int,
    frequency_prior: jax.Array | None = None,
    frequency_prior_tokens: float = 1024.0,
    frequency_epsilon: float = 0.05,
    transition_mask: jax.Array | None = None,
    transition_capacity: int = 256,
    transition_strength: float = 8.0,
    occurrence_cohort: jax.Array | None = None,
    occurrence_capacity: int = 256,
    occurrence_strength: float = 8.0,
    unseen_mask: jax.Array | None = None,
    unseen_cohort: jax.Array | None = None,
    unseen_capacity: int = 256,
    unseen_strength: float = 8.0,
) -> TrainMetrics:
    if (unseen_mask is None) != (unseen_cohort is None):
        raise ValueError("Unseen occurrence mask and cohorts must be supplied together")
    input_ids = streams[:, :seq_len]
    target_ids = streams[:, 1 : seq_len + 1]
    states_h = jnp.zeros((streams.shape[0], params["decay_raw"].shape[0]), dtype=jnp.float32)
    if frequency_prior is None and transition_mask is None and occurrence_cohort is None and unseen_mask is None:
        loss, (accuracy, _) = loss_fn(params, input_ids, target_ids, states_h)
    else:
        logits, _ = forward(params, input_ids, states_h)
        if frequency_prior is not None:
            history, seen = initial_frequency_state(frequency_prior, streams.shape[0], frequency_prior_tokens)
            logits, _, _ = causal_frequency_logits(logits, input_ids, history, seen, frequency_prior,
                                                   prior_tokens=frequency_prior_tokens, epsilon=frequency_epsilon)
        if transition_mask is not None:
            cache = initial_rare_transition_state(streams.shape[0], transition_capacity)
            logits, _ = rare_transition_logits(logits, input_ids, cache, transition_mask, prior_strength=transition_strength)
        if occurrence_cohort is not None:
            if transition_mask is None:
                raise ValueError("Occurrence correction requires a transition target mask")
            occurrence = initial_rare_occurrence_state(streams.shape[0], occurrence_capacity)
            logits, _ = rare_occurrence_logits(logits, input_ids, occurrence, transition_mask,
                                               occurrence_cohort, prior_strength=occurrence_strength)
        if unseen_mask is not None:
            unseen_state = initial_unseen_occurrence_state(streams.shape[0], unseen_capacity)
            logits, _ = unseen_occurrence_logits(logits, input_ids, unseen_state, unseen_mask,
                                                 unseen_cohort, prior_strength=unseen_strength)
        loss = language_cross_entropy(logits, target_ids)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == target_ids)
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
    *,
    frequency_prior: jax.Array | None = None,
    frequency_prior_tokens: float = 1024.0,
    frequency_epsilon: float = 0.05,
    transition_mask: jax.Array | None = None,
    transition_capacity: int = 256,
    transition_strength: float = 8.0,
    occurrence_cohort: jax.Array | None = None,
    occurrence_capacity: int = 256,
    occurrence_strength: float = 8.0,
    unseen_mask: jax.Array | None = None,
    unseen_cohort: jax.Array | None = None,
    unseen_capacity: int = 256,
    unseen_strength: float = 8.0,
) -> str:
    key = jax.random.PRNGKey(seed)
    unk_id = token_to_id[UNK]
    tokenizer_config = tokenizer_config or default_word_tokenizer_config()
    prompt_tokens = tokenize_with_config(prompt, tokenizer_config)
    prompt_ids = [token_to_id.get(token, unk_id) for token in prompt_tokens] or [token_to_id[BOS]]
    state_h = init_stream_state(params, 1)
    if frequency_prior is not None:
        history, seen = initial_frequency_state(frequency_prior, 1, frequency_prior_tokens)
    if transition_mask is not None:
        transition_state = initial_rare_transition_state(1, transition_capacity)

    if occurrence_cohort is not None:
        if transition_mask is None:
            raise ValueError("Occurrence correction requires a transition target mask")
        occurrence_state = initial_rare_occurrence_state(1, occurrence_capacity)

    if (unseen_mask is None) != (unseen_cohort is None):
        raise ValueError("Unseen occurrence mask and cohorts must be supplied together")
    if unseen_mask is not None:
        unseen_state = initial_unseen_occurrence_state(1, unseen_capacity)

    generated = list(prompt_tokens)
    current_id = prompt_ids[0]
    for token_id in prompt_ids:
        logits, state_h = forward(
            params,
            jnp.asarray([[token_id]], dtype=jnp.int32),
            state_h,
        )
        if frequency_prior is not None:
            logits, history, seen = causal_frequency_logits(logits, jnp.asarray([[token_id]], dtype=jnp.int32),
                history, seen, frequency_prior, prior_tokens=frequency_prior_tokens, epsilon=frequency_epsilon)
        if transition_mask is not None:
            logits, transition_state = rare_transition_logits(logits, jnp.asarray([[token_id]], dtype=jnp.int32),
                transition_state, transition_mask, prior_strength=transition_strength)
        if occurrence_cohort is not None:
            logits, occurrence_state = rare_occurrence_logits(logits, jnp.asarray([[token_id]], dtype=jnp.int32),
                occurrence_state, transition_mask, occurrence_cohort, prior_strength=occurrence_strength)
        if unseen_mask is not None:
            logits, unseen_state = unseen_occurrence_logits(logits, jnp.asarray([[token_id]], dtype=jnp.int32),
                unseen_state, unseen_mask, unseen_cohort, prior_strength=unseen_strength)
        current_id = token_id

    for _ in range(steps):
        next_logits = logits[0, -1] / max(temperature, 1.0e-6)
        if temperature <= 0.0:
            current_id = int(jnp.argmax(next_logits))
        else:
            key, sample_key = jax.random.split(key)
            current_id = int(jax.random.categorical(sample_key, next_logits))
        generated.append(vocab[current_id])
        logits, state_h = forward(
            params, jnp.asarray([[current_id]], dtype=jnp.int32), state_h
        )
        if frequency_prior is not None:
            logits, history, seen = causal_frequency_logits(logits, jnp.asarray([[current_id]], dtype=jnp.int32),
                history, seen, frequency_prior, prior_tokens=frequency_prior_tokens, epsilon=frequency_epsilon)
        if transition_mask is not None:
            logits, transition_state = rare_transition_logits(logits, jnp.asarray([[current_id]], dtype=jnp.int32),
                transition_state, transition_mask, prior_strength=transition_strength)

        if occurrence_cohort is not None:
            logits, occurrence_state = rare_occurrence_logits(logits, jnp.asarray([[current_id]], dtype=jnp.int32),
                occurrence_state, transition_mask, occurrence_cohort, prior_strength=occurrence_strength)

        if unseen_mask is not None:
            logits, unseen_state = unseen_occurrence_logits(logits, jnp.asarray([[current_id]], dtype=jnp.int32),
                unseen_state, unseen_mask, unseen_cohort, prior_strength=unseen_strength)

    return detokenize_with_config(generated, tokenizer_config)


def run(args: argparse.Namespace) -> None:
    if args.chunk_size < 0:
        raise ValueError("--chunk-size must be nonnegative")
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
    fitted = None
    if not args.load_checkpoint and tokenizer_config.get("type") == "bpe":
        fitted = fit_bpe_training_data(text, tokenizer_config, streams=args.streams,
                                      tokens_per_stream=args.tokens_per_stream, seq_len=args.seq_len)
        tokenizer_config = fitted["tokenizer_config"]
        tokens = fitted["train_tokens"]
    else:
        tokens = tokenize_with_config(text, tokenizer_config)
    args.vocab_coverage = fitted["coverage"] if fitted is not None else checkpoint_config.get("vocab_coverage")
    if len(tokens) < 16:
        raise SystemExit("Need at least 16 tokens to train the text SSM prototype.")

    if not args.load_checkpoint:
        if fitted is not None:
            token_to_id, vocab = fitted["token_to_id"], fitted["vocab"]
        else:
            token_to_id, vocab = build_vocab_for_tokenizer(tokens, tokenizer_config)
        params = init_params(
            key,
            len(vocab),
            args.input_dim,
            args.state_dim,
            variant=args.ssm_variant,
            skip_rank=args.ssm_skip_rank,
            attention_window=getattr(args, "ssm_attention_window", 128),
            attention_dim=getattr(args, "ssm_attention_dim", 32),
        )
        opt_state = init_adam(params)

    streams_np = (fitted["train_streams"] if fitted is not None else
                  make_streams(encode(tokens, token_to_id), args.streams, args.tokens_per_stream))
    streams = jnp.asarray(streams_np)

    if args.seq_len + 1 > args.tokens_per_stream:
        raise SystemExit("--seq-len must be smaller than --tokens-per-stream.")

    if getattr(args, "output_bias_init", "zero") == "train-unigram":
        if args.load_checkpoint or args.eval_only:
            raise ValueError("--output-bias-init train-unigram is only for fresh training")
        params = initialize_output_bias(params, streams_np, seq_len=args.seq_len,
                                       label_smoothing=args.label_smoothing)
        opt_state = init_adam(params)

    states_h = init_stream_state(params, args.streams)

    inference_kwargs = frequency_kwargs(checkpoint_config, len(vocab))
    args.causal_frequency_config = checkpoint_config.get("causal_frequency")
    args.rare_transition_config = checkpoint_config.get("rare_transition")
    args.rare_occurrence_config = checkpoint_config.get("rare_occurrence")
    args.unseen_occurrence_config = checkpoint_config.get("unseen_occurrence")
    baseline = eval_loss(params, streams, seq_len=args.seq_len, **inference_kwargs)
    jax.block_until_ready(baseline.loss)

    print("TextPy/SoA trainable SSM text prototype")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("SoA rails")
    print(f"  TOKEN_STREAMS: {tuple(streams.shape)} int32")
    print(f"  STATES_H:      {tuple(states_h.h.shape)} float32")
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
        states_h = init_stream_state(params, args.streams)
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
                chunk_size=args.chunk_size,
                label_smoothing=args.label_smoothing,
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
    final_eval = eval_loss(params, streams, seq_len=args.seq_len, **inference_kwargs)
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
            stream_state=states_h,
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
            **inference_kwargs,
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
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Internal differentiable portions per optimizer window; 0 means one portion.")
    parser.add_argument("--input-dim", type=int, default=48)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument(
        "--ssm-variant",
        choices=SSM_VARIANTS,
        default="static",
    )
    parser.add_argument("--ssm-skip-rank", type=int, default=DEFAULT_SKIP_RANK)
    parser.add_argument("--ssm-attention-window", type=int, default=128)
    parser.add_argument("--ssm-attention-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--learning-rate", type=float, default=0.012)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--memory-gate-l1", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Training-only uniform target smoothing; evaluation uses ordinary NLL.")
    parser.add_argument("--output-bias-init", choices=("zero", "train-unigram"), default="zero", help="Fresh training: initialize classifier bias from training targets only.")
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
    args = parser.parse_args()
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be finite and in [0, 1)")
    return args


if __name__ == "__main__":
    run(parse_args())
