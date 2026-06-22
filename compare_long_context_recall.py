#!/usr/bin/env python3
"""Compare SSM and context-capped Transformer on synthetic delayed recall.

The ordinary language-model benchmark rewards local statistics heavily. This
probe isolates the opposite regime: a key token appears early, many filler tokens
follow, and a later query must predict the original key. The SSM receives full
recurrent credit assignment across the record, while the Transformer is trained
with a causal attention cap that can be shorter than the key-query distance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
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

from compare_text_architectures import (
    choose_transformer_d_model,
    count_bytes,
    count_params,
    format_bytes,
    init_transformer_params,
    layer_norm,
)
from train_ssm_text import (
    SSM_VARIANTS,
    adam_update,
    clip_grads,
    decay_from_raw,
    forward as ssm_forward,
    init_adam,
    init_params as init_ssm_params,
    infer_ssm_variant,
    kv_read_gates,
)


PAD_TOKEN = 0
QUERY_TOKEN = 1
KEY_OFFSET = 2
ASSIGN_TOKEN = 2
SEP_TOKEN = 3
ASSIGN_NAME_OFFSET = 4
PY_DEF = 0
PY_FN = 1
PY_LPAREN = 2
PY_RPAREN = 3
PY_COLON = 4
PY_RETURN = 5
PY_PRINT = 6
PY_SYNTAX_TOKEN_COUNT = 7


def syntax_offset(key_count: int, value_count: int) -> int:
    return ASSIGN_NAME_OFFSET + key_count + value_count


def py_token(key_count: int, value_count: int, syntax_index: int) -> int:
    return syntax_offset(key_count, value_count) + syntax_index


def stable_index(text: str, modulo: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


class RecallMetrics(NamedTuple):
    loss: jax.Array
    accuracy: jax.Array
    all_token_loss: jax.Array
    all_token_accuracy: jax.Array
    grad_norm: jax.Array
    kv_read_gate_mean: jax.Array
    kv_read_gate_query_mean: jax.Array
    kv_read_gate_non_query_mean: jax.Array
    kv_read_gate_entropy_mean: jax.Array
    kv_read_gate_query_entropy_mean: jax.Array
    kv_read_gate_non_query_entropy_mean: jax.Array
    kv_read_gate_lt_0p01_frac: jax.Array
    kv_read_gate_query_gt_0p5_frac: jax.Array
    kv_read_gate_query_gt_0p9_frac: jax.Array
    kv_read_gate_non_query_lt_0p01_frac: jax.Array


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_delayed_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    key_indexes = rng.integers(0, key_count, size=records, dtype=np.int32)
    keys = key_indexes + KEY_OFFSET
    sequence = np.full((records, delay + 2), PAD_TOKEN, dtype=np.int32)
    sequence[:, 0] = keys
    sequence[:, delay] = QUERY_TOKEN
    sequence[:, delay + 1] = keys
    return sequence[:, :-1], sequence[:, 1:], key_indexes


def make_assignment_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    name_indexes = rng.integers(0, key_count, size=records, dtype=np.int32)
    distractor_offsets = rng.integers(1, key_count, size=records, dtype=np.int32)
    distractor_indexes = (name_indexes + distractor_offsets) % key_count
    value_indexes = rng.integers(0, value_count, size=records, dtype=np.int32)
    distractor_values = rng.integers(0, value_count, size=records, dtype=np.int32)
    name_tokens = name_indexes + ASSIGN_NAME_OFFSET
    distractor_name_tokens = distractor_indexes + ASSIGN_NAME_OFFSET
    value_offset = ASSIGN_NAME_OFFSET + key_count
    value_tokens = value_indexes + value_offset
    distractor_value_tokens = distractor_values + value_offset

    sequence = np.full((records, delay + 2), PAD_TOKEN, dtype=np.int32)
    # Code-like prefix: name = value ; other_name = other_value ; ... name ? value
    sequence[:, 0] = name_tokens
    sequence[:, 1] = ASSIGN_TOKEN
    sequence[:, 2] = value_tokens
    sequence[:, 3] = SEP_TOKEN
    sequence[:, 4] = distractor_name_tokens
    sequence[:, 5] = ASSIGN_TOKEN
    sequence[:, 6] = distractor_value_tokens
    sequence[:, 7] = SEP_TOKEN
    sequence[:, delay - 1] = name_tokens
    sequence[:, delay] = QUERY_TOKEN
    sequence[:, delay + 1] = value_tokens
    return sequence[:, :-1], sequence[:, 1:], value_indexes


def make_multi_assignment_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    binding_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if binding_count < 1:
        raise ValueError("binding_count must be positive.")
    if binding_count > key_count:
        raise ValueError("binding_count must not exceed key_count when unique names are required.")
    prefix_len = 4 * binding_count
    if delay < prefix_len + 1:
        raise ValueError(
            f"delay must be at least {prefix_len + 1} for {binding_count} bindings "
            "plus the final query name."
        )

    rng = np.random.default_rng(seed)
    name_indexes = np.empty((records, binding_count), dtype=np.int32)
    value_indexes = np.empty((records, binding_count), dtype=np.int32)
    unique_values = value_count >= binding_count
    for row in range(records):
        name_indexes[row] = rng.choice(key_count, size=binding_count, replace=False).astype(np.int32)
        value_indexes[row] = rng.choice(value_count, size=binding_count, replace=not unique_values).astype(np.int32)

    query_slots = rng.integers(0, binding_count, size=records, dtype=np.int32)
    row_indexes = np.arange(records)
    query_name_indexes = name_indexes[row_indexes, query_slots]
    query_value_indexes = value_indexes[row_indexes, query_slots]

    value_offset = ASSIGN_NAME_OFFSET + key_count
    sequence = np.full((records, delay + 2), PAD_TOKEN, dtype=np.int32)
    for slot in range(binding_count):
        base = 4 * slot
        sequence[:, base] = name_indexes[:, slot] + ASSIGN_NAME_OFFSET
        sequence[:, base + 1] = ASSIGN_TOKEN
        sequence[:, base + 2] = value_indexes[:, slot] + value_offset
        sequence[:, base + 3] = SEP_TOKEN

    sequence[:, delay - 1] = query_name_indexes + ASSIGN_NAME_OFFSET
    sequence[:, delay] = QUERY_TOKEN
    sequence[:, delay + 1] = query_value_indexes + value_offset
    return sequence[:, :-1], sequence[:, 1:], query_value_indexes


def make_multi_query_assignment_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    binding_count: int,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if binding_count < 1:
        raise ValueError("binding_count must be positive.")
    if query_count < 1:
        raise ValueError("query_count must be positive.")
    if binding_count > key_count:
        raise ValueError("binding_count must not exceed key_count when unique names are required.")
    if query_count > binding_count:
        raise ValueError("query_count must not exceed binding_count for unique queried names.")
    prefix_len = 4 * binding_count
    suffix_len = 4 * query_count
    if delay < prefix_len + 1:
        raise ValueError(
            f"delay must be at least {prefix_len + 1} for {binding_count} bindings "
            "before the first query."
        )

    rng = np.random.default_rng(seed)
    name_indexes = np.empty((records, binding_count), dtype=np.int32)
    value_indexes = np.empty((records, binding_count), dtype=np.int32)
    unique_values = value_count >= binding_count
    for row in range(records):
        name_indexes[row] = rng.choice(key_count, size=binding_count, replace=False).astype(np.int32)
        value_indexes[row] = rng.choice(value_count, size=binding_count, replace=not unique_values).astype(np.int32)

    query_slots = np.empty((records, query_count), dtype=np.int32)
    for row in range(records):
        query_slots[row] = rng.choice(binding_count, size=query_count, replace=False).astype(np.int32)
    row_indexes = np.arange(records)[:, None]
    query_name_indexes = name_indexes[row_indexes, query_slots]
    query_value_indexes = value_indexes[row_indexes, query_slots]

    value_offset = ASSIGN_NAME_OFFSET + key_count
    sequence = np.full((records, delay + suffix_len), PAD_TOKEN, dtype=np.int32)
    for slot in range(binding_count):
        base = 4 * slot
        sequence[:, base] = name_indexes[:, slot] + ASSIGN_NAME_OFFSET
        sequence[:, base + 1] = ASSIGN_TOKEN
        sequence[:, base + 2] = value_indexes[:, slot] + value_offset
        sequence[:, base + 3] = SEP_TOKEN
    for slot in range(query_count):
        base = delay - 1 + 4 * slot
        sequence[:, base] = query_name_indexes[:, slot] + ASSIGN_NAME_OFFSET
        sequence[:, base + 1] = QUERY_TOKEN
        sequence[:, base + 2] = query_value_indexes[:, slot] + value_offset
        sequence[:, base + 3] = SEP_TOKEN
    return sequence[:, :-1], sequence[:, 1:], query_value_indexes.reshape(-1)


def make_assignment_distractor_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if key_count < 2:
        raise ValueError("key_count must be at least 2 for assignment-distractor.")
    if value_count < 2:
        raise ValueError("value_count must be at least 2 for assignment-distractor.")
    if delay < 12:
        raise ValueError("delay must be at least 12 for --task assignment-distractor.")

    rng = np.random.default_rng(seed)
    name_indexes = rng.integers(0, key_count, size=records, dtype=np.int32)
    distractor_offsets = rng.integers(1, key_count, size=records, dtype=np.int32)
    distractor_indexes = (name_indexes + distractor_offsets) % key_count
    value_indexes = rng.integers(0, value_count, size=records, dtype=np.int32)
    wrong_offsets = rng.integers(1, value_count, size=records, dtype=np.int32)
    wrong_values = (value_indexes + wrong_offsets) % value_count
    other_values = rng.integers(0, value_count, size=records, dtype=np.int32)

    value_offset = ASSIGN_NAME_OFFSET + key_count
    name_tokens = name_indexes + ASSIGN_NAME_OFFSET
    distractor_name_tokens = distractor_indexes + ASSIGN_NAME_OFFSET

    sequence = np.full((records, delay + 8), PAD_TOKEN, dtype=np.int32)
    # Assignment, first read, non-assignment literal, second read. The literal
    # must not overwrite the key-value binding.
    sequence[:, 0] = name_tokens
    sequence[:, 1] = ASSIGN_TOKEN
    sequence[:, 2] = value_indexes + value_offset
    sequence[:, 3] = SEP_TOKEN
    sequence[:, 4] = distractor_name_tokens
    sequence[:, 5] = ASSIGN_TOKEN
    sequence[:, 6] = other_values + value_offset
    sequence[:, 7] = SEP_TOKEN
    sequence[:, delay - 1] = name_tokens
    sequence[:, delay] = QUERY_TOKEN
    sequence[:, delay + 1] = value_indexes + value_offset
    sequence[:, delay + 2] = SEP_TOKEN
    sequence[:, delay + 3] = wrong_values + value_offset
    sequence[:, delay + 4] = SEP_TOKEN
    sequence[:, delay + 5] = name_tokens
    sequence[:, delay + 6] = QUERY_TOKEN
    sequence[:, delay + 7] = value_indexes + value_offset
    return sequence[:, :-1], sequence[:, 1:], np.stack([value_indexes, value_indexes], axis=1).reshape(-1)


def make_assignment_alias_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if key_count < 3:
        raise ValueError("key_count must be at least 3 for assignment-alias.")
    if delay < 12:
        raise ValueError("delay must be at least 12 for --task assignment-alias.")

    rng = np.random.default_rng(seed)
    source_indexes = rng.integers(0, key_count, size=records, dtype=np.int32)
    alias_offsets = rng.integers(1, key_count, size=records, dtype=np.int32)
    alias_indexes = (source_indexes + alias_offsets) % key_count
    distractor_offsets = rng.integers(1, key_count, size=records, dtype=np.int32)
    distractor_indexes = (alias_indexes + distractor_offsets) % key_count
    same_as_source = distractor_indexes == source_indexes
    distractor_indexes = np.where(same_as_source, (distractor_indexes + 1) % key_count, distractor_indexes)
    value_indexes = rng.integers(0, value_count, size=records, dtype=np.int32)
    distractor_values = rng.integers(0, value_count, size=records, dtype=np.int32)

    value_offset = ASSIGN_NAME_OFFSET + key_count
    source_tokens = source_indexes + ASSIGN_NAME_OFFSET
    alias_tokens = alias_indexes + ASSIGN_NAME_OFFSET
    distractor_tokens = distractor_indexes + ASSIGN_NAME_OFFSET

    sequence = np.full((records, delay + 2), PAD_TOKEN, dtype=np.int32)
    # Alias/copy assignment: src = value ; alias = src ; other = value ; ... alias ? value
    sequence[:, 0] = source_tokens
    sequence[:, 1] = ASSIGN_TOKEN
    sequence[:, 2] = value_indexes + value_offset
    sequence[:, 3] = SEP_TOKEN
    sequence[:, 4] = alias_tokens
    sequence[:, 5] = ASSIGN_TOKEN
    sequence[:, 6] = source_tokens
    sequence[:, 7] = SEP_TOKEN
    sequence[:, 8] = distractor_tokens
    sequence[:, 9] = ASSIGN_TOKEN
    sequence[:, 10] = distractor_values + value_offset
    sequence[:, 11] = SEP_TOKEN
    sequence[:, delay - 1] = alias_tokens
    sequence[:, delay] = QUERY_TOKEN
    sequence[:, delay + 1] = value_indexes + value_offset
    return sequence[:, :-1], sequence[:, 1:], value_indexes


def make_mixed_code_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if key_count < 5:
        raise ValueError("key_count must be at least 5 for mixed-code.")
    if value_count < 4:
        raise ValueError("value_count must be at least 4 for mixed-code.")
    if query_count < 2:
        raise ValueError("query_count must be at least 2 for mixed-code.")
    if delay < 24:
        raise ValueError("delay must be at least 24 for --task mixed-code.")

    rng = np.random.default_rng(seed)
    value_offset = ASSIGN_NAME_OFFSET + key_count
    suffix_len = 4 * query_count
    sequence = np.full((records, delay + suffix_len), PAD_TOKEN, dtype=np.int32)
    targets = np.empty((records, query_count), dtype=np.int32)

    for row in range(records):
        names = rng.choice(key_count, size=5, replace=False).astype(np.int32)
        a, b, c, d, e = names.tolist()
        v0, v1, v2, v3 = rng.choice(value_count, size=4, replace=False).astype(np.int32)
        env = {a: v0, b: v1}
        prefix: list[int] = [
            a + ASSIGN_NAME_OFFSET,
            ASSIGN_TOKEN,
            v0 + value_offset,
            SEP_TOKEN,
            b + ASSIGN_NAME_OFFSET,
            ASSIGN_TOKEN,
            v1 + value_offset,
            SEP_TOKEN,
            c + ASSIGN_NAME_OFFSET,
            ASSIGN_TOKEN,
            a + ASSIGN_NAME_OFFSET,
            SEP_TOKEN,
        ]
        env[c] = env[a]
        prefix.extend(
            [
                d + ASSIGN_NAME_OFFSET,
                ASSIGN_TOKEN,
                v2 + value_offset,
                SEP_TOKEN,
            ]
        )
        env[d] = v2
        prefix.extend(
            [
                a + ASSIGN_NAME_OFFSET,
                ASSIGN_TOKEN,
                v3 + value_offset,
                SEP_TOKEN,
            ]
        )
        env[a] = v3
        prefix.extend(
            [
                e + ASSIGN_NAME_OFFSET,
                ASSIGN_TOKEN,
                c + ASSIGN_NAME_OFFSET,
                SEP_TOKEN,
            ]
        )
        env[e] = env[c]

        literal_noise = (env[e] + 1 + int(rng.integers(0, value_count - 1))) % value_count
        query_names = [c, a, e, b]
        rng.shuffle(query_names)
        query_names = query_names[:query_count]
        for index, token in enumerate(prefix[:delay]):
            sequence[row, index] = token
        for slot, name in enumerate(query_names):
            base = delay - 1 + 4 * slot
            if slot == 1:
                sequence[row, base - 1] = literal_noise + value_offset
            sequence[row, base] = name + ASSIGN_NAME_OFFSET
            sequence[row, base + 1] = QUERY_TOKEN
            sequence[row, base + 2] = env[name] + value_offset
            sequence[row, base + 3] = SEP_TOKEN
            targets[row, slot] = env[name]
    return sequence[:, :-1], sequence[:, 1:], targets.reshape(-1)


def make_generated_python_code_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if key_count < 6:
        raise ValueError("key_count must be at least 6 for generated-python-code.")
    if value_count < 5:
        raise ValueError("value_count must be at least 5 for generated-python-code.")
    if query_count < 2:
        raise ValueError("query_count must be at least 2 for generated-python-code.")
    if delay < 48:
        raise ValueError("delay must be at least 48 for --task generated-python-code.")

    rng = np.random.default_rng(seed)
    value_offset = ASSIGN_NAME_OFFSET + key_count
    def_token = py_token(key_count, value_count, PY_DEF)
    fn_token = py_token(key_count, value_count, PY_FN)
    lparen_token = py_token(key_count, value_count, PY_LPAREN)
    rparen_token = py_token(key_count, value_count, PY_RPAREN)
    colon_token = py_token(key_count, value_count, PY_COLON)
    return_token = py_token(key_count, value_count, PY_RETURN)
    print_token = py_token(key_count, value_count, PY_PRINT)

    suffix_len = 7 * query_count - 3
    sequence = np.full((records, delay + suffix_len), PAD_TOKEN, dtype=np.int32)
    targets = np.empty((records, query_count), dtype=np.int32)

    for row in range(records):
        names = rng.choice(key_count, size=6, replace=False).astype(np.int32)
        a, b, c, d, e, f = names.tolist()
        v0, v1, v2, v3, v4 = rng.choice(value_count, size=5, replace=False).astype(np.int32)
        env = {a: v0, b: v1}

        prefix: list[int] = [
            def_token,
            fn_token,
            lparen_token,
            rparen_token,
            colon_token,
            SEP_TOKEN,
            a + ASSIGN_NAME_OFFSET,
            ASSIGN_TOKEN,
            v0 + value_offset,
            SEP_TOKEN,
            b + ASSIGN_NAME_OFFSET,
            ASSIGN_TOKEN,
            v1 + value_offset,
            SEP_TOKEN,
            c + ASSIGN_NAME_OFFSET,
            ASSIGN_TOKEN,
            a + ASSIGN_NAME_OFFSET,
            SEP_TOKEN,
        ]
        env[c] = env[a]
        prefix.extend(
            [
                d + ASSIGN_NAME_OFFSET,
                ASSIGN_TOKEN,
                v2 + value_offset,
                SEP_TOKEN,
                a + ASSIGN_NAME_OFFSET,
                ASSIGN_TOKEN,
                v3 + value_offset,
                SEP_TOKEN,
            ]
        )
        env[d] = v2
        env[a] = v3
        prefix.extend(
            [
                e + ASSIGN_NAME_OFFSET,
                ASSIGN_TOKEN,
                c + ASSIGN_NAME_OFFSET,
                SEP_TOKEN,
            ]
        )
        env[e] = env[c]
        prefix.extend(
            [
                f + ASSIGN_NAME_OFFSET,
                ASSIGN_TOKEN,
                d + ASSIGN_NAME_OFFSET,
                SEP_TOKEN,
            ]
        )
        env[f] = env[d]

        literal_noise = (env[e] + 1 + int(rng.integers(0, value_count - 1))) % value_count
        prefix.extend(
            [
                print_token,
                lparen_token,
                literal_noise + value_offset,
                rparen_token,
                SEP_TOKEN,
                return_token,
                lparen_token,
                v4 + value_offset,
                rparen_token,
                SEP_TOKEN,
            ]
        )
        if len(prefix) > delay - 3:
            raise ValueError("generated-python-code prefix does not fit before the first query.")

        query_names = [c, a, e, b, d, f]
        rng.shuffle(query_names)
        query_names = query_names[:query_count]
        for index, token in enumerate(prefix):
            sequence[row, index] = token
        for slot, name in enumerate(query_names):
            base = delay - 3 + 7 * slot
            sequence[row, base] = print_token
            sequence[row, base + 1] = lparen_token
            sequence[row, base + 2] = name + ASSIGN_NAME_OFFSET
            sequence[row, base + 3] = QUERY_TOKEN
            sequence[row, base + 4] = env[name] + value_offset
            sequence[row, base + 5] = rparen_token
            sequence[row, base + 6] = SEP_TOKEN
            targets[row, slot] = env[name]
    return sequence[:, :-1], sequence[:, 1:], targets.reshape(-1)


def is_real_python_source(path: Path) -> bool:
    ignored_parts = {"artifacts", "__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
    return path.suffix == ".py" and not any(part in ignored_parts for part in path.parts)


def canonical_real_value(node: ast.AST, *, value_count: int) -> int | None:
    if isinstance(node, ast.Name):
        return None
    if isinstance(node, ast.Constant):
        label = f"const:{type(node.value).__name__}:{node.value!r}"
    elif isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            label = f"call:{func.id}"
        elif isinstance(func, ast.Attribute):
            label = f"callattr:{func.attr}"
        else:
            label = f"call:{type(func).__name__}"
    elif isinstance(node, ast.BinOp):
        label = f"binop:{type(node.op).__name__}"
    elif isinstance(node, ast.UnaryOp):
        label = f"unary:{type(node.op).__name__}"
    elif isinstance(node, ast.Compare):
        ops = ",".join(type(op).__name__ for op in node.ops)
        label = f"compare:{ops}"
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        label = f"literal:{type(node).__name__}:{len(getattr(node, 'elts', getattr(node, 'keys', [])))}"
    elif isinstance(node, ast.Subscript):
        label = "subscript"
    elif isinstance(node, ast.Attribute):
        label = f"attribute:{node.attr}"
    else:
        label = type(node).__name__
    return stable_index(label, value_count)


def extract_real_python_records(
    *,
    key_count: int,
    value_count: int,
    binding_count: int,
    query_count: int,
    seed: int,
    source_root: Path | None = None,
) -> list[tuple[list[int], list[int], list[int]]]:
    if key_count < 4:
        raise ValueError("key_count must be at least 4 for real-python-code.")
    if value_count < 4:
        raise ValueError("value_count must be at least 4 for real-python-code.")
    if binding_count < 2:
        raise ValueError("binding_count must be at least 2 for real-python-code.")
    if query_count < 1:
        raise ValueError("query_count must be positive for real-python-code.")

    root = source_root if source_root is not None else Path(__file__).resolve().parent
    records: list[tuple[list[int], list[int], list[int]]] = []
    value_offset = ASSIGN_NAME_OFFSET + key_count

    for path in sorted(root.rglob("*.py")):
        rel_path = path.relative_to(root)
        if not is_real_python_source(rel_path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fold = stable_index(f"{rel_path}:{node.name}:{getattr(node, 'lineno', 0)}", 2)
            if fold != seed % 2:
                continue
            name_to_index: dict[str, int] = {}
            env: dict[int, int] = {}
            prefix: list[int] = []
            assignments = [
                stmt
                for stmt in ast.walk(node)
                if isinstance(stmt, ast.Assign)
            ]
            assignments.sort(key=lambda stmt: (getattr(stmt, "lineno", 0), getattr(stmt, "col_offset", 0)))
            for stmt in assignments:
                if len(prefix) >= 4 * binding_count:
                    break
                if len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id not in name_to_index:
                    if len(name_to_index) >= key_count:
                        continue
                    name_to_index[target.id] = len(name_to_index)
                lhs = name_to_index[target.id]
                value_node = stmt.value
                if isinstance(value_node, ast.Name):
                    if value_node.id not in name_to_index:
                        if len(name_to_index) >= key_count:
                            continue
                        name_to_index[value_node.id] = len(name_to_index)
                    rhs_name = name_to_index[value_node.id]
                    if rhs_name not in env:
                        continue
                    value = env[rhs_name]
                    rhs_token = rhs_name + ASSIGN_NAME_OFFSET
                else:
                    value = canonical_real_value(value_node, value_count=value_count)
                    if value is None:
                        continue
                    rhs_token = value + value_offset
                env[lhs] = value
                prefix.extend([lhs + ASSIGN_NAME_OFFSET, ASSIGN_TOKEN, rhs_token, SEP_TOKEN])
            if len(prefix) < 4 * binding_count:
                continue
            queryable = sorted(env)
            if len(queryable) < query_count:
                continue
            rng = np.random.default_rng(seed + stable_index(f"{rel_path}:{node.name}", 1_000_003))
            query_slots = rng.choice(len(queryable), size=query_count, replace=False)
            query_names = [queryable[int(slot)] for slot in query_slots]
            targets = [env[name] for name in query_names]
            records.append((prefix[: 4 * binding_count], [int(name) for name in query_names], [int(value) for value in targets]))
    return records


def make_real_python_code_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    binding_count: int,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix_len = 4 * binding_count
    suffix_len = 4 * query_count
    if delay < prefix_len + 1:
        raise ValueError(f"delay must be at least {prefix_len + 1} for --task real-python-code.")

    extracted = extract_real_python_records(
        key_count=key_count,
        value_count=value_count,
        binding_count=binding_count,
        query_count=query_count,
        seed=seed,
    )
    if not extracted:
        raise ValueError("No real Python code records could be extracted from this repository.")

    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(extracted), size=records, dtype=np.int32)
    value_offset = ASSIGN_NAME_OFFSET + key_count
    sequence = np.full((records, delay + suffix_len), PAD_TOKEN, dtype=np.int32)
    target_values = np.empty((records, query_count), dtype=np.int32)

    for row, index in enumerate(indexes):
        prefix, query_names, targets = extracted[int(index)]
        for pos, token in enumerate(prefix[:delay]):
            sequence[row, pos] = token
        for slot, value in enumerate(targets):
            name = query_names[slot % len(query_names)]
            base = delay - 1 + 4 * slot
            sequence[row, base] = name + ASSIGN_NAME_OFFSET
            sequence[row, base + 1] = QUERY_TOKEN
            sequence[row, base + 2] = value + value_offset
            sequence[row, base + 3] = SEP_TOKEN
            target_values[row, slot] = value
    return sequence[:, :-1], sequence[:, 1:], target_values.reshape(-1)


def make_assignment_update_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if key_count < 2:
        raise ValueError("key_count must be at least 2 for assignment-update distractors.")
    if value_count < 2:
        raise ValueError("value_count must be at least 2 for assignment-update old/new values.")
    if delay < 12:
        raise ValueError("delay must be at least 12 for --task assignment-update.")

    rng = np.random.default_rng(seed)
    name_indexes = rng.integers(0, key_count, size=records, dtype=np.int32)
    distractor_offsets = rng.integers(1, key_count, size=records, dtype=np.int32)
    distractor_indexes = (name_indexes + distractor_offsets) % key_count
    old_values = rng.integers(0, value_count, size=records, dtype=np.int32)
    update_offsets = rng.integers(1, value_count, size=records, dtype=np.int32)
    new_values = (old_values + update_offsets) % value_count
    distractor_values = rng.integers(0, value_count, size=records, dtype=np.int32)

    value_offset = ASSIGN_NAME_OFFSET + key_count
    name_tokens = name_indexes + ASSIGN_NAME_OFFSET
    distractor_name_tokens = distractor_indexes + ASSIGN_NAME_OFFSET

    sequence = np.full((records, delay + 2), PAD_TOKEN, dtype=np.int32)
    # Code-like update: name = old ; other = value ; name = new ; ... name ? new
    sequence[:, 0] = name_tokens
    sequence[:, 1] = ASSIGN_TOKEN
    sequence[:, 2] = old_values + value_offset
    sequence[:, 3] = SEP_TOKEN
    sequence[:, 4] = distractor_name_tokens
    sequence[:, 5] = ASSIGN_TOKEN
    sequence[:, 6] = distractor_values + value_offset
    sequence[:, 7] = SEP_TOKEN
    sequence[:, 8] = name_tokens
    sequence[:, 9] = ASSIGN_TOKEN
    sequence[:, 10] = new_values + value_offset
    sequence[:, delay - 1] = name_tokens
    sequence[:, delay] = QUERY_TOKEN
    sequence[:, delay + 1] = new_values + value_offset
    return sequence[:, :-1], sequence[:, 1:], new_values


def make_mqar_recall(
    *,
    records: int,
    delay: int,
    key_count: int,
    binding_count: int,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical Multi-Query Associative Recall (MQAR).

    Shared vocabulary for keys and values (tokens 1..key_count; 0 is PAD/filler),
    no split name/value ranges, and no dedicated query-marker token: a query is
    simply a previously-seen key whose bound value must be emitted at the next
    position. Key-value pairs are scattered with random gaps in the first half of
    the sequence so every binding sits well before the query region; the second
    half is filler, then the query pairs follow. The third return value is an
    explicit float scoring mask (teacher-forcing bookkeeping, NOT a token the
    model sees) marking the query-key positions.
    """
    if binding_count < 1:
        raise ValueError("binding_count must be positive for mqar.")
    if query_count < 1:
        raise ValueError("query_count must be positive for mqar.")
    if query_count > binding_count:
        raise ValueError("query_count must not exceed binding_count for mqar.")
    if binding_count > key_count:
        raise ValueError("binding_count must not exceed key_count for unique mqar keys.")
    if delay < 4 * binding_count:
        raise ValueError(f"delay must be at least {4 * binding_count} for {binding_count} mqar pairs.")

    rng = np.random.default_rng(seed)
    symbol_count = key_count  # shared alphabet, tokens 1..key_count
    total_len = delay + 2 * query_count
    sequence = np.full((records, total_len), PAD_TOKEN, dtype=np.int32)

    # 2-slot write windows (even starts => adjacent k,v with no overlap) confined
    # to the first half so the latest binding still precedes the query region by
    # roughly delay/2 tokens, keeping it beyond the Transformer context cap.
    candidate_starts = np.arange(0, delay // 2, 2)
    if len(candidate_starts) < binding_count:
        raise ValueError("delay too small to scatter the requested mqar pairs.")

    for row in range(records):
        keys = rng.choice(symbol_count, size=binding_count, replace=False) + 1
        values = rng.integers(0, symbol_count, size=binding_count, dtype=np.int32) + 1
        starts = np.sort(rng.choice(candidate_starts, size=binding_count, replace=False))
        for k, v, s in zip(keys, values, starts):
            sequence[row, s] = int(k)
            sequence[row, s + 1] = int(v)
        q_slots = rng.choice(binding_count, size=query_count, replace=False)
        for j, slot in enumerate(q_slots):
            base = delay + 2 * j
            sequence[row, base] = int(keys[slot])       # query = the key itself, no marker
            sequence[row, base + 1] = int(values[slot])  # teacher-forced answer (next token)

    inputs = sequence[:, :-1]
    targets = sequence[:, 1:]
    mask = np.zeros_like(inputs, dtype=np.float32)
    for j in range(query_count):
        mask[:, delay + 2 * j] = 1.0
    return inputs, targets, mask


def make_task_data(
    *,
    task: str,
    records: int,
    delay: int,
    key_count: int,
    value_count: int,
    binding_count: int,
    query_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if task == "delayed-key":
        return make_delayed_recall(records=records, delay=delay, key_count=key_count, seed=seed)
    if task == "assignment":
        return make_assignment_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            value_count=value_count,
            seed=seed,
        )
    if task == "multi-assignment":
        return make_multi_assignment_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            value_count=value_count,
            binding_count=binding_count,
            seed=seed,
        )
    if task == "multi-query-assignment":
        return make_multi_query_assignment_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            value_count=value_count,
            binding_count=binding_count,
            query_count=query_count,
            seed=seed,
        )
    if task == "assignment-distractor":
        return make_assignment_distractor_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            value_count=value_count,
            seed=seed,
        )
    if task == "assignment-alias":
        return make_assignment_alias_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            value_count=value_count,
            seed=seed,
        )
    if task == "mixed-code":
        return make_mixed_code_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            value_count=value_count,
            query_count=query_count,
            seed=seed,
        )
    if task == "generated-python-code":
        return make_generated_python_code_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            value_count=value_count,
            query_count=query_count,
            seed=seed,
        )
    if task == "real-python-code":
        return make_real_python_code_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            value_count=value_count,
            binding_count=binding_count,
            query_count=query_count,
            seed=seed,
        )
    if task == "assignment-update":
        return make_assignment_update_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            value_count=value_count,
            seed=seed,
        )
    if task == "mqar":
        return make_mqar_recall(
            records=records,
            delay=delay,
            key_count=key_count,
            binding_count=binding_count,
            query_count=query_count,
            seed=seed,
        )
    raise ValueError(f"Unknown task: {task!r}")


def vocab_size_for_task(task: str, key_count: int, value_count: int) -> int:
    if task == "delayed-key":
        return KEY_OFFSET + key_count
    if task in {
        "assignment",
        "multi-assignment",
        "multi-query-assignment",
        "assignment-distractor",
        "assignment-alias",
        "mixed-code",
        "assignment-update",
    }:
        return ASSIGN_NAME_OFFSET + key_count + value_count
    if task == "generated-python-code":
        return ASSIGN_NAME_OFFSET + key_count + value_count + PY_SYNTAX_TOKEN_COUNT
    if task == "real-python-code":
        return ASSIGN_NAME_OFFSET + key_count + value_count
    if task == "mqar":
        return 1 + key_count  # PAD/filler + shared key/value alphabet
    raise ValueError(f"Unknown task: {task!r}")


def init_delayed_recall_memory_params(
    params: dict[str, jax.Array],
    *,
    key_count: int,
    write_logit: float,
    closed_logit: float,
    read_logit: float,
    memory_decay_raw: float,
    memory_logit_scale: float,
) -> dict[str, jax.Array]:
    if "memory_write_token" not in params or "memory_read_token" not in params:
        return params
    vocab_size = int(params["memory_write_token"].shape[0])
    write = jnp.full((vocab_size,), closed_logit, dtype=jnp.float32)
    read = jnp.full((vocab_size,), closed_logit, dtype=jnp.float32)
    key_tokens = jnp.arange(KEY_OFFSET, KEY_OFFSET + key_count)
    write = write.at[key_tokens].set(write_logit)
    read = read.at[QUERY_TOKEN].set(read_logit)
    params = dict(params)
    params["memory_write_token"] = write
    params["memory_read_token"] = read
    params["memory_decay_raw"] = jnp.asarray(memory_decay_raw, dtype=jnp.float32)
    params["memory_logit_scale"] = jnp.asarray(memory_logit_scale, dtype=jnp.float32)
    return params


def init_assignment_kv_memory_params(
    params: dict[str, jax.Array],
    *,
    key_count: int,
    value_count: int,
    key_logit: float,
    value_logit: float,
    erase_logit: float,
    query_logit: float,
    closed_logit: float,
    kv_decay_raw: float,
    kv_logit_scale: float,
) -> dict[str, jax.Array]:
    if "kv_key_token" not in params or "kv_value_token" not in params or "kv_query_token" not in params:
        return params
    vocab_size = int(params["kv_key_token"].shape[0])
    key = jnp.full((vocab_size,), closed_logit, dtype=jnp.float32)
    value = jnp.full((vocab_size,), closed_logit, dtype=jnp.float32)
    erase = jnp.full((vocab_size,), closed_logit, dtype=jnp.float32)
    write_context = jnp.full((vocab_size,), closed_logit, dtype=jnp.float32)
    deref = jnp.full((vocab_size,), closed_logit, dtype=jnp.float32)
    query = jnp.full((vocab_size,), closed_logit, dtype=jnp.float32)
    name_tokens = jnp.arange(ASSIGN_NAME_OFFSET, ASSIGN_NAME_OFFSET + key_count)
    value_tokens = jnp.arange(ASSIGN_NAME_OFFSET + key_count, ASSIGN_NAME_OFFSET + key_count + value_count)
    key = key.at[name_tokens].set(key_logit)
    deref = deref.at[name_tokens].set(value_logit)
    value = value.at[value_tokens].set(value_logit)
    erase = erase.at[value_tokens].set(erase_logit)
    write_context = write_context.at[ASSIGN_TOKEN].set(value_logit)
    query = query.at[QUERY_TOKEN].set(query_logit)
    params = dict(params)
    params["kv_key_token"] = key
    params["kv_value_token"] = value
    if "kv_erase_token" in params:
        params["kv_erase_token"] = erase
    if "kv_write_context_token" in params:
        params["kv_write_context_token"] = write_context
    if "kv_deref_token" in params:
        params["kv_deref_token"] = deref
    params["kv_query_token"] = query
    params["kv_decay_raw"] = jnp.asarray(kv_decay_raw, dtype=jnp.float32)
    params["kv_logit_scale"] = jnp.asarray(kv_logit_scale, dtype=jnp.float32)
    return params


def recall_loss_and_accuracy(
    logits: jax.Array,
    input_ids: jax.Array,
    target_ids: jax.Array,
    query_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1)[..., 0]
    predictions = jnp.argmax(logits, axis=-1).astype(target_ids.dtype)
    correct = predictions == target_ids
    if query_mask is None:
        query_mask = (input_ids == QUERY_TOKEN).astype(jnp.float32)
    else:
        query_mask = query_mask.astype(jnp.float32)
    query_count = jnp.maximum(1.0, jnp.sum(query_mask))
    recall_loss = jnp.sum(nll * query_mask) / query_count
    recall_accuracy = jnp.sum(correct.astype(jnp.float32) * query_mask) / query_count
    return recall_loss, recall_accuracy, jnp.mean(nll), jnp.mean(correct)


def ssm_recall_loss(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    *,
    kv_read_gate_l1: float = 0.0,
    kv_read_gate_target: float = 0.0,
    kv_read_gate_target_weight: float = 0.0,
    kv_read_gate_entropy_weight: float = 0.0,
    kv_read_gate_query_margin: float = 0.0,
    kv_read_gate_query_margin_weight: float = 0.0,
    kv_read_gate_hard_threshold: float | None = None,
    query_mask: jax.Array | None = None,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    initial_h = jnp.zeros((input_ids.shape[0], params["decay_raw"].shape[0]), dtype=jnp.float32)
    logits, _ = ssm_forward(
        params,
        input_ids,
        initial_h,
        kv_read_gate_hard_threshold=kv_read_gate_hard_threshold,
    )
    if query_mask is None:
        query_mask = (input_ids == QUERY_TOKEN).astype(jnp.float32)
    else:
        query_mask = query_mask.astype(jnp.float32)
    loss, accuracy, all_loss, all_accuracy = recall_loss_and_accuracy(logits, input_ids, target_ids, query_mask)
    gates = kv_read_gates(
        params,
        input_ids,
        initial_h,
        kv_read_gate_hard_threshold=kv_read_gate_hard_threshold,
    )
    non_query_mask = 1.0 - query_mask
    query_count = jnp.maximum(1.0, jnp.sum(query_mask))
    non_query_count = jnp.maximum(1.0, jnp.sum(non_query_mask))
    gate_mean = jnp.mean(gates)
    query_gate_mean = jnp.sum(gates * query_mask) / query_count
    non_query_gate_mean = jnp.sum(gates * non_query_mask) / non_query_count
    clipped_gates = jnp.clip(gates, 1.0e-6, 1.0 - 1.0e-6)
    gate_entropy = -(
        clipped_gates * jnp.log(clipped_gates)
        + (1.0 - clipped_gates) * jnp.log(1.0 - clipped_gates)
    )
    gate_entropy_mean = jnp.mean(gate_entropy)
    query_gate_entropy_mean = jnp.sum(gate_entropy * query_mask) / query_count
    non_query_gate_entropy_mean = jnp.sum(gate_entropy * non_query_mask) / non_query_count
    gate_lt_0p01_frac = jnp.mean((gates < 0.01).astype(jnp.float32))
    query_gate_gt_0p5_frac = jnp.sum((gates > 0.5).astype(jnp.float32) * query_mask) / query_count
    query_gate_gt_0p9_frac = jnp.sum((gates > 0.9).astype(jnp.float32) * query_mask) / query_count
    non_query_gate_lt_0p01_frac = jnp.sum((gates < 0.01).astype(jnp.float32) * non_query_mask) / non_query_count
    target_penalty = jnp.square(gate_mean - jnp.asarray(kv_read_gate_target, dtype=jnp.float32))
    query_margin_penalty = (
        jnp.sum(jnp.square(jnp.maximum(kv_read_gate_query_margin - gates, 0.0)) * query_mask)
        / query_count
    )
    total_loss = (
        loss
        + kv_read_gate_l1 * gate_mean
        + kv_read_gate_target_weight * target_penalty
        + kv_read_gate_entropy_weight * gate_entropy_mean
        + kv_read_gate_query_margin_weight * query_margin_penalty
    )
    return total_loss, (
        loss,
        accuracy,
        all_loss,
        all_accuracy,
        gate_mean,
        query_gate_mean,
        non_query_gate_mean,
        gate_entropy_mean,
        query_gate_entropy_mean,
        non_query_gate_entropy_mean,
        gate_lt_0p01_frac,
        query_gate_gt_0p5_frac,
        query_gate_gt_0p9_frac,
        non_query_gate_lt_0p01_frac,
    )


@jax.jit
def ssm_eval(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    query_mask: jax.Array | None = None,
) -> RecallMetrics:
    _, (
        loss,
        accuracy,
        all_loss,
        all_accuracy,
        gate_mean,
        query_gate_mean,
        non_query_gate_mean,
        gate_entropy_mean,
        query_gate_entropy_mean,
        non_query_gate_entropy_mean,
        gate_lt_0p01_frac,
        query_gate_gt_0p5_frac,
        query_gate_gt_0p9_frac,
        non_query_gate_lt_0p01_frac,
    ) = ssm_recall_loss(
        params,
        input_ids,
        target_ids,
        kv_read_gate_l1=0.0,
        kv_read_gate_target=0.0,
        kv_read_gate_target_weight=0.0,
        kv_read_gate_entropy_weight=0.0,
        kv_read_gate_query_margin=0.0,
        kv_read_gate_query_margin_weight=0.0,
        kv_read_gate_hard_threshold=None,
        query_mask=query_mask,
    )
    return RecallMetrics(
        loss,
        accuracy,
        all_loss,
        all_accuracy,
        jnp.asarray(0.0, dtype=jnp.float32),
        gate_mean,
        query_gate_mean,
        non_query_gate_mean,
        gate_entropy_mean,
        query_gate_entropy_mean,
        non_query_gate_entropy_mean,
        gate_lt_0p01_frac,
        query_gate_gt_0p5_frac,
        query_gate_gt_0p9_frac,
        non_query_gate_lt_0p01_frac,
    )


@jax.jit
def ssm_confusion(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    query_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    initial_h = jnp.zeros((input_ids.shape[0], params["decay_raw"].shape[0]), dtype=jnp.float32)
    logits, _ = ssm_forward(params, input_ids, initial_h)
    predictions = jnp.argmax(logits, axis=-1).astype(target_ids.dtype)
    out_mask = (input_ids == QUERY_TOKEN) if query_mask is None else (query_mask > 0.5)
    return predictions, target_ids, out_mask


def ssm_confusion_hard(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    *,
    kv_read_gate_hard_threshold: float,
    query_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    initial_h = jnp.zeros((input_ids.shape[0], params["decay_raw"].shape[0]), dtype=jnp.float32)
    logits, _ = ssm_forward(
        params,
        input_ids,
        initial_h,
        kv_read_gate_hard_threshold=kv_read_gate_hard_threshold,
    )
    predictions = jnp.argmax(logits, axis=-1).astype(target_ids.dtype)
    out_mask = (input_ids == QUERY_TOKEN) if query_mask is None else (query_mask > 0.5)
    return predictions, target_ids, out_mask


ssm_confusion_hard = jax.jit(ssm_confusion_hard, static_argnames=("kv_read_gate_hard_threshold",))


def transformer_forward_limited(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    *,
    layers: int,
    heads: int,
    context: int,
) -> jax.Array:
    batch, seq_len = input_ids.shape
    d_model = params["token_embed"].shape[1]
    head_dim = d_model // heads
    x = params["token_embed"][input_ids] + params["pos_embed"][:seq_len][None, :, :]
    positions = jnp.arange(seq_len)
    distance = positions[:, None] - positions[None, :]
    causal_mask = (distance >= 0) & (distance < context)
    causal_mask = causal_mask[None, None, :, :]

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


def transformer_recall_loss(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    *,
    layers: int,
    heads: int,
    context: int,
    query_mask: jax.Array | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    logits = transformer_forward_limited(params, input_ids, layers=layers, heads=heads, context=context)
    loss, accuracy, all_loss, all_accuracy = recall_loss_and_accuracy(logits, input_ids, target_ids, query_mask)
    return loss, (accuracy, all_loss, all_accuracy)


def transformer_confusion(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    *,
    layers: int,
    heads: int,
    context: int,
    query_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    logits = transformer_forward_limited(params, input_ids, layers=layers, heads=heads, context=context)
    predictions = jnp.argmax(logits, axis=-1).astype(target_ids.dtype)
    out_mask = (input_ids == QUERY_TOKEN) if query_mask is None else (query_mask > 0.5)
    return predictions, target_ids, out_mask


transformer_confusion = jax.jit(transformer_confusion, static_argnames=("layers", "heads", "context"))


def transformer_eval(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    *,
    layers: int,
    heads: int,
    context: int,
    query_mask: jax.Array | None = None,
) -> RecallMetrics:
    loss, (accuracy, all_loss, all_accuracy) = transformer_recall_loss(
        params,
        input_ids,
        target_ids,
        layers=layers,
        heads=heads,
        context=context,
        query_mask=query_mask,
    )
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return RecallMetrics(
        loss,
        accuracy,
        all_loss,
        all_accuracy,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    )


transformer_eval = jax.jit(transformer_eval, static_argnames=("layers", "heads", "context"))


def ssm_train_step(
    params: dict[str, jax.Array],
    opt_state: dict[str, object],
    input_ids: jax.Array,
    target_ids: jax.Array,
    *,
    learning_rate: float,
    grad_clip: float,
    kv_read_gate_l1: float,
    kv_read_gate_target: float,
    kv_read_gate_target_weight: float,
    kv_read_gate_entropy_weight: float,
    kv_read_gate_query_margin: float,
    kv_read_gate_query_margin_weight: float,
    query_mask: jax.Array | None = None,
) -> tuple[dict[str, jax.Array], dict[str, object], RecallMetrics]:
    (_, (
        pure_loss,
        accuracy,
        all_loss,
        all_accuracy,
        gate_mean,
        query_gate_mean,
        non_query_gate_mean,
        gate_entropy_mean,
        query_gate_entropy_mean,
        non_query_gate_entropy_mean,
        gate_lt_0p01_frac,
        query_gate_gt_0p5_frac,
        query_gate_gt_0p9_frac,
        non_query_gate_lt_0p01_frac,
    )), grads = jax.value_and_grad(
        ssm_recall_loss,
        has_aux=True,
    )(
        params,
        input_ids,
        target_ids,
        kv_read_gate_l1=kv_read_gate_l1,
        kv_read_gate_target=kv_read_gate_target,
        kv_read_gate_target_weight=kv_read_gate_target_weight,
        kv_read_gate_entropy_weight=kv_read_gate_entropy_weight,
        kv_read_gate_query_margin=kv_read_gate_query_margin,
        kv_read_gate_query_margin_weight=kv_read_gate_query_margin_weight,
        query_mask=query_mask,
    )
    grads, grad_norm = clip_grads(grads, grad_clip)
    params, opt_state = adam_update(params, grads, opt_state, learning_rate)
    return params, opt_state, RecallMetrics(
        pure_loss,
        accuracy,
        all_loss,
        all_accuracy,
        grad_norm,
        gate_mean,
        query_gate_mean,
        non_query_gate_mean,
        gate_entropy_mean,
        query_gate_entropy_mean,
        non_query_gate_entropy_mean,
        gate_lt_0p01_frac,
        query_gate_gt_0p5_frac,
        query_gate_gt_0p9_frac,
        non_query_gate_lt_0p01_frac,
    )


ssm_train_step = jax.jit(ssm_train_step, static_argnames=("learning_rate", "grad_clip"))


def ssm_eval_hard(
    params: dict[str, jax.Array],
    input_ids: jax.Array,
    target_ids: jax.Array,
    *,
    kv_read_gate_hard_threshold: float,
    query_mask: jax.Array | None = None,
) -> RecallMetrics:
    _, (
        loss,
        accuracy,
        all_loss,
        all_accuracy,
        gate_mean,
        query_gate_mean,
        non_query_gate_mean,
        gate_entropy_mean,
        query_gate_entropy_mean,
        non_query_gate_entropy_mean,
        gate_lt_0p01_frac,
        query_gate_gt_0p5_frac,
        query_gate_gt_0p9_frac,
        non_query_gate_lt_0p01_frac,
    ) = ssm_recall_loss(
        params,
        input_ids,
        target_ids,
        kv_read_gate_l1=0.0,
        kv_read_gate_target=0.0,
        kv_read_gate_target_weight=0.0,
        kv_read_gate_entropy_weight=0.0,
        kv_read_gate_query_margin=0.0,
        kv_read_gate_query_margin_weight=0.0,
        kv_read_gate_hard_threshold=kv_read_gate_hard_threshold,
        query_mask=query_mask,
    )
    return RecallMetrics(
        loss,
        accuracy,
        all_loss,
        all_accuracy,
        jnp.asarray(0.0, dtype=jnp.float32),
        gate_mean,
        query_gate_mean,
        non_query_gate_mean,
        gate_entropy_mean,
        query_gate_entropy_mean,
        non_query_gate_entropy_mean,
        gate_lt_0p01_frac,
        query_gate_gt_0p5_frac,
        query_gate_gt_0p9_frac,
        non_query_gate_lt_0p01_frac,
    )


ssm_eval_hard = jax.jit(ssm_eval_hard, static_argnames=("kv_read_gate_hard_threshold",))


def scheduled_kv_read_gate_l1(args: argparse.Namespace, *, update_index: int, batches_per_epoch: int) -> float:
    if args.kv_read_gate_l1_start is None:
        return float(args.kv_read_gate_l1)
    ramp_updates = int(args.kv_read_gate_l1_ramp_epochs) * int(batches_per_epoch)
    if ramp_updates <= 1:
        progress = 1.0
    else:
        progress = min(1.0, update_index / float(ramp_updates - 1))
    start = float(args.kv_read_gate_l1_start)
    end = float(args.kv_read_gate_l1)
    return start + (end - start) * progress


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
    context: int,
    query_mask: jax.Array | None = None,
) -> tuple[dict[str, jax.Array], dict[str, object], RecallMetrics]:
    (loss, (accuracy, all_loss, all_accuracy)), grads = jax.value_and_grad(
        transformer_recall_loss,
        has_aux=True,
    )(
        params,
        input_ids,
        target_ids,
        layers=layers,
        heads=heads,
        context=context,
        query_mask=query_mask,
    )
    grads, grad_norm = clip_grads(grads, grad_clip)
    params, opt_state = adam_update(params, grads, opt_state, learning_rate)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return params, opt_state, RecallMetrics(
        loss,
        accuracy,
        all_loss,
        all_accuracy,
        grad_norm,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    )


transformer_train_step = jax.jit(
    transformer_train_step,
    static_argnames=("learning_rate", "grad_clip", "layers", "heads", "context"),
)


def train_model(
    *,
    params: dict[str, jax.Array],
    train_inputs: jax.Array,
    train_targets: jax.Array,
    args: argparse.Namespace,
    model: str,
    train_query_mask: jax.Array | None = None,
) -> tuple[dict[str, jax.Array], dict[str, object], float, RecallMetrics]:
    opt_state = init_adam(params)
    records = train_inputs.shape[0]
    batches = math.ceil(records / args.batch_size)
    metrics = RecallMetrics(
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    start = time.perf_counter()
    update_index = 0
    for _ in range(args.epochs):
        for batch_index in range(batches):
            start_row = batch_index * args.batch_size
            end_row = min(records, start_row + args.batch_size)
            input_batch = train_inputs[start_row:end_row]
            target_batch = train_targets[start_row:end_row]
            mask_batch = None if train_query_mask is None else train_query_mask[start_row:end_row]
            if model == "ssm":
                kv_read_gate_l1 = scheduled_kv_read_gate_l1(
                    args,
                    update_index=update_index,
                    batches_per_epoch=batches,
                )
                params, opt_state, metrics = ssm_train_step(
                    params,
                    opt_state,
                    input_batch,
                    target_batch,
                    learning_rate=args.learning_rate,
                    grad_clip=args.grad_clip,
                    kv_read_gate_l1=kv_read_gate_l1,
                    kv_read_gate_target=args.kv_read_gate_target,
                    kv_read_gate_target_weight=args.kv_read_gate_target_weight,
                    kv_read_gate_entropy_weight=args.kv_read_gate_entropy_weight,
                    kv_read_gate_query_margin=args.kv_read_gate_query_margin,
                    kv_read_gate_query_margin_weight=args.kv_read_gate_query_margin_weight,
                    query_mask=mask_batch,
                )
            else:
                params, opt_state, metrics = transformer_train_step(
                    params,
                    opt_state,
                    input_batch,
                    target_batch,
                    learning_rate=args.learning_rate,
                    grad_clip=args.grad_clip,
                    layers=args.transformer_layers,
                    heads=args.transformer_heads,
                    context=args.transformer_context,
                    query_mask=mask_batch,
                )
            update_index += 1
    jax.block_until_ready(metrics.loss)
    return params, opt_state, time.perf_counter() - start, metrics


def count_target_hits(predictions: np.ndarray, targets: np.ndarray, target_tokens: list[int]) -> dict[str, int]:
    hits = {}
    for index, token in enumerate(target_tokens):
        mask = targets == token
        hits[str(index)] = int(np.sum(predictions[mask] == token))
    return hits


def target_tokens_for_task(task: str, key_count: int, value_count: int) -> list[int]:
    if task == "delayed-key":
        return [int(KEY_OFFSET + index) for index in range(key_count)]
    if task in {
        "assignment",
        "multi-assignment",
        "multi-query-assignment",
        "assignment-distractor",
        "assignment-alias",
        "mixed-code",
        "generated-python-code",
        "real-python-code",
        "assignment-update",
    }:
        value_offset = ASSIGN_NAME_OFFSET + key_count
        return [int(value_offset + index) for index in range(value_count)]
    if task == "mqar":
        # Shared alphabet: any symbol 1..key_count can be a queried value.
        return [int(1 + index) for index in range(key_count)]
    raise ValueError(f"Unknown task: {task!r}")


def eval_model(
    *,
    params: dict[str, jax.Array],
    eval_inputs: jax.Array,
    eval_targets: jax.Array,
    args: argparse.Namespace,
    model: str,
    kv_read_gate_hard_threshold: float | None = None,
    eval_query_mask: jax.Array | None = None,
) -> dict[str, Any]:
    if model == "ssm":
        if kv_read_gate_hard_threshold is None:
            warmup = ssm_eval(params, eval_inputs, eval_targets, eval_query_mask)
        else:
            warmup = ssm_eval_hard(
                params,
                eval_inputs,
                eval_targets,
                kv_read_gate_hard_threshold=kv_read_gate_hard_threshold,
                query_mask=eval_query_mask,
            )
        jax.block_until_ready(warmup.loss)
        start = time.perf_counter()
        if kv_read_gate_hard_threshold is None:
            metrics = ssm_eval(params, eval_inputs, eval_targets, eval_query_mask)
        else:
            metrics = ssm_eval_hard(
                params,
                eval_inputs,
                eval_targets,
                kv_read_gate_hard_threshold=kv_read_gate_hard_threshold,
                query_mask=eval_query_mask,
            )
        jax.block_until_ready(metrics.loss)
        elapsed = time.perf_counter() - start
        if kv_read_gate_hard_threshold is None:
            predictions, targets, query_mask = ssm_confusion(params, eval_inputs, eval_targets, eval_query_mask)
        else:
            predictions, targets, query_mask = ssm_confusion_hard(
                params,
                eval_inputs,
                eval_targets,
                kv_read_gate_hard_threshold=kv_read_gate_hard_threshold,
                query_mask=eval_query_mask,
            )
    else:
        warmup = transformer_eval(
            params,
            eval_inputs,
            eval_targets,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            context=args.transformer_context,
            query_mask=eval_query_mask,
        )
        jax.block_until_ready(warmup.loss)
        start = time.perf_counter()
        metrics = transformer_eval(
            params,
            eval_inputs,
            eval_targets,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            context=args.transformer_context,
            query_mask=eval_query_mask,
        )
        jax.block_until_ready(metrics.loss)
        elapsed = time.perf_counter() - start
        predictions, targets, query_mask = transformer_confusion(
            params,
            eval_inputs,
            eval_targets,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            context=args.transformer_context,
            query_mask=eval_query_mask,
        )
    query_mask_np = np.asarray(query_mask)
    predictions_np = np.asarray(predictions)[query_mask_np]
    targets_np = np.asarray(targets)[query_mask_np]
    target_tokens = target_tokens_for_task(args.task, args.key_count, args.value_count)
    return {
        "recall_loss": float(metrics.loss),
        "recall_accuracy": float(metrics.accuracy),
        "all_token_loss": float(metrics.all_token_loss),
        "all_token_accuracy": float(metrics.all_token_accuracy),
        "kv_read_gate_hard_threshold": kv_read_gate_hard_threshold,
        "kv_read_gate_mean": float(metrics.kv_read_gate_mean),
        "kv_read_gate_query_mean": float(metrics.kv_read_gate_query_mean),
        "kv_read_gate_non_query_mean": float(metrics.kv_read_gate_non_query_mean),
        "kv_read_gate_entropy_mean": float(metrics.kv_read_gate_entropy_mean),
        "kv_read_gate_query_entropy_mean": float(metrics.kv_read_gate_query_entropy_mean),
        "kv_read_gate_non_query_entropy_mean": float(metrics.kv_read_gate_non_query_entropy_mean),
        "kv_read_gate_lt_0p01_frac": float(metrics.kv_read_gate_lt_0p01_frac),
        "kv_read_gate_query_gt_0p5_frac": float(metrics.kv_read_gate_query_gt_0p5_frac),
        "kv_read_gate_query_gt_0p9_frac": float(metrics.kv_read_gate_query_gt_0p9_frac),
        "kv_read_gate_non_query_lt_0p01_frac": float(metrics.kv_read_gate_non_query_lt_0p01_frac),
        "eval_s": elapsed,
        "records_per_s": int(eval_inputs.shape[0] / max(elapsed, 1.0e-9)),
        "query_count": int(np.sum(query_mask_np)),
        "predicted_target_histogram": {
            str(index): int(np.sum(predictions_np == token))
            for index, token in enumerate(target_tokens)
        },
        "correct_by_target": count_target_hits(predictions_np, targets_np, target_tokens),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    bypass_mode = args.delay > args.transformer_context
    if not bypass_mode and not args.allow_visible_key:
        raise SystemExit("--delay must be larger than --transformer-context for this bypass probe.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive.")
    if args.task == "assignment" and args.delay < 8:
        raise SystemExit("--delay must be at least 8 for --task assignment.")
    if args.task in {"assignment-distractor", "assignment-alias", "assignment-update"} and args.delay < 12:
        raise SystemExit(f"--delay must be at least 12 for --task {args.task}.")
    if args.task == "mixed-code" and args.delay < 24:
        raise SystemExit("--delay must be at least 24 for --task mixed-code.")
    if args.task == "generated-python-code" and args.delay < 48:
        raise SystemExit("--delay must be at least 48 for --task generated-python-code.")
    if args.task in {"multi-assignment", "multi-query-assignment"}:
        if args.binding_count < 1:
            raise SystemExit("--binding-count must be positive.")
        if args.binding_count > args.key_count:
            raise SystemExit("--binding-count must not exceed --key-count for unique-name multi-assignment.")
        min_delay = 4 * args.binding_count + 1
        if args.delay < min_delay:
            raise SystemExit(f"--delay must be at least {min_delay} for --task {args.task}.")
    if args.task in {"multi-query-assignment", "mixed-code", "generated-python-code", "real-python-code"}:
        if args.query_count < 1:
            raise SystemExit("--query-count must be positive.")
    if args.task == "multi-query-assignment":
        if args.query_count > args.binding_count:
            raise SystemExit("--query-count must not exceed --binding-count.")
    if args.task == "mixed-code" and args.query_count < 2:
        raise SystemExit("--query-count must be at least 2 for --task mixed-code.")
    if args.task == "generated-python-code" and args.query_count < 2:
        raise SystemExit("--query-count must be at least 2 for --task generated-python-code.")
    if args.task == "real-python-code":
        min_delay = 4 * args.binding_count + 1
        if args.delay < min_delay:
            raise SystemExit(f"--delay must be at least {min_delay} for --task real-python-code.")
    if args.task == "mqar":
        if args.binding_count < 1 or args.query_count < 1:
            raise SystemExit("--binding-count and --query-count must be positive for --task mqar.")
        if args.query_count > args.binding_count:
            raise SystemExit("--query-count must not exceed --binding-count for --task mqar.")
        if args.binding_count > args.key_count:
            raise SystemExit("--binding-count must not exceed --key-count for --task mqar.")
        if args.delay < 4 * args.binding_count:
            raise SystemExit(f"--delay must be at least {4 * args.binding_count} for --task mqar.")

    vocab_size = vocab_size_for_task(args.task, args.key_count, args.value_count)
    train_inputs_np, train_targets_np, train_aux_np = make_task_data(
        task=args.task,
        records=args.train_records,
        delay=args.delay,
        key_count=args.key_count,
        value_count=args.value_count,
        binding_count=args.binding_count,
        query_count=args.query_count,
        seed=args.seed,
    )
    eval_inputs_np, eval_targets_np, eval_aux_np = make_task_data(
        task=args.task,
        records=args.eval_records,
        delay=args.delay,
        key_count=args.key_count,
        value_count=args.value_count,
        binding_count=args.binding_count,
        query_count=args.query_count,
        seed=args.seed + 1,
    )
    train_inputs = jnp.asarray(train_inputs_np)
    train_targets = jnp.asarray(train_targets_np)
    eval_inputs = jnp.asarray(eval_inputs_np)
    eval_targets = jnp.asarray(eval_targets_np)
    # MQAR has no query-marker token, so scoring positions ride an explicit mask
    # (returned as the third element). Legacy tasks keep deriving the mask from
    # the QUERY_TOKEN internally, so they pass None and stay byte-identical.
    if args.task == "mqar":
        train_query_mask = jnp.asarray(train_aux_np)
        eval_query_mask = jnp.asarray(eval_aux_np)
    else:
        train_query_mask = None
        eval_query_mask = None
    seq_len = train_inputs_np.shape[1]

    key = jax.random.PRNGKey(args.seed)
    ssm_key, transformer_key = jax.random.split(key)
    ssm_params = init_ssm_params(
        ssm_key,
        vocab_size,
        args.ssm_input_dim,
        args.ssm_state_dim,
        variant=args.ssm_variant,
        skip_rank=args.ssm_skip_rank,
    )
    if args.ssm_decay_init is not None:
        ssm_params["decay_raw"] = jnp.full(
            ssm_params["decay_raw"].shape,
            args.ssm_decay_init,
            dtype=jnp.float32,
        )
    if args.init_recall_memory:
        ssm_params = init_delayed_recall_memory_params(
            ssm_params,
            key_count=args.key_count,
            write_logit=args.memory_write_logit,
            closed_logit=args.memory_closed_logit,
            read_logit=args.memory_read_logit,
            memory_decay_raw=args.memory_decay_raw,
            memory_logit_scale=args.memory_logit_scale,
        )
    if args.init_assignment_kv_memory:
        ssm_params = init_assignment_kv_memory_params(
            ssm_params,
            key_count=args.key_count,
            value_count=args.value_count,
            key_logit=args.kv_key_logit,
            value_logit=args.kv_value_logit,
            erase_logit=args.kv_erase_logit,
            query_logit=args.kv_query_logit,
            closed_logit=args.kv_closed_logit,
            kv_decay_raw=args.kv_decay_raw,
            kv_logit_scale=args.kv_logit_scale,
        )
    ssm_param_count = count_params(ssm_params)

    transformer_d_model = args.transformer_d_model
    if transformer_d_model is None:
        transformer_d_model = choose_transformer_d_model(
            target_params=ssm_param_count,
            vocab_size=vocab_size,
            seq_len=seq_len,
            layers=args.transformer_layers,
            heads=args.transformer_heads,
            ff_mult=args.transformer_ff_mult,
            max_d_model=args.transformer_max_d_model,
        )
    transformer_params = init_transformer_params(
        transformer_key,
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=transformer_d_model,
        layers=args.transformer_layers,
        ff_mult=args.transformer_ff_mult,
    )
    transformer_param_count = count_params(transformer_params)
    budget_ratio = transformer_param_count / max(1, ssm_param_count)

    print("TextPy/SoA long-context delayed-recall comparison")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("Protocol")
    print(f"  train_records: {args.train_records:,}")
    print(f"  eval_records: {args.eval_records:,}")
    print(f"  task: {args.task}")
    print(f"  key_count: {args.key_count}")
    if args.task in {
        "assignment",
        "multi-assignment",
        "multi-query-assignment",
        "assignment-distractor",
        "assignment-alias",
        "mixed-code",
        "generated-python-code",
        "real-python-code",
        "assignment-update",
    }:
        print(f"  value_count: {args.value_count}")
    if args.task in {"multi-assignment", "multi-query-assignment", "real-python-code"}:
        print(f"  binding_count: {args.binding_count}")
    if args.task in {"multi-query-assignment", "mixed-code", "generated-python-code", "real-python-code"}:
        print(f"  query_count: {args.query_count}")
    print(f"  delay: {args.delay}")
    print(f"  seq_len: {seq_len}")
    print(f"  transformer_context: {args.transformer_context}")
    print(f"  bypass_mode: {bypass_mode}")
    print(f"  epochs: {args.epochs}")
    print("")
    print("Parameter budget")
    print(f"  ssm_variant: {infer_ssm_variant(ssm_params)}")
    print(f"  ssm_params: {ssm_param_count:,}")
    print(f"  transformer_d_model: {transformer_d_model}")
    print(f"  transformer_params: {transformer_param_count:,}")
    print(f"  transformer/ssm: {budget_ratio:.3f}")
    print("")

    print("Training SSM")
    ssm_params, _, ssm_train_s, ssm_train_metrics = train_model(
        params=ssm_params,
        train_inputs=train_inputs,
        train_targets=train_targets,
        args=args,
        model="ssm",
        train_query_mask=train_query_mask,
    )
    print(
        f"  recall_loss={float(ssm_train_metrics.loss):.4f} "
        f"recall_acc={float(ssm_train_metrics.accuracy):.4f} "
        f"train_s={ssm_train_s:.2f}"
    )

    print("Training Transformer")
    transformer_params, _, transformer_train_s, transformer_train_metrics = train_model(
        params=transformer_params,
        train_inputs=train_inputs,
        train_targets=train_targets,
        args=args,
        model="transformer",
        train_query_mask=train_query_mask,
    )
    print(
        f"  recall_loss={float(transformer_train_metrics.loss):.4f} "
        f"recall_acc={float(transformer_train_metrics.accuracy):.4f} "
        f"train_s={transformer_train_s:.2f}"
    )

    ssm_eval_payload = eval_model(
        params=ssm_params,
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        args=args,
        model="ssm",
        eval_query_mask=eval_query_mask,
    )
    ssm_hard_eval_payload = None
    if args.kv_read_gate_hard_eval_threshold is not None:
        ssm_hard_eval_payload = eval_model(
            params=ssm_params,
            eval_inputs=eval_inputs,
            eval_targets=eval_targets,
            args=args,
            model="ssm",
            kv_read_gate_hard_threshold=args.kv_read_gate_hard_eval_threshold,
            eval_query_mask=eval_query_mask,
        )
    transformer_eval_payload = eval_model(
        params=transformer_params,
        eval_inputs=eval_inputs,
        eval_targets=eval_targets,
        args=args,
        model="transformer",
        eval_query_mask=eval_query_mask,
    )
    winner = "ssm" if ssm_eval_payload["recall_accuracy"] > transformer_eval_payload["recall_accuracy"] else "transformer"
    if ssm_eval_payload["recall_accuracy"] == transformer_eval_payload["recall_accuracy"]:
        winner = "tie"

    print("")
    print("Held-out delayed recall")
    print(
        "  ssm: "
        f"loss={ssm_eval_payload['recall_loss']:.4f} "
        f"acc={ssm_eval_payload['recall_accuracy']:.4f} "
        f"records/s={ssm_eval_payload['records_per_s']:,}"
    )
    if "kv_key_token" in ssm_params:
        print(
            "  ssm_kv_read_gate: "
            f"mean={ssm_eval_payload['kv_read_gate_mean']:.4f} "
            f"query={ssm_eval_payload['kv_read_gate_query_mean']:.4f} "
            f"non_query={ssm_eval_payload['kv_read_gate_non_query_mean']:.4f}"
        )
        print(
            "  ssm_kv_read_gate_hardness: "
            f"query>0.5={ssm_eval_payload['kv_read_gate_query_gt_0p5_frac']:.4f} "
            f"query>0.9={ssm_eval_payload['kv_read_gate_query_gt_0p9_frac']:.4f} "
            f"non_query<0.01={ssm_eval_payload['kv_read_gate_non_query_lt_0p01_frac']:.4f}"
        )
        if ssm_hard_eval_payload is not None:
            print(
                "  ssm_hard_inference: "
                f"threshold={args.kv_read_gate_hard_eval_threshold:.4f} "
                f"loss={ssm_hard_eval_payload['recall_loss']:.4f} "
                f"acc={ssm_hard_eval_payload['recall_accuracy']:.4f} "
                f"gate_mean={ssm_hard_eval_payload['kv_read_gate_mean']:.4f} "
                f"query={ssm_hard_eval_payload['kv_read_gate_query_mean']:.4f} "
                f"non_query={ssm_hard_eval_payload['kv_read_gate_non_query_mean']:.4f}"
            )
    print(
        "  transformer: "
        f"loss={transformer_eval_payload['recall_loss']:.4f} "
        f"acc={transformer_eval_payload['recall_accuracy']:.4f} "
        f"records/s={transformer_eval_payload['records_per_s']:,}"
    )
    print("")
    print("Memory/context")
    print(f"  ssm_parameter_bytes: {format_bytes(count_bytes(ssm_params))}")
    print(f"  transformer_parameter_bytes: {format_bytes(count_bytes(transformer_params))}")
    print(f"  ssm_effective_context_tokens: {seq_len}")
    print(f"  transformer_effective_context_tokens: {args.transformer_context}")
    print(f"  recall_distance_over_transformer_context: {args.delay / args.transformer_context:.2f}x")
    print("")
    print("Comparison")
    print(f"  winner_by_recall_accuracy: {winner}")
    print(
        "  ssm_minus_transformer_recall_accuracy: "
        f"{ssm_eval_payload['recall_accuracy'] - transformer_eval_payload['recall_accuracy']:.6f}"
    )

    task_extra: dict[str, Any] = {}
    if args.task == "real-python-code":
        train_extracted = extract_real_python_records(
            key_count=args.key_count,
            value_count=args.value_count,
            binding_count=args.binding_count,
            query_count=args.query_count,
            seed=args.seed,
        )
        eval_extracted = extract_real_python_records(
            key_count=args.key_count,
            value_count=args.value_count,
            binding_count=args.binding_count,
            query_count=args.query_count,
            seed=args.seed + 1,
        )
        task_extra = {
            "data_source": "static AST extraction from Python files in this repository",
            "execution": "source files are parsed but never executed",
            "train_extracted_records": len(train_extracted),
            "eval_extracted_records": len(eval_extracted),
            "fold_split": "stable hash parity; train seed and eval seed use opposite folds",
            "target_derivation": (
                "Assignments are statically simulated in a compact variable environment; "
                "non-name RHS expressions are canonicalized into value buckets."
            ),
        }

    payload = {
        "created_at": timestamp(),
        "kind": "long_context_delayed_recall",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "protocol": {
            "train_records": args.train_records,
            "eval_records": args.eval_records,
            "task": args.task,
            "key_count": args.key_count,
            "value_count": args.value_count,
            "binding_count": args.binding_count,
            "query_count": args.query_count,
            "vocab_size": vocab_size,
            "delay": args.delay,
            "seq_len": seq_len,
            "transformer_context": args.transformer_context,
            "bypass_mode": bypass_mode,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "grad_clip": args.grad_clip,
            "kv_read_gate_l1": args.kv_read_gate_l1,
            "kv_read_gate_l1_start": args.kv_read_gate_l1_start,
            "kv_read_gate_l1_ramp_epochs": args.kv_read_gate_l1_ramp_epochs,
            "kv_read_gate_target": args.kv_read_gate_target,
            "kv_read_gate_target_weight": args.kv_read_gate_target_weight,
            "kv_read_gate_entropy_weight": args.kv_read_gate_entropy_weight,
            "kv_read_gate_query_margin": args.kv_read_gate_query_margin,
            "kv_read_gate_query_margin_weight": args.kv_read_gate_query_margin_weight,
            "kv_read_gate_hard_eval_threshold": args.kv_read_gate_hard_eval_threshold,
            "seed": args.seed,
            "claim_guardrail": (
                "This task is a bounded long-context/code-memory probe when bypass_mode=true: "
                "the key is outside the Transformer attention cap but inside the SSM "
                "recurrent training window. With --allow-visible-key it becomes a "
                "control run, not bypass evidence. It is not a general natural-language "
                "superiority claim."
            ),
            **task_extra,
        },
        "ssm": {
            "variant": infer_ssm_variant(ssm_params),
            "input_dim": args.ssm_input_dim,
            "state_dim": args.ssm_state_dim,
            "decay_init_raw": args.ssm_decay_init,
            "init_recall_memory": args.init_recall_memory,
            "init_assignment_kv_memory": args.init_assignment_kv_memory,
            "parameter_count": ssm_param_count,
            "parameter_bytes": count_bytes(ssm_params),
            "mean_decay_A": float(jnp.mean(decay_from_raw(ssm_params["decay_raw"]))),
            "train_s": ssm_train_s,
            "train": {
                "recall_loss": float(ssm_train_metrics.loss),
                "recall_accuracy": float(ssm_train_metrics.accuracy),
                "all_token_loss": float(ssm_train_metrics.all_token_loss),
                "all_token_accuracy": float(ssm_train_metrics.all_token_accuracy),
                "kv_read_gate_mean": float(ssm_train_metrics.kv_read_gate_mean),
                "kv_read_gate_query_mean": float(ssm_train_metrics.kv_read_gate_query_mean),
                "kv_read_gate_non_query_mean": float(ssm_train_metrics.kv_read_gate_non_query_mean),
                "kv_read_gate_entropy_mean": float(ssm_train_metrics.kv_read_gate_entropy_mean),
                "kv_read_gate_query_entropy_mean": float(ssm_train_metrics.kv_read_gate_query_entropy_mean),
                "kv_read_gate_non_query_entropy_mean": float(ssm_train_metrics.kv_read_gate_non_query_entropy_mean),
                "kv_read_gate_lt_0p01_frac": float(ssm_train_metrics.kv_read_gate_lt_0p01_frac),
                "kv_read_gate_query_gt_0p5_frac": float(ssm_train_metrics.kv_read_gate_query_gt_0p5_frac),
                "kv_read_gate_query_gt_0p9_frac": float(ssm_train_metrics.kv_read_gate_query_gt_0p9_frac),
                "kv_read_gate_non_query_lt_0p01_frac": float(ssm_train_metrics.kv_read_gate_non_query_lt_0p01_frac),
            },
            "eval": ssm_eval_payload,
            "hard_eval": ssm_hard_eval_payload,
        },
        "transformer": {
            "d_model": transformer_d_model,
            "layers": args.transformer_layers,
            "heads": args.transformer_heads,
            "ff_mult": args.transformer_ff_mult,
            "context": args.transformer_context,
            "parameter_count": transformer_param_count,
            "parameter_bytes": count_bytes(transformer_params),
            "train_s": transformer_train_s,
            "train": {
                "recall_loss": float(transformer_train_metrics.loss),
                "recall_accuracy": float(transformer_train_metrics.accuracy),
                "all_token_loss": float(transformer_train_metrics.all_token_loss),
                "all_token_accuracy": float(transformer_train_metrics.all_token_accuracy),
            },
            "eval": transformer_eval_payload,
        },
        "comparison": {
            "winner_by_recall_accuracy": winner,
            "ssm_minus_transformer_recall_accuracy": (
                ssm_eval_payload["recall_accuracy"] - transformer_eval_payload["recall_accuracy"]
            ),
            "ssm_minus_transformer_recall_loss": (
                ssm_eval_payload["recall_loss"] - transformer_eval_payload["recall_loss"]
            ),
            "transformer_to_ssm_param_ratio": budget_ratio,
        },
    }
    output_json = Path(args.output_json) if args.output_json else Path("artifacts/long_context_recall") / f"{timestamp()}.json"
    write_json(output_json, payload)
    print("")
    print(f"Saved JSON: {output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SSM vs context-capped Transformer on delayed recall.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--task",
        choices=(
            "delayed-key",
            "assignment",
            "multi-assignment",
            "multi-query-assignment",
            "assignment-distractor",
            "assignment-alias",
            "mixed-code",
            "generated-python-code",
            "real-python-code",
            "assignment-update",
            "mqar",
        ),
        default="delayed-key",
    )
    parser.add_argument("--train-records", type=int, default=4096)
    parser.add_argument("--eval-records", type=int, default=2048)
    parser.add_argument("--key-count", type=int, default=32)
    parser.add_argument("--value-count", type=int, default=32)
    parser.add_argument("--binding-count", type=int, default=4)
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--delay", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--kv-read-gate-l1",
        type=float,
        default=0.0,
        help="L1 penalty on SSM KV read-gate usage; eval reports pure recall loss.",
    )
    parser.add_argument(
        "--kv-read-gate-l1-start",
        type=float,
        default=None,
        help="Optional starting L1 value for a linear KV read-gate penalty curriculum.",
    )
    parser.add_argument(
        "--kv-read-gate-l1-ramp-epochs",
        type=int,
        default=0,
        help="Epochs over which to linearly ramp from --kv-read-gate-l1-start to --kv-read-gate-l1.",
    )
    parser.add_argument(
        "--kv-read-gate-target",
        type=float,
        default=0.0,
        help="Target mean KV read-gate usage for quadratic target sparsity penalty.",
    )
    parser.add_argument(
        "--kv-read-gate-target-weight",
        type=float,
        default=0.0,
        help="Weight for quadratic target sparsity penalty on mean KV read-gate usage.",
    )
    parser.add_argument(
        "--kv-read-gate-entropy-weight",
        type=float,
        default=0.0,
        help="Weight for gate entropy penalty; encourages more binary read decisions.",
    )
    parser.add_argument(
        "--kv-read-gate-query-margin",
        type=float,
        default=0.0,
        help="Minimum desired read-gate value on query tokens for squared hinge margin penalty.",
    )
    parser.add_argument(
        "--kv-read-gate-query-margin-weight",
        type=float,
        default=0.0,
        help="Weight for squared hinge penalty that pushes query-token KV read gates upward.",
    )
    parser.add_argument(
        "--kv-read-gate-hard-eval-threshold",
        type=float,
        default=None,
        help="If set, also evaluate SSM with KV read gates thresholded to 0/1 at this value.",
    )
    parser.add_argument("--ssm-input-dim", type=int, default=64)
    parser.add_argument("--ssm-state-dim", type=int, default=96)
    parser.add_argument("--ssm-variant", choices=SSM_VARIANTS, default="static")
    parser.add_argument("--ssm-skip-rank", type=int, default=32)
    parser.add_argument("--ssm-decay-init", type=float, default=6.0)
    parser.add_argument("--init-recall-memory", action="store_true")
    parser.add_argument("--memory-write-logit", type=float, default=8.0)
    parser.add_argument("--memory-read-logit", type=float, default=8.0)
    parser.add_argument("--memory-closed-logit", type=float, default=-8.0)
    parser.add_argument("--memory-decay-raw", type=float, default=8.0)
    parser.add_argument("--memory-logit-scale", type=float, default=32.0)
    parser.add_argument("--init-assignment-kv-memory", action="store_true")
    parser.add_argument("--kv-key-logit", type=float, default=8.0)
    parser.add_argument("--kv-value-logit", type=float, default=8.0)
    parser.add_argument("--kv-erase-logit", type=float, default=8.0)
    parser.add_argument("--kv-query-logit", type=float, default=8.0)
    parser.add_argument("--kv-closed-logit", type=float, default=-8.0)
    parser.add_argument("--kv-decay-raw", type=float, default=8.0)
    parser.add_argument("--kv-logit-scale", type=float, default=32.0)
    parser.add_argument("--transformer-d-model", type=int, default=None)
    parser.add_argument("--transformer-max-d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=2)
    parser.add_argument("--transformer-ff-mult", type=int, default=2)
    parser.add_argument("--transformer-context", type=int, default=32)
    parser.add_argument("--allow-visible-key", action="store_true")
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
