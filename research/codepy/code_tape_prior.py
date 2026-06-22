#!/usr/bin/env python3
"""Shared feature code for CodePy tape trace priors."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TRACE_FEATURE_NAMES = [
    "trace_active_frac",
    "trace_read_frac",
    "trace_write_frac",
    "trace_move_frac",
    "trace_loop_frac",
    "trace_accumulate_frac",
    "trace_predicate_frac",
    "trace_conditional_frac",
    "trace_max_ptr_frac",
    "trace_shaping_bonus",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def trace_embedding_from_metrics(
    metrics: dict[str, Any],
    *,
    program_length: int,
    max_steps: int,
    tape_len: int,
) -> list[float]:
    """Normalize aggregate execution metrics into a small behavior embedding."""
    program_denom = max(program_length, 1)
    step_denom = max(max_steps, 1)
    ptr_denom = max(tape_len - 1, 1)
    return [
        safe_float(metrics.get("active_blocks")) / program_denom,
        safe_float(metrics.get("read_steps")) / step_denom,
        safe_float(metrics.get("write_steps")) / step_denom,
        safe_float(metrics.get("move_steps")) / step_denom,
        safe_float(metrics.get("loop_steps")) / step_denom,
        safe_float(metrics.get("accumulate_steps")) / step_denom,
        safe_float(metrics.get("predicate_steps")) / step_denom,
        safe_float(metrics.get("conditional_steps")) / step_denom,
        safe_float(metrics.get("max_ptr")) / ptr_denom,
        safe_float(metrics.get("shaping_bonus")),
    ]


def trace_embedding_from_record(record: dict[str, Any]) -> list[float]:
    embedded = record.get("trace_embedding")
    if isinstance(embedded, dict) and isinstance(embedded.get("values"), list):
        return [safe_float(v) for v in embedded["values"]]
    args = record.get("args", {})
    return trace_embedding_from_metrics(
        record,
        program_length=int(args.get("program_length", len(record.get("program_ids", [])) or 1)),
        max_steps=int(args.get("max_steps", 1)),
        tape_len=int(args.get("tape_len", 8)),
    )


def feature_names(program_length: int, vocab_size: int) -> list[str]:
    names: list[str] = []
    for pos in range(program_length):
        for op_id in range(vocab_size):
            names.append(f"pos{pos}_op{op_id}")
    names.extend(f"count_op{op_id}" for op_id in range(vocab_size))
    names.extend(TRACE_FEATURE_NAMES)
    return names


def vectorize_program(
    program_ids: list[int] | np.ndarray,
    trace_embedding: list[float] | np.ndarray,
    *,
    program_length: int,
    vocab_size: int,
) -> np.ndarray:
    """Encode fixed-length tokens plus aggregate trace behavior."""
    features = np.zeros(program_length * vocab_size + vocab_size + len(TRACE_FEATURE_NAMES))
    program = [int(op) for op in list(program_ids)[:program_length]]
    program.extend([0] * max(program_length - len(program), 0))
    for pos, op_id in enumerate(program):
        if 0 <= op_id < vocab_size:
            features[pos * vocab_size + op_id] = 1.0
            features[program_length * vocab_size + op_id] += 1.0 / max(program_length, 1)
    trace = np.asarray(trace_embedding, dtype=np.float64)
    if trace.shape[0] != len(TRACE_FEATURE_NAMES):
        padded = np.zeros(len(TRACE_FEATURE_NAMES), dtype=np.float64)
        padded[: min(len(padded), trace.shape[0])] = trace[: len(padded)]
        trace = padded
    features[-len(TRACE_FEATURE_NAMES) :] = trace
    return features


def vectorize_record(record: dict[str, Any], *, program_length: int, vocab_size: int) -> np.ndarray:
    return vectorize_program(
        record.get("program_ids", []),
        trace_embedding_from_record(record),
        program_length=program_length,
        vocab_size=vocab_size,
    )


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x)
    out = np.empty_like(arr, dtype=np.float64)
    positive = arr >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def predict_probability(model: dict[str, Any], features: np.ndarray) -> float:
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model.get("bias", 0.0))
    return float(sigmoid(float(np.dot(features, weights) + bias)))


def logit(probability: float) -> float:
    clipped = min(max(probability, 1.0e-9), 1.0 - 1.0e-9)
    return math.log(clipped / (1.0 - clipped))


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
