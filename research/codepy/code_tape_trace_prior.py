#!/usr/bin/env python3
"""Trace-only features for contrastive CodePy tape priors.

The functions in this module intentionally avoid program token identity.  The
prior is meant to estimate P(trace | task): whether the observed execution
behavior solves the task, not whether a particular opcode sequence looks
familiar.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TRACE_NUMERIC_FEATURES = [
    "depth_frac",
    "active_block_frac",
    "prefix_complete",
    "output_hit",
    "index_hit",
    "max_hit",
    "coverage",
    "predicate_entropy",
    "loop_hit",
    "halt_hit",
    "running_value_hit",
    "running_index_hit",
    "paired_running_hit",
    "value_pair_gap",
    "index_pair_gap",
    "output_pair_gap",
    "trajectory_score_norm",
]

PREVIEW_BINS = list(range(-9, 10))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def safe_log(probability: float) -> float:
    return math.log(min(max(float(probability), 1.0e-9), 1.0 - 1.0e-9))


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


def classify_trace_record(
    record: dict[str, Any],
    *,
    success_mse: float = 1.0e-8,
    near_miss_output_hit: float = 0.45,
    near_miss_max_hit: float = 0.90,
    near_miss_running_value_hit: float = 0.70,
    near_miss_paired_running_hit: float = 0.95,
) -> dict[str, Any]:
    """Assign success / near_miss / failure labels from execution metrics."""
    train_mse = safe_float(record.get("train_mse"), math.inf)
    holdout_mse = safe_float(record.get("holdout_mse"), math.inf)
    success = train_mse <= success_mse and holdout_mse <= success_mse
    output_hit = safe_float(record.get("output_hit"))
    max_hit = safe_float(record.get("max_hit"))
    running_value_hit = safe_float(record.get("running_value_hit"))
    paired_running_hit = safe_float(record.get("paired_running_hit"))
    value_like = (
        output_hit >= near_miss_output_hit
        or max_hit >= near_miss_max_hit
        or running_value_hit >= near_miss_running_value_hit
    )
    missing_full_pair = paired_running_hit < near_miss_paired_running_hit or not success
    near_miss = bool((not success) and value_like and missing_full_pair)
    if success:
        class_label = "success"
        reason = "zero_train_and_holdout_error"
    elif near_miss:
        class_label = "near_miss"
        reason = "value_like_trace_without_full_argmax_pair"
    else:
        class_label = "failure"
        reason = "low_value_or_control_flow_match"
    return {
        "success": bool(success),
        "near_miss": bool(near_miss),
        "failure": bool(not success and not near_miss),
        "class_label": class_label,
        "binary_label": 1 if success else 0,
        "label_reason": reason,
    }


def trace_feature_names(hash_buckets: int) -> list[str]:
    names = list(TRACE_NUMERIC_FEATURES)
    names.extend(f"preview_bin_{value}" for value in PREVIEW_BINS)
    names.extend(f"signature_bucket_{i}" for i in range(hash_buckets))
    return names


def _signature_bucket(signature: str, hash_buckets: int) -> int:
    if hash_buckets <= 0:
        return -1
    digest = hashlib.sha1(str(signature).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % hash_buckets


def vectorize_trace_record(record: dict[str, Any], *, hash_buckets: int = 64) -> np.ndarray:
    """Convert trace metrics and compressed signatures to a trace-only vector."""
    args = record.get("args", {}) if isinstance(record.get("args"), dict) else {}
    program_length = max(int(args.get("program_length", len(record.get("program_ids", [])) or 1)), 1)
    depth = safe_float(record.get("depth"), program_length)
    active_blocks = len(record.get("compact_program_ids", []) or [])
    output_hit = safe_float(record.get("output_hit"))
    index_hit = safe_float(record.get("index_hit"))
    max_hit = safe_float(record.get("max_hit"))
    paired_running_hit = safe_float(record.get("paired_running_hit"))
    running_value_hit = safe_float(record.get("running_value_hit"))
    running_index_hit = safe_float(record.get("running_index_hit"))
    trajectory_score = safe_float(record.get("trajectory_score"))
    numeric = [
        depth / program_length,
        active_blocks / program_length,
        1.0 if depth >= program_length else 0.0,
        output_hit,
        index_hit,
        max_hit,
        safe_float(record.get("coverage")),
        safe_float(record.get("predicate_entropy")),
        safe_float(record.get("loop_hit")),
        safe_float(record.get("halt_hit")),
        running_value_hit,
        running_index_hit,
        paired_running_hit,
        max(running_value_hit - paired_running_hit, 0.0),
        max(running_index_hit - paired_running_hit, 0.0),
        max(output_hit - paired_running_hit, 0.0),
        trajectory_score / 10.0,
    ]
    preview = record.get("signature_preview", [])
    hist = np.zeros(len(PREVIEW_BINS), dtype=np.float64)
    if isinstance(preview, list) and preview:
        for value in preview:
            clipped = int(max(min(round(safe_float(value)), 9), -9))
            hist[clipped + 9] += 1.0
        hist /= max(float(len(preview)), 1.0)
    buckets = np.zeros(max(hash_buckets, 0), dtype=np.float64)
    bucket = _signature_bucket(str(record.get("signature", "")), hash_buckets)
    if bucket >= 0:
        buckets[bucket] = 1.0
    return np.concatenate([np.asarray(numeric, dtype=np.float64), hist, buckets])


def predict_trace_probability(model_payload: dict[str, Any], record: dict[str, Any]) -> float:
    model = model_payload["model"]
    features = vectorize_trace_record(record, hash_buckets=int(model.get("hash_buckets", 64)))
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model.get("bias", 0.0))
    logit = float(features @ weights + bias)
    if logit >= 0:
        return float(1.0 / (1.0 + math.exp(-logit)))
    exp_logit = math.exp(logit)
    return float(exp_logit / (1.0 + exp_logit))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def append_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
