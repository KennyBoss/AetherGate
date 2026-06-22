#!/usr/bin/env python3
"""Train a lightweight success prior from CodePy tape execution traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from code_tape_prior import (
    feature_names,
    save_json,
    sigmoid,
    vectorize_record,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def infer_program_length(records: list[dict[str, Any]]) -> int:
    lengths = []
    for record in records:
        args = record.get("args", {})
        if "program_length" in args:
            lengths.append(int(args["program_length"]))
        elif "program_ids" in record:
            lengths.append(len(record["program_ids"]))
    if not lengths:
        raise SystemExit("Could not infer program length from trace records.")
    return max(lengths)


def labels_from_records(records: list[dict[str, Any]], success_mse: float) -> np.ndarray:
    labels = []
    for record in records:
        train_mse = float(record.get("train_mse", math.inf))
        holdout_mse = float(record.get("holdout_mse", math.inf))
        labels.append(1.0 if train_mse <= success_mse and holdout_mse <= success_mse else 0.0)
    return np.asarray(labels, dtype=np.float64)


def weighted_log_loss(y: np.ndarray, prob: np.ndarray, sample_weight: np.ndarray) -> float:
    clipped = np.clip(prob, 1.0e-9, 1.0 - 1.0e-9)
    losses = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    return float(np.sum(losses * sample_weight) / max(float(np.sum(sample_weight)), 1.0))


def accuracy(y: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob >= 0.5) == (y >= 0.5)))


def auc_score(y: np.ndarray, prob: np.ndarray) -> float:
    positives = int(np.sum(y == 1.0))
    negatives = int(np.sum(y == 0.0))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(prob)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(prob) + 1, dtype=np.float64)
    pos_rank_sum = float(np.sum(ranks[y == 1.0]))
    return (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def top_k_success_rate(y: np.ndarray, prob: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return 0.0
    k = min(max(k, 1), len(y))
    top = np.argsort(prob)[-k:]
    return float(np.mean(y[top]))


def metric_block(y: np.ndarray, prob: np.ndarray, sample_weight: np.ndarray) -> dict[str, float]:
    positives = int(np.sum(y == 1.0))
    negatives = int(np.sum(y == 0.0))
    return {
        "records": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "log_loss": weighted_log_loss(y, prob, sample_weight),
        "accuracy": accuracy(y, prob),
        "auc": auc_score(y, prob),
        "top_10_success_rate": top_k_success_rate(y, prob, 10),
        "top_32_success_rate": top_k_success_rate(y, prob, 32),
        "has_auc_support": positives > 0 and negatives > 0,
    }


def train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    test_fraction: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if int(np.sum(y == 1.0)) == 0 or int(np.sum(y == 0.0)) == 0:
        raise SystemExit("Trace dataset needs both successful and unsuccessful programs.")
    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(np.flatnonzero(y == 1.0))
    neg_idx = rng.permutation(np.flatnonzero(y == 0.0))
    test_pos_count = 0 if len(pos_idx) <= 1 else min(max(int(len(pos_idx) * test_fraction), 1), len(pos_idx) - 1)
    test_neg_count = 0 if len(neg_idx) <= 1 else min(max(int(len(neg_idx) * test_fraction), 1), len(neg_idx) - 1)
    test_idx = np.concatenate([pos_idx[:test_pos_count], neg_idx[:test_neg_count]])
    train_idx = np.concatenate([pos_idx[test_pos_count:], neg_idx[test_neg_count:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    positives = max(float(np.sum(y_train == 1.0)), 1.0)
    negatives = max(float(np.sum(y_train == 0.0)), 1.0)
    train_weights = np.where(
        y_train == 1.0,
        len(y_train) / (2.0 * positives),
        len(y_train) / (2.0 * negatives),
    )
    test_weights = np.ones_like(y_test, dtype=np.float64)

    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    history = []
    for epoch in range(epochs):
        logits = x_train @ weights + bias
        prob = sigmoid(logits)
        err = (prob - y_train) * train_weights
        denom = max(float(np.sum(train_weights)), 1.0)
        weights -= lr * ((x_train.T @ err) / denom + l2 * weights)
        bias -= lr * float(np.sum(err) / denom)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(epochs // 5, 1) == 0:
            train_prob = sigmoid(x_train @ weights + bias)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_log_loss": weighted_log_loss(y_train, train_prob, train_weights),
                }
            )

    train_prob = sigmoid(x_train @ weights + bias)
    test_prob = sigmoid(x_test @ weights + bias)
    metrics = {
        "train": metric_block(y_train, train_prob, train_weights),
        "test": metric_block(y_test, test_prob, test_weights),
        "history": history,
    }
    return weights, bias, metrics


def run(args: argparse.Namespace) -> None:
    records = read_jsonl(Path(args.trace_jsonl))
    if not records:
        raise SystemExit("Trace JSONL is empty.")
    program_length = args.program_length or infer_program_length(records)
    x = np.vstack(
        [
            vectorize_record(record, program_length=program_length, vocab_size=args.vocab_size)
            for record in records
        ]
    )
    y = labels_from_records(records, args.success_mse)
    weights, bias, metrics = train_logistic(
        x,
        y,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )
    payload = {
        "format_version": 1,
        "prototype": "CodePy tape lightweight success prior",
        "guardrail": "Trains on safe DSL trace records only; no arbitrary generated source is executed.",
        "trace_jsonl": args.trace_jsonl,
        "label": {
            "definition": "train_mse <= success_mse and holdout_mse <= success_mse",
            "success_mse": args.success_mse,
        },
        "model": {
            "type": "logistic_regression",
            "program_length": program_length,
            "vocab_size": args.vocab_size,
            "feature_names": feature_names(program_length, args.vocab_size),
            "weights": weights.tolist(),
            "bias": bias,
        },
        "metrics": metrics,
        "coverage": {
            "records": int(len(records)),
            "positives": int(np.sum(y == 1.0)),
            "negatives": int(np.sum(y == 0.0)),
            "insufficient_positive_coverage": int(np.sum(y == 1.0)) < args.min_positive_coverage,
            "min_positive_coverage": args.min_positive_coverage,
            "note": (
                "Low positive coverage can make test AUC undefined or unstable; "
                "the prior may still be useful as a warm-start heuristic but is weak statistical evidence."
            ),
        },
        "args": vars(args),
    }
    save_json(args.output_json, payload)
    print("CodePy tape prior training")
    print(f"  records: {len(records)}")
    print(f"  positives: {int(np.sum(y == 1.0))}")
    print(f"  output_json: {args.output_json}")
    print(f"  train_auc: {metrics['train']['auc']:.4g}")
    print(f"  test_auc: {metrics['test']['auc']:.4g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CodePy tape trace success prior.")
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--program-length", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=20)
    parser.add_argument("--success-mse", type=float, default=1.0e-8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-positive-coverage", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
