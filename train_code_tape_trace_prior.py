#!/usr/bin/env python3
"""Train a contrastive trace-only prior for CodePy tape programs."""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

import numpy as np

from code_tape_trace_prior import (
    read_jsonl,
    save_json,
    sigmoid,
    trace_feature_names,
    vectorize_trace_record,
)


def labels_from_records(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([1.0 if record.get("class_label") == "success" else 0.0 for record in records])


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


def split_indices(labels: np.ndarray, classes: list[str], *, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for class_label in sorted(set(classes)):
        idx = np.asarray([i for i, label in enumerate(classes) if label == class_label], dtype=np.int64)
        idx = rng.permutation(idx)
        if len(idx) <= 1:
            train_parts.append(idx)
            continue
        test_count = min(max(int(len(idx) * test_fraction), 1), len(idx) - 1)
        test_parts.append(idx[:test_count])
        train_parts.append(idx[test_count:])
    train_idx = np.concatenate(train_parts) if train_parts else np.asarray([], dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.asarray([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    if len(test_idx) == 0:
        test_idx = train_idx.copy()
    return train_idx, test_idx


def class_weights(y: np.ndarray) -> np.ndarray:
    positives = max(float(np.sum(y == 1.0)), 1.0)
    negatives = max(float(np.sum(y == 0.0)), 1.0)
    return np.where(y == 1.0, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))


def metric_block(y: np.ndarray, prob: np.ndarray, sample_weight: np.ndarray) -> dict[str, Any]:
    positives = int(np.sum(y == 1.0))
    negatives = int(np.sum(y == 0.0))
    return {
        "records": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "log_loss": weighted_log_loss(y, prob, sample_weight),
        "accuracy": accuracy(y, prob),
        "auc": auc_score(y, prob),
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
    classes: list[str],
) -> tuple[np.ndarray, float, dict[str, Any], np.ndarray]:
    if int(np.sum(y == 1.0)) == 0 or int(np.sum(y == 0.0)) == 0:
        raise SystemExit("Contrastive trace dataset needs both success and non-success records.")
    train_idx, test_idx = split_indices(y, classes, test_fraction=test_fraction, seed=seed)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    train_weights = class_weights(y_train)
    test_weights = np.ones_like(y_test, dtype=np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    history: list[dict[str, float]] = []
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
    all_prob = sigmoid(x @ weights + bias)
    metrics = {
        "train": metric_block(y_train, train_prob, train_weights),
        "test": metric_block(y_test, test_prob, test_weights),
        "history": history,
    }
    return weights, bias, metrics, all_prob


def probability_summary(records: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for record, probability in zip(records, probabilities):
        grouped.setdefault(str(record.get("class_label", "failure")), []).append(float(probability))
    summary: dict[str, Any] = {}
    for class_label, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[class_label] = {
            "records": int(len(arr)),
            "mean_probability": float(np.mean(arr)),
            "median_probability": float(np.median(arr)),
            "max_probability": float(np.max(arr)),
            "min_probability": float(np.min(arr)),
        }
    if "success" in grouped and "near_miss" in grouped:
        summary["success_vs_near_miss_gap"] = (
            float(np.mean(grouped["success"])) - float(np.mean(grouped["near_miss"]))
        )
    if "success" in grouped and "failure" in grouped:
        summary["success_vs_failure_gap"] = (
            float(np.mean(grouped["success"])) - float(np.mean(grouped["failure"]))
        )
    return summary


def run(args: argparse.Namespace) -> None:
    records = read_jsonl(args.trace_jsonl)
    if not records:
        raise SystemExit("Trace dataset is empty.")
    classes = [str(record.get("class_label", "failure")) for record in records]
    class_counts = Counter(classes)
    x = np.vstack([vectorize_trace_record(record, hash_buckets=args.hash_buckets) for record in records])
    y = labels_from_records(records)
    weights, bias, metrics, probabilities = train_logistic(
        x,
        y,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        test_fraction=args.test_fraction,
        classes=classes,
    )
    payload = {
        "format_version": 1,
        "prototype": "CodePy trace-only contrastive prior",
        "guardrail": "Trains on safe DSL execution summaries only; no arbitrary generated source is executed.",
        "trace_jsonl": args.trace_jsonl,
        "objective": "Estimate P(trace | task) and separate exact argmax traces from value-only near misses.",
        "model": {
            "type": "logistic_regression",
            "hash_buckets": args.hash_buckets,
            "feature_names": trace_feature_names(args.hash_buckets),
            "weights": weights.tolist(),
            "bias": bias,
        },
        "metrics": metrics,
        "contrastive_separation": probability_summary(records, probabilities),
        "coverage": {
            "records": len(records),
            "class_counts": dict(class_counts),
            "positives": int(np.sum(y == 1.0)),
            "negatives": int(np.sum(y == 0.0)),
            "has_near_misses": class_counts.get("near_miss", 0) > 0,
        },
        "args": vars(args),
    }
    save_json(args.output_json, payload)
    sep = payload["contrastive_separation"]
    print("CodePy trace-prior training")
    print(f"  records: {len(records)}")
    print(f"  class_counts: {dict(class_counts)}")
    print(f"  output_json: {args.output_json}")
    print(f"  train_auc: {metrics['train']['auc']:.4g}")
    print(f"  test_auc: {metrics['test']['auc']:.4g}")
    if "success_vs_near_miss_gap" in sep:
        print(f"  success_vs_near_miss_gap: {sep['success_vs_near_miss_gap']:.4g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CodePy trace-only contrastive prior.")
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--hash-buckets", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--lr", type=float, default=0.28)
    parser.add_argument("--l2", type=float, default=1.0e-4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
