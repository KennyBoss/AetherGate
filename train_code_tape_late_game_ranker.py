#!/usr/bin/env python3
"""Train the CodePy late-game structural ranker for argmax-style tape programs.

The H4 late-game analysis localized the ``argmax_index4`` bottleneck to
control-flow *ordering*: survival-protected near-misses already collect the
canonical argmax blocks but place loop/update/return/halt in the wrong order.
This trainer learns a lightweight logistic ranker over **structural opcode
features** (``code_tape_late_game_features``) that separates verified-canonical
programs from those structural-failure families.

Dataset (built internally):
  * positives  : verified zero-error programs from ``--search-json``
                 (``reference_program`` / ``first_solution``).
  * negatives  : (a) structurally-broken final-depth near-misses from the
                 candidate JSONL, capped to the most informative rows, and
                 (b) synthetic reference corruptions per the H4 failure families
                 (drop halt, swap value/index update, loop-test before move).

Guardrail: features come from the registered safe-DSL reference program; the
decision is a learned logistic score, never a hardcoded opcode order; no
generated Python is executed.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_code_tape_trace_dataset import balance_records, load_reference_records
from code_tape_late_game_features import (
    ops_from_record,
    reference_corruptions,
    structural_feature_names,
    structural_feature_vector,
)
from code_tape_trace_prior import (
    classify_trace_record,
    read_jsonl,
    safe_float,
    save_json,
)
from evolve_code_tape import REFERENCE_PROGRAMS
from train_code_tape_trace_prior import probability_summary, train_logistic


def collect_positives(search_json: str, success_mse: float) -> list[dict[str, Any]]:
    records = load_reference_records(Path(search_json), success_mse)
    positives = [row for row in records if row.get("class_label") == "success"]
    for row in positives:
        row["late_game_class"] = "success"
    return positives


def collect_near_miss_negatives(
    candidates_jsonl: str,
    *,
    program_length: int,
    success_mse: float,
    max_negatives: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(candidates_jsonl):
        if int(record.get("depth", 0)) < program_length:
            continue
        if safe_float(record.get("train_mse"), math.inf) <= success_mse:
            continue
        record = dict(record)
        record.update(classify_trace_record(record, success_mse=success_mse))
        rows.append(record)
    rows = balance_records(rows, max_per_class=max_negatives, seed=seed)
    for row in rows:
        row["late_game_class"] = str(row.get("class_label", "failure"))
    return rows


def collect_synthetic_negatives(reference: list[int], target: str, program_length: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for corruption in reference_corruptions(reference):
        program_ids = [int(op) for op in corruption["program_ids"]]
        records.append(
            {
                "source": f"synthetic_corruption:{corruption['name']}",
                "corruption_name": corruption["name"],
                "target": target,
                "depth": program_length,
                "program_ids": program_ids,
                "compact_program_ids": program_ids,
                "class_label": "synthetic_negative",
                "late_game_class": "synthetic_negative",
                "binary_label": 0,
            }
        )
    return records


def run(args: argparse.Namespace) -> None:
    reference = [int(op) for op in REFERENCE_PROGRAMS.get(args.target, [])]
    if not reference:
        raise SystemExit(f"No reference program registered for target {args.target!r}.")
    feature_names = structural_feature_names(reference)

    positives = collect_positives(args.search_json, args.success_mse)
    if not positives:
        raise SystemExit("No verified-success positive program found in --search-json.")
    near_miss = collect_near_miss_negatives(
        args.candidates_jsonl,
        program_length=args.program_length,
        success_mse=args.success_mse,
        max_negatives=args.max_negatives,
        seed=args.seed,
    )
    synthetic = collect_synthetic_negatives(reference, args.target, args.program_length)

    records = positives + near_miss + synthetic
    x = np.vstack([structural_feature_vector(ops_from_record(row), reference) for row in records])
    y = np.asarray([1.0 if row.get("late_game_class") == "success" else 0.0 for row in records])
    classes = [str(row.get("late_game_class", "failure")) for row in records]

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

    # contrastive separation uses each record's class label; map late_game_class
    # so probability_summary reports success-vs-near_miss style gaps.
    for row, probability in zip(records, probabilities):
        row["class_label"] = str(row.get("late_game_class", "failure"))
    separation = probability_summary(records, probabilities)
    pos_mask = y == 1.0
    neg_mask = y == 0.0
    separation["positive_mean"] = float(np.mean(probabilities[pos_mask])) if pos_mask.any() else 0.0
    separation["negative_mean"] = float(np.mean(probabilities[neg_mask])) if neg_mask.any() else 0.0
    separation["positive_vs_negative_gap"] = separation["positive_mean"] - separation["negative_mean"]

    class_counts = Counter(classes)
    payload = {
        "format_version": 1,
        "prototype": "CodePy late-game structural ranker",
        "guardrail": (
            "Structural opcode features derived from the registered safe-DSL reference; "
            "the ranker learns a logistic score and hardcodes no opcode order; no "
            "generated Python is executed."
        ),
        "objective": (
            "Score whether a tape program assembles the canonical control-flow order "
            "(read/compare/update/loop/return/halt) versus the H4 structural-failure families."
        ),
        "target": args.target,
        "candidates_jsonl": args.candidates_jsonl,
        "search_json": args.search_json,
        "model": {
            "type": "logistic_regression",
            "feature_space": "structural_opcode_sequence",
            "reference": reference,
            "feature_names": feature_names,
            "weights": weights.tolist(),
            "bias": bias,
        },
        "metrics": metrics,
        "contrastive_separation": separation,
        "coverage": {
            "records": len(records),
            "class_counts": dict(class_counts),
            "positives": int(np.sum(pos_mask)),
            "negatives": int(np.sum(neg_mask)),
            "synthetic_negatives": int(class_counts.get("synthetic_negative", 0)),
            "near_miss_negatives": int(class_counts.get("near_miss", 0)),
            "failure_negatives": int(class_counts.get("failure", 0)),
        },
        "args": vars(args),
    }
    save_json(args.output_json, payload)

    print("CodePy late-game structural ranker training")
    print(f"  records: {len(records)} ({class_counts})")
    print(f"  features: {len(feature_names)}")
    print(f"  output_json: {args.output_json}")
    print(f"  train_auc: {metrics['train']['auc']:.4g}")
    print(f"  positive_mean: {separation['positive_mean']:.4g}")
    print(f"  negative_mean: {separation['negative_mean']:.4g}")
    print(f"  positive_vs_negative_gap: {separation['positive_vs_negative_gap']:.4g}")
    if "success_vs_near_miss_gap" in separation:
        print(f"  success_vs_near_miss_gap: {separation['success_vs_near_miss_gap']:.4g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CodePy late-game structural ranker.")
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--search-json", required=True)
    parser.add_argument("--target", default="argmax_index4")
    parser.add_argument("--program-length", type=int, default=11)
    parser.add_argument("--success-mse", type=float, default=1.0e-8)
    parser.add_argument("--max-negatives", type=int, default=1500)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--l2", type=float, default=1.0e-3)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
