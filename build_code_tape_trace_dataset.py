#!/usr/bin/env python3
"""Build a contrastive trace dataset for CodePy tape trajectory priors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from code_tape_trace_prior import (
    append_jsonl,
    classify_trace_record,
    read_jsonl,
    save_json,
)


def load_reference_records(path: Path, success_mse: float) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload.get("args", {})
    target = payload.get("target", {}).get("name", args.get("target", "unknown"))
    records: list[dict[str, Any]] = []
    for key, source in [
        ("reference_program", "reference_program"),
        ("first_solution", "seeded_first_solution"),
    ]:
        row = payload.get(key)
        if not isinstance(row, dict):
            continue
        program_ids = row.get("program_ids") or row.get("compact_program_ids") or []
        record = {
            **row,
            "source": source,
            "target": target,
            "depth": int(args.get("program_length", len(program_ids) or 1)),
            "prefix_ids": list(program_ids),
            "program_ids": list(program_ids),
            "seeded_reference_prefix": source != "reference_program",
            "args": {
                "target": target,
                "program_length": int(args.get("program_length", len(program_ids) or 1)),
                "max_steps": int(args.get("max_steps", 64)),
                "tape_len": int(args.get("tape_len", 8)),
                "input_len": int(args.get("input_len", 4)),
                "cases": int(args.get("cases", 32)),
                "backend": args.get("backend", "cpu"),
            },
        }
        record.update(classify_trace_record(record, success_mse=success_mse))
        records.append(record)
    return records


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    for record in records:
        key = (
            str(record.get("source", "")),
            tuple(int(op) for op in record.get("program_ids", []) or []),
            str(record.get("signature", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def balance_records(
    records: list[dict[str, Any]],
    *,
    max_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_per_class <= 0:
        return records
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("class_label", "failure")), []).append(record)
    balanced: list[dict[str, Any]] = []
    for label, rows in grouped.items():
        if len(rows) <= max_per_class:
            balanced.extend(rows)
            continue
        # Keep the most informative near misses: high value match but imperfect
        # paired state.  For other classes, sample deterministically.
        if label == "near_miss":
            ranked = sorted(
                rows,
                key=lambda r: (
                    float(r.get("max_hit", 0.0)),
                    float(r.get("running_value_hit", 0.0)),
                    -float(r.get("paired_running_hit", 0.0)),
                    -float(r.get("train_mse", math.inf)),
                ),
                reverse=True,
            )
            balanced.extend(ranked[:max_per_class])
        else:
            indices = rng.choice(len(rows), size=max_per_class, replace=False)
            balanced.extend(rows[int(i)] for i in sorted(indices))
    return sorted(
        balanced,
        key=lambda r: (
            str(r.get("class_label", "")),
            int(r.get("depth", 0)),
            str(r.get("signature", "")),
            r.get("program_ids", []),
        ),
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record.get("class_label", "failure")) for record in records)
    sources = Counter(str(record.get("source", "unknown")) for record in records)
    near_misses = [record for record in records if record.get("class_label") == "near_miss"]
    successes = [record for record in records if record.get("class_label") == "success"]
    failures = [record for record in records if record.get("class_label") == "failure"]

    def mean_metric(rows: list[dict[str, Any]], name: str) -> float:
        if not rows:
            return 0.0
        return float(np.mean([float(row.get(name, 0.0)) for row in rows]))

    return {
        "records": len(records),
        "class_counts": dict(counts),
        "source_counts": dict(sources),
        "mean_metrics_by_class": {
            label: {
                metric: mean_metric(rows, metric)
                for metric in [
                    "output_hit",
                    "max_hit",
                    "index_hit",
                    "running_value_hit",
                    "running_index_hit",
                    "paired_running_hit",
                    "train_mse",
                    "holdout_mse",
                ]
            }
            for label, rows in [
                ("success", successes),
                ("near_miss", near_misses),
                ("failure", failures),
            ]
        },
    }


def run(args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    for path in args.input_jsonl:
        for record in read_jsonl(path):
            record = dict(record)
            record.update(classify_trace_record(record, success_mse=args.success_mse))
            records.append(record)
    for path in args.reference_json:
        records.extend(load_reference_records(Path(path), args.success_mse))
    if args.dedupe:
        records = dedupe_records(records)
    records = balance_records(records, max_per_class=args.max_per_class, seed=args.seed)
    if not records:
        raise SystemExit("No records found for contrastive trace dataset.")
    success_count = sum(1 for record in records if record.get("class_label") == "success")
    near_miss_count = sum(1 for record in records if record.get("class_label") == "near_miss")
    failure_count = sum(1 for record in records if record.get("class_label") == "failure")
    if success_count < args.min_successes:
        raise SystemExit(f"Need at least {args.min_successes} success records, found {success_count}.")
    if near_miss_count < args.min_near_misses:
        raise SystemExit(f"Need at least {args.min_near_misses} near-miss records, found {near_miss_count}.")
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).write_text("", encoding="utf-8")
    append_jsonl(args.output_jsonl, records)
    summary = {
        "format_version": 1,
        "prototype": "CodePy contrastive trace dataset",
        "guardrail": "Dataset contains safe DSL execution summaries only; no arbitrary generated source is executed.",
        "input_jsonl": args.input_jsonl,
        "reference_json": args.reference_json,
        "labeling": {
            "success_mse": args.success_mse,
            "class_labels": ["success", "near_miss", "failure"],
            "near_miss_definition": (
                "non-zero-error traces that look value-like, especially max/running-value hits, "
                "but do not preserve the complete argmax value-index pair"
            ),
        },
        "dataset": summarize(records),
        "args": vars(args),
    }
    save_json(args.summary_json, summary)
    print("CodePy contrastive trace dataset")
    print(f"  output_jsonl: {args.output_jsonl}")
    print(f"  summary_json: {args.summary_json}")
    print(f"  records: {len(records)}")
    print(f"  success/near_miss/failure: {success_count}/{near_miss_count}/{failure_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CodePy contrastive trace dataset.")
    parser.add_argument("--input-jsonl", action="append", default=[])
    parser.add_argument("--reference-json", action="append", default=[])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--success-mse", type=float, default=1.0e-8)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--min-successes", type=int, default=1)
    parser.add_argument("--min-near-misses", type=int, default=1)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
