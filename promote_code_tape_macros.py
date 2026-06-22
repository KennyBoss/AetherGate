#!/usr/bin/env python3
"""Promote top CodePy tape traces into explicit macro candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from code_tape_prior import load_json, save_json
from evolve_code_tape import TAPE_BLOCKS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def candidate_key(record: dict[str, Any]) -> tuple[int, ...]:
    compact = record.get("compact_program_ids") or [
        int(op) for op in record.get("program_ids", []) if int(op) != 0
    ]
    return tuple(int(op) for op in compact)


def candidate_score(record: dict[str, Any]) -> tuple[float, float, float, float]:
    train_mse = float(record.get("train_mse", 1.0e12))
    holdout_mse = float(record.get("holdout_mse", 1.0e12))
    prior = float(record.get("prior_probability", 0.0))
    reward = float(record.get("reward", -1.0e12))
    active = float(record.get("active_blocks", len(candidate_key(record))))
    return (-train_mse - holdout_mse, prior, reward, -active)


def collect_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_text in args.trace_jsonl:
        records.extend(read_jsonl(Path(path_text)))
    for path_text in args.search_json:
        payload = load_json(path_text)
        for key in ("best", "first_solution", "reference_program"):
            value = payload.get(key)
            if isinstance(value, dict):
                records.append(value)
        for layer in payload.get("layers", []):
            if isinstance(layer, dict):
                records.append(
                    {
                        "program_ids": layer.get("best_prefix_ids", []),
                        "compact_program_ids": layer.get("best_prefix_ids", []),
                        "reward": layer.get("best_reward", 0.0),
                        "prior_probability": layer.get("best_prior_probability", 0.0),
                        "train_mse": layer.get("best_train_mse", 1.0e12),
                        "holdout_mse": layer.get("best_holdout_mse", 1.0e12),
                    }
                )
    return records


def promote(args: argparse.Namespace) -> dict[str, Any]:
    records = collect_records(args)
    grouped: dict[tuple[int, ...], dict[str, Any]] = {}
    for record in records:
        key = candidate_key(record)
        if not key:
            continue
        train_mse = float(record.get("train_mse", 1.0e12))
        holdout_mse = float(record.get("holdout_mse", 1.0e12))
        if train_mse > args.max_mse or holdout_mse > args.max_mse:
            continue
        existing = grouped.get(key)
        if existing is None or candidate_score(record) > candidate_score(existing):
            grouped[key] = record
    ranked = sorted(grouped.items(), key=lambda item: candidate_score(item[1]), reverse=True)
    macros = []
    for macro_id, (program_ids, record) in enumerate(ranked[: args.top_k]):
        macros.append(
            {
                "id": macro_id,
                "name": f"macro_{macro_id:02d}_{'_'.join(str(op) for op in program_ids)}",
                "program_ids": list(program_ids),
                "program_blocks": [TAPE_BLOCKS[op] for op in program_ids],
                "source": record.get("source", "trace_or_search"),
                "train_mse": float(record.get("train_mse", 0.0)),
                "holdout_mse": float(record.get("holdout_mse", 0.0)),
                "reward": float(record.get("reward", 0.0)),
                "prior_probability": float(record.get("prior_probability", 0.0)),
                "active_blocks": int(record.get("active_blocks", len(program_ids))),
            }
        )
    return {
        "format_version": 1,
        "prototype": "CodePy tape automatic macro promotion",
        "guardrail": "Promoted macros are explicit safe DSL token sequences, not generated Python source.",
        "selection": {
            "top_k": args.top_k,
            "max_mse": args.max_mse,
            "input_records": len(records),
            "eligible_unique_programs": len(ranked),
        },
        "macros": macros,
        "args": vars(args),
    }


def run(args: argparse.Namespace) -> None:
    payload = promote(args)
    save_json(args.output_json, payload)
    print("CodePy tape macro promotion")
    print(f"  input_records: {payload['selection']['input_records']}")
    print(f"  macros: {len(payload['macros'])}")
    print(f"  output_json: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote CodePy tape traces into macro candidates.")
    parser.add_argument("--trace-jsonl", action="append", default=[])
    parser.add_argument("--search-json", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-mse", type=float, default=1.0e-8)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
