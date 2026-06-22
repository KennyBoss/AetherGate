#!/usr/bin/env python3
"""Offline re-ranking proof for the CodePy late-game structural ranker.

Loads a trained late-game ranker and re-scores the final-depth tape candidates
(plus the registered reference program) purely on control-flow structure.  The
report proves whether the ranker lifts the canonical argmax program above the
reward-exploit near-misses, and whether canonical-ordered programs outscore each
H4 structural-failure family (missing halt / swapped value-index update /
loop-test before first move).

This is an offline diagnostic over safe DSL summaries; no generated Python is
executed and the live beam search is not modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_code_tape_late_game import compact_ops, find_stage_positions, failure_stage
from code_tape_late_game_features import (
    OP_FIRST_MOVE,
    OP_HALT,
    OP_INDEX_UPDATE,
    OP_LOOP_TEST,
    OP_VALUE_UPDATE,
    first_positions,
    predict_late_game_score,
)
from code_tape_trace_prior import load_json, read_jsonl, safe_float, save_json
from evolve_code_tape import REFERENCE_PROGRAMS, TAPE_BLOCKS


def family_flags(ops: list[int]) -> dict[str, bool]:
    positions = first_positions(ops)
    missing_read_max = failure_stage(find_stage_positions(ops)) == "read_max"
    missing_halt = OP_HALT not in positions
    swap_update = (
        OP_VALUE_UPDATE in positions
        and OP_INDEX_UPDATE in positions
        and positions[OP_INDEX_UPDATE] < positions[OP_VALUE_UPDATE]
    )
    loop_before_move = (
        OP_LOOP_TEST in positions
        and OP_FIRST_MOVE in positions
        and positions[OP_LOOP_TEST] < positions[OP_FIRST_MOVE]
    )
    canonical_ordered = (
        not missing_halt
        and not swap_update
        and not loop_before_move
        and OP_VALUE_UPDATE in positions
        and OP_INDEX_UPDATE in positions
        and OP_FIRST_MOVE in positions
        and OP_LOOP_TEST in positions
    )
    return {
        "reward_exploit_missing_read_max": missing_read_max,
        "missing_halt": missing_halt,
        "swap_index_before_value_update": swap_update,
        "loop_test_before_first_move": loop_before_move,
        "canonical_ordered": canonical_ordered,
    }


def family_mean(rows: list[dict[str, Any]], flag: str) -> float | None:
    scores = [row["structural_score"] for row in rows if row["family"].get(flag)]
    if not scores:
        return None
    return sum(scores) / len(scores)


def collect_final_depth(candidates_jsonl: str, program_length: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for record in read_jsonl(candidates_jsonl):
        if int(record.get("depth", 0)) < program_length:
            continue
        ops = compact_ops(record)
        key = tuple(ops)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "ops": ops,
                "train_mse": safe_float(record.get("train_mse"), math.inf),
                "holdout_mse": safe_float(record.get("holdout_mse"), math.inf),
                "output_hit": safe_float(record.get("output_hit")),
                "survival_ttl": int(record.get("survival_ttl") or 0),
                "source": "candidate",
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    model = load_json(args.model_json)
    reference = [int(op) for op in REFERENCE_PROGRAMS.get(args.target, [])]
    if not reference:
        raise SystemExit(f"No reference program registered for target {args.target!r}.")

    rows = collect_final_depth(args.candidates_jsonl, args.program_length)
    candidate_keys = {tuple(row["ops"]) for row in rows}
    reference_present_in_candidates = tuple(reference) in candidate_keys
    # Inject the registered reference as an explicit candidate for the lift test.
    rows.append(
        {
            "ops": list(reference),
            "train_mse": 0.0,
            "holdout_mse": 0.0,
            "output_hit": 1.0,
            "survival_ttl": 0,
            "source": "reference_program",
        }
    )

    for row in rows:
        row["structural_score"] = predict_late_game_score(model, row["ops"])
        row["family"] = family_flags(row["ops"])
    rows.sort(key=lambda row: (-row["structural_score"], row["train_mse"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    reference_row = next(row for row in rows if row["source"] == "reference_program")
    reference_rank = reference_row["rank"]
    reference_score = reference_row["structural_score"]

    reward_exploit_scores = [
        row["structural_score"]
        for row in rows
        if row["source"] == "candidate" and row["family"]["reward_exploit_missing_read_max"]
    ]
    best_reward_exploit = max(reward_exploit_scores) if reward_exploit_scores else None
    best_candidate_score = max(
        (row["structural_score"] for row in rows if row["source"] == "candidate"),
        default=None,
    )
    lift_achieved = reference_rank == 1
    lift_over_reward_exploit = (
        best_reward_exploit is None or reference_score > best_reward_exploit
    )

    canonical_mean = family_mean(rows, "canonical_ordered")
    family_gaps: dict[str, Any] = {}
    for flag in ["missing_halt", "swap_index_before_value_update", "loop_test_before_first_move"]:
        broken_mean = family_mean(rows, flag)
        gap = None
        if canonical_mean is not None and broken_mean is not None:
            gap = canonical_mean - broken_mean
        family_gaps[flag] = {"broken_mean": broken_mean, "gap_vs_canonical": gap}

    top_rows = [
        {
            "rank": row["rank"],
            "source": row["source"],
            "structural_score": row["structural_score"],
            "train_mse": row["train_mse"],
            "output_hit": row["output_hit"],
            "ops": row["ops"],
            "blocks": [TAPE_BLOCKS[op] for op in row["ops"]],
            "family": row["family"],
        }
        for row in rows[: args.top_n]
    ]

    payload = {
        "format_version": 1,
        "prototype": "CodePy late-game structural re-ranking proof",
        "guardrail": (
            "Offline re-ranking of safe DSL candidate summaries with a learned structural "
            "score; the live beam search is unchanged and no generated Python is executed."
        ),
        "target": args.target,
        "model_json": args.model_json,
        "candidates_jsonl": args.candidates_jsonl,
        "reference": {
            "program_ids": reference,
            "program_blocks": [TAPE_BLOCKS[op] for op in reference],
            "present_in_candidates": reference_present_in_candidates,
            "rank": reference_rank,
            "structural_score": reference_score,
        },
        "lift": {
            "lift_achieved": bool(lift_achieved),
            "reference_rank": reference_rank,
            "reference_structural_score": reference_score,
            "best_candidate_structural_score": best_candidate_score,
            "best_reward_exploit_structural_score": best_reward_exploit,
            "lift_over_reward_exploit": bool(lift_over_reward_exploit),
            "reward_exploit_candidates": len(reward_exploit_scores),
        },
        "canonical_mean_structural_score": canonical_mean,
        "family_gaps": family_gaps,
        "records": {
            "final_depth_candidates": len(rows) - 1,
            "ranked_rows": len(rows),
            "top_n": args.top_n,
        },
        "top_programs": top_rows,
        "args": vars(args),
    }
    save_json(args.output_json, payload)
    if args.output_md:
        write_markdown(Path(args.output_md), payload)

    print("CodePy late-game structural re-ranking proof")
    print(f"  output_json: {args.output_json}")
    if args.output_md:
        print(f"  output_md: {args.output_md}")
    print(f"  reference_rank: {reference_rank} (score {reference_score:.4g})")
    print(f"  lift_achieved: {lift_achieved}")
    print(f"  lift_over_reward_exploit: {lift_over_reward_exploit}")
    if canonical_mean is not None:
        print(f"  canonical_mean: {canonical_mean:.4g}")
    for flag, info in family_gaps.items():
        print(f"  gap_vs_{flag}: {info['gap_vs_canonical']}")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    reference = payload["reference"]
    lift = payload["lift"]
    lines = [
        "# CodePy Late-Game Structural Re-Ranking Proof",
        "",
        "Offline re-ranking of final-depth tape candidates with the learned structural",
        "ranker. No generated Python is executed; the live beam search is unchanged.",
        "",
        "## Summary",
        "",
        f"- Target: `{payload['target']}`",
        f"- Reference program: `{reference['program_ids']}`",
        f"- Reference rank: {reference['rank']} / {payload['records']['ranked_rows']} "
        f"(score {reference['structural_score']:.4g})",
        f"- Reference present among autonomous candidates: {reference['present_in_candidates']}",
        f"- lift_achieved (reference ranked #1): {lift['lift_achieved']}",
        f"- lift_over_reward_exploit: {lift['lift_over_reward_exploit']} "
        f"(best reward-exploit score {lift['best_reward_exploit_structural_score']})",
        f"- Canonical-ordered mean structural score: {payload['canonical_mean_structural_score']}",
        "",
        "## Family gaps (canonical minus broken family mean)",
        "",
    ]
    for flag, info in payload["family_gaps"].items():
        lines.append(f"- {flag}: gap {info['gap_vs_canonical']} (broken mean {info['broken_mean']})")
    lines.extend(["", "## Top programs", ""])
    for row in payload["top_programs"][:12]:
        flags = ", ".join(name for name, value in row["family"].items() if value) or "none"
        lines.append(
            f"- #{row['rank']} [{row['source']}] score={row['structural_score']:.4g} "
            f"mse={row['train_mse']:.4g} ops={row['ops']} flags={flags}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline re-ranking proof for the late-game ranker.")
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--target", default="argmax_index4")
    parser.add_argument("--program-length", type=int, default=11)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
