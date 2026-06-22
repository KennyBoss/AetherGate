#!/usr/bin/env python3
"""Analyze macro pruning and regularization from Stage 4B results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare_code_block_library import save_json


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def macro_cost(depth: int) -> int:
    return max(depth, 0)


def utility(row: dict[str, Any], penalty: float) -> float:
    return float(row["success_rate"]) - penalty * macro_cost(int(row["depth"]))


def best_by_utility(rows: list[dict[str, Any]], penalty: float) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            utility(row, penalty),
            row["success_rate"],
            -(row["mean_generation_found"] if row["mean_generation_found"] is not None else 1.0e9),
            -row["depth"],
        ),
    )


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_gen = other["mean_generation_found"]
            row_gen = row["mean_generation_found"]
            other_blocks = other["mean_minimal_blocks"]
            row_blocks = row["mean_minimal_blocks"]
            if other_gen is None:
                other_gen = 1.0e9
            if row_gen is None:
                row_gen = 1.0e9
            if other_blocks is None:
                other_blocks = 1.0e9
            if row_blocks is None:
                row_blocks = 1.0e9
            at_least_as_good = (
                other["success_rate"] >= row["success_rate"]
                and other["depth"] <= row["depth"]
                and other_gen <= row_gen
                and other_blocks <= row_blocks
            )
            strictly_better = (
                other["success_rate"] > row["success_rate"]
                or other["depth"] < row["depth"]
                or other_gen < row_gen
                or other_blocks < row_blocks
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: row["depth"])


def analyze_target(item: dict[str, Any], penalties: list[float]) -> dict[str, Any]:
    rows = item["depths"]
    best_success = max(rows, key=lambda row: row["success_rate"])
    best_generation = min(
        (row for row in rows if row["mean_generation_found"] is not None),
        key=lambda row: row["mean_generation_found"],
        default=None,
    )
    frontier = pareto_frontier(rows)
    penalty_choices = []
    for penalty in penalties:
        best = best_by_utility(rows, penalty)
        penalty_choices.append(
            {
                "penalty": penalty,
                "chosen_depth": best["depth"],
                "chosen_profile": best["profile"],
                "success_rate": best["success_rate"],
                "mean_generation_found": best["mean_generation_found"],
                "mean_minimal_blocks": best["mean_minimal_blocks"],
                "utility": utility(best, penalty),
            }
        )
    noisy_depths = []
    running_best = -1.0
    for row in rows:
        if row["success_rate"] < running_best:
            noisy_depths.append(
                {
                    "depth": row["depth"],
                    "profile": row["profile"],
                    "success_rate": row["success_rate"],
                    "previous_best_success_rate": running_best,
                    "drop": running_best - row["success_rate"],
                }
            )
        running_best = max(running_best, row["success_rate"])

    return {
        "target": item["target"],
        "label": item["label"],
        "best_success_depth": best_success["depth"],
        "best_success_rate": best_success["success_rate"],
        "best_generation_depth": best_generation["depth"] if best_generation else None,
        "best_mean_generation_found": (
            best_generation["mean_generation_found"] if best_generation else None
        ),
        "pareto_frontier": [
            {
                "depth": row["depth"],
                "profile": row["profile"],
                "success_rate": row["success_rate"],
                "mean_generation_found": row["mean_generation_found"],
                "mean_minimal_blocks": row["mean_minimal_blocks"],
            }
            for row in frontier
        ],
        "penalty_choices": penalty_choices,
        "noisy_depths": noisy_depths,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# CodePy Macro Pruning Analysis",
        "",
        "This report analyzes Stage 4B success-rate curves with a simple macro-regularized utility:",
        "",
        "`utility = success_rate - penalty * hierarchy_depth`",
        "",
    ]
    for item in payload["targets"]:
        lines.append(f"## {item['label']}")
        lines.append("")
        lines.append(
            f"- Best success depth: {item['best_success_depth']} "
            f"({item['best_success_rate']:.3f})"
        )
        if item["best_generation_depth"] is not None:
            lines.append(
                f"- Fastest successful depth: {item['best_generation_depth']} "
                f"(mean generation {item['best_mean_generation_found']:.3f})"
            )
        if item["noisy_depths"]:
            noisy = ", ".join(
                f"depth {row['depth']} drop {row['drop']:.3f}" for row in item["noisy_depths"]
            )
            lines.append(f"- Noisy/degrading depths: {noisy}")
        else:
            lines.append("- Noisy/degrading depths: none detected")
        lines.append("- Pareto frontier:")
        for row in item["pareto_frontier"]:
            lines.append(
                f"  - depth {row['depth']} {row['profile']}: "
                f"success={row['success_rate']:.3f}, "
                f"mean_gen={row['mean_generation_found']}, "
                f"mean_blocks={row['mean_minimal_blocks']}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_penalties(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> None:
    stage4b = load_json(Path(args.stage4b_json))
    penalties = parse_penalties(args.penalties)
    targets = [analyze_target(item, penalties) for item in stage4b["summary"]]
    payload = {
        "format_version": 1,
        "experiment": "CodePy macro pruning and regularization analysis",
        "source_json": args.stage4b_json,
        "regularization": {
            "utility": "success_rate - penalty * hierarchy_depth",
            "penalties": penalties,
        },
        "targets": targets,
        "status": {
            "macro_pruning": "analysis_pass",
            "interpretation": (
                "Pruning can select shallower or non-final depths when later macro additions "
                "do not improve success rate, but this is an offline analysis over Stage 4B "
                "rather than an online evolutionary deletion mechanism."
            ),
        },
        "args": vars(args),
    }
    save_json(Path(args.output_json), payload)
    if args.output_md:
        write_markdown(Path(args.output_md), payload)
    print("CodePy macro pruning analysis")
    for item in targets:
        noisy = ", ".join(f"d{row['depth']}" for row in item["noisy_depths"]) or "none"
        print(
            f"  {item['label']}: best_success_depth={item['best_success_depth']} "
            f"best_success={item['best_success_rate']:.3f} noisy_depths={noisy}"
        )
    print(f"  output_json: {args.output_json}")
    if args.output_md:
        print(f"  output_md: {args.output_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CodePy macro pruning from Stage 4B.")
    parser.add_argument("--stage4b-json", default="artifacts/code_block_stage4b/stage4b.json")
    parser.add_argument("--penalties", default="0,0.005,0.01,0.02,0.05")
    parser.add_argument("--output-json", default="artifacts/code_block_stage4c/pruning.json")
    parser.add_argument("--output-md", default="artifacts/code_block_stage4c/pruning.md")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
