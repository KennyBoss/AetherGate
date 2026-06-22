#!/usr/bin/env python3
"""Measure CodePy hierarchical synthesis across a growing task ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from compare_code_block_library import run_trial, save_json


TASK_LADDER = (
    {
        "stage": 1,
        "target": "square",
        "label": "square",
        "hierarchical_profile": "base",
        "program_length": 6,
        "promotes_profile": "stage4_after_square",
    },
    {
        "stage": 2,
        "target": "abs",
        "label": "abs",
        "hierarchical_profile": "stage4_after_square",
        "program_length": 6,
        "promotes_profile": "stage4_after_abs",
    },
    {
        "stage": 3,
        "target": "max_xy",
        "label": "max",
        "hierarchical_profile": "stage4_after_abs",
        "program_length": 8,
        "promotes_profile": "stage4_after_max",
    },
    {
        "stage": 4,
        "target": "clamp01",
        "label": "clamp",
        "hierarchical_profile": "stage4_after_max",
        "program_length": 8,
        "promotes_profile": "stage4_after_clamp",
    },
    {
        "stage": 5,
        "target": "max_abs_x_y",
        "label": "max(abs(x), y)",
        "hierarchical_profile": "stage4_after_clamp",
        "program_length": 8,
        "promotes_profile": "stage4_after_max_abs",
    },
    {
        "stage": 6,
        "target": "clamp_abs_01",
        "label": "clamp(abs(x),0,1)",
        "hierarchical_profile": "stage4_after_max_abs",
        "program_length": 8,
        "promotes_profile": "stage4_after_clamp_abs",
    },
    {
        "stage": 7,
        "target": "piecewise_max_abs_x_y",
        "label": "piecewise(max(abs(x),y))",
        "hierarchical_profile": "stage4_after_clamp_abs",
        "program_length": 10,
        "promotes_profile": None,
    },
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def task_aliases() -> dict[str, dict[str, Any]]:
    aliases = {task["target"]: task for task in TASK_LADDER}
    aliases.update({task["label"]: task for task in TASK_LADDER})
    aliases["piecewise(max(abs(x),y))"] = TASK_LADDER[-1]
    return aliases


def parse_task_names(value: str) -> set[str]:
    stripped = value.strip()
    if not stripped:
        return set()
    aliases = task_aliases()
    if stripped in aliases:
        return {stripped}
    return set(parse_csv(value))


def selected_tasks(task_names: set[str]) -> list[dict[str, Any]]:
    if not task_names:
        return [dict(task) for task in TASK_LADDER]
    tasks = []
    aliases = task_aliases()
    valid = set(aliases)
    unknown = sorted(task_names - valid)
    if unknown:
        raise SystemExit(f"Unknown task(s): {', '.join(unknown)}")
    for task in TASK_LADDER:
        if any(aliases[name]["target"] == task["target"] for name in task_names):
            tasks.append(dict(task))
    return tasks


def mean(values: list[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record["success"]]
    divisor = max(len(records), 1)
    return {
        "trials": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / divisor,
        "mean_generation_found": mean([record["generation_found"] for record in records]),
        "mean_active_blocks": mean([record["search_blocks"] for record in records]),
        "mean_minimal_blocks": mean([record["minimal_blocks"] for record in records]),
        "mean_compression_ratio": mean([record["compression_ratio"] for record in records]),
        "mean_search_time_s": mean([record["evolution_s"] for record in records]),
        "mean_evaluated_programs": mean([record["evaluated_programs"] for record in records]),
    }


def delta(base: float | None, hierarchical: float | None) -> float | None:
    if base is None or hierarchical is None:
        return None
    return base - hierarchical


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def summarize_ladder(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for stage in sorted({record["stage"] for record in records}):
        stage_records = [record for record in records if record["stage"] == stage]
        base_records = [record for record in stage_records if record["comparison"] == "base"]
        hierarchical_records = [
            record for record in stage_records if record["comparison"] == "hierarchical"
        ]
        base = summarize_group(base_records)
        hierarchical = summarize_group(hierarchical_records)
        first = stage_records[0]
        hierarchy_solved_base_failed = (
            hierarchical["success_rate"] > 0.0 and base["success_rate"] < hierarchical["success_rate"]
        )
        summaries.append(
            {
                "stage": stage,
                "target": first["target"],
                "label": first["label"],
                "hierarchical_profile": first["hierarchical_profile"],
                "base": base,
                "hierarchical": hierarchical,
                "hierarchical_vs_base": {
                    "success_rate_delta": hierarchical["success_rate"] - base["success_rate"],
                    "hierarchy_solved_base_failed": hierarchy_solved_base_failed,
                    "generation_gain": delta(
                        base["mean_generation_found"], hierarchical["mean_generation_found"]
                    ),
                    "active_block_gain": delta(
                        base["mean_active_blocks"], hierarchical["mean_active_blocks"]
                    ),
                    "minimal_block_gain": delta(
                        base["mean_minimal_blocks"], hierarchical["mean_minimal_blocks"]
                    ),
                    "compression_ratio_delta": delta(
                        hierarchical["mean_compression_ratio"], base["mean_compression_ratio"]
                    ),
                    "search_time_speedup": ratio(
                        base["mean_search_time_s"], hierarchical["mean_search_time_s"]
                    ),
                    "evaluated_program_reduction": (
                        1.0
                        - hierarchical["mean_evaluated_programs"] / base["mean_evaluated_programs"]
                        if base["mean_evaluated_programs"] not in (None, 0)
                        and hierarchical["mean_evaluated_programs"] is not None
                        else None
                    ),
                },
            }
        )
    return summaries


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    task_names = parse_task_names(args.tasks)
    tasks = selected_tasks(task_names)

    records: list[dict[str, Any]] = []
    print("TextPy/SoA CodePy Stage 4 hierarchical synthesis")
    print(f"tasks: {', '.join(task['label'] for task in tasks)}")
    print(f"seeds: {', '.join(map(str, seeds))}")
    print("")

    for task in tasks:
        for comparison, profile in (
            ("base", "base"),
            ("hierarchical", task["hierarchical_profile"]),
        ):
            for seed in seeds:
                print(
                    f"stage={task['stage']} target={task['target']} "
                    f"comparison={comparison} profile={profile} seed={seed}"
                )
                record = run_trial(
                    python=args.python,
                    output_dir=output_dir / "trials",
                    target=task["target"],
                    profile=profile,
                    seed=seed,
                    population=args.population,
                    generations=args.generations,
                    elites=args.elites,
                    program_length=task["program_length"],
                    cases=args.cases,
                    timeout=args.timeout,
                )
                record.update(
                    {
                        "stage": task["stage"],
                        "label": task["label"],
                        "comparison": comparison,
                        "hierarchical_profile": task["hierarchical_profile"],
                        "promotes_profile": task["promotes_profile"],
                        "program_length": task["program_length"],
                    }
                )
                records.append(record)
                print(
                    f"  success={record['success']} "
                    f"generation_found={record['generation_found']} "
                    f"active_blocks={record['search_blocks']} "
                    f"minimal_blocks={record['minimal_blocks']} "
                    f"compression={record['compression_ratio']:.3f} "
                    f"search_time_s={record['evolution_s']:.4f}"
                )

    summary = summarize_ladder(records)
    payload = {
        "format_version": 1,
        "experiment": "CodePy Stage 4 hierarchical synthesis ladder",
        "guardrail": "Constrained register DSL search; no arbitrary generated Python is executed.",
        "ladder": tasks,
        "records": records,
        "summary": summary,
        "args": vars(args),
    }
    save_json(Path(args.output_json), payload)

    print("")
    print("Stage 4 Summary")
    for item in summary:
        gain = item["hierarchical_vs_base"]
        print(
            f"  stage={item['stage']} {item['label']}: "
            f"gen_gain={gain['generation_gain']} "
            f"minimal_gain={gain['minimal_block_gain']} "
            f"time_speedup={gain['search_time_speedup']}"
        )
    print(f"  output_json: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodePy Stage 4 hierarchical synthesis ladder.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seeds", default="31")
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--elites", type=int, default=128)
    parser.add_argument("--cases", type=int, default=49)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-dir", default="artifacts/code_block_hierarchy")
    parser.add_argument("--output-json", default="artifacts/code_block_hierarchy/hierarchy.json")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
