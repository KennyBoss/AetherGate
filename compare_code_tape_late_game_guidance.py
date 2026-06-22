#!/usr/bin/env python3
"""Compare CodePy survival beam search with and without the late-game ranker.

All variants share the same budget and the same survival + trace-prior +
value-rollout + optional prefix-stage configuration; the only per-variant toggle
is the late-game structural ranker (``--late-game-ranker-json``). This is the
equal-budget test of whether adding the late-game re-ranker autonomously discovers
the exact ``argmax_index4`` program without injecting the reference.

Guardrail: runs safe DSL trajectory searches only; no generated Python source is
executed and the reference program is never injected into the beam.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_cmd(command: list[str]) -> tuple[dict[str, Any], str, float]:
    start = time.perf_counter()
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit(
            "Command failed:\n" + " ".join(command) + "\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr
        )
    output_json = Path(command[command.index("--output-json") + 1])
    return json.loads(output_json.read_text(encoding="utf-8")), proc.stdout, elapsed


def build_search_args(
    args: argparse.Namespace,
    *,
    output_json: Path,
    candidate_jsonl: Path,
    late_game_ranker_json: str | None,
    late_game_weight: float | None,
) -> list[str]:
    command = [
        sys.executable,
        "search_code_tape_trajectory.py",
        "--target", args.target,
        "--program-length", str(args.program_length),
        "--max-steps", str(args.max_steps),
        "--cases", str(args.cases),
        "--beam-width", str(args.beam_width),
        "--signature-cases", str(args.signature_cases),
        "--signature-steps", str(args.signature_steps),
        "--diversity-penalty", str(args.diversity_penalty),
        "--max-same-signature", str(args.max_same_signature),
        "--selection-mode", args.selection_mode,
        "--mse-weight", str(args.mse_weight),
        "--length-penalty", str(args.length_penalty),
        "--value-rollouts", str(args.value_rollouts),
        "--value-rollout-top-k", str(args.value_rollout_top_k),
        "--value-rollout-role-top-k", str(args.value_rollout_role_top_k),
        "--value-rollout-max-depth", str(args.value_rollout_max_depth),
        "--value-weight", str(args.value_weight),
        "--prefix-stage-lane-fraction", str(args.prefix_stage_lane_fraction),
        "--survival-lane-fraction", str(args.survival_lane_fraction),
        "--survival-ttl", str(args.survival_ttl),
        "--survival-min-value", str(args.survival_min_value),
        "--survival-max-per-root", str(args.survival_max_per_root),
        "--candidate-jsonl", str(candidate_jsonl),
        "--output-json", str(output_json),
    ]
    if args.survival_ignore_signature_limit:
        command.append("--survival-ignore-signature-limit")
    if args.prefix_stage_ignore_signature_limit:
        command.append("--prefix-stage-ignore-signature-limit")
    if args.stop_on_solution:
        command.append("--stop-on-solution")
    if args.trace_prior_json:
        command.extend(["--trace-prior-json", args.trace_prior_json, "--trace-prior-weight", str(args.trace_prior_weight)])
    if late_game_ranker_json:
        command.extend(
            [
                "--late-game-ranker-json", late_game_ranker_json,
                "--late-game-weight", str(late_game_weight),
                "--late-game-lane-fraction", str(args.late_game_lane_fraction),
                "--late-game-min-remaining", str(args.late_game_min_remaining),
            ]
        )
    else:
        # keep the late-game lane disabled for the baseline budget
        command.extend(["--late-game-lane-fraction", "0.0", "--late-game-weight", "0.0"])
    return command


def summarize_run(name: str, payload: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    best = payload["best"]
    graph = payload["trajectory_inference"]["latent_trace_graph"]
    search = payload["search"]
    first_solution = payload.get("first_solution")
    return {
        "name": name,
        "solved": first_solution is not None,
        "best_class_label": best.get("class_label"),
        "best_train_mse": float(best["train_mse"]),
        "best_holdout_mse": float(best["holdout_mse"]),
        "best_paired_running_hit": float(best.get("paired_running_hit", 0.0)),
        "best_late_game_score": best.get("late_game_score"),
        "best_argmax_role_score": best.get("argmax_role_score"),
        "evaluated_programs": int(search["evaluated_programs"]),
        "total_evaluated_programs": int(search.get("total_evaluated_programs", search["evaluated_programs"])),
        "late_game_lane_selected_total": int(search.get("late_game_lane_selected_total", 0)),
        "prefix_stage_lane_selected_total": int(search.get("prefix_stage_lane_selected_total", 0)),
        "survival_selected_total": int(search.get("survival_selected_total", 0)),
        "graph_nodes": int(graph["nodes"]),
        "elapsed_s": elapsed_s,
        "program_ids": best["program_ids"],
        "compact_program_ids": best["compact_program_ids"],
        "first_solution_compact": (first_solution or {}).get("compact_program_ids"),
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = [
        {"name": "baseline_survival", "late_game_ranker_json": None, "late_game_weight": None}
    ]
    for weight in args.late_game_weight:
        variants.append(
            {
                "name": f"late_game_w{weight:g}",
                "late_game_ranker_json": args.late_game_ranker_json,
                "late_game_weight": weight,
            }
        )

    runs: list[dict[str, Any]] = []
    stdout_logs: dict[str, str] = {}
    for variant in variants:
        name = variant["name"]
        output_json = output_dir / f"{args.target}_{name}.json"
        candidate_jsonl = output_dir / f"{args.target}_{name}_candidates.jsonl"
        payload, stdout, elapsed_s = run_cmd(
            build_search_args(
                args,
                output_json=output_json,
                candidate_jsonl=candidate_jsonl,
                late_game_ranker_json=variant["late_game_ranker_json"],
                late_game_weight=variant["late_game_weight"],
            )
        )
        summary = summarize_run(name, payload, elapsed_s)
        summary["output_json"] = str(output_json)
        summary["late_game_weight"] = variant["late_game_weight"]
        runs.append(summary)
        stdout_logs[name] = stdout
        print(
            f"{name}: solved={summary['solved']} "
            f"mse={summary['best_train_mse']:.6g}/{summary['best_holdout_mse']:.6g} "
            f"paired={summary['best_paired_running_hit']:.3f} "
            f"prefix_stage_lane={summary['prefix_stage_lane_selected_total']} "
            f"late_game_lane={summary['late_game_lane_selected_total']} "
            f"eval={summary['evaluated_programs']}"
        )

    baseline = runs[0]
    comparisons = []
    for run_summary in runs[1:]:
        comparisons.append(
            {
                "name": run_summary["name"],
                "vs": baseline["name"],
                "same_budget": run_summary["evaluated_programs"] == baseline["evaluated_programs"],
                "same_total_budget": run_summary["total_evaluated_programs"] == baseline["total_evaluated_programs"],
                "solved_delta": int(run_summary["solved"]) - int(baseline["solved"]),
                "train_mse_delta": run_summary["best_train_mse"] - baseline["best_train_mse"],
                "holdout_mse_delta": run_summary["best_holdout_mse"] - baseline["best_holdout_mse"],
                "paired_running_hit_delta": run_summary["best_paired_running_hit"] - baseline["best_paired_running_hit"],
            }
        )
    any_solved = any(run_summary["solved"] for run_summary in runs[1:])
    baseline_solved = bool(baseline["solved"])
    payload = {
        "format_version": 1,
        "prototype": "CodePy late-game guidance comparison",
        "guardrail": (
            "Runs safe DSL survival searches only; no generated Python is executed and the reference "
            "program is never injected into the beam."
        ),
        "objective": (
            "Compare the shared Beam+Survival(+PrefixStage) setup against the same setup plus "
            "the Late-Game ranker on equal budget."
        ),
        "target": args.target,
        "late_game_ranker_json": args.late_game_ranker_json,
        "headline": {
            "baseline_solved": baseline_solved,
            "late_game_solved": bool(any_solved),
            "autonomous_breakthrough": bool(any_solved and not baseline_solved),
        },
        "runs": runs,
        "comparisons": comparisons,
        "stdout_logs": stdout_logs if args.include_stdout else {},
        "args": vars(args),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CodePy late-game guidance comparison")
    print(f"  output_json: {args.output_json}")
    print(f"  runs: {len(runs)}")
    print(f"  autonomous_breakthrough: {payload['headline']['autonomous_breakthrough']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CodePy survival search with late-game guidance.")
    parser.add_argument("--target", default="argmax_index4")
    parser.add_argument("--late-game-ranker-json", required=True)
    parser.add_argument("--late-game-weight", type=float, action="append", default=None)
    parser.add_argument("--late-game-lane-fraction", type=float, default=0.25)
    parser.add_argument("--late-game-min-remaining", type=int, default=2)
    parser.add_argument("--output-dir", default="artifacts/code_tape_prior/late_game_guidance")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--program-length", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--signature-cases", type=int, default=8)
    parser.add_argument("--signature-steps", type=int, default=18)
    parser.add_argument("--diversity-penalty", type=float, default=0.65)
    parser.add_argument("--max-same-signature", type=int, default=1)
    parser.add_argument("--selection-mode", choices=("diverse", "multilane"), default="diverse")
    parser.add_argument("--mse-weight", type=float, default=0.45)
    parser.add_argument("--length-penalty", type=float, default=1.0e-6)
    parser.add_argument("--value-rollouts", type=int, default=25)
    parser.add_argument("--value-rollout-top-k", type=int, default=128)
    parser.add_argument("--value-rollout-role-top-k", type=int, default=128)
    parser.add_argument("--value-rollout-max-depth", type=int, default=4)
    parser.add_argument("--value-weight", type=float, default=0.03)
    parser.add_argument("--prefix-stage-lane-fraction", type=float, default=0.0)
    parser.add_argument("--prefix-stage-ignore-signature-limit", action="store_true")
    parser.add_argument("--survival-lane-fraction", type=float, default=0.25)
    parser.add_argument("--survival-ttl", type=int, default=8)
    parser.add_argument("--survival-min-value", type=float, default=11.9)
    parser.add_argument("--survival-max-per-root", type=int, default=0)
    parser.add_argument("--survival-ignore-signature-limit", action="store_true")
    parser.add_argument("--trace-prior-json", default=None)
    parser.add_argument("--trace-prior-weight", type=float, default=0.25)
    parser.add_argument("--stop-on-solution", action="store_true")
    parser.add_argument("--include-stdout", action="store_true")
    args = parser.parse_args()
    if args.late_game_weight is None:
        args.late_game_weight = [0.25, 0.5, 1.0]
    return args


if __name__ == "__main__":
    run(parse_args())
