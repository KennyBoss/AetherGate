#!/usr/bin/env python3
"""Compare autonomous trajectory beam search with and without a trace prior."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_cmd(args: list[str]) -> tuple[dict[str, Any], str, float]:
    start = time.perf_counter()
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit(
            "Command failed:\n"
            + " ".join(args)
            + "\nSTDOUT:\n"
            + proc.stdout
            + "\nSTDERR:\n"
            + proc.stderr
        )
    output_json = Path(args[args.index("--output-json") + 1])
    return json.loads(output_json.read_text(encoding="utf-8")), proc.stdout, elapsed


def build_search_args(
    args: argparse.Namespace,
    *,
    output_json: Path,
    candidate_jsonl: Path,
    trace_prior_json: str | None = None,
    trace_prior_weight: float | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "search_code_tape_trajectory.py",
        "--target",
        args.target,
        "--program-length",
        str(args.program_length),
        "--max-steps",
        str(args.max_steps),
        "--cases",
        str(args.cases),
        "--beam-width",
        str(args.beam_width),
        "--signature-cases",
        str(args.signature_cases),
        "--signature-steps",
        str(args.signature_steps),
        "--diversity-penalty",
        str(args.diversity_penalty),
        "--max-same-signature",
        str(args.max_same_signature),
        "--selection-mode",
        args.selection_mode,
        "--raw-lane-fraction",
        str(args.raw_lane_fraction),
        "--prior-lane-fraction",
        str(args.prior_lane_fraction),
        "--role-lane-fraction",
        str(args.role_lane_fraction),
        "--behavior-lane-fraction",
        str(args.behavior_lane_fraction),
        "--mse-weight",
        str(args.mse_weight),
        "--length-penalty",
        str(args.length_penalty),
        "--value-rollouts",
        str(args.value_rollouts),
        "--value-rollout-top-k",
        str(args.value_rollout_top_k),
        "--value-rollout-role-top-k",
        str(args.value_rollout_role_top_k),
        "--value-rollout-max-depth",
        str(args.value_rollout_max_depth),
        "--value-weight",
        str(args.value_weight),
        "--value-lane-fraction",
        str(args.value_lane_fraction),
        "--value-rollout-pair-weight",
        str(args.value_rollout_pair_weight),
        "--value-rollout-output-weight",
        str(args.value_rollout_output_weight),
        "--value-rollout-role-weight",
        str(args.value_rollout_role_weight),
        "--value-rollout-prior-weight",
        str(args.value_rollout_prior_weight),
        "--survival-lane-fraction",
        str(args.survival_lane_fraction),
        "--survival-ttl",
        str(args.survival_ttl),
        "--survival-min-value",
        str(args.survival_min_value),
        "--survival-score-weight",
        str(args.survival_score_weight),
        "--survival-max-per-root",
        str(args.survival_max_per_root),
        "--candidate-jsonl",
        str(candidate_jsonl),
        "--output-json",
        str(output_json),
    ]
    if args.survival_ignore_signature_limit:
        command.append("--survival-ignore-signature-limit")
    if args.stop_on_solution:
        command.append("--stop-on-solution")
    if trace_prior_json:
        command.extend(
            [
                "--trace-prior-json",
                trace_prior_json,
                "--trace-prior-weight",
                str(trace_prior_weight),
            ]
        )
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
        "best_output_hit": float(best.get("output_hit", 0.0)),
        "best_max_hit": float(best.get("max_hit", 0.0)),
        "best_index_hit": float(best.get("index_hit", 0.0)),
        "best_paired_running_hit": float(best.get("paired_running_hit", 0.0)),
        "best_trace_prior_probability": best.get("trace_prior_probability"),
        "best_argmax_role_score": best.get("argmax_role_score"),
        "evaluated_programs": int(search["evaluated_programs"]),
        "value_rollout_evaluations": int(search.get("value_rollout_evaluations", 0)),
        "total_evaluated_programs": int(search.get("total_evaluated_programs", search["evaluated_programs"])),
        "survival_activations": int(search.get("survival_activations", 0)),
        "survival_selected_total": int(search.get("survival_selected_total", 0)),
        "survival_lane_fraction": float(search.get("survival_lane_fraction", 0.0)),
        "survival_ttl": int(search.get("survival_ttl", 0)),
        "survival_max_per_root": int(search.get("survival_max_per_root", 0)),
        "candidate_records_written": int(search.get("candidate_records_written", 0)),
        "graph_nodes": int(graph["nodes"]),
        "graph_edges": int(graph["edges"]),
        "selected_signature_entropy": float(graph["selected_signature_entropy"]),
        "elapsed_s": elapsed_s,
        "program_ids": best["program_ids"],
        "compact_program_ids": best["compact_program_ids"],
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        {
            "name": "baseline",
            "trace_prior_json": None,
            "trace_prior_weight": None,
        }
    ]
    for weight in args.trace_prior_weight:
        variants.append(
            {
                "name": f"trace_prior_w{weight:g}",
                "trace_prior_json": args.trace_prior_json,
                "trace_prior_weight": weight,
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
                trace_prior_json=variant["trace_prior_json"],
                trace_prior_weight=variant["trace_prior_weight"],
            )
        )
        summary = summarize_run(name, payload, elapsed_s)
        summary["output_json"] = str(output_json)
        summary["candidate_jsonl"] = str(candidate_jsonl)
        summary["trace_prior_weight"] = variant["trace_prior_weight"]
        runs.append(summary)
        stdout_logs[name] = stdout
        print(
            f"{name}: solved={summary['solved']} "
            f"mse={summary['best_train_mse']:.6g}/{summary['best_holdout_mse']:.6g} "
            f"paired={summary['best_paired_running_hit']:.3f} "
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
                "same_total_budget": (
                    run_summary["total_evaluated_programs"] == baseline["total_evaluated_programs"]
                ),
                "solved_delta": int(run_summary["solved"]) - int(baseline["solved"]),
                "train_mse_delta": run_summary["best_train_mse"] - baseline["best_train_mse"],
                "holdout_mse_delta": run_summary["best_holdout_mse"] - baseline["best_holdout_mse"],
                "paired_running_hit_delta": (
                    run_summary["best_paired_running_hit"] - baseline["best_paired_running_hit"]
                ),
            }
        )
    payload = {
        "format_version": 1,
        "prototype": "CodePy trace-prior guidance comparison",
        "guardrail": "Runs safe DSL trajectory searches only; no arbitrary generated Python source is executed.",
        "objective": "Compare autonomous beam search against beam + contrastive P(trace | task) on equal budget.",
        "target": args.target,
        "trace_prior_json": args.trace_prior_json,
        "runs": runs,
        "comparisons": comparisons,
        "stdout_logs": stdout_logs if args.include_stdout else {},
        "args": vars(args),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CodePy trace-guidance comparison")
    print(f"  output_json: {args.output_json}")
    print(f"  runs: {len(runs)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CodePy trajectory search with trace-prior guidance.")
    parser.add_argument("--target", default="argmax_index4")
    parser.add_argument("--trace-prior-json", required=True)
    parser.add_argument("--trace-prior-weight", type=float, action="append", default=None)
    parser.add_argument("--output-dir", default="artifacts/code_tape_prior/trace_guidance")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--program-length", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--signature-cases", type=int, default=8)
    parser.add_argument("--signature-steps", type=int, default=18)
    parser.add_argument("--diversity-penalty", type=float, default=0.65)
    parser.add_argument("--max-same-signature", type=int, default=1)
    parser.add_argument("--selection-mode", choices=("diverse", "multilane"), default="diverse")
    parser.add_argument("--raw-lane-fraction", type=float, default=0.45)
    parser.add_argument("--prior-lane-fraction", type=float, default=0.2)
    parser.add_argument("--role-lane-fraction", type=float, default=0.25)
    parser.add_argument("--behavior-lane-fraction", type=float, default=0.1)
    parser.add_argument("--mse-weight", type=float, default=0.45)
    parser.add_argument("--length-penalty", type=float, default=1.0e-6)
    parser.add_argument("--value-rollouts", type=int, default=0)
    parser.add_argument("--value-rollout-top-k", type=int, default=64)
    parser.add_argument("--value-rollout-role-top-k", type=int, default=64)
    parser.add_argument("--value-rollout-max-depth", type=int, default=4)
    parser.add_argument("--value-weight", type=float, default=0.15)
    parser.add_argument("--value-lane-fraction", type=float, default=0.0)
    parser.add_argument("--value-rollout-pair-weight", type=float, default=1.0)
    parser.add_argument("--value-rollout-output-weight", type=float, default=1.0)
    parser.add_argument("--value-rollout-role-weight", type=float, default=0.5)
    parser.add_argument("--value-rollout-prior-weight", type=float, default=0.1)
    parser.add_argument("--survival-lane-fraction", type=float, default=0.0)
    parser.add_argument("--survival-ttl", type=int, default=0)
    parser.add_argument("--survival-min-value", type=float, default=0.0)
    parser.add_argument("--survival-score-weight", type=float, default=0.0)
    parser.add_argument("--survival-max-per-root", type=int, default=0)
    parser.add_argument("--survival-ignore-signature-limit", action="store_true")
    parser.add_argument("--stop-on-solution", action="store_true")
    parser.add_argument("--include-stdout", action="store_true")
    args = parser.parse_args()
    if args.trace_prior_weight is None:
        args.trace_prior_weight = [0.25, 0.5, 1.0, 2.0]
    return args


if __name__ == "__main__":
    run(parse_args())
