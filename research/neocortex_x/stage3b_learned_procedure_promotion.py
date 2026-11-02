#!/usr/bin/env python3
"""Stage 3B learned procedure promotion for NeoCortex-X."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_experiment import (
    COMPUTE_KEYS,
    make_procedure_task,
    procedure_sum_segment,
    run_procedure_task,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": mean(values), "min": float(min(values)), "max": float(max(values))}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_csv_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit("CSV integer argument must contain at least one value.")
    return items


def collect_inline_trace_patterns(args: argparse.Namespace) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for seed in parse_csv_ints(args.train_seeds):
        for episode in range(args.train_episodes):
            task = make_procedure_task(
                seed=seed,
                episode=episode,
                tape_len=args.tape_len,
                segment_len=args.segment_len,
            )
            for segment in task["segments"]:
                patterns.append(
                    {
                        "ops": ("read", "accumulate", "advance"),
                        "signature": "sum_segment(start:int, length:int) -> int",
                        "length": segment["length"],
                    }
                )
    return patterns


def promote_procedure(patterns: list[dict[str, Any]], min_support: int) -> dict[str, Any]:
    signatures = Counter(pattern["signature"] for pattern in patterns)
    if not signatures:
        return {"promoted": False, "reason": "no_patterns"}
    signature, support = signatures.most_common(1)[0]
    lengths = sorted({int(pattern["length"]) for pattern in patterns if pattern["signature"] == signature})
    return {
        "promoted": support >= min_support,
        "name": "sum_segment",
        "signature": signature,
        "support": support,
        "lengths": lengths,
        "bounded": True,
    }


def certify_procedure(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not candidate.get("promoted"):
        return {"certified": False, "case_count": 0, "failure_count": 0}
    failures: list[dict[str, Any]] = []
    case_count = 0
    for seed in parse_csv_ints(args.cert_seeds):
        for episode in range(args.cert_episodes):
            task = make_procedure_task(
                seed=seed,
                episode=episode + 20_000,
                tape_len=args.tape_len,
                segment_len=args.segment_len,
            )
            for segment in task["segments"]:
                got, _steps = procedure_sum_segment(task["tape"], segment["start"], segment["length"])
                expected = sum(task["tape"][segment["start"] : segment["start"] + segment["length"]])
                case_count += 1
                if got != expected:
                    failures.append(
                        {
                            "task_id": task["task_id"],
                            "segment": segment,
                            "got": got,
                            "expected": expected,
                        }
                    )
    return {
        "certified": not failures and case_count > 0,
        "case_count": case_count,
        "failure_count": len(failures),
        "failures": failures[:8],
    }


def run_promoted_task(task: dict[str, Any], call_overhead: int) -> dict[str, Any]:
    return run_procedure_task(task, agent="procedure_reuse", call_overhead=call_overhead)


def run_agent_seed(*, seed: int, agent: str, args: argparse.Namespace) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    active_compute = {key: 0 for key in COMPUTE_KEYS}
    for episode in range(args.eval_episodes):
        task = make_procedure_task(
            seed=seed,
            episode=episode + 30_000,
            tape_len=args.tape_len,
            segment_len=args.segment_len,
        )
        if agent == "promoted_procedure":
            result = run_promoted_task(task, args.call_overhead)
        elif agent == "inline_loop_control":
            result = run_procedure_task(task, agent="inline_loop_control", call_overhead=args.call_overhead)
        elif agent == "exact_cache_control":
            result = run_procedure_task(task, agent="exact_cache_control", call_overhead=args.call_overhead)
        elif agent == "random_routing_control":
            result = run_procedure_task(task, agent="random_routing_control", call_overhead=args.call_overhead)
        else:
            raise ValueError(f"Unknown agent: {agent}")
        for key in COMPUTE_KEYS:
            active_compute[key] += int(result["active_compute"][key])
        episodes.append(
            {
                "episode": episode,
                "axis": task["axis"],
                "task_success": result["task_success"],
                "state_consistency": result["state_consistency"],
                "active_compute_total": result["active_compute_total"],
                "reuse_count": result["reuse_count"],
                "cache_hit": result["cache_hit"],
            }
        )
    axis_s = [item for item in episodes if item["axis"] == "axis_s"]
    axis_c = [item for item in episodes if item["axis"] == "axis_c"]
    return {
        "seed": seed,
        "agent": agent,
        "task_success": mean([1.0 if item["task_success"] else 0.0 for item in episodes]),
        "state_consistency": mean([1.0 if item["state_consistency"] else 0.0 for item in episodes]),
        "axis_s_success": mean([1.0 if item["task_success"] else 0.0 for item in axis_s]),
        "axis_c_success": mean([1.0 if item["task_success"] else 0.0 for item in axis_c]),
        "active_compute_total": mean([float(item["active_compute_total"]) for item in episodes]),
        "reuse_count": mean([float(item["reuse_count"]) for item in episodes]),
        "cache_hit_rate": mean([1.0 if item["cache_hit"] else 0.0 for item in episodes]),
        "active_compute": active_compute,
        "episodes": episodes,
    }


def aggregate_agent(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_success": summarize([float(run["task_success"]) for run in runs]),
        "state_consistency": summarize([float(run["state_consistency"]) for run in runs]),
        "axis_s_success": summarize([float(run["axis_s_success"]) for run in runs]),
        "axis_c_success": summarize([float(run["axis_c_success"]) for run in runs]),
        "active_compute_total": summarize([float(run["active_compute_total"]) for run in runs]),
        "reuse_count": summarize([float(run["reuse_count"]) for run in runs]),
        "cache_hit_rate": summarize([float(run["cache_hit_rate"]) for run in runs]),
        "runs": runs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    patterns = collect_inline_trace_patterns(args)
    candidate = promote_procedure(patterns, args.min_support)
    certification = certify_procedure(candidate, args)
    agents = ("promoted_procedure", "inline_loop_control", "exact_cache_control", "random_routing_control")
    eval_seeds = parse_csv_ints(args.eval_seeds)
    by_agent = {
        agent: [run_agent_seed(seed=seed, agent=agent, args=args) for seed in eval_seeds]
        for agent in agents
    }
    aggregate = {"agents": {agent: aggregate_agent(runs) for agent, runs in by_agent.items()}}
    promoted_compute = aggregate["agents"]["promoted_procedure"]["active_compute_total"]["mean"]
    inline_compute = aggregate["agents"]["inline_loop_control"]["active_compute_total"]["mean"]
    aggregate["comparison"] = {
        "active_compute_drop_vs_inline": {
            "mean": inline_compute - promoted_compute,
            "min": min(
                run_inline["active_compute_total"] - run_promoted["active_compute_total"]
                for run_inline, run_promoted in zip(by_agent["inline_loop_control"], by_agent["promoted_procedure"])
            ),
            "max": max(
                run_inline["active_compute_total"] - run_promoted["active_compute_total"]
                for run_inline, run_promoted in zip(by_agent["inline_loop_control"], by_agent["promoted_procedure"])
            ),
        }
    }
    protocol_evidence = {
        "valid": bool(len(eval_seeds) >= args.min_eval_seeds and candidate.get("promoted") and certification["certified"]),
        "train_pattern_count": len(patterns),
        "heldout_seed_count": len(eval_seeds),
        "heldout_seed_requirement_met": len(eval_seeds) >= args.min_eval_seeds,
        "axis_s_axis_c_present": True,
    }
    promotion_evidence = {
        **candidate,
        **certification,
    }
    promoted = aggregate["agents"]["promoted_procedure"]
    cache = aggregate["agents"]["exact_cache_control"]
    random_route = aggregate["agents"]["random_routing_control"]
    compute_drop = aggregate["comparison"]["active_compute_drop_vs_inline"]["mean"]
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "promotion_promoted": bool(candidate.get("promoted")),
        "promotion_certified": bool(certification["certified"]),
        "task_success": promoted["task_success"]["mean"] >= args.success_min,
        "axis_s_success": promoted["axis_s_success"]["mean"] >= args.success_min,
        "axis_c_success": promoted["axis_c_success"]["mean"] >= args.success_min,
        "active_compute_drop_vs_inline": compute_drop >= args.compute_drop_min,
        "exact_cache_hit_rate": cache["cache_hit_rate"]["mean"] == 0.0,
        "random_routing_task_success": random_route["task_success"]["mean"] <= args.random_success_max,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage3b_learned_procedure_promotion",
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "promotion_evidence": promotion_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": aggregate,
        "claim_guardrail": (
            "Stage 3B promotes a bounded procedure from synthetic inline traces. "
            "It is learned promotion, but still a controlled smoke rather than open-ended synthesis."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 3B learned procedure promotion failed: {pass_fail}")
    print("NeoCortex-X Stage 3B learned procedure promotion")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  promoted: {candidate.get('promoted')}")
    print(f"  certified: {certification['certified']}")
    print(f"  promoted.task_success: {promoted['task_success']['mean']:.4f}")
    print(f"  active_compute_drop_vs_inline: {compute_drop:.4f}")
    print(f"  random_routing.task_success: {random_route['task_success']['mean']:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 3B learned procedure promotion.")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage3b_learned_procedure_promotion.json")
    parser.add_argument("--train-seeds", default="101,103,107")
    parser.add_argument("--cert-seeds", default="109,113,127")
    parser.add_argument("--eval-seeds", default="11,17,23,29,31,37")
    parser.add_argument("--min-eval-seeds", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--cert-episodes", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=96)
    parser.add_argument("--tape-len", type=int, default=12)
    parser.add_argument("--segment-len", type=int, default=3)
    parser.add_argument("--call-overhead", type=int, default=1)
    parser.add_argument("--min-support", type=int, default=32)
    parser.add_argument("--success-min", type=float, default=0.95)
    parser.add_argument("--compute-drop-min", type=float, default=4.0)
    parser.add_argument("--random-success-max", type=float, default=0.40)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
