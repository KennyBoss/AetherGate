#!/usr/bin/env python3
"""Stage 2B learned tape-world policy for NeoCortex-X.

This runner learns a compact state-action table from Stage 2 oracle traces and
evaluates the induced policy on held-out tape-world tasks.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_experiment import (
    COMPUTE_KEYS,
    active_compute_increment,
    build_tape_delta,
    make_tape_task,
    state_digest,
    tape_semantic_state,
    tape_transition,
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


def state_features(state: dict[str, Any]) -> tuple[int, str, bool]:
    ptr = int(state["ptr"])
    input_len = int(state["input_len"])
    if ptr < input_len:
        ptr_relation = "inside"
    elif ptr == input_len:
        ptr_relation = "at_end"
    else:
        ptr_relation = "past_end"
    return int(state["pc"]), ptr_relation, bool(state["halted"])


def action_label(action: dict[str, Any]) -> str:
    return str(action["kind"])


def oracle_step(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    after, _delta, _gates, action, _reward = tape_transition(state, agent="codepy_loop")
    return after, action


def collect_oracle_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    train_seeds = parse_csv_ints(args.train_seeds)
    for seed in train_seeds:
        for episode in range(args.train_episodes):
            task = make_tape_task(
                seed=seed,
                episode=episode,
                tape_len=args.tape_len,
                input_len_min=args.input_len_min,
                input_len_max=args.input_len_max,
            )
            state = {
                "agent": "codepy_loop",
                "task_id": task["task_id"],
                "target": task["target"],
                "tape": list(task["tape"]),
                "input_len": int(task["input_len"]),
                "target_sum": int(task["target_sum"]),
                "ptr": 0,
                "pc": 0,
                "registers": {"r0": 0, "r1": 0},
                "predicate": False,
                "halted": False,
                "errors": [],
                "active_compute": {key: 0 for key in COMPUTE_KEYS},
            }
            for _step in range(args.max_steps):
                if state["halted"]:
                    break
                features = state_features(state)
                next_state, action = oracle_step(state)
                examples.append(
                    {
                        "features": features,
                        "action": action_label(action),
                        "raw_action": action,
                    }
                )
                state = next_state
    return examples


def train_table_policy(examples: list[dict[str, Any]]) -> dict[tuple[int, str, bool], str]:
    votes: dict[tuple[int, str, bool], Counter[str]] = defaultdict(Counter)
    for example in examples:
        votes[example["features"]][example["action"]] += 1
    return {features: counter.most_common(1)[0][0] for features, counter in votes.items()}


def action_from_label(label: str, state: dict[str, Any]) -> dict[str, Any]:
    if label == "tape_read":
        return {"kind": "tape_read", "ptr": state["ptr"], "value": state["tape"][state["ptr"]]}
    if label == "accumulate":
        return {"kind": "accumulate", "r1": state["registers"]["r1"], "r0_after": state["registers"]["r0"] + state["registers"]["r1"]}
    if label == "move_ptr":
        return {"kind": "move_ptr", "ptr_after": min(state["ptr"] + 1, len(state["tape"]))}
    if label == "set_predicate":
        return {"kind": "set_predicate", "expr": "ptr < input_len", "value": state["ptr"] < state["input_len"]}
    if label == "branch":
        return {"kind": "branch", "predicate": state["predicate"], "pc_after": 0 if state["predicate"] else 5}
    if label == "stop":
        return {"kind": "stop"}
    return {"kind": "unknown_stop", "label": label}


def apply_learned_action(state: dict[str, Any], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    before = copy.deepcopy(state)
    after = copy.deepcopy(state)
    action = action_from_label(label, state)
    kind = action["kind"]
    if kind == "tape_read":
        after["registers"]["r1"] = before["tape"][before["ptr"]]
        after["pc"] = 1
        after["active_compute"]["tape_reads"] += 1
        after["active_compute"]["register_writes"] += 1
    elif kind == "accumulate":
        after["registers"]["r0"] = before["registers"]["r0"] + before["registers"]["r1"]
        after["pc"] = 2
        after["active_compute"]["register_writes"] += 1
    elif kind == "move_ptr":
        after["ptr"] = min(before["ptr"] + 1, len(before["tape"]))
        after["pc"] = 3
        after["active_compute"]["pointer_moves"] += 1
    elif kind == "set_predicate":
        after["predicate"] = before["ptr"] < before["input_len"]
        after["pc"] = 4
        after["active_compute"]["register_writes"] += 1
    elif kind == "branch":
        after["pc"] = 0 if before["predicate"] else 5
        after["active_compute"]["branches"] += 1
    elif kind == "stop":
        after["halted"] = True
        after["active_compute"]["stops"] += 1
    else:
        after["halted"] = True
        after["errors"].append({"reason": "unknown_action", "label": label})
        after["active_compute"]["stops"] += 1
    after["active_compute"]["total_ticks"] += 1
    if kind != "stop":
        after["active_compute"]["env_actions"] += 1
    delta = build_tape_delta(before, after, action)
    return after, delta


def initial_state(task: dict[str, Any], agent: str) -> dict[str, Any]:
    return {
        "agent": agent,
        "task_id": task["task_id"],
        "target": task["target"],
        "tape": list(task["tape"]),
        "input_len": int(task["input_len"]),
        "target_sum": int(task["target_sum"]),
        "ptr": 0,
        "pc": 0,
        "registers": {"r0": 0, "r1": 0},
        "predicate": False,
        "halted": False,
        "errors": [],
        "active_compute": {key: 0 for key in COMPUTE_KEYS},
    }


def run_policy_episode(
    task: dict[str, Any],
    *,
    policy: dict[tuple[int, str, bool], str],
    agent: str,
    max_steps: int,
    majority_action: str,
) -> dict[str, Any]:
    oracle_state = initial_state(task, "codepy_loop")
    state = initial_state(task, agent)
    action_correct = 0
    action_count = 0
    first_divergence: int | None = None
    for step in range(max_steps):
        if state["halted"]:
            break
        oracle_before_digest = state_digest(oracle_state)
        oracle_next, oracle_action = oracle_step(oracle_state)
        oracle_label = action_label(oracle_action)
        if agent == "learned_trace_policy":
            label = policy.get(state_features(state), majority_action)
            state, _delta = apply_learned_action(state, label)
        elif agent == "dense_majority_control":
            label = majority_action
            state, _delta = apply_learned_action(state, label)
        elif agent == "single_cell_baseline":
            state, _delta, _gates, action, _reward = tape_transition(state, agent="single_cell_baseline")
            label = action_label(action)
        else:
            raise ValueError(f"Unknown agent: {agent}")
        action_count += 1
        if label == oracle_label:
            action_correct += 1
        if first_divergence is None and state_digest(state) != state_digest(oracle_next):
            first_divergence = step + 1
        oracle_state = oracle_next
        _ = oracle_before_digest
    state_consistent = tape_semantic_state(state) == tape_semantic_state(oracle_state)
    task_success = bool(state["halted"] and state["registers"]["r0"] == state["target_sum"] and not state["errors"])
    return {
        "task_id": task["task_id"],
        "input_len": task["input_len"],
        "action_accuracy": action_correct / action_count if action_count else 0.0,
        "state_consistency": state_consistent,
        "task_success": task_success,
        "trajectory_horizon": first_divergence if first_divergence is not None else action_count,
        "tick_count": action_count,
    }


def run_agent_eval(
    *,
    seeds: list[int],
    agent: str,
    args: argparse.Namespace,
    policy: dict[tuple[int, str, bool], str],
    majority_action: str,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        for episode in range(args.eval_episodes):
            task = make_tape_task(
                seed=seed,
                episode=episode + 10_000,
                tape_len=args.tape_len,
                input_len_min=args.input_len_min,
                input_len_max=args.input_len_max,
            )
            episodes.append(
                run_policy_episode(
                    task,
                    policy=policy,
                    agent=agent,
                    max_steps=args.max_steps,
                    majority_action=majority_action,
                )
            )
    return {
        "agent": agent,
        "episode_count": len(episodes),
        "action_accuracy": mean([float(item["action_accuracy"]) for item in episodes]),
        "state_consistency": mean([1.0 if item["state_consistency"] else 0.0 for item in episodes]),
        "task_success": mean([1.0 if item["task_success"] else 0.0 for item in episodes]),
        "trajectory_horizon": mean([float(item["trajectory_horizon"]) for item in episodes]),
        "episodes": episodes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    examples = collect_oracle_examples(args)
    policy = train_table_policy(examples)
    action_counts = Counter(example["action"] for example in examples)
    majority_action = action_counts.most_common(1)[0][0]
    eval_seeds = parse_csv_ints(args.eval_seeds)
    agents = ("learned_trace_policy", "single_cell_baseline", "dense_majority_control")
    runs = {
        agent: run_agent_eval(
            seeds=eval_seeds,
            agent=agent,
            args=args,
            policy=policy,
            majority_action=majority_action,
        )
        for agent in agents
    }
    comparison = {
        "learned_minus_baseline_state_consistency": (
            runs["learned_trace_policy"]["state_consistency"]
            - runs["single_cell_baseline"]["state_consistency"]
        ),
        "learned_minus_dense_state_consistency": (
            runs["learned_trace_policy"]["state_consistency"]
            - runs["dense_majority_control"]["state_consistency"]
        ),
    }
    protocol_evidence = {
        "valid": bool(len(eval_seeds) >= args.min_eval_seeds and len(examples) > 0 and len(policy) > 0),
        "train_example_count": len(examples),
        "learned_rule_count": len(policy),
        "heldout_seed_count": len(eval_seeds),
        "heldout_seed_requirement_met": len(eval_seeds) >= args.min_eval_seeds,
        "train_seeds": parse_csv_ints(args.train_seeds),
        "eval_seeds": eval_seeds,
        "majority_action": majority_action,
    }
    thresholds = {
        "action_accuracy_min": args.action_accuracy_min,
        "state_consistency_min": args.state_consistency_min,
        "task_success_min": args.task_success_min,
        "state_margin_min": args.state_margin_min,
    }
    learned = runs["learned_trace_policy"]
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "action_accuracy": learned["action_accuracy"] >= thresholds["action_accuracy_min"],
        "state_consistency": learned["state_consistency"] >= thresholds["state_consistency_min"],
        "task_success": learned["task_success"] >= thresholds["task_success_min"],
        "learned_minus_baseline_state_consistency": (
            comparison["learned_minus_baseline_state_consistency"] >= thresholds["state_margin_min"]
        ),
        "learned_minus_dense_state_consistency": (
            comparison["learned_minus_dense_state_consistency"] >= thresholds["state_margin_min"]
        ),
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage2b_learned_tape_policy",
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "thresholds": thresholds,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "policy": {str(key): value for key, value in policy.items()},
        "aggregate": {
            "agents": {agent: {k: v for k, v in run.items() if k != "episodes"} for agent, run in runs.items()},
            "comparison": comparison,
        },
        "runs": runs,
        "claim_guardrail": (
            "Stage 2B is imitation learning from oracle traces in the bounded "
            "tape world. It is not yet reinforcement learning or broad program synthesis."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 2B learned tape policy failed: {pass_fail}")
    print("NeoCortex-X Stage 2B learned tape policy")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  train_examples: {len(examples)}")
    print(f"  learned_rules: {len(policy)}")
    print(f"  learned.action_accuracy: {learned['action_accuracy']:.4f}")
    print(f"  learned.state_consistency: {learned['state_consistency']:.4f}")
    print(f"  learned.task_success: {learned['task_success']:.4f}")
    print(f"  learned_minus_baseline_state_consistency: {comparison['learned_minus_baseline_state_consistency']:.4f}")
    print(f"  learned_minus_dense_state_consistency: {comparison['learned_minus_dense_state_consistency']:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 2B learned tape policy.")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage2b_learned_tape_policy.json")
    parser.add_argument("--train-seeds", default="101,103,107")
    parser.add_argument("--eval-seeds", default="11,17,23,29,31,37")
    parser.add_argument("--min-eval-seeds", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--tape-len", type=int, default=8)
    parser.add_argument("--input-len-min", type=int, default=4)
    parser.add_argument("--input-len-max", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--action-accuracy-min", type=float, default=0.95)
    parser.add_argument("--state-consistency-min", type=float, default=0.95)
    parser.add_argument("--task-success-min", type=float, default=0.95)
    parser.add_argument("--state-margin-min", type=float, default=0.50)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
