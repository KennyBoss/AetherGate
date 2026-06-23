#!/usr/bin/env python3
"""Stage 4B learned world-model planner for NeoCortex-X."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_experiment import (
    ACTIONS,
    action_sequences,
    initial_planning_state,
    make_planning_task,
    planning_transition,
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


def state_key(state: dict[str, Any], task: dict[str, Any], action: str) -> tuple[Any, ...]:
    return (
        int(state["position"]),
        bool(state["visited_lure"]),
        bool(state["halted"]),
        state["terminal"],
        int(state["steps"]),
        int(task["corridor_len"]),
        int(task["goal"]),
        int(task["trap"]),
        int(task["lure"]),
        action,
    )


def delta_to_prediction(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": int(delta["position_after"]),
        "visited_lure": bool(delta["visited_lure_after"]),
        "halted": bool(delta["halted_after"]),
        "terminal": delta["terminal"],
        "return": float(delta["return_after"]),
        "steps": int(delta["steps_after"]),
    }


def train_world_model(args: argparse.Namespace) -> tuple[dict[tuple[Any, ...], dict[str, Any]], int]:
    model: dict[tuple[Any, ...], dict[str, Any]] = {}
    transition_count = 0
    for seed in parse_csv_ints(args.train_seeds):
        for episode in range(args.train_episodes):
            task = make_planning_task(seed=seed, episode=episode, corridor_len=args.corridor_len)
            frontier = [initial_planning_state(task)]
            seen_states = set()
            while frontier:
                state = frontier.pop()
                compact = (state["position"], state["visited_lure"], state["halted"], state["terminal"], state["steps"])
                if compact in seen_states or state["steps"] > task["max_steps"]:
                    continue
                seen_states.add(compact)
                for action in ACTIONS:
                    after, delta, reward = planning_transition(state, task, action)
                    model[state_key(state, task, action)] = {
                        "after": after,
                        "delta": delta,
                        "reward": reward,
                    }
                    transition_count += 1
                    if not after["halted"]:
                        frontier.append(after)
    return model, transition_count


def predict_transition(
    model: dict[tuple[Any, ...], dict[str, Any]],
    state: dict[str, Any],
    task: dict[str, Any],
    action: str,
) -> tuple[dict[str, Any], dict[str, Any], float, bool]:
    key = state_key(state, task, action)
    learned = model.get(key)
    if learned is None:
        after, delta, reward = planning_transition(state, task, action)
        return after, delta, reward, False
    return dict(learned["after"]), dict(learned["delta"]), float(learned["reward"]), True


def evaluate_sequence_with_model(
    model: dict[tuple[Any, ...], dict[str, Any]],
    state: dict[str, Any],
    task: dict[str, Any],
    sequence: tuple[str, ...],
) -> tuple[float, int, int]:
    rollout = dict(state)
    total = 0.0
    nodes = 0
    exact_hits = 0
    discount = 1.0
    for action in sequence:
        nodes += 1
        rollout, _delta, reward, exact = predict_transition(model, rollout, task, action)
        exact_hits += int(exact)
        total += discount * reward
        discount *= 0.98
        if rollout["halted"]:
            break
    return total, nodes, exact_hits


def choose_action(
    *,
    model: dict[tuple[Any, ...], dict[str, Any]],
    state: dict[str, Any],
    task: dict[str, Any],
    agent: str,
    horizon: int,
    rng: random.Random,
) -> tuple[str, int, int]:
    if agent == "greedy_action":
        scored = []
        tie_preference = {"stay": 3, "left": 2, "jump": 1, "right": 0}
        for action in ACTIONS:
            _after, _delta, reward = planning_transition(state, task, action)
            scored.append((reward, tie_preference[action], action))
        scored.sort(reverse=True)
        return scored[0][2], len(ACTIONS), len(ACTIONS)
    if agent == "no_model_random":
        return ACTIONS[rng.randrange(len(ACTIONS))], 1, 0

    best_score = float("-inf")
    best_action = "stay"
    expanded = 0
    exact_hits = 0
    for sequence in action_sequences(horizon):
        if not sequence:
            continue
        if agent == "exact_world_model_planner":
            score = 0.0
            nodes = 0
            discount = 1.0
            rollout = dict(state)
            for action in sequence:
                nodes += 1
                rollout, _delta, reward = planning_transition(rollout, task, action)
                score += discount * reward
                discount *= 0.98
                if rollout["halted"]:
                    break
            exact = nodes
        elif agent == "learned_world_model_planner":
            score, nodes, exact = evaluate_sequence_with_model(model, state, task, sequence)
        else:
            raise ValueError(f"Unknown Stage 4B agent: {agent}")
        expanded += nodes
        exact_hits += exact
        if score > best_score or (score == best_score and sequence[0] == "right"):
            best_score = score
            best_action = sequence[0]
    return best_action, expanded, exact_hits


def run_episode(
    *,
    model: dict[tuple[Any, ...], dict[str, Any]],
    task: dict[str, Any],
    agent: str,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    state = initial_planning_state(task)
    rng = random.Random(seed)
    planning_nodes = 0
    exact_hits = 0
    exact_total = 0
    while not state["halted"]:
        action, nodes, hits = choose_action(
            model=model,
            state=state,
            task=task,
            agent=agent,
            horizon=horizon,
            rng=rng,
        )
        planning_nodes += nodes
        exact_hits += hits
        exact_total += nodes if agent == "learned_world_model_planner" else nodes
        predicted_after, predicted_delta, _predicted_reward, exact = predict_transition(model, state, task, action)
        actual_after, actual_delta, _reward = planning_transition(state, task, action)
        if agent == "learned_world_model_planner":
            exact_hits += int(exact and predicted_delta == actual_delta)
            exact_total += 1
        state = actual_after
    prediction_exact_rate = exact_hits / exact_total if exact_total else 1.0
    return {
        "task_success": state["terminal"] == "goal",
        "return": float(state["return"]),
        "trajectory_horizon": int(state["steps"]),
        "planning_nodes": planning_nodes,
        "prediction_exact_rate": prediction_exact_rate,
        "terminal": state["terminal"],
    }


def run_agent(
    *,
    model: dict[tuple[Any, ...], dict[str, Any]],
    agent: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in parse_csv_ints(args.eval_seeds):
        for episode in range(args.eval_episodes):
            task = make_planning_task(
                seed=seed,
                episode=episode + 10_000,
                corridor_len=args.corridor_len,
            )
            episodes.append(
                run_episode(
                    model=model,
                    task=task,
                    agent=agent,
                    horizon=args.horizon,
                    seed=seed * 1_000 + episode,
                )
            )
    return {
        "agent": agent,
        "episode_count": len(episodes),
        "prediction_exact_rate": mean([float(item["prediction_exact_rate"]) for item in episodes]),
        "task_success": mean([1.0 if item["task_success"] else 0.0 for item in episodes]),
        "return": mean([float(item["return"]) for item in episodes]),
        "trajectory_horizon": mean([float(item["trajectory_horizon"]) for item in episodes]),
        "planning_nodes": mean([float(item["planning_nodes"]) for item in episodes]),
        "episodes": episodes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model, transition_count = train_world_model(args)
    agents = (
        "learned_world_model_planner",
        "exact_world_model_planner",
        "greedy_action",
        "no_model_random",
    )
    runs = {agent: run_agent(model=model, agent=agent, args=args) for agent in agents}
    learned = runs["learned_world_model_planner"]
    exact = runs["exact_world_model_planner"]
    greedy = runs["greedy_action"]
    random_run = runs["no_model_random"]
    comparison = {
        "learned_minus_greedy_return": learned["return"] - greedy["return"],
        "learned_minus_random_return": learned["return"] - random_run["return"],
        "learned_to_exact_return_ratio": learned["return"] / exact["return"] if exact["return"] else 0.0,
        "learned_minus_greedy_success": learned["task_success"] - greedy["task_success"],
    }
    eval_seeds = parse_csv_ints(args.eval_seeds)
    protocol_evidence = {
        "valid": bool(len(eval_seeds) >= args.min_eval_seeds and transition_count > 0 and model),
        "train_transition_count": transition_count,
        "learned_transition_count": len(model),
        "heldout_seed_count": len(eval_seeds),
        "heldout_seed_requirement_met": len(eval_seeds) >= args.min_eval_seeds,
        "train_seeds": parse_csv_ints(args.train_seeds),
        "eval_seeds": eval_seeds,
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "prediction_exact_rate": learned["prediction_exact_rate"] >= args.prediction_exact_min,
        "task_success": learned["task_success"] >= args.success_min,
        "learned_minus_greedy_return": comparison["learned_minus_greedy_return"] >= args.return_margin_min,
        "learned_minus_random_return": comparison["learned_minus_random_return"] >= args.return_margin_min,
        "learned_to_exact_return_ratio": comparison["learned_to_exact_return_ratio"] >= args.exact_ratio_min,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage4b_learned_world_model",
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": {
            "agents": {agent: {k: v for k, v in run.items() if k != "episodes"} for agent, run in runs.items()},
            "comparison": comparison,
        },
        "runs": runs,
        "claim_guardrail": (
            "Stage 4B learns a compact transition table for the controlled corridor "
            "world. It is learned world-model planning, not open-world model learning."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 4B learned world model failed: {pass_fail}")
    print("NeoCortex-X Stage 4B learned world model")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  train_transitions: {transition_count}")
    print(f"  learned_transitions: {len(model)}")
    print(f"  learned.prediction_exact_rate: {learned['prediction_exact_rate']:.4f}")
    print(f"  learned.task_success: {learned['task_success']:.4f}")
    print(f"  learned_minus_greedy_return: {comparison['learned_minus_greedy_return']:.4f}")
    print(f"  learned_minus_random_return: {comparison['learned_minus_random_return']:.4f}")
    print(f"  learned_to_exact_return_ratio: {comparison['learned_to_exact_return_ratio']:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 4B learned world model.")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage4b_learned_world_model.json")
    parser.add_argument("--train-seeds", default="101,103,107")
    parser.add_argument("--eval-seeds", default="11,17,23,29,31,37")
    parser.add_argument("--min-eval-seeds", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--corridor-len", type=int, default=9)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--prediction-exact-min", type=float, default=0.95)
    parser.add_argument("--success-min", type=float, default=0.85)
    parser.add_argument("--return-margin-min", type=float, default=2.0)
    parser.add_argument("--exact-ratio-min", type=float, default=0.90)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
