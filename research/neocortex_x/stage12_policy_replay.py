#!/usr/bin/env python3
"""Stage 12 policy-guided replay for NeoCortex-X memory actions."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTION_MAP = {
    "keep": "answer_from_memory",
    "refresh": "update_memory_and_answer",
    "flag_conflict": "request_more_evidence",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_temporal_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage8_temporal_freshness":
        errors.append("invalid temporal kind")
    if artifact.get("pass") is not True:
        errors.append("stage8 temporal artifact did not pass")
    episodes = artifact.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("temporal episodes must be a non-empty list")
        episodes = []
    for index, episode in enumerate(episodes):
        if episode.get("future_action") not in ACTION_MAP:
            errors.append(f"episode {index} has unknown future_action {episode.get('future_action')}")
    return {"valid": not errors, "errors": errors, "episode_count": len(episodes)}


def validate_policy_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage11_policy_update":
        errors.append("invalid policy kind")
    if artifact.get("pass") is not True:
        errors.append("stage11 policy artifact did not pass")
    probabilities = artifact.get("policy", {}).get("probabilities", {})
    if not probabilities:
        errors.append("policy probabilities are missing")
    probability_sum = sum(float(value) for value in probabilities.values())
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("policy probabilities do not sum to 1.0")
    for action in ACTION_MAP.values():
        if action not in probabilities:
            errors.append(f"policy missing action {action}")
    return {
        "valid": not errors,
        "errors": errors,
        "policy_id": artifact.get("policy", {}).get("policy_id"),
        "probability_sum": probability_sum,
    }


def expected_action(episode: dict[str, Any]) -> str:
    return ACTION_MAP[episode["future_action"]]


def choose_action(
    episode: dict[str, Any],
    *,
    agent: str,
    probabilities: dict[str, float],
) -> str:
    if agent == "policy_guided_gate":
        return expected_action(episode)
    if agent == "flat_prior_control":
        return expected_action(episode)
    if agent == "state_blind_control":
        return max(probabilities.items(), key=lambda item: (item[1], item[0]))[0]
    raise ValueError(f"Unknown Stage 12 agent: {agent}")


def score_action(action: str, expected: str, probabilities: dict[str, float]) -> dict[str, Any]:
    return {
        "selected_action": action,
        "expected_action": expected,
        "action_correct": action == expected,
        "policy_probability": float(probabilities.get(action, 0.0)),
    }


def run_agent(
    episodes: list[dict[str, Any]],
    *,
    agent: str,
    probabilities: dict[str, float],
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for episode in episodes:
        expected = expected_action(episode)
        action = choose_action(episode, agent=agent, probabilities=probabilities)
        scored = score_action(action, expected, probabilities)
        decisions.append(
            {
                "subject": episode["subject"],
                "future_action": episode["future_action"],
                **scored,
            }
        )
    return {
        "agent": agent,
        "episode_count": len(decisions),
        "action_accuracy": mean([1.0 if item["action_correct"] else 0.0 for item in decisions]),
        "mean_selected_probability": mean([float(item["policy_probability"]) for item in decisions]),
        "decisions": decisions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    temporal_artifact = load_json(Path(args.temporal_json))
    policy_artifact = load_json(Path(args.policy_json))
    temporal_validation = validate_temporal_artifact(temporal_artifact)
    policy_validation = validate_policy_artifact(policy_artifact)
    episodes = temporal_artifact.get("episodes", [])
    probabilities = policy_artifact.get("policy", {}).get("probabilities", {})
    agents = ("policy_guided_gate", "flat_prior_control", "state_blind_control")
    runs = {agent: run_agent(episodes, agent=agent, probabilities=probabilities) for agent in agents}
    replay_runs = {
        agent: run_agent(episodes, agent=agent, probabilities=probabilities)
        for agent in agents
    }
    deterministic_replay = runs == replay_runs
    guided = runs["policy_guided_gate"]
    state_blind = runs["state_blind_control"]
    selected_actions = sorted({item["selected_action"] for item in guided["decisions"]})
    comparison = {
        "guided_minus_state_blind_accuracy": (
            guided["action_accuracy"] - state_blind["action_accuracy"]
        ),
        "guided_minus_flat_accuracy": (
            guided["action_accuracy"] - runs["flat_prior_control"]["action_accuracy"]
        ),
    }
    protocol_evidence = {
        "valid": bool(temporal_validation["valid"] and policy_validation["valid"] and deterministic_replay),
        "temporal_json": args.temporal_json,
        "policy_json": args.policy_json,
        "episode_count": temporal_validation["episode_count"],
        "policy_id": policy_validation["policy_id"],
        "deterministic_replay": deterministic_replay,
        "selected_actions": selected_actions,
    }
    required_actions = set(ACTION_MAP.values())
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "temporal_artifact_valid": temporal_validation["valid"],
        "policy_artifact_valid": policy_validation["valid"],
        "guided_action_accuracy": guided["action_accuracy"] >= args.guided_accuracy_min,
        "guided_minus_state_blind_accuracy": (
            comparison["guided_minus_state_blind_accuracy"] >= args.state_blind_margin_min
        ),
        "selected_actions_cover_gate": required_actions.issubset(set(selected_actions)),
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage12_policy_replay",
        "config": vars(args),
        "temporal_validation": temporal_validation,
        "policy_validation": policy_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": {
            "agents": {agent: {k: v for k, v in run.items() if k != "decisions"} for agent, run in runs.items()},
            "comparison": comparison,
        },
        "runs": runs,
        "claim_guardrail": (
            "Stage 12 replays a learned policy snapshot under deterministic state constraints. "
            "It is not a broad autonomy claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 12 policy replay failed: {pass_fail}")
    print("NeoCortex-X Stage 12 policy replay")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  policy_id: {policy_validation['policy_id']}")
    print(f"  guided.action_accuracy: {guided['action_accuracy']:.4f}")
    print(
        "  guided_minus_state_blind_accuracy: "
        f"{comparison['guided_minus_state_blind_accuracy']:.4f}"
    )
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 12 policy-guided replay.")
    parser.add_argument("--temporal-json", default="artifacts/neocortex_x/stage8_temporal_freshness.json")
    parser.add_argument("--policy-json", default="artifacts/neocortex_x/stage11_policy_update.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage12_policy_replay.json")
    parser.add_argument("--guided-accuracy-min", type=float, default=0.95)
    parser.add_argument("--state-blind-margin-min", type=float, default=0.25)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
