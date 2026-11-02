#!/usr/bin/env python3
"""Stage 20 contextual policy-guided replay for NeoCortex-X."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ACTION = {
    "context_match": "answer_from_context_memory",
    "same_subject_different_context": "answer_with_context_scope",
    "request_context": "ask_context_question",
    "flag_conflict": "request_more_evidence",
}

ACTIONS = tuple(sorted(set(EXPECTED_ACTION.values())))


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


def validate_contextual_action_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage17_contextual_action_gate":
        errors.append("invalid contextual action kind")
    if artifact.get("pass") is not True:
        errors.append("stage17 contextual action artifact did not pass")
    runs = artifact.get("runs", {})
    decisions = runs.get("contextual_action_gate", {}).get("decisions")
    if not isinstance(decisions, list) or not decisions:
        errors.append("contextual_action_gate decisions must be a non-empty list")
        decisions = []
    required = {"observation_id", "context_decision", "agent_action", "expected_action"}
    for index, decision in enumerate(decisions):
        missing = required.difference(decision)
        if missing:
            errors.append(f"decision {index} missing {sorted(missing)}")
        if decision.get("context_decision") not in EXPECTED_ACTION:
            errors.append(f"decision {index} has unknown context_decision")
    return {"valid": not errors, "errors": errors, "decision_count": len(decisions)}


def validate_policy_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage19_contextual_policy_update":
        errors.append("invalid policy kind")
    if artifact.get("pass") is not True:
        errors.append("stage19 contextual policy artifact did not pass")
    policy = artifact.get("policy", {})
    probabilities = policy.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        errors.append("policy probabilities are missing")
        probabilities = {}
    probability_sums: dict[str, float] = {}
    for decision, expected_action in EXPECTED_ACTION.items():
        bucket = probabilities.get(decision)
        if not isinstance(bucket, dict):
            errors.append(f"policy missing bucket {decision}")
            continue
        probability_sums[decision] = sum(float(value) for value in bucket.values())
        if abs(probability_sums[decision] - 1.0) > 1e-9:
            errors.append(f"policy bucket {decision} does not sum to 1.0")
        for action in ACTIONS:
            if action not in bucket:
                errors.append(f"policy bucket {decision} missing action {action}")
        selected = top_action(bucket)
        if selected != expected_action:
            errors.append(f"policy bucket {decision} top action {selected} != {expected_action}")
    return {
        "valid": not errors,
        "errors": errors,
        "policy_id": policy.get("policy_id"),
        "probability_sums": probability_sums,
    }


def top_action(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=lambda action: (float(probabilities[action]), action))


def global_prior(probabilities: dict[str, dict[str, float]]) -> dict[str, float]:
    totals = {action: 0.0 for action in ACTIONS}
    for bucket in probabilities.values():
        for action in ACTIONS:
            totals[action] += float(bucket.get(action, 0.0))
    total = sum(totals.values())
    if total <= 0.0:
        return {action: 1.0 / len(ACTIONS) for action in ACTIONS}
    return {action: value / total for action, value in totals.items()}


def expected_action(decision: dict[str, Any]) -> str:
    return EXPECTED_ACTION[decision["context_decision"]]


def choose_action(
    decision: dict[str, Any],
    *,
    agent: str,
    probabilities: dict[str, dict[str, float]],
    global_probabilities: dict[str, float],
) -> str:
    if agent == "contextual_policy_guided":
        return top_action(probabilities[decision["context_decision"]])
    if agent == "global_prior_control":
        return top_action(global_probabilities)
    if agent == "always_evidence_control":
        return "request_more_evidence"
    raise ValueError(f"Unknown Stage 20 agent: {agent}")


def run_agent(
    decisions: list[dict[str, Any]],
    *,
    agent: str,
    probabilities: dict[str, dict[str, float]],
    global_probabilities: dict[str, float],
) -> dict[str, Any]:
    replay_decisions: list[dict[str, Any]] = []
    for decision in decisions:
        expected = expected_action(decision)
        selected = choose_action(
            decision,
            agent=agent,
            probabilities=probabilities,
            global_probabilities=global_probabilities,
        )
        probability_bucket = (
            probabilities[decision["context_decision"]]
            if agent == "contextual_policy_guided"
            else global_probabilities
        )
        replay_decisions.append(
            {
                "observation_id": decision["observation_id"],
                "context_decision": decision["context_decision"],
                "selected_action": selected,
                "expected_action": expected,
                "action_correct": selected == expected,
                "selected_probability": float(probability_bucket.get(selected, 0.0)),
                "false_evidence_request": (
                    selected == "request_more_evidence"
                    and decision["context_decision"] != "flag_conflict"
                ),
            }
        )
    return {
        "agent": agent,
        "decision_count": len(replay_decisions),
        "action_accuracy": mean([1.0 if item["action_correct"] else 0.0 for item in replay_decisions]),
        "mean_selected_probability": mean(
            [float(item["selected_probability"]) for item in replay_decisions]
        ),
        "false_evidence_request_rate": mean(
            [1.0 if item["false_evidence_request"] else 0.0 for item in replay_decisions]
        ),
        "decisions": replay_decisions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    action_artifact = load_json(Path(args.contextual_action_json))
    policy_artifact = load_json(Path(args.policy_json))
    action_validation = validate_contextual_action_artifact(action_artifact)
    policy_validation = validate_policy_artifact(policy_artifact)
    decisions = action_artifact.get("runs", {}).get("contextual_action_gate", {}).get("decisions", [])
    probabilities = policy_artifact.get("policy", {}).get("probabilities", {})
    global_probabilities = global_prior(probabilities)
    agents = (
        "contextual_policy_guided",
        "global_prior_control",
        "always_evidence_control",
    )
    runs = {
        agent: run_agent(
            decisions,
            agent=agent,
            probabilities=probabilities,
            global_probabilities=global_probabilities,
        )
        for agent in agents
    }
    replay_runs = {
        agent: run_agent(
            decisions,
            agent=agent,
            probabilities=probabilities,
            global_probabilities=global_probabilities,
        )
        for agent in agents
    }
    deterministic_replay = runs == replay_runs
    guided = runs["contextual_policy_guided"]
    global_control = runs["global_prior_control"]
    always_evidence = runs["always_evidence_control"]
    selected_actions = sorted({item["selected_action"] for item in guided["decisions"]})
    false_evidence_for_context_alias = any(
        item["selected_action"] == "request_more_evidence"
        for item in guided["decisions"]
        if item["context_decision"] == "same_subject_different_context"
    )
    comparison = {
        "guided_minus_global_prior_accuracy": (
            guided["action_accuracy"] - global_control["action_accuracy"]
        ),
        "guided_minus_always_evidence_accuracy": (
            guided["action_accuracy"] - always_evidence["action_accuracy"]
        ),
    }
    protocol_evidence = {
        "valid": bool(action_validation["valid"] and policy_validation["valid"] and deterministic_replay),
        "contextual_action_json": args.contextual_action_json,
        "policy_json": args.policy_json,
        "decision_count": action_validation["decision_count"],
        "policy_id": policy_validation["policy_id"],
        "deterministic_replay": deterministic_replay,
        "selected_actions": selected_actions,
    }
    required_actions = set(EXPECTED_ACTION.values())
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "contextual_action_artifact_valid": action_validation["valid"],
        "policy_artifact_valid": policy_validation["valid"],
        "guided_action_accuracy": guided["action_accuracy"] >= args.guided_accuracy_min,
        "selected_actions_cover_gate": required_actions.issubset(set(selected_actions)),
        "guided_minus_global_prior_accuracy": (
            comparison["guided_minus_global_prior_accuracy"] >= args.global_prior_margin_min
        ),
        "no_false_evidence_request_for_contextual_alias": not false_evidence_for_context_alias,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage20_contextual_policy_replay",
        "config": vars(args),
        "action_validation": action_validation,
        "policy_validation": policy_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": {
            "agents": {agent: {k: v for k, v in run.items() if k != "decisions"} for agent, run in runs.items()},
            "comparison": comparison,
            "global_probabilities": global_probabilities,
        },
        "runs": runs,
        "claim_guardrail": (
            "Stage 20 replays a contextual policy snapshot under deterministic "
            "context decisions. It is not a broad autonomy claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 20 contextual policy replay failed: {pass_fail}")
    print("NeoCortex-X Stage 20 contextual policy replay")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  policy_id: {policy_validation['policy_id']}")
    print(f"  guided.action_accuracy: {guided['action_accuracy']:.4f}")
    print(
        "  guided_minus_global_prior_accuracy: "
        f"{comparison['guided_minus_global_prior_accuracy']:.4f}"
    )
    print(f"  guided.false_evidence_request_rate: {guided['false_evidence_request_rate']:.4f}")
    print(f"  selected_actions: {','.join(selected_actions)}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 20 contextual policy replay.")
    parser.add_argument("--contextual-action-json", default="artifacts/neocortex_x/stage17_contextual_action_gate.json")
    parser.add_argument("--policy-json", default="artifacts/neocortex_x/stage19_contextual_policy_update.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage20_contextual_policy_replay.json")
    parser.add_argument("--guided-accuracy-min", type=float, default=0.95)
    parser.add_argument("--global-prior-margin-min", type=float, default=0.25)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
