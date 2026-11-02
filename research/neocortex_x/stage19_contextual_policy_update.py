#!/usr/bin/env python3
"""Stage 19 contextual policy update from NeoCortex-X contextual feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DECISIONS = (
    "context_match",
    "same_subject_different_context",
    "request_context",
    "flag_conflict",
)

ACTIONS = (
    "answer_from_context_memory",
    "answer_with_context_scope",
    "ask_context_question",
    "request_more_evidence",
)

EXPECTED_TOP_ACTION = {
    "context_match": "answer_from_context_memory",
    "same_subject_different_context": "answer_with_context_scope",
    "request_context": "ask_context_question",
    "flag_conflict": "request_more_evidence",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def stable_hash(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_feedback(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_feedback_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage18_contextual_feedback":
        errors.append("invalid kind")
    if artifact.get("pass") is not True:
        errors.append("stage18 contextual feedback artifact did not pass")
    events = artifact.get("feedback_events")
    if not isinstance(events, list) or not events:
        errors.append("feedback_events must be a non-empty list")
        events = []
    required = {
        "feedback_id",
        "context_decision",
        "agent_action",
        "reward",
        "reinforce_action",
        "replay_hash",
    }
    for index, event in enumerate(events):
        missing = required.difference(event)
        if missing:
            errors.append(f"feedback event {index} missing {sorted(missing)}")
        if event.get("context_decision") not in DECISIONS:
            errors.append(f"feedback event {index} has unknown decision {event.get('context_decision')}")
        if event.get("agent_action") not in ACTIONS:
            errors.append(f"feedback event {index} has unknown action {event.get('agent_action')}")
    return {"valid": not errors, "errors": errors, "feedback_event_count": len(events)}


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        return {action: 1.0 / len(weights) for action in weights}
    return {action: max(0.0, value) / total for action, value in weights.items()}


def top_action(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=lambda action: (probabilities[action], action))


def init_weights(base_weight: float) -> dict[str, dict[str, float]]:
    return {
        decision: {action: base_weight for action in ACTIONS}
        for decision in DECISIONS
    }


def update_policy(
    events: list[dict[str, Any]],
    *,
    base_weight: float,
    learning_rate: float,
) -> dict[str, Any]:
    before = init_weights(base_weight)
    after = {decision: dict(weights) for decision, weights in before.items()}
    updates: list[dict[str, Any]] = []
    for event in events:
        decision = event["context_decision"]
        action = event["agent_action"]
        reward = float(event["reward"])
        delta = learning_rate * reward
        after[decision][action] = max(0.0, after[decision][action] + delta)
        updates.append(
            {
                "feedback_id": event["feedback_id"],
                "context_decision": decision,
                "action": action,
                "reward": reward,
                "delta": delta,
                "weight_after": after[decision][action],
            }
        )
    probabilities = {
        decision: normalize(weights)
        for decision, weights in after.items()
    }
    weight_gains = {
        decision: {
            action: after[decision][action] - before[decision][action]
            for action in ACTIONS
        }
        for decision in DECISIONS
    }
    top_actions = {
        decision: top_action(probabilities[decision])
        for decision in DECISIONS
    }
    snapshot_basis = {
        "before": before,
        "after": after,
        "probabilities": probabilities,
        "top_actions": top_actions,
        "updates": updates,
    }
    return {
        "policy_id": f"ctxpolicy_{stable_hash(snapshot_basis)}",
        "before_weights": before,
        "after_weights": after,
        "weight_gains": weight_gains,
        "probabilities": probabilities,
        "top_actions": top_actions,
        "updates": updates,
        "replay_hash": stable_hash(snapshot_basis, length=24),
    }


def reinforced_pairs(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (event["context_decision"], event["agent_action"])
        for event in events
        if event.get("reinforce_action")
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_feedback(Path(args.feedback_json))
    feedback_validation = validate_feedback_artifact(artifact)
    events = artifact.get("feedback_events", [])
    policy = update_policy(events, base_weight=args.base_weight, learning_rate=args.learning_rate)
    replay_policy = update_policy(events, base_weight=args.base_weight, learning_rate=args.learning_rate)
    deterministic_replay = policy == replay_policy
    pairs = reinforced_pairs(events)
    reinforced_gains = [
        policy["weight_gains"][decision][action]
        for decision, action in pairs
    ]
    probability_sums = {
        decision: sum(policy["probabilities"][decision].values())
        for decision in DECISIONS
    }
    probability_sums_valid = all(abs(value - 1.0) <= 1e-9 for value in probability_sums.values())
    expected_top_actions_valid = all(
        policy["top_actions"][decision] == EXPECTED_TOP_ACTION[decision]
        for decision in DECISIONS
    )
    no_cross_context_leakage = (
        policy["top_actions"]["same_subject_different_context"] != "request_more_evidence"
        and policy["top_actions"]["request_context"] != "answer_from_context_memory"
        and policy["top_actions"]["flag_conflict"] != "answer_with_context_scope"
    )
    protocol_evidence = {
        "valid": bool(feedback_validation["valid"] and events and deterministic_replay),
        "feedback_json": args.feedback_json,
        "feedback_event_count": feedback_validation["feedback_event_count"],
        "policy_id": policy["policy_id"],
        "deterministic_replay": deterministic_replay,
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "feedback_artifact_valid": feedback_validation["valid"],
        "reinforced_pairs_gain_positive": all(gain > 0.0 for gain in reinforced_gains),
        "probability_sums": probability_sums_valid,
        "expected_top_actions": expected_top_actions_valid,
        "no_cross_context_leakage": no_cross_context_leakage,
        "mean_reinforced_gain": mean(reinforced_gains) >= args.mean_gain_min,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage19_contextual_policy_update",
        "config": vars(args),
        "feedback_validation": feedback_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "mean_reinforced_gain": mean(reinforced_gains),
            "reinforced_pair_count": len(pairs),
            "probability_sums": probability_sums,
            "no_cross_context_leakage": no_cross_context_leakage,
        },
        "policy": policy,
        "claim_guardrail": (
            "Stage 19 updates a small contextual action policy from deterministic "
            "feedback. It is not autonomous infinite RL."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 19 contextual policy update failed: {pass_fail}")
    print("NeoCortex-X Stage 19 contextual policy update")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  policy_id: {policy['policy_id']}")
    print(f"  mean_reinforced_gain: {payload['metrics']['mean_reinforced_gain']:.4f}")
    print(f"  top_actions: {json.dumps(policy['top_actions'], sort_keys=True)}")
    print(f"  no_cross_context_leakage: {no_cross_context_leakage}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 19 contextual policy update.")
    parser.add_argument("--feedback-json", default="artifacts/neocortex_x/stage18_contextual_feedback.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage19_contextual_policy_update.json")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--mean-gain-min", type=float, default=0.05)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
