#!/usr/bin/env python3
"""Stage 11 policy update from NeoCortex-X feedback events."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIONS = (
    "answer_from_memory",
    "update_memory_and_answer",
    "request_more_evidence",
)


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
    if artifact.get("kind") != "neocortex_x_stage10_feedback_loop":
        errors.append("invalid kind")
    if artifact.get("pass") is not True:
        errors.append("stage10 feedback artifact did not pass")
    events = artifact.get("feedback_events")
    if not isinstance(events, list) or not events:
        errors.append("feedback_events must be a non-empty list")
        events = []
    required = {"feedback_id", "agent_action", "reward", "reinforce_action", "replay_hash"}
    for index, event in enumerate(events):
        missing = required.difference(event)
        if missing:
            errors.append(f"feedback event {index} missing {sorted(missing)}")
        if event.get("agent_action") not in ACTIONS:
            errors.append(f"feedback event {index} has unknown action {event.get('agent_action')}")
    return {"valid": not errors, "errors": errors, "feedback_event_count": len(events)}


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        return {action: 1.0 / len(weights) for action in weights}
    return {action: max(0.0, value) / total for action, value in weights.items()}


def update_policy(events: list[dict[str, Any]], *, base_weight: float, learning_rate: float) -> dict[str, Any]:
    before = {action: base_weight for action in ACTIONS}
    after = dict(before)
    updates: list[dict[str, Any]] = []
    for event in events:
        action = event["agent_action"]
        reward = float(event["reward"])
        delta = learning_rate * reward
        after[action] = max(0.0, after[action] + delta)
        updates.append(
            {
                "feedback_id": event["feedback_id"],
                "action": action,
                "reward": reward,
                "delta": delta,
                "weight_after": after[action],
            }
        )
    gains = {action: after[action] - before[action] for action in ACTIONS}
    probabilities = normalize(after)
    snapshot_basis = {
        "before": before,
        "after": after,
        "probabilities": probabilities,
        "updates": updates,
    }
    return {
        "before_weights": before,
        "after_weights": after,
        "weight_gains": gains,
        "probabilities": probabilities,
        "updates": updates,
        "policy_id": f"policy_{stable_hash(snapshot_basis)}",
        "replay_hash": stable_hash(snapshot_basis, length=24),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_feedback(Path(args.feedback_json))
    feedback_validation = validate_feedback_artifact(artifact)
    events = artifact.get("feedback_events", [])
    policy = update_policy(events, base_weight=args.base_weight, learning_rate=args.learning_rate)
    replay_policy = update_policy(events, base_weight=args.base_weight, learning_rate=args.learning_rate)
    deterministic_replay = policy == replay_policy
    reinforced_actions = sorted({event["agent_action"] for event in events if event.get("reinforce_action")})
    reinforced_gains = [policy["weight_gains"][action] for action in reinforced_actions]
    probability_sum = sum(policy["probabilities"].values())
    protocol_evidence = {
        "valid": bool(feedback_validation["valid"] and events and deterministic_replay),
        "feedback_json": args.feedback_json,
        "feedback_event_count": feedback_validation["feedback_event_count"],
        "reinforced_actions": reinforced_actions,
        "policy_id": policy["policy_id"],
        "deterministic_replay": deterministic_replay,
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "feedback_artifact_valid": feedback_validation["valid"],
        "reinforced_actions_gain_positive": all(gain > 0.0 for gain in reinforced_gains),
        "probability_sum": abs(probability_sum - 1.0) <= 1e-9,
        "mean_reinforced_gain": mean(reinforced_gains) >= args.mean_gain_min,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage11_policy_update",
        "config": vars(args),
        "feedback_validation": feedback_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "probability_sum": probability_sum,
            "mean_reinforced_gain": mean(reinforced_gains),
            "reinforced_action_count": len(reinforced_actions),
        },
        "policy": policy,
        "claim_guardrail": (
            "Stage 11 updates a small action policy from deterministic feedback. "
            "It is not autonomous infinite RL."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 11 policy update failed: {pass_fail}")
    print("NeoCortex-X Stage 11 policy update")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  policy_id: {policy['policy_id']}")
    print(f"  reinforced_actions: {','.join(reinforced_actions)}")
    print(f"  mean_reinforced_gain: {payload['metrics']['mean_reinforced_gain']:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 11 policy update.")
    parser.add_argument("--feedback-json", default="artifacts/neocortex_x/stage10_feedback_loop.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage11_policy_update.json")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--mean-gain-min", type=float, default=0.05)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
