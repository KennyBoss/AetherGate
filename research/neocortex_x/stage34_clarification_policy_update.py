#!/usr/bin/env python3
"""Stage 34 clarification-key policy update for NeoCortex-X memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stable_hash(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def validate_feedback_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if artifact.get("kind") != "neocortex_x_stage33_clarification_feedback_ledger":
        errors.append("invalid stage33 kind")
    if artifact.get("pass") is not True:
        errors.append("stage33 feedback artifact did not pass")
    events = artifact.get("feedback_events")
    if not isinstance(events, list) or not events:
        errors.append("feedback_events missing")
        events = []
    required = {"feedback_id", "event_type", "reward", "success", "replay_hash"}
    for index, event in enumerate(events):
        missing = required.difference(event)
        if missing:
            errors.append(f"feedback event {index} missing {sorted(missing)}")
        key = event.get("reinforce_key")
        if key is not None and key not in CONTEXT_KEYS:
            errors.append(f"feedback event {index} unknown reinforce_key {key}")
    return {"valid": not errors, "errors": errors, "feedback_event_count": len(events)}


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        return {key: 1.0 / len(weights) for key in weights}
    return {key: max(0.0, value) / total for key, value in weights.items()}


def top_key(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=lambda key: (probabilities[key], key))


def update_policy(
    events: list[dict[str, Any]],
    *,
    base_weight: float,
    learning_rate: float,
) -> dict[str, Any]:
    before = {key: base_weight for key in CONTEXT_KEYS}
    after = dict(before)
    updates = []
    for event in events:
        key = event.get("reinforce_key")
        if not key:
            continue
        reward = float(event["reward"])
        delta = learning_rate * reward
        after[key] = max(0.0, after[key] + delta)
        updates.append(
            {
                "feedback_id": event["feedback_id"],
                "reinforce_key": key,
                "reward": reward,
                "delta": delta,
                "weight_after": after[key],
            }
        )
    probabilities = normalize(after)
    gains = {key: after[key] - before[key] for key in CONTEXT_KEYS}
    ranked_keys = sorted(
        CONTEXT_KEYS,
        key=lambda key: (probabilities[key], gains[key], key),
        reverse=True,
    )
    snapshot_basis = {
        "before": before,
        "after": after,
        "gains": gains,
        "probabilities": probabilities,
        "ranked_keys": ranked_keys,
        "updates": updates,
    }
    return {
        "policy_id": f"clarpolicy_{stable_hash(snapshot_basis)}",
        "before_weights": before,
        "after_weights": after,
        "weight_gains": gains,
        "probabilities": probabilities,
        "top_key": top_key(probabilities),
        "ranked_keys": ranked_keys,
        "updates": updates,
        "replay_hash": stable_hash(snapshot_basis, length=24),
    }


def reinforced_keys(events: list[dict[str, Any]]) -> list[str]:
    return [event["reinforce_key"] for event in events if event.get("reinforce_key")]


def run(args: argparse.Namespace) -> dict[str, Any]:
    feedback = load_json(Path(args.feedback_json))
    feedback_validation = validate_feedback_artifact(feedback)
    events = feedback.get("feedback_events", [])
    policy = update_policy(events, base_weight=args.base_weight, learning_rate=args.learning_rate)
    replay_policy = update_policy(events, base_weight=args.base_weight, learning_rate=args.learning_rate)
    deterministic_replay = policy == replay_policy
    keys = reinforced_keys(events)
    reinforced_key_set = set(keys)
    reinforced_gains = [policy["weight_gains"][key] for key in reinforced_key_set]
    non_reinforced_keys = [key for key in CONTEXT_KEYS if key not in reinforced_key_set]
    non_reinforced_gains = [policy["weight_gains"][key] for key in non_reinforced_keys]
    probability_sum = sum(policy["probabilities"].values())
    top_key_reinforced = policy["top_key"] in reinforced_key_set
    ranked_reinforced_first = all(
        policy["ranked_keys"].index(key) < len(reinforced_key_set)
        for key in reinforced_key_set
    )
    pass_fail = {
        "feedback_artifact_valid": feedback_validation["valid"],
        "updates_present": bool(policy["updates"]),
        "reinforced_keys_present": bool(reinforced_key_set),
        "reinforced_key_gains_positive": all(gain > 0.0 for gain in reinforced_gains),
        "non_reinforced_keys_unchanged": all(abs(gain) <= 1e-12 for gain in non_reinforced_gains),
        "mean_reinforced_gain": mean(reinforced_gains) >= args.mean_gain_min,
        "probability_sum": abs(probability_sum - 1.0) <= 1e-9,
        "top_key_reinforced": top_key_reinforced,
        "ranked_reinforced_keys_first": ranked_reinforced_first,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "feedback_json": args.feedback_json,
        "feedback_event_count": feedback_validation["feedback_event_count"],
        "policy_id": policy["policy_id"],
        "reinforced_keys": sorted(reinforced_key_set),
        "top_key": policy["top_key"],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage34_clarification_policy_update",
        "config": vars(args),
        "feedback_validation": feedback_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "mean_reinforced_gain": mean(reinforced_gains),
            "reinforced_key_count": len(reinforced_key_set),
            "update_count": len(policy["updates"]),
            "probability_sum": probability_sum,
        },
        "policy": policy,
        "claim_guardrail": (
            "Stage 34 updates a deterministic clarification-key policy from "
            "bounded feedback. It is not autonomous RL."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 34 clarification policy update failed: {pass_fail}")
    print("NeoCortex-X Stage 34 clarification policy update")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  policy_id: {policy['policy_id']}")
    print(f"  top_key: {policy['top_key']}")
    print(f"  ranked_keys: {','.join(policy['ranked_keys'])}")
    print(f"  mean_reinforced_gain: {payload['metrics']['mean_reinforced_gain']:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X clarification policy update.")
    parser.add_argument(
        "--feedback-json",
        default="artifacts/neocortex_x/stage33_clarification_feedback_ledger.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage34_clarification_policy_update.json",
    )
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--mean-gain-min", type=float, default=1.0)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
