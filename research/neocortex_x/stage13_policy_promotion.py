#!/usr/bin/env python3
"""Stage 13 promotion manifest for NeoCortex-X runtime action policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_ACTIONS = {
    "answer_from_memory",
    "update_memory_and_answer",
    "request_more_evidence",
}


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


def validate_replay_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage12_policy_replay":
        errors.append("invalid replay kind")
    if artifact.get("pass") is not True:
        errors.append("stage12 replay artifact did not pass")
    aggregate = artifact.get("aggregate", {})
    guided = aggregate.get("agents", {}).get("policy_guided_gate", {})
    comparison = aggregate.get("comparison", {})
    selected_actions = set(artifact.get("protocol_evidence", {}).get("selected_actions", []))
    return {
        "valid": not errors,
        "errors": errors,
        "guided_action_accuracy": float(guided.get("action_accuracy", 0.0)),
        "guided_minus_state_blind_accuracy": float(
            comparison.get("guided_minus_state_blind_accuracy", 0.0)
        ),
        "selected_actions": sorted(selected_actions),
        "selected_actions_cover_gate": REQUIRED_ACTIONS.issubset(selected_actions),
        "policy_id": artifact.get("protocol_evidence", {}).get("policy_id"),
    }


def validate_policy_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage11_policy_update":
        errors.append("invalid policy kind")
    if artifact.get("pass") is not True:
        errors.append("stage11 policy artifact did not pass")
    policy = artifact.get("policy", {})
    probabilities = policy.get("probabilities", {})
    missing = REQUIRED_ACTIONS.difference(probabilities)
    if missing:
        errors.append(f"policy missing actions {sorted(missing)}")
    probability_sum = sum(float(probabilities.get(action, 0.0)) for action in probabilities)
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("policy probabilities do not sum to 1.0")
    return {
        "valid": not errors,
        "errors": errors,
        "policy_id": policy.get("policy_id"),
        "replay_hash": policy.get("replay_hash"),
        "probabilities": probabilities,
        "probability_sum": probability_sum,
    }


def build_manifest(
    *,
    replay_validation: dict[str, Any],
    policy_validation: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest_basis = {
        "policy_id": policy_validation["policy_id"],
        "policy_hash": policy_validation["replay_hash"],
        "replay_json": args.replay_json,
        "policy_json": args.policy_json,
        "probabilities": policy_validation["probabilities"],
    }
    return {
        "manifest_id": f"promoted_{stable_hash(manifest_basis)}",
        "policy_id": policy_validation["policy_id"],
        "policy_replay_hash": policy_validation["replay_hash"],
        "runtime_probabilities": policy_validation["probabilities"],
        "promotion_thresholds": {
            "guided_accuracy_min": args.guided_accuracy_min,
            "state_blind_margin_min": args.state_blind_margin_min,
        },
        "promotion_evidence": {
            "replay_json": args.replay_json,
            "policy_json": args.policy_json,
            "guided_action_accuracy": replay_validation["guided_action_accuracy"],
            "guided_minus_state_blind_accuracy": replay_validation[
                "guided_minus_state_blind_accuracy"
            ],
            "selected_actions": replay_validation["selected_actions"],
        },
        "rollback": {
            "previous_policy": "uniform_action_policy_v0",
            "restore_command": "make neocortex-x-stage11-policy-smoke",
        },
        "created_at": timestamp(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    replay_artifact = load_json(Path(args.replay_json))
    policy_artifact = load_json(Path(args.policy_json))
    replay_validation = validate_replay_artifact(replay_artifact)
    policy_validation = validate_policy_artifact(policy_artifact)
    policy_ids_match = replay_validation["policy_id"] == policy_validation["policy_id"]
    manifest = build_manifest(
        replay_validation=replay_validation,
        policy_validation=policy_validation,
        args=args,
    )
    pass_fail = {
        "replay_artifact_valid": replay_validation["valid"],
        "policy_artifact_valid": policy_validation["valid"],
        "policy_ids_match": policy_ids_match,
        "guided_action_accuracy": (
            replay_validation["guided_action_accuracy"] >= args.guided_accuracy_min
        ),
        "guided_minus_state_blind_accuracy": (
            replay_validation["guided_minus_state_blind_accuracy"] >= args.state_blind_margin_min
        ),
        "selected_actions_cover_gate": replay_validation["selected_actions_cover_gate"],
        "rollback_present": bool(manifest["rollback"]["previous_policy"]),
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "replay_json": args.replay_json,
        "policy_json": args.policy_json,
        "manifest_id": manifest["manifest_id"],
        "policy_id": manifest["policy_id"],
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage13_policy_promotion",
        "config": vars(args),
        "replay_validation": replay_validation,
        "policy_validation": policy_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "manifest": manifest,
        "claim_guardrail": (
            "Stage 13 promotes a deterministic replay-validated policy manifest. "
            "It is not a production autonomy guarantee."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 13 policy promotion failed: {pass_fail}")
    print("NeoCortex-X Stage 13 policy promotion")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  manifest_id: {manifest['manifest_id']}")
    print(f"  policy_id: {manifest['policy_id']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a NeoCortex-X replay-validated policy.")
    parser.add_argument("--replay-json", default="artifacts/neocortex_x/stage12_policy_replay.json")
    parser.add_argument("--policy-json", default="artifacts/neocortex_x/stage11_policy_update.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage13_policy_promotion.json")
    parser.add_argument("--guided-accuracy-min", type=float, default=0.95)
    parser.add_argument("--state-blind-margin-min", type=float, default=0.25)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
