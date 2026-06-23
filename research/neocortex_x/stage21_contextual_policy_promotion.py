#!/usr/bin/env python3
"""Stage 21 promotion manifest for NeoCortex-X contextual action policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_ACTIONS = {
    "answer_from_context_memory",
    "answer_with_context_scope",
    "ask_context_question",
    "request_more_evidence",
}

EXPECTED_TOP_ACTIONS = {
    "context_match": "answer_from_context_memory",
    "same_subject_different_context": "answer_with_context_scope",
    "request_context": "ask_context_question",
    "flag_conflict": "request_more_evidence",
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
    if artifact.get("kind") != "neocortex_x_stage20_contextual_policy_replay":
        errors.append("invalid replay kind")
    if artifact.get("pass") is not True:
        errors.append("stage20 contextual replay artifact did not pass")
    aggregate = artifact.get("aggregate", {})
    guided = aggregate.get("agents", {}).get("contextual_policy_guided", {})
    comparison = aggregate.get("comparison", {})
    selected_actions = set(artifact.get("protocol_evidence", {}).get("selected_actions", []))
    pass_fail = artifact.get("pass_fail", {})
    no_false_evidence = bool(pass_fail.get("no_false_evidence_request_for_contextual_alias"))
    return {
        "valid": not errors,
        "errors": errors,
        "guided_action_accuracy": float(guided.get("action_accuracy", 0.0)),
        "guided_minus_global_prior_accuracy": float(
            comparison.get("guided_minus_global_prior_accuracy", 0.0)
        ),
        "selected_actions": sorted(selected_actions),
        "selected_actions_cover_gate": REQUIRED_ACTIONS.issubset(selected_actions),
        "no_false_evidence_request_for_contextual_alias": no_false_evidence,
        "policy_id": artifact.get("protocol_evidence", {}).get("policy_id"),
    }


def validate_policy_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage19_contextual_policy_update":
        errors.append("invalid policy kind")
    if artifact.get("pass") is not True:
        errors.append("stage19 contextual policy artifact did not pass")
    policy = artifact.get("policy", {})
    probabilities = policy.get("probabilities", {})
    top_actions = policy.get("top_actions", {})
    if not probabilities:
        errors.append("policy probabilities missing")
    for decision, expected_action in EXPECTED_TOP_ACTIONS.items():
        bucket = probabilities.get(decision)
        if not isinstance(bucket, dict):
            errors.append(f"policy missing bucket {decision}")
            continue
        missing_actions = REQUIRED_ACTIONS.difference(bucket)
        if missing_actions:
            errors.append(f"policy bucket {decision} missing actions {sorted(missing_actions)}")
        probability_sum = sum(float(value) for value in bucket.values())
        if abs(probability_sum - 1.0) > 1e-9:
            errors.append(f"policy bucket {decision} probabilities do not sum to 1.0")
        if top_actions.get(decision) != expected_action:
            errors.append(f"policy top action for {decision} is not {expected_action}")
    return {
        "valid": not errors,
        "errors": errors,
        "policy_id": policy.get("policy_id"),
        "replay_hash": policy.get("replay_hash"),
        "probabilities": probabilities,
        "top_actions": top_actions,
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
        "top_actions": policy_validation["top_actions"],
    }
    return {
        "manifest_id": f"ctxpromoted_{stable_hash(manifest_basis)}",
        "policy_id": policy_validation["policy_id"],
        "policy_replay_hash": policy_validation["replay_hash"],
        "runtime_probabilities": policy_validation["probabilities"],
        "runtime_top_actions": policy_validation["top_actions"],
        "promotion_thresholds": {
            "guided_accuracy_min": args.guided_accuracy_min,
            "global_prior_margin_min": args.global_prior_margin_min,
        },
        "promotion_evidence": {
            "replay_json": args.replay_json,
            "policy_json": args.policy_json,
            "guided_action_accuracy": replay_validation["guided_action_accuracy"],
            "guided_minus_global_prior_accuracy": replay_validation[
                "guided_minus_global_prior_accuracy"
            ],
            "selected_actions": replay_validation["selected_actions"],
            "no_false_evidence_request_for_contextual_alias": replay_validation[
                "no_false_evidence_request_for_contextual_alias"
            ],
        },
        "rollback": {
            "previous_policy": "uniform_contextual_action_policy_v0",
            "restore_command": "make neocortex-x-stage19-contextual-policy-smoke",
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
        "guided_minus_global_prior_accuracy": (
            replay_validation["guided_minus_global_prior_accuracy"] >= args.global_prior_margin_min
        ),
        "selected_actions_cover_gate": replay_validation["selected_actions_cover_gate"],
        "no_false_evidence_request_for_contextual_alias": replay_validation[
            "no_false_evidence_request_for_contextual_alias"
        ],
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
        "kind": "neocortex_x_stage21_contextual_policy_promotion",
        "config": vars(args),
        "replay_validation": replay_validation,
        "policy_validation": policy_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "manifest": manifest,
        "claim_guardrail": (
            "Stage 21 promotes a deterministic replay-validated contextual policy "
            "manifest. It is not a production autonomy guarantee."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 21 contextual policy promotion failed: {pass_fail}")
    print("NeoCortex-X Stage 21 contextual policy promotion")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  manifest_id: {manifest['manifest_id']}")
    print(f"  policy_id: {manifest['policy_id']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a NeoCortex-X contextual policy.")
    parser.add_argument("--replay-json", default="artifacts/neocortex_x/stage20_contextual_policy_replay.json")
    parser.add_argument("--policy-json", default="artifacts/neocortex_x/stage19_contextual_policy_update.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage21_contextual_policy_promotion.json")
    parser.add_argument("--guided-accuracy-min", type=float, default=0.95)
    parser.add_argument("--global-prior-margin-min", type=float, default=0.25)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
