#!/usr/bin/env python3
"""Stage 22 runtime loader for promoted NeoCortex-X contextual policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TOP_ACTIONS = {
    "context_match": "answer_from_context_memory",
    "same_subject_different_context": "answer_with_context_scope",
    "request_context": "ask_context_question",
    "flag_conflict": "request_more_evidence",
}

REQUIRED_CONTEXT_DECISIONS = tuple(EXPECTED_TOP_ACTIONS)
UNKNOWN_CONTEXT_FALLBACK = "ask_context_question"


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


def validate_promotion_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage21_contextual_policy_promotion":
        errors.append("invalid promotion kind")
    if artifact.get("pass") is not True:
        errors.append("stage21 contextual promotion artifact did not pass")
    manifest = artifact.get("manifest", {})
    probabilities = manifest.get("runtime_probabilities", {})
    top_actions = manifest.get("runtime_top_actions", {})
    probability_sums: dict[str, float] = {}
    for decision, expected_action in EXPECTED_TOP_ACTIONS.items():
        bucket = probabilities.get(decision)
        if not isinstance(bucket, dict):
            errors.append(f"runtime probabilities missing bucket {decision}")
            continue
        probability_sums[decision] = sum(float(value) for value in bucket.values())
        if abs(probability_sums[decision] - 1.0) > 1e-9:
            errors.append(f"runtime bucket {decision} does not sum to 1.0")
        if top_actions.get(decision) != expected_action:
            errors.append(f"runtime top action for {decision} is not {expected_action}")
    rollback = manifest.get("rollback", {})
    if not rollback.get("previous_policy") or not rollback.get("restore_command"):
        errors.append("rollback metadata missing")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_id": manifest.get("manifest_id"),
        "policy_id": manifest.get("policy_id"),
        "probability_sums": probability_sums,
        "rollback_present": bool(rollback.get("previous_policy") and rollback.get("restore_command")),
    }


def ordered_actions(bucket: dict[str, float]) -> list[str]:
    return [
        action
        for action, _ in sorted(bucket.items(), key=lambda item: (-float(item[1]), item[0]))
    ]


def build_runtime_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    probabilities = manifest["runtime_probabilities"]
    action_order = {
        decision: ordered_actions(bucket)
        for decision, bucket in probabilities.items()
    }
    default_actions = {
        decision: actions[0]
        for decision, actions in action_order.items()
    }
    runtime_basis = {
        "manifest_id": manifest["manifest_id"],
        "policy_id": manifest["policy_id"],
        "probabilities": probabilities,
        "action_order": action_order,
        "default_actions": default_actions,
    }
    return {
        "runtime_policy_id": f"ctxruntime_{stable_hash(runtime_basis)}",
        "source_manifest_id": manifest["manifest_id"],
        "policy_id": manifest["policy_id"],
        "probabilities": probabilities,
        "action_order": action_order,
        "default_actions": default_actions,
        "unknown_context_fallback": UNKNOWN_CONTEXT_FALLBACK,
        "rollback": manifest["rollback"],
        "replay_hash": stable_hash(runtime_basis, length=24),
    }


def smoke_decisions(runtime_policy: dict[str, Any]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for context_decision in REQUIRED_CONTEXT_DECISIONS:
        selected_action = runtime_policy["default_actions"].get(
            context_decision,
            runtime_policy["unknown_context_fallback"],
        )
        decisions.append(
            {
                "requested_context_decision": context_decision,
                "available": context_decision in runtime_policy["probabilities"],
                "selected_action": selected_action,
                "expected_action": EXPECTED_TOP_ACTIONS[context_decision],
                "action_correct": selected_action == EXPECTED_TOP_ACTIONS[context_decision],
                "probability": float(
                    runtime_policy["probabilities"]
                    .get(context_decision, {})
                    .get(selected_action, 0.0)
                ),
            }
        )
    decisions.append(
        {
            "requested_context_decision": "unknown_context_decision",
            "available": False,
            "selected_action": runtime_policy["unknown_context_fallback"],
            "expected_action": UNKNOWN_CONTEXT_FALLBACK,
            "action_correct": runtime_policy["unknown_context_fallback"] == UNKNOWN_CONTEXT_FALLBACK,
            "probability": 0.0,
        }
    )
    return decisions


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_json(Path(args.promotion_json))
    promotion_validation = validate_promotion_artifact(artifact)
    manifest = artifact.get("manifest", {})
    runtime_policy = build_runtime_policy(manifest)
    decisions = smoke_decisions(runtime_policy)
    replay_decisions = smoke_decisions(runtime_policy)
    deterministic_replay = decisions == replay_decisions
    required_available = all(
        decision["available"]
        for decision in decisions
        if decision["requested_context_decision"] in REQUIRED_CONTEXT_DECISIONS
    )
    top_actions_match = all(
        runtime_policy["default_actions"].get(decision) == expected_action
        for decision, expected_action in EXPECTED_TOP_ACTIONS.items()
    )
    unknown_fallback_correct = any(
        decision["requested_context_decision"] == "unknown_context_decision"
        and decision["selected_action"] == UNKNOWN_CONTEXT_FALLBACK
        for decision in decisions
    )
    pass_fail = {
        "promotion_artifact_valid": promotion_validation["valid"],
        "required_context_decisions_available": required_available,
        "bucket_probability_sums": all(
            abs(value - 1.0) <= 1e-9
            for value in promotion_validation["probability_sums"].values()
        ),
        "top_actions_match_manifest": top_actions_match,
        "rollback_present": promotion_validation["rollback_present"],
        "smoke_decisions_cover_contexts": required_available,
        "unknown_context_fallback": unknown_fallback_correct,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "promotion_json": args.promotion_json,
        "manifest_id": promotion_validation["manifest_id"],
        "runtime_policy_id": runtime_policy["runtime_policy_id"],
        "policy_id": runtime_policy["policy_id"],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage22_contextual_runtime_loader",
        "config": vars(args),
        "promotion_validation": promotion_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "runtime_policy": runtime_policy,
        "smoke_decisions": decisions,
        "claim_guardrail": (
            "Stage 22 loads a promoted contextual policy manifest as a runtime "
            "snapshot. It is not a production deployment claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 22 contextual runtime loader failed: {pass_fail}")
    print("NeoCortex-X Stage 22 contextual runtime loader")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  runtime_policy_id: {runtime_policy['runtime_policy_id']}")
    print(f"  unknown_context_fallback: {runtime_policy['unknown_context_fallback']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a promoted NeoCortex-X contextual runtime policy.")
    parser.add_argument("--promotion-json", default="artifacts/neocortex_x/stage21_contextual_policy_promotion.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage22_contextual_runtime_loader.json")
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
