#!/usr/bin/env python3
"""Stage 14 runtime loader for promoted NeoCortex-X action policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_ACTIONS = (
    "answer_from_memory",
    "update_memory_and_answer",
    "request_more_evidence",
)


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
    if artifact.get("kind") != "neocortex_x_stage13_policy_promotion":
        errors.append("invalid promotion kind")
    if artifact.get("pass") is not True:
        errors.append("stage13 promotion artifact did not pass")
    manifest = artifact.get("manifest", {})
    probabilities = manifest.get("runtime_probabilities", {})
    for action in REQUIRED_ACTIONS:
        if action not in probabilities:
            errors.append(f"runtime probabilities missing {action}")
    probability_sum = sum(float(probabilities.get(action, 0.0)) for action in probabilities)
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("runtime probabilities do not sum to 1.0")
    rollback = manifest.get("rollback", {})
    if not rollback.get("previous_policy") or not rollback.get("restore_command"):
        errors.append("rollback metadata missing")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_id": manifest.get("manifest_id"),
        "policy_id": manifest.get("policy_id"),
        "probability_sum": probability_sum,
        "rollback_present": bool(rollback.get("previous_policy") and rollback.get("restore_command")),
    }


def build_runtime_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    probabilities = manifest["runtime_probabilities"]
    sorted_actions = sorted(
        probabilities.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )
    runtime_basis = {
        "manifest_id": manifest["manifest_id"],
        "policy_id": manifest["policy_id"],
        "probabilities": probabilities,
        "sorted_actions": sorted_actions,
    }
    return {
        "runtime_policy_id": f"runtime_{stable_hash(runtime_basis)}",
        "source_manifest_id": manifest["manifest_id"],
        "policy_id": manifest["policy_id"],
        "probabilities": probabilities,
        "action_order": [action for action, _ in sorted_actions],
        "default_action": sorted_actions[0][0],
        "rollback": manifest["rollback"],
        "replay_hash": stable_hash(runtime_basis, length=24),
    }


def smoke_decisions(runtime_policy: dict[str, Any]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for action in REQUIRED_ACTIONS:
        decisions.append(
            {
                "requested_action": action,
                "available": action in runtime_policy["probabilities"],
                "probability": float(runtime_policy["probabilities"].get(action, 0.0)),
                "selected_action": action if action in runtime_policy["probabilities"] else runtime_policy["default_action"],
            }
        )
    decisions.append(
        {
            "requested_action": "unknown",
            "available": False,
            "probability": 0.0,
            "selected_action": runtime_policy["default_action"],
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
        decision["available"] for decision in decisions if decision["requested_action"] in REQUIRED_ACTIONS
    )
    pass_fail = {
        "promotion_artifact_valid": promotion_validation["valid"],
        "probability_sum": abs(promotion_validation["probability_sum"] - 1.0) <= 1e-9,
        "required_actions_available": required_available,
        "rollback_present": promotion_validation["rollback_present"],
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
        "kind": "neocortex_x_stage14_runtime_loader",
        "config": vars(args),
        "promotion_validation": promotion_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "runtime_policy": runtime_policy,
        "smoke_decisions": decisions,
        "claim_guardrail": (
            "Stage 14 loads a promoted policy manifest as a runtime snapshot. "
            "It is not a production deployment claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 14 runtime loader failed: {pass_fail}")
    print("NeoCortex-X Stage 14 runtime loader")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  runtime_policy_id: {runtime_policy['runtime_policy_id']}")
    print(f"  default_action: {runtime_policy['default_action']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a promoted NeoCortex-X runtime policy.")
    parser.add_argument("--promotion-json", default="artifacts/neocortex_x/stage13_policy_promotion.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage14_runtime_loader.json")
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
