#!/usr/bin/env python3
"""Stage 36 promotion manifest for NeoCortex-X clarification-key policy."""

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


def validate_replay_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if artifact.get("kind") != "neocortex_x_stage35_clarification_policy_replay":
        errors.append("invalid stage35 replay kind")
    if artifact.get("pass") is not True:
        errors.append("stage35 replay artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    aggregate = artifact.get("aggregate", {})
    agents = aggregate.get("agents", {})
    policy = agents.get("policy_guided_replay", {})
    comparison = aggregate.get("comparison", {})
    ranked_keys = protocol.get("ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("replay ranked_keys missing")
        ranked_keys = []
    ranked_set = set(ranked_keys)
    if ranked_set != set(CONTEXT_KEYS):
        errors.append("replay ranked_keys do not cover CONTEXT_KEYS")
    selected_keys = policy.get("selected_keys")
    if not isinstance(selected_keys, list):
        errors.append("policy replay selected_keys missing")
        selected_keys = []
    selected_from_ranked = set(selected_keys).issubset(ranked_set)
    if not selected_from_ranked:
        errors.append("policy replay selected keys are not subset of ranked_keys")
    return {
        "valid": not errors,
        "errors": errors,
        "policy_id": protocol.get("policy_id"),
        "ranked_keys": ranked_keys,
        "top_key": protocol.get("top_key"),
        "query_count": int(protocol.get("query_count", 0)),
        "resolution_accuracy": float(policy.get("resolution_accuracy", 0.0)),
        "ambiguous_resolution_accuracy": float(
            policy.get("ambiguous_resolution_accuracy", 0.0)
        ),
        "target_preservation_rate": float(policy.get("target_preservation_rate", 0.0)),
        "max_steps": int(policy.get("max_steps", 0)),
        "mean_steps_for_ambiguous": float(policy.get("mean_steps_for_ambiguous", 0.0)),
        "selected_keys": sorted(selected_keys),
        "selected_keys_from_ranked_list": selected_from_ranked,
        "fixed_minus_policy_mean_ambiguous_steps": float(
            comparison.get("fixed_minus_policy_mean_ambiguous_steps", 0.0)
        ),
        "policy_minus_always_top_accuracy": float(
            comparison.get("policy_minus_always_top_accuracy", 0.0)
        ),
    }


def validate_policy_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if artifact.get("kind") != "neocortex_x_stage34_clarification_policy_update":
        errors.append("invalid stage34 policy kind")
    if artifact.get("pass") is not True:
        errors.append("stage34 policy artifact did not pass")
    policy = artifact.get("policy", {})
    ranked_keys = policy.get("ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("policy ranked_keys missing")
        ranked_keys = []
    if set(ranked_keys) != set(CONTEXT_KEYS):
        errors.append("policy ranked_keys do not cover CONTEXT_KEYS")
    top_key = policy.get("top_key")
    if top_key not in set(ranked_keys):
        errors.append("policy top_key missing from ranked_keys")
    probabilities = policy.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        errors.append("policy probabilities missing")
        probabilities = {}
    missing_probabilities = set(CONTEXT_KEYS).difference(probabilities)
    if missing_probabilities:
        errors.append(f"policy probabilities missing {sorted(missing_probabilities)}")
    probability_sum = sum(float(probabilities.get(key, 0.0)) for key in CONTEXT_KEYS)
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("policy probabilities do not sum to 1.0")
    return {
        "valid": not errors,
        "errors": errors,
        "policy_id": policy.get("policy_id"),
        "replay_hash": policy.get("replay_hash"),
        "ranked_keys": ranked_keys,
        "top_key": top_key,
        "probabilities": probabilities,
        "probability_sum": probability_sum,
    }


def build_manifest(
    *,
    replay_validation: dict[str, Any],
    policy_validation: dict[str, Any],
    args: argparse.Namespace,
    created_at: str,
) -> dict[str, Any]:
    manifest_basis = {
        "policy_id": policy_validation["policy_id"],
        "policy_hash": policy_validation["replay_hash"],
        "replay_json": args.replay_json,
        "policy_json": args.policy_json,
        "ranked_keys": policy_validation["ranked_keys"],
        "top_key": policy_validation["top_key"],
        "probabilities": policy_validation["probabilities"],
        "promotion_metrics": {
            "resolution_accuracy": replay_validation["resolution_accuracy"],
            "ambiguous_resolution_accuracy": replay_validation[
                "ambiguous_resolution_accuracy"
            ],
            "target_preservation_rate": replay_validation["target_preservation_rate"],
            "fixed_minus_policy_mean_ambiguous_steps": replay_validation[
                "fixed_minus_policy_mean_ambiguous_steps"
            ],
            "policy_minus_always_top_accuracy": replay_validation[
                "policy_minus_always_top_accuracy"
            ],
        },
    }
    return {
        "manifest_id": f"clarpromoted_{stable_hash(manifest_basis)}",
        "policy_id": policy_validation["policy_id"],
        "policy_replay_hash": policy_validation["replay_hash"],
        "runtime_ranked_keys": policy_validation["ranked_keys"],
        "runtime_top_key": policy_validation["top_key"],
        "runtime_probabilities": policy_validation["probabilities"],
        "runtime_contract": {
            "selection_rule": "ask first available missing key from runtime_ranked_keys",
            "unique_candidate_rule": "answer without clarification",
            "stop_rule": "stop after unique target slot or after all context dimensions",
            "context_dimensions": list(CONTEXT_KEYS),
        },
        "promotion_thresholds": {
            "resolution_accuracy_min": args.resolution_accuracy_min,
            "step_gain_vs_fixed_min": args.step_gain_vs_fixed_min,
            "always_top_accuracy_margin_min": args.always_top_accuracy_margin_min,
        },
        "promotion_evidence": {
            "replay_json": args.replay_json,
            "policy_json": args.policy_json,
            "query_count": replay_validation["query_count"],
            "resolution_accuracy": replay_validation["resolution_accuracy"],
            "ambiguous_resolution_accuracy": replay_validation[
                "ambiguous_resolution_accuracy"
            ],
            "target_preservation_rate": replay_validation["target_preservation_rate"],
            "max_steps": replay_validation["max_steps"],
            "mean_steps_for_ambiguous": replay_validation["mean_steps_for_ambiguous"],
            "selected_keys": replay_validation["selected_keys"],
            "fixed_minus_policy_mean_ambiguous_steps": replay_validation[
                "fixed_minus_policy_mean_ambiguous_steps"
            ],
            "policy_minus_always_top_accuracy": replay_validation[
                "policy_minus_always_top_accuracy"
            ],
        },
        "rollback": {
            "previous_policy": "static_entropy_clarification_key_policy_v0",
            "fallback_agent": "static_entropy_control",
            "restore_command": "make neocortex-x-stage31-disambiguation-gate-smoke",
        },
        "created_at": created_at,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    replay_artifact = load_json(Path(args.replay_json))
    policy_artifact = load_json(Path(args.policy_json))
    replay_validation = validate_replay_artifact(replay_artifact)
    policy_validation = validate_policy_artifact(policy_artifact)
    policy_ids_match = replay_validation["policy_id"] == policy_validation["policy_id"]
    ranked_keys_match = replay_validation["ranked_keys"] == policy_validation["ranked_keys"]
    top_key_matches = replay_validation["top_key"] == policy_validation["top_key"]
    created_at = timestamp()
    manifest = build_manifest(
        replay_validation=replay_validation,
        policy_validation=policy_validation,
        args=args,
        created_at=created_at,
    )
    pass_fail = {
        "replay_artifact_valid": replay_validation["valid"],
        "policy_artifact_valid": policy_validation["valid"],
        "policy_ids_match": policy_ids_match,
        "ranked_keys_match": ranked_keys_match,
        "top_key_matches": top_key_matches,
        "policy_resolution_accuracy": (
            replay_validation["resolution_accuracy"] >= args.resolution_accuracy_min
        ),
        "policy_ambiguous_resolution_accuracy": (
            replay_validation["ambiguous_resolution_accuracy"]
            >= args.resolution_accuracy_min
        ),
        "policy_target_preservation_rate": (
            replay_validation["target_preservation_rate"] == 1.0
        ),
        "policy_bounded_steps": replay_validation["max_steps"] <= len(CONTEXT_KEYS),
        "policy_step_efficiency_vs_fixed": (
            replay_validation["fixed_minus_policy_mean_ambiguous_steps"]
            >= args.step_gain_vs_fixed_min
        ),
        "policy_beats_always_top_accuracy": (
            replay_validation["policy_minus_always_top_accuracy"]
            >= args.always_top_accuracy_margin_min
        ),
        "selected_keys_from_ranked_list": replay_validation[
            "selected_keys_from_ranked_list"
        ],
        "runtime_contract_present": bool(manifest["runtime_contract"]["selection_rule"]),
        "rollback_present": bool(manifest["rollback"]["previous_policy"]),
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "replay_json": args.replay_json,
        "policy_json": args.policy_json,
        "manifest_id": manifest["manifest_id"],
        "policy_id": manifest["policy_id"],
        "runtime_ranked_keys": manifest["runtime_ranked_keys"],
        "runtime_top_key": manifest["runtime_top_key"],
    }
    payload = {
        "created_at": created_at,
        "kind": "neocortex_x_stage36_clarification_policy_promotion",
        "config": vars(args),
        "replay_validation": replay_validation,
        "policy_validation": policy_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "manifest": manifest,
        "claim_guardrail": (
            "Stage 36 promotes a deterministic replay-validated "
            "clarification-key policy manifest. It is not a production "
            "autonomy guarantee."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 36 clarification-policy promotion failed: {pass_fail}")
    print("NeoCortex-X Stage 36 clarification-policy promotion")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  manifest_id: {manifest['manifest_id']}")
    print(f"  policy_id: {manifest['policy_id']}")
    print(f"  runtime_ranked_keys: {','.join(manifest['runtime_ranked_keys'])}")
    print(f"  runtime_top_key: {manifest['runtime_top_key']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a NeoCortex-X clarification-key policy.")
    parser.add_argument(
        "--replay-json",
        default="artifacts/neocortex_x/stage35_clarification_policy_replay.json",
    )
    parser.add_argument(
        "--policy-json",
        default="artifacts/neocortex_x/stage34_clarification_policy_update.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage36_clarification_policy_promotion.json",
    )
    parser.add_argument("--resolution-accuracy-min", type=float, default=0.95)
    parser.add_argument("--step-gain-vs-fixed-min", type=float, default=0.25)
    parser.add_argument("--always-top-accuracy-margin-min", type=float, default=0.10)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
