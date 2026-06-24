#!/usr/bin/env python3
"""Stage 37 runtime loader for promoted NeoCortex-X clarification-key policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS


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
    errors = []
    if artifact.get("kind") != "neocortex_x_stage36_clarification_policy_promotion":
        errors.append("invalid stage36 promotion kind")
    if artifact.get("pass") is not True:
        errors.append("stage36 promotion artifact did not pass")
    manifest = artifact.get("manifest", {})
    ranked_keys = manifest.get("runtime_ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("runtime_ranked_keys missing")
        ranked_keys = []
    if set(ranked_keys) != set(CONTEXT_KEYS):
        errors.append("runtime_ranked_keys do not cover CONTEXT_KEYS")
    top_key = manifest.get("runtime_top_key")
    if top_key not in set(ranked_keys):
        errors.append("runtime_top_key missing from runtime_ranked_keys")
    probabilities = manifest.get("runtime_probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        errors.append("runtime_probabilities missing")
        probabilities = {}
    missing_probabilities = set(CONTEXT_KEYS).difference(probabilities)
    if missing_probabilities:
        errors.append(f"runtime_probabilities missing {sorted(missing_probabilities)}")
    probability_sum = sum(float(probabilities.get(key, 0.0)) for key in CONTEXT_KEYS)
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("runtime_probabilities do not sum to 1.0")
    runtime_contract = manifest.get("runtime_contract", {})
    if not runtime_contract.get("selection_rule"):
        errors.append("runtime selection rule missing")
    rollback = manifest.get("rollback", {})
    if not rollback.get("previous_policy") or not rollback.get("restore_command"):
        errors.append("rollback metadata missing")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_id": manifest.get("manifest_id"),
        "policy_id": manifest.get("policy_id"),
        "ranked_keys": ranked_keys,
        "top_key": top_key,
        "probabilities": probabilities,
        "probability_sum": probability_sum,
        "runtime_contract_present": bool(runtime_contract.get("selection_rule")),
        "rollback_present": bool(rollback.get("previous_policy") and rollback.get("restore_command")),
    }


def build_runtime_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    ranked_keys = manifest["runtime_ranked_keys"]
    probabilities = manifest["runtime_probabilities"]
    runtime_basis = {
        "manifest_id": manifest["manifest_id"],
        "policy_id": manifest["policy_id"],
        "ranked_keys": ranked_keys,
        "top_key": manifest["runtime_top_key"],
        "probabilities": probabilities,
        "runtime_contract": manifest["runtime_contract"],
    }
    return {
        "runtime_policy_id": f"clarruntime_{stable_hash(runtime_basis)}",
        "source_manifest_id": manifest["manifest_id"],
        "policy_id": manifest["policy_id"],
        "ranked_keys": ranked_keys,
        "top_key": manifest["runtime_top_key"],
        "probabilities": probabilities,
        "selection_rule": manifest["runtime_contract"]["selection_rule"],
        "unknown_context_fallback": UNKNOWN_CONTEXT_FALLBACK,
        "rollback": manifest["rollback"],
        "replay_hash": stable_hash(runtime_basis, length=24),
    }


def select_runtime_key(
    runtime_policy: dict[str, Any],
    partial_context: dict[str, str],
) -> str | None:
    for key in runtime_policy["ranked_keys"]:
        if key not in partial_context:
            return key
    return None


def smoke_cases(runtime_policy: dict[str, Any]) -> list[dict[str, Any]]:
    ranked_keys = runtime_policy["ranked_keys"]
    return [
        {
            "case_id": "empty_context",
            "partial_context": {},
            "expected_key": ranked_keys[0],
        },
        {
            "case_id": "top_key_known",
            "partial_context": {ranked_keys[0]: "known"},
            "expected_key": ranked_keys[1],
        },
        {
            "case_id": "first_two_known",
            "partial_context": {ranked_keys[0]: "known", ranked_keys[1]: "known"},
            "expected_key": ranked_keys[2],
        },
        {
            "case_id": "all_known",
            "partial_context": {key: "known" for key in ranked_keys},
            "expected_key": None,
        },
        {
            "case_id": "unknown_context_shape",
            "partial_context": {"unknown_context_key": "value"},
            "expected_key": ranked_keys[0],
        },
    ]


def smoke_decisions(runtime_policy: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = []
    for case in smoke_cases(runtime_policy):
        selected_key = select_runtime_key(runtime_policy, case["partial_context"])
        unknown_shape = any(key not in CONTEXT_KEYS for key in case["partial_context"])
        selected_action = (
            "answer_without_clarification"
            if selected_key is None
            else "ask_disambiguating_context"
        )
        decisions.append(
            {
                "case_id": case["case_id"],
                "partial_context": case["partial_context"],
                "selected_key": selected_key,
                "expected_key": case["expected_key"],
                "key_correct": selected_key == case["expected_key"],
                "selected_action": selected_action,
                "unknown_shape": unknown_shape,
                "unknown_context_fallback": (
                    runtime_policy["unknown_context_fallback"] if unknown_shape else None
                ),
                "selected_from_ranked_keys": (
                    selected_key is None or selected_key in runtime_policy["ranked_keys"]
                ),
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
    all_ranked_keys_available = set(runtime_policy["ranked_keys"]) == set(CONTEXT_KEYS)
    top_key_matches_rank = runtime_policy["top_key"] == runtime_policy["ranked_keys"][0]
    smoke_keys_correct = all(decision["key_correct"] for decision in decisions)
    full_context_no_key = any(
        decision["case_id"] == "all_known"
        and decision["selected_key"] is None
        and decision["selected_action"] == "answer_without_clarification"
        for decision in decisions
    )
    unknown_context_fallback_present = any(
        decision["case_id"] == "unknown_context_shape"
        and decision["unknown_context_fallback"] == UNKNOWN_CONTEXT_FALLBACK
        for decision in decisions
    )
    selected_from_ranked = all(
        decision["selected_from_ranked_keys"] for decision in decisions
    )
    pass_fail = {
        "promotion_artifact_valid": promotion_validation["valid"],
        "ranked_keys_cover_context_dimensions": all_ranked_keys_available,
        "probability_sum": abs(promotion_validation["probability_sum"] - 1.0) <= 1e-9,
        "top_key_matches_rank": top_key_matches_rank,
        "runtime_contract_present": promotion_validation["runtime_contract_present"],
        "rollback_present": promotion_validation["rollback_present"],
        "smoke_keys_correct": smoke_keys_correct,
        "full_context_no_clarification_key": full_context_no_key,
        "unknown_context_fallback_present": unknown_context_fallback_present,
        "selected_keys_from_ranked_list": selected_from_ranked,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "promotion_json": args.promotion_json,
        "manifest_id": promotion_validation["manifest_id"],
        "runtime_policy_id": runtime_policy["runtime_policy_id"],
        "policy_id": runtime_policy["policy_id"],
        "ranked_keys": runtime_policy["ranked_keys"],
        "top_key": runtime_policy["top_key"],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage37_clarification_runtime_loader",
        "config": vars(args),
        "promotion_validation": promotion_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "runtime_policy": runtime_policy,
        "smoke_decisions": decisions,
        "claim_guardrail": (
            "Stage 37 loads a promoted clarification-key policy manifest as "
            "a runtime snapshot. It is not a production deployment claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 37 clarification runtime loader failed: {pass_fail}")
    print("NeoCortex-X Stage 37 clarification runtime loader")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  runtime_policy_id: {runtime_policy['runtime_policy_id']}")
    print(f"  top_key: {runtime_policy['top_key']}")
    print(f"  unknown_context_fallback: {runtime_policy['unknown_context_fallback']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a promoted NeoCortex-X clarification runtime policy.")
    parser.add_argument(
        "--promotion-json",
        default="artifacts/neocortex_x/stage36_clarification_policy_promotion.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage37_clarification_runtime_loader.json",
    )
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
