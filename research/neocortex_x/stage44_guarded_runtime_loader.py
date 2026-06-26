#!/usr/bin/env python3
"""Stage 44 guarded runtime loader for NeoCortex-X clarification policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS


REQUIRED_NOT_ALLOWED = {
    "live_search_memory",
    "production_scale_memory",
    "unguarded_runtime_policy",
    "AGI_or_consciousness_claim",
    "neuromorphic_energy_claim",
}

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


def validate_stage43_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage43_guarded_policy_promotion":
        errors.append("invalid stage43 promotion kind")
    if artifact.get("pass") is not True:
        errors.append("stage43 artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    decision = artifact.get("promotion_decision", {})
    manifest = artifact.get("manifest", {})
    ranked_keys = manifest.get("runtime_ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("runtime_ranked_keys missing")
        ranked_keys = []
    if set(ranked_keys) != set(CONTEXT_KEYS):
        errors.append("runtime_ranked_keys do not cover CONTEXT_KEYS")
    top_key = manifest.get("runtime_top_key")
    if top_key not in set(ranked_keys):
        errors.append("runtime_top_key missing from ranked_keys")
    probabilities = manifest.get("runtime_probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        errors.append("runtime_probabilities missing")
        probabilities = {}
    probability_sum = sum(float(probabilities.get(key, 0.0)) for key in CONTEXT_KEYS)
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("runtime_probabilities do not sum to 1.0")
    runtime_contract = manifest.get("runtime_contract", {})
    required_contract = {
        "selection_rule",
        "clean_answer_rule",
        "same_context_contradiction_rule",
        "noisy_context_rule",
        "temporal_drift_rule",
        "unknown_context_fallback",
        "context_dimensions",
    }
    missing_contract = required_contract.difference(runtime_contract)
    if missing_contract:
        errors.append(f"runtime_contract missing {sorted(missing_contract)}")
    if runtime_contract.get("unknown_context_fallback") != UNKNOWN_CONTEXT_FALLBACK:
        errors.append("unknown_context_fallback mismatch")
    if set(runtime_contract.get("context_dimensions", [])) != set(CONTEXT_KEYS):
        errors.append("runtime contract context_dimensions mismatch")
    scope = manifest.get("promotion_scope", {})
    if scope.get("allowed") != "guarded_cached_replay_runtime_candidate":
        errors.append("promotion scope is not guarded cached replay")
    not_allowed = set(scope.get("not_allowed", []))
    if not REQUIRED_NOT_ALLOWED.issubset(not_allowed):
        errors.append("claim guardrails missing required forbidden claims")
    rollback = manifest.get("rollback", {})
    if not rollback.get("restore_command") or not rollback.get("previous_runtime_policy"):
        errors.append("rollback metadata incomplete")
    evidence = manifest.get("promotion_evidence", {})
    if int(evidence.get("critical_unsafe_answer_count_after_guard", 1)) != 0:
        errors.append("critical unsafe answers remain in promotion evidence")
    if float(evidence.get("adversarial_safe_rate_after_guard", 0.0)) != 1.0:
        errors.append("adversarial safe rate after guard is not 1.0")
    if protocol.get("deterministic_replay") is not True:
        errors.append("stage43 deterministic replay missing")
    if decision.get("promotion_allowed") is not True:
        errors.append("stage43 promotion was not allowed")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest_id": manifest.get("manifest_id"),
        "candidate_policy_id": manifest.get("candidate_policy_id"),
        "guard_id": manifest.get("guard_id"),
        "ranked_keys": ranked_keys,
        "top_key": top_key,
        "probabilities": probabilities,
        "probability_sum": probability_sum,
        "runtime_contract": runtime_contract,
        "promotion_scope": scope,
        "rollback": rollback,
        "critical_unsafe_answer_count_after_guard": int(
            evidence.get("critical_unsafe_answer_count_after_guard", 0)
        ),
        "adversarial_safe_rate_after_guard": float(
            evidence.get("adversarial_safe_rate_after_guard", 0.0)
        ),
        "promotion_allowed": bool(decision.get("promotion_allowed")),
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
    }


def build_runtime_snapshot(manifest: dict[str, Any]) -> dict[str, Any]:
    basis = {
        "manifest_id": manifest["manifest_id"],
        "candidate_policy_id": manifest["candidate_policy_id"],
        "guard_id": manifest["guard_id"],
        "ranked_keys": manifest["runtime_ranked_keys"],
        "runtime_contract": manifest["runtime_contract"],
        "promotion_scope": manifest["promotion_scope"],
    }
    return {
        "runtime_policy_id": f"clarguardedruntime_{stable_hash(basis)}",
        "source_manifest_id": manifest["manifest_id"],
        "candidate_policy_id": manifest["candidate_policy_id"],
        "guard_id": manifest["guard_id"],
        "ranked_keys": manifest["runtime_ranked_keys"],
        "top_key": manifest["runtime_top_key"],
        "probabilities": manifest["runtime_probabilities"],
        "runtime_contract": manifest["runtime_contract"],
        "promotion_scope": manifest["promotion_scope"],
        "rollback": manifest["rollback"],
        "guard_active": True,
        "claim_guardrails_active": True,
        "replay_hash": stable_hash(basis, length=24),
    }


def select_runtime_key(runtime_snapshot: dict[str, Any], partial_context: dict[str, str]) -> str | None:
    for key in runtime_snapshot["ranked_keys"]:
        if key not in partial_context:
            return key
    return None


def guarded_action(runtime_snapshot: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    contract = runtime_snapshot["runtime_contract"]
    risk_type = case["risk_type"]
    partial_context = case["partial_context"]
    unknown_shape = any(key not in CONTEXT_KEYS for key in partial_context)
    if unknown_shape:
        selected_key = select_runtime_key(runtime_snapshot, {})
        return {
            "selected_action": contract["unknown_context_fallback"],
            "selected_key": selected_key,
            "answer_allowed": False,
            "guard_applied": True,
            "unknown_shape": True,
        }
    if risk_type == "clean_unique":
        return {
            "selected_action": "allow_contextual_answer",
            "selected_key": None,
            "answer_allowed": True,
            "guard_applied": True,
            "unknown_shape": False,
        }
    if risk_type == "same_context_contradiction":
        return {
            "selected_action": contract["same_context_contradiction_rule"],
            "selected_key": None,
            "answer_allowed": False,
            "guard_applied": True,
            "unknown_shape": False,
        }
    if risk_type == "noisy_context":
        return {
            "selected_action": contract["noisy_context_rule"],
            "selected_key": None,
            "answer_allowed": False,
            "guard_applied": True,
            "unknown_shape": False,
        }
    if risk_type == "temporal_drift":
        return {
            "selected_action": contract["temporal_drift_rule"],
            "selected_key": None,
            "answer_allowed": False,
            "guard_applied": True,
            "unknown_shape": False,
        }
    selected_key = select_runtime_key(runtime_snapshot, partial_context)
    selected_action = (
        "allow_contextual_answer"
        if selected_key is None
        else "ask_disambiguating_context"
    )
    return {
        "selected_action": selected_action,
        "selected_key": selected_key,
        "answer_allowed": selected_action == "allow_contextual_answer",
        "guard_applied": True,
        "unknown_shape": False,
    }


def smoke_cases(runtime_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    ranked_keys = runtime_snapshot["ranked_keys"]
    return [
        {
            "case_id": "missing_context_selects_top_key",
            "risk_type": "missing_context",
            "partial_context": {},
            "expected_action": "ask_disambiguating_context",
            "expected_key": ranked_keys[0],
            "expected_answer_allowed": False,
        },
        {
            "case_id": "clean_unique_answer_allowed",
            "risk_type": "clean_unique",
            "partial_context": {key: "known" for key in ranked_keys},
            "expected_action": "allow_contextual_answer",
            "expected_key": None,
            "expected_answer_allowed": True,
        },
        {
            "case_id": "same_context_contradiction_guarded",
            "risk_type": "same_context_contradiction",
            "partial_context": {key: "known" for key in ranked_keys},
            "expected_action": "request_evidence_on_value_conflict",
            "expected_key": None,
            "expected_answer_allowed": False,
        },
        {
            "case_id": "noisy_context_guarded",
            "risk_type": "noisy_context",
            "partial_context": {ranked_keys[0]: "corrupted"},
            "expected_action": "request_context_integrity_check",
            "expected_key": None,
            "expected_answer_allowed": False,
        },
        {
            "case_id": "temporal_drift_guarded",
            "risk_type": "temporal_drift",
            "partial_context": {key: "known" for key in ranked_keys},
            "expected_action": "request_validity_window_check",
            "expected_key": None,
            "expected_answer_allowed": False,
        },
        {
            "case_id": "unknown_context_shape_fallback",
            "risk_type": "unknown_shape",
            "partial_context": {"unknown_context_key": "value"},
            "expected_action": UNKNOWN_CONTEXT_FALLBACK,
            "expected_key": ranked_keys[0],
            "expected_answer_allowed": False,
        },
    ]


def smoke_decisions(runtime_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = []
    for case in smoke_cases(runtime_snapshot):
        result = guarded_action(runtime_snapshot, case)
        decisions.append(
            {
                "case_id": case["case_id"],
                "risk_type": case["risk_type"],
                "partial_context": case["partial_context"],
                "selected_action": result["selected_action"],
                "expected_action": case["expected_action"],
                "selected_key": result["selected_key"],
                "expected_key": case["expected_key"],
                "answer_allowed": result["answer_allowed"],
                "expected_answer_allowed": case["expected_answer_allowed"],
                "guard_applied": result["guard_applied"],
                "unknown_shape": result["unknown_shape"],
                "action_correct": result["selected_action"] == case["expected_action"],
                "key_correct": result["selected_key"] == case["expected_key"],
                "answer_permission_correct": (
                    result["answer_allowed"] == case["expected_answer_allowed"]
                ),
            }
        )
    return decisions


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_json(Path(args.promotion_json))
    stage43_validation = validate_stage43_artifact(artifact)
    manifest = artifact.get("manifest", {})
    runtime_snapshot = build_runtime_snapshot(manifest)
    decisions = smoke_decisions(runtime_snapshot)
    replay_decisions = smoke_decisions(runtime_snapshot)
    deterministic_replay = decisions == replay_decisions
    actions_correct = all(decision["action_correct"] for decision in decisions)
    keys_correct = all(decision["key_correct"] for decision in decisions)
    answer_permissions_correct = all(
        decision["answer_permission_correct"] for decision in decisions
    )
    guard_applied = all(decision["guard_applied"] for decision in decisions)
    conflict_answers_blocked = all(
        not decision["answer_allowed"]
        for decision in decisions
        if decision["risk_type"]
        in {"same_context_contradiction", "noisy_context", "temporal_drift"}
    )
    clean_answer_allowed = any(
        decision["case_id"] == "clean_unique_answer_allowed"
        and decision["answer_allowed"]
        and decision["selected_action"] == "allow_contextual_answer"
        for decision in decisions
    )
    claim_guardrails_active = runtime_snapshot["claim_guardrails_active"] and REQUIRED_NOT_ALLOWED.issubset(
        set(runtime_snapshot["promotion_scope"].get("not_allowed", []))
    )
    pass_fail = {
        "stage43_artifact_valid": stage43_validation["valid"],
        "promotion_allowed": stage43_validation["promotion_allowed"],
        "guard_active": runtime_snapshot["guard_active"],
        "claim_guardrails_active": claim_guardrails_active,
        "rollback_present": bool(runtime_snapshot["rollback"].get("restore_command")),
        "runtime_scope_limited": (
            runtime_snapshot["promotion_scope"].get("allowed")
            == "guarded_cached_replay_runtime_candidate"
        ),
        "smoke_actions_correct": actions_correct,
        "smoke_keys_correct": keys_correct,
        "answer_permissions_correct": answer_permissions_correct,
        "guard_applied_to_all_smoke_cases": guard_applied,
        "conflict_answers_blocked": conflict_answers_blocked,
        "clean_answer_allowed": clean_answer_allowed,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "promotion_json": args.promotion_json,
        "manifest_id": stage43_validation["manifest_id"],
        "runtime_policy_id": runtime_snapshot["runtime_policy_id"],
        "candidate_policy_id": runtime_snapshot["candidate_policy_id"],
        "guard_id": runtime_snapshot["guard_id"],
        "runtime_scope": runtime_snapshot["promotion_scope"].get("allowed"),
        "guard_active": runtime_snapshot["guard_active"],
        "rollback_present": pass_fail["rollback_present"],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage44_guarded_runtime_loader",
        "config": vars(args),
        "stage43_validation": stage43_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "runtime_snapshot": runtime_snapshot,
        "smoke_decisions": decisions,
        "claim_guardrail": (
            "Stage 44 loads a guarded cached-replay runtime snapshot. It keeps "
            "the evidence/conflict guard and claim guardrails active. It is not "
            "live-source trust, production memory, AGI, or consciousness evidence."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 44 guarded runtime loader failed: {pass_fail}")
    print("NeoCortex-X Stage 44 guarded runtime loader")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  runtime_policy_id: {runtime_snapshot['runtime_policy_id']}")
    print(f"  manifest_id: {runtime_snapshot['source_manifest_id']}")
    print(f"  guard_id: {runtime_snapshot['guard_id']}")
    print(f"  runtime_scope: {protocol_evidence['runtime_scope']}")
    print(f"  guard_active: {runtime_snapshot['guard_active']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a NeoCortex-X guarded clarification runtime snapshot."
    )
    parser.add_argument(
        "--promotion-json",
        default="artifacts/neocortex_x/stage43_guarded_policy_promotion.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage44_guarded_runtime_loader.json",
    )
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
