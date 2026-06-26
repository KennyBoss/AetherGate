#!/usr/bin/env python3
"""Stage 43 guarded promotion manifest for NeoCortex-X clarification policy."""

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


def validate_stage42_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage42_clean_adversarial_replay":
        errors.append("invalid stage42 replay kind")
    if artifact.get("pass") is not True:
        errors.append("stage42 artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    promotion = artifact.get("promotion_decision", {})
    aggregate = artifact.get("aggregate", {})
    agents = aggregate.get("agents", {})
    candidate = agents.get("candidate_policy", {})
    ranked_keys = candidate.get("ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("candidate ranked_keys missing")
        ranked_keys = []
    if set(ranked_keys) != set(CONTEXT_KEYS):
        errors.append("candidate ranked_keys do not cover CONTEXT_KEYS")
    if not protocol.get("deterministic_replay"):
        errors.append("stage42 deterministic replay missing")
    if promotion.get("promotion_allowed") is not False:
        errors.append("stage42 did not block unguarded promotion")
    if not promotion.get("critical_risks_remain"):
        errors.append("stage42 did not record critical risks before guard")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_policy_id": protocol.get("candidate_policy_id"),
        "clean_case_count": int(protocol.get("clean_case_count", 0)),
        "adversarial_case_count": int(protocol.get("adversarial_case_count", 0)),
        "ranked_keys": ranked_keys,
        "clean_resolution_accuracy": float(candidate.get("clean_resolution_accuracy", 0.0)),
        "clean_target_preservation_rate": float(
            candidate.get("clean_target_preservation_rate", 0.0)
        ),
        "safe_context_target_preservation_rate": float(
            candidate.get("safe_context_target_preservation_rate", 0.0)
        ),
        "critical_unsafe_answer_count_before_guard": int(
            candidate.get("critical_unsafe_answer_count", 0)
        ),
        "promotion_allowed": bool(promotion.get("promotion_allowed")),
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
    }


def validate_stage42c_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage42c_evidence_conflict_guard":
        errors.append("invalid stage42c guard kind")
    if artifact.get("pass") is not True:
        errors.append("stage42c artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    aggregate = artifact.get("aggregate", {})
    decision = artifact.get("decision", {})
    manifest = artifact.get("guard_manifest", {})
    rules = manifest.get("rules")
    if not protocol.get("deterministic_replay"):
        errors.append("stage42c deterministic replay missing")
    if protocol.get("runtime_promotion_allowed") is not False:
        errors.append("stage42c allowed runtime promotion")
    if protocol.get("candidate_guard_allowed_for_stage43") is not True:
        errors.append("stage42c did not allow stage43 review")
    if int(protocol.get("critical_unsafe_answer_count_after_guard", 1)) != 0:
        errors.append("stage42c critical unsafe answers remain")
    if float(aggregate.get("clean_answer_preservation_rate", 0.0)) != 1.0:
        errors.append("stage42c did not preserve clean answers")
    if float(aggregate.get("adversarial_safe_rate_after_guard", 0.0)) != 1.0:
        errors.append("stage42c adversarial safety after guard incomplete")
    if not isinstance(rules, dict) or not rules:
        errors.append("stage42c guard rules missing")
        rules = {}
    required_rules = {
        "same_context_contradiction",
        "noisy_context",
        "temporal_drift",
        "clean",
    }
    missing_rules = required_rules.difference(rules)
    if missing_rules:
        errors.append(f"stage42c guard rules missing {sorted(missing_rules)}")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_policy_id": protocol.get("candidate_policy_id"),
        "guard_id": protocol.get("guard_id"),
        "critical_unsafe_answer_count_before_guard": int(
            protocol.get("critical_unsafe_answer_count_before_guard", 0)
        ),
        "critical_unsafe_answer_count_after_guard": int(
            protocol.get("critical_unsafe_answer_count_after_guard", 0)
        ),
        "clean_answer_preservation_rate": float(
            aggregate.get("clean_answer_preservation_rate", 0.0)
        ),
        "adversarial_safe_rate_after_guard": float(
            aggregate.get("adversarial_safe_rate_after_guard", 0.0)
        ),
        "same_context_evidence_route_rate": float(
            aggregate.get("same_context_evidence_route_rate", 0.0)
        ),
        "noisy_context_integrity_route_rate": float(
            aggregate.get("noisy_context_integrity_route_rate", 0.0)
        ),
        "temporal_drift_validity_route_rate": float(
            aggregate.get("temporal_drift_validity_route_rate", 0.0)
        ),
        "runtime_promotion_allowed": bool(protocol.get("runtime_promotion_allowed")),
        "candidate_guard_allowed_for_stage43": bool(
            protocol.get("candidate_guard_allowed_for_stage43")
        ),
        "decision": decision.get("decision"),
        "rules": rules,
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
    }


def validate_guarded_review_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage41b_guarded_policy_review":
        errors.append("invalid stage41b review kind")
    if artifact.get("pass") is not True:
        errors.append("stage41b artifact did not pass")
    candidate = artifact.get("candidate_policy", {})
    decision = artifact.get("guarded_decision", {})
    probabilities = candidate.get("probabilities")
    ranked_keys = candidate.get("ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("stage41b ranked_keys missing")
        ranked_keys = []
    if set(ranked_keys) != set(CONTEXT_KEYS):
        errors.append("stage41b ranked_keys do not cover CONTEXT_KEYS")
    if not isinstance(probabilities, dict) or not probabilities:
        errors.append("stage41b probabilities missing")
        probabilities = {}
    probability_sum = sum(float(probabilities.get(key, 0.0)) for key in CONTEXT_KEYS)
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("stage41b probabilities do not sum to 1.0")
    if decision.get("runtime_promotion_allowed") is not False:
        errors.append("stage41b allowed runtime promotion")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_policy_id": candidate.get("candidate_policy_id"),
        "review_state": candidate.get("review_state"),
        "ranked_keys": ranked_keys,
        "top_key": candidate.get("top_key"),
        "probabilities": probabilities,
        "probability_sum": probability_sum,
        "runtime_promotion_allowed": bool(decision.get("runtime_promotion_allowed")),
    }


def build_manifest(
    *,
    args: argparse.Namespace,
    created_at: str,
    stage42_validation: dict[str, Any],
    stage42c_validation: dict[str, Any],
    review_validation: dict[str, Any],
) -> dict[str, Any]:
    runtime_contract = {
        "selection_rule": "ask first available missing key from guarded ranked_keys",
        "clean_answer_rule": "allow contextual answer when unique target is preserved",
        "same_context_contradiction_rule": stage42c_validation["rules"][
            "same_context_contradiction"
        ],
        "noisy_context_rule": stage42c_validation["rules"]["noisy_context"],
        "temporal_drift_rule": stage42c_validation["rules"]["temporal_drift"],
        "unknown_context_fallback": "ask_context_question",
        "context_dimensions": list(CONTEXT_KEYS),
    }
    rollback = {
        "previous_runtime_policy": "clarruntime_39694f57388ae602",
        "previous_manifest": "clarpromoted_b519c2257b02c3b7",
        "fallback_agent": "static_entropy_control",
        "fallback_guard": "block_answers_on_conflict_until_evidence_review",
        "restore_command": "make neocortex-x-stage42c-evidence-conflict-guard-smoke",
        "rollback_reason": (
            "Restore previous clarification runtime plus conflict guard if guarded "
            "promotion fails replay, monitor feedback, or live-source trust gates."
        ),
    }
    basis = {
        "candidate_policy_id": stage42_validation["candidate_policy_id"],
        "guard_id": stage42c_validation["guard_id"],
        "ranked_keys": review_validation["ranked_keys"],
        "probabilities": review_validation["probabilities"],
        "runtime_contract": runtime_contract,
        "rollback": rollback,
    }
    return {
        "manifest_id": f"clarguardedpromoted_{stable_hash(basis)}",
        "candidate_policy_id": stage42_validation["candidate_policy_id"],
        "guard_id": stage42c_validation["guard_id"],
        "runtime_ranked_keys": review_validation["ranked_keys"],
        "runtime_top_key": review_validation["top_key"],
        "runtime_probabilities": review_validation["probabilities"],
        "runtime_contract": runtime_contract,
        "promotion_evidence": {
            "stage41b_guarded_review_json": args.guarded_review_json,
            "stage42_replay_json": args.stage42_json,
            "stage42c_guard_json": args.stage42c_json,
            "clean_case_count": stage42_validation["clean_case_count"],
            "adversarial_case_count": stage42_validation["adversarial_case_count"],
            "clean_resolution_accuracy": stage42_validation["clean_resolution_accuracy"],
            "clean_target_preservation_rate": stage42_validation[
                "clean_target_preservation_rate"
            ],
            "safe_context_target_preservation_rate": stage42_validation[
                "safe_context_target_preservation_rate"
            ],
            "critical_unsafe_answer_count_before_guard": stage42c_validation[
                "critical_unsafe_answer_count_before_guard"
            ],
            "critical_unsafe_answer_count_after_guard": stage42c_validation[
                "critical_unsafe_answer_count_after_guard"
            ],
            "adversarial_safe_rate_after_guard": stage42c_validation[
                "adversarial_safe_rate_after_guard"
            ],
            "same_context_evidence_route_rate": stage42c_validation[
                "same_context_evidence_route_rate"
            ],
            "noisy_context_integrity_route_rate": stage42c_validation[
                "noisy_context_integrity_route_rate"
            ],
            "temporal_drift_validity_route_rate": stage42c_validation[
                "temporal_drift_validity_route_rate"
            ],
        },
        "promotion_scope": {
            "allowed": "guarded_cached_replay_runtime_candidate",
            "not_allowed": [
                "live_search_memory",
                "production_scale_memory",
                "unguarded_runtime_policy",
                "AGI_or_consciousness_claim",
                "neuromorphic_energy_claim",
            ],
        },
        "rollback": rollback,
        "created_at": created_at,
        "replay_hash": stable_hash(basis, length=24),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage42 = load_json(Path(args.stage42_json))
    stage42c = load_json(Path(args.stage42c_json))
    guarded_review = load_json(Path(args.guarded_review_json))
    stage42_validation = validate_stage42_artifact(stage42)
    stage42c_validation = validate_stage42c_artifact(stage42c)
    review_validation = validate_guarded_review_artifact(guarded_review)
    candidate_ids_match = (
        stage42_validation["candidate_policy_id"]
        == stage42c_validation["candidate_policy_id"]
        == review_validation["candidate_policy_id"]
    )
    ranked_keys_match = stage42_validation["ranked_keys"] == review_validation["ranked_keys"]
    created_at = timestamp()
    manifest = build_manifest(
        args=args,
        created_at=created_at,
        stage42_validation=stage42_validation,
        stage42c_validation=stage42c_validation,
        review_validation=review_validation,
    )
    replay_manifest = build_manifest(
        args=args,
        created_at=created_at,
        stage42_validation=stage42_validation,
        stage42c_validation=stage42c_validation,
        review_validation=review_validation,
    )
    deterministic_replay = manifest == replay_manifest
    pass_fail = {
        "stage42_artifact_valid": stage42_validation["valid"],
        "stage42c_artifact_valid": stage42c_validation["valid"],
        "stage41b_artifact_valid": review_validation["valid"],
        "candidate_ids_match": candidate_ids_match,
        "ranked_keys_match": ranked_keys_match,
        "stage42_blocked_unguarded_promotion": (
            stage42_validation["promotion_allowed"] is False
        ),
        "stage42c_allowed_stage43_review": stage42c_validation[
            "candidate_guard_allowed_for_stage43"
        ],
        "no_critical_unsafe_answers_after_guard": (
            stage42c_validation["critical_unsafe_answer_count_after_guard"] == 0
        ),
        "clean_answers_preserved": (
            stage42c_validation["clean_answer_preservation_rate"] == 1.0
        ),
        "adversarial_safe_after_guard": (
            stage42c_validation["adversarial_safe_rate_after_guard"] == 1.0
        ),
        "guard_rules_present": bool(manifest["runtime_contract"]),
        "rollback_metadata_present": bool(manifest["rollback"]["restore_command"]),
        "promotion_scope_limited": (
            manifest["promotion_scope"]["allowed"] == "guarded_cached_replay_runtime_candidate"
        ),
        "claim_guardrails_present": bool(manifest["promotion_scope"]["not_allowed"]),
        "deterministic_replay": deterministic_replay,
    }
    promotion_decision = {
        "decision": "promote_guarded_runtime_candidate",
        "promotion_allowed": all(pass_fail.values()),
        "manifest_id": manifest["manifest_id"],
        "candidate_policy_id": manifest["candidate_policy_id"],
        "guard_id": manifest["guard_id"],
        "runtime_promotion_scope": manifest["promotion_scope"]["allowed"],
        "rollback_required": True,
        "required_next_stage": "Stage 44 guarded runtime loader",
        "reason": (
            "Stage 43 promotes only a guarded cached-replay runtime candidate. "
            "The guard manifest cleared critical unsafe answers and rollback "
            "metadata is attached; live-source trust and production-scale memory "
            "remain separate gates."
        ),
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "stage42_json": args.stage42_json,
        "stage42c_json": args.stage42c_json,
        "guarded_review_json": args.guarded_review_json,
        "manifest_id": manifest["manifest_id"],
        "candidate_policy_id": manifest["candidate_policy_id"],
        "guard_id": manifest["guard_id"],
        "promotion_allowed": promotion_decision["promotion_allowed"],
        "runtime_promotion_scope": promotion_decision["runtime_promotion_scope"],
        "rollback_present": bool(manifest["rollback"]["restore_command"]),
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": created_at,
        "kind": "neocortex_x_stage43_guarded_policy_promotion",
        "config": vars(args),
        "stage42_validation": stage42_validation,
        "stage42c_validation": stage42c_validation,
        "stage41b_validation": review_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "manifest": manifest,
        "promotion_decision": promotion_decision,
        "claim_guardrail": (
            "Stage 43 promotes only a guarded cached-replay runtime candidate "
            "with rollback metadata. It is not a live-source trust claim, not "
            "production-scale memory, not AGI, and not consciousness evidence."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 43 guarded policy promotion failed: {pass_fail}")
    print("NeoCortex-X Stage 43 guarded policy promotion")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  manifest_id: {manifest['manifest_id']}")
    print(f"  candidate_policy_id: {manifest['candidate_policy_id']}")
    print(f"  guard_id: {manifest['guard_id']}")
    print(f"  promotion_allowed: {promotion_decision['promotion_allowed']}")
    print(f"  runtime_promotion_scope: {promotion_decision['runtime_promotion_scope']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeoCortex-X Stage 43 guarded policy promotion."
    )
    parser.add_argument(
        "--stage42-json",
        default="artifacts/neocortex_x/stage42_clean_adversarial_replay.json",
    )
    parser.add_argument(
        "--stage42c-json",
        default="artifacts/neocortex_x/stage42c_evidence_conflict_guard.json",
    )
    parser.add_argument(
        "--guarded-review-json",
        default="artifacts/neocortex_x/stage41b_guarded_policy_review.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage43_guarded_policy_promotion.json",
    )
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
