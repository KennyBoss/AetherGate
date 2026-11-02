#!/usr/bin/env python3
"""Stage 41B guarded policy review for NeoCortex-X clarification feedback."""

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


def validate_stage40_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage40_clarification_monitor_feedback":
        errors.append("invalid stage40 feedback kind")
    if artifact.get("pass") is not True:
        errors.append("stage40 feedback artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    if not protocol.get("deterministic_replay"):
        errors.append("stage40 deterministic replay missing")
    key_rewards = artifact.get("key_rewards")
    if not isinstance(key_rewards, dict) or not key_rewards:
        errors.append("stage40 key_rewards missing")
        key_rewards = {}
    unknown_keys = set(key_rewards).difference(CONTEXT_KEYS)
    if unknown_keys:
        errors.append(f"stage40 key_rewards unknown keys {sorted(unknown_keys)}")
    events = artifact.get("feedback_events")
    if not isinstance(events, list) or not events:
        errors.append("stage40 feedback_events missing")
        events = []
    return {
        "valid": not errors,
        "errors": errors,
        "feedback_event_count": len(events),
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
        "key_rewards_present": bool(key_rewards),
    }


def validate_stage41a_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage41a_adversarial_risk_audit":
        errors.append("invalid stage41a audit kind")
    if artifact.get("pass") is not True:
        errors.append("stage41a adversarial risk audit artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    promotion_decision = artifact.get("promotion_decision", {})
    if not protocol.get("deterministic_replay"):
        errors.append("stage41a deterministic replay missing")
    if promotion_decision.get("promotion_allowed") is not False:
        errors.append("stage41a did not block promotion")
    if int(protocol.get("critical_blocking_case_count", 0)) <= 0:
        errors.append("stage41a critical blocking cases missing")
    if not promotion_decision.get("rollback_required"):
        errors.append("stage41a rollback requirement missing")
    return {
        "valid": not errors,
        "errors": errors,
        "case_count": int(protocol.get("case_count", 0)),
        "critical_blocking_case_count": int(protocol.get("critical_blocking_case_count", 0)),
        "promotion_allowed": bool(promotion_decision.get("promotion_allowed")),
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
        "rollback_required": bool(promotion_decision.get("rollback_required")),
    }


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        return {key: 1.0 / len(weights) for key in weights}
    return {key: max(0.0, value) / total for key, value in weights.items()}


def build_candidate_policy(
    stage40: dict[str, Any],
    *,
    base_weight: float,
    learning_rate: float,
) -> dict[str, Any]:
    before = {key: base_weight for key in CONTEXT_KEYS}
    after = dict(before)
    updates: list[dict[str, Any]] = []
    for key, reward_summary in sorted(stage40.get("key_rewards", {}).items()):
        if key not in CONTEXT_KEYS:
            continue
        total_reward = float(reward_summary.get("total_reward", 0.0))
        count = int(reward_summary.get("count", 0))
        delta = learning_rate * total_reward
        after[key] = max(0.0, after[key] + delta)
        updates.append(
            {
                "reinforce_key": key,
                "feedback_count": count,
                "total_reward": total_reward,
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
    basis = {
        "before": before,
        "after": after,
        "gains": gains,
        "probabilities": probabilities,
        "ranked_keys": ranked_keys,
        "updates": updates,
        "review_state": "quarantined_until_stage42_replay",
    }
    return {
        "candidate_policy_id": f"clarreview_{stable_hash(basis)}",
        "review_state": "quarantined_until_stage42_replay",
        "before_weights": before,
        "after_weights": after,
        "weight_gains": gains,
        "probabilities": probabilities,
        "top_key": ranked_keys[0],
        "ranked_keys": ranked_keys,
        "updates": updates,
        "replay_hash": stable_hash(basis, length=24),
    }


def build_guarded_decision(
    candidate_policy: dict[str, Any],
    stage41a_validation: dict[str, Any],
) -> dict[str, Any]:
    blocking = stage41a_validation["critical_blocking_case_count"]
    return {
        "decision": "quarantine_candidate_policy",
        "candidate_policy_id": candidate_policy["candidate_policy_id"],
        "candidate_policy_allowed_for_replay": True,
        "runtime_promotion_allowed": False,
        "blocked_by_stage41a": blocking > 0,
        "critical_blocking_case_count": blocking,
        "required_next_stage": "Stage 42 clean + adversarial replay",
        "rollback_required": True,
        "promotion_requirements": [
            "Stage 42 replay must compare current runtime, candidate policy, fixed-order control, and entropy control.",
            "No critical adversarial regressions may be introduced.",
            "Rollback metadata must be attached before any Stage 43 promotion.",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage40 = load_json(Path(args.feedback_json))
    stage41a = load_json(Path(args.risk_audit_json))
    stage40_validation = validate_stage40_artifact(stage40)
    stage41a_validation = validate_stage41a_artifact(stage41a)
    candidate_policy = build_candidate_policy(
        stage40,
        base_weight=args.base_weight,
        learning_rate=args.learning_rate,
    )
    replay_candidate_policy = build_candidate_policy(
        stage40,
        base_weight=args.base_weight,
        learning_rate=args.learning_rate,
    )
    deterministic_replay = candidate_policy == replay_candidate_policy
    guarded_decision = build_guarded_decision(candidate_policy, stage41a_validation)
    reinforced_keys = {
        update["reinforce_key"]
        for update in candidate_policy["updates"]
        if update["delta"] > 0.0
    }
    reinforced_gains = [candidate_policy["weight_gains"][key] for key in reinforced_keys]
    non_reinforced_keys = [key for key in CONTEXT_KEYS if key not in reinforced_keys]
    non_reinforced_gains = [candidate_policy["weight_gains"][key] for key in non_reinforced_keys]
    probability_sum = sum(candidate_policy["probabilities"].values())
    pass_fail = {
        "stage40_artifact_valid": stage40_validation["valid"],
        "stage41a_artifact_valid": stage41a_validation["valid"],
        "candidate_policy_present": bool(candidate_policy["updates"]),
        "candidate_policy_quarantined": candidate_policy["review_state"]
        == "quarantined_until_stage42_replay",
        "reinforced_keys_present": bool(reinforced_keys),
        "reinforced_key_gains_positive": all(gain > 0.0 for gain in reinforced_gains),
        "non_reinforced_keys_unchanged": all(abs(gain) <= 1e-12 for gain in non_reinforced_gains),
        "probability_sum": abs(probability_sum - 1.0) <= 1e-9,
        "runtime_promotion_blocked": guarded_decision["runtime_promotion_allowed"] is False,
        "candidate_allowed_for_stage42_replay": guarded_decision["candidate_policy_allowed_for_replay"],
        "blocked_by_stage41a": guarded_decision["blocked_by_stage41a"],
        "rollback_required": guarded_decision["rollback_required"],
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "feedback_json": args.feedback_json,
        "risk_audit_json": args.risk_audit_json,
        "candidate_policy_id": candidate_policy["candidate_policy_id"],
        "review_state": candidate_policy["review_state"],
        "ranked_keys": candidate_policy["ranked_keys"],
        "top_key": candidate_policy["top_key"],
        "runtime_promotion_allowed": guarded_decision["runtime_promotion_allowed"],
        "candidate_policy_allowed_for_replay": guarded_decision[
            "candidate_policy_allowed_for_replay"
        ],
        "critical_blocking_case_count": stage41a_validation["critical_blocking_case_count"],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage41b_guarded_policy_review",
        "config": vars(args),
        "stage40_validation": stage40_validation,
        "stage41a_validation": stage41a_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "mean_reinforced_gain": mean(reinforced_gains),
            "reinforced_key_count": len(reinforced_keys),
            "candidate_update_count": len(candidate_policy["updates"]),
            "probability_sum": probability_sum,
        },
        "candidate_policy": candidate_policy,
        "guarded_decision": guarded_decision,
        "claim_guardrail": (
            "Stage 41B creates a quarantined candidate policy for replay only. "
            "It does not promote runtime policy and does not claim live-world robustness."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 41B guarded policy review failed: {pass_fail}")
    print("NeoCortex-X Stage 41B guarded policy review")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  candidate_policy_id: {candidate_policy['candidate_policy_id']}")
    print(f"  review_state: {candidate_policy['review_state']}")
    print(f"  ranked_keys: {','.join(candidate_policy['ranked_keys'])}")
    print(f"  runtime_promotion_allowed: {guarded_decision['runtime_promotion_allowed']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 41B guarded policy review.")
    parser.add_argument(
        "--feedback-json",
        default="artifacts/neocortex_x/stage40_clarification_monitor_feedback.json",
    )
    parser.add_argument(
        "--risk-audit-json",
        default="artifacts/neocortex_x/stage41a_adversarial_risk_audit.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage41b_guarded_policy_review.json",
    )
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
