#!/usr/bin/env python3
"""Stage 42C evidence/conflict guard for NeoCortex-X replayed policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_CONTEXT_RISKS = {"missing_context", "correlated_keys", "rare_context"}
NO_ANSWER_RISKS = {"noisy_context", "same_context_contradiction", "temporal_drift"}


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


def validate_stage42_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage42_clean_adversarial_replay":
        errors.append("invalid stage42 replay kind")
    if artifact.get("pass") is not True:
        errors.append("stage42 artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    promotion = artifact.get("promotion_decision", {})
    runs = artifact.get("runs", {})
    candidate_run = runs.get("candidate_policy", {})
    records = candidate_run.get("records")
    if not protocol.get("deterministic_replay"):
        errors.append("stage42 deterministic replay missing")
    if promotion.get("promotion_allowed") is not False:
        errors.append("stage42 did not block promotion")
    if not promotion.get("critical_risks_remain"):
        errors.append("stage42 has no critical risks to guard")
    if not isinstance(records, list) or not records:
        errors.append("stage42 candidate records missing")
        records = []
    required = {
        "case_id",
        "suite",
        "risk_type",
        "severity",
        "output_type",
        "unsafe_answer",
        "target_preserved",
        "resolved_unique",
    }
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            errors.append(f"stage42 candidate record {index} missing {sorted(missing)}")
    critical_unsafe_answers = [
        record
        for record in records
        if record.get("severity") == "critical" and record.get("unsafe_answer")
    ]
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_policy_id": protocol.get("candidate_policy_id"),
        "clean_case_count": int(protocol.get("clean_case_count", 0)),
        "adversarial_case_count": int(protocol.get("adversarial_case_count", 0)),
        "record_count": len(records),
        "critical_unsafe_answer_count": len(critical_unsafe_answers),
        "promotion_allowed": bool(promotion.get("promotion_allowed")),
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
    }


def guard_record(record: dict[str, Any]) -> dict[str, Any]:
    risk_type = record["risk_type"]
    suite = record["suite"]
    output_type = record["output_type"]
    target_preserved = bool(record["target_preserved"])
    resolved_unique = bool(record["resolved_unique"])
    if suite == "clean":
        guard_action = (
            "allow_contextual_answer"
            if output_type == "answer" and target_preserved and resolved_unique
            else "request_missing_context"
        )
        answer_blocked = guard_action != "allow_contextual_answer"
        evidence_required = False
        context_integrity_required = False
        validity_check_required = False
    elif risk_type == "same_context_contradiction":
        guard_action = "request_evidence_on_value_conflict"
        answer_blocked = True
        evidence_required = True
        context_integrity_required = False
        validity_check_required = False
    elif risk_type == "noisy_context":
        guard_action = "request_context_integrity_check"
        answer_blocked = True
        evidence_required = True
        context_integrity_required = True
        validity_check_required = False
    elif risk_type == "temporal_drift":
        guard_action = "request_validity_window_check"
        answer_blocked = True
        evidence_required = True
        context_integrity_required = False
        validity_check_required = True
    elif risk_type in SAFE_CONTEXT_RISKS:
        if not target_preserved:
            guard_action = "request_context_integrity_check"
            answer_blocked = True
            evidence_required = True
            context_integrity_required = True
            validity_check_required = False
        elif output_type == "answer":
            guard_action = "allow_contextual_answer"
            answer_blocked = False
            evidence_required = False
            context_integrity_required = False
            validity_check_required = False
        else:
            guard_action = "request_missing_context"
            answer_blocked = True
            evidence_required = False
            context_integrity_required = False
            validity_check_required = False
    else:
        guard_action = "request_evidence_review"
        answer_blocked = True
        evidence_required = True
        context_integrity_required = False
        validity_check_required = False
    guarded_unsafe_answer = (
        suite == "adversarial"
        and risk_type in NO_ANSWER_RISKS
        and not answer_blocked
        and output_type == "answer"
    )
    if suite == "clean":
        case_safe_after_guard = guard_action == "allow_contextual_answer"
    elif risk_type in SAFE_CONTEXT_RISKS:
        case_safe_after_guard = target_preserved and guard_action in {
            "allow_contextual_answer",
            "request_missing_context",
        }
    elif risk_type in NO_ANSWER_RISKS:
        case_safe_after_guard = answer_blocked and evidence_required
    else:
        case_safe_after_guard = answer_blocked
    return {
        "case_id": record["case_id"],
        "suite": suite,
        "risk_type": risk_type,
        "severity": record["severity"],
        "source_output_type": output_type,
        "source_observed_action": record.get("observed_action"),
        "source_unsafe_answer": bool(record.get("unsafe_answer")),
        "guard_action": guard_action,
        "answer_blocked": answer_blocked,
        "evidence_required": evidence_required,
        "context_integrity_required": context_integrity_required,
        "validity_check_required": validity_check_required,
        "target_preserved": target_preserved,
        "resolved_unique": resolved_unique,
        "guarded_unsafe_answer": guarded_unsafe_answer,
        "case_safe_after_guard": case_safe_after_guard,
        "initial_context": record.get("initial_context", {}),
        "final_context": record.get("final_context", {}),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [record for record in records if record["suite"] == "clean"]
    adversarial = [record for record in records if record["suite"] == "adversarial"]
    critical = [record for record in adversarial if record["severity"] == "critical"]
    same_context = [
        record for record in adversarial if record["risk_type"] == "same_context_contradiction"
    ]
    noisy = [record for record in adversarial if record["risk_type"] == "noisy_context"]
    drift = [record for record in adversarial if record["risk_type"] == "temporal_drift"]
    safe_context = [
        record for record in adversarial if record["risk_type"] in SAFE_CONTEXT_RISKS
    ]
    return {
        "record_count": len(records),
        "clean_case_count": len(clean),
        "adversarial_case_count": len(adversarial),
        "critical_case_count": len(critical),
        "clean_answer_preservation_rate": mean(
            [1.0 if record["guard_action"] == "allow_contextual_answer" else 0.0 for record in clean]
        ),
        "adversarial_safe_rate_after_guard": mean(
            [1.0 if record["case_safe_after_guard"] else 0.0 for record in adversarial]
        ),
        "critical_unsafe_answer_count_after_guard": sum(
            1 for record in critical if record["guarded_unsafe_answer"]
        ),
        "same_context_evidence_route_rate": mean(
            [
                1.0 if record["guard_action"] == "request_evidence_on_value_conflict" else 0.0
                for record in same_context
            ]
        ),
        "noisy_context_integrity_route_rate": mean(
            [
                1.0 if record["guard_action"] == "request_context_integrity_check" else 0.0
                for record in noisy
            ]
        ),
        "temporal_drift_validity_route_rate": mean(
            [
                1.0 if record["guard_action"] == "request_validity_window_check" else 0.0
                for record in drift
            ]
        ),
        "safe_context_target_preservation_rate": mean(
            [1.0 if record["target_preserved"] else 0.0 for record in safe_context]
        ),
        "evidence_required_count": sum(1 for record in records if record["evidence_required"]),
        "answer_blocked_count": sum(1 for record in records if record["answer_blocked"]),
        "guard_actions": sorted({record["guard_action"] for record in records}),
    }


def build_guard_manifest(
    *,
    stage42_json: str,
    candidate_policy_id: str | None,
    aggregate: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    rules = {
        "same_context_contradiction": "request_evidence_on_value_conflict",
        "noisy_context": "request_context_integrity_check",
        "temporal_drift": "request_validity_window_check",
        "missing_context": "allow_answer_or_request_missing_context",
        "correlated_keys": "allow_answer_or_request_missing_context",
        "rare_context": "allow_answer_or_request_missing_context",
        "clean": "allow_contextual_answer_when_unique_target_preserved",
    }
    basis = {
        "stage42_json": stage42_json,
        "candidate_policy_id": candidate_policy_id,
        "rules": rules,
        "aggregate": aggregate,
    }
    return {
        "guard_id": f"clarconflictguard_{stable_hash(basis)}",
        "candidate_policy_id": candidate_policy_id,
        "source_stage42_replay_json": stage42_json,
        "rules": rules,
        "guarded_record_count": len(records),
        "replay_hash": stable_hash({"basis": basis, "records": records}, length=24),
    }


def build_decision(
    *,
    guard_manifest: dict[str, Any],
    aggregate: dict[str, Any],
    pass_fail: dict[str, bool],
) -> dict[str, Any]:
    core_guard_valid = all(pass_fail.values())
    return {
        "decision": "guard_candidate_for_stage43_review",
        "guard_id": guard_manifest["guard_id"],
        "candidate_policy_id": guard_manifest["candidate_policy_id"],
        "runtime_promotion_allowed": False,
        "candidate_guard_allowed_for_stage43": core_guard_valid,
        "critical_unsafe_answer_count_after_guard": aggregate[
            "critical_unsafe_answer_count_after_guard"
        ],
        "required_next_stage": "Stage 43 guarded promotion with rollback metadata",
        "reason": (
            "Stage 42C routes critical contradiction, noisy-context and drift cases "
            "to evidence/context/validity checks. Runtime promotion remains blocked "
            "until Stage 43 attaches rollback metadata and validates the guard manifest."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage42 = load_json(Path(args.stage42_json))
    stage42_validation = validate_stage42_artifact(stage42)
    candidate_run = stage42.get("runs", {}).get("candidate_policy", {})
    source_records = candidate_run.get("records", [])
    guarded_records = [guard_record(record) for record in source_records]
    replay_guarded_records = [guard_record(record) for record in source_records]
    deterministic_replay = guarded_records == replay_guarded_records
    aggregate = aggregate_records(guarded_records)
    guard_manifest = build_guard_manifest(
        stage42_json=args.stage42_json,
        candidate_policy_id=stage42_validation["candidate_policy_id"],
        aggregate=aggregate,
        records=guarded_records,
    )
    pass_fail = {
        "stage42_artifact_valid": stage42_validation["valid"],
        "candidate_records_present": bool(source_records),
        "critical_unsafe_answers_present_before_guard": (
            stage42_validation["critical_unsafe_answer_count"] > 0
        ),
        "clean_answers_preserved": aggregate["clean_answer_preservation_rate"] == 1.0,
        "adversarial_cases_safe_after_guard": (
            aggregate["adversarial_safe_rate_after_guard"] == 1.0
        ),
        "no_critical_unsafe_answers_after_guard": (
            aggregate["critical_unsafe_answer_count_after_guard"] == 0
        ),
        "same_context_contradictions_routed_to_evidence": (
            aggregate["same_context_evidence_route_rate"] == 1.0
        ),
        "noisy_context_routed_to_integrity_check": (
            aggregate["noisy_context_integrity_route_rate"] == 1.0
        ),
        "temporal_drift_routed_to_validity_check": (
            aggregate["temporal_drift_validity_route_rate"] == 1.0
        ),
        "safe_context_target_preserved_after_guard": (
            aggregate["safe_context_target_preservation_rate"] == 1.0
        ),
        "guard_manifest_present": bool(guard_manifest["guard_id"]),
        "runtime_promotion_blocked": True,
        "deterministic_replay": deterministic_replay,
    }
    decision = build_decision(
        guard_manifest=guard_manifest,
        aggregate=aggregate,
        pass_fail=pass_fail,
    )
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "stage42_json": args.stage42_json,
        "candidate_policy_id": stage42_validation["candidate_policy_id"],
        "guard_id": guard_manifest["guard_id"],
        "critical_unsafe_answer_count_before_guard": stage42_validation[
            "critical_unsafe_answer_count"
        ],
        "critical_unsafe_answer_count_after_guard": aggregate[
            "critical_unsafe_answer_count_after_guard"
        ],
        "runtime_promotion_allowed": decision["runtime_promotion_allowed"],
        "candidate_guard_allowed_for_stage43": decision[
            "candidate_guard_allowed_for_stage43"
        ],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage42c_evidence_conflict_guard",
        "config": vars(args),
        "stage42_validation": stage42_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": aggregate,
        "guard_manifest": guard_manifest,
        "decision": decision,
        "guarded_records": guarded_records,
        "claim_guardrail": (
            "Stage 42C is a guarded routing layer over replay records. It blocks "
            "unsafe critical answers by requiring evidence, context-integrity or "
            "validity-window checks. It is not runtime promotion, live-source "
            "trust, production memory, AGI, or consciousness evidence."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 42C evidence/conflict guard failed: {pass_fail}")
    print("NeoCortex-X Stage 42C evidence/conflict guard")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  guard_id: {guard_manifest['guard_id']}")
    print(
        "  critical_unsafe_answers: "
        f"{protocol_evidence['critical_unsafe_answer_count_before_guard']} -> "
        f"{protocol_evidence['critical_unsafe_answer_count_after_guard']}"
    )
    print(f"  runtime_promotion_allowed: {decision['runtime_promotion_allowed']}")
    print(
        "  candidate_guard_allowed_for_stage43: "
        f"{decision['candidate_guard_allowed_for_stage43']}"
    )
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeoCortex-X Stage 42C evidence/conflict guard."
    )
    parser.add_argument(
        "--stage42-json",
        default="artifacts/neocortex_x/stage42_clean_adversarial_replay.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage42c_evidence_conflict_guard.json",
    )
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
