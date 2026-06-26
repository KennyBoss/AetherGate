#!/usr/bin/env python3
"""Stage 41A adversarial risk audit for NeoCortex-X clarification policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS
from stage38_clarification_runtime_execution import (
    candidate_pairs,
    filter_candidates,
    output_for_result,
    select_runtime_key,
    target_present,
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


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def validate_stage30_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage30_partial_context_scale_retrieval":
        errors.append("invalid stage30 kind")
    if artifact.get("pass") is not True:
        errors.append("stage30 artifact did not pass")
    queries = artifact.get("probe", {}).get("evaluated_queries")
    if not isinstance(queries, list) or not queries:
        errors.append("stage30 evaluated_queries missing")
        queries = []
    required = {
        "query_id",
        "target_slot_id",
        "subject",
        "property",
        "partial_context",
        "full_context",
        "target_value",
        "candidate_count",
        "candidate_ids",
        "candidate_contexts",
        "target_in_candidates",
    }
    for index, query in enumerate(queries):
        missing = required.difference(query)
        if missing:
            errors.append(f"stage30 query {index} missing {sorted(missing)}")
            continue
        if len(query["candidate_ids"]) != len(query["candidate_contexts"]):
            errors.append(f"stage30 query {index} candidate ids/contexts length mismatch")
        if not query.get("target_in_candidates"):
            errors.append(f"stage30 query {index} target missing from candidates")
    return {"valid": not errors, "errors": errors, "query_count": len(queries)}


def validate_runtime_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage37_clarification_runtime_loader":
        errors.append("invalid stage37 runtime kind")
    if artifact.get("pass") is not True:
        errors.append("stage37 runtime artifact did not pass")
    runtime_policy = artifact.get("runtime_policy", {})
    ranked_keys = runtime_policy.get("ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("runtime ranked_keys missing")
        ranked_keys = []
    if set(ranked_keys) != set(CONTEXT_KEYS):
        errors.append("runtime ranked_keys do not cover CONTEXT_KEYS")
    rollback = runtime_policy.get("rollback", {})
    if not rollback.get("restore_command"):
        errors.append("runtime rollback restore_command missing")
    return {
        "valid": not errors,
        "errors": errors,
        "runtime_policy_id": runtime_policy.get("runtime_policy_id"),
        "policy_id": runtime_policy.get("policy_id"),
        "ranked_keys": ranked_keys,
        "rollback_present": bool(rollback.get("restore_command")),
    }


def validate_stage40_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage40_clarification_monitor_feedback":
        errors.append("invalid stage40 feedback kind")
    if artifact.get("pass") is not True:
        errors.append("stage40 feedback artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    alerts = artifact.get("alert_summary", {})
    feedback_events = artifact.get("feedback_events")
    if not protocol.get("deterministic_replay"):
        errors.append("stage40 deterministic replay missing")
    if not isinstance(feedback_events, list) or not feedback_events:
        errors.append("stage40 feedback_events missing")
        feedback_events = []
    if int(alerts.get("critical_alert_count", 0)) != 0:
        errors.append("stage40 already has critical alerts")
    return {
        "valid": not errors,
        "errors": errors,
        "feedback_event_count": len(feedback_events),
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
        "critical_alert_count": int(alerts.get("critical_alert_count", 0)),
        "info_alert_count": int(alerts.get("info_alert_count", 0)),
    }


def alternate_context_value(
    query: dict[str, Any],
    key: str,
    *,
    avoid_value: str,
) -> str:
    for context in query["candidate_contexts"]:
        value = context.get(key)
        if value is not None and value != avoid_value:
            return value
    return f"adversarial_{key}_value"


def make_variant(
    query: dict[str, Any],
    risk_type: str,
    partial_context: dict[str, str],
    *,
    expected_safe_action: str,
    expected_target_preserved: bool,
    severity: str,
    description: str,
) -> dict[str, Any]:
    basis = {
        "query_id": query["query_id"],
        "risk_type": risk_type,
        "partial_context": partial_context,
        "expected_safe_action": expected_safe_action,
    }
    return {
        "audit_case_id": f"riskcase_{stable_hash(basis)}",
        "source_query_id": query["query_id"],
        "target_slot_id": query["target_slot_id"],
        "risk_type": risk_type,
        "subject": query["subject"],
        "property": query["property"],
        "partial_context": partial_context,
        "full_context": query["full_context"],
        "target_value": query["target_value"],
        "expected_safe_action": expected_safe_action,
        "expected_target_preserved": expected_target_preserved,
        "severity": severity,
        "description": description,
    }


def build_adversarial_cases(
    queries: list[dict[str, Any]],
    *,
    max_cases_per_type: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    ambiguous = [query for query in queries if int(query["candidate_count"]) > 1]
    unique = [query for query in queries if int(query["candidate_count"]) == 1]
    source_queries = ambiguous or queries

    for query in source_queries[:max_cases_per_type]:
        cases.append(
            make_variant(
                query,
                "missing_context",
                {},
                expected_safe_action="clarify_then_answer",
                expected_target_preserved=True,
                severity="medium",
                description="Remove all known context and require bounded clarification.",
            )
        )

    for query in source_queries[:max_cases_per_type]:
        noisy_key = next(iter(query["partial_context"]), "time_of_day")
        noisy_value = alternate_context_value(
            query,
            noisy_key,
            avoid_value=query["full_context"][noisy_key],
        )
        cases.append(
            make_variant(
                query,
                "noisy_context",
                {**query["partial_context"], noisy_key: noisy_value},
                expected_safe_action="reject_or_request_evidence",
                expected_target_preserved=False,
                severity="critical",
                description="Corrupt one supplied context key; promotion must not treat this as safe.",
            )
        )

    for query in source_queries[:max_cases_per_type]:
        correlated_context = {
            key: query["full_context"][key]
            for key in ("location", "season")
            if key in query["full_context"]
        }
        if len(correlated_context) < 2:
            continue
        cases.append(
            make_variant(
                query,
                "correlated_keys",
                correlated_context,
                expected_safe_action="clarify_then_answer",
                expected_target_preserved=True,
                severity="medium",
                description="Use only correlated keys; audit checks whether target remains recoverable.",
            )
        )

    for query in (unique or queries)[:max_cases_per_type]:
        cases.append(
            make_variant(
                query,
                "rare_context",
                dict(query["partial_context"]),
                expected_safe_action="direct_or_clarify_answer",
                expected_target_preserved=True,
                severity="medium",
                description="Use low-candidate context as rare/direct-answer control.",
            )
        )

    for query in source_queries[:max_cases_per_type]:
        full_context = query["full_context"]
        cases.append(
            make_variant(
                query,
                "same_context_contradiction",
                dict(full_context),
                expected_safe_action="request_evidence_on_value_conflict",
                expected_target_preserved=True,
                severity="critical",
                description="Same context with hypothetical incompatible value must route to evidence.",
            )
        )

    for query in source_queries[:max_cases_per_type]:
        drift_key = "season" if "season" in query["full_context"] else next(iter(CONTEXT_KEYS))
        drift_value = alternate_context_value(
            query,
            drift_key,
            avoid_value=query["full_context"][drift_key],
        )
        drift_context = {**query["full_context"], drift_key: drift_value}
        cases.append(
            make_variant(
                query,
                "temporal_drift",
                drift_context,
                expected_safe_action="new_slot_or_validity_check",
                expected_target_preserved=False,
                severity="medium",
                description="Change a validity-like key and require non-conflict handling.",
            )
        )

    return cases


def case_candidate_pairs(case: dict[str, Any], query: dict[str, Any]) -> list[dict[str, Any]]:
    return candidate_pairs(
        {
            "candidate_ids": query["candidate_ids"],
            "candidate_contexts": query["candidate_contexts"],
        }
    )


def execute_case(
    case: dict[str, Any],
    query_by_id: dict[str, dict[str, Any]],
    runtime_policy: dict[str, Any],
) -> dict[str, Any]:
    query = query_by_id[case["source_query_id"]]
    current_context = dict(case["partial_context"])
    candidates = filter_candidates(case_candidate_pairs(case, query), current_context)
    initial_count = len(candidates)
    steps: list[dict[str, Any]] = []
    while len(candidates) > 1 and len(current_context) < len(CONTEXT_KEYS):
        key = select_runtime_key(runtime_policy, current_context)
        if not key:
            break
        answer = case["full_context"].get(key)
        if answer is None:
            break
        before_count = len(candidates)
        current_context[key] = answer
        candidates = filter_candidates(candidates, {key: answer})
        after_count = len(candidates)
        steps.append(
            {
                "action": "ask_disambiguating_context",
                "asked_key": key,
                "answer_value": answer,
                "before_candidate_count": before_count,
                "after_candidate_count": after_count,
                "reduced_candidates": after_count < before_count,
                "target_preserved": target_present(candidates, case["target_slot_id"]),
                "selected_from_ranked_keys": key in runtime_policy["ranked_keys"],
            }
        )
    output = output_for_result(
        {
            "target_slot_id": case["target_slot_id"],
            "target_value": case["target_value"],
        },
        candidates,
        current_context,
        steps,
    )
    target_kept = target_present(candidates, case["target_slot_id"])
    resolved_unique = len(candidates) == 1 and target_kept
    if resolved_unique:
        observed_action = "direct_answer" if not steps else "clarify_then_answer"
    elif target_kept:
        observed_action = "clarification_incomplete"
    else:
        observed_action = "target_lost_or_rejected"
    expected_target_preserved = bool(case["expected_target_preserved"])
    if expected_target_preserved:
        case_safe = target_kept and output["type"] != "target_lost"
    else:
        case_safe = not resolved_unique and output["type"] != "answer"
    risk_detected = not case_safe or case["severity"] == "critical"
    return {
        "audit_case_id": case["audit_case_id"],
        "source_query_id": case["source_query_id"],
        "risk_type": case["risk_type"],
        "severity": case["severity"],
        "expected_safe_action": case["expected_safe_action"],
        "observed_action": observed_action,
        "initial_candidate_count": initial_count,
        "final_candidate_count": len(candidates),
        "initial_context": case["partial_context"],
        "final_context": current_context,
        "steps": steps,
        "target_preserved": target_kept,
        "resolved_unique": resolved_unique,
        "output_type": output["type"],
        "case_safe": case_safe,
        "risk_detected": risk_detected,
        "promotion_blocking": case["severity"] == "critical" and risk_detected,
        "description": case["description"],
    }


def aggregate_by_risk_type(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["risk_type"], []).append(record)
    return {
        risk_type: {
            "case_count": len(items),
            "safe_rate": mean([1.0 if item["case_safe"] else 0.0 for item in items]),
            "risk_detected_count": sum(1 for item in items if item["risk_detected"]),
            "promotion_blocking_count": sum(1 for item in items if item["promotion_blocking"]),
            "target_preservation_rate": mean(
                [1.0 if item["target_preserved"] else 0.0 for item in items]
            ),
        }
        for risk_type, items in sorted(grouped.items())
    }


def promotion_decision(
    records: list[dict[str, Any]],
    stage40_validation: dict[str, Any],
) -> dict[str, Any]:
    blocking_records = [record for record in records if record["promotion_blocking"]]
    critical_records = [record for record in records if record["severity"] == "critical"]
    return {
        "decision": "block_policy_update",
        "promotion_allowed": False,
        "blocking_case_count": len(blocking_records),
        "critical_case_count": len(critical_records),
        "stage40_critical_alert_count": stage40_validation["critical_alert_count"],
        "required_next_stage": "Stage 41B guarded policy review after adversarial replay",
        "rollback_required": True,
        "reason": (
            "Risk-first gate requires held-out/adversarial replay and rollback "
            "metadata before Stage 40 feedback can update runtime policy."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage30 = load_json(Path(args.partial_context_json))
    runtime = load_json(Path(args.runtime_json))
    stage40 = load_json(Path(args.feedback_json))
    stage30_validation = validate_stage30_artifact(stage30)
    runtime_validation = validate_runtime_artifact(runtime)
    stage40_validation = validate_stage40_artifact(stage40)
    queries = stage30.get("probe", {}).get("evaluated_queries", [])
    runtime_policy = runtime.get("runtime_policy", {})
    cases = build_adversarial_cases(queries, max_cases_per_type=args.max_cases_per_type)
    query_by_id = {query["query_id"]: query for query in queries}
    records = [execute_case(case, query_by_id, runtime_policy) for case in cases]
    replay_records = [execute_case(case, query_by_id, runtime_policy) for case in cases]
    deterministic_replay = records == replay_records
    risk_summary = aggregate_by_risk_type(records)
    decision = promotion_decision(records, stage40_validation)
    risk_types_present = set(risk_summary)
    critical_blocking_cases = [
        record for record in records if record["severity"] == "critical" and record["promotion_blocking"]
    ]
    safe_required_cases = [
        record
        for record in records
        if record["risk_type"] in {"missing_context", "correlated_keys", "rare_context"}
    ]
    risky_required_cases = [
        record
        for record in records
        if record["risk_type"] in {"noisy_context", "same_context_contradiction"}
    ]
    pass_fail = {
        "stage30_artifact_valid": stage30_validation["valid"],
        "runtime_artifact_valid": runtime_validation["valid"],
        "stage40_artifact_valid": stage40_validation["valid"],
        "adversarial_cases_present": bool(records),
        "required_risk_types_present": {
            "missing_context",
            "noisy_context",
            "correlated_keys",
            "rare_context",
            "same_context_contradiction",
            "temporal_drift",
        }.issubset(risk_types_present),
        "safe_required_cases_preserve_target": all(record["case_safe"] for record in safe_required_cases),
        "risky_cases_detected": all(record["risk_detected"] for record in risky_required_cases),
        "critical_risks_block_promotion": bool(critical_blocking_cases),
        "promotion_blocked": decision["promotion_allowed"] is False,
        "rollback_metadata_present": runtime_validation["rollback_present"],
        "stage40_replay_valid": stage40_validation["deterministic_replay"],
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "partial_context_json": args.partial_context_json,
        "runtime_json": args.runtime_json,
        "feedback_json": args.feedback_json,
        "case_count": len(records),
        "risk_types_present": sorted(risk_types_present),
        "critical_blocking_case_count": len(critical_blocking_cases),
        "promotion_allowed": decision["promotion_allowed"],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage41a_adversarial_risk_audit",
        "config": vars(args),
        "stage30_validation": stage30_validation,
        "runtime_validation": runtime_validation,
        "stage40_validation": stage40_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "risk_summary": risk_summary,
        "promotion_decision": decision,
        "audit_records": records,
        "claim_guardrail": (
            "Stage 41A is a risk-first adversarial audit. Passing means risks "
            "were measured and unsafe policy update was blocked; it is not a "
            "policy promotion or live-world robustness claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 41A adversarial risk audit failed: {pass_fail}")
    print("NeoCortex-X Stage 41A adversarial risk audit")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  cases: {len(records)}")
    print(f"  risk_types: {','.join(protocol_evidence['risk_types_present'])}")
    print(f"  critical_blocking_cases: {len(critical_blocking_cases)}")
    print(f"  promotion_allowed: {decision['promotion_allowed']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 41A adversarial risk audit.")
    parser.add_argument(
        "--partial-context-json",
        default="artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json",
    )
    parser.add_argument(
        "--runtime-json",
        default="artifacts/neocortex_x/stage37_clarification_runtime_loader.json",
    )
    parser.add_argument(
        "--feedback-json",
        default="artifacts/neocortex_x/stage40_clarification_monitor_feedback.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage41a_adversarial_risk_audit.json",
    )
    parser.add_argument("--max-cases-per-type", type=int, default=32)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
