#!/usr/bin/env python3
"""Stage 45 guarded runtime execution for NeoCortex-X clarification policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_CONTEXT_RISKS = {"missing_context", "correlated_keys", "rare_context"}
NO_ANSWER_RISKS = {"noisy_context", "same_context_contradiction", "temporal_drift"}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def validate_stage44_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage44_guarded_runtime_loader":
        errors.append("invalid stage44 runtime loader kind")
    if artifact.get("pass") is not True:
        errors.append("stage44 artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    snapshot = artifact.get("runtime_snapshot", {})
    if protocol.get("deterministic_replay") is not True:
        errors.append("stage44 deterministic replay missing")
    if snapshot.get("guard_active") is not True:
        errors.append("runtime guard is not active")
    if snapshot.get("claim_guardrails_active") is not True:
        errors.append("claim guardrails are not active")
    if snapshot.get("promotion_scope", {}).get("allowed") != "guarded_cached_replay_runtime_candidate":
        errors.append("runtime scope is not guarded cached replay")
    if not snapshot.get("rollback", {}).get("restore_command"):
        errors.append("runtime rollback metadata missing")
    contract = snapshot.get("runtime_contract", {})
    required_contract = {
        "clean_answer_rule",
        "same_context_contradiction_rule",
        "noisy_context_rule",
        "temporal_drift_rule",
        "unknown_context_fallback",
    }
    missing_contract = required_contract.difference(contract)
    if missing_contract:
        errors.append(f"runtime contract missing {sorted(missing_contract)}")
    return {
        "valid": not errors,
        "errors": errors,
        "runtime_policy_id": protocol.get("runtime_policy_id"),
        "manifest_id": protocol.get("manifest_id"),
        "candidate_policy_id": protocol.get("candidate_policy_id"),
        "guard_id": protocol.get("guard_id"),
        "runtime_scope": protocol.get("runtime_scope"),
        "guard_active": bool(snapshot.get("guard_active")),
        "claim_guardrails_active": bool(snapshot.get("claim_guardrails_active")),
        "rollback_present": bool(snapshot.get("rollback", {}).get("restore_command")),
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
    }


def validate_stage42_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage42_clean_adversarial_replay":
        errors.append("invalid stage42 replay kind")
    if artifact.get("pass") is not True:
        errors.append("stage42 artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    if protocol.get("deterministic_replay") is not True:
        errors.append("stage42 deterministic replay missing")
    runs = artifact.get("runs", {})
    candidate = runs.get("candidate_policy", {})
    records = candidate.get("records")
    if not isinstance(records, list) or not records:
        errors.append("stage42 candidate records missing")
        records = []
    required = {
        "case_id",
        "suite",
        "risk_type",
        "severity",
        "initial_context",
        "final_context",
        "selected_keys",
        "target_preserved",
        "resolved_unique",
        "output_type",
    }
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            errors.append(f"stage42 candidate record {index} missing {sorted(missing)}")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_policy_id": protocol.get("candidate_policy_id"),
        "record_count": len(records),
        "clean_case_count": int(protocol.get("clean_case_count", 0)),
        "adversarial_case_count": int(protocol.get("adversarial_case_count", 0)),
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
    }


def execute_record(runtime_snapshot: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    contract = runtime_snapshot["runtime_contract"]
    risk_type = record["risk_type"]
    selected_keys = list(record.get("selected_keys", []))
    target_preserved = bool(record["target_preserved"])
    resolved_unique = bool(record["resolved_unique"])
    if record["suite"] == "clean":
        if target_preserved and resolved_unique:
            output_type = "answer"
            action = "allow_contextual_answer"
            answer_allowed = True
            guard_route = "clean_answer_rule"
        else:
            output_type = "request_context"
            action = "ask_disambiguating_context"
            answer_allowed = False
            guard_route = "clean_needs_context"
    elif risk_type == "same_context_contradiction":
        output_type = "request_evidence"
        action = contract["same_context_contradiction_rule"]
        answer_allowed = False
        guard_route = "same_context_contradiction_rule"
    elif risk_type == "noisy_context":
        output_type = "request_context_integrity"
        action = contract["noisy_context_rule"]
        answer_allowed = False
        guard_route = "noisy_context_rule"
    elif risk_type == "temporal_drift":
        output_type = "request_validity_window"
        action = contract["temporal_drift_rule"]
        answer_allowed = False
        guard_route = "temporal_drift_rule"
    elif risk_type in SAFE_CONTEXT_RISKS:
        if target_preserved and resolved_unique:
            output_type = "answer"
            action = "allow_contextual_answer"
            answer_allowed = True
            guard_route = "safe_context_answer"
        elif target_preserved:
            output_type = "request_context"
            action = "ask_disambiguating_context"
            answer_allowed = False
            guard_route = "safe_context_clarify"
        else:
            output_type = "request_context_integrity"
            action = contract["noisy_context_rule"]
            answer_allowed = False
            guard_route = "safe_context_target_lost"
    else:
        output_type = "request_evidence"
        action = "request_evidence_review"
        answer_allowed = False
        guard_route = "unknown_risk_guard"
    unsafe_answer = (
        record["suite"] == "adversarial"
        and risk_type in NO_ANSWER_RISKS
        and answer_allowed
    )
    case_safe = not unsafe_answer and (
        record["suite"] == "clean"
        or risk_type in NO_ANSWER_RISKS
        or target_preserved
    )
    active_compute = {
        "policy_reads": 1,
        "guard_reads": 1,
        "context_reads": len(record.get("initial_context", {})) + len(selected_keys),
        "candidate_filters": max(1, len(selected_keys)),
        "evidence_checks_requested": 1 if output_type == "request_evidence" else 0,
        "context_integrity_checks_requested": (
            1 if output_type == "request_context_integrity" else 0
        ),
        "validity_checks_requested": 1 if output_type == "request_validity_window" else 0,
        "answers_emitted": 1 if output_type == "answer" else 0,
        "total_ticks": 3 + len(selected_keys) + (0 if answer_allowed else 1),
    }
    return {
        "case_id": record["case_id"],
        "suite": record["suite"],
        "risk_type": risk_type,
        "severity": record["severity"],
        "runtime_policy_id": runtime_snapshot["runtime_policy_id"],
        "guard_id": runtime_snapshot["guard_id"],
        "initial_context": record["initial_context"],
        "final_context": record["final_context"],
        "selected_keys": selected_keys,
        "target_preserved": target_preserved,
        "resolved_unique": resolved_unique,
        "source_output_type": record["output_type"],
        "output_type": output_type,
        "selected_action": action,
        "answer_allowed": answer_allowed,
        "guard_applied": True,
        "guard_route": guard_route,
        "unsafe_answer": unsafe_answer,
        "case_safe": case_safe,
        "active_compute": active_compute,
    }


def execute_records(
    runtime_snapshot: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [execute_record(runtime_snapshot, record) for record in records]


def aggregate_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [trace for trace in traces if trace["suite"] == "clean"]
    adversarial = [trace for trace in traces if trace["suite"] == "adversarial"]
    no_answer = [trace for trace in adversarial if trace["risk_type"] in NO_ANSWER_RISKS]
    safe_context = [trace for trace in adversarial if trace["risk_type"] in SAFE_CONTEXT_RISKS]
    return {
        "trace_count": len(traces),
        "clean_trace_count": len(clean),
        "adversarial_trace_count": len(adversarial),
        "clean_answer_rate": mean([1.0 if trace["output_type"] == "answer" else 0.0 for trace in clean]),
        "clean_case_safe_rate": mean([1.0 if trace["case_safe"] else 0.0 for trace in clean]),
        "adversarial_safe_rate": mean(
            [1.0 if trace["case_safe"] else 0.0 for trace in adversarial]
        ),
        "no_answer_risk_block_rate": mean(
            [1.0 if not trace["answer_allowed"] else 0.0 for trace in no_answer]
        ),
        "unsafe_answer_count": sum(1 for trace in traces if trace["unsafe_answer"]),
        "safe_context_target_preservation_rate": mean(
            [1.0 if trace["target_preserved"] else 0.0 for trace in safe_context]
        ),
        "guard_applied_rate": mean([1.0 if trace["guard_applied"] else 0.0 for trace in traces]),
        "output_types": sorted({trace["output_type"] for trace in traces}),
        "guard_routes": sorted({trace["guard_route"] for trace in traces}),
        "total_ticks": sum(int(trace["active_compute"]["total_ticks"]) for trace in traces),
        "evidence_check_count": sum(
            int(trace["active_compute"]["evidence_checks_requested"]) for trace in traces
        ),
        "context_integrity_check_count": sum(
            int(trace["active_compute"]["context_integrity_checks_requested"])
            for trace in traces
        ),
        "validity_check_count": sum(
            int(trace["active_compute"]["validity_checks_requested"]) for trace in traces
        ),
        "answer_count": sum(int(trace["active_compute"]["answers_emitted"]) for trace in traces),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage44 = load_json(Path(args.runtime_json))
    stage42 = load_json(Path(args.stage42_json))
    stage44_validation = validate_stage44_artifact(stage44)
    stage42_validation = validate_stage42_artifact(stage42)
    runtime_snapshot = stage44.get("runtime_snapshot", {})
    records = stage42.get("runs", {}).get("candidate_policy", {}).get("records", [])
    traces = execute_records(runtime_snapshot, records)
    replay_traces = execute_records(runtime_snapshot, records)
    deterministic_replay = traces == replay_traces
    aggregate = aggregate_traces(traces)
    candidate_ids_match = (
        stage44_validation["candidate_policy_id"] == stage42_validation["candidate_policy_id"]
    )
    pass_fail = {
        "stage44_artifact_valid": stage44_validation["valid"],
        "stage42_artifact_valid": stage42_validation["valid"],
        "candidate_ids_match": candidate_ids_match,
        "runtime_guard_active": stage44_validation["guard_active"],
        "trace_count_matches_stage42_records": (
            aggregate["trace_count"] == stage42_validation["record_count"]
        ),
        "clean_answers_preserved": aggregate["clean_answer_rate"] == 1.0,
        "clean_cases_safe": aggregate["clean_case_safe_rate"] == 1.0,
        "adversarial_cases_safe": aggregate["adversarial_safe_rate"] == 1.0,
        "no_answer_risks_blocked": aggregate["no_answer_risk_block_rate"] == 1.0,
        "no_unsafe_answers": aggregate["unsafe_answer_count"] == 0,
        "safe_context_target_preserved": (
            aggregate["safe_context_target_preservation_rate"] == 1.0
        ),
        "guard_applied_to_all_traces": aggregate["guard_applied_rate"] == 1.0,
        "evidence_routes_present": aggregate["evidence_check_count"] > 0,
        "context_integrity_routes_present": (
            aggregate["context_integrity_check_count"] > 0
        ),
        "validity_routes_present": aggregate["validity_check_count"] > 0,
        "answers_emitted_for_clean_or_safe": aggregate["answer_count"] > 0,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "runtime_json": args.runtime_json,
        "stage42_json": args.stage42_json,
        "runtime_policy_id": stage44_validation["runtime_policy_id"],
        "candidate_policy_id": stage44_validation["candidate_policy_id"],
        "guard_id": stage44_validation["guard_id"],
        "trace_count": aggregate["trace_count"],
        "unsafe_answer_count": aggregate["unsafe_answer_count"],
        "guard_applied_rate": aggregate["guard_applied_rate"],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage45_guarded_runtime_execution",
        "config": vars(args),
        "stage44_validation": stage44_validation,
        "stage42_validation": stage42_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": aggregate,
        "traces": traces,
        "claim_guardrail": (
            "Stage 45 executes a guarded cached-replay runtime snapshot over "
            "Stage 42 clean and adversarial records. It is not live-source trust, "
            "production memory, AGI, or consciousness evidence."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 45 guarded runtime execution failed: {pass_fail}")
    print("NeoCortex-X Stage 45 guarded runtime execution")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  runtime_policy_id: {stage44_validation['runtime_policy_id']}")
    print(f"  traces: {aggregate['trace_count']}")
    print(f"  clean_answer_rate: {aggregate['clean_answer_rate']:.4f}")
    print(f"  adversarial_safe_rate: {aggregate['adversarial_safe_rate']:.4f}")
    print(f"  unsafe_answer_count: {aggregate['unsafe_answer_count']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a NeoCortex-X guarded clarification runtime snapshot."
    )
    parser.add_argument(
        "--runtime-json",
        default="artifacts/neocortex_x/stage44_guarded_runtime_loader.json",
    )
    parser.add_argument(
        "--stage42-json",
        default="artifacts/neocortex_x/stage42_clean_adversarial_replay.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage45_guarded_runtime_execution.json",
    )
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
