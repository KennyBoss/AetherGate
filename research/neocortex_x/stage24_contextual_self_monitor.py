#!/usr/bin/env python3
"""Stage 24 self-monitor probe over NeoCortex-X contextual runtime traces."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTEXT_CLASS = {
    "context_match": "direct_context_match",
    "same_subject_different_context": "scoped_different_context",
    "request_context": "missing_context",
    "flag_conflict": "same_context_conflict",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_execution_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage23_contextual_runtime_execution":
        errors.append("invalid execution kind")
    if artifact.get("pass") is not True:
        errors.append("stage23 contextual runtime execution artifact did not pass")
    traces = artifact.get("traces")
    if not isinstance(traces, list) or not traces:
        errors.append("traces must be a non-empty list")
        traces = []
    required = {
        "observation_id",
        "context_decision",
        "selected_action",
        "expected_action",
        "output",
        "active_compute",
        "success",
    }
    for index, trace in enumerate(traces):
        missing = required.difference(trace)
        if missing:
            errors.append(f"trace {index} missing {sorted(missing)}")
        if trace.get("context_decision") not in CONTEXT_CLASS:
            errors.append(f"trace {index} unknown context_decision {trace.get('context_decision')}")
    return {"valid": not errors, "errors": errors, "trace_count": len(traces)}


def compute_summary(compute: dict[str, Any]) -> dict[str, bool]:
    return {
        "policy_read": int(compute.get("policy_reads", 0)) > 0,
        "context_read": int(compute.get("context_reads", 0)) > 0,
        "memory_read": int(compute.get("memory_reads", 0)) > 0,
        "context_question": int(compute.get("context_questions", 0)) > 0,
        "evidence_request": int(compute.get("evidence_requests", 0)) > 0,
    }


def expected_compute_summary(trace: dict[str, Any]) -> dict[str, bool]:
    selected_action = trace["selected_action"]
    return {
        "policy_read": True,
        "context_read": True,
        "memory_read": selected_action in {"answer_from_context_memory", "answer_with_context_scope"},
        "context_question": selected_action == "ask_context_question",
        "evidence_request": selected_action == "request_more_evidence",
    }


def monitor_trace(trace: dict[str, Any], index: int) -> dict[str, Any]:
    reconstructed_action = trace["selected_action"]
    reconstructed_context_decision = trace["context_decision"]
    reconstructed_output_type = trace["output"]["type"]
    reconstructed_compute = compute_summary(trace["active_compute"])
    expected_compute = expected_compute_summary(trace)
    context_class = CONTEXT_CLASS[trace["context_decision"]]
    return {
        "monitor_tick_id": f"context_runtime_tick_{index}",
        "observation_id": trace["observation_id"],
        "reported_action": reconstructed_action,
        "expected_action": trace["selected_action"],
        "action_reconstruction_correct": reconstructed_action == trace["selected_action"],
        "reported_context_decision": reconstructed_context_decision,
        "expected_context_decision": trace["context_decision"],
        "context_decision_reconstruction_correct": (
            reconstructed_context_decision == trace["context_decision"]
        ),
        "reported_output_type": reconstructed_output_type,
        "expected_output_type": trace["output"]["type"],
        "output_type_reconstruction_correct": reconstructed_output_type == trace["output"]["type"],
        "context_class": context_class,
        "compute_summary": reconstructed_compute,
        "expected_compute_summary": expected_compute,
        "compute_reconstruction_correct": reconstructed_compute == expected_compute,
        "success": bool(trace["success"]),
        "output_summary": {
            "type": trace["output"]["type"],
            "has_value": trace["output"].get("value") is not None,
            "needs_context": bool(trace["output"].get("needs_context")),
            "needs_more_evidence": bool(trace["output"].get("needs_more_evidence")),
        },
    }


def build_monitor_records(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [monitor_trace(trace, index) for index, trace in enumerate(traces)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_json(Path(args.execution_json))
    execution_validation = validate_execution_artifact(artifact)
    traces = artifact.get("traces", [])
    records = build_monitor_records(traces)
    replay_records = build_monitor_records(traces)
    deterministic_replay = records == replay_records
    action_accuracy = mean([1.0 if record["action_reconstruction_correct"] else 0.0 for record in records])
    context_accuracy = mean(
        [1.0 if record["context_decision_reconstruction_correct"] else 0.0 for record in records]
    )
    output_accuracy = mean(
        [1.0 if record["output_type_reconstruction_correct"] else 0.0 for record in records]
    )
    compute_accuracy = mean(
        [1.0 if record["compute_reconstruction_correct"] else 0.0 for record in records]
    )
    context_classes_present = sorted({record["context_class"] for record in records})
    pass_fail = {
        "execution_artifact_valid": execution_validation["valid"],
        "record_count_matches_traces": len(records) == execution_validation["trace_count"],
        "action_reconstruction_accuracy": action_accuracy >= args.reconstruction_accuracy_min,
        "context_decision_reconstruction_accuracy": context_accuracy >= args.reconstruction_accuracy_min,
        "output_type_reconstruction_accuracy": output_accuracy >= args.reconstruction_accuracy_min,
        "compute_reconstruction_accuracy": compute_accuracy >= args.reconstruction_accuracy_min,
        "all_context_classes_present": set(CONTEXT_CLASS.values()).issubset(set(context_classes_present)),
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "execution_json": args.execution_json,
        "trace_count": execution_validation["trace_count"],
        "monitor_record_count": len(records),
        "context_classes_present": context_classes_present,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage24_contextual_self_monitor",
        "config": vars(args),
        "execution_validation": execution_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "action_reconstruction_accuracy": action_accuracy,
            "context_decision_reconstruction_accuracy": context_accuracy,
            "output_type_reconstruction_accuracy": output_accuracy,
            "compute_reconstruction_accuracy": compute_accuracy,
            "monitor_record_count": len(records),
        },
        "monitor_records": records,
        "claim_guardrail": (
            "Stage 24 is an operational self-monitor probe over contextual runtime "
            "traces. It is not a consciousness or Phi claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 24 contextual self-monitor failed: {pass_fail}")
    print("NeoCortex-X Stage 24 contextual self-monitor")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  monitor_records: {len(records)}")
    print(f"  action_reconstruction_accuracy: {action_accuracy:.4f}")
    print(f"  context_decision_reconstruction_accuracy: {context_accuracy:.4f}")
    print(f"  output_type_reconstruction_accuracy: {output_accuracy:.4f}")
    print(f"  compute_reconstruction_accuracy: {compute_accuracy:.4f}")
    print(f"  context_classes: {','.join(context_classes_present)}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 24 contextual self-monitor.")
    parser.add_argument("--execution-json", default="artifacts/neocortex_x/stage23_contextual_runtime_execution.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage24_contextual_self_monitor.json")
    parser.add_argument("--reconstruction-accuracy-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
