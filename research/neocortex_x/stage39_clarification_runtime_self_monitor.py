#!/usr/bin/env python3
"""Stage 39 self-monitor over NeoCortex-X clarification runtime traces."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS


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


def validate_execution_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage38_clarification_runtime_execution":
        errors.append("invalid stage38 execution kind")
    if artifact.get("pass") is not True:
        errors.append("stage38 clarification runtime execution artifact did not pass")
    traces = artifact.get("traces")
    if not isinstance(traces, list) or not traces:
        errors.append("stage38 traces must be a non-empty list")
        traces = []
    required = {
        "query_id",
        "target_slot_id",
        "runtime_policy_id",
        "initial_context",
        "final_context",
        "initial_candidate_count",
        "final_candidate_count",
        "initial_ambiguous",
        "steps",
        "resolved_unique",
        "target_preserved",
        "selected_keys",
        "output",
        "success",
        "active_compute",
    }
    for index, trace in enumerate(traces):
        missing = required.difference(trace)
        if missing:
            errors.append(f"trace {index} missing {sorted(missing)}")
            continue
        if not isinstance(trace["steps"], list):
            errors.append(f"trace {index} steps must be a list")
        if not isinstance(trace["selected_keys"], list):
            errors.append(f"trace {index} selected_keys must be a list")
        if not isinstance(trace["output"], dict):
            errors.append(f"trace {index} output must be an object")
        if not isinstance(trace["active_compute"], dict):
            errors.append(f"trace {index} active_compute must be an object")
    return {"valid": not errors, "errors": errors, "trace_count": len(traces)}


def classify_runtime_path(trace: dict[str, Any]) -> str:
    output_type = trace["output"].get("type")
    if output_type == "target_lost":
        return "target_lost"
    if output_type == "clarification_incomplete":
        return "clarification_incomplete"
    if trace["initial_ambiguous"]:
        return "clarify_then_answer" if trace["steps"] else "ambiguous_direct_answer"
    return "direct_answer"


def expected_runtime_path(trace: dict[str, Any]) -> str:
    if not trace["success"]:
        return trace["output"].get("type", "failed_execution")
    if trace["initial_ambiguous"]:
        return "clarify_then_answer"
    return "direct_answer"


def expected_active_compute(trace: dict[str, Any]) -> dict[str, int]:
    step_count = len(trace["steps"])
    output_type = trace["output"].get("type")
    return {
        "policy_reads": max(1, step_count),
        "context_reads": len(trace["initial_context"]) + step_count,
        "candidate_filters": 1 + step_count,
        "memory_reads": 1,
        "context_questions": step_count,
        "answers_emitted": 1 if output_type == "answer" else 0,
        "total_ticks": 2 + (2 * step_count) + (1 if output_type == "answer" else 0),
    }


def compute_reconstruction_correct(trace: dict[str, Any]) -> bool:
    expected = expected_active_compute(trace)
    observed = trace["active_compute"]
    return all(int(observed.get(key, -1)) == value for key, value in expected.items())


def output_consistency_correct(trace: dict[str, Any]) -> bool:
    output = trace["output"]
    if trace["success"]:
        return (
            output.get("type") == "answer"
            and output.get("slot_id") == trace["target_slot_id"]
            and output.get("context") == trace["final_context"]
            and output.get("clarification_steps") == len(trace["steps"])
            and output.get("needs_more_context") is False
            and output.get("value") is not None
        )
    if output.get("type") == "clarification_incomplete":
        return output.get("needs_more_context") is True
    if output.get("type") == "target_lost":
        return trace["target_preserved"] is False
    return False


def step_summary(trace: dict[str, Any]) -> dict[str, Any]:
    steps = trace["steps"]
    selected_from_steps = [step.get("asked_key") for step in steps]
    non_increasing_candidates = all(
        int(step.get("after_candidate_count", 0)) <= int(step.get("before_candidate_count", 0))
        for step in steps
    )
    all_steps_reduce_candidates = all(bool(step.get("reduced_candidates")) for step in steps)
    any_step_reduces_candidates = any(bool(step.get("reduced_candidates")) for step in steps)
    trace_reduces_candidates = int(trace["final_candidate_count"]) < int(
        trace["initial_candidate_count"]
    )
    target_preserved = all(bool(step.get("target_preserved")) for step in steps)
    selected_from_ranked = all(bool(step.get("selected_from_ranked_keys")) for step in steps)
    valid_context_keys = all(key in CONTEXT_KEYS for key in selected_from_steps)
    return {
        "step_count": len(steps),
        "selected_keys_from_steps": selected_from_steps,
        "selected_keys_match_trace": selected_from_steps == trace["selected_keys"],
        "all_steps_non_increasing_candidates": non_increasing_candidates,
        "all_steps_reduce_candidates": all_steps_reduce_candidates,
        "any_step_reduces_candidates": any_step_reduces_candidates,
        "trace_reduces_candidates": trace_reduces_candidates,
        "all_steps_preserve_target": target_preserved,
        "all_steps_selected_from_ranked_keys": selected_from_ranked,
        "all_selected_keys_are_context_keys": valid_context_keys,
    }


def monitor_trace(trace: dict[str, Any], index: int) -> dict[str, Any]:
    runtime_path = classify_runtime_path(trace)
    expected_path = expected_runtime_path(trace)
    steps = step_summary(trace)
    expected_compute = expected_active_compute(trace)
    return {
        "monitor_tick_id": f"clarification_runtime_tick_{index}",
        "query_id": trace["query_id"],
        "runtime_policy_id": trace["runtime_policy_id"],
        "reported_runtime_path": runtime_path,
        "expected_runtime_path": expected_path,
        "runtime_path_reconstruction_correct": runtime_path == expected_path,
        "initial_ambiguous": bool(trace["initial_ambiguous"]),
        "initial_candidate_count": trace["initial_candidate_count"],
        "final_candidate_count": trace["final_candidate_count"],
        "resolved_unique": bool(trace["resolved_unique"]),
        "target_preserved": bool(trace["target_preserved"]),
        "output_type": trace["output"].get("type"),
        "output_consistency_correct": output_consistency_correct(trace),
        "step_summary": steps,
        "expected_active_compute": expected_compute,
        "observed_active_compute": trace["active_compute"],
        "compute_reconstruction_correct": compute_reconstruction_correct(trace),
        "success": bool(trace["success"]),
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
    stepped_records = [record for record in records if record["step_summary"]["step_count"] > 0]
    runtime_paths_present = sorted({record["reported_runtime_path"] for record in records})
    runtime_path_accuracy = mean(
        [1.0 if record["runtime_path_reconstruction_correct"] else 0.0 for record in records]
    )
    output_consistency = mean(
        [1.0 if record["output_consistency_correct"] else 0.0 for record in records]
    )
    compute_accuracy = mean(
        [1.0 if record["compute_reconstruction_correct"] else 0.0 for record in records]
    )
    selected_key_accuracy = mean(
        [
            1.0
            if record["step_summary"]["selected_keys_match_trace"]
            and record["step_summary"]["all_steps_selected_from_ranked_keys"]
            and record["step_summary"]["all_selected_keys_are_context_keys"]
            else 0.0
            for record in records
        ]
    )
    strict_step_reduction_rate = mean(
        [
            1.0
            if record["step_summary"]["all_steps_reduce_candidates"]
            and record["step_summary"]["all_steps_preserve_target"]
            else 0.0
            for record in stepped_records
        ]
    )
    clarification_trace_progress_accuracy = mean(
        [
            1.0
            if record["step_summary"]["all_steps_non_increasing_candidates"]
            and record["step_summary"]["any_step_reduces_candidates"]
            and record["step_summary"]["trace_reduces_candidates"]
            and record["step_summary"]["all_steps_preserve_target"]
            and record["resolved_unique"]
            and record["success"]
            else 0.0
            for record in stepped_records
        ]
    )
    target_preservation = mean([1.0 if record["target_preserved"] else 0.0 for record in records])
    pass_fail = {
        "execution_artifact_valid": execution_validation["valid"],
        "record_count_matches_traces": len(records) == execution_validation["trace_count"],
        "direct_and_clarification_paths_present": {"direct_answer", "clarify_then_answer"}.issubset(
            set(runtime_paths_present)
        ),
        "runtime_path_reconstruction_accuracy": runtime_path_accuracy
        >= args.reconstruction_accuracy_min,
        "output_consistency_accuracy": output_consistency >= args.reconstruction_accuracy_min,
        "compute_reconstruction_accuracy": compute_accuracy >= args.reconstruction_accuracy_min,
        "selected_key_reconstruction_accuracy": selected_key_accuracy
        >= args.reconstruction_accuracy_min,
        "clarification_trace_progress_accuracy": clarification_trace_progress_accuracy
        >= args.reconstruction_accuracy_min,
        "target_preservation_rate": target_preservation == 1.0,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "execution_json": args.execution_json,
        "runtime_policy_id": artifact.get("protocol_evidence", {}).get("runtime_policy_id"),
        "trace_count": execution_validation["trace_count"],
        "monitor_record_count": len(records),
        "runtime_paths_present": runtime_paths_present,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage39_clarification_runtime_self_monitor",
        "config": vars(args),
        "execution_validation": execution_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "runtime_path_reconstruction_accuracy": runtime_path_accuracy,
            "output_consistency_accuracy": output_consistency,
            "compute_reconstruction_accuracy": compute_accuracy,
            "selected_key_reconstruction_accuracy": selected_key_accuracy,
            "strict_step_reduction_rate": strict_step_reduction_rate,
            "clarification_trace_progress_accuracy": clarification_trace_progress_accuracy,
            "target_preservation_rate": target_preservation,
            "monitor_record_count": len(records),
            "stepped_record_count": len(stepped_records),
        },
        "monitor_records": records,
        "claim_guardrail": (
            "Stage 39 is an operational self-monitor over clarification runtime "
            "traces. It is not a consciousness, Phi, live deployment, or "
            "production memory-scale claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 39 clarification runtime self-monitor failed: {pass_fail}")
    print("NeoCortex-X Stage 39 clarification runtime self-monitor")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  monitor_records: {len(records)}")
    print(f"  runtime_path_reconstruction_accuracy: {runtime_path_accuracy:.4f}")
    print(f"  output_consistency_accuracy: {output_consistency:.4f}")
    print(f"  compute_reconstruction_accuracy: {compute_accuracy:.4f}")
    print(f"  selected_key_reconstruction_accuracy: {selected_key_accuracy:.4f}")
    print(f"  runtime_paths: {','.join(runtime_paths_present)}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeoCortex-X Stage 39 clarification runtime self-monitor."
    )
    parser.add_argument(
        "--execution-json",
        default="artifacts/neocortex_x/stage38_clarification_runtime_execution.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage39_clarification_runtime_self_monitor.json",
    )
    parser.add_argument("--reconstruction-accuracy-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
