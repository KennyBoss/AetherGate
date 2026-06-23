#!/usr/bin/env python3
"""Stage 23 contextual runtime execution traces for NeoCortex-X."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ACTION = {
    "context_match": "answer_from_context_memory",
    "same_subject_different_context": "answer_with_context_scope",
    "request_context": "ask_context_question",
    "flag_conflict": "request_more_evidence",
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


def validate_runtime_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage22_contextual_runtime_loader":
        errors.append("invalid runtime kind")
    if artifact.get("pass") is not True:
        errors.append("stage22 contextual runtime artifact did not pass")
    runtime_policy = artifact.get("runtime_policy", {})
    default_actions = runtime_policy.get("default_actions", {})
    for decision, expected_action in EXPECTED_ACTION.items():
        if default_actions.get(decision) != expected_action:
            errors.append(f"runtime default for {decision} is not {expected_action}")
    if runtime_policy.get("unknown_context_fallback") != "ask_context_question":
        errors.append("unknown context fallback is not ask_context_question")
    return {
        "valid": not errors,
        "errors": errors,
        "runtime_policy_id": runtime_policy.get("runtime_policy_id"),
        "policy_id": runtime_policy.get("policy_id"),
    }


def validate_contextual_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage16_contextual_memory":
        errors.append("invalid contextual memory kind")
    if artifact.get("pass") is not True:
        errors.append("stage16 contextual memory artifact did not pass")
    episodes = artifact.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("contextual episodes must be a non-empty list")
        episodes = []
    required = {"observation_id", "decision", "observed_value", "context", "subject", "property"}
    for index, episode in enumerate(episodes):
        missing = required.difference(episode)
        if missing:
            errors.append(f"episode {index} missing {sorted(missing)}")
        if episode.get("decision") not in EXPECTED_ACTION:
            errors.append(f"episode {index} unknown decision {episode.get('decision')}")
    return {"valid": not errors, "errors": errors, "episode_count": len(episodes)}


def context_label(context: dict[str, str]) -> str:
    if not context:
        return "missing_context"
    return ", ".join(f"{key}={context[key]}" for key in sorted(context))


def output_for_action(episode: dict[str, Any], selected_action: str) -> dict[str, Any]:
    if selected_action == "answer_from_context_memory":
        return {
            "type": "answer",
            "value": episode["observed_value"],
            "context_scope": episode["context"],
            "needs_context": False,
            "needs_more_evidence": False,
        }
    if selected_action == "answer_with_context_scope":
        return {
            "type": "answer_with_scope",
            "value": episode["observed_value"],
            "context_scope": episode["context"],
            "context_label": context_label(episode["context"]),
            "needs_context": False,
            "needs_more_evidence": False,
        }
    if selected_action == "ask_context_question":
        return {
            "type": "clarification",
            "value": None,
            "question": (
                f"Need context for {episode['subject']}.{episode['property']}: "
                "time_of_day, weather, location, or season."
            ),
            "context_scope": {},
            "needs_context": True,
            "needs_more_evidence": False,
        }
    return {
        "type": "evidence_request",
        "value": None,
        "conflict_value": episode["observed_value"],
        "context_scope": episode["context"],
        "context_label": context_label(episode["context"]),
        "needs_context": False,
        "needs_more_evidence": True,
    }


def execute_episode(episode: dict[str, Any], runtime_policy: dict[str, Any]) -> dict[str, Any]:
    context_decision = episode["decision"]
    expected_action = EXPECTED_ACTION[context_decision]
    selected_action = runtime_policy["default_actions"].get(
        context_decision,
        runtime_policy["unknown_context_fallback"],
    )
    output = output_for_action(episode, selected_action)
    success = selected_action == expected_action
    active_compute = {
        "policy_reads": 1,
        "context_reads": 1,
        "memory_reads": 1 if selected_action in {"answer_from_context_memory", "answer_with_context_scope"} else 0,
        "context_questions": 1 if selected_action == "ask_context_question" else 0,
        "evidence_requests": 1 if selected_action == "request_more_evidence" else 0,
        "total_ticks": 2
        + (1 if output["needs_context"] else 0)
        + (1 if output["needs_more_evidence"] else 0),
    }
    return {
        "observation_id": episode["observation_id"],
        "subject": episode["subject"],
        "property": episode["property"],
        "context": episode["context"],
        "runtime_policy_id": runtime_policy["runtime_policy_id"],
        "context_decision": context_decision,
        "selected_action": selected_action,
        "expected_action": expected_action,
        "success": success,
        "output": output,
        "active_compute": active_compute,
        "false_evidence_request": (
            selected_action == "request_more_evidence"
            and context_decision != "flag_conflict"
        ),
    }


def execute_runtime(episodes: list[dict[str, Any]], runtime_policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [execute_episode(episode, runtime_policy) for episode in episodes]


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime_artifact = load_json(Path(args.runtime_json))
    contextual_artifact = load_json(Path(args.contextual_json))
    runtime_validation = validate_runtime_artifact(runtime_artifact)
    contextual_validation = validate_contextual_artifact(contextual_artifact)
    runtime_policy = runtime_artifact.get("runtime_policy", {})
    episodes = contextual_artifact.get("episodes", [])
    traces = execute_runtime(episodes, runtime_policy)
    replay_traces = execute_runtime(episodes, runtime_policy)
    deterministic_replay = traces == replay_traces
    execution_success = mean([1.0 if trace["success"] else 0.0 for trace in traces])
    output_types = sorted({trace["output"]["type"] for trace in traces})
    active_compute_recorded = all(trace.get("active_compute") for trace in traces)
    no_false_evidence = not any(trace["false_evidence_request"] for trace in traces)
    missing_context_clarified = all(
        trace["output"]["type"] == "clarification" and trace["output"]["needs_context"]
        for trace in traces
        if trace["context_decision"] == "request_context"
    )
    pass_fail = {
        "runtime_artifact_valid": runtime_validation["valid"],
        "contextual_artifact_valid": contextual_validation["valid"],
        "execution_success": execution_success >= args.execution_success_min,
        "trace_count_matches_episodes": len(traces) == contextual_validation["episode_count"],
        "all_output_types_present": {
            "answer",
            "answer_with_scope",
            "clarification",
            "evidence_request",
        }.issubset(set(output_types)),
        "no_false_evidence_request_for_contextual_alias": no_false_evidence,
        "missing_context_clarified": missing_context_clarified,
        "active_compute_recorded": active_compute_recorded,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "runtime_json": args.runtime_json,
        "contextual_json": args.contextual_json,
        "runtime_policy_id": runtime_validation["runtime_policy_id"],
        "trace_count": len(traces),
        "output_types": output_types,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage23_contextual_runtime_execution",
        "config": vars(args),
        "runtime_validation": runtime_validation,
        "contextual_validation": contextual_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "execution_success": execution_success,
            "trace_count": len(traces),
            "total_ticks": sum(int(trace["active_compute"]["total_ticks"]) for trace in traces),
            "false_evidence_request_count": sum(1 for trace in traces if trace["false_evidence_request"]),
        },
        "traces": traces,
        "claim_guardrail": (
            "Stage 23 executes a promoted contextual runtime policy in cached "
            "deterministic episodes. It is not a live production deployment."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 23 contextual runtime execution failed: {pass_fail}")
    print("NeoCortex-X Stage 23 contextual runtime execution")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  runtime_policy_id: {runtime_validation['runtime_policy_id']}")
    print(f"  execution_success: {execution_success:.4f}")
    print(f"  traces: {len(traces)}")
    print(f"  output_types: {','.join(output_types)}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a NeoCortex-X contextual runtime policy.")
    parser.add_argument("--runtime-json", default="artifacts/neocortex_x/stage22_contextual_runtime_loader.json")
    parser.add_argument("--contextual-json", default="artifacts/neocortex_x/stage16_contextual_memory.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage23_contextual_runtime_execution.json")
    parser.add_argument("--execution-success-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
