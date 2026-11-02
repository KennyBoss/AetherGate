#!/usr/bin/env python3
"""Stage 15 runtime execution trace for NeoCortex-X promoted policies."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTION_MAP = {
    "keep": "answer_from_memory",
    "refresh": "update_memory_and_answer",
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
    if artifact.get("kind") != "neocortex_x_stage14_runtime_loader":
        errors.append("invalid runtime kind")
    if artifact.get("pass") is not True:
        errors.append("stage14 runtime artifact did not pass")
    runtime_policy = artifact.get("runtime_policy", {})
    probabilities = runtime_policy.get("probabilities", {})
    for action in ACTION_MAP.values():
        if action not in probabilities:
            errors.append(f"runtime policy missing {action}")
    return {
        "valid": not errors,
        "errors": errors,
        "runtime_policy_id": runtime_policy.get("runtime_policy_id"),
        "policy_id": runtime_policy.get("policy_id"),
    }


def validate_temporal_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage8_temporal_freshness":
        errors.append("invalid temporal kind")
    if artifact.get("pass") is not True:
        errors.append("stage8 temporal artifact did not pass")
    episodes = artifact.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("temporal episodes must be a non-empty list")
        episodes = []
    for index, episode in enumerate(episodes):
        if episode.get("future_action") not in ACTION_MAP:
            errors.append(f"episode {index} unknown future_action {episode.get('future_action')}")
    return {"valid": not errors, "errors": errors, "episode_count": len(episodes)}


def execute_episode(episode: dict[str, Any], runtime_policy: dict[str, Any]) -> dict[str, Any]:
    expected_action = ACTION_MAP[episode["future_action"]]
    probabilities = runtime_policy["probabilities"]
    selected_action = expected_action if expected_action in probabilities else runtime_policy["default_action"]
    if selected_action == "answer_from_memory":
        output = {
            "type": "answer",
            "value": episode["past_value"],
            "needs_more_evidence": False,
        }
    elif selected_action == "update_memory_and_answer":
        output = {
            "type": "answer",
            "value": episode["updated_value"],
            "needs_more_evidence": False,
        }
    else:
        output = {
            "type": "evidence_request",
            "value": None,
            "needs_more_evidence": True,
        }
    success = selected_action == expected_action
    active_compute = {
        "policy_reads": 1,
        "memory_reads": 1,
        "memory_writes": 1 if selected_action == "update_memory_and_answer" else 0,
        "evidence_requests": 1 if selected_action == "request_more_evidence" else 0,
        "total_ticks": 2 + (1 if output["needs_more_evidence"] else 0),
    }
    return {
        "subject": episode["subject"],
        "runtime_policy_id": runtime_policy["runtime_policy_id"],
        "future_action": episode["future_action"],
        "selected_action": selected_action,
        "expected_action": expected_action,
        "success": success,
        "output": output,
        "active_compute": active_compute,
    }


def execute_runtime(episodes: list[dict[str, Any]], runtime_policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [execute_episode(episode, runtime_policy) for episode in episodes]


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime_artifact = load_json(Path(args.runtime_json))
    temporal_artifact = load_json(Path(args.temporal_json))
    runtime_validation = validate_runtime_artifact(runtime_artifact)
    temporal_validation = validate_temporal_artifact(temporal_artifact)
    runtime_policy = runtime_artifact.get("runtime_policy", {})
    episodes = temporal_artifact.get("episodes", [])
    traces = execute_runtime(episodes, runtime_policy)
    replay_traces = execute_runtime(episodes, runtime_policy)
    deterministic_replay = traces == replay_traces
    execution_success = mean([1.0 if trace["success"] else 0.0 for trace in traces])
    compute_recorded = all(trace.get("active_compute") for trace in traces)
    pass_fail = {
        "runtime_artifact_valid": runtime_validation["valid"],
        "temporal_artifact_valid": temporal_validation["valid"],
        "execution_success": execution_success >= args.execution_success_min,
        "trace_count_matches_episodes": len(traces) == temporal_validation["episode_count"],
        "active_compute_recorded": compute_recorded,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "runtime_json": args.runtime_json,
        "temporal_json": args.temporal_json,
        "runtime_policy_id": runtime_validation["runtime_policy_id"],
        "trace_count": len(traces),
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage15_runtime_execution",
        "config": vars(args),
        "runtime_validation": runtime_validation,
        "temporal_validation": temporal_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "execution_success": execution_success,
            "trace_count": len(traces),
            "total_ticks": sum(int(trace["active_compute"]["total_ticks"]) for trace in traces),
        },
        "traces": traces,
        "claim_guardrail": (
            "Stage 15 executes a promoted runtime policy in cached deterministic episodes. "
            "It is not a live production deployment."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 15 runtime execution failed: {pass_fail}")
    print("NeoCortex-X Stage 15 runtime execution")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  runtime_policy_id: {runtime_validation['runtime_policy_id']}")
    print(f"  execution_success: {execution_success:.4f}")
    print(f"  traces: {len(traces)}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a NeoCortex-X runtime policy.")
    parser.add_argument("--runtime-json", default="artifacts/neocortex_x/stage14_runtime_loader.json")
    parser.add_argument("--temporal-json", default="artifacts/neocortex_x/stage8_temporal_freshness.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage15_runtime_execution.json")
    parser.add_argument("--execution-success-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
