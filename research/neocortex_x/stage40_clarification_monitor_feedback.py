#!/usr/bin/env python3
"""Stage 40 feedback ledger over NeoCortex-X clarification runtime monitors."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def validate_stage39_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage39_clarification_runtime_self_monitor":
        errors.append("invalid stage39 monitor kind")
    if artifact.get("pass") is not True:
        errors.append("stage39 clarification runtime self-monitor artifact did not pass")
    records = artifact.get("monitor_records")
    if not isinstance(records, list) or not records:
        errors.append("stage39 monitor_records must be a non-empty list")
        records = []
    required = {
        "monitor_tick_id",
        "query_id",
        "runtime_policy_id",
        "reported_runtime_path",
        "runtime_path_reconstruction_correct",
        "resolved_unique",
        "target_preserved",
        "output_consistency_correct",
        "step_summary",
        "compute_reconstruction_correct",
        "success",
    }
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            errors.append(f"monitor record {index} missing {sorted(missing)}")
            continue
        if not isinstance(record["step_summary"], dict):
            errors.append(f"monitor record {index} step_summary must be an object")
    return {"valid": not errors, "errors": errors, "record_count": len(records)}


def feedback_id(prefix: str, basis: dict[str, Any]) -> str:
    return f"{prefix}_{stable_hash(basis)}"


def score_runtime_path(record: dict[str, Any]) -> dict[str, Any]:
    success = (
        bool(record["runtime_path_reconstruction_correct"])
        and bool(record["success"])
        and bool(record["target_preserved"])
    )
    reward = 1.0 if success else -1.0
    basis = {
        "monitor_tick_id": record["monitor_tick_id"],
        "event_type": "runtime_path_feedback",
        "runtime_path": record["reported_runtime_path"],
        "success": success,
        "reward": reward,
    }
    return {
        "feedback_id": feedback_id("clarrtfb", basis),
        "event_type": "runtime_path_feedback",
        "monitor_tick_id": record["monitor_tick_id"],
        "query_id": record["query_id"],
        "runtime_policy_id": record["runtime_policy_id"],
        "runtime_path": record["reported_runtime_path"],
        "success": success,
        "reward": reward,
        "reinforce_runtime_path": record["reported_runtime_path"] if success else None,
        "reinforce_keys": [],
        "needs_runtime_review": not success,
        "target_loss_penalty": not bool(record["target_preserved"]),
        "alert_severity": "critical" if not success else "none",
        "replay_hash": stable_hash(basis, length=24),
    }


def score_output(record: dict[str, Any]) -> dict[str, Any]:
    success = bool(record["output_consistency_correct"])
    reward = 0.9 if success else -1.0
    basis = {
        "monitor_tick_id": record["monitor_tick_id"],
        "event_type": "output_consistency_feedback",
        "success": success,
        "reward": reward,
    }
    return {
        "feedback_id": feedback_id("clarrtfb", basis),
        "event_type": "output_consistency_feedback",
        "monitor_tick_id": record["monitor_tick_id"],
        "query_id": record["query_id"],
        "runtime_policy_id": record["runtime_policy_id"],
        "runtime_path": record["reported_runtime_path"],
        "success": success,
        "reward": reward,
        "reinforce_runtime_path": record["reported_runtime_path"] if success else None,
        "reinforce_keys": [],
        "needs_runtime_review": not success,
        "target_loss_penalty": False,
        "alert_severity": "critical" if not success else "none",
        "replay_hash": stable_hash(basis, length=24),
    }


def score_compute(record: dict[str, Any]) -> dict[str, Any]:
    success = bool(record["compute_reconstruction_correct"])
    reward = 0.6 if success else -0.8
    basis = {
        "monitor_tick_id": record["monitor_tick_id"],
        "event_type": "active_compute_feedback",
        "success": success,
        "reward": reward,
    }
    return {
        "feedback_id": feedback_id("clarrtfb", basis),
        "event_type": "active_compute_feedback",
        "monitor_tick_id": record["monitor_tick_id"],
        "query_id": record["query_id"],
        "runtime_policy_id": record["runtime_policy_id"],
        "runtime_path": record["reported_runtime_path"],
        "success": success,
        "reward": reward,
        "reinforce_runtime_path": None,
        "reinforce_keys": [],
        "needs_runtime_review": not success,
        "target_loss_penalty": False,
        "alert_severity": "warning" if not success else "none",
        "replay_hash": stable_hash(basis, length=24),
    }


def score_selected_keys(record: dict[str, Any]) -> dict[str, Any] | None:
    steps = record["step_summary"]
    if int(steps.get("step_count", 0)) == 0:
        return None
    success = (
        bool(steps.get("selected_keys_match_trace"))
        and bool(steps.get("all_steps_selected_from_ranked_keys"))
        and bool(steps.get("all_selected_keys_are_context_keys"))
        and bool(steps.get("all_steps_non_increasing_candidates"))
        and bool(steps.get("any_step_reduces_candidates"))
        and bool(steps.get("trace_reduces_candidates"))
        and bool(steps.get("all_steps_preserve_target"))
        and bool(record["resolved_unique"])
        and bool(record["target_preserved"])
    )
    selected_keys = list(steps.get("selected_keys_from_steps", []))
    reward = 0.7 if success else -1.0
    basis = {
        "monitor_tick_id": record["monitor_tick_id"],
        "event_type": "selected_key_feedback",
        "selected_keys": selected_keys,
        "success": success,
        "reward": reward,
    }
    return {
        "feedback_id": feedback_id("clarrtfb", basis),
        "event_type": "selected_key_feedback",
        "monitor_tick_id": record["monitor_tick_id"],
        "query_id": record["query_id"],
        "runtime_policy_id": record["runtime_policy_id"],
        "runtime_path": record["reported_runtime_path"],
        "success": success,
        "reward": reward,
        "reinforce_runtime_path": None,
        "reinforce_keys": selected_keys if success else [],
        "needs_runtime_review": not success,
        "target_loss_penalty": not bool(record["target_preserved"]),
        "alert_severity": "critical" if not success else "none",
        "replay_hash": stable_hash(basis, length=24),
    }


def score_efficiency_diagnostic(record: dict[str, Any]) -> dict[str, Any] | None:
    steps = record["step_summary"]
    if int(steps.get("step_count", 0)) == 0:
        return None
    if bool(steps.get("all_steps_reduce_candidates")):
        return None
    if not bool(steps.get("trace_reduces_candidates")):
        return None
    basis = {
        "monitor_tick_id": record["monitor_tick_id"],
        "event_type": "clarification_efficiency_diagnostic",
        "selected_keys": list(steps.get("selected_keys_from_steps", [])),
        "strict_step_reduction": False,
    }
    return {
        "feedback_id": feedback_id("clarrtinfo", basis),
        "event_type": "clarification_efficiency_diagnostic",
        "monitor_tick_id": record["monitor_tick_id"],
        "query_id": record["query_id"],
        "runtime_policy_id": record["runtime_policy_id"],
        "runtime_path": record["reported_runtime_path"],
        "success": True,
        "reward": 0.1,
        "reinforce_runtime_path": None,
        "reinforce_keys": [],
        "needs_runtime_review": False,
        "target_loss_penalty": False,
        "alert_severity": "info",
        "diagnostic": (
            "Not every clarification step reduced candidates, but the full "
            "trace made progress, preserved target, and resolved uniquely."
        ),
        "replay_hash": stable_hash(basis, length=24),
    }


def build_feedback_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for record in records:
        events.append(score_runtime_path(record))
        events.append(score_output(record))
        events.append(score_compute(record))
        key_event = score_selected_keys(record)
        if key_event:
            events.append(key_event)
        diagnostic_event = score_efficiency_diagnostic(record)
        if diagnostic_event:
            events.append(diagnostic_event)
    return events


def aggregate_key_rewards(events: list[dict[str, Any]]) -> dict[str, Any]:
    rewards_by_key: dict[str, list[float]] = {}
    for event in events:
        if event["event_type"] != "selected_key_feedback":
            continue
        for key in event.get("reinforce_keys", []):
            rewards_by_key.setdefault(key, []).append(float(event["reward"]))
    return {
        key: {
            "count": len(rewards),
            "mean_reward": mean(rewards),
            "total_reward": sum(rewards),
        }
        for key, rewards in sorted(rewards_by_key.items())
    }


def aggregate_runtime_path_rewards(events: list[dict[str, Any]]) -> dict[str, Any]:
    rewards_by_path: dict[str, list[float]] = {}
    for event in events:
        path = event.get("reinforce_runtime_path")
        if not path:
            continue
        rewards_by_path.setdefault(path, []).append(float(event["reward"]))
    return {
        path: {
            "count": len(rewards),
            "mean_reward": mean(rewards),
            "total_reward": sum(rewards),
        }
        for path, rewards in sorted(rewards_by_path.items())
    }


def alert_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    critical = [event for event in events if event.get("alert_severity") == "critical"]
    warning = [event for event in events if event.get("alert_severity") == "warning"]
    info = [event for event in events if event.get("alert_severity") == "info"]
    return {
        "critical_alert_count": len(critical),
        "warning_alert_count": len(warning),
        "info_alert_count": len(info),
        "critical_event_types": sorted({event["event_type"] for event in critical}),
        "warning_event_types": sorted({event["event_type"] for event in warning}),
        "info_event_types": sorted({event["event_type"] for event in info}),
    }


def expected_efficiency_diagnostics(records: list[dict[str, Any]]) -> int:
    count = 0
    for record in records:
        steps = record["step_summary"]
        if (
            int(steps.get("step_count", 0)) > 0
            and not bool(steps.get("all_steps_reduce_candidates"))
            and bool(steps.get("trace_reduces_candidates"))
        ):
            count += 1
    return count


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage39 = load_json(Path(args.monitor_json))
    validation = validate_stage39_artifact(stage39)
    records = stage39.get("monitor_records", [])
    feedback_events = build_feedback_events(records)
    replay_events = build_feedback_events(records)
    deterministic_replay = feedback_events == replay_events
    path_events = [event for event in feedback_events if event["event_type"] == "runtime_path_feedback"]
    output_events = [
        event for event in feedback_events if event["event_type"] == "output_consistency_feedback"
    ]
    compute_events = [event for event in feedback_events if event["event_type"] == "active_compute_feedback"]
    key_events = [event for event in feedback_events if event["event_type"] == "selected_key_feedback"]
    diagnostic_events = [
        event
        for event in feedback_events
        if event["event_type"] == "clarification_efficiency_diagnostic"
    ]
    success_rate = mean([1.0 if event["success"] else 0.0 for event in feedback_events])
    mean_reward = mean([float(event["reward"]) for event in feedback_events])
    policy_review_count = sum(1 for event in feedback_events if event["needs_runtime_review"])
    target_loss_penalty_count = sum(1 for event in feedback_events if event["target_loss_penalty"])
    replay_hashes_present = all(event.get("replay_hash") for event in feedback_events)
    key_rewards = aggregate_key_rewards(feedback_events)
    runtime_path_rewards = aggregate_runtime_path_rewards(feedback_events)
    alerts = alert_summary(feedback_events)
    expected_diagnostics = expected_efficiency_diagnostics(records)
    pass_fail = {
        "stage39_artifact_valid": validation["valid"],
        "feedback_events_present": bool(feedback_events),
        "runtime_path_feedback_count_matches": len(path_events) == validation["record_count"],
        "output_feedback_count_matches": len(output_events) == validation["record_count"],
        "compute_feedback_count_matches": len(compute_events) == validation["record_count"],
        "selected_key_feedback_present": bool(key_events),
        "efficiency_diagnostic_count_matches": len(diagnostic_events) == expected_diagnostics,
        "success_rate": success_rate >= args.success_rate_min,
        "mean_reward": mean_reward >= args.mean_reward_min,
        "no_runtime_review_needed": policy_review_count == 0,
        "no_target_loss_penalties": target_loss_penalty_count == 0,
        "no_critical_alerts": alerts["critical_alert_count"] == 0,
        "key_rewards_present": bool(key_rewards),
        "runtime_path_rewards_present": bool(runtime_path_rewards),
        "replay_hashes_present": replay_hashes_present,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "monitor_json": args.monitor_json,
        "monitor_record_count": validation["record_count"],
        "feedback_event_count": len(feedback_events),
        "runtime_path_event_count": len(path_events),
        "selected_key_event_count": len(key_events),
        "diagnostic_event_count": len(diagnostic_events),
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage40_clarification_monitor_feedback",
        "config": vars(args),
        "stage39_validation": validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "feedback_event_count": len(feedback_events),
            "policy_review_count": policy_review_count,
            "target_loss_penalty_count": target_loss_penalty_count,
            "expected_efficiency_diagnostics": expected_diagnostics,
        },
        "alert_summary": alerts,
        "key_rewards": key_rewards,
        "runtime_path_rewards": runtime_path_rewards,
        "feedback_events": feedback_events,
        "claim_guardrail": (
            "Stage 40 converts deterministic self-monitor records into feedback "
            "and alert events. It is not RL training, live deployment, or a "
            "production memory-scale claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 40 clarification monitor feedback failed: {pass_fail}")
    print("NeoCortex-X Stage 40 clarification monitor feedback")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  feedback_events: {len(feedback_events)}")
    print(f"  success_rate: {success_rate:.4f}")
    print(f"  mean_reward: {mean_reward:.4f}")
    print(f"  critical_alerts: {alerts['critical_alert_count']}")
    print(f"  info_alerts: {alerts['info_alert_count']}")
    print(f"  key_rewards: {','.join(sorted(key_rewards))}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeoCortex-X Stage 40 clarification monitor feedback."
    )
    parser.add_argument(
        "--monitor-json",
        default="artifacts/neocortex_x/stage39_clarification_runtime_self_monitor.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage40_clarification_monitor_feedback.json",
    )
    parser.add_argument("--success-rate-min", type=float, default=0.95)
    parser.add_argument("--mean-reward-min", type=float, default=0.75)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
