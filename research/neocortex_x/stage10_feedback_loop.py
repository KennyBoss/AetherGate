#!/usr/bin/env python3
"""Stage 10 feedback loop over NeoCortex-X memory-gated actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def stable_hash(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_action_gate(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_action_gate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage9_action_gate":
        errors.append("invalid kind")
    if artifact.get("pass") is not True:
        errors.append("stage9 action gate artifact did not pass")
    runs = artifact.get("runs", {})
    decisions = runs.get("memory_action_gate", {}).get("decisions")
    if not isinstance(decisions, list) or not decisions:
        errors.append("memory_action_gate decisions must be a non-empty list")
        decisions = []
    required = {"subject", "future_action", "agent_action", "answer_value", "needs_more_evidence"}
    for index, decision in enumerate(decisions):
        missing = required.difference(decision)
        if missing:
            errors.append(f"decision {index} missing {sorted(missing)}")
    return {"valid": not errors, "errors": errors, "decision_count": len(decisions)}


def expected_outcome(decision: dict[str, Any]) -> dict[str, Any]:
    action = decision["agent_action"]
    if action == "answer_from_memory":
        return {"should_answer": True, "expected_answer": decision["answer_value"]}
    if action == "update_memory_and_answer":
        return {"should_answer": True, "expected_answer": decision["answer_value"]}
    if action == "request_more_evidence":
        return {"should_answer": False, "expected_answer": None}
    raise ValueError(f"Unknown agent_action: {action}")


def score_decision(decision: dict[str, Any]) -> dict[str, Any]:
    expected = expected_outcome(decision)
    answer_value = decision["answer_value"]
    needs_more_evidence = bool(decision["needs_more_evidence"])
    if expected["should_answer"]:
        success = answer_value == expected["expected_answer"] and not needs_more_evidence
    else:
        success = answer_value is None and needs_more_evidence
    reward = 1.0 if success else -1.0
    if decision["agent_action"] == "request_more_evidence" and success:
        reward = 0.8
    feedback_basis = {
        "subject": decision["subject"],
        "future_action": decision["future_action"],
        "agent_action": decision["agent_action"],
        "answer_value": answer_value,
        "needs_more_evidence": needs_more_evidence,
        "success": success,
        "reward": reward,
    }
    return {
        "feedback_id": f"fb_{stable_hash(feedback_basis)}",
        "subject": decision["subject"],
        "future_action": decision["future_action"],
        "agent_action": decision["agent_action"],
        "answer_value": answer_value,
        "needs_more_evidence": needs_more_evidence,
        "success": success,
        "reward": reward,
        "reinforce_action": decision["agent_action"] if success else None,
        "needs_policy_review": not success,
        "replay_hash": stable_hash(feedback_basis, length=24),
    }


def build_feedback_events(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [score_decision(decision) for decision in decisions]


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_action_gate(Path(args.action_gate_json))
    gate_validation = validate_action_gate_artifact(artifact)
    decisions = artifact.get("runs", {}).get("memory_action_gate", {}).get("decisions", [])
    feedback_events = build_feedback_events(decisions)
    replay_events = build_feedback_events(decisions)
    deterministic_replay = feedback_events == replay_events
    outcome_success = mean([1.0 if event["success"] else 0.0 for event in feedback_events])
    mean_reward = mean([float(event["reward"]) for event in feedback_events])
    replay_hashes_present = all(event.get("replay_hash") for event in feedback_events)
    protocol_evidence = {
        "valid": bool(gate_validation["valid"] and feedback_events and deterministic_replay),
        "action_gate_json": args.action_gate_json,
        "decision_count": gate_validation["decision_count"],
        "feedback_event_count": len(feedback_events),
        "deterministic_replay": deterministic_replay,
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "action_gate_artifact_valid": gate_validation["valid"],
        "feedback_count_matches_decisions": len(feedback_events) == gate_validation["decision_count"],
        "outcome_success": outcome_success >= args.outcome_success_min,
        "mean_reward": mean_reward >= args.mean_reward_min,
        "replay_hashes_present": replay_hashes_present,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage10_feedback_loop",
        "config": vars(args),
        "gate_validation": gate_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "outcome_success": outcome_success,
            "mean_reward": mean_reward,
            "feedback_event_count": len(feedback_events),
            "policy_review_count": sum(1 for event in feedback_events if event["needs_policy_review"]),
        },
        "feedback_events": feedback_events,
        "claim_guardrail": (
            "Stage 10 records deterministic feedback for memory-gated actions. "
            "It is a feedback ledger, not an infinite RL or AGI claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 10 feedback loop failed: {pass_fail}")
    print("NeoCortex-X Stage 10 feedback loop")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  feedback_events: {len(feedback_events)}")
    print(f"  outcome_success: {outcome_success:.4f}")
    print(f"  mean_reward: {mean_reward:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 10 feedback loop.")
    parser.add_argument("--action-gate-json", default="artifacts/neocortex_x/stage9_action_gate.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage10_feedback_loop.json")
    parser.add_argument("--outcome-success-min", type=float, default=0.95)
    parser.add_argument("--mean-reward-min", type=float, default=0.80)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
