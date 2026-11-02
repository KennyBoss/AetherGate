#!/usr/bin/env python3
"""Stage 18 feedback loop for NeoCortex-X contextual actions."""

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


def load_contextual_gate(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contextual_gate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage17_contextual_action_gate":
        errors.append("invalid kind")
    if artifact.get("pass") is not True:
        errors.append("stage17 contextual action gate artifact did not pass")
    runs = artifact.get("runs", {})
    decisions = runs.get("contextual_action_gate", {}).get("decisions")
    if not isinstance(decisions, list) or not decisions:
        errors.append("contextual_action_gate decisions must be a non-empty list")
        decisions = []
    required = {
        "observation_id",
        "subject",
        "property",
        "context",
        "context_decision",
        "agent_action",
        "expected_action",
        "observed_value",
        "payload",
    }
    for index, decision in enumerate(decisions):
        missing = required.difference(decision)
        if missing:
            errors.append(f"decision {index} missing {sorted(missing)}")
    return {"valid": not errors, "errors": errors, "decision_count": len(decisions)}


def score_decision(decision: dict[str, Any]) -> dict[str, Any]:
    action = decision["agent_action"]
    context_decision = decision["context_decision"]
    payload = decision["payload"]
    needs_context = bool(payload.get("needs_context"))
    needs_more_evidence = bool(payload.get("needs_more_evidence"))
    answer_value = payload.get("answer_value")
    context_scope = payload.get("context_scope") or {}

    if action == "answer_from_context_memory":
        success = (
            context_decision == "context_match"
            and answer_value is not None
            and bool(context_scope)
            and not needs_context
            and not needs_more_evidence
        )
        reward = 1.0 if success else -1.0
    elif action == "answer_with_context_scope":
        success = (
            context_decision == "same_subject_different_context"
            and answer_value == decision["observed_value"]
            and bool(context_scope)
            and not needs_context
            and not needs_more_evidence
        )
        reward = 1.0 if success else -1.0
    elif action == "ask_context_question":
        success = (
            context_decision == "request_context"
            and needs_context
            and not needs_more_evidence
            and payload.get("question")
        )
        reward = 0.9 if success else -1.0
    elif action == "request_more_evidence":
        success = (
            context_decision == "flag_conflict"
            and needs_more_evidence
            and not needs_context
            and answer_value is None
        )
        reward = 0.8 if success else -1.0
    else:
        raise ValueError(f"Unknown contextual agent_action: {action}")

    feedback_basis = {
        "observation_id": decision["observation_id"],
        "context_decision": context_decision,
        "agent_action": action,
        "payload": payload,
        "success": bool(success),
        "reward": reward,
    }
    return {
        "feedback_id": f"ctxfb_{stable_hash(feedback_basis)}",
        "observation_id": decision["observation_id"],
        "subject": decision["subject"],
        "property": decision["property"],
        "context": decision["context"],
        "context_decision": context_decision,
        "agent_action": action,
        "observed_value": decision["observed_value"],
        "success": bool(success),
        "reward": reward,
        "reinforce_action": action if success else None,
        "needs_policy_review": not success,
        "false_evidence_penalty": (
            action == "request_more_evidence"
            and context_decision == "same_subject_different_context"
        ),
        "replay_hash": stable_hash(feedback_basis, length=24),
    }


def build_feedback_events(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [score_decision(decision) for decision in decisions]


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_contextual_gate(Path(args.contextual_action_json))
    gate_validation = validate_contextual_gate_artifact(artifact)
    decisions = artifact.get("runs", {}).get("contextual_action_gate", {}).get("decisions", [])
    feedback_events = build_feedback_events(decisions)
    replay_events = build_feedback_events(decisions)
    deterministic_replay = feedback_events == replay_events
    outcome_success = mean([1.0 if event["success"] else 0.0 for event in feedback_events])
    mean_reward = mean([float(event["reward"]) for event in feedback_events])
    replay_hashes_present = all(event.get("replay_hash") for event in feedback_events)
    scoped_answers_reinforced = all(
        event["success"] and event["reinforce_action"] == "answer_with_context_scope"
        for event in feedback_events
        if event["context_decision"] == "same_subject_different_context"
    )
    missing_context_reinforced = all(
        event["success"] and event["reinforce_action"] == "ask_context_question"
        for event in feedback_events
        if event["context_decision"] == "request_context"
    )
    conflicts_request_evidence = all(
        event["success"] and event["reinforce_action"] == "request_more_evidence"
        for event in feedback_events
        if event["context_decision"] == "flag_conflict"
    )
    no_false_evidence_penalties = not any(event["false_evidence_penalty"] for event in feedback_events)
    protocol_evidence = {
        "valid": bool(gate_validation["valid"] and feedback_events and deterministic_replay),
        "contextual_action_json": args.contextual_action_json,
        "decision_count": gate_validation["decision_count"],
        "feedback_event_count": len(feedback_events),
        "deterministic_replay": deterministic_replay,
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "contextual_action_gate_valid": gate_validation["valid"],
        "feedback_count_matches_decisions": len(feedback_events) == gate_validation["decision_count"],
        "outcome_success": outcome_success >= args.outcome_success_min,
        "mean_reward": mean_reward >= args.mean_reward_min,
        "contextual_scoped_answers_reinforced": scoped_answers_reinforced,
        "missing_context_clarifications_reinforced": missing_context_reinforced,
        "real_conflicts_routed_to_evidence_request": conflicts_request_evidence,
        "no_false_evidence_penalties_for_contextual_alias": no_false_evidence_penalties,
        "replay_hashes_present": replay_hashes_present,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage18_contextual_feedback",
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
            "false_evidence_penalty_count": sum(
                1 for event in feedback_events if event["false_evidence_penalty"]
            ),
        },
        "feedback_events": feedback_events,
        "claim_guardrail": (
            "Stage 18 records deterministic feedback for contextual actions. "
            "It is a bounded feedback ledger, not an infinite RL, consciousness, or AGI claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 18 contextual feedback failed: {pass_fail}")
    print("NeoCortex-X Stage 18 contextual feedback")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  feedback_events: {len(feedback_events)}")
    print(f"  outcome_success: {outcome_success:.4f}")
    print(f"  mean_reward: {mean_reward:.4f}")
    print(f"  false_evidence_penalty_count: {payload['metrics']['false_evidence_penalty_count']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 18 contextual feedback loop.")
    parser.add_argument("--contextual-action-json", default="artifacts/neocortex_x/stage17_contextual_action_gate.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage18_contextual_feedback.json")
    parser.add_argument("--outcome-success-min", type=float, default=0.95)
    parser.add_argument("--mean-reward-min", type=float, default=0.90)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
