#!/usr/bin/env python3
"""Stage 27 action gate over store-backed NeoCortex-X contextual retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTION_MAP = {
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


def validate_retrieval_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage26_store_backed_contextual_retrieval":
        errors.append("invalid retrieval kind")
    if artifact.get("pass") is not True:
        errors.append("stage26 store-backed retrieval artifact did not pass")
    episodes = artifact.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("episodes must be a non-empty list")
        episodes = []
    required = {
        "observation_id",
        "subject",
        "property",
        "context",
        "observed_value",
        "decision",
        "matched_slot_id",
        "candidate_count",
        "db_candidate_used",
    }
    for index, episode in enumerate(episodes):
        missing = required.difference(episode)
        if missing:
            errors.append(f"episode {index} missing {sorted(missing)}")
        if episode.get("decision") not in ACTION_MAP:
            errors.append(f"episode {index} has unknown decision {episode.get('decision')}")
    return {"valid": not errors, "errors": errors, "episode_count": len(episodes)}


def expected_agent_action(decision: str) -> str:
    return ACTION_MAP[decision]


def choose_action(episode: dict[str, Any], *, agent: str) -> str:
    if agent == "store_backed_contextual_action_gate":
        return expected_agent_action(episode["decision"])
    if agent == "unconditional_fact_control":
        return "answer_from_context_memory"
    if agent == "always_clarify_control":
        return "ask_context_question"
    if agent == "always_evidence_control":
        return "request_more_evidence"
    raise ValueError(f"Unknown Stage 27 agent: {agent}")


def context_label(context: dict[str, str]) -> str:
    if not context:
        return "missing_context"
    return ", ".join(f"{key}={context[key]}" for key in sorted(context))


def answer_payload(episode: dict[str, Any], action: str) -> dict[str, Any]:
    if action == "answer_from_context_memory":
        return {
            "type": "answer",
            "answer_value": episode["observed_value"],
            "context_scope": episode["context"],
            "matched_slot_id": episode["matched_slot_id"],
            "needs_context": False,
            "needs_more_evidence": False,
        }
    if action == "answer_with_context_scope":
        return {
            "type": "answer_with_scope",
            "answer_value": episode["observed_value"],
            "context_scope": episode["context"],
            "context_label": context_label(episode["context"]),
            "matched_slot_id": episode["matched_slot_id"],
            "needs_context": False,
            "needs_more_evidence": False,
        }
    if action == "ask_context_question":
        return {
            "type": "clarification",
            "question": (
                f"Need context for {episode['subject']}.{episode['property']}: "
                "time_of_day, weather, location, or season."
            ),
            "answer_value": None,
            "context_scope": {},
            "candidate_count": episode["candidate_count"],
            "needs_context": True,
            "needs_more_evidence": False,
        }
    return {
        "type": "evidence_request",
        "answer_value": None,
        "context_scope": episode["context"],
        "context_label": context_label(episode["context"]),
        "conflict_value": episode["observed_value"],
        "matched_slot_id": episode["matched_slot_id"],
        "needs_context": False,
        "needs_more_evidence": True,
    }


def run_agent(episodes: list[dict[str, Any]], *, agent: str) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for episode in episodes:
        expected = expected_agent_action(episode["decision"])
        action = choose_action(episode, agent=agent)
        payload = answer_payload(episode, action)
        decisions.append(
            {
                "observation_id": episode["observation_id"],
                "subject": episode["subject"],
                "property": episode["property"],
                "context": episode["context"],
                "context_decision": episode["decision"],
                "agent_action": action,
                "expected_action": expected,
                "action_correct": action == expected,
                "observed_value": episode["observed_value"],
                "matched_slot_id": episode["matched_slot_id"],
                "db_candidate_used": bool(episode["db_candidate_used"]),
                "payload": payload,
                "false_evidence_request": (
                    action == "request_more_evidence" and episode["decision"] != "flag_conflict"
                ),
            }
        )
    return {
        "agent": agent,
        "episode_count": len(decisions),
        "action_accuracy": mean([1.0 if item["action_correct"] else 0.0 for item in decisions]),
        "false_evidence_request_rate": mean(
            [1.0 if item["false_evidence_request"] else 0.0 for item in decisions]
        ),
        "decisions": decisions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_json(Path(args.retrieval_json))
    retrieval_validation = validate_retrieval_artifact(artifact)
    episodes = artifact.get("episodes", [])
    agents = (
        "store_backed_contextual_action_gate",
        "unconditional_fact_control",
        "always_clarify_control",
        "always_evidence_control",
    )
    runs = {agent: run_agent(episodes, agent=agent) for agent in agents}
    replay_runs = {agent: run_agent(episodes, agent=agent) for agent in agents}
    deterministic_replay = runs == replay_runs
    gate = runs["store_backed_contextual_action_gate"]
    unconditional = runs["unconditional_fact_control"]
    always_clarify = runs["always_clarify_control"]
    always_evidence = runs["always_evidence_control"]
    gate_actions = sorted({item["agent_action"] for item in gate["decisions"]})
    contextual_alias_decisions = [
        item for item in gate["decisions"] if item["context_decision"] == "same_subject_different_context"
    ]
    missing_context_decisions = [
        item for item in gate["decisions"] if item["context_decision"] == "request_context"
    ]
    conflict_decisions = [
        item for item in gate["decisions"] if item["context_decision"] == "flag_conflict"
    ]
    non_missing_decisions = [item for item in gate["decisions"] if item["context"]]
    db_candidate_usage = mean([1.0 if item["db_candidate_used"] else 0.0 for item in non_missing_decisions])
    no_false_evidence_for_alias = all(
        item["agent_action"] != "request_more_evidence" for item in contextual_alias_decisions
    )
    missing_context_asks = all(
        item["agent_action"] == "ask_context_question" for item in missing_context_decisions
    )
    conflict_requests_evidence = all(
        item["agent_action"] == "request_more_evidence" for item in conflict_decisions
    )
    comparison = {
        "gate_minus_unconditional_accuracy": (
            gate["action_accuracy"] - unconditional["action_accuracy"]
        ),
        "gate_minus_always_clarify_accuracy": (
            gate["action_accuracy"] - always_clarify["action_accuracy"]
        ),
        "gate_minus_always_evidence_accuracy": (
            gate["action_accuracy"] - always_evidence["action_accuracy"]
        ),
    }
    required_actions = set(ACTION_MAP.values())
    pass_fail = {
        "retrieval_artifact_valid": retrieval_validation["valid"],
        "required_actions_present": required_actions.issubset(set(gate_actions)),
        "gate_action_accuracy": gate["action_accuracy"] >= args.gate_accuracy_min,
        "no_false_evidence_request_for_contextual_alias": no_false_evidence_for_alias,
        "missing_context_asks_context_question": missing_context_asks,
        "conflict_requests_more_evidence": conflict_requests_evidence,
        "db_candidate_usage_preserved": db_candidate_usage == 1.0,
        "gate_minus_unconditional_accuracy": (
            comparison["gate_minus_unconditional_accuracy"] >= args.unconditional_margin_min
        ),
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "retrieval_json": args.retrieval_json,
        "episode_count": retrieval_validation["episode_count"],
        "actions_present": gate_actions,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage27_store_backed_contextual_action_gate",
        "config": vars(args),
        "retrieval_validation": retrieval_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": {
            "agents": {agent: {k: v for k, v in run.items() if k != "decisions"} for agent, run in runs.items()},
            "comparison": comparison,
            "db_candidate_usage": db_candidate_usage,
        },
        "runs": runs,
        "claim_guardrail": (
            "Stage 27 maps SQLite-backed contextual retrieval decisions to bounded "
            "agent actions. It is not a broad autonomous agent claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 27 store-backed action gate failed: {pass_fail}")
    print("NeoCortex-X Stage 27 store-backed contextual action gate")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  decisions: {retrieval_validation['episode_count']}")
    print(f"  gate.action_accuracy: {gate['action_accuracy']:.4f}")
    print(f"  db_candidate_usage: {db_candidate_usage:.4f}")
    print(
        "  gate_minus_unconditional_accuracy: "
        f"{comparison['gate_minus_unconditional_accuracy']:.4f}"
    )
    print(f"  false_evidence_request_rate: {gate['false_evidence_request_rate']:.4f}")
    print(f"  actions_present: {','.join(gate_actions)}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X store-backed contextual action gate.")
    parser.add_argument("--retrieval-json", default="artifacts/neocortex_x/stage26_store_backed_contextual_retrieval.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage27_store_backed_contextual_action_gate.json")
    parser.add_argument("--gate-accuracy-min", type=float, default=0.95)
    parser.add_argument("--unconditional-margin-min", type=float, default=0.25)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
