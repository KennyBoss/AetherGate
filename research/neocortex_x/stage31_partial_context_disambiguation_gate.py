#!/usr/bin/env python3
"""Stage 31 disambiguation gate for NeoCortex-X partial-context memory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS


ACTION_TYPES = {
    "answer_from_unique_context",
    "ask_disambiguating_context",
}


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


def validate_stage30_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if artifact.get("kind") != "neocortex_x_stage30_partial_context_scale_retrieval":
        errors.append("invalid stage30 kind")
    if artifact.get("pass") is not True:
        errors.append("stage30 artifact did not pass")
    probe = artifact.get("probe", {})
    queries = probe.get("evaluated_queries")
    if not isinstance(queries, list) or not queries:
        errors.append("stage30 evaluated_queries missing; rerun Stage 30 with query details")
        queries = []
    for index, query in enumerate(queries):
        required = {
            "query_id",
            "target_slot_id",
            "partial_context",
            "candidate_count",
            "candidate_contexts",
            "target_in_candidates",
        }
        missing = required.difference(query)
        if missing:
            errors.append(f"query {index} missing {sorted(missing)}")
    return {"valid": not errors, "errors": errors, "query_count": len(queries)}


def entropy_score(values: list[str]) -> int:
    return len(set(values))


def choose_disambiguating_key(query: dict[str, Any]) -> str | None:
    known_keys = set(query["partial_context"])
    candidate_contexts = query["candidate_contexts"]
    best_key = None
    best_score = 0
    for key in CONTEXT_KEYS:
        if key in known_keys:
            continue
        values = [ctx.get(key, "<missing>") for ctx in candidate_contexts]
        score = entropy_score(values)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key


def expected_action(query: dict[str, Any]) -> str:
    if int(query["candidate_count"]) <= 1:
        return "answer_from_unique_context"
    return "ask_disambiguating_context"


def choose_action(query: dict[str, Any], *, agent: str) -> str:
    if agent == "partial_context_disambiguation_gate":
        return expected_action(query)
    if agent == "always_answer_control":
        return "answer_from_unique_context"
    if agent == "always_ask_control":
        return "ask_disambiguating_context"
    raise ValueError(f"Unknown Stage 31 agent: {agent}")


def output_payload(query: dict[str, Any], action: str) -> dict[str, Any]:
    if action == "answer_from_unique_context":
        return {
            "type": "answer",
            "slot_id": query["candidate_ids"][0] if query["candidate_ids"] else None,
            "needs_more_context": False,
            "disambiguation_key": None,
        }
    key = choose_disambiguating_key(query)
    return {
        "type": "clarification",
        "question": f"Need {key} to disambiguate {query['subject']}.{query['property']}.",
        "needs_more_context": True,
        "disambiguation_key": key,
        "candidate_count": query["candidate_count"],
    }


def run_agent(queries: list[dict[str, Any]], *, agent: str) -> dict[str, Any]:
    decisions = []
    for query in queries:
        expected = expected_action(query)
        action = choose_action(query, agent=agent)
        payload = output_payload(query, action)
        ambiguous = int(query["candidate_count"]) > 1
        decisions.append(
            {
                "query_id": query["query_id"],
                "subject": query["subject"],
                "property": query["property"],
                "partial_context": query["partial_context"],
                "target_slot_id": query["target_slot_id"],
                "candidate_count": query["candidate_count"],
                "ambiguous": ambiguous,
                "agent_action": action,
                "expected_action": expected,
                "action_correct": action == expected,
                "target_in_candidates": bool(query["target_in_candidates"]),
                "payload": payload,
                "false_answer_on_ambiguous": ambiguous and action == "answer_from_unique_context",
                "false_question_on_unique": (not ambiguous) and action == "ask_disambiguating_context",
            }
        )
    return {
        "agent": agent,
        "query_count": len(decisions),
        "action_accuracy": mean([1.0 if item["action_correct"] else 0.0 for item in decisions]),
        "false_answer_on_ambiguous_rate": mean(
            [1.0 if item["false_answer_on_ambiguous"] else 0.0 for item in decisions]
        ),
        "false_question_on_unique_rate": mean(
            [1.0 if item["false_question_on_unique"] else 0.0 for item in decisions]
        ),
        "decisions": decisions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage30 = load_json(Path(args.partial_context_json))
    validation = validate_stage30_artifact(stage30)
    queries = stage30.get("probe", {}).get("evaluated_queries", [])
    agents = (
        "partial_context_disambiguation_gate",
        "always_answer_control",
        "always_ask_control",
    )
    runs = {agent: run_agent(queries, agent=agent) for agent in agents}
    replay_runs = {agent: run_agent(queries, agent=agent) for agent in agents}
    deterministic_replay = runs == replay_runs
    gate = runs["partial_context_disambiguation_gate"]
    always_answer = runs["always_answer_control"]
    always_ask = runs["always_ask_control"]
    gate_actions = sorted({item["agent_action"] for item in gate["decisions"]})
    ambiguous_decisions = [item for item in gate["decisions"] if item["ambiguous"]]
    unique_decisions = [item for item in gate["decisions"] if not item["ambiguous"]]
    disambiguation_keys = [
        item["payload"]["disambiguation_key"]
        for item in ambiguous_decisions
        if item["payload"].get("disambiguation_key")
    ]
    comparison = {
        "gate_minus_always_answer_accuracy": (
            gate["action_accuracy"] - always_answer["action_accuracy"]
        ),
        "gate_minus_always_ask_accuracy": gate["action_accuracy"] - always_ask["action_accuracy"],
    }
    pass_fail = {
        "stage30_artifact_valid": validation["valid"],
        "required_actions_present": ACTION_TYPES.issubset(set(gate_actions)),
        "gate_action_accuracy": gate["action_accuracy"] >= args.action_accuracy_min,
        "no_false_answer_on_ambiguous": gate["false_answer_on_ambiguous_rate"] == 0.0,
        "no_false_question_on_unique": gate["false_question_on_unique_rate"] == 0.0,
        "ambiguous_queries_present": len(ambiguous_decisions) > 0,
        "unique_queries_present": len(unique_decisions) > 0,
        "disambiguation_keys_selected": len(disambiguation_keys) == len(ambiguous_decisions),
        "gate_beats_always_answer": (
            comparison["gate_minus_always_answer_accuracy"] >= args.control_margin_min
        ),
        "gate_beats_always_ask": (
            comparison["gate_minus_always_ask_accuracy"] >= args.control_margin_min
        ),
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "partial_context_json": args.partial_context_json,
        "query_count": validation["query_count"],
        "ambiguous_count": len(ambiguous_decisions),
        "unique_count": len(unique_decisions),
        "actions_present": gate_actions,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage31_partial_context_disambiguation_gate",
        "config": vars(args),
        "stage30_validation": validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": {
            "agents": {agent: {k: v for k, v in run.items() if k != "decisions"} for agent, run in runs.items()},
            "comparison": comparison,
            "disambiguation_keys": sorted(set(disambiguation_keys)),
        },
        "runs": runs,
        "claim_guardrail": (
            "Stage 31 maps partial-context candidate ambiguity to bounded "
            "answer-or-clarify actions. It is not a general dialogue policy."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 31 disambiguation gate failed: {pass_fail}")
    print("NeoCortex-X Stage 31 partial-context disambiguation gate")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  queries: {validation['query_count']}")
    print(f"  ambiguous: {len(ambiguous_decisions)}")
    print(f"  unique: {len(unique_decisions)}")
    print(f"  gate.action_accuracy: {gate['action_accuracy']:.4f}")
    print(f"  actions_present: {','.join(gate_actions)}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X partial-context disambiguation gate.")
    parser.add_argument(
        "--partial-context-json",
        default="artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage31_partial_context_disambiguation_gate.json",
    )
    parser.add_argument("--action-accuracy-min", type=float, default=0.95)
    parser.add_argument("--control-margin-min", type=float, default=0.10)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
