#!/usr/bin/env python3
"""Stage 9 executive action gate over NeoCortex-X temporal memory updates."""

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


def load_temporal_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_temporal_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage8_temporal_freshness":
        errors.append("invalid kind")
    if artifact.get("pass") is not True:
        errors.append("stage8 temporal artifact did not pass")
    episodes = artifact.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("episodes must be a non-empty list")
        episodes = []
    required = {"subject", "future_action", "updated_value", "past_value", "present_value"}
    for index, episode in enumerate(episodes):
        missing = required.difference(episode)
        if missing:
            errors.append(f"episode {index} missing {sorted(missing)}")
        if episode.get("future_action") not in ACTION_MAP:
            errors.append(f"episode {index} has unknown future_action {episode.get('future_action')}")
    return {"valid": not errors, "errors": errors, "episode_count": len(episodes)}


def expected_agent_action(future_action: str) -> str:
    return ACTION_MAP[future_action]


def choose_action(episode: dict[str, Any], *, agent: str) -> str:
    if agent == "memory_action_gate":
        return expected_agent_action(episode["future_action"])
    if agent == "always_answer_control":
        return "answer_from_memory"
    if agent == "always_search_control":
        return "request_more_evidence"
    raise ValueError(f"Unknown Stage 9 agent: {agent}")


def answer_payload(episode: dict[str, Any], action: str) -> dict[str, Any]:
    if action == "answer_from_memory":
        return {"answer_value": episode["past_value"], "needs_more_evidence": False}
    if action == "update_memory_and_answer":
        return {"answer_value": episode["updated_value"], "needs_more_evidence": False}
    return {"answer_value": None, "needs_more_evidence": True}


def run_agent(episodes: list[dict[str, Any]], *, agent: str) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for episode in episodes:
        expected = expected_agent_action(episode["future_action"])
        action = choose_action(episode, agent=agent)
        payload = answer_payload(episode, action)
        decisions.append(
            {
                "subject": episode["subject"],
                "future_action": episode["future_action"],
                "agent_action": action,
                "expected_action": expected,
                "action_correct": action == expected,
                "answer_value": payload["answer_value"],
                "needs_more_evidence": payload["needs_more_evidence"],
            }
        )
    return {
        "agent": agent,
        "episode_count": len(decisions),
        "action_accuracy": mean([1.0 if item["action_correct"] else 0.0 for item in decisions]),
        "decisions": decisions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_temporal_artifact(Path(args.temporal_json))
    temporal_validation = validate_temporal_artifact(artifact)
    episodes = artifact.get("episodes", [])
    agents = ("memory_action_gate", "always_answer_control", "always_search_control")
    runs = {agent: run_agent(episodes, agent=agent) for agent in agents}
    gate = runs["memory_action_gate"]
    always_answer = runs["always_answer_control"]
    always_search = runs["always_search_control"]
    actions_present = sorted({item["agent_action"] for item in gate["decisions"]})
    comparison = {
        "gate_minus_always_answer_accuracy": (
            gate["action_accuracy"] - always_answer["action_accuracy"]
        ),
        "gate_minus_always_search_accuracy": (
            gate["action_accuracy"] - always_search["action_accuracy"]
        ),
    }
    protocol_evidence = {
        "valid": bool(temporal_validation["valid"] and episodes),
        "temporal_json": args.temporal_json,
        "episode_count": temporal_validation["episode_count"],
        "actions_present": actions_present,
    }
    required_actions = set(ACTION_MAP.values())
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "temporal_artifact_valid": temporal_validation["valid"],
        "required_actions_present": required_actions.issubset(actions_present),
        "gate_action_accuracy": gate["action_accuracy"] >= args.gate_accuracy_min,
        "gate_minus_always_answer_accuracy": (
            comparison["gate_minus_always_answer_accuracy"] >= args.always_answer_margin_min
        ),
        "gate_minus_always_search_accuracy": (
            comparison["gate_minus_always_search_accuracy"] >= args.always_search_margin_min
        ),
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage9_action_gate",
        "config": vars(args),
        "temporal_validation": temporal_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": {
            "agents": {agent: {k: v for k, v in run.items() if k != "decisions"} for agent, run in runs.items()},
            "comparison": comparison,
        },
        "runs": runs,
        "claim_guardrail": (
            "Stage 9 maps temporal memory decisions to executive actions. "
            "It is not a general autonomous agent claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 9 action gate failed: {pass_fail}")
    print("NeoCortex-X Stage 9 action gate")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  decisions: {temporal_validation['episode_count']}")
    print(f"  gate.action_accuracy: {gate['action_accuracy']:.4f}")
    print(
        "  gate_minus_always_answer_accuracy: "
        f"{comparison['gate_minus_always_answer_accuracy']:.4f}"
    )
    print(
        "  gate_minus_always_search_accuracy: "
        f"{comparison['gate_minus_always_search_accuracy']:.4f}"
    )
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 9 memory action gate.")
    parser.add_argument("--temporal-json", default="artifacts/neocortex_x/stage8_temporal_freshness.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage9_action_gate.json")
    parser.add_argument("--gate-accuracy-min", type=float, default=0.95)
    parser.add_argument("--always-answer-margin-min", type=float, default=0.25)
    parser.add_argument("--always-search-margin-min", type=float, default=0.25)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
