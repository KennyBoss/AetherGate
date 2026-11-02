#!/usr/bin/env python3
"""Stage 32 clarification-resolution loop for NeoCortex-X contextual memory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS
from stage31_partial_context_disambiguation_gate import choose_disambiguating_key


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
    queries = artifact.get("probe", {}).get("evaluated_queries")
    if not isinstance(queries, list) or not queries:
        errors.append("stage30 evaluated_queries missing")
        queries = []
    required = {
        "query_id",
        "target_slot_id",
        "subject",
        "property",
        "partial_context",
        "full_context",
        "candidate_count",
        "candidate_ids",
        "candidate_contexts",
        "target_in_candidates",
    }
    for index, query in enumerate(queries):
        missing = required.difference(query)
        if missing:
            errors.append(f"stage30 query {index} missing {sorted(missing)}")
    return {"valid": not errors, "errors": errors, "query_count": len(queries)}


def validate_stage31_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if artifact.get("kind") != "neocortex_x_stage31_partial_context_disambiguation_gate":
        errors.append("invalid stage31 kind")
    if artifact.get("pass") is not True:
        errors.append("stage31 artifact did not pass")
    decisions = (
        artifact.get("runs", {})
        .get("partial_context_disambiguation_gate", {})
        .get("decisions")
    )
    if not isinstance(decisions, list) or not decisions:
        errors.append("stage31 gate decisions missing")
        decisions = []
    required = {"query_id", "agent_action", "payload", "ambiguous"}
    for index, decision in enumerate(decisions):
        missing = required.difference(decision)
        if missing:
            errors.append(f"stage31 decision {index} missing {sorted(missing)}")
    return {"valid": not errors, "errors": errors, "decision_count": len(decisions)}


def candidate_pairs(query: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"slot_id": slot_id, "context": context}
        for slot_id, context in zip(query["candidate_ids"], query["candidate_contexts"])
    ]


def target_present(candidates: list[dict[str, Any]], target_slot_id: str) -> bool:
    return any(candidate["slot_id"] == target_slot_id for candidate in candidates)


def filter_candidates(
    candidates: list[dict[str, Any]],
    context: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if all(candidate["context"].get(key) == value for key, value in context.items())
    ]


def make_query_view(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    current_context: dict[str, str],
) -> dict[str, Any]:
    return {
        "partial_context": current_context,
        "candidate_contexts": [candidate["context"] for candidate in candidates],
    }


def resolve_query(query: dict[str, Any], gate_decision: dict[str, Any]) -> dict[str, Any]:
    current_context = dict(query["partial_context"])
    candidates = filter_candidates(candidate_pairs(query), current_context)
    initial_count = len(candidates)
    steps = []
    first_gate_key = gate_decision.get("payload", {}).get("disambiguation_key")
    ambiguous = initial_count > 1
    if not ambiguous:
        return {
            "query_id": query["query_id"],
            "target_slot_id": query["target_slot_id"],
            "initial_candidate_count": initial_count,
            "final_candidate_count": initial_count,
            "steps": steps,
            "resolved_unique": initial_count == 1 and target_present(candidates, query["target_slot_id"]),
            "target_preserved": target_present(candidates, query["target_slot_id"]),
            "first_step_matches_stage31_key": True,
            "unnecessary_clarification": False,
            "final_context": current_context,
            "resolved_slot_id": candidates[0]["slot_id"] if len(candidates) == 1 else None,
        }

    used_keys = set(current_context)
    first_step_matches_stage31_key = True
    while len(candidates) > 1 and len(used_keys) < len(CONTEXT_KEYS):
        if not steps and first_gate_key:
            key = first_gate_key
        else:
            key = choose_disambiguating_key(make_query_view(query, candidates, current_context))
        if not key:
            break
        if key not in query["full_context"]:
            break
        if key in current_context:
            used_keys.add(key)
            continue
        before_count = len(candidates)
        answer = query["full_context"][key]
        current_context[key] = answer
        used_keys.add(key)
        candidates = filter_candidates(candidates, {key: answer})
        after_count = len(candidates)
        if not steps:
            first_step_matches_stage31_key = key == first_gate_key
        steps.append(
            {
                "asked_key": key,
                "answer_value": answer,
                "before_candidate_count": before_count,
                "after_candidate_count": after_count,
                "reduced_candidates": after_count < before_count,
                "target_preserved": target_present(candidates, query["target_slot_id"]),
            }
        )
    return {
        "query_id": query["query_id"],
        "target_slot_id": query["target_slot_id"],
        "initial_candidate_count": initial_count,
        "final_candidate_count": len(candidates),
        "steps": steps,
        "resolved_unique": len(candidates) == 1 and target_present(candidates, query["target_slot_id"]),
        "target_preserved": target_present(candidates, query["target_slot_id"]),
        "first_step_matches_stage31_key": first_step_matches_stage31_key,
        "unnecessary_clarification": False,
        "final_context": current_context,
        "resolved_slot_id": candidates[0]["slot_id"] if len(candidates) == 1 else None,
    }


def run_resolution(
    queries: list[dict[str, Any]],
    gate_decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    episodes = []
    for query in queries:
        decision = gate_decisions.get(query["query_id"], {})
        episodes.append(resolve_query(query, decision))
    ambiguous = [episode for episode in episodes if episode["initial_candidate_count"] > 1]
    unique = [episode for episode in episodes if episode["initial_candidate_count"] == 1]
    all_steps = [step for episode in episodes for step in episode["steps"]]
    return {
        "episode_count": len(episodes),
        "ambiguous_count": len(ambiguous),
        "unique_count": len(unique),
        "resolution_accuracy": mean([1.0 if item["resolved_unique"] else 0.0 for item in episodes]),
        "ambiguous_resolution_accuracy": mean(
            [1.0 if item["resolved_unique"] else 0.0 for item in ambiguous]
        ),
        "target_preservation_rate": mean([1.0 if item["target_preserved"] else 0.0 for item in episodes]),
        "first_step_stage31_key_match_rate": mean(
            [1.0 if item["first_step_matches_stage31_key"] else 0.0 for item in ambiguous]
        ),
        "candidate_reduction_step_rate": mean(
            [1.0 if step["reduced_candidates"] else 0.0 for step in all_steps]
        ),
        "max_steps": max([len(item["steps"]) for item in episodes], default=0),
        "mean_steps_for_ambiguous": mean([float(len(item["steps"])) for item in ambiguous]),
        "episodes": episodes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage30 = load_json(Path(args.partial_context_json))
    stage31 = load_json(Path(args.disambiguation_json))
    stage30_validation = validate_stage30_artifact(stage30)
    stage31_validation = validate_stage31_artifact(stage31)
    queries = stage30.get("probe", {}).get("evaluated_queries", [])
    decisions = (
        stage31.get("runs", {})
        .get("partial_context_disambiguation_gate", {})
        .get("decisions", [])
    )
    decision_by_query = {decision["query_id"]: decision for decision in decisions}
    result = run_resolution(queries, decision_by_query)
    replay_result = run_resolution(queries, decision_by_query)
    deterministic_replay = result["episodes"] == replay_result["episodes"]
    pass_fail = {
        "stage30_artifact_valid": stage30_validation["valid"],
        "stage31_artifact_valid": stage31_validation["valid"],
        "query_decision_count_matches": stage30_validation["query_count"] == stage31_validation["decision_count"],
        "ambiguous_queries_present": result["ambiguous_count"] > 0,
        "unique_queries_present": result["unique_count"] > 0,
        "resolution_accuracy": result["resolution_accuracy"] >= args.resolution_accuracy_min,
        "ambiguous_resolution_accuracy": (
            result["ambiguous_resolution_accuracy"] >= args.resolution_accuracy_min
        ),
        "target_preservation_rate": result["target_preservation_rate"] == 1.0,
        "first_step_matches_stage31_key": result["first_step_stage31_key_match_rate"] == 1.0,
        "candidate_reduction_steps": result["candidate_reduction_step_rate"] == 1.0,
        "bounded_resolution_steps": result["max_steps"] <= len(CONTEXT_KEYS),
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "partial_context_json": args.partial_context_json,
        "disambiguation_json": args.disambiguation_json,
        "episode_count": result["episode_count"],
        "ambiguous_count": result["ambiguous_count"],
        "unique_count": result["unique_count"],
        "resolution_accuracy": result["resolution_accuracy"],
        "ambiguous_resolution_accuracy": result["ambiguous_resolution_accuracy"],
        "max_steps": result["max_steps"],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage32_clarification_resolution_loop",
        "config": vars(args),
        "stage30_validation": stage30_validation,
        "stage31_validation": stage31_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "resolution": result,
        "claim_guardrail": (
            "Stage 32 simulates bounded clarification answers and verifies "
            "candidate-set resolution. It is not an open-ended dialogue system."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 32 clarification-resolution loop failed: {pass_fail}")
    print("NeoCortex-X Stage 32 clarification-resolution loop")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  episodes: {result['episode_count']}")
    print(f"  ambiguous: {result['ambiguous_count']}")
    print(f"  unique: {result['unique_count']}")
    print(f"  resolution_accuracy: {result['resolution_accuracy']:.4f}")
    print(f"  mean_steps_for_ambiguous: {result['mean_steps_for_ambiguous']:.4f}")
    print(f"  max_steps: {result['max_steps']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X clarification-resolution loop.")
    parser.add_argument(
        "--partial-context-json",
        default="artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json",
    )
    parser.add_argument(
        "--disambiguation-json",
        default="artifacts/neocortex_x/stage31_partial_context_disambiguation_gate.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage32_clarification_resolution_loop.json",
    )
    parser.add_argument("--resolution-accuracy-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
