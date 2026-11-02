#!/usr/bin/env python3
"""Stage 35 clarification-policy replay for NeoCortex-X memory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS


AGENTS = (
    "policy_guided_replay",
    "fixed_order_control",
    "static_entropy_control",
    "always_top_key_control",
)


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
            continue
        if len(query["candidate_ids"]) != len(query["candidate_contexts"]):
            errors.append(f"stage30 query {index} candidate ids/contexts length mismatch")
        if int(query["candidate_count"]) != len(query["candidate_ids"]):
            errors.append(f"stage30 query {index} candidate count mismatch")
        if not query.get("target_in_candidates"):
            errors.append(f"stage30 query {index} target missing from candidates")
        for context_name in ("partial_context", "full_context"):
            if not isinstance(query[context_name], dict):
                errors.append(f"stage30 query {index} {context_name} must be dict")
        for ctx_index, context in enumerate(query["candidate_contexts"]):
            if not isinstance(context, dict):
                errors.append(f"stage30 query {index} candidate context {ctx_index} must be dict")
                continue
            missing_keys = set(CONTEXT_KEYS).difference(context)
            if missing_keys:
                errors.append(
                    f"stage30 query {index} candidate context {ctx_index} "
                    f"missing {sorted(missing_keys)}"
                )
    return {"valid": not errors, "errors": errors, "query_count": len(queries)}


def validate_policy_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if artifact.get("kind") != "neocortex_x_stage34_clarification_policy_update":
        errors.append("invalid stage34 policy kind")
    if artifact.get("pass") is not True:
        errors.append("stage34 policy artifact did not pass")
    policy = artifact.get("policy", {})
    ranked_keys = policy.get("ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("policy ranked_keys missing")
        ranked_keys = []
    probabilities = policy.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        errors.append("policy probabilities missing")
        probabilities = {}
    ranked_set = set(ranked_keys)
    unknown_keys = ranked_set.difference(CONTEXT_KEYS)
    missing_keys = set(CONTEXT_KEYS).difference(ranked_set)
    if unknown_keys:
        errors.append(f"policy ranked_keys contain unknown keys {sorted(unknown_keys)}")
    if missing_keys:
        errors.append(f"policy ranked_keys missing {sorted(missing_keys)}")
    probability_sum = sum(float(probabilities.get(key, 0.0)) for key in CONTEXT_KEYS)
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("policy probabilities do not sum to 1.0")
    for key in CONTEXT_KEYS:
        if key not in probabilities:
            errors.append(f"policy probabilities missing {key}")
    top_key = policy.get("top_key")
    if top_key not in ranked_set:
        errors.append("policy top_key missing from ranked_keys")
    return {
        "valid": not errors,
        "errors": errors,
        "policy_id": policy.get("policy_id"),
        "ranked_keys": ranked_keys,
        "top_key": top_key,
        "probability_sum": probability_sum,
    }


def candidate_pairs(query: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"slot_id": slot_id, "context": context}
        for slot_id, context in zip(query["candidate_ids"], query["candidate_contexts"])
    ]


def filter_candidates(
    candidates: list[dict[str, Any]],
    context: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if all(candidate["context"].get(key) == value for key, value in context.items())
    ]


def target_present(candidates: list[dict[str, Any]], target_slot_id: str) -> bool:
    return any(candidate["slot_id"] == target_slot_id for candidate in candidates)


def entropy_score(candidates: list[dict[str, Any]], key: str) -> int:
    return len({candidate["context"].get(key, "<missing>") for candidate in candidates})


def choose_key(
    *,
    agent: str,
    candidates: list[dict[str, Any]],
    current_context: dict[str, str],
    query: dict[str, Any],
    ranked_keys: list[str],
    top_key: str | None,
) -> str | None:
    if agent == "policy_guided_replay":
        for key in ranked_keys:
            if key not in current_context and key in query["full_context"]:
                return key
        return None
    if agent == "fixed_order_control":
        for key in CONTEXT_KEYS:
            if key not in current_context and key in query["full_context"]:
                return key
        return None
    if agent == "static_entropy_control":
        best_key = None
        best_score = 0
        for key in CONTEXT_KEYS:
            if key in current_context:
                continue
            score = entropy_score(candidates, key)
            if score > best_score:
                best_key = key
                best_score = score
        return best_key
    if agent == "always_top_key_control":
        if top_key and top_key not in current_context and top_key in query["full_context"]:
            return top_key
        return None
    raise ValueError(f"Unknown Stage 35 agent: {agent}")


def resolve_query(
    query: dict[str, Any],
    *,
    agent: str,
    ranked_keys: list[str],
    top_key: str | None,
) -> dict[str, Any]:
    current_context = dict(query["partial_context"])
    candidates = filter_candidates(candidate_pairs(query), current_context)
    initial_count = len(candidates)
    steps = []
    while len(candidates) > 1 and len(current_context) < len(CONTEXT_KEYS):
        key = choose_key(
            agent=agent,
            candidates=candidates,
            current_context=current_context,
            query=query,
            ranked_keys=ranked_keys,
            top_key=top_key,
        )
        if not key:
            break
        before_count = len(candidates)
        answer = query["full_context"].get(key)
        if answer is None:
            break
        current_context[key] = answer
        candidates = filter_candidates(candidates, {key: answer})
        after_count = len(candidates)
        policy_rank = ranked_keys.index(key) + 1 if key in ranked_keys else None
        steps.append(
            {
                "asked_key": key,
                "answer_value": answer,
                "policy_rank": policy_rank,
                "before_candidate_count": before_count,
                "after_candidate_count": after_count,
                "reduced_candidates": after_count < before_count,
                "target_preserved": target_present(candidates, query["target_slot_id"]),
                "selected_from_policy_ranked_keys": key in ranked_keys,
            }
        )
    target_kept = target_present(candidates, query["target_slot_id"])
    return {
        "query_id": query["query_id"],
        "target_slot_id": query["target_slot_id"],
        "subject": query["subject"],
        "property": query["property"],
        "initial_context": query["partial_context"],
        "final_context": current_context,
        "initial_candidate_count": initial_count,
        "final_candidate_count": len(candidates),
        "initial_ambiguous": initial_count > 1,
        "steps": steps,
        "resolved_unique": len(candidates) == 1 and target_kept,
        "target_preserved": target_kept,
        "unnecessary_clarification": initial_count <= 1 and bool(steps),
        "resolved_slot_id": candidates[0]["slot_id"] if len(candidates) == 1 else None,
    }


def run_agent(
    queries: list[dict[str, Any]],
    *,
    agent: str,
    ranked_keys: list[str],
    top_key: str | None,
) -> dict[str, Any]:
    episodes = [
        resolve_query(query, agent=agent, ranked_keys=ranked_keys, top_key=top_key)
        for query in queries
    ]
    ambiguous = [episode for episode in episodes if episode["initial_ambiguous"]]
    unique = [episode for episode in episodes if not episode["initial_ambiguous"]]
    all_steps = [step for episode in episodes for step in episode["steps"]]
    selected_keys = sorted({step["asked_key"] for step in all_steps})
    return {
        "agent": agent,
        "episode_count": len(episodes),
        "ambiguous_count": len(ambiguous),
        "unique_count": len(unique),
        "resolution_accuracy": mean([1.0 if item["resolved_unique"] else 0.0 for item in episodes]),
        "ambiguous_resolution_accuracy": mean(
            [1.0 if item["resolved_unique"] else 0.0 for item in ambiguous]
        ),
        "target_preservation_rate": mean(
            [1.0 if item["target_preserved"] else 0.0 for item in episodes]
        ),
        "unnecessary_clarification_rate": mean(
            [1.0 if item["unnecessary_clarification"] else 0.0 for item in unique]
        ),
        "candidate_reduction_step_rate": mean(
            [1.0 if step["reduced_candidates"] else 0.0 for step in all_steps]
        ),
        "selected_keys_from_policy_ranked_list": all(
            step["selected_from_policy_ranked_keys"] for step in all_steps
        ),
        "selected_keys": selected_keys,
        "max_steps": max([len(item["steps"]) for item in episodes], default=0),
        "mean_steps_for_ambiguous": mean([float(len(item["steps"])) for item in ambiguous]),
        "episodes": episodes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage30 = load_json(Path(args.partial_context_json))
    stage34 = load_json(Path(args.policy_json))
    stage30_validation = validate_stage30_artifact(stage30)
    policy_validation = validate_policy_artifact(stage34)
    queries = stage30.get("probe", {}).get("evaluated_queries", [])
    ranked_keys = list(policy_validation["ranked_keys"])
    top_key = policy_validation["top_key"]
    runs = {
        agent: run_agent(queries, agent=agent, ranked_keys=ranked_keys, top_key=top_key)
        for agent in AGENTS
    }
    replay_runs = {
        agent: run_agent(queries, agent=agent, ranked_keys=ranked_keys, top_key=top_key)
        for agent in AGENTS
    }
    deterministic_replay = runs == replay_runs
    policy = runs["policy_guided_replay"]
    fixed = runs["fixed_order_control"]
    static_entropy = runs["static_entropy_control"]
    always_top = runs["always_top_key_control"]
    comparison = {
        "policy_minus_fixed_accuracy": (
            policy["resolution_accuracy"] - fixed["resolution_accuracy"]
        ),
        "policy_minus_always_top_accuracy": (
            policy["resolution_accuracy"] - always_top["resolution_accuracy"]
        ),
        "fixed_minus_policy_mean_ambiguous_steps": (
            fixed["mean_steps_for_ambiguous"] - policy["mean_steps_for_ambiguous"]
        ),
        "fixed_minus_policy_max_steps": fixed["max_steps"] - policy["max_steps"],
        "static_entropy_minus_policy_mean_ambiguous_steps": (
            static_entropy["mean_steps_for_ambiguous"] - policy["mean_steps_for_ambiguous"]
        ),
    }
    protocol_evidence = {
        "valid": bool(
            stage30_validation["valid"]
            and policy_validation["valid"]
            and queries
            and deterministic_replay
        ),
        "partial_context_json": args.partial_context_json,
        "policy_json": args.policy_json,
        "query_count": stage30_validation["query_count"],
        "policy_id": policy_validation["policy_id"],
        "ranked_keys": ranked_keys,
        "top_key": top_key,
        "deterministic_replay": deterministic_replay,
    }
    pass_fail = {
        "protocol_evidence_valid": protocol_evidence["valid"],
        "stage30_artifact_valid": stage30_validation["valid"],
        "policy_artifact_valid": policy_validation["valid"],
        "ambiguous_queries_present": policy["ambiguous_count"] > 0,
        "unique_queries_present": policy["unique_count"] > 0,
        "policy_resolution_accuracy": (
            policy["resolution_accuracy"] >= args.resolution_accuracy_min
        ),
        "policy_ambiguous_resolution_accuracy": (
            policy["ambiguous_resolution_accuracy"] >= args.resolution_accuracy_min
        ),
        "policy_target_preservation_rate": policy["target_preservation_rate"] == 1.0,
        "policy_bounded_steps": policy["max_steps"] <= len(CONTEXT_KEYS),
        "policy_no_unnecessary_unique_clarification": (
            policy["unnecessary_clarification_rate"] == 0.0
        ),
        "policy_selected_keys_present": bool(policy["selected_keys"]),
        "policy_selected_keys_from_ranked_list": policy["selected_keys_from_policy_ranked_list"],
        "policy_step_efficiency_vs_fixed": (
            comparison["fixed_minus_policy_mean_ambiguous_steps"]
            >= args.step_gain_vs_fixed_min
        ),
        "policy_beats_always_top_accuracy": (
            comparison["policy_minus_always_top_accuracy"]
            >= args.always_top_accuracy_margin_min
        ),
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage35_clarification_policy_replay",
        "config": vars(args),
        "stage30_validation": stage30_validation,
        "policy_validation": policy_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": {
            "agents": {
                agent: {key: value for key, value in run.items() if key != "episodes"}
                for agent, run in runs.items()
            },
            "comparison": comparison,
        },
        "runs": runs,
        "claim_guardrail": (
            "Stage 35 replays a deterministic clarification-key policy over "
            "partial-context candidate sets. JSON is an artifact/report format "
            "here, not the primary memory store."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 35 clarification-policy replay failed: {pass_fail}")
    print("NeoCortex-X Stage 35 clarification-policy replay")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  policy_id: {policy_validation['policy_id']}")
    print(f"  ranked_keys: {','.join(ranked_keys)}")
    print(f"  policy.resolution_accuracy: {policy['resolution_accuracy']:.4f}")
    print(f"  policy.ambiguous_resolution_accuracy: {policy['ambiguous_resolution_accuracy']:.4f}")
    print(f"  policy.mean_steps_for_ambiguous: {policy['mean_steps_for_ambiguous']:.4f}")
    print(
        "  fixed_minus_policy_mean_ambiguous_steps: "
        f"{comparison['fixed_minus_policy_mean_ambiguous_steps']:.4f}"
    )
    print(
        "  policy_minus_always_top_accuracy: "
        f"{comparison['policy_minus_always_top_accuracy']:.4f}"
    )
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 35 clarification-policy replay.")
    parser.add_argument(
        "--partial-context-json",
        default="artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json",
    )
    parser.add_argument(
        "--policy-json",
        default="artifacts/neocortex_x/stage34_clarification_policy_update.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage35_clarification_policy_replay.json",
    )
    parser.add_argument("--resolution-accuracy-min", type=float, default=0.95)
    parser.add_argument("--step-gain-vs-fixed-min", type=float, default=0.25)
    parser.add_argument("--always-top-accuracy-margin-min", type=float, default=0.10)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
