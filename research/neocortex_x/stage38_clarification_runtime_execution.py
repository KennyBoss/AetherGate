#!/usr/bin/env python3
"""Stage 38 runtime execution traces for NeoCortex-X clarification policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS


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


def validate_runtime_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if artifact.get("kind") != "neocortex_x_stage37_clarification_runtime_loader":
        errors.append("invalid stage37 runtime kind")
    if artifact.get("pass") is not True:
        errors.append("stage37 runtime artifact did not pass")
    runtime_policy = artifact.get("runtime_policy", {})
    ranked_keys = runtime_policy.get("ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("runtime ranked_keys missing")
        ranked_keys = []
    if set(ranked_keys) != set(CONTEXT_KEYS):
        errors.append("runtime ranked_keys do not cover CONTEXT_KEYS")
    top_key = runtime_policy.get("top_key")
    if top_key not in set(ranked_keys):
        errors.append("runtime top_key missing from ranked_keys")
    if runtime_policy.get("unknown_context_fallback") != "ask_context_question":
        errors.append("runtime unknown_context_fallback is not ask_context_question")
    return {
        "valid": not errors,
        "errors": errors,
        "runtime_policy_id": runtime_policy.get("runtime_policy_id"),
        "policy_id": runtime_policy.get("policy_id"),
        "ranked_keys": ranked_keys,
        "top_key": top_key,
    }


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
        "target_value",
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
    return {"valid": not errors, "errors": errors, "query_count": len(queries)}


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


def select_runtime_key(runtime_policy: dict[str, Any], current_context: dict[str, str]) -> str | None:
    for key in runtime_policy["ranked_keys"]:
        if key not in current_context:
            return key
    return None


def output_for_result(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    current_context: dict[str, str],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    target_kept = target_present(candidates, query["target_slot_id"])
    if len(candidates) == 1 and target_kept:
        return {
            "type": "answer",
            "slot_id": candidates[0]["slot_id"],
            "value": query["target_value"],
            "context": current_context,
            "clarification_steps": len(steps),
            "needs_more_context": False,
        }
    if target_kept:
        return {
            "type": "clarification_incomplete",
            "slot_id": None,
            "value": None,
            "context": current_context,
            "candidate_count": len(candidates),
            "clarification_steps": len(steps),
            "needs_more_context": True,
        }
    return {
        "type": "target_lost",
        "slot_id": None,
        "value": None,
        "context": current_context,
        "candidate_count": len(candidates),
        "clarification_steps": len(steps),
        "needs_more_context": True,
    }


def execute_query(query: dict[str, Any], runtime_policy: dict[str, Any]) -> dict[str, Any]:
    current_context = dict(query["partial_context"])
    candidates = filter_candidates(candidate_pairs(query), current_context)
    initial_count = len(candidates)
    steps = []
    while len(candidates) > 1 and len(current_context) < len(CONTEXT_KEYS):
        key = select_runtime_key(runtime_policy, current_context)
        if not key:
            break
        answer = query["full_context"].get(key)
        if answer is None:
            break
        before_count = len(candidates)
        current_context[key] = answer
        candidates = filter_candidates(candidates, {key: answer})
        after_count = len(candidates)
        steps.append(
            {
                "action": "ask_disambiguating_context",
                "asked_key": key,
                "answer_value": answer,
                "before_candidate_count": before_count,
                "after_candidate_count": after_count,
                "reduced_candidates": after_count < before_count,
                "target_preserved": target_present(candidates, query["target_slot_id"]),
                "selected_from_ranked_keys": key in runtime_policy["ranked_keys"],
            }
        )
    output = output_for_result(query, candidates, current_context, steps)
    target_kept = target_present(candidates, query["target_slot_id"])
    resolved_unique = len(candidates) == 1 and target_kept
    unnecessary_clarification = initial_count <= 1 and bool(steps)
    active_compute = {
        "policy_reads": max(1, len(steps)),
        "context_reads": len(query["partial_context"]) + len(steps),
        "candidate_filters": 1 + len(steps),
        "memory_reads": 1,
        "context_questions": len(steps),
        "answers_emitted": 1 if output["type"] == "answer" else 0,
        "total_ticks": 2 + (2 * len(steps)) + (1 if output["type"] == "answer" else 0),
    }
    return {
        "query_id": query["query_id"],
        "subject": query["subject"],
        "property": query["property"],
        "target_slot_id": query["target_slot_id"],
        "runtime_policy_id": runtime_policy["runtime_policy_id"],
        "initial_context": query["partial_context"],
        "final_context": current_context,
        "initial_candidate_count": initial_count,
        "final_candidate_count": len(candidates),
        "initial_ambiguous": initial_count > 1,
        "steps": steps,
        "resolved_unique": resolved_unique,
        "target_preserved": target_kept,
        "unnecessary_clarification": unnecessary_clarification,
        "selected_keys": [step["asked_key"] for step in steps],
        "output": output,
        "success": resolved_unique,
        "active_compute": active_compute,
    }


def execute_runtime(queries: list[dict[str, Any]], runtime_policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [execute_query(query, runtime_policy) for query in queries]


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime_artifact = load_json(Path(args.runtime_json))
    stage30 = load_json(Path(args.partial_context_json))
    runtime_validation = validate_runtime_artifact(runtime_artifact)
    stage30_validation = validate_stage30_artifact(stage30)
    runtime_policy = runtime_artifact.get("runtime_policy", {})
    queries = stage30.get("probe", {}).get("evaluated_queries", [])
    traces = execute_runtime(queries, runtime_policy)
    replay_traces = execute_runtime(queries, runtime_policy)
    deterministic_replay = traces == replay_traces
    ambiguous = [trace for trace in traces if trace["initial_ambiguous"]]
    unique = [trace for trace in traces if not trace["initial_ambiguous"]]
    all_steps = [step for trace in traces for step in trace["steps"]]
    output_types = sorted({trace["output"]["type"] for trace in traces})
    execution_success = mean([1.0 if trace["success"] else 0.0 for trace in traces])
    ambiguous_success = mean([1.0 if trace["success"] else 0.0 for trace in ambiguous])
    target_preservation = mean([1.0 if trace["target_preserved"] else 0.0 for trace in traces])
    active_compute_recorded = all(trace.get("active_compute") for trace in traces)
    selected_keys_from_ranked = all(
        step["selected_from_ranked_keys"] for step in all_steps
    )
    no_unnecessary_unique_clarification = not any(
        trace["unnecessary_clarification"] for trace in unique
    )
    context_questions_present = bool(all_steps)
    pass_fail = {
        "runtime_artifact_valid": runtime_validation["valid"],
        "stage30_artifact_valid": stage30_validation["valid"],
        "ambiguous_queries_present": bool(ambiguous),
        "unique_queries_present": bool(unique),
        "execution_success": execution_success >= args.execution_success_min,
        "ambiguous_execution_success": ambiguous_success >= args.execution_success_min,
        "target_preservation_rate": target_preservation == 1.0,
        "trace_count_matches_queries": len(traces) == stage30_validation["query_count"],
        "max_steps_bounded": max([len(trace["steps"]) for trace in traces], default=0) <= len(CONTEXT_KEYS),
        "no_unnecessary_unique_clarification": no_unnecessary_unique_clarification,
        "selected_keys_from_ranked_list": selected_keys_from_ranked,
        "context_questions_present": context_questions_present,
        "answer_outputs_present": output_types == ["answer"],
        "active_compute_recorded": active_compute_recorded,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "runtime_json": args.runtime_json,
        "partial_context_json": args.partial_context_json,
        "runtime_policy_id": runtime_validation["runtime_policy_id"],
        "policy_id": runtime_validation["policy_id"],
        "trace_count": len(traces),
        "ambiguous_count": len(ambiguous),
        "unique_count": len(unique),
        "output_types": output_types,
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage38_clarification_runtime_execution",
        "config": vars(args),
        "runtime_validation": runtime_validation,
        "stage30_validation": stage30_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "execution_success": execution_success,
            "ambiguous_execution_success": ambiguous_success,
            "target_preservation_rate": target_preservation,
            "trace_count": len(traces),
            "mean_steps_for_ambiguous": mean([float(len(trace["steps"])) for trace in ambiguous]),
            "max_steps": max([len(trace["steps"]) for trace in traces], default=0),
            "total_ticks": sum(int(trace["active_compute"]["total_ticks"]) for trace in traces),
            "context_question_count": sum(len(trace["steps"]) for trace in traces),
        },
        "traces": traces,
        "claim_guardrail": (
            "Stage 38 executes a promoted clarification runtime policy over "
            "cached partial-context episodes. It is not a live production "
            "deployment or a primary-memory storage claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 38 clarification runtime execution failed: {pass_fail}")
    print("NeoCortex-X Stage 38 clarification runtime execution")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  runtime_policy_id: {runtime_validation['runtime_policy_id']}")
    print(f"  execution_success: {execution_success:.4f}")
    print(f"  ambiguous_execution_success: {ambiguous_success:.4f}")
    print(f"  traces: {len(traces)}")
    print(f"  output_types: {','.join(output_types)}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a NeoCortex-X clarification runtime policy.")
    parser.add_argument(
        "--runtime-json",
        default="artifacts/neocortex_x/stage37_clarification_runtime_loader.json",
    )
    parser.add_argument(
        "--partial-context-json",
        default="artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage38_clarification_runtime_execution.json",
    )
    parser.add_argument("--execution-success-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
