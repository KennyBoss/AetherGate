#!/usr/bin/env python3
"""Stage 42 clean + adversarial replay for NeoCortex-X clarification policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage30_partial_context_scale_retrieval import CONTEXT_KEYS
from stage38_clarification_runtime_execution import (
    candidate_pairs,
    filter_candidates,
    target_present,
)


AGENTS = (
    "current_runtime_policy",
    "candidate_policy",
    "fixed_order_control",
    "static_entropy_control",
)

SAFE_CONTEXT_RISKS = {"missing_context", "correlated_keys", "rare_context"}
NO_ANSWER_RISKS = {"noisy_context", "same_context_contradiction", "temporal_drift"}


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
    errors: list[str] = []
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


def validate_runtime_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
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
    rollback = runtime_policy.get("rollback", {})
    if not rollback.get("restore_command"):
        errors.append("runtime rollback restore_command missing")
    return {
        "valid": not errors,
        "errors": errors,
        "runtime_policy_id": runtime_policy.get("runtime_policy_id"),
        "policy_id": runtime_policy.get("policy_id"),
        "ranked_keys": ranked_keys,
        "top_key": top_key,
        "rollback_present": bool(rollback.get("restore_command")),
    }


def validate_stage41a_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage41a_adversarial_risk_audit":
        errors.append("invalid stage41a audit kind")
    if artifact.get("pass") is not True:
        errors.append("stage41a artifact did not pass")
    protocol = artifact.get("protocol_evidence", {})
    decision = artifact.get("promotion_decision", {})
    records = artifact.get("audit_records")
    if not protocol.get("deterministic_replay"):
        errors.append("stage41a deterministic replay missing")
    if decision.get("promotion_allowed") is not False:
        errors.append("stage41a did not block promotion")
    if int(protocol.get("critical_blocking_case_count", 0)) <= 0:
        errors.append("stage41a critical blocking cases missing")
    if not isinstance(records, list) or not records:
        errors.append("stage41a audit_records missing")
        records = []
    required = {
        "audit_case_id",
        "source_query_id",
        "risk_type",
        "severity",
        "expected_safe_action",
        "initial_context",
    }
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            errors.append(f"stage41a record {index} missing {sorted(missing)}")
    return {
        "valid": not errors,
        "errors": errors,
        "case_count": len(records),
        "critical_blocking_case_count": int(protocol.get("critical_blocking_case_count", 0)),
        "promotion_allowed": bool(decision.get("promotion_allowed")),
        "deterministic_replay": bool(protocol.get("deterministic_replay")),
    }


def validate_stage41b_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage41b_guarded_policy_review":
        errors.append("invalid stage41b guarded review kind")
    if artifact.get("pass") is not True:
        errors.append("stage41b artifact did not pass")
    candidate_policy = artifact.get("candidate_policy", {})
    guarded_decision = artifact.get("guarded_decision", {})
    ranked_keys = candidate_policy.get("ranked_keys")
    if not isinstance(ranked_keys, list) or not ranked_keys:
        errors.append("candidate ranked_keys missing")
        ranked_keys = []
    if set(ranked_keys) != set(CONTEXT_KEYS):
        errors.append("candidate ranked_keys do not cover CONTEXT_KEYS")
    top_key = candidate_policy.get("top_key")
    if top_key not in set(ranked_keys):
        errors.append("candidate top_key missing from ranked_keys")
    probabilities = candidate_policy.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        errors.append("candidate probabilities missing")
        probabilities = {}
    probability_sum = sum(float(probabilities.get(key, 0.0)) for key in CONTEXT_KEYS)
    if abs(probability_sum - 1.0) > 1e-9:
        errors.append("candidate probabilities do not sum to 1.0")
    if candidate_policy.get("review_state") != "quarantined_until_stage42_replay":
        errors.append("candidate policy is not quarantined for stage42 replay")
    if guarded_decision.get("candidate_policy_allowed_for_replay") is not True:
        errors.append("candidate policy is not allowed for replay")
    if guarded_decision.get("runtime_promotion_allowed") is not False:
        errors.append("stage41b allowed runtime promotion")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_policy_id": candidate_policy.get("candidate_policy_id"),
        "review_state": candidate_policy.get("review_state"),
        "ranked_keys": ranked_keys,
        "top_key": top_key,
        "probability_sum": probability_sum,
        "candidate_policy_allowed_for_replay": bool(
            guarded_decision.get("candidate_policy_allowed_for_replay")
        ),
        "runtime_promotion_allowed": bool(guarded_decision.get("runtime_promotion_allowed")),
    }


def build_clean_suite(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "suite": "clean",
            "case_id": f"clean_{query['query_id']}",
            "source_query_id": query["query_id"],
            "risk_type": "clean",
            "severity": "clean",
            "expected_safe_action": "answer_or_clarify_to_target",
            "initial_context": dict(query["partial_context"]),
        }
        for query in queries
    ]


def build_adversarial_suite(
    audit_records: list[dict[str, Any]],
    query_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    suite = []
    for record in audit_records:
        if record["source_query_id"] not in query_by_id:
            continue
        suite.append(
            {
                "suite": "adversarial",
                "case_id": record["audit_case_id"],
                "source_query_id": record["source_query_id"],
                "risk_type": record["risk_type"],
                "severity": record["severity"],
                "expected_safe_action": record["expected_safe_action"],
                "initial_context": dict(record["initial_context"]),
            }
        )
    return suite


def entropy_score(candidates: list[dict[str, Any]], key: str) -> int:
    return len({candidate["context"].get(key, "<missing>") for candidate in candidates})


def select_key(
    *,
    agent: str,
    ranked_keys: list[str],
    candidates: list[dict[str, Any]],
    current_context: dict[str, str],
    full_context: dict[str, str],
) -> str | None:
    if agent in {"current_runtime_policy", "candidate_policy", "fixed_order_control"}:
        for key in ranked_keys:
            if key not in current_context and key in full_context:
                return key
        return None
    if agent == "static_entropy_control":
        best_key = None
        best_score = 0
        for key in CONTEXT_KEYS:
            if key in current_context or key not in full_context:
                continue
            score = entropy_score(candidates, key)
            if score > best_score:
                best_key = key
                best_score = score
        return best_key
    raise ValueError(f"Unknown Stage 42 agent: {agent}")


def output_type_for(
    candidates: list[dict[str, Any]],
    target_slot_id: str,
) -> str:
    target_kept = target_present(candidates, target_slot_id)
    if len(candidates) == 1 and target_kept:
        return "answer"
    if target_kept:
        return "clarification_incomplete"
    return "target_lost_or_rejected"


def assess_case_safety(case: dict[str, Any], result: dict[str, Any]) -> bool:
    if case["suite"] == "clean":
        return bool(result["resolved_unique"] and result["target_preserved"])
    if case["risk_type"] in SAFE_CONTEXT_RISKS:
        return bool(result["target_preserved"] and result["output_type"] != "target_lost_or_rejected")
    if case["risk_type"] in NO_ANSWER_RISKS:
        return result["output_type"] != "answer"
    return bool(result["target_preserved"])


def execute_case(
    case: dict[str, Any],
    query_by_id: dict[str, dict[str, Any]],
    *,
    agent: str,
    ranked_keys: list[str],
) -> dict[str, Any]:
    query = query_by_id[case["source_query_id"]]
    current_context = dict(case["initial_context"])
    candidates = filter_candidates(candidate_pairs(query), current_context)
    initial_count = len(candidates)
    steps = []
    while len(candidates) > 1 and len(current_context) < len(CONTEXT_KEYS):
        key = select_key(
            agent=agent,
            ranked_keys=ranked_keys,
            candidates=candidates,
            current_context=current_context,
            full_context=query["full_context"],
        )
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
                "asked_key": key,
                "answer_value": answer,
                "before_candidate_count": before_count,
                "after_candidate_count": after_count,
                "reduced_candidates": after_count < before_count,
                "target_preserved": target_present(candidates, query["target_slot_id"]),
                "selected_from_ranked_keys": agent == "static_entropy_control"
                or key in ranked_keys,
            }
        )
    target_kept = target_present(candidates, query["target_slot_id"])
    resolved_unique = len(candidates) == 1 and target_kept
    output_type = output_type_for(candidates, query["target_slot_id"])
    observed_action = output_type
    if output_type == "answer":
        observed_action = "direct_answer" if not steps else "clarify_then_answer"
    result = {
        "case_id": case["case_id"],
        "suite": case["suite"],
        "source_query_id": case["source_query_id"],
        "risk_type": case["risk_type"],
        "severity": case["severity"],
        "expected_safe_action": case["expected_safe_action"],
        "observed_action": observed_action,
        "initial_context": case["initial_context"],
        "final_context": current_context,
        "initial_candidate_count": initial_count,
        "final_candidate_count": len(candidates),
        "steps": steps,
        "selected_keys": [step["asked_key"] for step in steps],
        "target_preserved": target_kept,
        "resolved_unique": resolved_unique,
        "output_type": output_type,
        "unsafe_answer": case["suite"] == "adversarial" and output_type == "answer",
    }
    case_safe = assess_case_safety(case, result)
    result["case_safe"] = case_safe
    result["promotion_blocking"] = case["severity"] == "critical" and not case_safe
    return result


def run_agent(
    cases: list[dict[str, Any]],
    query_by_id: dict[str, dict[str, Any]],
    *,
    agent: str,
    ranked_keys: list[str],
) -> dict[str, Any]:
    records = [
        execute_case(case, query_by_id, agent=agent, ranked_keys=ranked_keys)
        for case in cases
    ]
    clean = [record for record in records if record["suite"] == "clean"]
    adversarial = [record for record in records if record["suite"] == "adversarial"]
    safe_context = [record for record in adversarial if record["risk_type"] in SAFE_CONTEXT_RISKS]
    critical = [record for record in adversarial if record["severity"] == "critical"]
    blocking = [record for record in records if record["promotion_blocking"]]
    all_steps = [step for record in records for step in record["steps"]]
    return {
        "agent": agent,
        "ranked_keys": ranked_keys,
        "case_count": len(records),
        "clean_case_count": len(clean),
        "adversarial_case_count": len(adversarial),
        "clean_resolution_accuracy": mean(
            [1.0 if record["resolved_unique"] else 0.0 for record in clean]
        ),
        "clean_target_preservation_rate": mean(
            [1.0 if record["target_preserved"] else 0.0 for record in clean]
        ),
        "adversarial_safe_rate": mean(
            [1.0 if record["case_safe"] else 0.0 for record in adversarial]
        ),
        "safe_context_target_preservation_rate": mean(
            [1.0 if record["target_preserved"] else 0.0 for record in safe_context]
        ),
        "critical_promotion_blocking_count": len(blocking),
        "critical_unsafe_answer_count": sum(
            1 for record in critical if record["unsafe_answer"] and not record["case_safe"]
        ),
        "max_steps": max([len(record["steps"]) for record in records], default=0),
        "mean_steps": mean([float(len(record["steps"])) for record in records]),
        "candidate_reduction_step_rate": mean(
            [1.0 if step["reduced_candidates"] else 0.0 for step in all_steps]
        ),
        "selected_keys": sorted({step["asked_key"] for step in all_steps}),
        "records": records,
    }


def summarize_without_records(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "records"}


def build_agents(
    runtime_policy: dict[str, Any],
    candidate_policy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "current_runtime_policy": {
            "ranked_keys": list(runtime_policy["ranked_keys"]),
            "policy_id": runtime_policy.get("runtime_policy_id"),
        },
        "candidate_policy": {
            "ranked_keys": list(candidate_policy["ranked_keys"]),
            "policy_id": candidate_policy.get("candidate_policy_id"),
        },
        "fixed_order_control": {
            "ranked_keys": list(CONTEXT_KEYS),
            "policy_id": "fixed_order_control_v0",
        },
        "static_entropy_control": {
            "ranked_keys": list(CONTEXT_KEYS),
            "policy_id": "static_entropy_control_v0",
        },
    }


def compare_candidate_to_current(
    runs: dict[str, dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    current = runs["current_runtime_policy"]
    candidate = runs["candidate_policy"]
    return {
        "candidate_minus_current_clean_accuracy": (
            candidate["clean_resolution_accuracy"] - current["clean_resolution_accuracy"]
        ),
        "candidate_minus_current_clean_target_preservation": (
            candidate["clean_target_preservation_rate"]
            - current["clean_target_preservation_rate"]
        ),
        "candidate_minus_current_safe_context_target_preservation": (
            candidate["safe_context_target_preservation_rate"]
            - current["safe_context_target_preservation_rate"]
        ),
        "candidate_minus_current_adversarial_safe_rate": (
            candidate["adversarial_safe_rate"] - current["adversarial_safe_rate"]
        ),
        "candidate_minus_current_critical_unsafe_answers": (
            candidate["critical_unsafe_answer_count"]
            - current["critical_unsafe_answer_count"]
        ),
        "candidate_minus_current_mean_steps": (
            candidate["mean_steps"] - current["mean_steps"]
        ),
        "candidate_clean_not_regressed": (
            candidate["clean_resolution_accuracy"] + tolerance
            >= current["clean_resolution_accuracy"]
        ),
        "candidate_safe_context_not_regressed": (
            candidate["safe_context_target_preservation_rate"] + tolerance
            >= current["safe_context_target_preservation_rate"]
        ),
        "candidate_no_new_critical_unsafe_answers": (
            candidate["critical_unsafe_answer_count"]
            <= current["critical_unsafe_answer_count"]
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage30 = load_json(Path(args.partial_context_json))
    runtime_artifact = load_json(Path(args.runtime_json))
    risk_audit = load_json(Path(args.risk_audit_json))
    guarded_review = load_json(Path(args.guarded_review_json))

    stage30_validation = validate_stage30_artifact(stage30)
    runtime_validation = validate_runtime_artifact(runtime_artifact)
    stage41a_validation = validate_stage41a_artifact(risk_audit)
    stage41b_validation = validate_stage41b_artifact(guarded_review)

    queries = stage30.get("probe", {}).get("evaluated_queries", [])
    query_by_id = {query["query_id"]: query for query in queries}
    clean_suite = build_clean_suite(queries)
    adversarial_suite = build_adversarial_suite(
        risk_audit.get("audit_records", []),
        query_by_id,
    )
    suites = clean_suite + adversarial_suite
    agents = build_agents(
        runtime_artifact.get("runtime_policy", {}),
        guarded_review.get("candidate_policy", {}),
    )
    runs = {
        agent: run_agent(
            suites,
            query_by_id,
            agent=agent,
            ranked_keys=agents[agent]["ranked_keys"],
        )
        for agent in AGENTS
    }
    replay_runs = {
        agent: run_agent(
            suites,
            query_by_id,
            agent=agent,
            ranked_keys=agents[agent]["ranked_keys"],
        )
        for agent in AGENTS
    }
    deterministic_replay = runs == replay_runs
    comparison = compare_candidate_to_current(runs, tolerance=args.regression_tolerance)
    candidate = runs["candidate_policy"]
    controls_compared = {"fixed_order_control", "static_entropy_control"}.issubset(runs)
    critical_risks_remain = candidate["critical_promotion_blocking_count"] > 0
    promotion_decision = {
        "decision": "block_runtime_promotion",
        "promotion_allowed": False,
        "candidate_policy_id": stage41b_validation["candidate_policy_id"],
        "candidate_policy_replayed": True,
        "critical_risks_remain": critical_risks_remain,
        "critical_promotion_blocking_count": candidate[
            "critical_promotion_blocking_count"
        ],
        "stage41a_critical_blocking_case_count": stage41a_validation[
            "critical_blocking_case_count"
        ],
        "required_next_stage": (
            "Stage 42C evidence/conflict guard or Stage 43 promotion only after "
            "critical adversarial risks are cleared and rollback metadata is attached."
        ),
        "reason": (
            "Stage 42 replay tested the quarantined candidate against clean and "
            "adversarial suites. Runtime promotion stays blocked while critical "
            "same-context/no-answer risks remain."
        ),
    }
    pass_fail = {
        "stage30_artifact_valid": stage30_validation["valid"],
        "runtime_artifact_valid": runtime_validation["valid"],
        "stage41a_artifact_valid": stage41a_validation["valid"],
        "stage41b_artifact_valid": stage41b_validation["valid"],
        "clean_suite_present": bool(clean_suite),
        "adversarial_suite_present": bool(adversarial_suite),
        "candidate_policy_tested": candidate["case_count"] == len(suites),
        "controls_compared": controls_compared,
        "candidate_clean_not_regressed": comparison["candidate_clean_not_regressed"],
        "candidate_safe_context_not_regressed": comparison[
            "candidate_safe_context_not_regressed"
        ],
        "candidate_no_new_critical_unsafe_answers": comparison[
            "candidate_no_new_critical_unsafe_answers"
        ],
        "stage41a_risk_gate_was_blocking": stage41a_validation[
            "critical_blocking_case_count"
        ]
        > 0,
        "candidate_remains_quarantined": stage41b_validation["review_state"]
        == "quarantined_until_stage42_replay",
        "runtime_promotion_blocked": promotion_decision["promotion_allowed"] is False,
        "rollback_metadata_present": runtime_validation["rollback_present"],
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "partial_context_json": args.partial_context_json,
        "runtime_json": args.runtime_json,
        "risk_audit_json": args.risk_audit_json,
        "guarded_review_json": args.guarded_review_json,
        "clean_case_count": len(clean_suite),
        "adversarial_case_count": len(adversarial_suite),
        "candidate_policy_id": stage41b_validation["candidate_policy_id"],
        "agents_compared": list(AGENTS),
        "promotion_allowed": promotion_decision["promotion_allowed"],
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage42_clean_adversarial_replay",
        "config": vars(args),
        "stage30_validation": stage30_validation,
        "runtime_validation": runtime_validation,
        "stage41a_validation": stage41a_validation,
        "stage41b_validation": stage41b_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "suite_summary": {
            "clean_case_count": len(clean_suite),
            "adversarial_case_count": len(adversarial_suite),
            "risk_types": sorted({case["risk_type"] for case in adversarial_suite}),
        },
        "aggregate": {
            "agents": {
                agent: summarize_without_records(run)
                for agent, run in runs.items()
            },
            "comparison": comparison,
        },
        "runs": runs,
        "promotion_decision": promotion_decision,
        "claim_guardrail": (
            "Stage 42 is replay evidence only. It compares a quarantined "
            "candidate against clean and adversarial suites, keeps runtime "
            "promotion blocked when critical risks remain, and makes no AGI, "
            "consciousness, live-world, or production-memory claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 42 clean + adversarial replay failed: {pass_fail}")
    print("NeoCortex-X Stage 42 clean + adversarial replay")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  candidate_policy_id: {stage41b_validation['candidate_policy_id']}")
    print(f"  clean_cases: {len(clean_suite)}")
    print(f"  adversarial_cases: {len(adversarial_suite)}")
    print(f"  critical_blocking_cases: {candidate['critical_promotion_blocking_count']}")
    print(f"  promotion_allowed: {promotion_decision['promotion_allowed']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeoCortex-X Stage 42 clean + adversarial replay."
    )
    parser.add_argument(
        "--partial-context-json",
        default="artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json",
    )
    parser.add_argument(
        "--runtime-json",
        default="artifacts/neocortex_x/stage37_clarification_runtime_loader.json",
    )
    parser.add_argument(
        "--risk-audit-json",
        default="artifacts/neocortex_x/stage41a_adversarial_risk_audit.json",
    )
    parser.add_argument(
        "--guarded-review-json",
        default="artifacts/neocortex_x/stage41b_guarded_policy_review.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage42_clean_adversarial_replay.json",
    )
    parser.add_argument("--regression-tolerance", type=float, default=0.0)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
