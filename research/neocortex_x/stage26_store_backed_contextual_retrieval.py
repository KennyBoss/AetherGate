#!/usr/bin/env python3
"""Stage 26 store-backed contextual retrieval for NeoCortex-X."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage16_contextual_memory import compatible_values, make_observations


DECISION_TYPES = {
    "context_match",
    "same_subject_different_context",
    "request_context",
    "flag_conflict",
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


def open_readonly_store(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def validate_store_report(artifact: dict[str, Any], db_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage25_indexed_contextual_memory_store":
        errors.append("invalid store report kind")
    if artifact.get("pass") is not True:
        errors.append("stage25 indexed contextual store artifact did not pass")
    if not db_path.exists() or db_path.stat().st_size <= 0:
        errors.append("sqlite db artifact missing or empty")
    store = artifact.get("store", {})
    if int(store.get("slot_count", 0)) <= 0:
        errors.append("store slot count missing")
    return {
        "valid": not errors,
        "errors": errors,
        "slot_count": int(store.get("slot_count", 0)),
        "db_path": str(db_path),
    }


def row_to_slot(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "slot_id": row["slot_id"],
        "memory_key": row["memory_key"],
        "subject": row["subject"],
        "property": row["property"],
        "value": row["value"],
        "confidence": float(row["confidence"]),
        "context_signature": row["context_signature"],
        "replay_hash": row["replay_hash"],
    }


def subject_property_candidates(
    conn: sqlite3.Connection,
    *,
    subject: str,
    prop: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM memory_slots
        WHERE subject = ? AND property = ?
        ORDER BY confidence DESC, slot_id ASC
        """,
        (subject, prop),
    ).fetchall()
    return [row_to_slot(row) for row in rows]


def slot_context(conn: sqlite3.Connection, slot_id: str) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT context_key, context_value
        FROM context_dimensions
        WHERE slot_id = ?
        ORDER BY context_key ASC
        """,
        (slot_id,),
    ).fetchall()
    return {row["context_key"]: row["context_value"] for row in rows}


def enrich_slots(conn: sqlite3.Connection, slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for slot in slots:
        item = dict(slot)
        item["context"] = slot_context(conn, slot["slot_id"])
        enriched.append(item)
    return enriched


def context_matches(slot_context_map: dict[str, str], observation_context: dict[str, str]) -> bool:
    return all(observation_context.get(key) == value for key, value in slot_context_map.items())


def context_present(observation_context: dict[str, str]) -> bool:
    return bool(observation_context)


def best_by_confidence(slots: list[dict[str, Any]]) -> dict[str, Any]:
    return max(slots, key=lambda slot: (float(slot["confidence"]), slot["slot_id"]))


def explain_subject_lookup_uses_index(conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT *
        FROM memory_slots
        WHERE subject = 'sky' AND property = 'expected_color'
        """
    ).fetchall()
    return any("INDEX" in str(row["detail"]).upper() for row in rows)


def explain_context_lookup_uses_index(conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT ms.slot_id
        FROM memory_slots ms
        JOIN context_dimensions cd ON cd.slot_id = ms.slot_id
        WHERE cd.context_key = 'time_of_day'
          AND cd.context_value = 'night'
        """
    ).fetchall()
    return any("INDEX" in str(row["detail"]).upper() for row in rows)


def decide_from_store(conn: sqlite3.Connection, observation: dict[str, Any]) -> dict[str, Any]:
    candidates = enrich_slots(
        conn,
        subject_property_candidates(
            conn,
            subject=observation["subject"],
            prop=observation["property"],
        ),
    )
    if not candidates:
        return {
            "decision": "request_context",
            "matched_slot_id": None,
            "reason": "no_subject_property_slots",
            "candidate_count": 0,
        }
    if not context_present(observation["context"]):
        return {
            "decision": "request_context",
            "matched_slot_id": None,
            "reason": "missing_context",
            "candidate_count": len(candidates),
        }
    matching_slots = [
        slot for slot in candidates if context_matches(slot["context"], observation["context"])
    ]
    if matching_slots:
        best = best_by_confidence(matching_slots)
        if compatible_values(observation["observed_value"], best["value"]):
            exact = observation["observed_value"] == best["value"]
            return {
                "decision": "context_match" if exact else "same_subject_different_context",
                "matched_slot_id": best["slot_id"],
                "reason": "store_context_value_match" if exact else "store_context_value_alias",
                "candidate_count": len(candidates),
            }
        return {
            "decision": "flag_conflict",
            "matched_slot_id": best["slot_id"],
            "reason": "store_same_context_incompatible_value",
            "candidate_count": len(candidates),
        }
    value_compatible_slots = [
        slot for slot in candidates if compatible_values(observation["observed_value"], slot["value"])
    ]
    if value_compatible_slots:
        best = best_by_confidence(value_compatible_slots)
        return {
            "decision": "same_subject_different_context",
            "matched_slot_id": best["slot_id"],
            "reason": "store_value_known_under_different_context",
            "candidate_count": len(candidates),
        }
    return {
        "decision": "request_context",
        "matched_slot_id": None,
        "reason": "store_known_subject_unknown_context_value_pair",
        "candidate_count": len(candidates),
    }


def run_policy(conn: sqlite3.Connection, observations: list[dict[str, Any]]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for observation in observations:
        decision = decide_from_store(conn, observation)
        actual = decision["decision"]
        expected = observation["expected_decision"]
        episodes.append(
            {
                "observation_id": observation["observation_id"],
                "subject": observation["subject"],
                "property": observation["property"],
                "context": observation["context"],
                "observed_value": observation["observed_value"],
                "decision": actual,
                "expected_decision": expected,
                "decision_correct": actual == expected,
                "matched_slot_id": decision["matched_slot_id"],
                "reason": decision["reason"],
                "candidate_count": decision["candidate_count"],
                "db_candidate_used": bool(decision["matched_slot_id"]),
                "false_conflict": actual == "flag_conflict" and expected != "flag_conflict",
            }
        )
    return {
        "episodes": episodes,
        "decision_accuracy": mean([1.0 if episode["decision_correct"] else 0.0 for episode in episodes]),
        "decision_types_present": sorted({episode["decision"] for episode in episodes}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    store_report = load_json(Path(args.store_json))
    db_path = Path(args.store_db)
    store_validation = validate_store_report(store_report, db_path)
    observations = make_observations()
    conn = open_readonly_store(db_path)
    try:
        result = run_policy(conn, observations)
        replay_result = run_policy(conn, observations)
        subject_index_used = explain_subject_lookup_uses_index(conn)
        context_index_used = explain_context_lookup_uses_index(conn)
    finally:
        conn.close()
    deterministic_replay = result["episodes"] == replay_result["episodes"]
    false_conflict_rate = mean([1.0 if item["false_conflict"] else 0.0 for item in result["episodes"]])
    missing_context_detection_rate = mean(
        [
            1.0 if item["decision"] == "request_context" else 0.0
            for item in result["episodes"]
            if not item["context"]
        ]
    )
    non_missing = [item for item in result["episodes"] if item["context"]]
    db_candidate_usage = mean([1.0 if item["db_candidate_used"] else 0.0 for item in non_missing])
    decision_types_present = set(result["decision_types_present"])
    indexed_query_plans = subject_index_used and context_index_used
    pass_fail = {
        "store_report_valid": store_validation["valid"],
        "store_backed_decision_accuracy": result["decision_accuracy"] >= args.decision_accuracy_min,
        "required_decision_types_present": DECISION_TYPES.issubset(decision_types_present),
        "false_contextual_conflict_rate": false_conflict_rate == 0.0,
        "missing_context_detection_rate": missing_context_detection_rate == 1.0,
        "db_candidate_used_for_non_missing_decisions": db_candidate_usage == 1.0,
        "indexed_query_plans_used": indexed_query_plans,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "store_json": args.store_json,
        "store_db": args.store_db,
        "observation_count": len(observations),
        "decision_types_present": sorted(decision_types_present),
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage26_store_backed_contextual_retrieval",
        "config": vars(args),
        "store_validation": store_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "decision_accuracy": result["decision_accuracy"],
            "false_contextual_conflict_rate": false_conflict_rate,
            "missing_context_detection_rate": missing_context_detection_rate,
            "db_candidate_usage": db_candidate_usage,
            "subject_index_used": subject_index_used,
            "context_index_used": context_index_used,
        },
        "episodes": result["episodes"],
        "claim_guardrail": (
            "Stage 26 makes contextual decisions through the indexed SQLite store. "
            "It is still a small deterministic smoke, not a 10M+ benchmark."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 26 store-backed contextual retrieval failed: {pass_fail}")
    print("NeoCortex-X Stage 26 store-backed contextual retrieval")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  observations: {len(observations)}")
    print(f"  decision_accuracy: {result['decision_accuracy']:.4f}")
    print(f"  db_candidate_usage: {db_candidate_usage:.4f}")
    print(f"  indexed_query_plans_used: {indexed_query_plans}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X store-backed contextual retrieval.")
    parser.add_argument("--store-json", default="artifacts/neocortex_x/stage25_indexed_contextual_memory_store.json")
    parser.add_argument("--store-db", default="artifacts/neocortex_x/stage25_contextual_memory.sqlite")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage26_store_backed_contextual_retrieval.json")
    parser.add_argument("--decision-accuracy-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
