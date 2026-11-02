#!/usr/bin/env python3
"""Stage 30 partial-context scale retrieval for NeoCortex-X memory."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage16_contextual_memory import context_signature, stable_hash
from stage28_append_only_evidence_log_compaction import (
    append_events,
    compact_from_log,
    create_schema,
    open_store,
    validate_event_rows,
)
from stage29_contextual_memory_scale_probe import percentile


CONTEXT_KEYS = ("time_of_day", "weather", "location", "season")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def make_context(index: int) -> dict[str, str]:
    times = ("daytime", "night", "sunset", "dawn")
    weather = ("clear", "cloudy", "rain", "snow", "haze")
    locations = ("earth_surface", "mountain", "coast", "city", "desert", "orbit")
    seasons = ("spring", "summer", "autumn", "winter")
    return {
        "time_of_day": times[index % len(times)],
        "weather": weather[(index // 3) % len(weather)],
        "location": locations[(index // 5) % len(locations)],
        "season": seasons[(index // 7) % len(seasons)],
    }


def make_slot(index: int) -> dict[str, Any]:
    subject = f"entity_{index % 64:03d}"
    prop = f"property_{(index // 64) % 8:02d}"
    context = make_context(index)
    value = f"value_{stable_hash({'stage': 30, 'i': index, 'ctx': context}, length=10)}"
    basis = {
        "index": index,
        "subject": subject,
        "property": prop,
        "context": context,
        "value": value,
    }
    slot_hash = stable_hash(basis, length=16)
    return {
        "slot_id": f"partial_ctx_{slot_hash}",
        "memory_key": f"nx30:{subject}:{prop}:{stable_hash(context, length=8)}:{slot_hash[:6]}",
        "subject": subject,
        "property": prop,
        "context": context,
        "value": value,
        "confidence": 0.70 + ((index % 25) / 100.0),
        "provenance_refs": [f"scale_fixture://stage30/{index:08d}"],
        "replay_hash": stable_hash(basis, length=24),
    }


def generate_slots(record_count: int) -> list[dict[str, Any]]:
    return [make_slot(index) for index in range(record_count)]


def make_events(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for seq, slot in enumerate(slots, start=1):
        normalized = {
            "slot_id": slot["slot_id"],
            "memory_key": slot["memory_key"],
            "subject": slot["subject"],
            "property": slot["property"],
            "context": {key: slot["context"][key] for key in sorted(slot["context"])},
            "value": slot["value"],
            "confidence": float(slot["confidence"]),
            "provenance_refs": sorted(set(slot["provenance_refs"])),
            "replay_hash": slot["replay_hash"],
        }
        payload = {
            "seq": seq,
            "event_type": "upsert_conditional_slot",
            "slot": normalized,
            "logical_time": f"stage30:{seq:08d}",
        }
        event_hash = stable_hash(payload, length=24)
        events.append(
            {
                "seq": seq,
                "event_id": f"elog_{event_hash}",
                "event_type": payload["event_type"],
                "slot_id": normalized["slot_id"],
                "event_hash": event_hash,
                "payload": payload,
            }
        )
    return events


def partial_context_for(slot: dict[str, Any], variant: int) -> dict[str, str]:
    key_sets = (
        ("time_of_day",),
        ("weather",),
        ("time_of_day", "weather"),
        ("location", "season"),
        ("weather", "location"),
    )
    selected = key_sets[variant % len(key_sets)]
    return {key: slot["context"][key] for key in selected}


def make_queries(slots: list[dict[str, Any]], query_count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample_count = min(query_count, len(slots))
    indexes = rng.sample(range(len(slots)), sample_count)
    queries = []
    for variant, index in enumerate(indexes):
        slot = slots[index]
        partial_context = partial_context_for(slot, variant)
        queries.append(
            {
                "query_id": f"pq_{variant:04d}_{slot['slot_id']}",
                "target_slot_id": slot["slot_id"],
                "subject": slot["subject"],
                "property": slot["property"],
                "partial_context": partial_context,
                "full_context": slot["context"],
                "target_value": slot["value"],
            }
        )
    return queries


def partial_lookup(
    conn: sqlite3.Connection,
    query: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    conditions = list(query["partial_context"].items())
    params: list[Any] = [query["subject"], query["property"]]
    exists_clauses = []
    for key, value in conditions:
        exists_clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM context_dimensions cd
                WHERE cd.slot_id = ms.slot_id
                  AND cd.context_key = ?
                  AND cd.context_value = ?
            )
            """
        )
        params.extend([key, value])
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT ms.slot_id, ms.value, ms.confidence, ms.context_signature
        FROM memory_slots ms
        WHERE ms.subject = ?
          AND ms.property = ?
          AND {' AND '.join(exists_clauses)}
        ORDER BY ms.confidence DESC, ms.slot_id ASC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def json_partial_lookup(
    slots: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates = []
    partial_context = query["partial_context"]
    for slot in slots:
        if slot["subject"] != query["subject"] or slot["property"] != query["property"]:
            continue
        if all(slot["context"].get(key) == value for key, value in partial_context.items()):
            candidates.append(slot)
    candidates.sort(key=lambda item: (-float(item["confidence"]), item["slot_id"]))
    return candidates[:limit]


def exact_context_rerank(
    candidates: list[dict[str, Any]],
    query: dict[str, Any],
) -> dict[str, Any] | None:
    target_signature = context_signature(query["full_context"])
    for candidate in candidates:
        if candidate.get("context_signature") == target_signature:
            return candidate
        if "context" in candidate and context_signature(candidate["context"]) == target_signature:
            return candidate
    return None


def time_partial_indexed(
    conn: sqlite3.Connection,
    queries: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    durations = []
    target_hits = []
    rerank_hits = []
    candidate_counts = []
    evaluated_queries = []
    for query in queries:
        start = time.perf_counter()
        candidates = partial_lookup(conn, query, limit=limit)
        durations.append((time.perf_counter() - start) * 1000.0)
        candidate_ids = {candidate["slot_id"] for candidate in candidates}
        target_hits.append(query["target_slot_id"] in candidate_ids)
        reranked = exact_context_rerank(candidates, query)
        rerank_hits.append(bool(reranked and reranked["slot_id"] == query["target_slot_id"]))
        candidate_counts.append(len(candidates))
        evaluated_queries.append(
            {
                "query_id": query["query_id"],
                "target_slot_id": query["target_slot_id"],
                "subject": query["subject"],
                "property": query["property"],
                "partial_context": query["partial_context"],
                "full_context": query["full_context"],
                "target_value": query["target_value"],
                "candidate_count": len(candidates),
                "candidate_ids": [candidate["slot_id"] for candidate in candidates],
                "candidate_context_signatures": [
                    candidate["context_signature"] for candidate in candidates
                ],
                "candidate_contexts": [
                    dict(part.split("=", 1) for part in candidate["context_signature"].split("|"))
                    for candidate in candidates
                ],
                "target_in_candidates": query["target_slot_id"] in candidate_ids,
                "reranked_slot_id": reranked["slot_id"] if reranked else None,
            }
        )
    return {
        "target_recall": mean([1.0 if hit else 0.0 for hit in target_hits]),
        "exact_rerank_accuracy": mean([1.0 if hit else 0.0 for hit in rerank_hits]),
        "candidate_count_mean": mean([float(count) for count in candidate_counts]),
        "candidate_count_p95": percentile([float(count) for count in candidate_counts], 95),
        "multi_candidate_rate": mean([1.0 if count > 1 else 0.0 for count in candidate_counts]),
        "latency_ms_p50": percentile(durations, 50),
        "latency_ms_p95": percentile(durations, 95),
        "latency_ms_max": max(durations) if durations else 0.0,
        "evaluated_queries": evaluated_queries,
    }


def time_partial_json_scan(
    slots: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    limit: int,
    max_queries: int,
) -> dict[str, Any]:
    durations = []
    target_hits = []
    candidate_counts = []
    for query in queries[:max_queries]:
        start = time.perf_counter()
        candidates = json_partial_lookup(slots, query, limit=limit)
        durations.append((time.perf_counter() - start) * 1000.0)
        candidate_ids = {candidate["slot_id"] for candidate in candidates}
        target_hits.append(query["target_slot_id"] in candidate_ids)
        candidate_counts.append(len(candidates))
    return {
        "query_count": len(durations),
        "target_recall": mean([1.0 if hit else 0.0 for hit in target_hits]),
        "candidate_count_mean": mean([float(count) for count in candidate_counts]),
        "candidate_count_p95": percentile([float(count) for count in candidate_counts], 95),
        "latency_ms_p50": percentile(durations, 50),
        "latency_ms_p95": percentile(durations, 95),
        "latency_ms_max": max(durations) if durations else 0.0,
    }


def explain_partial_lookup_uses_index(conn: sqlite3.Connection, query: dict[str, Any]) -> bool:
    conditions = list(query["partial_context"].items())
    params: list[Any] = [query["subject"], query["property"]]
    exists_clauses = []
    for key, value in conditions:
        exists_clauses.append(
            """
            EXISTS (
                SELECT 1
                FROM context_dimensions cd
                WHERE cd.slot_id = ms.slot_id
                  AND cd.context_key = ?
                  AND cd.context_value = ?
            )
            """
        )
        params.extend([key, value])
    rows = conn.execute(
        f"""
        EXPLAIN QUERY PLAN
        SELECT ms.slot_id
        FROM memory_slots ms
        WHERE ms.subject = ?
          AND ms.property = ?
          AND {' AND '.join(exists_clauses)}
        """,
        params,
    ).fetchall()
    details = [str(row["detail"]).upper() for row in rows]
    return any("INDEX" in detail for detail in details)


def explain_context_dimension_index(conn: sqlite3.Connection, query: dict[str, Any]) -> bool:
    key, value = next(iter(query["partial_context"].items()))
    rows = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT slot_id
        FROM context_dimensions
        WHERE context_key = ? AND context_value = ?
        """,
        (key, value),
    ).fetchall()
    return any("IDX_CONTEXT_KEY_VALUE" in str(row["detail"]).upper() for row in rows)


def build_probe(args: argparse.Namespace) -> dict[str, Any]:
    slots = generate_slots(args.records)
    events = make_events(slots)
    source_hash = stable_hash([event["payload"] for event in events], length=24)
    db_path = Path(args.output_db)
    conn = open_store(db_path)
    try:
        create_schema(conn)
        append_events(conn, events)
        event_validation = validate_event_rows(conn)
        first_compaction = compact_from_log(conn, source_hash=source_hash)
        second_compaction = compact_from_log(conn, source_hash=source_hash)
        deterministic_compaction = (
            first_compaction["compaction_hash"] == second_compaction["compaction_hash"]
            and first_compaction["materialized_slots"] == second_compaction["materialized_slots"]
        )
        queries = make_queries(slots, args.queries, args.seed)
        indexed = time_partial_indexed(conn, queries, limit=args.candidate_limit)
        json_scan = time_partial_json_scan(
            slots,
            queries,
            limit=args.candidate_limit,
            max_queries=args.json_scan_queries,
        )
        partial_index_used = explain_partial_lookup_uses_index(conn, queries[0])
        context_index_used = explain_context_dimension_index(conn, queries[0])
        materialized_count = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_slots"
        ).fetchone()["n"]
        event_count = conn.execute(
            "SELECT COUNT(*) AS n FROM evidence_events"
        ).fetchone()["n"]
    finally:
        conn.close()
    return {
        "record_count": len(slots),
        "event_count": int(event_count),
        "materialized_slot_count": int(materialized_count),
        "event_validation": event_validation,
        "deterministic_compaction": deterministic_compaction,
        "compaction_hash": first_compaction["compaction_hash"],
        "db_size_bytes": db_path.stat().st_size,
        "query_count": len(queries),
        "candidate_limit": args.candidate_limit,
        "partial_indexed_lookup": indexed,
        "evaluated_queries": indexed["evaluated_queries"],
        "json_scan_baseline": json_scan,
        "index_usage": {
            "partial_lookup_uses_index": partial_index_used,
            "context_dimension_index_used": context_index_used,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    probe = build_probe(args)
    indexed_p95 = probe["partial_indexed_lookup"]["latency_ms_p95"]
    json_p95 = probe["json_scan_baseline"]["latency_ms_p95"]
    speedup = json_p95 / indexed_p95 if indexed_p95 > 0 else 0.0
    pass_fail = {
        "record_count_minimum_met": probe["record_count"] >= args.min_records,
        "event_count_matches_records": probe["event_count"] == probe["record_count"],
        "materialized_count_matches_records": (
            probe["materialized_slot_count"] == probe["record_count"]
        ),
        "append_only_sequence_valid": probe["event_validation"]["sequence_is_append_only"],
        "event_hashes_valid": probe["event_validation"]["event_hashes_valid"],
        "deterministic_compaction": probe["deterministic_compaction"],
        "partial_target_recall": (
            probe["partial_indexed_lookup"]["target_recall"] >= args.target_recall_min
        ),
        "exact_rerank_accuracy": (
            probe["partial_indexed_lookup"]["exact_rerank_accuracy"] >= args.rerank_accuracy_min
        ),
        "multi_candidate_queries_present": (
            probe["partial_indexed_lookup"]["multi_candidate_rate"] >= args.multi_candidate_rate_min
        ),
        "indexed_partial_lookup_p95_under_budget": indexed_p95 <= args.indexed_p95_ms_max,
        "index_usage_verified": (
            probe["index_usage"]["partial_lookup_uses_index"]
            and probe["index_usage"]["context_dimension_index_used"]
        ),
        "json_scan_baseline_recalls_target": (
            probe["json_scan_baseline"]["target_recall"] >= args.target_recall_min
        ),
        "indexed_lookup_beats_json_scan": speedup >= args.min_json_scan_speedup,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "record_count": probe["record_count"],
        "query_count": probe["query_count"],
        "candidate_limit": probe["candidate_limit"],
        "compaction_hash": probe["compaction_hash"],
        "partial_target_recall": probe["partial_indexed_lookup"]["target_recall"],
        "exact_rerank_accuracy": probe["partial_indexed_lookup"]["exact_rerank_accuracy"],
        "indexed_partial_p95_ms": indexed_p95,
        "json_scan_p95_ms": json_p95,
        "json_scan_speedup_ratio": speedup,
        "json_role": "baseline_only",
        "evidence_log_role": "append_only_source_of_truth",
        "query_store_role": "compacted_indexed_view",
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage30_partial_context_scale_retrieval",
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "probe": probe,
        "scale_guardrail": {
            "target_record_count": 10_000_000,
            "current_probe_record_count": probe["record_count"],
            "json_role": "baseline_only",
            "claim": (
                "Stage 30 probes partial-context retrieval over thousands of "
                "records. It is not the final 10M+ benchmark."
            ),
        },
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 30 partial-context retrieval failed: {pass_fail}")
    print("NeoCortex-X Stage 30 partial-context scale retrieval")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  records: {probe['record_count']}")
    print(f"  target_recall: {protocol_evidence['partial_target_recall']:.4f}")
    print(f"  exact_rerank_accuracy: {protocol_evidence['exact_rerank_accuracy']:.4f}")
    print(f"  indexed_partial_p95_ms: {indexed_p95:.6f}")
    print(f"  json_scan_p95_ms: {json_p95:.6f}")
    print(f"  json_scan_speedup_ratio: {speedup:.2f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X partial-context scale retrieval.")
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--json-scan-queries", type=int, default=64)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--output-db", default="artifacts/neocortex_x/stage30_partial_context.sqlite")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json")
    parser.add_argument("--min-records", type=int, default=1000)
    parser.add_argument("--target-recall-min", type=float, default=1.0)
    parser.add_argument("--rerank-accuracy-min", type=float, default=1.0)
    parser.add_argument("--multi-candidate-rate-min", type=float, default=0.10)
    parser.add_argument("--indexed-p95-ms-max", type=float, default=8.0)
    parser.add_argument("--min-json-scan-speedup", type=float, default=1.2)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
