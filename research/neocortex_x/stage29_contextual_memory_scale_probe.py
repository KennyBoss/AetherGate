#!/usr/bin/env python3
"""Stage 29 scale probe for NeoCortex-X contextual memory storage."""

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
    explain_event_lookup_uses_index,
    open_store,
    validate_event_rows,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def make_context(index: int) -> dict[str, str]:
    times = ("daytime", "night", "sunset", "dawn")
    weather = ("clear", "cloudy", "rain", "snow", "haze")
    locations = ("earth_surface", "mountain", "coast", "city", "desert")
    seasons = ("spring", "summer", "autumn", "winter")
    return {
        "time_of_day": times[index % len(times)],
        "weather": weather[(index // len(times)) % len(weather)],
        "location": locations[(index // (len(times) * len(weather))) % len(locations)],
        "season": seasons[(index // (len(times) * len(weather) * len(locations))) % len(seasons)],
    }


def make_slot(index: int) -> dict[str, Any]:
    subject = f"entity_{index % 257:04d}"
    prop = f"property_{(index // 257) % 17:02d}"
    context = make_context(index)
    value = f"value_{stable_hash({'i': index, 'ctx': context}, length=10)}"
    basis = {
        "index": index,
        "subject": subject,
        "property": prop,
        "context": context,
        "value": value,
    }
    slot_hash = stable_hash(basis, length=16)
    return {
        "slot_id": f"scale_ctx_{slot_hash}",
        "memory_key": f"nx29:{subject}:{prop}:{stable_hash(context, length=8)}:{slot_hash[:6]}",
        "subject": subject,
        "property": prop,
        "context": context,
        "value": value,
        "confidence": 0.70 + ((index % 30) / 100.0),
        "provenance_refs": [f"scale_fixture://stage29/{index:08d}"],
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
            "logical_time": f"stage29:{seq:08d}",
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


def lookup_slot(conn: sqlite3.Connection, slot: dict[str, Any]) -> dict[str, Any] | None:
    rows = conn.execute(
        """
        SELECT ms.slot_id, ms.value, ms.confidence
        FROM memory_slots ms
        WHERE ms.subject = ?
          AND ms.property = ?
          AND ms.context_signature = ?
        ORDER BY ms.confidence DESC, ms.slot_id ASC
        LIMIT 1
        """,
        (slot["subject"], slot["property"], context_signature(slot["context"])),
    ).fetchall()
    if not rows:
        return None
    return dict(rows[0])


def json_scan_lookup(slots: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any] | None:
    target_signature = context_signature(target["context"])
    for slot in slots:
        if (
            slot["subject"] == target["subject"]
            and slot["property"] == target["property"]
            and context_signature(slot["context"]) == target_signature
        ):
            return slot
    return None


def explain_lookup_uses_index(conn: sqlite3.Connection, slot: dict[str, Any]) -> bool:
    rows = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT ms.slot_id
        FROM memory_slots ms
        WHERE ms.subject = ?
          AND ms.property = ?
          AND ms.context_signature = ?
        """,
        (slot["subject"], slot["property"], context_signature(slot["context"])),
    ).fetchall()
    return any("INDEX" in str(row["detail"]).upper() for row in rows)


def sample_queries(slots: list[dict[str, Any]], query_count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if query_count >= len(slots):
        return list(slots)
    indexes = rng.sample(range(len(slots)), query_count)
    return [slots[index] for index in indexes]


def time_indexed_queries(
    conn: sqlite3.Connection,
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    durations = []
    correct = []
    for query in queries:
        start = time.perf_counter()
        found = lookup_slot(conn, query)
        durations.append((time.perf_counter() - start) * 1000.0)
        correct.append(bool(found and found["slot_id"] == query["slot_id"]))
    return {
        "accuracy": sum(1 for item in correct if item) / len(correct),
        "latency_ms_p50": percentile(durations, 50),
        "latency_ms_p95": percentile(durations, 95),
        "latency_ms_max": max(durations) if durations else 0.0,
    }


def time_json_scan_queries(
    slots: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    max_queries: int,
) -> dict[str, Any]:
    durations = []
    correct = []
    for query in queries[:max_queries]:
        start = time.perf_counter()
        found = json_scan_lookup(slots, query)
        durations.append((time.perf_counter() - start) * 1000.0)
        correct.append(bool(found and found["slot_id"] == query["slot_id"]))
    return {
        "query_count": len(durations),
        "accuracy": sum(1 for item in correct if item) / len(correct),
        "latency_ms_p50": percentile(durations, 50),
        "latency_ms_p95": percentile(durations, 95),
        "latency_ms_max": max(durations) if durations else 0.0,
    }


def build_probe(args: argparse.Namespace) -> dict[str, Any]:
    slots = generate_slots(args.records)
    events = make_events(slots)
    source_hash = stable_hash([event["payload"] for event in events], length=24)
    db_path = Path(args.output_db)
    conn = open_store(db_path)
    try:
        create_schema(conn)
        ingest_start = time.perf_counter()
        append_events(conn, events)
        ingest_seconds = time.perf_counter() - ingest_start
        event_validation = validate_event_rows(conn)
        compaction_start = time.perf_counter()
        first_compaction = compact_from_log(conn, source_hash=source_hash)
        compaction_seconds = time.perf_counter() - compaction_start
        second_compaction = compact_from_log(conn, source_hash=source_hash)
        deterministic_compaction = (
            first_compaction["compaction_hash"] == second_compaction["compaction_hash"]
            and first_compaction["materialized_slots"] == second_compaction["materialized_slots"]
        )
        queries = sample_queries(slots, args.queries, args.seed)
        indexed = time_indexed_queries(conn, queries)
        json_scan = time_json_scan_queries(slots, queries, max_queries=args.json_scan_queries)
        event_index_used = explain_event_lookup_uses_index(conn, events[0]["slot_id"])
        lookup_index_used = explain_lookup_uses_index(conn, queries[0])
        materialized_count = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_slots"
        ).fetchone()["n"]
        event_count = conn.execute(
            "SELECT COUNT(*) AS n FROM evidence_events"
        ).fetchone()["n"]
        context_row_count = conn.execute(
            "SELECT COUNT(*) AS n FROM context_dimensions"
        ).fetchone()["n"]
    finally:
        conn.close()
    return {
        "record_count": len(slots),
        "event_count": int(event_count),
        "materialized_slot_count": int(materialized_count),
        "context_row_count": int(context_row_count),
        "event_validation": event_validation,
        "deterministic_compaction": deterministic_compaction,
        "compaction_hash": first_compaction["compaction_hash"],
        "db_size_bytes": db_path.stat().st_size,
        "ingest": {
            "seconds": ingest_seconds,
            "events_per_second": len(events) / ingest_seconds if ingest_seconds > 0 else 0.0,
        },
        "compaction": {
            "seconds": compaction_seconds,
            "slots_per_second": len(slots) / compaction_seconds if compaction_seconds > 0 else 0.0,
        },
        "indexed_lookup": indexed,
        "json_scan_baseline": json_scan,
        "index_usage": {
            "event_lookup_uses_index": event_index_used,
            "materialized_lookup_uses_index": lookup_index_used,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    probe = build_probe(args)
    indexed_p95 = probe["indexed_lookup"]["latency_ms_p95"]
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
        "indexed_lookup_accuracy": probe["indexed_lookup"]["accuracy"] >= args.lookup_accuracy_min,
        "indexed_lookup_p95_under_budget": indexed_p95 <= args.indexed_p95_ms_max,
        "index_usage_verified": (
            probe["index_usage"]["event_lookup_uses_index"]
            and probe["index_usage"]["materialized_lookup_uses_index"]
        ),
        "json_scan_baseline_correct": (
            probe["json_scan_baseline"]["accuracy"] >= args.lookup_accuracy_min
        ),
        "indexed_lookup_beats_json_scan": speedup >= args.min_json_scan_speedup,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "record_count": probe["record_count"],
        "query_count": args.queries,
        "json_scan_query_count": probe["json_scan_baseline"]["query_count"],
        "compaction_hash": probe["compaction_hash"],
        "indexed_lookup_p95_ms": indexed_p95,
        "json_scan_p95_ms": json_p95,
        "json_scan_speedup_ratio": speedup,
        "json_role": "baseline_only",
        "evidence_log_role": "append_only_source_of_truth",
        "query_store_role": "compacted_indexed_view",
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage29_contextual_memory_scale_probe",
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
                "Stage 29 is a fast scale probe over thousands of records, "
                "not the final 10M+ benchmark."
            ),
        },
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 29 contextual memory scale probe failed: {pass_fail}")
    print("NeoCortex-X Stage 29 contextual memory scale probe")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  records: {probe['record_count']}")
    print(f"  events_per_second: {probe['ingest']['events_per_second']:.2f}")
    print(f"  compaction_slots_per_second: {probe['compaction']['slots_per_second']:.2f}")
    print(f"  indexed_lookup_p95_ms: {indexed_p95:.6f}")
    print(f"  json_scan_p95_ms: {json_p95:.6f}")
    print(f"  json_scan_speedup_ratio: {speedup:.2f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X contextual memory scale probe.")
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--json-scan-queries", type=int, default=64)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output-db", default="artifacts/neocortex_x/stage29_scale_probe.sqlite")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage29_contextual_memory_scale_probe.json")
    parser.add_argument("--min-records", type=int, default=1000)
    parser.add_argument("--lookup-accuracy-min", type=float, default=0.999)
    parser.add_argument("--indexed-p95-ms-max", type=float, default=5.0)
    parser.add_argument("--min-json-scan-speedup", type=float, default=1.5)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
