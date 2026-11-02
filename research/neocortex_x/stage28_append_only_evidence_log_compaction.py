#!/usr/bin/env python3
"""Stage 28 append-only evidence log compaction for NeoCortex-X memory."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage16_contextual_memory import context_signature, make_observations, stable_hash
from stage25_indexed_contextual_memory_store import validate_contextual_artifact
from stage26_store_backed_contextual_retrieval import (
    explain_context_lookup_uses_index,
    explain_subject_lookup_uses_index,
    run_policy,
)


REQUIRED_TABLES = {
    "evidence_events",
    "memory_slots",
    "context_dimensions",
    "provenance_refs",
    "store_metadata",
}

REQUIRED_INDEXES = {
    "idx_evidence_slot",
    "idx_evidence_hash",
    "idx_evidence_type_seq",
    "idx_memory_subject_property",
    "idx_memory_subject_property_value",
    "idx_context_key_value",
    "idx_context_slot",
    "idx_provenance_ref",
    "idx_provenance_slot",
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


def open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE evidence_events (
            seq INTEGER PRIMARY KEY,
            event_id TEXT NOT NULL UNIQUE,
            event_type TEXT NOT NULL,
            slot_id TEXT NOT NULL,
            event_hash TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE memory_slots (
            slot_id TEXT PRIMARY KEY,
            memory_key TEXT NOT NULL UNIQUE,
            subject TEXT NOT NULL,
            property TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL NOT NULL,
            context_signature TEXT NOT NULL,
            replay_hash TEXT NOT NULL
        );

        CREATE TABLE context_dimensions (
            slot_id TEXT NOT NULL,
            context_key TEXT NOT NULL,
            context_value TEXT NOT NULL,
            PRIMARY KEY (slot_id, context_key),
            FOREIGN KEY (slot_id) REFERENCES memory_slots(slot_id)
        );

        CREATE TABLE provenance_refs (
            slot_id TEXT NOT NULL,
            provenance_ref TEXT NOT NULL,
            PRIMARY KEY (slot_id, provenance_ref),
            FOREIGN KEY (slot_id) REFERENCES memory_slots(slot_id)
        );

        CREATE TABLE store_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX idx_evidence_slot
            ON evidence_events(slot_id);
        CREATE INDEX idx_evidence_hash
            ON evidence_events(event_hash);
        CREATE INDEX idx_evidence_type_seq
            ON evidence_events(event_type, seq);
        CREATE INDEX idx_memory_subject_property
            ON memory_slots(subject, property);
        CREATE INDEX idx_memory_subject_property_value
            ON memory_slots(subject, property, value);
        CREATE INDEX idx_context_key_value
            ON context_dimensions(context_key, context_value);
        CREATE INDEX idx_context_slot
            ON context_dimensions(slot_id);
        CREATE INDEX idx_provenance_ref
            ON provenance_refs(provenance_ref);
        CREATE INDEX idx_provenance_slot
            ON provenance_refs(slot_id);
        """
    )


def normalize_slot(slot: dict[str, Any]) -> dict[str, Any]:
    context = {key: slot["context"][key] for key in sorted(slot["context"])}
    provenance_refs = sorted(set(slot["provenance_refs"]))
    return {
        "slot_id": slot["slot_id"],
        "memory_key": slot["memory_key"],
        "subject": slot["subject"],
        "property": slot["property"],
        "context": context,
        "value": slot["value"],
        "confidence": float(slot["confidence"]),
        "provenance_refs": provenance_refs,
        "replay_hash": slot["replay_hash"],
    }


def make_evidence_events(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for seq, slot in enumerate(sorted(slots, key=lambda item: item["slot_id"]), start=1):
        normalized = normalize_slot(slot)
        payload = {
            "seq": seq,
            "event_type": "upsert_conditional_slot",
            "slot": normalized,
            "logical_time": f"stage28:{seq:08d}",
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


def append_events(conn: sqlite3.Connection, events: list[dict[str, Any]]) -> None:
    for event in events:
        conn.execute(
            """
            INSERT INTO evidence_events (
                seq, event_id, event_type, slot_id, event_hash, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event["seq"],
                event["event_id"],
                event["event_type"],
                event["slot_id"],
                event["event_hash"],
                json.dumps(event["payload"], sort_keys=True, separators=(",", ":")),
            ),
        )
    conn.commit()


def list_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def list_indexes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    return {row["name"] for row in rows}


def validate_event_rows(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT seq, event_hash, payload_json
        FROM evidence_events
        ORDER BY seq ASC
        """
    ).fetchall()
    seqs = [int(row["seq"]) for row in rows]
    expected = list(range(1, len(rows) + 1))
    hash_checks = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        hash_checks.append(stable_hash(payload, length=24) == row["event_hash"])
    return {
        "event_count": len(rows),
        "sequence_is_append_only": seqs == expected,
        "event_hashes_valid": all(hash_checks),
    }


def clear_materialized(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM context_dimensions")
    conn.execute("DELETE FROM provenance_refs")
    conn.execute("DELETE FROM memory_slots")
    conn.execute("DELETE FROM store_metadata")


def read_compacted_slots(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT payload_json
        FROM evidence_events
        ORDER BY seq ASC
        """
    ).fetchall()
    compacted: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = json.loads(row["payload_json"])
        slot = payload["slot"]
        if payload["event_type"] == "upsert_conditional_slot":
            compacted[slot["slot_id"]] = slot
        elif payload["event_type"] == "delete_conditional_slot":
            compacted.pop(slot["slot_id"], None)
        else:
            raise ValueError(f"Unknown event_type: {payload['event_type']}")
    return [compacted[key] for key in sorted(compacted)]


def materialize_slots(
    conn: sqlite3.Connection,
    slots: list[dict[str, Any]],
    *,
    source_hash: str,
) -> dict[str, Any]:
    clear_materialized(conn)
    for slot in slots:
        ctx_signature = context_signature(slot["context"])
        conn.execute(
            """
            INSERT INTO memory_slots (
                slot_id, memory_key, subject, property, value, confidence,
                context_signature, replay_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                slot["slot_id"],
                slot["memory_key"],
                slot["subject"],
                slot["property"],
                slot["value"],
                float(slot["confidence"]),
                ctx_signature,
                slot["replay_hash"],
            ),
        )
        for key, value in sorted(slot["context"].items()):
            conn.execute(
                """
                INSERT INTO context_dimensions (slot_id, context_key, context_value)
                VALUES (?, ?, ?)
                """,
                (slot["slot_id"], key, value),
            )
        for provenance_ref in sorted(set(slot["provenance_refs"])):
            conn.execute(
                """
                INSERT INTO provenance_refs (slot_id, provenance_ref)
                VALUES (?, ?)
                """,
                (slot["slot_id"], provenance_ref),
            )
    compaction_hash = stable_hash(slots, length=24)
    metadata = {
        "evidence_log_role": "append_only_source_of_truth",
        "json_role": "report_only",
        "query_store_role": "compacted_indexed_view",
        "source_hash": source_hash,
        "compaction_hash": compaction_hash,
        "slot_count": str(len(slots)),
    }
    conn.executemany(
        "INSERT INTO store_metadata (key, value) VALUES (?, ?)",
        sorted(metadata.items()),
    )
    conn.commit()
    slot_rows = conn.execute(
        """
        SELECT slot_id, memory_key, subject, property, value, confidence,
               context_signature, replay_hash
        FROM memory_slots
        ORDER BY slot_id ASC
        """
    ).fetchall()
    return {
        "slot_count": len(slots),
        "compaction_hash": compaction_hash,
        "materialized_slots": [dict(row) for row in slot_rows],
    }


def compact_from_log(conn: sqlite3.Connection, *, source_hash: str) -> dict[str, Any]:
    slots = read_compacted_slots(conn)
    return materialize_slots(conn, slots, source_hash=source_hash)


def explain_event_lookup_uses_index(conn: sqlite3.Connection, slot_id: str) -> bool:
    rows = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT payload_json
        FROM evidence_events
        WHERE slot_id = ?
        """,
        (slot_id,),
    ).fetchall()
    return any("INDEX" in str(row["detail"]).upper() for row in rows)


def run_retrieval_compatibility(conn: sqlite3.Connection) -> dict[str, Any]:
    result = run_policy(conn, make_observations())
    false_conflict_rate = mean(
        [1.0 if item["false_conflict"] else 0.0 for item in result["episodes"]]
    )
    non_missing = [item for item in result["episodes"] if item["context"]]
    db_candidate_usage = mean(
        [1.0 if item["db_candidate_used"] else 0.0 for item in non_missing]
    )
    return {
        "decision_accuracy": result["decision_accuracy"],
        "decision_types_present": result["decision_types_present"],
        "false_contextual_conflict_rate": false_conflict_rate,
        "db_candidate_usage": db_candidate_usage,
        "episodes": result["episodes"],
    }


def build_compacted_store(
    *,
    contextual_artifact: dict[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    slots = contextual_artifact.get("conditional_slots", [])
    events = make_evidence_events(slots)
    source_hash = stable_hash([event["payload"] for event in events], length=24)
    conn = open_store(db_path)
    try:
        create_schema(conn)
        append_events(conn, events)
        event_validation = validate_event_rows(conn)
        first_compaction = compact_from_log(conn, source_hash=source_hash)
        second_compaction = compact_from_log(conn, source_hash=source_hash)
        tables = list_tables(conn)
        indexes = list_indexes(conn)
        retrieval = run_retrieval_compatibility(conn)
        event_index_used = explain_event_lookup_uses_index(conn, events[0]["slot_id"])
        subject_index_used = explain_subject_lookup_uses_index(conn)
        context_index_used = explain_context_lookup_uses_index(conn)
        context_row_count = conn.execute(
            "SELECT COUNT(*) AS n FROM context_dimensions"
        ).fetchone()["n"]
        provenance_row_count = conn.execute(
            "SELECT COUNT(*) AS n FROM provenance_refs"
        ).fetchone()["n"]
    finally:
        conn.close()
    deterministic_compaction = (
        first_compaction["compaction_hash"] == second_compaction["compaction_hash"]
        and first_compaction["materialized_slots"] == second_compaction["materialized_slots"]
    )
    return {
        "source_hash": source_hash,
        "event_validation": event_validation,
        "event_count": len(events),
        "slot_count": first_compaction["slot_count"],
        "context_row_count": int(context_row_count),
        "provenance_row_count": int(provenance_row_count),
        "compaction_hash": first_compaction["compaction_hash"],
        "deterministic_compaction": deterministic_compaction,
        "tables": sorted(tables),
        "indexes": sorted(indexes),
        "index_usage": {
            "event_lookup_uses_index": event_index_used,
            "subject_lookup_uses_index": subject_index_used,
            "context_lookup_uses_index": context_index_used,
        },
        "retrieval_compatibility": retrieval,
        "db_size_bytes": db_path.stat().st_size,
    }


def scale_contract() -> dict[str, Any]:
    return {
        "json_role": "report_only",
        "evidence_log_role": "append_only_source_of_truth",
        "query_store_role": "compacted_indexed_view",
        "target_record_count": 10_000_000,
        "future_log_candidates": [
            "SQLite WAL table for deterministic local preflight",
            "Postgres append-only evidence table for durable relational scale",
            "RocksDB/LSM for high-write local event storage",
            "Parquet/Arrow immutable event segments for batch replay",
            "Kafka-like durable log for distributed ingestion",
        ],
        "future_query_store_candidates": [
            "SQLite/DuckDB for local analytical preflight",
            "Postgres for indexed symbolic lookup",
            "hybrid symbolic plus vector index for semantic candidate retrieval",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    contextual_artifact = load_json(Path(args.contextual_json))
    contextual_validation = validate_contextual_artifact(contextual_artifact)
    db_path = Path(args.output_db)
    store = build_compacted_store(contextual_artifact=contextual_artifact, db_path=db_path)
    contract = scale_contract()
    tables_ok = REQUIRED_TABLES.issubset(set(store["tables"]))
    indexes_ok = REQUIRED_INDEXES.issubset(set(store["indexes"]))
    event_validation = store["event_validation"]
    retrieval = store["retrieval_compatibility"]
    index_usage = store["index_usage"]
    pass_fail = {
        "contextual_artifact_valid": contextual_validation["valid"],
        "sqlite_db_created": db_path.exists() and db_path.stat().st_size > 0,
        "event_count_matches_contextual_slots": store["event_count"] == contextual_validation["slot_count"],
        "append_only_sequence_valid": event_validation["sequence_is_append_only"],
        "event_hashes_valid": event_validation["event_hashes_valid"],
        "materialized_slot_count_matches_events": store["slot_count"] == store["event_count"],
        "required_tables_exist": tables_ok,
        "required_indexes_exist": indexes_ok,
        "event_lookup_uses_index": index_usage["event_lookup_uses_index"],
        "materialized_lookup_uses_indexes": (
            index_usage["subject_lookup_uses_index"] and index_usage["context_lookup_uses_index"]
        ),
        "retrieval_compatibility_accuracy": (
            retrieval["decision_accuracy"] >= args.retrieval_accuracy_min
        ),
        "false_contextual_conflict_rate": retrieval["false_contextual_conflict_rate"] == 0.0,
        "db_candidate_usage_preserved": retrieval["db_candidate_usage"] == 1.0,
        "deterministic_compaction": store["deterministic_compaction"],
        "json_not_primary_memory": (
            contract["json_role"] == "report_only"
            and contract["evidence_log_role"] == "append_only_source_of_truth"
            and contract["query_store_role"] == "compacted_indexed_view"
        ),
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "contextual_json": args.contextual_json,
        "output_db": args.output_db,
        "event_count": store["event_count"],
        "slot_count": store["slot_count"],
        "compaction_hash": store["compaction_hash"],
        "deterministic_compaction": store["deterministic_compaction"],
        "json_role": contract["json_role"],
        "evidence_log_role": contract["evidence_log_role"],
        "query_store_role": contract["query_store_role"],
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage28_append_only_evidence_log_compaction",
        "config": vars(args),
        "contextual_validation": contextual_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "store": store,
        "scale_contract": contract,
        "claim_guardrail": (
            "Stage 28 proves append-only evidence log compaction into an indexed "
            "query store on a small deterministic fixture. It is not the 10M+ "
            "scale benchmark yet."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 28 evidence-log compaction failed: {pass_fail}")
    print("NeoCortex-X Stage 28 append-only evidence log compaction")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  output_db: {args.output_db}")
    print(f"  events: {store['event_count']}")
    print(f"  slots: {store['slot_count']}")
    print(f"  deterministic_compaction: {store['deterministic_compaction']}")
    print(f"  retrieval_accuracy: {retrieval['decision_accuracy']:.4f}")
    print(f"  db_candidate_usage: {retrieval['db_candidate_usage']:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NeoCortex-X append-only evidence log and compacted query store."
    )
    parser.add_argument(
        "--contextual-json",
        default="artifacts/neocortex_x/stage16_contextual_memory.json",
    )
    parser.add_argument(
        "--output-db",
        default="artifacts/neocortex_x/stage28_evidence_log_compacted_store.sqlite",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage28_append_only_evidence_log_compaction.json",
    )
    parser.add_argument("--retrieval-accuracy-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
