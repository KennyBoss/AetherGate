#!/usr/bin/env python3
"""Stage 25 indexed contextual memory store preflight for NeoCortex-X."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_TABLES = {
    "memory_slots",
    "context_dimensions",
    "provenance_refs",
    "store_metadata",
}

REQUIRED_INDEXES = {
    "idx_memory_subject_property",
    "idx_memory_subject_property_value",
    "idx_context_key_value",
    "idx_context_slot",
    "idx_provenance_ref",
    "idx_provenance_slot",
}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stable_hash(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contextual_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage16_contextual_memory":
        errors.append("invalid contextual memory kind")
    if artifact.get("pass") is not True:
        errors.append("stage16 contextual memory artifact did not pass")
    slots = artifact.get("conditional_slots")
    if not isinstance(slots, list) or not slots:
        errors.append("conditional_slots must be a non-empty list")
        slots = []
    required = {
        "slot_id",
        "memory_key",
        "subject",
        "property",
        "context",
        "value",
        "confidence",
        "provenance_refs",
        "replay_hash",
    }
    for index, slot in enumerate(slots):
        missing = required.difference(slot)
        if missing:
            errors.append(f"slot {index} missing {sorted(missing)}")
    return {"valid": not errors, "errors": errors, "slot_count": len(slots)}


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


def insert_slots(conn: sqlite3.Connection, slots: list[dict[str, Any]], *, source_hash: str) -> None:
    for slot in slots:
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
                slot["context_signature"],
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
        for provenance_ref in slot["provenance_refs"]:
            conn.execute(
                """
                INSERT INTO provenance_refs (slot_id, provenance_ref)
                VALUES (?, ?)
                """,
                (slot["slot_id"], provenance_ref),
            )
    metadata = {
        "created_at": timestamp(),
        "json_role": "report_only",
        "store_role": "indexed_preflight",
        "source_hash": source_hash,
        "slot_count": str(len(slots)),
    }
    conn.executemany(
        "INSERT INTO store_metadata (key, value) VALUES (?, ?)",
        sorted(metadata.items()),
    )
    conn.commit()


def list_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row["name"] for row in rows}


def list_indexes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    ).fetchall()
    return {row["name"] for row in rows}


def exact_context_lookup(conn: sqlite3.Connection, slot: dict[str, Any]) -> dict[str, Any] | None:
    conditions = list(slot["context"].items())
    params: list[Any] = [slot["subject"], slot["property"], len(conditions)]
    for key, value in conditions:
        params.extend([key, value])
    query = f"""
        SELECT ms.*
        FROM memory_slots ms
        JOIN context_dimensions cd ON cd.slot_id = ms.slot_id
        WHERE ms.subject = ?
          AND ms.property = ?
          AND (SELECT COUNT(*) FROM context_dimensions cdc WHERE cdc.slot_id = ms.slot_id) = ?
          AND ({' OR '.join(['(cd.context_key = ? AND cd.context_value = ?)'] * len(conditions))})
        GROUP BY ms.slot_id
        HAVING COUNT(DISTINCT cd.context_key) = ?
    """
    params.append(len(conditions))
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return None
    return dict(rows[0])


def partial_context_lookup(
    conn: sqlite3.Connection,
    *,
    subject: str,
    prop: str,
    context_key: str,
    context_value: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ms.*
        FROM memory_slots ms
        JOIN context_dimensions cd ON cd.slot_id = ms.slot_id
        WHERE ms.subject = ?
          AND ms.property = ?
          AND cd.context_key = ?
          AND cd.context_value = ?
        ORDER BY ms.confidence DESC, ms.slot_id ASC
        """,
        (subject, prop, context_key, context_value),
    ).fetchall()
    return [dict(row) for row in rows]


def subject_property_lookup(conn: sqlite3.Connection, *, subject: str, prop: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM memory_slots
        WHERE subject = ? AND property = ?
        ORDER BY confidence DESC, slot_id ASC
        """,
        (subject, prop),
    ).fetchall()
    return [dict(row) for row in rows]


def provenance_lookup(conn: sqlite3.Connection, provenance_ref: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ms.*
        FROM memory_slots ms
        JOIN provenance_refs pr ON pr.slot_id = ms.slot_id
        WHERE pr.provenance_ref = ?
        ORDER BY ms.slot_id ASC
        """,
        (provenance_ref,),
    ).fetchall()
    return [dict(row) for row in rows]


def explain_uses_index(conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT ms.slot_id
        FROM memory_slots ms
        JOIN context_dimensions cd ON cd.slot_id = ms.slot_id
        WHERE ms.subject = 'sky'
          AND ms.property = 'expected_color'
          AND cd.context_key = 'time_of_day'
          AND cd.context_value = 'night'
        """
    ).fetchall()
    details = [str(row["detail"]).upper() for row in rows]
    return any("INDEX" in detail for detail in details)


def run_queries(conn: sqlite3.Connection, slots: list[dict[str, Any]]) -> dict[str, Any]:
    exact_results = []
    for slot in slots:
        found = exact_context_lookup(conn, slot)
        exact_results.append(
            {
                "slot_id": slot["slot_id"],
                "found_slot_id": found["slot_id"] if found else None,
                "correct": bool(found and found["slot_id"] == slot["slot_id"]),
            }
        )
    partial_candidates = partial_context_lookup(
        conn,
        subject="sky",
        prop="expected_color",
        context_key="location",
        context_value="earth_surface",
    )
    subject_candidates = subject_property_lookup(conn, subject="sky", prop="expected_color")
    missing_context_candidates = subject_property_lookup(conn, subject="sky", prop="expected_color")
    provenance_ref = slots[0]["provenance_refs"][0]
    provenance_candidates = provenance_lookup(conn, provenance_ref)
    return {
        "exact_results": exact_results,
        "exact_accuracy": sum(1 for item in exact_results if item["correct"]) / len(exact_results),
        "partial_candidate_count": len(partial_candidates),
        "subject_candidate_count": len(subject_candidates),
        "missing_context_candidate_count": len(missing_context_candidates),
        "provenance_candidate_count": len(provenance_candidates),
        "query_plan_uses_index": explain_uses_index(conn),
    }


def build_store(
    *,
    contextual_artifact: dict[str, Any],
    db_path: Path,
) -> dict[str, Any]:
    slots = contextual_artifact.get("conditional_slots", [])
    source_hash = stable_hash(slots, length=24)
    conn = open_store(db_path)
    try:
        create_schema(conn)
        insert_slots(conn, slots, source_hash=source_hash)
        tables = list_tables(conn)
        indexes = list_indexes(conn)
        query_results = run_queries(conn, slots)
        slot_count = conn.execute("SELECT COUNT(*) AS n FROM memory_slots").fetchone()["n"]
        context_row_count = conn.execute("SELECT COUNT(*) AS n FROM context_dimensions").fetchone()["n"]
        provenance_row_count = conn.execute("SELECT COUNT(*) AS n FROM provenance_refs").fetchone()["n"]
    finally:
        conn.close()
    return {
        "source_hash": source_hash,
        "slot_count": int(slot_count),
        "context_row_count": int(context_row_count),
        "provenance_row_count": int(provenance_row_count),
        "tables": sorted(tables),
        "indexes": sorted(indexes),
        "query_results": query_results,
        "db_size_bytes": db_path.stat().st_size,
    }


def scale_guardrail() -> dict[str, Any]:
    return {
        "json_role": "report_only",
        "store_role": "indexed_preflight",
        "target_record_count": 10_000_000,
        "next_storage_candidates": [
            "SQLite for local deterministic smoke and schema proof",
            "DuckDB for analytical scans over columnar snapshots",
            "Postgres for durable relational conditional memory",
            "Parquet/Arrow for replayable batch snapshots",
            "RocksDB/LSM for large append-friendly key-value logs",
            "hybrid symbolic plus vector index for contextual retrieval",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    contextual_artifact = load_json(Path(args.contextual_json))
    contextual_validation = validate_contextual_artifact(contextual_artifact)
    db_path = Path(args.output_db)
    first_build = build_store(contextual_artifact=contextual_artifact, db_path=db_path)
    second_build = build_store(contextual_artifact=contextual_artifact, db_path=db_path)
    deterministic_rebuild = first_build["source_hash"] == second_build["source_hash"]
    tables_ok = REQUIRED_TABLES.issubset(set(first_build["tables"]))
    indexes_ok = REQUIRED_INDEXES.issubset(set(first_build["indexes"]))
    query_results = first_build["query_results"]
    guardrail = scale_guardrail()
    pass_fail = {
        "contextual_artifact_valid": contextual_validation["valid"],
        "sqlite_db_created": db_path.exists() and db_path.stat().st_size > 0,
        "slot_count_matches_contextual_slots": first_build["slot_count"] == contextual_validation["slot_count"],
        "required_tables_exist": tables_ok,
        "required_indexes_exist": indexes_ok,
        "exact_context_lookup_accuracy": query_results["exact_accuracy"] >= args.lookup_accuracy_min,
        "partial_lookup_returns_candidates": query_results["partial_candidate_count"] > 0,
        "subject_lookup_returns_candidates": query_results["subject_candidate_count"] == contextual_validation["slot_count"],
        "missing_context_candidate_lookup": query_results["missing_context_candidate_count"] == contextual_validation["slot_count"],
        "provenance_lookup_returns_slots": query_results["provenance_candidate_count"] > 0,
        "query_plan_uses_index": query_results["query_plan_uses_index"],
        "deterministic_rebuild": deterministic_rebuild,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "contextual_json": args.contextual_json,
        "output_db": args.output_db,
        "slot_count": first_build["slot_count"],
        "deterministic_rebuild": deterministic_rebuild,
        "json_role": guardrail["json_role"],
        "store_role": guardrail["store_role"],
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage25_indexed_contextual_memory_store",
        "config": vars(args),
        "contextual_validation": contextual_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "store": first_build,
        "scale_guardrail": guardrail,
        "claim_guardrail": (
            "Stage 25 is an indexed storage preflight over small contextual fixtures. "
            "It is not a 10M+ scale benchmark."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 25 indexed contextual store failed: {pass_fail}")
    print("NeoCortex-X Stage 25 indexed contextual memory store")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  output_db: {args.output_db}")
    print(f"  slots: {first_build['slot_count']}")
    print(f"  exact_context_lookup_accuracy: {query_results['exact_accuracy']:.4f}")
    print(f"  query_plan_uses_index: {query_results['query_plan_uses_index']}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NeoCortex-X indexed contextual memory store.")
    parser.add_argument("--contextual-json", default="artifacts/neocortex_x/stage16_contextual_memory.json")
    parser.add_argument("--output-db", default="artifacts/neocortex_x/stage25_contextual_memory.sqlite")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage25_indexed_contextual_memory_store.json")
    parser.add_argument("--lookup-accuracy-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
