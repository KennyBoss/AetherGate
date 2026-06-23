#!/usr/bin/env python3
"""Stage 16 context-aware memory over NeoCortex-X evidence packets.

The smoke verifies that memory stores conditional knowledge rather than
unconditional subject=value facts. A value that looks contradictory can be valid
under a different context, and missing context should trigger clarification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7_memory_retrieval import load_packets, validate_packet_artifact


CONTEXT_DIMENSIONS = ("time_of_day", "weather", "location", "season")
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


def stable_hash(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def context_matches(slot_context: dict[str, str], observation_context: dict[str, str]) -> bool:
    return all(observation_context.get(key) == value for key, value in slot_context.items())


def context_is_complete(observation_context: dict[str, str]) -> bool:
    return any(observation_context.get(key) for key in CONTEXT_DIMENSIONS)


def context_signature(context: dict[str, str]) -> str:
    parts = [f"{key}={context[key]}" for key in sorted(context)]
    return "|".join(parts)


def build_slot(
    *,
    subject: str,
    prop: str,
    context: dict[str, str],
    value: str,
    confidence: float,
    provenance_refs: list[str],
) -> dict[str, Any]:
    basis = {
        "subject": subject,
        "property": prop,
        "context": context,
        "value": value,
        "provenance_refs": provenance_refs,
    }
    slot_id = f"ctx_{stable_hash(basis)}"
    memory_key = f"nx16:{subject}:{prop}:{stable_hash(context, length=8)}"
    return {
        "slot_id": slot_id,
        "memory_key": memory_key,
        "subject": subject,
        "property": prop,
        "context": context,
        "context_signature": context_signature(context),
        "value": value,
        "confidence": confidence,
        "provenance_refs": provenance_refs,
        "replay_hash": stable_hash(basis, length=24),
        "compact_record": (
            f"{memory_key}|{value}|ctx={context_signature(context)}|c={confidence:.3f}"
        ),
    }


def seed_contextual_slots(packet_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    packets = packet_artifact.get("packets", [])
    provenance_refs = [packet["packet_id"] for packet in packets[:2]] or ["fixture://stage16/seed"]
    return [
        build_slot(
            subject="sky",
            prop="expected_color",
            context={
                "time_of_day": "daytime",
                "weather": "clear",
                "location": "earth_surface",
            },
            value="blue",
            confidence=0.94,
            provenance_refs=provenance_refs,
        ),
        build_slot(
            subject="sky",
            prop="expected_color",
            context={
                "time_of_day": "night",
                "location": "earth_surface",
            },
            value="dark",
            confidence=0.96,
            provenance_refs=provenance_refs,
        ),
        build_slot(
            subject="sky",
            prop="expected_color",
            context={
                "time_of_day": "sunset",
                "weather": "clear",
                "location": "earth_surface",
            },
            value="orange_red",
            confidence=0.88,
            provenance_refs=provenance_refs,
        ),
        build_slot(
            subject="sky",
            prop="expected_color",
            context={
                "weather": "cloudy",
                "location": "earth_surface",
            },
            value="gray_variable",
            confidence=0.82,
            provenance_refs=provenance_refs,
        ),
    ]


def build_index(slots: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for slot in slots:
        index.setdefault((slot["subject"], slot["property"]), []).append(slot)
    return index


def make_observations() -> list[dict[str, Any]]:
    return [
        {
            "observation_id": "obs_sky_day_blue",
            "subject": "sky",
            "property": "expected_color",
            "context": {
                "time_of_day": "daytime",
                "weather": "clear",
                "location": "earth_surface",
            },
            "observed_value": "blue",
            "expected_decision": "context_match",
        },
        {
            "observation_id": "obs_sky_night_black",
            "subject": "sky",
            "property": "expected_color",
            "context": {
                "time_of_day": "night",
                "location": "earth_surface",
            },
            "observed_value": "black",
            "expected_decision": "same_subject_different_context",
        },
        {
            "observation_id": "obs_sky_sunset_orange",
            "subject": "sky",
            "property": "expected_color",
            "context": {
                "time_of_day": "sunset",
                "weather": "clear",
                "location": "earth_surface",
            },
            "observed_value": "orange_red",
            "expected_decision": "context_match",
        },
        {
            "observation_id": "obs_sky_cloudy_gray",
            "subject": "sky",
            "property": "expected_color",
            "context": {
                "weather": "cloudy",
                "location": "earth_surface",
            },
            "observed_value": "gray_variable",
            "expected_decision": "context_match",
        },
        {
            "observation_id": "obs_sky_black_missing_context",
            "subject": "sky",
            "property": "expected_color",
            "context": {},
            "observed_value": "black",
            "expected_decision": "request_context",
        },
        {
            "observation_id": "obs_sky_day_green",
            "subject": "sky",
            "property": "expected_color",
            "context": {
                "time_of_day": "daytime",
                "weather": "clear",
                "location": "earth_surface",
            },
            "observed_value": "green",
            "expected_decision": "flag_conflict",
        },
    ]


def compatible_values(value_a: str, value_b: str) -> bool:
    if value_a == value_b:
        return True
    aliases = {
        frozenset({"dark", "black"}),
        frozenset({"orange_red", "red_orange"}),
        frozenset({"gray_variable", "gray"}),
    }
    return frozenset({value_a, value_b}) in aliases


def decide_observation(
    observation: dict[str, Any],
    index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    candidates = index.get((observation["subject"], observation["property"]), [])
    observation_context = observation["context"]
    if not candidates:
        return {
            "decision": "request_context",
            "matched_slot_id": None,
            "reason": "no_subject_property_slots",
        }
    if not context_is_complete(observation_context):
        return {
            "decision": "request_context",
            "matched_slot_id": None,
            "reason": "missing_context",
        }

    matching_slots = [
        slot for slot in candidates if context_matches(slot["context"], observation_context)
    ]
    if matching_slots:
        best = max(matching_slots, key=lambda slot: float(slot["confidence"]))
        if compatible_values(observation["observed_value"], best["value"]):
            exact = observation["observed_value"] == best["value"]
            return {
                "decision": "context_match" if exact else "same_subject_different_context",
                "matched_slot_id": best["slot_id"],
                "reason": "context_value_match" if exact else "context_value_alias",
            }
        return {
            "decision": "flag_conflict",
            "matched_slot_id": best["slot_id"],
            "reason": "same_context_incompatible_value",
        }

    value_compatible_slots = [
        slot for slot in candidates if compatible_values(observation["observed_value"], slot["value"])
    ]
    if value_compatible_slots:
        best = max(value_compatible_slots, key=lambda slot: float(slot["confidence"]))
        return {
            "decision": "same_subject_different_context",
            "matched_slot_id": best["slot_id"],
            "reason": "value_known_under_different_context",
        }

    return {
        "decision": "request_context",
        "matched_slot_id": None,
        "reason": "known_subject_unknown_context_value_pair",
    }


def run_policy(
    slots: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    index = build_index(slots)
    episodes: list[dict[str, Any]] = []
    for observation in observations:
        decision = decide_observation(observation, index)
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
                "false_conflict": actual == "flag_conflict" and expected != "flag_conflict",
            }
        )
    return {
        "episodes": episodes,
        "decision_accuracy": mean([1.0 if episode["decision_correct"] else 0.0 for episode in episodes]),
        "decision_types_present": sorted({episode["decision"] for episode in episodes}),
    }


def scale_research_track() -> dict[str, Any]:
    return {
        "json_role": "smoke_artifact_only",
        "target_record_count": 10_000_000,
        "required_indexes": [
            "subject",
            "property",
            "context_dimensions",
            "time",
            "confidence",
            "provenance",
        ],
        "storage_candidates": [
            "SQLite/DuckDB for local indexed smoke runs",
            "Postgres for durable relational conditional memory",
            "Parquet/Arrow for columnar replay and batch analysis",
            "RocksDB/LSM for large append-friendly key-value memory logs",
            "hybrid symbolic plus vector index for contextual retrieval",
        ],
        "graph_db_default": False,
        "scale_requirements": [
            "append-only evidence log",
            "compacted memory snapshots",
            "deterministic replay from log plus snapshot hash",
            "provenance-preserving updates",
            "bounded lookup latency for subject/property/context queries",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    packet_artifact = load_packets(Path(args.packet_json))
    packet_validation = validate_packet_artifact(packet_artifact)
    slots = seed_contextual_slots(packet_artifact)
    observations = make_observations()
    result = run_policy(slots, observations)
    replay_result = run_policy(slots, observations)
    deterministic_replay = result["episodes"] == replay_result["episodes"]
    decision_types_present = set(result["decision_types_present"])
    false_conflict_rate = mean([1.0 if episode["false_conflict"] else 0.0 for episode in result["episodes"]])
    missing_context_episodes = [
        episode for episode in result["episodes"] if not episode["context"]
    ]
    missing_context_detection_rate = mean(
        [
            1.0 if episode["decision"] == "request_context" else 0.0
            for episode in missing_context_episodes
        ]
    )
    scale_track = scale_research_track()
    protocol_evidence = {
        "valid": bool(packet_validation["valid"] and slots and observations and deterministic_replay),
        "packet_json": args.packet_json,
        "packet_count": packet_validation["packet_count"],
        "slot_count": len(slots),
        "observation_count": len(observations),
        "deterministic_replay": deterministic_replay,
        "decision_types_present": sorted(decision_types_present),
        "json_role": scale_track["json_role"],
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "packet_artifact_valid": packet_validation["valid"],
        "conditional_slot_count": len(slots) > 0,
        "required_decision_types_present": DECISION_TYPES.issubset(decision_types_present),
        "decision_accuracy": result["decision_accuracy"] >= args.decision_accuracy_min,
        "false_contextual_conflict_rate": false_conflict_rate == 0.0,
        "missing_context_detection_rate": missing_context_detection_rate == 1.0,
        "deterministic_replay": deterministic_replay,
        "scale_research_track_included": scale_track["target_record_count"] >= 10_000_000,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage16_contextual_memory",
        "config": vars(args),
        "packet_validation": packet_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "decision_accuracy": result["decision_accuracy"],
            "false_contextual_conflict_rate": false_conflict_rate,
            "missing_context_detection_rate": missing_context_detection_rate,
            "slot_count": len(slots),
            "observation_count": len(observations),
        },
        "conditional_slots": slots,
        "episodes": result["episodes"],
        "scale_research_track": scale_track,
        "claim_guardrail": (
            "Stage 16 tests conditional/context-aware memory over cached fixtures. "
            "It is not a scalable storage benchmark, consciousness claim, or AGI claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 16 contextual memory failed: {pass_fail}")
    print("NeoCortex-X Stage 16 contextual memory")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  slots: {len(slots)}")
    print(f"  observations: {len(observations)}")
    print(f"  decision_accuracy: {result['decision_accuracy']:.4f}")
    print(f"  false_contextual_conflict_rate: {false_conflict_rate:.4f}")
    print(f"  missing_context_detection_rate: {missing_context_detection_rate:.4f}")
    print(f"  decision_types: {','.join(protocol_evidence['decision_types_present'])}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 16 contextual memory.")
    parser.add_argument("--packet-json", default="artifacts/neocortex_x/stage6_evidence_packets.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage16_contextual_memory.json")
    parser.add_argument("--decision-accuracy-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
