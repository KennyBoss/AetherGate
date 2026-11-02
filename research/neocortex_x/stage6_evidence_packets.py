#!/usr/bin/env python3
"""Stage 6 evidence-packet builder for NeoCortex-X long-term memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_experiment import apply_provenance_adapter
from stage5b_live_cached_adapter import fixture_item_to_task, load_fixture, validate_fixture


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


def build_packet(item: dict[str, Any], index: int, *, contradiction_margin: float) -> dict[str, Any]:
    task = fixture_item_to_task(item, index)
    adapter_state = apply_provenance_adapter(task, contradiction_margin)["state"]
    selected_provenance = adapter_state["provenance"]
    all_scores = [float(obs["source_trust"]) for obs in item["observations"]]
    total_trust = sum(all_scores)
    selected_score = float(adapter_state["score"])
    confidence = selected_score / total_trust if total_trust else 0.0
    supporting_claim_ids = [obs["claim_id"] for obs in selected_provenance]
    rejected_values = [
        {
            "value": rejected["value"],
            "score": float(rejected["score"]),
            "claim_ids": [obs["claim_id"] for obs in rejected["provenance"]],
        }
        for rejected in adapter_state["rejected"]
    ]
    replay_basis = {
        "subject": adapter_state["subject"],
        "value": adapter_state["value"],
        "supporting_claim_ids": supporting_claim_ids,
        "rejected_values": rejected_values,
        "contradiction": bool(adapter_state["contradiction"]),
    }
    packet_id = f"evp_{stable_hash(replay_basis)}"
    memory_key = f"nx6:{adapter_state['subject']}:{stable_hash(replay_basis, length=8)}"
    return {
        "packet_id": packet_id,
        "memory_key": memory_key,
        "subject": adapter_state["subject"],
        "value": adapter_state["value"],
        "ground_truth": item["ground_truth"],
        "factual_accuracy": adapter_state["value"] == item["ground_truth"],
        "confidence": confidence,
        "selected_score": selected_score,
        "total_trust": total_trust,
        "supporting_claim_ids": supporting_claim_ids,
        "supporting_sources": [
            {
                "source_id": obs["source_id"],
                "source_trust": obs["source_trust"],
                "observed_at": obs["observed_at"],
                "provenance_url": obs["provenance_url"],
            }
            for obs in selected_provenance
        ],
        "rejected_values": rejected_values,
        "contradiction": bool(adapter_state["contradiction"]),
        "replay_hash": stable_hash(replay_basis, length=24),
        "compact_record": f"{memory_key}|{adapter_state['value']}|c={confidence:.3f}",
    }


def build_packets(items: list[dict[str, Any]], *, contradiction_margin: float) -> list[dict[str, Any]]:
    return [
        build_packet(item, index, contradiction_margin=contradiction_margin)
        for index, item in enumerate(items)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture = load_fixture(Path(args.fixture_json))
    fixture_validation = validate_fixture(fixture)
    items = fixture.get("items", [])
    packets = build_packets(items, contradiction_margin=args.contradiction_margin)
    replay_packets = build_packets(items, contradiction_margin=args.contradiction_margin)
    packet_ids = [packet["packet_id"] for packet in packets]
    memory_keys = [packet["memory_key"] for packet in packets]
    deterministic_replay = packets == replay_packets
    factual_accuracy = mean([1.0 if packet["factual_accuracy"] else 0.0 for packet in packets])
    provenance_coverage = mean([1.0 if packet["supporting_claim_ids"] else 0.0 for packet in packets])
    compact_key_unique = len(memory_keys) == len(set(memory_keys))
    protocol_evidence = {
        "valid": bool(fixture_validation["valid"] and items and deterministic_replay),
        "fixture_json": args.fixture_json,
        "fixture_item_count": fixture_validation["item_count"],
        "packet_count": len(packets),
        "deterministic_replay": deterministic_replay,
        "packet_id_unique": len(packet_ids) == len(set(packet_ids)),
        "compact_key_unique": compact_key_unique,
        "live_backend_used": False,
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "cache_fixture_valid": fixture_validation["valid"],
        "packet_count_matches_fixture": len(packets) == fixture_validation["item_count"],
        "factual_accuracy": factual_accuracy >= args.factual_accuracy_min,
        "provenance_coverage": provenance_coverage == 1.0,
        "deterministic_replay": deterministic_replay,
        "compact_key_unique": compact_key_unique,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage6_evidence_packets",
        "config": vars(args),
        "fixture_validation": fixture_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "factual_accuracy": factual_accuracy,
            "provenance_coverage": provenance_coverage,
            "packet_count": len(packets),
            "contradiction_count": sum(1 for packet in packets if packet["contradiction"]),
        },
        "packets": packets,
        "claim_guardrail": (
            "Stage 6 converts replayable provenance states into memory evidence packets. "
            "It is not a live-search, consciousness, or AGI claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 6 evidence packets failed: {pass_fail}")
    print("NeoCortex-X Stage 6 evidence packets")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  packets: {len(packets)}")
    print(f"  factual_accuracy: {factual_accuracy:.4f}")
    print(f"  provenance_coverage: {provenance_coverage:.4f}")
    print(f"  deterministic_replay: {deterministic_replay}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NeoCortex-X Stage 6 evidence packets.")
    parser.add_argument("--fixture-json", default="research/neocortex_x/fixtures/stage5b_cached_observations.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage6_evidence_packets.json")
    parser.add_argument("--contradiction-margin", type=float, default=0.20)
    parser.add_argument("--factual-accuracy-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
