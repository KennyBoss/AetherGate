#!/usr/bin/env python3
"""Stage 7 long-term memory retrieval over NeoCortex-X evidence packets."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_packets(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_packet_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if artifact.get("kind") != "neocortex_x_stage6_evidence_packets":
        errors.append("invalid kind")
    if artifact.get("pass") is not True:
        errors.append("stage6 packet artifact did not pass")
    packets = artifact.get("packets")
    if not isinstance(packets, list) or not packets:
        errors.append("packets must be a non-empty list")
        packets = []
    required = {"packet_id", "memory_key", "subject", "value", "supporting_claim_ids"}
    for index, packet in enumerate(packets):
        missing = required.difference(packet)
        if missing:
            errors.append(f"packet {index} missing {sorted(missing)}")
    return {"valid": not errors, "errors": errors, "packet_count": len(packets)}


def build_memory_index(packets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for packet in packets:
        index[packet["subject"]] = packet
        index[packet["memory_key"]] = packet
    return index


def make_queries(packets: list[dict[str, Any]]) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    for packet in packets:
        queries.append(
            {
                "query": packet["subject"],
                "lookup_key": packet["subject"],
                "subject": packet["subject"],
                "expected_value": packet["value"],
                "query_type": "subject",
            }
        )
        queries.append(
            {
                "query": f"{packet['subject']} current_value",
                "lookup_key": packet["subject"],
                "subject": packet["subject"],
                "expected_value": packet["value"],
                "query_type": "compact_natural",
            }
        )
        queries.append(
            {
                "query": packet["memory_key"],
                "lookup_key": packet["memory_key"],
                "subject": packet["subject"],
                "expected_value": packet["value"],
                "query_type": "memory_key",
            }
        )
    return queries


def retrieve_packet(
    query: dict[str, str],
    *,
    agent: str,
    index: dict[str, dict[str, Any]],
    packets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if agent == "packet_memory":
        return index.get(query["lookup_key"])
    if agent == "latest_string_control":
        return packets[-1] if packets else None
    if agent == "no_memory_control":
        return None
    raise ValueError(f"Unknown Stage 7 agent: {agent}")


def run_agent(
    queries: list[dict[str, str]],
    *,
    agent: str,
    index: dict[str, dict[str, Any]],
    packets: list[dict[str, Any]],
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for query in queries:
        packet = retrieve_packet(query, agent=agent, index=index, packets=packets)
        selected_value = packet["value"] if packet else None
        provenance_claim_ids = packet.get("supporting_claim_ids", []) if packet else []
        episodes.append(
            {
                "query": query["query"],
                "query_type": query["query_type"],
                "subject": query["subject"],
                "selected_value": selected_value,
                "expected_value": query["expected_value"],
                "factual_accuracy": selected_value == query["expected_value"],
                "provenance_coverage": bool(provenance_claim_ids),
                "packet_id": packet.get("packet_id") if packet else None,
                "memory_key": packet.get("memory_key") if packet else None,
                "supporting_claim_ids": provenance_claim_ids,
            }
        )
    return {
        "agent": agent,
        "episode_count": len(episodes),
        "factual_accuracy": mean([1.0 if item["factual_accuracy"] else 0.0 for item in episodes]),
        "provenance_coverage": mean([1.0 if item["provenance_coverage"] else 0.0 for item in episodes]),
        "episodes": episodes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_packets(Path(args.packet_json))
    packet_validation = validate_packet_artifact(artifact)
    packets = artifact.get("packets", [])
    index = build_memory_index(packets)
    queries = make_queries(packets)
    agents = ("packet_memory", "latest_string_control", "no_memory_control")
    runs = {
        agent: run_agent(queries, agent=agent, index=index, packets=packets)
        for agent in agents
    }
    memory = runs["packet_memory"]
    latest = runs["latest_string_control"]
    no_memory = runs["no_memory_control"]
    comparison = {
        "packet_minus_latest_accuracy": memory["factual_accuracy"] - latest["factual_accuracy"],
        "packet_minus_no_memory_accuracy": memory["factual_accuracy"] - no_memory["factual_accuracy"],
    }
    protocol_evidence = {
        "valid": bool(packet_validation["valid"] and queries),
        "packet_json": args.packet_json,
        "packet_count": packet_validation["packet_count"],
        "query_count": len(queries),
        "index_entry_count": len(index),
        "query_types": sorted({query["query_type"] for query in queries}),
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "packet_artifact_valid": packet_validation["valid"],
        "memory_factual_accuracy": memory["factual_accuracy"] >= args.memory_accuracy_min,
        "memory_provenance_coverage": memory["provenance_coverage"] == 1.0,
        "packet_minus_latest_accuracy": (
            comparison["packet_minus_latest_accuracy"] >= args.latest_margin_min
        ),
        "packet_minus_no_memory_accuracy": (
            comparison["packet_minus_no_memory_accuracy"] >= args.no_memory_margin_min
        ),
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage7_memory_retrieval",
        "config": vars(args),
        "packet_validation": packet_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": {
            "agents": {agent: {k: v for k, v in run.items() if k != "episodes"} for agent, run in runs.items()},
            "comparison": comparison,
        },
        "runs": runs,
        "claim_guardrail": (
            "Stage 7 tests retrieval from Stage 6 evidence packets. It is not a broad "
            "semantic-memory or live internet claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 7 memory retrieval failed: {pass_fail}")
    print("NeoCortex-X Stage 7 memory retrieval")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  packets: {packet_validation['packet_count']}")
    print(f"  queries: {len(queries)}")
    print(f"  memory.factual_accuracy: {memory['factual_accuracy']:.4f}")
    print(f"  packet_minus_latest_accuracy: {comparison['packet_minus_latest_accuracy']:.4f}")
    print(f"  packet_minus_no_memory_accuracy: {comparison['packet_minus_no_memory_accuracy']:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 7 memory retrieval.")
    parser.add_argument("--packet-json", default="artifacts/neocortex_x/stage6_evidence_packets.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage7_memory_retrieval.json")
    parser.add_argument("--memory-accuracy-min", type=float, default=0.95)
    parser.add_argument("--latest-margin-min", type=float, default=0.25)
    parser.add_argument("--no-memory-margin-min", type=float, default=0.50)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
