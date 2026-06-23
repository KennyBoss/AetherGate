#!/usr/bin/env python3
"""Stage 8 temporal freshness policy over NeoCortex-X evidence packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage7_memory_retrieval import load_packets, validate_packet_artifact


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


def make_present_observations(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index, packet in enumerate(packets):
        if index % 3 == 0:
            observed_value = packet["value"]
            source_trust = min(1.0, float(packet["confidence"]) + 0.25)
            expected_action = "keep"
        elif index % 3 == 1:
            observed_value = f"{packet['value']}_refresh"
            source_trust = 0.95
            expected_action = "refresh"
        else:
            observed_value = f"{packet['value']}_weak_conflict"
            source_trust = 0.40
            expected_action = "flag_conflict"
        observations.append(
            {
                "subject": packet["subject"],
                "packet_id": packet["packet_id"],
                "observed_value": observed_value,
                "source_id": f"present_source_{index}",
                "source_trust": source_trust,
                "observed_at": f"2026-06-24T08:{index:02d}:00Z",
                "provenance_url": f"fixture://stage8/{packet['subject']}/{index}",
                "expected_action": expected_action,
            }
        )
    return observations


def decide_action(packet: dict[str, Any], observation: dict[str, Any], *, refresh_margin: float) -> str:
    if observation["observed_value"] == packet["value"]:
        return "keep"
    packet_confidence = float(packet["confidence"])
    observation_trust = float(observation["source_trust"])
    if observation_trust - packet_confidence >= refresh_margin:
        return "refresh"
    return "flag_conflict"


def update_packet(packet: dict[str, Any], observation: dict[str, Any], action: str) -> dict[str, Any]:
    updated = dict(packet)
    event = {
        "action": action,
        "observed_value": observation["observed_value"],
        "source_id": observation["source_id"],
        "source_trust": observation["source_trust"],
        "observed_at": observation["observed_at"],
        "provenance_url": observation["provenance_url"],
    }
    history = list(packet.get("temporal_events", []))
    history.append(event)
    updated["temporal_events"] = history
    updated["freshness_action"] = action
    updated["last_observed_at"] = observation["observed_at"]
    if action == "refresh":
        updated["previous_value"] = packet["value"]
        updated["value"] = observation["observed_value"]
        updated["confidence"] = observation["source_trust"]
        updated["memory_key"] = f"nx8:{packet['subject']}:{stable_hash([packet['packet_id'], observation], length=8)}"
        updated["replay_hash"] = stable_hash([packet["replay_hash"], observation], length=24)
    elif action == "flag_conflict":
        conflicts = list(packet.get("temporal_conflicts", []))
        conflicts.append(event)
        updated["temporal_conflicts"] = conflicts
    return updated


def run_policy(
    packets: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    refresh_margin: float,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    updated_packets: list[dict[str, Any]] = []
    by_packet_id = {packet["packet_id"]: packet for packet in packets}
    for observation in observations:
        packet = by_packet_id[observation["packet_id"]]
        action = decide_action(packet, observation, refresh_margin=refresh_margin)
        updated = update_packet(packet, observation, action)
        updated_packets.append(updated)
        episodes.append(
            {
                "subject": observation["subject"],
                "packet_id": observation["packet_id"],
                "past_value": packet["value"],
                "present_value": observation["observed_value"],
                "future_action": action,
                "expected_action": observation["expected_action"],
                "action_correct": action == observation["expected_action"],
                "updated_value": updated["value"],
            }
        )
    return {
        "episodes": episodes,
        "updated_packets": updated_packets,
        "action_accuracy": mean([1.0 if item["action_correct"] else 0.0 for item in episodes]),
        "actions_present": sorted({item["future_action"] for item in episodes}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact = load_packets(Path(args.packet_json))
    packet_validation = validate_packet_artifact(artifact)
    packets = artifact.get("packets", [])
    observations = make_present_observations(packets)
    result = run_policy(packets, observations, refresh_margin=args.refresh_margin)
    replay_result = run_policy(packets, observations, refresh_margin=args.refresh_margin)
    deterministic_replay = result["episodes"] == replay_result["episodes"]
    required_actions = {"keep", "refresh", "flag_conflict"}
    actions_present = set(result["actions_present"])
    protocol_evidence = {
        "valid": bool(packet_validation["valid"] and observations and deterministic_replay),
        "packet_json": args.packet_json,
        "packet_count": packet_validation["packet_count"],
        "observation_count": len(observations),
        "deterministic_replay": deterministic_replay,
        "actions_present": sorted(actions_present),
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "packet_artifact_valid": packet_validation["valid"],
        "required_actions_present": required_actions.issubset(actions_present),
        "policy_action_accuracy": result["action_accuracy"] >= args.action_accuracy_min,
        "deterministic_replay": deterministic_replay,
        "updated_packet_count": len(result["updated_packets"]) == len(packets),
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage8_temporal_freshness",
        "config": vars(args),
        "packet_validation": packet_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "action_accuracy": result["action_accuracy"],
            "updated_packet_count": len(result["updated_packets"]),
            "keep_count": sum(1 for item in result["episodes"] if item["future_action"] == "keep"),
            "refresh_count": sum(1 for item in result["episodes"] if item["future_action"] == "refresh"),
            "flag_conflict_count": sum(
                1 for item in result["episodes"] if item["future_action"] == "flag_conflict"
            ),
        },
        "episodes": result["episodes"],
        "updated_packets": result["updated_packets"],
        "claim_guardrail": (
            "Stage 8 tests cached temporal freshness policy over evidence packets. "
            "It is not a live-search or AGI claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 8 temporal freshness failed: {pass_fail}")
    print("NeoCortex-X Stage 8 temporal freshness")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  observations: {len(observations)}")
    print(f"  action_accuracy: {result['action_accuracy']:.4f}")
    print(f"  actions_present: {','.join(protocol_evidence['actions_present'])}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 8 temporal freshness policy.")
    parser.add_argument("--packet-json", default="artifacts/neocortex_x/stage6_evidence_packets.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage8_temporal_freshness.json")
    parser.add_argument("--refresh-margin", type=float, default=0.20)
    parser.add_argument("--action-accuracy-min", type=float, default=0.95)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
