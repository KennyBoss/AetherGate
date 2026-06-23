#!/usr/bin/env python3
"""Stage 5B cached/live-shaped provenance adapter for NeoCortex-X."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core_experiment import apply_provenance_adapter, run_open_world_task


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": mean(values), "min": float(min(values)), "max": float(max(values))}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if fixture.get("kind") != "neocortex_x_stage5b_cached_observations":
        errors.append("invalid kind")
    items = fixture.get("items")
    if not isinstance(items, list) or not items:
        errors.append("items must be a non-empty list")
        items = []
    required_observation = {
        "claim_id",
        "subject",
        "value",
        "source_id",
        "source_trust",
        "observed_at",
        "provenance_url",
    }
    for index, item in enumerate(items):
        for field in ("query", "subject", "ground_truth", "observations"):
            if field not in item:
                errors.append(f"item {index} missing {field}")
        observations = item.get("observations", [])
        if not observations:
            errors.append(f"item {index} has no observations")
        for obs_index, observation in enumerate(observations):
            missing = required_observation.difference(observation)
            if missing:
                errors.append(f"item {index} observation {obs_index} missing {sorted(missing)}")
    return {"valid": not errors, "errors": errors, "item_count": len(items)}


def fixture_item_to_task(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "task_id": f"fixture_{index}_{item['subject']}",
        "seed": 0,
        "episode": index,
        "subject": item["subject"],
        "ground_truth": item["ground_truth"],
        "observations": item["observations"],
    }


def run_agent(items: list[dict[str, Any]], *, agent: str, contradiction_margin: float) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        task = fixture_item_to_task(item, index)
        result = run_open_world_task(task, agent=agent, contradiction_margin=contradiction_margin)
        episodes.append(
            {
                "task_id": task["task_id"],
                "query": item["query"],
                "subject": item["subject"],
                "selected_value": result["selected_value"],
                "ground_truth": result["ground_truth"],
                "factual_accuracy": result["factual_accuracy"],
                "provenance_coverage": result["provenance_coverage"],
                "replay_consistency": result["replay_consistency"],
            }
        )
    return {
        "agent": agent,
        "episode_count": len(episodes),
        "factual_accuracy": mean([1.0 if item["factual_accuracy"] else 0.0 for item in episodes]),
        "provenance_coverage": mean([1.0 if item["provenance_coverage"] else 0.0 for item in episodes]),
        "replay_consistency": mean([1.0 if item["replay_consistency"] else 0.0 for item in episodes]),
        "episodes": episodes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode != "cached":
        raise SystemExit("Stage 5B live mode is reserved for a future backend; use --mode cached for smoke.")
    fixture = load_fixture(Path(args.fixture_json))
    fixture_validation = validate_fixture(fixture)
    items = fixture.get("items", [])
    agents = ("provenance_adapter", "naive_latest_control", "no_external_control")
    runs = {
        agent: run_agent(items, agent=agent, contradiction_margin=args.contradiction_margin)
        for agent in agents
    }
    adapter = runs["provenance_adapter"]
    naive = runs["naive_latest_control"]
    no_external = runs["no_external_control"]
    comparison = {
        "adapter_minus_naive_accuracy": adapter["factual_accuracy"] - naive["factual_accuracy"],
        "adapter_minus_no_external_accuracy": adapter["factual_accuracy"] - no_external["factual_accuracy"],
    }
    replay_states = [
        apply_provenance_adapter(fixture_item_to_task(item, index), args.contradiction_margin)["state"]
        for index, item in enumerate(items)
    ]
    protocol_evidence = {
        "valid": bool(args.mode == "cached" and fixture_validation["valid"] and items),
        "mode": args.mode,
        "fixture_json": args.fixture_json,
        "cache_fixture_valid": fixture_validation["valid"],
        "fixture_item_count": fixture_validation["item_count"],
        "live_backend_used": False,
        "deterministic_replay_states": len(replay_states),
    }
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "cache_fixture_valid": fixture_validation["valid"],
        "adapter_factual_accuracy": adapter["factual_accuracy"] >= args.adapter_accuracy_min,
        "adapter_provenance_coverage": adapter["provenance_coverage"] == 1.0,
        "adapter_replay_consistency": adapter["replay_consistency"] == 1.0,
        "adapter_minus_naive_accuracy": comparison["adapter_minus_naive_accuracy"] >= args.naive_margin_min,
        "adapter_minus_no_external_accuracy": comparison["adapter_minus_no_external_accuracy"] >= args.no_external_margin_min,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage5b_live_cached_adapter",
        "config": vars(args),
        "fixture_validation": fixture_validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "aggregate": {
            "agents": {agent: {k: v for k, v in run.items() if k != "episodes"} for agent, run in runs.items()},
            "comparison": comparison,
        },
        "runs": runs,
        "replay_states": replay_states,
        "claim_guardrail": (
            "Stage 5B smoke uses cached fixtures only. Live search must write this "
            "same schema before results can be treated as replayable evidence."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 5B cached adapter failed: {pass_fail}")
    print("NeoCortex-X Stage 5B live/cached adapter")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  mode: {args.mode}")
    print(f"  fixture_items: {fixture_validation['item_count']}")
    print(f"  adapter.factual_accuracy: {adapter['factual_accuracy']:.4f}")
    print(f"  adapter_minus_naive_accuracy: {comparison['adapter_minus_naive_accuracy']:.4f}")
    print(f"  adapter_minus_no_external_accuracy: {comparison['adapter_minus_no_external_accuracy']:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X Stage 5B cached/live-shaped adapter.")
    parser.add_argument("--mode", choices=("cached", "live"), default="cached")
    parser.add_argument("--fixture-json", default="research/neocortex_x/fixtures/stage5b_cached_observations.json")
    parser.add_argument("--output-json", default="artifacts/neocortex_x/stage5b_live_cached_adapter.json")
    parser.add_argument("--contradiction-margin", type=float, default=0.20)
    parser.add_argument("--adapter-accuracy-min", type=float, default=0.95)
    parser.add_argument("--naive-margin-min", type=float, default=0.25)
    parser.add_argument("--no-external-margin-min", type=float, default=0.50)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
