#!/usr/bin/env python3
"""Stage 33 clarification feedback ledger for NeoCortex-X memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def validate_stage32_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if artifact.get("kind") != "neocortex_x_stage32_clarification_resolution_loop":
        errors.append("invalid stage32 kind")
    if artifact.get("pass") is not True:
        errors.append("stage32 artifact did not pass")
    episodes = artifact.get("resolution", {}).get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("stage32 resolution episodes missing")
        episodes = []
    required = {
        "query_id",
        "target_slot_id",
        "initial_candidate_count",
        "final_candidate_count",
        "steps",
        "resolved_unique",
        "target_preserved",
    }
    for index, episode in enumerate(episodes):
        missing = required.difference(episode)
        if missing:
            errors.append(f"episode {index} missing {sorted(missing)}")
    return {"valid": not errors, "errors": errors, "episode_count": len(episodes)}


def score_step(episode: dict[str, Any], step: dict[str, Any], step_index: int) -> dict[str, Any]:
    before_count = int(step["before_candidate_count"])
    after_count = int(step["after_candidate_count"])
    reduced = bool(step["reduced_candidates"])
    target_preserved = bool(step["target_preserved"])
    reduction_ratio = (before_count - after_count) / before_count if before_count else 0.0
    success = reduced and target_preserved
    reward = reduction_ratio
    if target_preserved:
        reward += 0.25
    if after_count == 1:
        reward += 0.25
    if not success:
        reward = -1.0
    basis = {
        "query_id": episode["query_id"],
        "step_index": step_index,
        "asked_key": step["asked_key"],
        "answer_value": step["answer_value"],
        "before": before_count,
        "after": after_count,
        "target_preserved": target_preserved,
        "reward": round(reward, 6),
    }
    return {
        "feedback_id": f"clarfb_{stable_hash(basis)}",
        "event_type": "clarification_step_feedback",
        "query_id": episode["query_id"],
        "target_slot_id": episode["target_slot_id"],
        "step_index": step_index,
        "asked_key": step["asked_key"],
        "answer_value": step["answer_value"],
        "before_candidate_count": before_count,
        "after_candidate_count": after_count,
        "reduction_ratio": reduction_ratio,
        "target_preserved": target_preserved,
        "success": success,
        "reward": reward,
        "reinforce_key": step["asked_key"] if success else None,
        "needs_policy_review": not success,
        "target_loss_penalty": not target_preserved,
        "replay_hash": stable_hash(basis, length=24),
    }


def score_episode(episode: dict[str, Any]) -> dict[str, Any]:
    success = bool(episode["resolved_unique"] and episode["target_preserved"])
    initial_count = int(episode["initial_candidate_count"])
    final_count = int(episode["final_candidate_count"])
    steps = episode["steps"]
    if initial_count <= 1:
        reward = 0.4 if success else -1.0
        event_type = "unique_context_answer_feedback"
    else:
        reward = 1.0 if success else -1.0
        reward -= 0.05 * max(0, len(steps) - 1)
        event_type = "clarification_resolution_feedback"
    basis = {
        "query_id": episode["query_id"],
        "event_type": event_type,
        "initial": initial_count,
        "final": final_count,
        "steps": len(steps),
        "success": success,
        "reward": round(reward, 6),
    }
    return {
        "feedback_id": f"clarep_{stable_hash(basis)}",
        "event_type": event_type,
        "query_id": episode["query_id"],
        "target_slot_id": episode["target_slot_id"],
        "initial_candidate_count": initial_count,
        "final_candidate_count": final_count,
        "step_count": len(steps),
        "resolved_slot_id": episode.get("resolved_slot_id"),
        "success": success,
        "reward": reward,
        "reinforce_resolution": success,
        "needs_policy_review": not success,
        "target_loss_penalty": not bool(episode["target_preserved"]),
        "replay_hash": stable_hash(basis, length=24),
    }


def build_feedback_events(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for episode in episodes:
        for step_index, step in enumerate(episode["steps"], start=1):
            events.append(score_step(episode, step, step_index))
        events.append(score_episode(episode))
    return events


def aggregate_key_rewards(events: list[dict[str, Any]]) -> dict[str, Any]:
    rewards_by_key: dict[str, list[float]] = {}
    for event in events:
        key = event.get("reinforce_key")
        if not key:
            continue
        rewards_by_key.setdefault(key, []).append(float(event["reward"]))
    return {
        key: {
            "count": len(rewards),
            "mean_reward": mean(rewards),
            "total_reward": sum(rewards),
        }
        for key, rewards in sorted(rewards_by_key.items())
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage32 = load_json(Path(args.resolution_json))
    validation = validate_stage32_artifact(stage32)
    episodes = stage32.get("resolution", {}).get("episodes", [])
    feedback_events = build_feedback_events(episodes)
    replay_events = build_feedback_events(episodes)
    deterministic_replay = feedback_events == replay_events
    step_events = [event for event in feedback_events if event["event_type"] == "clarification_step_feedback"]
    episode_events = [event for event in feedback_events if event["event_type"] != "clarification_step_feedback"]
    success_rate = mean([1.0 if event["success"] else 0.0 for event in feedback_events])
    mean_reward = mean([float(event["reward"]) for event in feedback_events])
    step_success_rate = mean([1.0 if event["success"] else 0.0 for event in step_events])
    episode_success_rate = mean([1.0 if event["success"] else 0.0 for event in episode_events])
    target_loss_penalty_count = sum(1 for event in feedback_events if event["target_loss_penalty"])
    policy_review_count = sum(1 for event in feedback_events if event["needs_policy_review"])
    replay_hashes_present = all(event.get("replay_hash") for event in feedback_events)
    key_rewards = aggregate_key_rewards(feedback_events)
    pass_fail = {
        "stage32_artifact_valid": validation["valid"],
        "feedback_events_present": bool(feedback_events),
        "episode_feedback_count_matches": len(episode_events) == validation["episode_count"],
        "clarification_step_events_present": bool(step_events),
        "success_rate": success_rate >= args.success_rate_min,
        "mean_reward": mean_reward >= args.mean_reward_min,
        "step_success_rate": step_success_rate >= args.success_rate_min,
        "episode_success_rate": episode_success_rate >= args.success_rate_min,
        "no_target_loss_penalties": target_loss_penalty_count == 0,
        "no_policy_review_needed": policy_review_count == 0,
        "key_rewards_present": bool(key_rewards),
        "replay_hashes_present": replay_hashes_present,
        "deterministic_replay": deterministic_replay,
    }
    protocol_evidence = {
        "valid": all(pass_fail.values()),
        "resolution_json": args.resolution_json,
        "episode_count": validation["episode_count"],
        "feedback_event_count": len(feedback_events),
        "step_event_count": len(step_events),
        "episode_event_count": len(episode_events),
        "deterministic_replay": deterministic_replay,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage33_clarification_feedback_ledger",
        "config": vars(args),
        "stage32_validation": validation,
        "protocol_evidence": protocol_evidence,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "metrics": {
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "step_success_rate": step_success_rate,
            "episode_success_rate": episode_success_rate,
            "target_loss_penalty_count": target_loss_penalty_count,
            "policy_review_count": policy_review_count,
        },
        "key_rewards": key_rewards,
        "feedback_events": feedback_events,
        "claim_guardrail": (
            "Stage 33 records deterministic feedback for clarification and "
            "resolution events. It is a bounded feedback ledger, not RL training."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 33 clarification feedback failed: {pass_fail}")
    print("NeoCortex-X Stage 33 clarification feedback ledger")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  feedback_events: {len(feedback_events)}")
    print(f"  step_events: {len(step_events)}")
    print(f"  success_rate: {success_rate:.4f}")
    print(f"  mean_reward: {mean_reward:.4f}")
    print(f"  key_rewards: {','.join(sorted(key_rewards))}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoCortex-X clarification feedback ledger.")
    parser.add_argument(
        "--resolution-json",
        default="artifacts/neocortex_x/stage32_clarification_resolution_loop.json",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage33_clarification_feedback_ledger.json",
    )
    parser.add_argument("--success-rate-min", type=float, default=0.95)
    parser.add_argument("--mean-reward-min", type=float, default=0.50)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
