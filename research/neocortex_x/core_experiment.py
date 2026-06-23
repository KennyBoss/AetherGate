#!/usr/bin/env python3
"""NeoCortex-X Stage 0/1 controlled experiment harness.

This is intentionally small: it validates the trace/replay contract and the
recall-to-action protocol before any trained model or CodePy procedure memory
is plugged in.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGENTS = ("kv_memory", "context_window")
TRACE_REQUIRED_FIELDS = (
    "tick_id",
    "phase",
    "event_type",
    "observation",
    "gates",
    "predicted_delta",
    "actual_delta",
    "state_before_digest",
    "state_after_digest",
    "reward",
    "active_compute",
)
GATE_KEYS = ("read_memory", "write_memory", "env_action", "call_codepy", "stop")
COMPUTE_KEYS = (
    "total_ticks",
    "reads",
    "writes",
    "env_actions",
    "codepy_calls",
    "stops",
    "context_scans",
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def state_digest(state: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(state).encode("utf-8")).hexdigest()[:20]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_csv_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit("CSV integer argument must contain at least one value.")
    return items


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def empty_gates() -> dict[str, bool]:
    return {key: False for key in GATE_KEYS}


def initial_state(agent: str) -> dict[str, Any]:
    return {
        "agent": agent,
        "memory": {},
        "history": [],
        "pending_query_key": None,
        "pending_expected_value": None,
        "last_read_value": None,
        "last_action_value": None,
        "last_action_correct": None,
        "halted": False,
        "errors": [],
        "active_compute": {key: 0 for key in COMPUTE_KEYS},
    }


def value_name(index: int) -> str:
    return f"value_{index}"


def key_name(index: int) -> str:
    return f"key_{index}"


def fallback_action_value(value_count: int) -> str:
    if value_count < 1:
        raise ValueError("value_count must be positive.")
    return value_name(0)


def lookup_binding(history: list[dict[str, Any]], key: str) -> str | None:
    for event in reversed(history):
        if event["type"] == "bind" and event["key"] == key:
            return event["value"]
    return None


def visible_history(state: dict[str, Any], agent: str, baseline_window: int) -> list[dict[str, Any]]:
    if agent == "context_window":
        if baseline_window <= 0:
            return []
        return state["history"][-baseline_window:]
    return state["history"][-4:]


def make_episode_events(
    *,
    seed: int,
    episode: int,
    key_count: int,
    value_count: int,
    binding_count: int,
    delay: int,
) -> list[dict[str, Any]]:
    if binding_count < 1:
        raise ValueError("binding_count must be positive.")
    if binding_count > key_count:
        raise ValueError("binding_count must not exceed key_count.")
    if value_count < 1:
        raise ValueError("value_count must be positive.")
    if delay < 0:
        raise ValueError("delay must be non-negative.")

    rng = random.Random(seed * 1_000_003 + episode)
    key_ids = rng.sample(range(key_count), binding_count)
    query_slot = rng.randrange(binding_count)
    query_value_id = (episode + seed) % value_count
    used_values = {query_value_id}
    value_ids: list[int] = []
    for slot in range(binding_count):
        if slot == query_slot:
            value_ids.append(query_value_id)
            continue
        candidate = (query_value_id + slot + 1 + rng.randrange(value_count)) % value_count
        while candidate in used_values and len(used_values) < value_count:
            candidate = (candidate + 1) % value_count
        value_ids.append(candidate)
        used_values.add(candidate)

    events: list[dict[str, Any]] = []
    for slot, (key_id, value_id) in enumerate(zip(key_ids, value_ids)):
        events.append(
            {
                "type": "bind",
                "episode": episode,
                "slot": slot,
                "key": key_name(key_id),
                "value": value_name(value_id),
            }
        )
    for offset in range(delay):
        events.append(
            {
                "type": "filler",
                "episode": episode,
                "offset": offset,
                "token": f"noise_{(seed + episode + offset) % 997}",
            }
        )
    events.append(
        {
            "type": "query",
            "episode": episode,
            "key": key_name(key_ids[query_slot]),
        }
    )
    events.append({"type": "act", "episode": episode})
    events.append({"type": "stop", "episode": episode})
    return events


def active_compute_increment(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int]:
    return {
        key: int(after["active_compute"][key] - before["active_compute"][key])
        for key in COMPUTE_KEYS
    }


def build_delta(
    before: dict[str, Any], after: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    memory_write = None
    if event["type"] == "bind" and after["memory"] != before["memory"]:
        memory_write = {"key": event["key"], "value": event["value"]}
    return {
        "history_appended": event,
        "memory_write": memory_write,
        "memory_size": len(after["memory"]),
        "pending_query_key": after["pending_query_key"],
        "pending_expected_value": after["pending_expected_value"],
        "last_read_value": after["last_read_value"],
        "last_action_value": after["last_action_value"],
        "last_action_correct": after["last_action_correct"],
        "halted": after["halted"],
        "errors_added": after["errors"][len(before["errors"]) :],
        "active_compute_increment": active_compute_increment(before, after),
    }


def transition(
    state: dict[str, Any],
    event: dict[str, Any],
    *,
    agent: str,
    baseline_window: int,
    value_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool], dict[str, Any], float]:
    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent}")

    before = copy.deepcopy(state)
    after = copy.deepcopy(state)
    gates = empty_gates()
    action: dict[str, Any] = {"kind": "observe"}
    reward = 0.0
    phase = event["type"]

    if phase == "bind":
        if agent == "kv_memory":
            gates["write_memory"] = True
            after["memory"][event["key"]] = event["value"]
            action = {
                "kind": "write_memory",
                "key": event["key"],
                "value": event["value"],
            }
        else:
            action = {"kind": "observe_context", "key": event["key"]}
    elif phase == "filler":
        action = {"kind": "ignore_filler", "token": event["token"]}
    elif phase == "query":
        key = event["key"]
        expected_value = lookup_binding(before["history"], key)
        if agent == "kv_memory":
            gates["read_memory"] = True
            read_value = before["memory"].get(key)
            action = {"kind": "read_memory", "key": key, "value": read_value}
        else:
            read_value = lookup_binding(visible_history(before, agent, baseline_window), key)
            action = {
                "kind": "scan_context",
                "key": key,
                "window": baseline_window,
                "value": read_value,
            }
        after["pending_query_key"] = key
        after["pending_expected_value"] = expected_value
        after["last_read_value"] = read_value
    elif phase == "act":
        gates["env_action"] = True
        expected_value = before["pending_expected_value"]
        chosen_value = before["last_read_value"] or fallback_action_value(value_count)
        correct = expected_value is not None and chosen_value == expected_value
        action = {"kind": "emit_value", "value": chosen_value}
        after["last_action_value"] = chosen_value
        after["last_action_correct"] = correct
        reward = 1.0 if correct else 0.0
        if not correct:
            after["errors"].append(
                {
                    "phase": phase,
                    "expected": expected_value,
                    "actual": chosen_value,
                }
            )
    elif phase == "stop":
        gates["stop"] = True
        after["halted"] = True
        action = {"kind": "stop"}
    else:
        raise ValueError(f"Unknown event type: {phase}")

    after["history"].append(copy.deepcopy(event))
    after["active_compute"]["total_ticks"] += 1
    if gates["read_memory"]:
        after["active_compute"]["reads"] += 1
    if gates["write_memory"]:
        after["active_compute"]["writes"] += 1
    if gates["env_action"]:
        after["active_compute"]["env_actions"] += 1
    if gates["call_codepy"]:
        after["active_compute"]["codepy_calls"] += 1
    if gates["stop"]:
        after["active_compute"]["stops"] += 1
    if action["kind"] == "scan_context":
        after["active_compute"]["context_scans"] += 1

    delta = build_delta(before, after, event)
    return after, delta, gates, action, reward


def execute_episode(
    events: list[dict[str, Any]],
    *,
    agent: str,
    baseline_window: int,
    value_count: int,
    tick_start: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = initial_state(agent)
    records: list[dict[str, Any]] = []
    read_correct = 0
    query_count = 0
    action_correct = 0
    action_count = 0
    exact_delta_count = 0
    first_error_tick: int | None = None

    for local_tick, event in enumerate(events):
        tick_id = tick_start + local_tick
        before_digest = state_digest(state)
        observation = {
            "event": copy.deepcopy(event),
            "agent_visible_history": visible_history(state, agent, baseline_window),
        }
        predicted_after, predicted_delta, gates, action, predicted_reward = transition(
            state,
            event,
            agent=agent,
            baseline_window=baseline_window,
            value_count=value_count,
        )
        actual_after, actual_delta, actual_gates, actual_action, reward = transition(
            state,
            event,
            agent=agent,
            baseline_window=baseline_window,
            value_count=value_count,
        )
        if predicted_after != actual_after:
            raise RuntimeError("Deterministic predictor diverged from transition.")
        if predicted_delta == actual_delta:
            exact_delta_count += 1

        record = {
            "tick_id": tick_id,
            "episode": event.get("episode"),
            "agent": agent,
            "phase": event["type"],
            "event_type": event["type"],
            "observation": observation,
            "gates": actual_gates,
            "action": actual_action,
            "predicted_delta": predicted_delta,
            "actual_delta": actual_delta,
            "state_before_digest": before_digest,
            "state_after_digest": state_digest(actual_after),
            "reward": reward,
            "predicted_reward": predicted_reward,
            "active_compute": copy.deepcopy(actual_after["active_compute"]),
        }
        records.append(record)

        if event["type"] == "query":
            query_count += 1
            if actual_after["last_read_value"] == actual_after["pending_expected_value"]:
                read_correct += 1
        if event["type"] == "act":
            action_count += 1
            if actual_after["last_action_correct"]:
                action_correct += 1
        if first_error_tick is None and len(actual_after["errors"]) > len(state["errors"]):
            first_error_tick = local_tick + 1

        state = actual_after

    tick_count = len(events)
    metrics = {
        "tick_count": tick_count,
        "query_count": query_count,
        "action_count": action_count,
        "read_correct": read_correct,
        "action_correct": action_correct,
        "write_read_fidelity": read_correct / query_count if query_count else 0.0,
        "action_accuracy": action_correct / action_count if action_count else 0.0,
        "trajectory_horizon": first_error_tick if first_error_tick is not None else tick_count,
        "world_delta_exact_rate": exact_delta_count / tick_count if tick_count else 0.0,
        "active_compute": copy.deepcopy(state["active_compute"]),
    }
    return records, metrics


def replay_trace_records(
    records: list[dict[str, Any]],
    *,
    agent: str,
    baseline_window: int,
    value_count: int,
) -> dict[str, Any]:
    state = initial_state(agent)
    mismatches: list[dict[str, Any]] = []
    delta_mismatches: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        event = record["observation"]["event"]
        after, delta, _gates, _action, _reward = transition(
            state,
            event,
            agent=agent,
            baseline_window=baseline_window,
            value_count=value_count,
        )
        digest = state_digest(after)
        if digest != record["state_after_digest"]:
            mismatches.append(
                {
                    "index": index,
                    "tick_id": record["tick_id"],
                    "expected_digest": record["state_after_digest"],
                    "actual_digest": digest,
                }
            )
        if delta != record["actual_delta"]:
            delta_mismatches.append({"index": index, "tick_id": record["tick_id"]})
        state = after
    return {
        "valid": not mismatches and not delta_mismatches,
        "total_ticks": len(records),
        "state_mismatch_count": len(mismatches),
        "delta_mismatch_count": len(delta_mismatches),
        "mismatches": mismatches[:8],
        "delta_mismatches": delta_mismatches[:8],
    }


def validate_trace_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    bad_gates: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        absent = [field for field in TRACE_REQUIRED_FIELDS if field not in record]
        if absent:
            missing.append({"index": index, "missing": absent})
        gates = record.get("gates", {})
        absent_gates = [key for key in GATE_KEYS if key not in gates]
        if absent_gates:
            bad_gates.append({"index": index, "missing_gates": absent_gates})
    phases = sorted({record.get("phase") for record in records})
    required_phases = {"bind", "filler", "query", "act", "stop"}
    return {
        "valid": not missing and not bad_gates and required_phases.issubset(set(phases)),
        "record_count": len(records),
        "phases": phases,
        "required_phases_present": required_phases.issubset(set(phases)),
        "missing_fields": missing[:8],
        "bad_gates": bad_gates[:8],
    }


def combine_episode_metrics(episode_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total_queries = sum(int(m["query_count"]) for m in episode_metrics)
    total_actions = sum(int(m["action_count"]) for m in episode_metrics)
    total_read_correct = sum(int(m["read_correct"]) for m in episode_metrics)
    total_action_correct = sum(int(m["action_correct"]) for m in episode_metrics)
    active_compute = {key: 0 for key in COMPUTE_KEYS}
    for metrics in episode_metrics:
        for key in COMPUTE_KEYS:
            active_compute[key] += int(metrics["active_compute"][key])
    return {
        "episode_count": len(episode_metrics),
        "query_count": total_queries,
        "action_count": total_actions,
        "write_read_fidelity": total_read_correct / total_queries if total_queries else 0.0,
        "action_accuracy": total_action_correct / total_actions if total_actions else 0.0,
        "trajectory_horizon_mean": mean(
            [float(m["trajectory_horizon"]) for m in episode_metrics]
        ),
        "world_delta_exact_rate": mean(
            [float(m["world_delta_exact_rate"]) for m in episode_metrics]
        ),
        "active_compute": active_compute,
    }


def run_stage0(args: argparse.Namespace) -> dict[str, Any]:
    events = make_episode_events(
        seed=args.seed,
        episode=0,
        key_count=args.key_count,
        value_count=args.value_count,
        binding_count=args.binding_count,
        delay=args.delay,
    )
    records, metrics = execute_episode(
        events,
        agent="kv_memory",
        baseline_window=args.baseline_window,
        value_count=args.value_count,
    )
    output_trace = Path(args.output_trace)
    write_jsonl(output_trace, records)

    replay = replay_trace_records(
        read_jsonl(output_trace),
        agent="kv_memory",
        baseline_window=args.baseline_window,
        value_count=args.value_count,
    )
    schema = validate_trace_schema(records)
    world_delta_exact_rate = mean(
        [1.0 if record["predicted_delta"] == record["actual_delta"] else 0.0 for record in records]
    )
    protocol_evidence = {
        "valid": bool(
            schema["valid"]
            and replay["valid"]
            and metrics["active_compute"]["total_ticks"] > 0
            and world_delta_exact_rate == 1.0
        ),
        "trace_schema_valid": schema["valid"],
        "replay_valid": replay["valid"],
        "required_phases_present": schema["required_phases_present"],
        "active_compute_positive": metrics["active_compute"]["total_ticks"] > 0,
        "world_delta_exact": world_delta_exact_rate == 1.0,
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage0_instrumentation_smoke",
        "config": vars(args),
        "trace_path": str(output_trace),
        "metrics": {
            **metrics,
            "world_delta_exact_rate": world_delta_exact_rate,
        },
        "trace_schema": schema,
        "replay_check": replay,
        "protocol_evidence": protocol_evidence,
        "claim_guardrail": (
            "Stage 0 validates deterministic instrumentation and replay only; "
            "it is not a learning or consciousness claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not protocol_evidence["valid"]:
        raise SystemExit(f"Stage 0 protocol failed: {protocol_evidence}")
    print("NeoCortex-X Stage 0 smoke")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  trace_ticks: {len(records)}")
    print(f"  world_delta_exact_rate: {world_delta_exact_rate:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def run_agent_seed(
    *,
    seed: int,
    agent: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    episode_metrics: list[dict[str, Any]] = []
    replay_valid_count = 0
    for episode in range(args.episodes):
        events = make_episode_events(
            seed=seed,
            episode=episode,
            key_count=args.key_count,
            value_count=args.value_count,
            binding_count=args.binding_count,
            delay=args.delay,
        )
        records, metrics = execute_episode(
            events,
            agent=agent,
            baseline_window=args.baseline_window,
            value_count=args.value_count,
            tick_start=episode * len(events),
        )
        replay = replay_trace_records(
            records,
            agent=agent,
            baseline_window=args.baseline_window,
            value_count=args.value_count,
        )
        if replay["valid"]:
            replay_valid_count += 1
        episode_metrics.append({**metrics, "state_consistency": 1.0 if replay["valid"] else 0.0})

    combined = combine_episode_metrics(episode_metrics)
    combined["state_consistency"] = replay_valid_count / args.episodes if args.episodes else 0.0
    combined["seed"] = seed
    combined["agent"] = agent
    return combined


def aggregate_agent(agent_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "action_accuracy": summarize([float(run["action_accuracy"]) for run in agent_runs]),
        "write_read_fidelity": summarize(
            [float(run["write_read_fidelity"]) for run in agent_runs]
        ),
        "trajectory_horizon": summarize(
            [float(run["trajectory_horizon_mean"]) for run in agent_runs]
        ),
        "state_consistency": summarize([float(run["state_consistency"]) for run in agent_runs]),
        "world_delta_exact_rate": summarize(
            [float(run["world_delta_exact_rate"]) for run in agent_runs]
        ),
        "active_compute_total_ticks": summarize(
            [float(run["active_compute"]["total_ticks"]) for run in agent_runs]
        ),
        "runs": agent_runs,
    }


def run_stage1(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_csv_ints(args.seeds)
    seed_runs: list[dict[str, Any]] = []
    by_agent: dict[str, list[dict[str, Any]]] = {agent: [] for agent in AGENTS}
    for seed in seeds:
        seed_payload: dict[str, Any] = {"seed": seed, "agents": {}}
        for agent in AGENTS:
            run = run_agent_seed(seed=seed, agent=agent, args=args)
            seed_payload["agents"][agent] = run
            by_agent[agent].append(run)
        kv_acc = seed_payload["agents"]["kv_memory"]["action_accuracy"]
        ctx_acc = seed_payload["agents"]["context_window"]["action_accuracy"]
        seed_payload["comparison"] = {
            "kv_minus_context_action_accuracy": kv_acc - ctx_acc,
        }
        seed_runs.append(seed_payload)

    aggregate = {
        "agents": {agent: aggregate_agent(runs) for agent, runs in by_agent.items()},
        "comparison": {
            "kv_minus_context_action_accuracy": summarize(
                [
                    float(run["comparison"]["kv_minus_context_action_accuracy"])
                    for run in seed_runs
                ]
            )
        },
    }
    chance = 1.0 / args.value_count
    protocol_evidence = {
        "valid": bool(
            len(seeds) >= args.min_seeds
            and args.episodes >= args.min_episodes
            and args.delay > args.baseline_window
            and args.binding_count <= args.key_count
            and args.value_count > 0
        ),
        "seed_count": len(seeds),
        "seed_requirement_met": len(seeds) >= args.min_seeds,
        "episode_count": args.episodes,
        "episode_requirement_met": args.episodes >= args.min_episodes,
        "delay": args.delay,
        "baseline_window": args.baseline_window,
        "delay_beyond_context_window": args.delay > args.baseline_window,
        "same_task_stream_both_agents": True,
        "chance_action_accuracy": chance,
    }
    thresholds = {
        "kv_memory_action_accuracy_min": args.kv_accuracy_min,
        "context_window_action_accuracy_max": chance + args.context_chance_margin,
        "kv_minus_context_action_accuracy_min": args.kv_margin_min,
        "state_consistency_mean": 1.0,
    }
    kv_action_mean = aggregate["agents"]["kv_memory"]["action_accuracy"]["mean"]
    ctx_action_mean = aggregate["agents"]["context_window"]["action_accuracy"]["mean"]
    kv_minus_ctx_mean = aggregate["comparison"]["kv_minus_context_action_accuracy"]["mean"]
    state_consistency_means = [
        aggregate["agents"][agent]["state_consistency"]["mean"] for agent in AGENTS
    ]
    pass_fail = {
        "kv_memory_action_accuracy": kv_action_mean >= thresholds["kv_memory_action_accuracy_min"],
        "context_window_action_accuracy": (
            ctx_action_mean <= thresholds["context_window_action_accuracy_max"]
        ),
        "kv_minus_context_action_accuracy": (
            kv_minus_ctx_mean >= thresholds["kv_minus_context_action_accuracy_min"]
        ),
        "state_consistency": all(value == 1.0 for value in state_consistency_means),
        "protocol_evidence": protocol_evidence["valid"],
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage1_recall_to_action_smoke",
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "thresholds": thresholds,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "seed_runs": seed_runs,
        "aggregate": aggregate,
        "claim_guardrail": (
            "Stage 1 uses deterministic hand-coded agents to validate "
            "recall-to-action metrics. It is a harness smoke, not a trained "
            "SSM result yet."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 1 smoke failed: {pass_fail}")
    print("NeoCortex-X Stage 1 recall-to-action smoke")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  kv_memory.action_accuracy_mean: {kv_action_mean:.4f}")
    print(f"  context_window.action_accuracy_mean: {ctx_action_mean:.4f}")
    print(f"  kv_minus_context_action_accuracy_mean: {kv_minus_ctx_mean:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeoCortex-X Stage 0/1 controlled experiment smoke tests."
    )
    parser.add_argument("--stage", choices=("stage0", "stage1"), required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seeds", default="11,17,23,29,31,37")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--min-episodes", type=int, default=64)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--key-count", type=int, default=16)
    parser.add_argument("--value-count", type=int, default=16)
    parser.add_argument("--binding-count", type=int, default=4)
    parser.add_argument("--delay", type=int, default=24)
    parser.add_argument("--baseline-window", type=int, default=8)
    parser.add_argument(
        "--output-json",
        default="artifacts/neocortex_x/stage1_recall_to_action.json",
    )
    parser.add_argument(
        "--output-trace",
        default="artifacts/neocortex_x/stage0_trace.jsonl",
    )
    parser.add_argument("--kv-accuracy-min", type=float, default=0.95)
    parser.add_argument("--context-chance-margin", type=float, default=0.20)
    parser.add_argument("--kv-margin-min", type=float, default=0.50)
    parser.add_argument("--no-require-pass", dest="require_pass", action="store_false")
    parser.set_defaults(require_pass=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive.")
    if args.key_count < 1:
        raise SystemExit("--key-count must be positive.")
    if args.value_count < 1:
        raise SystemExit("--value-count must be positive.")
    if args.binding_count < 1:
        raise SystemExit("--binding-count must be positive.")
    if args.binding_count > args.key_count:
        raise SystemExit("--binding-count must not exceed --key-count.")
    if args.delay < 0:
        raise SystemExit("--delay must be non-negative.")
    if args.baseline_window < 0:
        raise SystemExit("--baseline-window must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.stage == "stage0":
        if args.output_json == "artifacts/neocortex_x/stage1_recall_to_action.json":
            args.output_json = "artifacts/neocortex_x/stage0_smoke.json"
        run_stage0(args)
    else:
        run_stage1(args)


if __name__ == "__main__":
    main()
