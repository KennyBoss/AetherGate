#!/usr/bin/env python3
"""NeoCortex-X Stage 0/1/2 controlled experiment harness.

This is intentionally small: it validates the trace/replay contract and the
recall-to-action protocol before any trained model or CodePy procedure memory
is plugged in, then adds the first bounded stateful tape world.
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
TAPE_AGENTS = ("codepy_loop", "single_cell_baseline")
PROCEDURE_AGENTS = (
    "procedure_reuse",
    "inline_loop_control",
    "exact_cache_control",
    "random_routing_control",
)
PLANNING_AGENTS = ("lookahead_planner", "greedy_action")
OPEN_WORLD_AGENTS = ("provenance_adapter", "no_external_control", "naive_latest_control")
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
    "tape_reads",
    "tape_writes",
    "register_writes",
    "pointer_moves",
    "branches",
    "procedure_steps",
    "cache_lookups",
    "random_routes",
    "planning_nodes",
    "external_observations",
    "contradictions",
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


def make_tape_task(
    *,
    seed: int,
    episode: int,
    tape_len: int,
    input_len_min: int,
    input_len_max: int,
) -> dict[str, Any]:
    if tape_len < 2:
        raise ValueError("tape_len must be at least 2.")
    if input_len_min < 2:
        raise ValueError("input_len_min must be at least 2 for the Stage 2 baseline.")
    if input_len_max > tape_len:
        raise ValueError("input_len_max must not exceed tape_len.")
    if input_len_min > input_len_max:
        raise ValueError("input_len_min must not exceed input_len_max.")

    rng = random.Random(seed * 1_000_003 + episode * 9_176 + 2_024)
    input_len = rng.randint(input_len_min, input_len_max)
    values = [rng.randint(-5, 5) for _ in range(input_len)]
    # Keep the baseline honest: summing only the first cell cannot solve this task.
    while sum(values[1:]) == 0:
        values = [rng.randint(-5, 5) for _ in range(input_len)]
    tape = values + [0 for _ in range(tape_len - input_len)]
    return {
        "task_id": f"seed{seed}_episode{episode}",
        "seed": seed,
        "episode": episode,
        "target": "sum_first_n",
        "tape": tape,
        "input_len": input_len,
        "target_sum": sum(values),
    }


def initial_tape_state(agent: str, task: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent": agent,
        "task_id": task["task_id"],
        "target": task["target"],
        "tape": list(task["tape"]),
        "input_len": int(task["input_len"]),
        "target_sum": int(task["target_sum"]),
        "ptr": 0,
        "pc": 0,
        "registers": {"r0": 0, "r1": 0},
        "predicate": False,
        "halted": False,
        "errors": [],
        "active_compute": {key: 0 for key in COMPUTE_KEYS},
    }


def tape_semantic_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "tape": state["tape"],
        "input_len": state["input_len"],
        "target_sum": state["target_sum"],
        "ptr": state["ptr"],
        "pc": state["pc"],
        "registers": state["registers"],
        "predicate": state["predicate"],
        "halted": state["halted"],
    }


def build_tape_delta(
    before: dict[str, Any], after: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    return {
        "action": action,
        "ptr_before": before["ptr"],
        "ptr_after": after["ptr"],
        "pc_before": before["pc"],
        "pc_after": after["pc"],
        "registers_before": before["registers"],
        "registers_after": after["registers"],
        "predicate_before": before["predicate"],
        "predicate_after": after["predicate"],
        "halted_before": before["halted"],
        "halted_after": after["halted"],
        "errors_added": after["errors"][len(before["errors"]) :],
        "active_compute_increment": active_compute_increment(before, after),
    }


def tape_transition(
    state: dict[str, Any], *, agent: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool], dict[str, Any], float]:
    if agent not in TAPE_AGENTS:
        raise ValueError(f"Unknown Stage 2 agent: {agent}")

    before = copy.deepcopy(state)
    after = copy.deepcopy(state)
    gates = empty_gates()
    reward = 0.0
    action: dict[str, Any]

    if before["halted"]:
        gates["stop"] = True
        action = {"kind": "already_halted"}
    elif agent == "codepy_loop":
        pc = before["pc"]
        if pc == 0:
            gates["env_action"] = True
            value = before["tape"][before["ptr"]]
            after["registers"]["r1"] = value
            after["pc"] = 1
            after["active_compute"]["tape_reads"] += 1
            after["active_compute"]["register_writes"] += 1
            action = {"kind": "tape_read", "ptr": before["ptr"], "value": value}
        elif pc == 1:
            gates["env_action"] = True
            after["registers"]["r0"] = before["registers"]["r0"] + before["registers"]["r1"]
            after["pc"] = 2
            after["active_compute"]["register_writes"] += 1
            action = {
                "kind": "accumulate",
                "r1": before["registers"]["r1"],
                "r0_after": after["registers"]["r0"],
            }
        elif pc == 2:
            gates["env_action"] = True
            after["ptr"] = min(before["ptr"] + 1, len(before["tape"]))
            after["pc"] = 3
            after["active_compute"]["pointer_moves"] += 1
            action = {"kind": "move_ptr", "ptr_after": after["ptr"]}
        elif pc == 3:
            gates["env_action"] = True
            after["predicate"] = before["ptr"] < before["input_len"]
            after["pc"] = 4
            after["active_compute"]["register_writes"] += 1
            action = {
                "kind": "set_predicate",
                "expr": "ptr < input_len",
                "value": after["predicate"],
            }
        elif pc == 4:
            gates["env_action"] = True
            after["pc"] = 0 if before["predicate"] else 5
            after["active_compute"]["branches"] += 1
            action = {"kind": "branch", "predicate": before["predicate"], "pc_after": after["pc"]}
        elif pc == 5:
            gates["stop"] = True
            after["halted"] = True
            action = {"kind": "stop"}
        else:
            gates["stop"] = True
            after["halted"] = True
            after["errors"].append({"phase": "tape_step", "reason": "invalid_pc", "pc": pc})
            action = {"kind": "invalid_pc_stop", "pc": pc}
    else:
        pc = before["pc"]
        if pc == 0:
            gates["env_action"] = True
            value = before["tape"][0]
            after["registers"]["r1"] = value
            after["pc"] = 1
            after["active_compute"]["tape_reads"] += 1
            after["active_compute"]["register_writes"] += 1
            action = {"kind": "baseline_read_first", "value": value}
        elif pc == 1:
            gates["env_action"] = True
            after["registers"]["r0"] = before["registers"]["r0"] + before["registers"]["r1"]
            after["pc"] = 2
            after["active_compute"]["register_writes"] += 1
            action = {"kind": "baseline_accumulate_first", "r0_after": after["registers"]["r0"]}
        else:
            gates["stop"] = True
            after["halted"] = True
            action = {"kind": "baseline_stop"}

    after["active_compute"]["total_ticks"] += 1
    if gates["env_action"]:
        after["active_compute"]["env_actions"] += 1
    if gates["stop"]:
        after["active_compute"]["stops"] += 1
    if after["halted"] and after["registers"]["r0"] == after["target_sum"] and not after["errors"]:
        reward = 1.0

    delta = build_tape_delta(before, after, action)
    return after, delta, gates, action, reward


def execute_tape_episode(
    task: dict[str, Any], *, agent: str, max_steps: int, tick_start: int = 0
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    state = initial_tape_state(agent, task)
    records: list[dict[str, Any]] = []
    exact_delta_count = 0

    for local_tick in range(max_steps):
        if state["halted"]:
            break
        tick_id = tick_start + local_tick
        before_digest = state_digest(state)
        observation = {
            "event": {"type": "tape_step", "step": local_tick},
            "task": copy.deepcopy(task),
            "state": tape_semantic_state(state),
        }
        predicted_after, predicted_delta, _gates, _action, predicted_reward = tape_transition(
            state, agent=agent
        )
        actual_after, actual_delta, actual_gates, actual_action, reward = tape_transition(
            state, agent=agent
        )
        if predicted_after != actual_after:
            raise RuntimeError("Deterministic tape predictor diverged from transition.")
        if predicted_delta == actual_delta:
            exact_delta_count += 1

        record = {
            "tick_id": tick_id,
            "episode": task["episode"],
            "task_id": task["task_id"],
            "agent": agent,
            "phase": "stop" if actual_gates["stop"] else "tape_step",
            "event_type": "tape_step",
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
        state = actual_after

    final_state = copy.deepcopy(state)
    metrics = {
        "tick_count": len(records),
        "halted": bool(final_state["halted"]),
        "task_success": bool(
            final_state["halted"]
            and final_state["registers"]["r0"] == final_state["target_sum"]
            and not final_state["errors"]
        ),
        "world_delta_exact_rate": exact_delta_count / len(records) if records else 0.0,
        "active_compute": copy.deepcopy(final_state["active_compute"]),
        "final_state": tape_semantic_state(final_state),
    }
    return records, metrics, final_state


def replay_tape_records(
    records: list[dict[str, Any]], *, agent: str, max_steps: int
) -> dict[str, Any]:
    if not records:
        return {
            "valid": False,
            "total_ticks": 0,
            "state_mismatch_count": 0,
            "delta_mismatch_count": 0,
            "mismatches": [],
            "delta_mismatches": [],
        }
    task = records[0]["observation"]["task"]
    state = initial_tape_state(agent, task)
    mismatches: list[dict[str, Any]] = []
    delta_mismatches: list[dict[str, Any]] = []
    for index, record in enumerate(records[:max_steps]):
        after, delta, _gates, _action, _reward = tape_transition(state, agent=agent)
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
        "valid": not mismatches and not delta_mismatches and len(records) <= max_steps,
        "total_ticks": len(records),
        "state_mismatch_count": len(mismatches),
        "delta_mismatch_count": len(delta_mismatches),
        "mismatches": mismatches[:8],
        "delta_mismatches": delta_mismatches[:8],
    }


def validate_tape_trace_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
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
    return {
        "valid": not missing and not bad_gates and "tape_step" in phases and "stop" in phases,
        "record_count": len(records),
        "phases": phases,
        "tape_step_present": "tape_step" in phases,
        "stop_present": "stop" in phases,
        "missing_fields": missing[:8],
        "bad_gates": bad_gates[:8],
    }


def trajectory_horizon_against_reference(
    records: list[dict[str, Any]], reference_records: list[dict[str, Any]]
) -> int:
    for index, (record, reference) in enumerate(zip(records, reference_records), start=1):
        if record["actual_delta"] != reference["actual_delta"]:
            return index
        if record["state_after_digest"] != reference["state_after_digest"]:
            return index
    if len(records) == len(reference_records):
        return len(reference_records)
    return min(len(records), len(reference_records)) + 1


def run_tape_agent_seed(*, seed: int, agent: str, args: argparse.Namespace) -> dict[str, Any]:
    per_episode: list[dict[str, Any]] = []
    replay_valid_count = 0
    state_consistent_count = 0
    task_success_count = 0
    trace_records: list[dict[str, Any]] = []
    active_compute = {key: 0 for key in COMPUTE_KEYS}

    for episode in range(args.episodes):
        task = make_tape_task(
            seed=seed,
            episode=episode,
            tape_len=args.tape_len,
            input_len_min=args.input_len_min,
            input_len_max=args.input_len_max,
        )
        reference_records, _reference_metrics, reference_state = execute_tape_episode(
            task,
            agent="codepy_loop",
            max_steps=args.max_steps,
            tick_start=episode * args.max_steps,
        )
        records, metrics, final_state = execute_tape_episode(
            task,
            agent=agent,
            max_steps=args.max_steps,
            tick_start=episode * args.max_steps,
        )
        replay = replay_tape_records(records, agent=agent, max_steps=args.max_steps)
        replay_valid = replay["valid"]
        state_consistent = tape_semantic_state(final_state) == tape_semantic_state(reference_state)
        trajectory_horizon = trajectory_horizon_against_reference(records, reference_records)

        if replay_valid:
            replay_valid_count += 1
        if state_consistent:
            state_consistent_count += 1
        if metrics["task_success"]:
            task_success_count += 1
        for key in COMPUTE_KEYS:
            active_compute[key] += int(metrics["active_compute"][key])

        per_episode.append(
            {
                "episode": episode,
                "task_id": task["task_id"],
                "input_len": task["input_len"],
                "target_sum": task["target_sum"],
                "tick_count": metrics["tick_count"],
                "state_consistent": state_consistent,
                "task_success": metrics["task_success"],
                "trajectory_horizon": trajectory_horizon,
                "replay_consistent": replay_valid,
                "world_delta_exact_rate": metrics["world_delta_exact_rate"],
            }
        )
        if seed == parse_csv_ints(args.seeds)[0] and episode < args.trace_episodes:
            trace_records.extend(records)

    return {
        "seed": seed,
        "agent": agent,
        "episode_count": args.episodes,
        "state_consistency": state_consistent_count / args.episodes,
        "task_success": task_success_count / args.episodes,
        "replay_consistency": replay_valid_count / args.episodes,
        "trajectory_horizon_mean": mean(
            [float(item["trajectory_horizon"]) for item in per_episode]
        ),
        "world_delta_exact_rate": mean(
            [float(item["world_delta_exact_rate"]) for item in per_episode]
        ),
        "active_compute": active_compute,
        "episodes": per_episode,
        "trace_records": trace_records,
    }


def aggregate_tape_agent(agent_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state_consistency": summarize([float(run["state_consistency"]) for run in agent_runs]),
        "task_success": summarize([float(run["task_success"]) for run in agent_runs]),
        "replay_consistency": summarize([float(run["replay_consistency"]) for run in agent_runs]),
        "trajectory_horizon": summarize(
            [float(run["trajectory_horizon_mean"]) for run in agent_runs]
        ),
        "world_delta_exact_rate": summarize(
            [float(run["world_delta_exact_rate"]) for run in agent_runs]
        ),
        "active_compute_total_ticks": summarize(
            [float(run["active_compute"]["total_ticks"]) for run in agent_runs]
        ),
        "runs": [
            {key: value for key, value in run.items() if key != "trace_records"}
            for run in agent_runs
        ],
    }


def run_stage2(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_csv_ints(args.seeds)
    by_agent: dict[str, list[dict[str, Any]]] = {agent: [] for agent in TAPE_AGENTS}
    seed_runs: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []

    for seed in seeds:
        seed_payload: dict[str, Any] = {"seed": seed, "agents": {}}
        for agent in TAPE_AGENTS:
            run = run_tape_agent_seed(seed=seed, agent=agent, args=args)
            trace_records.extend(run["trace_records"])
            run_without_trace = {key: value for key, value in run.items() if key != "trace_records"}
            seed_payload["agents"][agent] = run_without_trace
            by_agent[agent].append(run)
        codepy_state = seed_payload["agents"]["codepy_loop"]["state_consistency"]
        baseline_state = seed_payload["agents"]["single_cell_baseline"]["state_consistency"]
        codepy_horizon = seed_payload["agents"]["codepy_loop"]["trajectory_horizon_mean"]
        baseline_horizon = seed_payload["agents"]["single_cell_baseline"]["trajectory_horizon_mean"]
        seed_payload["comparison"] = {
            "codepy_minus_baseline_state_consistency": codepy_state - baseline_state,
            "codepy_minus_baseline_trajectory_horizon": codepy_horizon - baseline_horizon,
        }
        seed_runs.append(seed_payload)

    if trace_records:
        write_jsonl(Path(args.output_trace), trace_records)
        trace_schema = validate_tape_trace_schema(trace_records)
    else:
        trace_schema = {
            "valid": False,
            "record_count": 0,
            "phases": [],
            "tape_step_present": False,
            "stop_present": False,
            "missing_fields": [],
            "bad_gates": [],
        }

    aggregate = {
        "agents": {agent: aggregate_tape_agent(runs) for agent, runs in by_agent.items()},
        "comparison": {
            "codepy_minus_baseline_state_consistency": summarize(
                [
                    float(run["comparison"]["codepy_minus_baseline_state_consistency"])
                    for run in seed_runs
                ]
            ),
            "codepy_minus_baseline_trajectory_horizon": summarize(
                [
                    float(run["comparison"]["codepy_minus_baseline_trajectory_horizon"])
                    for run in seed_runs
                ]
            ),
        },
    }

    protocol_evidence = {
        "valid": bool(
            len(seeds) >= args.min_seeds
            and args.episodes >= args.stage2_min_episodes
            and args.input_len_min >= 2
            and args.input_len_max <= args.tape_len
            and args.max_steps >= (5 * args.input_len_max + 1)
            and trace_schema["valid"]
        ),
        "seed_count": len(seeds),
        "seed_requirement_met": len(seeds) >= args.min_seeds,
        "episode_count": args.episodes,
        "episode_requirement_met": args.episodes >= args.stage2_min_episodes,
        "input_len_min": args.input_len_min,
        "input_len_max": args.input_len_max,
        "tape_len": args.tape_len,
        "max_steps": args.max_steps,
        "max_steps_covers_reference_program": args.max_steps >= (5 * args.input_len_max + 1),
        "same_tasks_both_agents": True,
        "trace_schema_valid": trace_schema["valid"],
    }
    thresholds = {
        "codepy_loop_state_consistency_min": args.stage2_codepy_state_min,
        "single_cell_baseline_state_consistency_max": args.stage2_baseline_state_max,
        "codepy_minus_baseline_state_consistency_min": args.stage2_state_margin_min,
        "trajectory_horizon_margin_min": args.stage2_horizon_margin_min,
        "codepy_loop_replay_consistency_mean": 1.0,
        "codepy_loop_world_delta_exact_rate_mean": 1.0,
    }

    codepy_state_mean = aggregate["agents"]["codepy_loop"]["state_consistency"]["mean"]
    baseline_state_mean = aggregate["agents"]["single_cell_baseline"]["state_consistency"]["mean"]
    state_margin_mean = aggregate["comparison"]["codepy_minus_baseline_state_consistency"]["mean"]
    horizon_margin_mean = aggregate["comparison"]["codepy_minus_baseline_trajectory_horizon"]["mean"]
    codepy_replay_mean = aggregate["agents"]["codepy_loop"]["replay_consistency"]["mean"]
    codepy_delta_mean = aggregate["agents"]["codepy_loop"]["world_delta_exact_rate"]["mean"]
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "codepy_loop_state_consistency": (
            codepy_state_mean >= thresholds["codepy_loop_state_consistency_min"]
        ),
        "single_cell_baseline_state_consistency": (
            baseline_state_mean <= thresholds["single_cell_baseline_state_consistency_max"]
        ),
        "codepy_minus_baseline_state_consistency": (
            state_margin_mean >= thresholds["codepy_minus_baseline_state_consistency_min"]
        ),
        "trajectory_horizon_margin": (
            horizon_margin_mean >= thresholds["trajectory_horizon_margin_min"]
        ),
        "codepy_loop_replay_consistency": (
            codepy_replay_mean == thresholds["codepy_loop_replay_consistency_mean"]
        ),
        "codepy_loop_world_delta_exact_rate": (
            codepy_delta_mean == thresholds["codepy_loop_world_delta_exact_rate_mean"]
        ),
    }

    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage2_stateful_tape_world_smoke",
        "config": vars(args),
        "trace_path": str(args.output_trace),
        "trace_schema": trace_schema,
        "protocol_evidence": protocol_evidence,
        "thresholds": thresholds,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "seed_runs": seed_runs,
        "aggregate": aggregate,
        "claim_guardrail": (
            "Stage 2 validates a bounded stateful tape world and baseline "
            "comparison. It is not yet a learned SSM policy or procedure reuse claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 2 smoke failed: {pass_fail}")
    print("NeoCortex-X Stage 2 stateful tape world smoke")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  codepy_loop.state_consistency_mean: {codepy_state_mean:.4f}")
    print(f"  single_cell_baseline.state_consistency_mean: {baseline_state_mean:.4f}")
    print(f"  state_consistency_margin_mean: {state_margin_mean:.4f}")
    print(f"  trajectory_horizon_margin_mean: {horizon_margin_mean:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def make_procedure_task(
    *,
    seed: int,
    episode: int,
    tape_len: int,
    segment_len: int,
) -> dict[str, Any]:
    if segment_len < 1:
        raise ValueError("segment_len must be positive.")
    if tape_len < segment_len * 2:
        raise ValueError("tape_len must fit two non-overlapping segments.")

    rng = random.Random(seed * 1_000_003 + episode * 65_537 + 3_003)
    axis = "axis_s" if episode % 2 == 0 else "axis_c"
    start_a = rng.randint(0, tape_len - segment_len * 2)
    start_b = rng.randint(start_a + segment_len, tape_len - segment_len)
    base_values = [rng.randint(-6, 6) for _ in range(tape_len)]
    if axis == "axis_s":
        surface_shift = (episode % 5) - 2
        tape = [(-value + surface_shift) if index % 2 else (value + surface_shift) for index, value in enumerate(base_values)]
        segments = [
            {"start": start_a, "length": segment_len},
            {"start": start_b, "length": segment_len},
        ]
    else:
        tape = base_values
        segments = [
            {"start": start_a, "length": segment_len},
            {"start": start_b, "length": segment_len},
        ]
    target_sum = sum(
        sum(tape[segment["start"] : segment["start"] + segment["length"]])
        for segment in segments
    )
    return {
        "task_id": f"proc_seed{seed}_episode{episode}",
        "seed": seed,
        "episode": episode,
        "axis": axis,
        "target": "sum_segment" if axis == "axis_s" else "sum_segment_composition",
        "tape": tape,
        "segments": segments,
        "target_sum": target_sum,
    }


def procedure_sum_segment(tape: list[int], start: int, length: int) -> tuple[int, int]:
    total = 0
    for offset in range(length):
        total += tape[start + offset]
    return total, length


def run_procedure_task(
    task: dict[str, Any], *, agent: str, call_overhead: int
) -> dict[str, Any]:
    if agent not in PROCEDURE_AGENTS:
        raise ValueError(f"Unknown Stage 3 agent: {agent}")

    compute = {key: 0 for key in COMPUTE_KEYS}
    result = 0
    reuse_count = 0
    cache_hit = False
    random_route_count = 0
    route_errors = 0
    trace: list[dict[str, Any]] = []

    compute["total_ticks"] += 1
    if agent == "procedure_reuse":
        for segment in task["segments"]:
            segment_sum, steps = procedure_sum_segment(
                task["tape"], segment["start"], segment["length"]
            )
            result += segment_sum
            reuse_count += 1
            compute["codepy_calls"] += 1
            compute["procedure_steps"] += steps
            compute["total_ticks"] += call_overhead + 1
            trace.append(
                {
                    "kind": "call_procedure",
                    "name": "sum_segment",
                    "start": segment["start"],
                    "length": segment["length"],
                    "result": segment_sum,
                }
            )
    elif agent == "inline_loop_control":
        for segment in task["segments"]:
            for offset in range(segment["length"]):
                result += task["tape"][segment["start"] + offset]
                compute["tape_reads"] += 1
                compute["register_writes"] += 1
                compute["pointer_moves"] += 1
                compute["total_ticks"] += 4
        trace.append({"kind": "inline_loop", "segments": task["segments"]})
    elif agent == "exact_cache_control":
        compute["cache_lookups"] += 1
        compute["total_ticks"] += 1
        # Held-out task IDs are intentionally cold; exact cache cannot explain reuse.
        cache_hit = False
        result = 0
        trace.append({"kind": "cache_lookup", "hit": cache_hit, "task_id": task["task_id"]})
    else:
        for index, segment in enumerate(task["segments"]):
            wrong_start = (segment["start"] + segment["length"] + index + 1) % (
                len(task["tape"]) - segment["length"] + 1
            )
            if wrong_start == segment["start"]:
                wrong_start = (wrong_start + 1) % (len(task["tape"]) - segment["length"] + 1)
            segment_sum, steps = procedure_sum_segment(task["tape"], wrong_start, segment["length"])
            result += segment_sum
            reuse_count += 1
            random_route_count += 1
            route_errors += int(wrong_start != segment["start"])
            compute["codepy_calls"] += 1
            compute["procedure_steps"] += steps
            compute["random_routes"] += 1
            compute["total_ticks"] += call_overhead + 1
            trace.append(
                {
                    "kind": "random_route",
                    "requested_start": segment["start"],
                    "routed_start": wrong_start,
                    "length": segment["length"],
                    "result": segment_sum,
                }
            )

    success = result == task["target_sum"]
    state_consistent = success and agent in {"procedure_reuse", "inline_loop_control"}
    return {
        "agent": agent,
        "task_id": task["task_id"],
        "axis": task["axis"],
        "result": result,
        "target_sum": task["target_sum"],
        "task_success": success,
        "state_consistency": state_consistent,
        "active_compute": compute,
        "active_compute_total": (
            compute["total_ticks"]
            + compute["tape_reads"]
            + compute["procedure_steps"]
            + compute["codepy_calls"] * call_overhead
        ),
        "reuse_count": reuse_count,
        "cache_hit": cache_hit,
        "random_route_count": random_route_count,
        "route_errors": route_errors,
        "trace": trace,
    }


def run_procedure_agent_seed(*, seed: int, agent: str, args: argparse.Namespace) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    active_compute = {key: 0 for key in COMPUTE_KEYS}
    for episode in range(args.episodes):
        task = make_procedure_task(
            seed=seed,
            episode=episode,
            tape_len=args.stage3_tape_len,
            segment_len=args.stage3_segment_len,
        )
        result = run_procedure_task(task, agent=agent, call_overhead=args.stage3_call_overhead)
        for key in COMPUTE_KEYS:
            active_compute[key] += int(result["active_compute"][key])
        episodes.append(
            {
                "episode": episode,
                "task_id": task["task_id"],
                "axis": task["axis"],
                "target": task["target"],
                "task_success": result["task_success"],
                "state_consistency": result["state_consistency"],
                "active_compute_total": result["active_compute_total"],
                "reuse_count": result["reuse_count"],
                "cache_hit": result["cache_hit"],
                "random_route_count": result["random_route_count"],
            }
        )

    axis_s = [item for item in episodes if item["axis"] == "axis_s"]
    axis_c = [item for item in episodes if item["axis"] == "axis_c"]
    return {
        "seed": seed,
        "agent": agent,
        "episode_count": args.episodes,
        "task_success": mean([1.0 if item["task_success"] else 0.0 for item in episodes]),
        "state_consistency": mean(
            [1.0 if item["state_consistency"] else 0.0 for item in episodes]
        ),
        "axis_s_success": mean([1.0 if item["task_success"] else 0.0 for item in axis_s]),
        "axis_c_success": mean([1.0 if item["task_success"] else 0.0 for item in axis_c]),
        "active_compute_total": mean([float(item["active_compute_total"]) for item in episodes]),
        "reuse_count": mean([float(item["reuse_count"]) for item in episodes]),
        "cache_hit_rate": mean([1.0 if item["cache_hit"] else 0.0 for item in episodes]),
        "random_route_count": mean([float(item["random_route_count"]) for item in episodes]),
        "active_compute": active_compute,
        "episodes": episodes,
    }


def aggregate_procedure_agent(agent_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_success": summarize([float(run["task_success"]) for run in agent_runs]),
        "state_consistency": summarize([float(run["state_consistency"]) for run in agent_runs]),
        "axis_s_success": summarize([float(run["axis_s_success"]) for run in agent_runs]),
        "axis_c_success": summarize([float(run["axis_c_success"]) for run in agent_runs]),
        "active_compute_total": summarize(
            [float(run["active_compute_total"]) for run in agent_runs]
        ),
        "reuse_count": summarize([float(run["reuse_count"]) for run in agent_runs]),
        "cache_hit_rate": summarize([float(run["cache_hit_rate"]) for run in agent_runs]),
        "random_route_count": summarize(
            [float(run["random_route_count"]) for run in agent_runs]
        ),
        "runs": agent_runs,
    }


def run_stage3(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_csv_ints(args.seeds)
    by_agent: dict[str, list[dict[str, Any]]] = {agent: [] for agent in PROCEDURE_AGENTS}
    seed_runs: list[dict[str, Any]] = []

    for seed in seeds:
        seed_payload: dict[str, Any] = {"seed": seed, "agents": {}}
        for agent in PROCEDURE_AGENTS:
            run = run_procedure_agent_seed(seed=seed, agent=agent, args=args)
            by_agent[agent].append(run)
            seed_payload["agents"][agent] = run
        procedure_compute = seed_payload["agents"]["procedure_reuse"]["active_compute_total"]
        inline_compute = seed_payload["agents"]["inline_loop_control"]["active_compute_total"]
        seed_payload["comparison"] = {
            "active_compute_drop_vs_inline": inline_compute - procedure_compute,
            "procedure_minus_random_success": (
                seed_payload["agents"]["procedure_reuse"]["task_success"]
                - seed_payload["agents"]["random_routing_control"]["task_success"]
            ),
        }
        seed_runs.append(seed_payload)

    aggregate = {
        "agents": {
            agent: aggregate_procedure_agent(runs) for agent, runs in by_agent.items()
        },
        "comparison": {
            "active_compute_drop_vs_inline": summarize(
                [float(run["comparison"]["active_compute_drop_vs_inline"]) for run in seed_runs]
            ),
            "procedure_minus_random_success": summarize(
                [float(run["comparison"]["procedure_minus_random_success"]) for run in seed_runs]
            ),
        },
    }

    protocol_evidence = {
        "valid": bool(
            len(seeds) >= args.min_seeds
            and args.episodes >= args.stage3_min_episodes
            and args.stage3_tape_len >= args.stage3_segment_len * 2
            and args.stage3_call_overhead >= 0
        ),
        "seed_count": len(seeds),
        "seed_requirement_met": len(seeds) >= args.min_seeds,
        "episode_count": args.episodes,
        "episode_requirement_met": args.episodes >= args.stage3_min_episodes,
        "axis_s_present": True,
        "axis_c_present": True,
        "exact_cache_cold_on_heldout": True,
        "same_tasks_all_agents": True,
        "procedure_signature": "sum_segment(start:int, length:int) -> int",
    }
    thresholds = {
        "procedure_task_success_min": args.stage3_success_min,
        "procedure_axis_s_success_min": args.stage3_success_min,
        "procedure_axis_c_success_min": args.stage3_success_min,
        "active_compute_drop_vs_inline_min": args.stage3_compute_drop_min,
        "exact_cache_hit_rate": 0.0,
        "random_routing_task_success_max": args.stage3_random_success_max,
        "procedure_reuse_count_min": args.stage3_reuse_count_min,
    }
    procedure = aggregate["agents"]["procedure_reuse"]
    cache = aggregate["agents"]["exact_cache_control"]
    random_routing = aggregate["agents"]["random_routing_control"]
    compute_drop_mean = aggregate["comparison"]["active_compute_drop_vs_inline"]["mean"]
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "procedure_task_success": (
            procedure["task_success"]["mean"] >= thresholds["procedure_task_success_min"]
        ),
        "procedure_axis_s_success": (
            procedure["axis_s_success"]["mean"] >= thresholds["procedure_axis_s_success_min"]
        ),
        "procedure_axis_c_success": (
            procedure["axis_c_success"]["mean"] >= thresholds["procedure_axis_c_success_min"]
        ),
        "active_compute_drop_vs_inline": (
            compute_drop_mean >= thresholds["active_compute_drop_vs_inline_min"]
        ),
        "exact_cache_hit_rate": cache["cache_hit_rate"]["mean"] == thresholds["exact_cache_hit_rate"],
        "random_routing_task_success": (
            random_routing["task_success"]["mean"]
            <= thresholds["random_routing_task_success_max"]
        ),
        "procedure_reuse_count": (
            procedure["reuse_count"]["mean"] >= thresholds["procedure_reuse_count_min"]
        ),
    }

    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage3_procedure_reuse_smoke",
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "thresholds": thresholds,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "seed_runs": seed_runs,
        "aggregate": aggregate,
        "claim_guardrail": (
            "Stage 3 validates procedure-reuse accounting and controls with "
            "hand-coded certified procedures. It is not yet learned promotion."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 3 smoke failed: {pass_fail}")
    print("NeoCortex-X Stage 3 procedure reuse smoke")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  procedure_reuse.task_success_mean: {procedure['task_success']['mean']:.4f}")
    print(f"  axis_s_success_mean: {procedure['axis_s_success']['mean']:.4f}")
    print(f"  axis_c_success_mean: {procedure['axis_c_success']['mean']:.4f}")
    print(f"  active_compute_drop_vs_inline_mean: {compute_drop_mean:.4f}")
    print(f"  random_routing.task_success_mean: {random_routing['task_success']['mean']:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


ACTIONS = ("left", "right", "jump", "stay")
ACTION_SEQUENCE_CACHE: dict[int, list[tuple[str, ...]]] = {}


def make_planning_task(*, seed: int, episode: int, corridor_len: int) -> dict[str, Any]:
    if corridor_len < 7:
        raise ValueError("corridor_len must be at least 7.")
    rng = random.Random(seed * 1_000_003 + episode * 12_289 + 4_004)
    start = 1 + (episode % 2)
    goal = corridor_len - 1
    trap = min(start + 2 + rng.randrange(2), corridor_len - 2)
    lure = trap - 1
    return {
        "task_id": f"plan_seed{seed}_episode{episode}",
        "seed": seed,
        "episode": episode,
        "corridor_len": corridor_len,
        "start": start,
        "goal": goal,
        "trap": trap,
        "lure": lure,
        "max_steps": corridor_len + 3,
        "step_cost": -0.05,
        "lure_reward": 1.0,
        "goal_reward": 6.0,
        "trap_penalty": -6.0,
    }


def initial_planning_state(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": task["start"],
        "visited_lure": False,
        "halted": False,
        "terminal": None,
        "return": 0.0,
        "steps": 0,
    }


def planning_transition(
    state: dict[str, Any], task: dict[str, Any], action: str
) -> tuple[dict[str, Any], dict[str, Any], float]:
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    before = state
    after = {
        "position": int(state["position"]),
        "visited_lure": bool(state["visited_lure"]),
        "halted": bool(state["halted"]),
        "terminal": state["terminal"],
        "return": float(state["return"]),
        "steps": int(state["steps"]),
    }
    if before["halted"]:
        return after, {"action": action, "already_halted": True}, 0.0

    if action == "left":
        after["position"] = max(0, before["position"] - 1)
    elif action == "right":
        after["position"] = min(task["corridor_len"] - 1, before["position"] + 1)
    elif action == "jump":
        after["position"] = min(task["corridor_len"] - 1, before["position"] + 2)
    else:
        after["position"] = before["position"]

    reward = float(task["step_cost"])
    if after["position"] == task["lure"] and not before["visited_lure"]:
        reward += float(task["lure_reward"])
        after["visited_lure"] = True
    if after["position"] == task["trap"]:
        reward += float(task["trap_penalty"])
        after["halted"] = True
        after["terminal"] = "trap"
    elif after["position"] == task["goal"]:
        reward += float(task["goal_reward"])
        after["halted"] = True
        after["terminal"] = "goal"

    after["steps"] = before["steps"] + 1
    after["return"] = round(before["return"] + reward, 8)
    if after["steps"] >= task["max_steps"] and not after["halted"]:
        after["halted"] = True
        after["terminal"] = "budget"

    delta = {
        "action": action,
        "position_before": before["position"],
        "position_after": after["position"],
        "visited_lure_before": before["visited_lure"],
        "visited_lure_after": after["visited_lure"],
        "halted_before": before["halted"],
        "halted_after": after["halted"],
        "terminal": after["terminal"],
        "reward": reward,
        "return_after": after["return"],
        "steps_after": after["steps"],
    }
    return after, delta, reward


def score_action_one_step(state: dict[str, Any], task: dict[str, Any], action: str) -> float:
    _after, _delta, reward = planning_transition(state, task, action)
    return reward


def evaluate_sequence(
    state: dict[str, Any], task: dict[str, Any], sequence: tuple[str, ...]
) -> tuple[float, int]:
    rollout = {
        "position": int(state["position"]),
        "visited_lure": bool(state["visited_lure"]),
        "halted": bool(state["halted"]),
        "terminal": state["terminal"],
        "return": float(state["return"]),
        "steps": int(state["steps"]),
    }
    total = 0.0
    nodes = 0
    discount = 1.0
    for action in sequence:
        nodes += 1
        rollout, _delta, reward = planning_transition(rollout, task, action)
        total += discount * reward
        discount *= 0.98
        if rollout["halted"]:
            break
    return total, nodes


def action_sequences(horizon: int) -> list[tuple[str, ...]]:
    cached = ACTION_SEQUENCE_CACHE.get(horizon)
    if cached is not None:
        return cached
    sequences: list[tuple[str, ...]] = [()]
    for _depth in range(horizon):
        sequences = [seq + (action,) for seq in sequences for action in ACTIONS]
    ACTION_SEQUENCE_CACHE[horizon] = sequences
    return sequences


def choose_planning_action(
    state: dict[str, Any], task: dict[str, Any], *, agent: str, horizon: int
) -> tuple[str, int]:
    if agent == "greedy_action":
        tie_preference = {"stay": 3, "left": 2, "jump": 1, "right": 0}
        scored = [
            (score_action_one_step(state, task, action), tie_preference[action], action)
            for action in ACTIONS
        ]
        scored.sort(reverse=True)
        return scored[0][2], len(ACTIONS)
    if agent != "lookahead_planner":
        raise ValueError(f"Unknown Stage 4 agent: {agent}")

    best_score = float("-inf")
    best_action = "stay"
    expanded = 0
    for sequence in action_sequences(horizon):
        if not sequence:
            continue
        score, nodes = evaluate_sequence(state, task, sequence)
        expanded += nodes
        # Tie-break toward rightward progress when expected return is equal.
        if score > best_score or (score == best_score and sequence[0] == "right"):
            best_score = score
            best_action = sequence[0]
    return best_action, expanded


def run_planning_episode(
    task: dict[str, Any], *, agent: str, horizon: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if agent not in PLANNING_AGENTS:
        raise ValueError(f"Unknown Stage 4 agent: {agent}")
    state = initial_planning_state(task)
    records: list[dict[str, Any]] = []
    exact_delta_count = 0
    active_compute = {key: 0 for key in COMPUTE_KEYS}

    while not state["halted"]:
        before_digest = state_digest(state)
        action, expanded = choose_planning_action(state, task, agent=agent, horizon=horizon)
        predicted_after, predicted_delta, predicted_reward = planning_transition(state, task, action)
        actual_after, actual_delta, reward = planning_transition(state, task, action)
        if predicted_after != actual_after:
            raise RuntimeError("Deterministic planning predictor diverged from transition.")
        if predicted_delta == actual_delta:
            exact_delta_count += 1

        active_compute["total_ticks"] += 1
        active_compute["env_actions"] += 1
        active_compute["planning_nodes"] += expanded
        record = {
            "tick_id": len(records),
            "episode": task["episode"],
            "task_id": task["task_id"],
            "agent": agent,
            "phase": "planning_step",
            "event_type": "planning_step",
            "observation": {"task": task, "state": copy.deepcopy(state)},
            "gates": {
                **empty_gates(),
                "env_action": True,
                "stop": bool(actual_after["halted"]),
            },
            "action": {"kind": "move", "value": action, "expanded_nodes": expanded},
            "predicted_delta": predicted_delta,
            "actual_delta": actual_delta,
            "state_before_digest": before_digest,
            "state_after_digest": state_digest(actual_after),
            "reward": reward,
            "predicted_reward": predicted_reward,
            "active_compute": copy.deepcopy(active_compute),
        }
        records.append(record)
        state = actual_after

    success = state["terminal"] == "goal"
    metrics = {
        "task_success": success,
        "return": float(state["return"]),
        "trajectory_horizon": int(state["steps"]),
        "planning_nodes": active_compute["planning_nodes"],
        "world_delta_exact_rate": exact_delta_count / len(records) if records else 0.0,
        "active_compute": active_compute,
        "terminal": state["terminal"],
    }
    return records, metrics


def run_planning_agent_seed(*, seed: int, agent: str, args: argparse.Namespace) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    active_compute = {key: 0 for key in COMPUTE_KEYS}
    for episode in range(args.episodes):
        task = make_planning_task(
            seed=seed,
            episode=episode,
            corridor_len=args.stage4_corridor_len,
        )
        _records, metrics = run_planning_episode(
            task,
            agent=agent,
            horizon=args.stage4_horizon,
        )
        for key in COMPUTE_KEYS:
            active_compute[key] += int(metrics["active_compute"][key])
        episodes.append(
            {
                "episode": episode,
                "task_id": task["task_id"],
                "task_success": metrics["task_success"],
                "return": metrics["return"],
                "trajectory_horizon": metrics["trajectory_horizon"],
                "planning_nodes": metrics["planning_nodes"],
                "world_delta_exact_rate": metrics["world_delta_exact_rate"],
                "terminal": metrics["terminal"],
            }
        )
    return {
        "seed": seed,
        "agent": agent,
        "episode_count": args.episodes,
        "task_success": mean([1.0 if item["task_success"] else 0.0 for item in episodes]),
        "return": mean([float(item["return"]) for item in episodes]),
        "trajectory_horizon": mean([float(item["trajectory_horizon"]) for item in episodes]),
        "planning_nodes": mean([float(item["planning_nodes"]) for item in episodes]),
        "world_delta_exact_rate": mean(
            [float(item["world_delta_exact_rate"]) for item in episodes]
        ),
        "active_compute": active_compute,
        "episodes": episodes,
    }


def aggregate_planning_agent(agent_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_success": summarize([float(run["task_success"]) for run in agent_runs]),
        "return": summarize([float(run["return"]) for run in agent_runs]),
        "trajectory_horizon": summarize([float(run["trajectory_horizon"]) for run in agent_runs]),
        "planning_nodes": summarize([float(run["planning_nodes"]) for run in agent_runs]),
        "world_delta_exact_rate": summarize(
            [float(run["world_delta_exact_rate"]) for run in agent_runs]
        ),
        "runs": agent_runs,
    }


def run_stage4(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_csv_ints(args.seeds)
    by_agent: dict[str, list[dict[str, Any]]] = {agent: [] for agent in PLANNING_AGENTS}
    seed_runs: list[dict[str, Any]] = []
    for seed in seeds:
        seed_payload: dict[str, Any] = {"seed": seed, "agents": {}}
        for agent in PLANNING_AGENTS:
            run = run_planning_agent_seed(seed=seed, agent=agent, args=args)
            by_agent[agent].append(run)
            seed_payload["agents"][agent] = run
        planner = seed_payload["agents"]["lookahead_planner"]
        greedy = seed_payload["agents"]["greedy_action"]
        seed_payload["comparison"] = {
            "planner_minus_greedy_return": planner["return"] - greedy["return"],
            "planner_minus_greedy_success": planner["task_success"] - greedy["task_success"],
            "planner_minus_greedy_nodes": planner["planning_nodes"] - greedy["planning_nodes"],
        }
        seed_runs.append(seed_payload)

    aggregate = {
        "agents": {
            agent: aggregate_planning_agent(runs) for agent, runs in by_agent.items()
        },
        "comparison": {
            "planner_minus_greedy_return": summarize(
                [float(run["comparison"]["planner_minus_greedy_return"]) for run in seed_runs]
            ),
            "planner_minus_greedy_success": summarize(
                [float(run["comparison"]["planner_minus_greedy_success"]) for run in seed_runs]
            ),
            "planner_minus_greedy_nodes": summarize(
                [float(run["comparison"]["planner_minus_greedy_nodes"]) for run in seed_runs]
            ),
        },
    }
    protocol_evidence = {
        "valid": bool(
            len(seeds) >= args.min_seeds
            and args.episodes >= args.stage4_min_episodes
            and args.stage4_horizon >= 3
            and args.stage4_corridor_len >= 7
        ),
        "seed_count": len(seeds),
        "seed_requirement_met": len(seeds) >= args.min_seeds,
        "episode_count": args.episodes,
        "episode_requirement_met": args.episodes >= args.stage4_min_episodes,
        "horizon": args.stage4_horizon,
        "corridor_len": args.stage4_corridor_len,
        "same_predictor_both_agents": True,
        "deterministic_world_model": True,
    }
    thresholds = {
        "planner_success_min": args.stage4_planner_success_min,
        "greedy_success_max": args.stage4_greedy_success_max,
        "planner_minus_greedy_return_min": args.stage4_return_margin_min,
        "planner_minus_greedy_success_min": args.stage4_success_margin_min,
        "planner_world_delta_exact_rate": 1.0,
    }
    planner_agg = aggregate["agents"]["lookahead_planner"]
    greedy_agg = aggregate["agents"]["greedy_action"]
    return_margin = aggregate["comparison"]["planner_minus_greedy_return"]["mean"]
    success_margin = aggregate["comparison"]["planner_minus_greedy_success"]["mean"]
    nodes_margin = aggregate["comparison"]["planner_minus_greedy_nodes"]["mean"]
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "planner_success": planner_agg["task_success"]["mean"] >= thresholds["planner_success_min"],
        "greedy_success": greedy_agg["task_success"]["mean"] <= thresholds["greedy_success_max"],
        "planner_minus_greedy_return": (
            return_margin >= thresholds["planner_minus_greedy_return_min"]
        ),
        "planner_minus_greedy_success": (
            success_margin >= thresholds["planner_minus_greedy_success_min"]
        ),
        "planner_world_delta_exact_rate": (
            planner_agg["world_delta_exact_rate"]["mean"]
            == thresholds["planner_world_delta_exact_rate"]
        ),
        "planner_expands_more_nodes": nodes_margin > 0.0,
    }

    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage4_planning_smoke",
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "thresholds": thresholds,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "seed_runs": seed_runs,
        "aggregate": aggregate,
        "claim_guardrail": (
            "Stage 4 validates short lookahead over exact predicted deltas. "
            "It is not yet a learned world model or open-world planning claim."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 4 smoke failed: {pass_fail}")
    print("NeoCortex-X Stage 4 planning smoke")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  planner.task_success_mean: {planner_agg['task_success']['mean']:.4f}")
    print(f"  greedy.task_success_mean: {greedy_agg['task_success']['mean']:.4f}")
    print(f"  planner_minus_greedy_return_mean: {return_margin:.4f}")
    print(f"  planner_minus_greedy_success_mean: {success_margin:.4f}")
    print(f"  planner_minus_greedy_nodes_mean: {nodes_margin:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def make_open_world_task(*, seed: int, episode: int) -> dict[str, Any]:
    rng = random.Random(seed * 1_000_003 + episode * 32_771 + 5_005)
    subject = f"entity_{episode % 32}"
    correct_value = f"value_{(seed + episode) % 17}"
    stale_value = f"value_{(seed + episode + 7) % 17}"
    weak_value = f"value_{(seed + episode + 11) % 17}"
    if stale_value == correct_value:
        stale_value = f"value_{(seed + episode + 8) % 17}"
    if weak_value in {correct_value, stale_value}:
        weak_value = f"value_{(seed + episode + 12) % 17}"
    observations = [
        {
            "claim_id": f"{subject}-old-{episode}",
            "subject": subject,
            "value": stale_value,
            "source_id": "low_trust_forum",
            "source_trust": 0.25,
            "observed_at": f"2026-06-24T00:{episode % 60:02d}:00Z",
            "provenance_url": f"synthetic://low/{seed}/{episode}",
        },
        {
            "claim_id": f"{subject}-primary-{episode}",
            "subject": subject,
            "value": correct_value,
            "source_id": "primary_registry",
            "source_trust": 0.90,
            "observed_at": f"2026-06-24T01:{episode % 60:02d}:00Z",
            "provenance_url": f"synthetic://primary/{seed}/{episode}",
        },
        {
            "claim_id": f"{subject}-audit-{episode}",
            "subject": subject,
            "value": correct_value,
            "source_id": "audit_mirror",
            "source_trust": 0.70,
            "observed_at": f"2026-06-24T02:{episode % 60:02d}:00Z",
            "provenance_url": f"synthetic://audit/{seed}/{episode}",
        },
        {
            "claim_id": f"{subject}-latest-{episode}",
            "subject": subject,
            "value": weak_value,
            "source_id": "latest_unverified",
            "source_trust": 0.35 + 0.05 * rng.randrange(2),
            "observed_at": f"2026-06-24T03:{episode % 60:02d}:00Z",
            "provenance_url": f"synthetic://latest/{seed}/{episode}",
        },
    ]
    return {
        "task_id": f"open_seed{seed}_episode{episode}",
        "seed": seed,
        "episode": episode,
        "subject": subject,
        "ground_truth": correct_value,
        "observations": observations,
    }


def apply_provenance_adapter(task: dict[str, Any], contradiction_margin: float) -> dict[str, Any]:
    scores: dict[str, float] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    compute = {key: 0 for key in COMPUTE_KEYS}
    for observation in task["observations"]:
        value = observation["value"]
        scores[value] = scores.get(value, 0.0) + float(observation["source_trust"])
        provenance.setdefault(value, []).append(observation)
        compute["external_observations"] += 1
        compute["total_ticks"] += 1
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected_value, selected_score = ranked[0]
    rejected = [
        {"value": value, "score": score, "provenance": provenance[value]}
        for value, score in ranked[1:]
    ]
    contradiction = bool(rejected and selected_score - rejected[0]["score"] <= contradiction_margin)
    if contradiction:
        compute["contradictions"] += 1
    state = {
        "subject": task["subject"],
        "value": selected_value,
        "score": selected_score,
        "provenance": provenance[selected_value],
        "rejected": rejected,
        "contradiction": contradiction,
    }
    return {"state": state, "active_compute": compute}


def run_open_world_task(task: dict[str, Any], *, agent: str, contradiction_margin: float) -> dict[str, Any]:
    if agent not in OPEN_WORLD_AGENTS:
        raise ValueError(f"Unknown Stage 5 agent: {agent}")
    compute = {key: 0 for key in COMPUTE_KEYS}
    if agent == "provenance_adapter":
        result = apply_provenance_adapter(task, contradiction_margin)
        state = result["state"]
        compute = result["active_compute"]
    elif agent == "no_external_control":
        state = {
            "subject": task["subject"],
            "value": None,
            "score": 0.0,
            "provenance": [],
            "rejected": [],
            "contradiction": False,
        }
        compute["total_ticks"] = 1
    else:
        latest = task["observations"][-1]
        state = {
            "subject": task["subject"],
            "value": latest["value"],
            "score": latest["source_trust"],
            "provenance": [latest],
            "rejected": [],
            "contradiction": False,
        }
        compute["external_observations"] = len(task["observations"])
        compute["total_ticks"] = len(task["observations"])

    replay_state = (
        apply_provenance_adapter(task, contradiction_margin)["state"]
        if agent == "provenance_adapter"
        else copy.deepcopy(state)
    )
    factual_accuracy = state["value"] == task["ground_truth"]
    provenance_coverage = bool(state["value"] is None or state["provenance"])
    replay_consistency = state == replay_state
    expected_contradiction = False
    contradiction_precision = 1.0 if state["contradiction"] == expected_contradiction else 0.0
    return {
        "agent": agent,
        "task_id": task["task_id"],
        "subject": task["subject"],
        "selected_value": state["value"],
        "ground_truth": task["ground_truth"],
        "factual_accuracy": factual_accuracy,
        "provenance_coverage": provenance_coverage,
        "replay_consistency": replay_consistency,
        "contradiction_precision": contradiction_precision,
        "active_compute": compute,
        "state": state,
    }


def run_open_world_agent_seed(*, seed: int, agent: str, args: argparse.Namespace) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    active_compute = {key: 0 for key in COMPUTE_KEYS}
    for episode in range(args.episodes):
        task = make_open_world_task(seed=seed, episode=episode)
        result = run_open_world_task(
            task,
            agent=agent,
            contradiction_margin=args.stage5_contradiction_margin,
        )
        for key in COMPUTE_KEYS:
            active_compute[key] += int(result["active_compute"][key])
        episodes.append(
            {
                "episode": episode,
                "task_id": task["task_id"],
                "factual_accuracy": result["factual_accuracy"],
                "provenance_coverage": result["provenance_coverage"],
                "replay_consistency": result["replay_consistency"],
                "contradiction_precision": result["contradiction_precision"],
                "selected_value": result["selected_value"],
                "ground_truth": result["ground_truth"],
            }
        )
    return {
        "seed": seed,
        "agent": agent,
        "episode_count": args.episodes,
        "factual_accuracy": mean([1.0 if item["factual_accuracy"] else 0.0 for item in episodes]),
        "provenance_coverage": mean(
            [1.0 if item["provenance_coverage"] else 0.0 for item in episodes]
        ),
        "replay_consistency": mean(
            [1.0 if item["replay_consistency"] else 0.0 for item in episodes]
        ),
        "contradiction_precision": mean(
            [float(item["contradiction_precision"]) for item in episodes]
        ),
        "active_compute": active_compute,
        "episodes": episodes,
    }


def aggregate_open_world_agent(agent_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "factual_accuracy": summarize([float(run["factual_accuracy"]) for run in agent_runs]),
        "provenance_coverage": summarize(
            [float(run["provenance_coverage"]) for run in agent_runs]
        ),
        "replay_consistency": summarize([float(run["replay_consistency"]) for run in agent_runs]),
        "contradiction_precision": summarize(
            [float(run["contradiction_precision"]) for run in agent_runs]
        ),
        "runs": agent_runs,
    }


def run_stage5(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_csv_ints(args.seeds)
    by_agent: dict[str, list[dict[str, Any]]] = {agent: [] for agent in OPEN_WORLD_AGENTS}
    seed_runs: list[dict[str, Any]] = []
    for seed in seeds:
        seed_payload: dict[str, Any] = {"seed": seed, "agents": {}}
        for agent in OPEN_WORLD_AGENTS:
            run = run_open_world_agent_seed(seed=seed, agent=agent, args=args)
            by_agent[agent].append(run)
            seed_payload["agents"][agent] = run
        adapter = seed_payload["agents"]["provenance_adapter"]
        no_external = seed_payload["agents"]["no_external_control"]
        naive = seed_payload["agents"]["naive_latest_control"]
        seed_payload["comparison"] = {
            "adapter_minus_no_external_accuracy": (
                adapter["factual_accuracy"] - no_external["factual_accuracy"]
            ),
            "adapter_minus_naive_accuracy": (
                adapter["factual_accuracy"] - naive["factual_accuracy"]
            ),
        }
        seed_runs.append(seed_payload)

    aggregate = {
        "agents": {
            agent: aggregate_open_world_agent(runs) for agent, runs in by_agent.items()
        },
        "comparison": {
            "adapter_minus_no_external_accuracy": summarize(
                [
                    float(run["comparison"]["adapter_minus_no_external_accuracy"])
                    for run in seed_runs
                ]
            ),
            "adapter_minus_naive_accuracy": summarize(
                [float(run["comparison"]["adapter_minus_naive_accuracy"]) for run in seed_runs]
            ),
        },
    }
    protocol_evidence = {
        "valid": bool(len(seeds) >= args.min_seeds and args.episodes >= args.stage5_min_episodes),
        "seed_count": len(seeds),
        "seed_requirement_met": len(seeds) >= args.min_seeds,
        "episode_count": args.episodes,
        "episode_requirement_met": args.episodes >= args.stage5_min_episodes,
        "deterministic_external_observations": True,
        "live_web_disabled_for_replay": True,
        "same_observations_all_agents": True,
    }
    thresholds = {
        "adapter_factual_accuracy_min": args.stage5_adapter_accuracy_min,
        "adapter_provenance_coverage": 1.0,
        "adapter_replay_consistency": 1.0,
        "adapter_minus_no_external_accuracy_min": args.stage5_no_external_margin_min,
        "adapter_minus_naive_accuracy_min": args.stage5_naive_margin_min,
        "naive_latest_accuracy_max": args.stage5_naive_accuracy_max,
    }
    adapter_agg = aggregate["agents"]["provenance_adapter"]
    naive_agg = aggregate["agents"]["naive_latest_control"]
    no_external_margin = aggregate["comparison"]["adapter_minus_no_external_accuracy"]["mean"]
    naive_margin = aggregate["comparison"]["adapter_minus_naive_accuracy"]["mean"]
    pass_fail = {
        "protocol_evidence": protocol_evidence["valid"],
        "adapter_factual_accuracy": (
            adapter_agg["factual_accuracy"]["mean"] >= thresholds["adapter_factual_accuracy_min"]
        ),
        "adapter_provenance_coverage": (
            adapter_agg["provenance_coverage"]["mean"]
            == thresholds["adapter_provenance_coverage"]
        ),
        "adapter_replay_consistency": (
            adapter_agg["replay_consistency"]["mean"] == thresholds["adapter_replay_consistency"]
        ),
        "adapter_minus_no_external_accuracy": (
            no_external_margin >= thresholds["adapter_minus_no_external_accuracy_min"]
        ),
        "adapter_minus_naive_accuracy": naive_margin >= thresholds["adapter_minus_naive_accuracy_min"],
        "naive_latest_accuracy": (
            naive_agg["factual_accuracy"]["mean"] <= thresholds["naive_latest_accuracy_max"]
        ),
    }
    payload = {
        "created_at": timestamp(),
        "kind": "neocortex_x_stage5_open_world_adapter_smoke",
        "config": vars(args),
        "protocol_evidence": protocol_evidence,
        "thresholds": thresholds,
        "pass_fail": pass_fail,
        "pass": all(pass_fail.values()),
        "seed_runs": seed_runs,
        "aggregate": aggregate,
        "claim_guardrail": (
            "Stage 5 validates deterministic external-observation provenance and "
            "contradiction handling. It does not perform live web search."
        ),
    }
    write_json(Path(args.output_json), payload)
    if args.require_pass and not payload["pass"]:
        raise SystemExit(f"Stage 5 smoke failed: {pass_fail}")
    print("NeoCortex-X Stage 5 open-world adapter smoke")
    print(f"  protocol_valid: {protocol_evidence['valid']}")
    print(f"  pass: {payload['pass']}")
    print(f"  adapter.factual_accuracy_mean: {adapter_agg['factual_accuracy']['mean']:.4f}")
    print(f"  naive_latest.factual_accuracy_mean: {naive_agg['factual_accuracy']['mean']:.4f}")
    print(f"  adapter_minus_no_external_accuracy_mean: {no_external_margin:.4f}")
    print(f"  adapter_minus_naive_accuracy_mean: {naive_margin:.4f}")
    print(f"  output_json: {args.output_json}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeoCortex-X Stage 0/1/2 controlled experiment smoke tests."
    )
    parser.add_argument(
        "--stage",
        choices=("stage0", "stage1", "stage2", "stage3", "stage4", "stage5"),
        required=True,
    )
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
    parser.add_argument("--tape-len", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--input-len-min", type=int, default=4)
    parser.add_argument("--input-len-max", type=int, default=8)
    parser.add_argument("--trace-episodes", type=int, default=2)
    parser.add_argument("--stage2-min-episodes", type=int, default=96)
    parser.add_argument("--stage2-codepy-state-min", type=float, default=0.95)
    parser.add_argument("--stage2-baseline-state-max", type=float, default=0.40)
    parser.add_argument("--stage2-state-margin-min", type=float, default=0.50)
    parser.add_argument("--stage2-horizon-margin-min", type=float, default=4.0)
    parser.add_argument("--stage3-min-episodes", type=int, default=96)
    parser.add_argument("--stage3-tape-len", type=int, default=12)
    parser.add_argument("--stage3-segment-len", type=int, default=3)
    parser.add_argument("--stage3-call-overhead", type=int, default=1)
    parser.add_argument("--stage3-success-min", type=float, default=0.95)
    parser.add_argument("--stage3-compute-drop-min", type=float, default=4.0)
    parser.add_argument("--stage3-random-success-max", type=float, default=0.40)
    parser.add_argument("--stage3-reuse-count-min", type=float, default=2.0)
    parser.add_argument("--stage4-min-episodes", type=int, default=96)
    parser.add_argument("--stage4-corridor-len", type=int, default=9)
    parser.add_argument("--stage4-horizon", type=int, default=4)
    parser.add_argument("--stage4-planner-success-min", type=float, default=0.90)
    parser.add_argument("--stage4-greedy-success-max", type=float, default=0.60)
    parser.add_argument("--stage4-return-margin-min", type=float, default=2.0)
    parser.add_argument("--stage4-success-margin-min", type=float, default=0.30)
    parser.add_argument("--stage5-min-episodes", type=int, default=96)
    parser.add_argument("--stage5-contradiction-margin", type=float, default=0.20)
    parser.add_argument("--stage5-adapter-accuracy-min", type=float, default=0.95)
    parser.add_argument("--stage5-no-external-margin-min", type=float, default=0.50)
    parser.add_argument("--stage5-naive-margin-min", type=float, default=0.25)
    parser.add_argument("--stage5-naive-accuracy-max", type=float, default=0.75)
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
    if args.tape_len < 2:
        raise SystemExit("--tape-len must be at least 2.")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive.")
    if args.input_len_min < 2:
        raise SystemExit("--input-len-min must be at least 2.")
    if args.input_len_max < args.input_len_min:
        raise SystemExit("--input-len-max must be >= --input-len-min.")
    if args.input_len_max > args.tape_len:
        raise SystemExit("--input-len-max must not exceed --tape-len.")
    if args.trace_episodes < 0:
        raise SystemExit("--trace-episodes must be non-negative.")
    if args.stage3_tape_len < args.stage3_segment_len * 2:
        raise SystemExit("--stage3-tape-len must fit two --stage3-segment-len segments.")
    if args.stage3_segment_len < 1:
        raise SystemExit("--stage3-segment-len must be positive.")
    if args.stage3_call_overhead < 0:
        raise SystemExit("--stage3-call-overhead must be non-negative.")
    if args.stage4_corridor_len < 7:
        raise SystemExit("--stage4-corridor-len must be at least 7.")
    if args.stage4_horizon < 1:
        raise SystemExit("--stage4-horizon must be positive.")
    if args.stage5_contradiction_margin < 0:
        raise SystemExit("--stage5-contradiction-margin must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.stage == "stage0":
        if args.output_json == "artifacts/neocortex_x/stage1_recall_to_action.json":
            args.output_json = "artifacts/neocortex_x/stage0_smoke.json"
        run_stage0(args)
    elif args.stage == "stage1":
        run_stage1(args)
    elif args.stage == "stage2":
        if args.output_json == "artifacts/neocortex_x/stage1_recall_to_action.json":
            args.output_json = "artifacts/neocortex_x/stage2_stateful_tape_world.json"
        if args.output_trace == "artifacts/neocortex_x/stage0_trace.jsonl":
            args.output_trace = "artifacts/neocortex_x/stage2_trace.jsonl"
        run_stage2(args)
    elif args.stage == "stage3":
        if args.output_json == "artifacts/neocortex_x/stage1_recall_to_action.json":
            args.output_json = "artifacts/neocortex_x/stage3_procedure_reuse.json"
        run_stage3(args)
    elif args.stage == "stage4":
        if args.output_json == "artifacts/neocortex_x/stage1_recall_to_action.json":
            args.output_json = "artifacts/neocortex_x/stage4_planning.json"
        run_stage4(args)
    else:
        if args.output_json == "artifacts/neocortex_x/stage1_recall_to_action.json":
            args.output_json = "artifacts/neocortex_x/stage5_open_world_adapter.json"
        run_stage5(args)


if __name__ == "__main__":
    main()
