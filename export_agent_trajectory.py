#!/usr/bin/env python3
"""Export small agent trajectories for inspection or visualization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def requested_backend(argv: list[str]) -> str:
    backend = os.environ.get("JAX_PLATFORM_NAME", "cpu")
    for i, arg in enumerate(argv):
        if arg == "--backend" and i + 1 < len(argv):
            backend = argv[i + 1]
        elif arg.startswith("--backend="):
            backend = arg.split("=", 1)[1]
    return backend


REQUESTED_BACKEND = requested_backend(sys.argv[1:])
if REQUESTED_BACKEND == "auto":
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ.pop("JAX_PLATFORMS", None)
else:
    os.environ["JAX_PLATFORM_NAME"] = REQUESTED_BACKEND
    os.environ["JAX_PLATFORMS"] = REQUESTED_BACKEND

import jax
import jax.numpy as jnp
import numpy as np

from agent_ssm_core import (
    ACTION_NAMES,
    GOAL_POS,
    STATE_DIM,
    THREAT_POS,
    World,
    count_actions,
    init_handcoded_weights,
    init_positions_energy,
    init_world,
    norm2,
    normalized_policy_entropy,
    reward_score,
    step_genome,
    step_world_with_actions,
)
from run_evolved_agent import load_genome


def write_rows(
    writer: csv.writer,
    step: int,
    positions_xy: jax.Array,
    energy: jax.Array,
    actions: list[str],
) -> None:
    positions_np = np.asarray(jax.device_get(positions_xy))
    energy_np = np.asarray(jax.device_get(energy))
    for agent_id, ((x, y), e, action) in enumerate(zip(positions_np, energy_np, actions)):
        writer.writerow([step, agent_id, f"{x:.6f}", f"{y:.6f}", f"{e:.6f}", action])


def run_handcoded(args: argparse.Namespace, writer: csv.writer) -> dict[str, object]:
    world = init_world(args.agents, args.seed)
    weights = init_handcoded_weights()
    key = jax.random.PRNGKey(args.seed + 1)
    initial_goal = float(jnp.mean(norm2(world.positions_xy - GOAL_POS)))
    initial_threat = float(jnp.mean(norm2(world.positions_xy - THREAT_POS)))

    write_rows(writer, 0, world.positions_xy, world.energy, ["start"] * args.agents)
    total_counts = jnp.zeros((len(ACTION_NAMES),), dtype=jnp.int32)

    for step in range(1, args.steps + 1):
        key, step_key = jax.random.split(key)
        world, _, actions = step_world_with_actions(world, weights, step_key, args.speed)
        total_counts = total_counts + count_actions(actions)
        action_names = [ACTION_NAMES[int(action)] for action in np.asarray(jax.device_get(actions))]
        write_rows(writer, step, world.positions_xy, world.energy, action_names)

    final_goal = float(jnp.mean(norm2(world.positions_xy - GOAL_POS)))
    final_threat = float(jnp.mean(norm2(world.positions_xy - THREAT_POS)))
    mean_energy = float(jnp.mean(world.energy))
    entropy = float(normalized_policy_entropy(total_counts))
    reward = float(reward_score(final_goal, final_threat, mean_energy, entropy))

    return {
        "mode": "handcoded",
        "initial_goal_distance": initial_goal,
        "final_goal_distance": final_goal,
        "initial_threat_distance": initial_threat,
        "final_threat_distance": final_threat,
        "mean_energy": mean_energy,
        "policy_entropy": entropy,
        "reward": reward,
        "action_counts": action_count_dict(total_counts),
    }


def run_genome(args: argparse.Namespace, writer: csv.writer) -> dict[str, object]:
    if not args.genome:
        raise SystemExit("--genome is required when --mode genome.")
    genome, metadata = load_genome(args.genome)
    positions_xy, energy = init_positions_energy(args.agents, args.seed)
    states_h = jnp.zeros((args.agents, STATE_DIM), dtype=jnp.float32)
    key = jax.random.PRNGKey(args.seed + 1000)
    initial_goal = float(jnp.mean(norm2(positions_xy - GOAL_POS)))
    initial_threat = float(jnp.mean(norm2(positions_xy - THREAT_POS)))

    write_rows(writer, 0, positions_xy, energy, ["start"] * args.agents)
    total_counts = jnp.zeros((len(ACTION_NAMES),), dtype=jnp.int32)

    for step in range(1, args.steps + 1):
        key, step_key = jax.random.split(key)
        positions_xy, energy, states_h, actions = step_genome(
            positions_xy, energy, states_h, genome, step_key, args.speed
        )
        total_counts = total_counts + count_actions(actions)
        action_names = [ACTION_NAMES[int(action)] for action in np.asarray(jax.device_get(actions))]
        write_rows(writer, step, positions_xy, energy, action_names)

    final_goal = float(jnp.mean(norm2(positions_xy - GOAL_POS)))
    final_threat = float(jnp.mean(norm2(positions_xy - THREAT_POS)))
    mean_energy = float(jnp.mean(energy))
    entropy = float(normalized_policy_entropy(total_counts))
    reward = float(reward_score(final_goal, final_threat, mean_energy, entropy))

    return {
        "mode": "genome",
        "genome": args.genome,
        "stored_metadata": metadata,
        "initial_goal_distance": initial_goal,
        "final_goal_distance": final_goal,
        "initial_threat_distance": initial_threat,
        "final_threat_distance": final_threat,
        "mean_energy": mean_energy,
        "policy_entropy": entropy,
        "reward": reward,
        "action_counts": action_count_dict(total_counts),
    }


def action_count_dict(counts: jax.Array) -> dict[str, int]:
    counts_np = np.asarray(jax.device_get(counts))
    return {name: int(value) for name, value in zip(ACTION_NAMES, counts_np)}


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name
    summary_path = output_dir / args.summary_name

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "agent_id", "x", "y", "energy", "action"])
        if args.mode == "handcoded":
            summary = run_handcoded(args, writer)
        else:
            summary = run_genome(args, writer)

    summary.update(
        {
            "agents": args.agents,
            "steps": args.steps,
            "speed": args.speed,
            "seed": args.seed,
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
        }
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("TextPy/SoA trajectory export")
    print(f"  mode: {args.mode}")
    print(f"  csv: {csv_path}")
    print(f"  summary: {summary_path}")
    print(f"  rows: {(args.steps + 1) * args.agents:,}")
    print(
        f"  goal_distance: "
        f"{summary['initial_goal_distance']:.4f} -> {summary['final_goal_distance']:.4f}"
    )
    print(
        f"  threat_distance: "
        f"{summary['initial_threat_distance']:.4f} -> {summary['final_threat_distance']:.4f}"
    )
    print(f"  reward: {summary['reward']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export small agent trajectories to CSV/JSON.")
    parser.add_argument("--mode", choices=("handcoded", "genome"), default="handcoded")
    parser.add_argument("--genome", default=None)
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--speed", type=float, default=0.028)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-dir", default="artifacts/trajectories")
    parser.add_argument("--csv-name", default="trajectory.csv")
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
