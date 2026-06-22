#!/usr/bin/env python3
"""Replay/evaluate a saved evolved SSM agent genome.

The evolution script writes a `.npz` with A/B/C-style rule tensors. This runner
loads that artifact and evaluates it on a fresh world seed, proving that evolved
policies are reusable outside the search loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from typing import NamedTuple


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
    ACTION_DIM,
    ACTION_NAMES,
    GOAL_POS,
    INPUT_DIM,
    STATE_DIM,
    THREAT_POS,
    Genome,
    count_actions,
    init_positions_energy,
    norm2,
    normalized_policy_entropy,
    reward_score,
    step_genome,
)


class ReplayMetrics(NamedTuple):
    reward: jax.Array
    mean_energy: jax.Array
    initial_goal_distance: jax.Array
    mean_goal_distance: jax.Array
    initial_threat_distance: jax.Array
    mean_threat_distance: jax.Array
    policy_entropy: jax.Array
    action_counts: jax.Array


def load_genome(path: str) -> tuple[Genome, dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    required = ("decay_raw", "input_b", "output_c", "action_bias")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Genome file is missing keys: {missing}")

    genome = Genome(
        decay_raw=jnp.asarray(data["decay_raw"], dtype=jnp.float32),
        input_b=jnp.asarray(data["input_b"], dtype=jnp.float32),
        output_c=jnp.asarray(data["output_c"], dtype=jnp.float32),
        action_bias=jnp.asarray(data["action_bias"], dtype=jnp.float32),
    )
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"].item()))

    if genome.decay_raw.shape != (STATE_DIM,):
        raise ValueError(f"Expected decay_raw shape {(STATE_DIM,)}, got {genome.decay_raw.shape}")
    if genome.input_b.shape != (INPUT_DIM, STATE_DIM):
        raise ValueError(f"Expected input_b shape {(INPUT_DIM, STATE_DIM)}, got {genome.input_b.shape}")
    if genome.output_c.shape != (STATE_DIM, ACTION_DIM):
        raise ValueError(f"Expected output_c shape {(STATE_DIM, ACTION_DIM)}, got {genome.output_c.shape}")
    if genome.action_bias.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action_bias shape {(ACTION_DIM,)}, got {genome.action_bias.shape}")
    return genome, metadata


@partial(jax.jit, static_argnames=("steps",))
def replay(
    genome: Genome,
    init_positions_xy: jax.Array,
    init_energy: jax.Array,
    key: jax.Array,
    *,
    steps: int,
    speed: float,
) -> ReplayMetrics:
    states_h = jnp.zeros((init_positions_xy.shape[0], STATE_DIM), dtype=jnp.float32)
    keys = jax.random.split(key, steps)
    initial_goal_distance = jnp.mean(norm2(init_positions_xy - GOAL_POS))
    initial_threat_distance = jnp.mean(norm2(init_positions_xy - THREAT_POS))

    def body(carry: tuple[jax.Array, jax.Array, jax.Array], step_key: jax.Array):
        positions_xy, energy, states_h = carry
        positions_xy, energy, states_h, actions = step_genome(
            positions_xy, energy, states_h, genome, step_key, speed
        )
        counts = count_actions(actions)
        return (positions_xy, energy, states_h), counts

    (positions_xy, energy, _), action_counts = jax.lax.scan(
        body, (init_positions_xy, init_energy, states_h), keys
    )
    total_counts = jnp.sum(action_counts, axis=0)
    normalized_entropy = normalized_policy_entropy(total_counts)

    mean_goal_distance = jnp.mean(norm2(positions_xy - GOAL_POS))
    mean_threat_distance = jnp.mean(norm2(positions_xy - THREAT_POS))
    mean_energy = jnp.mean(energy)
    reward = reward_score(mean_goal_distance, mean_threat_distance, mean_energy, normalized_entropy)
    return ReplayMetrics(
        reward=reward,
        mean_energy=mean_energy,
        initial_goal_distance=initial_goal_distance,
        mean_goal_distance=mean_goal_distance,
        initial_threat_distance=initial_threat_distance,
        mean_threat_distance=mean_threat_distance,
        policy_entropy=normalized_entropy,
        action_counts=total_counts,
    )


def run(args: argparse.Namespace) -> None:
    genome, metadata = load_genome(args.genome)
    positions_xy, energy = init_positions_energy(args.agents, args.seed)
    key = jax.random.PRNGKey(args.seed + 1000)

    print("TextPy/SoA evolved genome replay")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("Artifact")
    print(f"  genome: {args.genome}")
    print(f"  GENOME_A: {tuple(genome.decay_raw.shape)}")
    print(f"  GENOME_B: {tuple(genome.input_b.shape)}")
    print(f"  GENOME_C: {tuple(genome.output_c.shape)}")
    if metadata:
        stored = {
            key: metadata.get(key)
            for key in ("reward", "goal_distance", "threat_distance", "energy", "policy_entropy")
            if key in metadata
        }
        print(f"  stored_metrics: {stored}")
    print("")

    start = time.perf_counter()
    metrics = replay(genome, positions_xy, energy, key, steps=args.steps, speed=args.speed)
    jax.block_until_ready(metrics.reward)
    elapsed = time.perf_counter() - start
    agent_steps = args.agents * args.steps
    throughput = agent_steps / max(elapsed, 1.0e-9)

    print("Run")
    print(f"  agents: {args.agents:,}")
    print(f"  steps: {args.steps:,}")
    print(f"  agent_steps: {agent_steps:,}")
    print(f"  replay_s: {elapsed:.4f}")
    print(f"  throughput_agent_steps_s: {throughput:,.0f}")
    print(f"  reward: {float(metrics.reward):.4f}")
    print(f"  mean_energy: {float(metrics.mean_energy):.4f}")
    print(
        f"  mean_goal_distance: "
        f"{float(metrics.initial_goal_distance):.4f} -> {float(metrics.mean_goal_distance):.4f}"
    )
    print(
        f"  mean_threat_distance: "
        f"{float(metrics.initial_threat_distance):.4f} -> {float(metrics.mean_threat_distance):.4f}"
    )
    print(f"  policy_entropy: {float(metrics.policy_entropy):.4f}")
    print("  action_counts:")
    for name, count in zip(ACTION_NAMES, np.asarray(jax.device_get(metrics.action_counts))):
        print(f"    {name:>7}: {int(count):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved evolved SSM agent genome.")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--agents", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--speed", type=float, default=0.028)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
