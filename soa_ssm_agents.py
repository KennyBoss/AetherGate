#!/usr/bin/env python3
"""SoA + SSM prototype for many independent logical agents.

The point of this script is not to be a trained model yet. It is a compact
runtime kernel: flat arrays, JAX JIT compilation, and an O(1) recurrent state
update per agent per tick.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial


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

from agent_ssm_core import (
    ACTION_NAMES,
    GOAL_POS,
    StepMetrics,
    THREAT_POS,
    Weights,
    World,
    init_handcoded_weights,
    init_world,
    norm2,
    observe,
    step_world,
)


@partial(jax.jit, static_argnames=("num_steps",))
def simulate(
    world: World,
    weights: Weights,
    key: jax.Array,
    *,
    num_steps: int,
    speed: float,
) -> tuple[World, StepMetrics]:
    keys = jax.random.split(key, num_steps)

    def body(carry: World, step_key: jax.Array) -> tuple[World, StepMetrics]:
        return step_world(carry, weights, step_key, speed)

    return jax.lax.scan(body, world, keys)


def block_until_ready(result: tuple[World, StepMetrics]) -> tuple[World, StepMetrics]:
    final_world, metrics = result
    jax.block_until_ready(final_world.positions_xy)
    jax.block_until_ready(final_world.energy)
    jax.block_until_ready(metrics.mean_energy)
    return result


def run(args: argparse.Namespace) -> None:
    world = init_world(args.agents, args.seed)
    weights = init_handcoded_weights()
    key = jax.random.PRNGKey(args.seed + 1)

    initial_x = observe(world.positions_xy, world.energy)
    initial_goal_distance = float(jnp.mean(norm2(world.positions_xy - GOAL_POS)))
    initial_threat_distance = float(jnp.mean(norm2(world.positions_xy - THREAT_POS)))

    print("TextPy/SoA SSM agent prototype")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("SoA rails")
    print(f"  TENSORS_X: {tuple(initial_x.shape)} float32")
    print(f"  STATES_H:  {tuple(world.states_h.shape)} float32")
    print(f"  POS_XY:    {tuple(world.positions_xy.shape)} float32")
    print(f"  ENERGY:    {tuple(world.energy.shape)} float32")
    print(f"  WEIGHTS_A: {tuple(weights.decay_a.shape)}")
    print(f"  WEIGHTS_B: {tuple(weights.input_b.shape)}")
    print(f"  WEIGHTS_C: {tuple(weights.output_c.shape)}")
    print("")

    compile_start = time.perf_counter()
    warmed = block_until_ready(
        simulate(world, weights, key, num_steps=args.steps, speed=args.speed)
    )
    compile_seconds = time.perf_counter() - compile_start

    exec_key = jax.random.PRNGKey(args.seed + 2)
    exec_start = time.perf_counter()
    final_world, metrics = block_until_ready(
        simulate(world, weights, exec_key, num_steps=args.steps, speed=args.speed)
    )
    exec_seconds = time.perf_counter() - exec_start

    del warmed

    agent_steps = args.agents * args.steps
    throughput = agent_steps / max(exec_seconds, 1.0e-9)
    final_counts = [int(x) for x in metrics.action_counts[-1]]
    final_energy = float(metrics.mean_energy[-1])
    final_goal_distance = float(metrics.mean_goal_distance[-1])
    final_threat_distance = float(metrics.mean_threat_distance[-1])
    state_abs_mean = float(jnp.mean(jnp.abs(final_world.states_h)))

    print("Run")
    print(f"  agents: {args.agents:,}")
    print(f"  steps: {args.steps:,}")
    print(f"  agent_steps: {agent_steps:,}")
    print(f"  compile_plus_first_run_s: {compile_seconds:.4f}")
    print(f"  cached_run_s: {exec_seconds:.4f}")
    print(f"  throughput_agent_steps_s: {throughput:,.0f}")
    print("")
    print("Behavior")
    print(f"  mean_energy: {final_energy:.4f}")
    print(f"  mean_goal_distance: {initial_goal_distance:.4f} -> {final_goal_distance:.4f}")
    print(f"  mean_threat_distance: {initial_threat_distance:.4f} -> {final_threat_distance:.4f}")
    print(f"  mean_abs_hidden_state: {state_abs_mean:.4f}")
    print("  final_action_counts:")
    for name, count in zip(ACTION_NAMES, final_counts):
        print(f"    {name:>7}: {count:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JAX SoA/SSM prototype: many logical agents, one vectorized memory update."
    )
    parser.add_argument("--agents", type=int, default=262_144)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--speed", type=float, default=0.028)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is still experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
