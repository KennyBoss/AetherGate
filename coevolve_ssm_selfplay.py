#!/usr/bin/env python3
"""Coevolve SSM agents in a small self-play world.

The normal agent runtime has one static goal and one static threat. This script
turns the threat into an evolving opponent: runners try to reach a goal while
blockers try to occupy positions that make runner policies fail. The point is
not to create AGI; it is to add an open-ended pressure source where the task
changes because another population is adapting.
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

from agent_ssm_core import ACTION_DIM, GOAL_POS, INPUT_DIM, STATE_DIM, Genome, decay_from_raw, norm2


SELFPLAY_INPUT_DIM = 10
ROLE_RUNNER = 0
ROLE_BLOCKER = 1
ROLE_NAMES = ("runner", "blocker")


class MatchMetrics(NamedTuple):
    runner_reward: jax.Array
    blocker_reward: jax.Array
    runner_goal_distance: jax.Array
    runner_blocker_distance: jax.Array
    runner_energy: jax.Array
    blocker_goal_distance: jax.Array
    runner_action_entropy: jax.Array
    blocker_action_entropy: jax.Array


class PopulationMetrics(NamedTuple):
    runner_reward: jax.Array  # [runner_population]
    blocker_reward: jax.Array  # [blocker_population]
    runner_goal_distance: jax.Array  # [runner_population]
    blocker_goal_distance: jax.Array  # [blocker_population]
    runner_diversity: jax.Array
    blocker_diversity: jax.Array
    payoff_matrix: jax.Array  # [runner_population, blocker_population]


def init_population(key: jax.Array, population: int) -> Genome:
    decay_key, b_key, c_key, bias_key = jax.random.split(key, 4)
    return Genome(
        decay_raw=0.35 * jax.random.normal(decay_key, (population, STATE_DIM), dtype=jnp.float32),
        input_b=0.65 * jax.random.normal(b_key, (population, SELFPLAY_INPUT_DIM, STATE_DIM), dtype=jnp.float32),
        output_c=0.55 * jax.random.normal(c_key, (population, STATE_DIM, ACTION_DIM), dtype=jnp.float32),
        action_bias=0.12 * jax.random.normal(bias_key, (population, ACTION_DIM), dtype=jnp.float32),
    )


def take_tree(tree: Genome, indices: jax.Array) -> Genome:
    return jax.tree_util.tree_map(lambda x: x[indices], tree)


def mutate_elites(key: jax.Array, elites: Genome, population: int, mutation_sigma: float) -> Genome:
    elite_count = elites.decay_raw.shape[0]
    repeats = int((population + elite_count - 1) // elite_count)
    parent_indices = jnp.tile(jnp.arange(elite_count), repeats)[:population]
    parents = take_tree(elites, parent_indices)
    keys = jax.random.split(key, 4)
    noise = Genome(
        decay_raw=jax.random.normal(keys[0], parents.decay_raw.shape, dtype=jnp.float32),
        input_b=jax.random.normal(keys[1], parents.input_b.shape, dtype=jnp.float32),
        output_c=jax.random.normal(keys[2], parents.output_c.shape, dtype=jnp.float32),
        action_bias=jax.random.normal(keys[3], parents.action_bias.shape, dtype=jnp.float32),
    )
    mutated = jax.tree_util.tree_map(lambda p, n: p + mutation_sigma * n, parents, noise)
    return Genome(
        decay_raw=mutated.decay_raw.at[:elite_count].set(elites.decay_raw),
        input_b=mutated.input_b.at[:elite_count].set(elites.input_b),
        output_c=mutated.output_c.at[:elite_count].set(elites.output_c),
        action_bias=mutated.action_bias.at[:elite_count].set(elites.action_bias),
    )


def init_positions(key: jax.Array, agents: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    runner_key, blocker_key, runner_energy_key, blocker_energy_key = jax.random.split(key, 4)
    runner_pos = jax.random.uniform(runner_key, (agents, 2), minval=-0.95, maxval=0.25, dtype=jnp.float32)
    blocker_pos = jax.random.uniform(blocker_key, (agents, 2), minval=-0.25, maxval=0.95, dtype=jnp.float32)
    runner_energy = jax.random.uniform(runner_energy_key, (agents,), minval=0.55, maxval=1.0, dtype=jnp.float32)
    blocker_energy = jax.random.uniform(blocker_energy_key, (agents,), minval=0.55, maxval=1.0, dtype=jnp.float32)
    return runner_pos, blocker_pos, runner_energy, blocker_energy


def unit(x: jax.Array) -> jax.Array:
    return x / norm2(x)[:, None]


def observe_selfplay(
    self_pos: jax.Array,
    other_pos: jax.Array,
    energy: jax.Array,
    *,
    role: int,
) -> jax.Array:
    to_goal = GOAL_POS - self_pos
    to_other = other_pos - self_pos
    goal_distance = norm2(to_goal)
    other_distance = norm2(to_other)
    goal_signal = jnp.exp(-3.0 * goal_distance * goal_distance)
    other_signal = jnp.exp(-7.0 * other_distance * other_distance)
    role_value = jnp.full_like(energy, -1.0 if role == ROLE_RUNNER else 1.0)
    bias = jnp.ones_like(energy)
    return jnp.stack(
        (
            to_goal[:, 0],
            to_goal[:, 1],
            goal_signal,
            to_other[:, 0],
            to_other[:, 1],
            other_signal,
            energy,
            1.0 - energy,
            role_value,
            bias,
        ),
        axis=1,
    ).astype(jnp.float32)


def step_policy(
    pos: jax.Array,
    other_pos: jax.Array,
    energy: jax.Array,
    states_h: jax.Array,
    genome: Genome,
    key: jax.Array,
    *,
    role: int,
    speed: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    tensors_x = observe_selfplay(pos, other_pos, energy, role=role)
    states_h = jnp.tanh(states_h * decay_from_raw(genome.decay_raw) + tensors_x @ genome.input_b)
    logits = states_h @ genome.output_c + genome.action_bias
    actions = jnp.argmax(logits, axis=1)

    to_goal = GOAL_POS - pos
    to_other = other_pos - pos
    toward_goal = unit(to_goal)
    toward_other = unit(to_other)
    away_other = -toward_other
    explore = unit(jax.random.normal(key, pos.shape, dtype=jnp.float32))

    if role == ROLE_RUNNER:
        dirs = jnp.stack((toward_goal, away_other, jnp.zeros_like(pos), explore), axis=1)
    else:
        intercept = unit(0.65 * to_goal - 0.35 * to_other)
        dirs = jnp.stack((toward_other, intercept, jnp.zeros_like(pos), explore), axis=1)

    chosen_dirs = jnp.take_along_axis(dirs, actions[:, None, None], axis=1)[:, 0, :]
    moved = actions != 2
    pos = jnp.clip(pos + speed * chosen_dirs, -1.0, 1.0)
    move_cost = 0.006 * moved
    rest_gain = 0.012 * (actions == 2)
    energy = jnp.clip(energy + rest_gain - move_cost, 0.0, 1.0)
    return pos, energy, states_h, actions


def normalized_entropy(actions: jax.Array) -> jax.Array:
    counts = jnp.sum(actions[:, None] == jnp.arange(ACTION_DIM), axis=0)
    probs = counts / jnp.maximum(jnp.sum(counts), 1)
    entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0.0))
    return entropy / jnp.log(jnp.asarray(ACTION_DIM, dtype=jnp.float32))


def evaluate_match(
    runner: Genome,
    blocker: Genome,
    key: jax.Array,
    *,
    agents: int,
    steps: int,
    speed: float,
) -> MatchMetrics:
    runner_pos, blocker_pos, runner_energy, blocker_energy = init_positions(key, agents)
    runner_h = jnp.zeros((agents, STATE_DIM), dtype=jnp.float32)
    blocker_h = jnp.zeros((agents, STATE_DIM), dtype=jnp.float32)
    step_keys = jax.random.split(key, steps * 2).reshape(steps, 2, 2)

    def body(carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], keys: jax.Array):
        runner_pos, blocker_pos, runner_energy, blocker_energy, runner_h, blocker_h = carry
        runner_key = keys[0]
        blocker_key = keys[1]
        runner_pos, runner_energy, runner_h, runner_actions = step_policy(
            runner_pos,
            blocker_pos,
            runner_energy,
            runner_h,
            runner,
            runner_key,
            role=ROLE_RUNNER,
            speed=speed,
        )
        blocker_pos, blocker_energy, blocker_h, blocker_actions = step_policy(
            blocker_pos,
            runner_pos,
            blocker_energy,
            blocker_h,
            blocker,
            blocker_key,
            role=ROLE_BLOCKER,
            speed=speed,
        )
        return (
            runner_pos,
            blocker_pos,
            runner_energy,
            blocker_energy,
            runner_h,
            blocker_h,
        ), (runner_actions, blocker_actions)

    (runner_pos, blocker_pos, runner_energy, blocker_energy, _, _), actions = jax.lax.scan(
        body,
        (runner_pos, blocker_pos, runner_energy, blocker_energy, runner_h, blocker_h),
        step_keys,
    )
    runner_actions, blocker_actions = actions
    runner_goal_distance = jnp.mean(norm2(runner_pos - GOAL_POS))
    blocker_goal_distance = jnp.mean(norm2(blocker_pos - GOAL_POS))
    runner_blocker_distance = jnp.mean(norm2(runner_pos - blocker_pos))
    runner_energy_mean = jnp.mean(runner_energy)
    runner_entropy = normalized_entropy(runner_actions.reshape(-1))
    blocker_entropy = normalized_entropy(blocker_actions.reshape(-1))

    runner_reward = (
        2.4 * (1.25 - runner_goal_distance)
        + 0.55 * runner_blocker_distance
        + 0.55 * runner_energy_mean
        + 0.10 * runner_entropy
    )
    blocker_reward = (
        2.0 * runner_goal_distance
        + 0.75 * (1.25 - runner_blocker_distance)
        + 0.35 * (1.25 - blocker_goal_distance)
        + 0.10 * blocker_entropy
    )
    return MatchMetrics(
        runner_reward=runner_reward,
        blocker_reward=blocker_reward,
        runner_goal_distance=runner_goal_distance,
        runner_blocker_distance=runner_blocker_distance,
        runner_energy=runner_energy_mean,
        blocker_goal_distance=blocker_goal_distance,
        runner_action_entropy=runner_entropy,
        blocker_action_entropy=blocker_entropy,
    )


@partial(jax.jit, static_argnames=("agents", "steps"))
def evaluate_populations(
    runners: Genome,
    blockers: Genome,
    keys: jax.Array,
    *,
    agents: int,
    steps: int,
    speed: float,
) -> PopulationMetrics:
    runner_population = runners.decay_raw.shape[0]
    blocker_population = blockers.decay_raw.shape[0]

    def eval_runner_blocker(runner: Genome, runner_keys: jax.Array):
        def eval_one_blocker(blocker: Genome, key: jax.Array):
            return evaluate_match(runner, blocker, key, agents=agents, steps=steps, speed=speed)

        return jax.vmap(eval_one_blocker, in_axes=(0, 0), out_axes=0)(blockers, runner_keys)

    matches = jax.vmap(eval_runner_blocker, in_axes=(0, 0), out_axes=0)(runners, keys)
    runner_payoffs = matches.runner_reward
    blocker_payoffs = matches.blocker_reward
    runner_reward = jnp.mean(runner_payoffs, axis=1)
    blocker_reward = jnp.mean(blocker_payoffs, axis=0)
    runner_goal_distance = jnp.mean(matches.runner_goal_distance, axis=1)
    blocker_goal_distance = jnp.mean(matches.blocker_goal_distance, axis=0)

    runner_centroid = jnp.mean(runners.action_bias, axis=0, keepdims=True)
    blocker_centroid = jnp.mean(blockers.action_bias, axis=0, keepdims=True)
    runner_diversity = jnp.mean(norm2(runners.action_bias - runner_centroid))
    blocker_diversity = jnp.mean(norm2(blockers.action_bias - blocker_centroid))
    assert runner_population == runner_payoffs.shape[0]
    assert blocker_population == blocker_payoffs.shape[1]
    return PopulationMetrics(
        runner_reward=runner_reward,
        blocker_reward=blocker_reward,
        runner_goal_distance=runner_goal_distance,
        blocker_goal_distance=blocker_goal_distance,
        runner_diversity=runner_diversity,
        blocker_diversity=blocker_diversity,
        payoff_matrix=runner_payoffs,
    )


def scalar(x: jax.Array) -> float:
    return float(jax.device_get(x))


def save_json(path: str, payload: dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace) -> None:
    if args.runner_elites < 1 or args.runner_elites > args.runners:
        raise SystemExit("--runner-elites must be between 1 and --runners.")
    if args.blocker_elites < 1 or args.blocker_elites > args.blockers:
        raise SystemExit("--blocker-elites must be between 1 and --blockers.")

    key = jax.random.PRNGKey(args.seed)
    runner_key, blocker_key = jax.random.split(key)
    runners = init_population(runner_key, args.runners)
    blockers = init_population(blocker_key, args.blockers)

    print("TextPy/SoA SSM self-play coevolution prototype")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("Roles")
    print("  runner: reaches the fixed goal while avoiding the blocker")
    print("  blocker: moves to make the runner fail")
    print("")
    print("Protocol")
    print(f"  runners: {args.runners}")
    print(f"  blockers: {args.blockers}")
    print(f"  agents_per_match: {args.agents}")
    print(f"  steps: {args.steps}")
    print(f"  generations: {args.generations}")
    print("")

    history: list[dict[str, object]] = []
    start = time.perf_counter()
    best_runner_reward = -1.0e9
    best_blocker_reward = -1.0e9
    best_runner_generation = 0
    best_blocker_generation = 0

    for generation in range(args.generations + 1):
        key, eval_key, runner_mutate_key, blocker_mutate_key = jax.random.split(key, 4)
        match_keys = jax.random.split(eval_key, args.runners * args.blockers).reshape(args.runners, args.blockers, 2)
        metrics = evaluate_populations(
            runners,
            blockers,
            match_keys,
            agents=args.agents,
            steps=args.steps,
            speed=args.speed,
        )
        jax.block_until_ready(metrics.runner_reward)
        runner_elite_idx = jnp.argsort(metrics.runner_reward)[-args.runner_elites :][::-1]
        blocker_elite_idx = jnp.argsort(metrics.blocker_reward)[-args.blocker_elites :][::-1]
        runner_best_idx = int(runner_elite_idx[0])
        blocker_best_idx = int(blocker_elite_idx[0])
        runner_best_reward = scalar(metrics.runner_reward[runner_best_idx])
        blocker_best_reward = scalar(metrics.blocker_reward[blocker_best_idx])
        mean_runner_reward = scalar(jnp.mean(metrics.runner_reward))
        mean_blocker_reward = scalar(jnp.mean(metrics.blocker_reward))
        runner_goal = scalar(metrics.runner_goal_distance[runner_best_idx])
        blocker_goal = scalar(metrics.blocker_goal_distance[blocker_best_idx])
        runner_diversity = scalar(metrics.runner_diversity)
        blocker_diversity = scalar(metrics.blocker_diversity)

        if runner_best_reward > best_runner_reward:
            best_runner_reward = runner_best_reward
            best_runner_generation = generation
        if blocker_best_reward > best_blocker_reward:
            best_blocker_reward = blocker_best_reward
            best_blocker_generation = generation

        record = {
            "generation": generation,
            "runner_best_reward": runner_best_reward,
            "runner_mean_reward": mean_runner_reward,
            "blocker_best_reward": blocker_best_reward,
            "blocker_mean_reward": mean_blocker_reward,
            "runner_best_goal_distance": runner_goal,
            "blocker_best_goal_distance": blocker_goal,
            "runner_diversity": runner_diversity,
            "blocker_diversity": blocker_diversity,
            "payoff_matrix": np.asarray(jax.device_get(metrics.payoff_matrix)).round(6).tolist(),
        }
        history.append(record)
        if generation == 0 or generation == args.generations or generation % args.log_every == 0:
            print(
                f"gen={generation:03d} "
                f"runner_best={runner_best_reward:.4f} "
                f"runner_mean={mean_runner_reward:.4f} "
                f"blocker_best={blocker_best_reward:.4f} "
                f"blocker_mean={mean_blocker_reward:.4f} "
                f"runner_goal={runner_goal:.4f} "
                f"diversity=({runner_diversity:.3f},{blocker_diversity:.3f})"
            )

        if generation < args.generations:
            runner_elites = take_tree(runners, runner_elite_idx)
            blocker_elites = take_tree(blockers, blocker_elite_idx)
            runners = mutate_elites(runner_mutate_key, runner_elites, args.runners, args.mutation_sigma)
            blockers = mutate_elites(blocker_mutate_key, blocker_elites, args.blockers, args.mutation_sigma)

    elapsed = time.perf_counter() - start
    evaluated_pair_agent_steps = (args.generations + 1) * args.runners * args.blockers * args.agents * args.steps
    throughput = evaluated_pair_agent_steps / max(elapsed, 1.0e-9)
    first = history[0]
    last = history[-1]
    runner_pressure_delta = last["runner_mean_reward"] - first["runner_mean_reward"]
    blocker_pressure_delta = last["blocker_mean_reward"] - first["blocker_mean_reward"]

    payload = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "protocol": {
            "runners": args.runners,
            "blockers": args.blockers,
            "runner_elites": args.runner_elites,
            "blocker_elites": args.blocker_elites,
            "agents": args.agents,
            "steps": args.steps,
            "generations": args.generations,
            "speed": args.speed,
            "mutation_sigma": args.mutation_sigma,
            "seed": args.seed,
        },
        "roles": {
            "runner": "reach the fixed goal while avoiding blockers",
            "blocker": "adapt positions and policy to reduce runner success",
        },
        "history": history,
        "summary": {
            "best_runner_reward": best_runner_reward,
            "best_runner_generation": best_runner_generation,
            "best_blocker_reward": best_blocker_reward,
            "best_blocker_generation": best_blocker_generation,
            "runner_mean_reward_delta": runner_pressure_delta,
            "blocker_mean_reward_delta": blocker_pressure_delta,
            "evaluated_pair_agent_steps": evaluated_pair_agent_steps,
            "elapsed_s": elapsed,
            "throughput_pair_agent_steps_s": throughput,
        },
        "claim_guardrail": (
            "This is a self-play pressure loop over two small SSM populations. "
            "It is open-ended evaluation scaffolding, not autonomous AGI."
        ),
    }

    print("")
    print("Self-play result")
    print(f"  best_runner_reward: {best_runner_reward:.4f} at gen {best_runner_generation}")
    print(f"  best_blocker_reward: {best_blocker_reward:.4f} at gen {best_blocker_generation}")
    print(f"  runner_mean_reward_delta: {runner_pressure_delta:+.4f}")
    print(f"  blocker_mean_reward_delta: {blocker_pressure_delta:+.4f}")
    print(f"  evaluated_pair_agent_steps: {evaluated_pair_agent_steps:,}")
    print(f"  throughput_pair_agent_steps_s: {throughput:,.0f}")
    print("  guardrail: self-play pressure loop, not autonomous AGI")
    if args.output_json:
        save_json(args.output_json, payload)
        print("")
        print(f"Saved JSON: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coevolve SSM runner/blocker populations in self-play.")
    parser.add_argument("--runners", type=int, default=8)
    parser.add_argument("--blockers", type=int, default=8)
    parser.add_argument("--runner-elites", type=int, default=2)
    parser.add_argument("--blocker-elites", type=int, default=2)
    parser.add_argument("--agents", type=int, default=256)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--mutation-sigma", type=float, default=0.08)
    parser.add_argument("--speed", type=float, default=0.028)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
