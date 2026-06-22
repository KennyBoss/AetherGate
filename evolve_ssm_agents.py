#!/usr/bin/env python3
"""Evolve SSM agent rules with a vectorized JAX population loop.

This is the third prototype layer. The text runtime learns A/B/C by gradients;
this script searches over whole rule sets by population selection and mutation.
The population itself is stored as SoA arrays: every genome field has a leading
population axis.
"""

from __future__ import annotations

import argparse
import json
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
import numpy as np

from typing import NamedTuple

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


class EvalMetrics(NamedTuple):
    reward: jax.Array  # [population]
    mean_energy: jax.Array  # [population]
    mean_goal_distance: jax.Array  # [population]
    mean_threat_distance: jax.Array  # [population]
    policy_entropy: jax.Array  # [population]


def init_population(key: jax.Array, population: int) -> Genome:
    decay_key, b_key, c_key, bias_key = jax.random.split(key, 4)
    return Genome(
        decay_raw=0.35 * jax.random.normal(decay_key, (population, STATE_DIM), dtype=jnp.float32),
        input_b=0.75 * jax.random.normal(b_key, (population, INPUT_DIM, STATE_DIM), dtype=jnp.float32),
        output_c=0.65 * jax.random.normal(c_key, (population, STATE_DIM, ACTION_DIM), dtype=jnp.float32),
        action_bias=0.18 * jax.random.normal(bias_key, (population, ACTION_DIM), dtype=jnp.float32),
    )


def evaluate_one(
    genome: Genome,
    init_positions_xy: jax.Array,
    init_energy: jax.Array,
    key: jax.Array,
    *,
    steps: int,
    speed: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    states_h = jnp.zeros((init_positions_xy.shape[0], STATE_DIM), dtype=jnp.float32)
    keys = jax.random.split(key, steps)

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

    goal_distance = norm2(positions_xy - GOAL_POS)
    threat_distance = norm2(positions_xy - THREAT_POS)
    mean_energy = jnp.mean(energy)
    mean_goal_distance = jnp.mean(goal_distance)
    mean_threat_distance = jnp.mean(threat_distance)

    normalized_entropy = normalized_policy_entropy(jnp.sum(action_counts, axis=0))
    reward = reward_score(mean_goal_distance, mean_threat_distance, mean_energy, normalized_entropy)
    return reward, mean_energy, mean_goal_distance, mean_threat_distance, normalized_entropy


@partial(jax.jit, static_argnames=("steps",))
def evaluate_population(
    population: Genome,
    init_positions_xy: jax.Array,
    init_energy: jax.Array,
    keys: jax.Array,
    *,
    steps: int,
    speed: float,
) -> EvalMetrics:
    def mapped_eval(genome: Genome, key: jax.Array):
        return evaluate_one(
            genome,
            init_positions_xy,
            init_energy,
            key,
            steps=steps,
            speed=speed,
        )

    reward, energy, goal_distance, threat_distance, entropy = jax.vmap(
        mapped_eval, in_axes=(0, 0), out_axes=0
    )(population, keys)
    return EvalMetrics(
        reward=reward,
        mean_energy=energy,
        mean_goal_distance=goal_distance,
        mean_threat_distance=threat_distance,
        policy_entropy=entropy,
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
    mutated = jax.tree_util.tree_map(
        lambda p, n: p + mutation_sigma * n,
        parents,
        noise,
    )
    return Genome(
        decay_raw=mutated.decay_raw.at[:elite_count].set(elites.decay_raw),
        input_b=mutated.input_b.at[:elite_count].set(elites.input_b),
        output_c=mutated.output_c.at[:elite_count].set(elites.output_c),
        action_bias=mutated.action_bias.at[:elite_count].set(elites.action_bias),
    )


def top_action_bias(genome: Genome) -> list[tuple[str, float]]:
    bias = np_array(genome.action_bias)
    return [(name, float(value)) for name, value in zip(ACTION_NAMES, bias)]


def np_array(x: jax.Array):
    return jax.device_get(x)


def save_genome(
    path: str,
    genome: Genome,
    metrics: tuple[float, float, float, float, float],
    args: argparse.Namespace,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    reward, goal_distance, threat_distance, energy, entropy = metrics
    metadata = {
        "format_version": 1,
        "reward": reward,
        "goal_distance": goal_distance,
        "threat_distance": threat_distance,
        "energy": energy,
        "policy_entropy": entropy,
        "input_dim": INPUT_DIM,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_names": ACTION_NAMES,
        "args": {
            "population": args.population,
            "elites": args.elites,
            "agents": args.agents,
            "steps": args.steps,
            "generations": args.generations,
            "mutation_sigma": args.mutation_sigma,
            "speed": args.speed,
            "seed": args.seed,
        },
    }
    np.savez_compressed(
        path,
        decay_raw=np.asarray(jax.device_get(genome.decay_raw)),
        input_b=np.asarray(jax.device_get(genome.input_b)),
        output_c=np.asarray(jax.device_get(genome.output_c)),
        action_bias=np.asarray(jax.device_get(genome.action_bias)),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def run(args: argparse.Namespace) -> None:
    if args.elites < 1 or args.elites > args.population:
        raise SystemExit("--elites must be between 1 and --population.")

    init_positions_xy, init_energy = init_positions_energy(args.agents, args.seed)
    key = jax.random.PRNGKey(args.seed + 100)
    population = init_population(key, args.population)

    print("TextPy/SoA evolutionary SSM agent prototype")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("SoA population rails")
    print(f"  POPULATION:   {args.population:,}")
    print(f"  AGENTS:       {args.agents:,}")
    print(f"  STEPS:        {args.steps:,}")
    print(f"  STATES_H:     {(args.agents, STATE_DIM)} float32 per candidate")
    print(f"  GENOME_A:     {tuple(population.decay_raw.shape)}")
    print(f"  GENOME_B:     {tuple(population.input_b.shape)}")
    print(f"  GENOME_C:     {tuple(population.output_c.shape)}")
    print("")

    start = time.perf_counter()
    best_metrics = None
    best_genome = None

    for generation in range(args.generations + 1):
        key, eval_key, mutate_key = jax.random.split(key, 3)
        eval_keys = jax.random.split(eval_key, args.population)
        metrics = evaluate_population(
            population,
            init_positions_xy,
            init_energy,
            eval_keys,
            steps=args.steps,
            speed=args.speed,
        )
        jax.block_until_ready(metrics.reward)

        elite_indices = jnp.argsort(metrics.reward)[-args.elites :][::-1]
        elites = take_tree(population, elite_indices)

        best_idx = int(elite_indices[0])
        best_reward = float(metrics.reward[best_idx])
        mean_reward = float(jnp.mean(metrics.reward))
        best_goal = float(metrics.mean_goal_distance[best_idx])
        best_threat = float(metrics.mean_threat_distance[best_idx])
        best_energy = float(metrics.mean_energy[best_idx])
        best_entropy = float(metrics.policy_entropy[best_idx])

        if best_metrics is None or best_reward > best_metrics[0]:
            best_metrics = (best_reward, best_goal, best_threat, best_energy, best_entropy)
            best_genome = take_tree(population, jnp.asarray([best_idx]))

        if generation == 0 or generation == args.generations or generation % args.log_every == 0:
            print(
                f"gen={generation:03d} "
                f"best_reward={best_reward:.4f} "
                f"mean_reward={mean_reward:.4f} "
                f"goal={best_goal:.4f} "
                f"threat={best_threat:.4f} "
                f"energy={best_energy:.4f} "
                f"entropy={best_entropy:.3f}"
            )

        if generation < args.generations:
            population = mutate_elites(mutate_key, elites, args.population, args.mutation_sigma)

    elapsed = time.perf_counter() - start
    eval_agent_steps = (args.generations + 1) * args.population * args.agents * args.steps
    throughput = eval_agent_steps / max(elapsed, 1.0e-9)

    assert best_metrics is not None
    assert best_genome is not None
    best_reward, best_goal, best_threat, best_energy, best_entropy = best_metrics
    best_single = jax.tree_util.tree_map(lambda x: x[0], best_genome)

    print("")
    print("Run")
    print(f"  evaluated_agent_steps: {eval_agent_steps:,}")
    print(f"  evolution_s: {elapsed:.4f}")
    print(f"  throughput_agent_steps_s: {throughput:,.0f}")
    print(f"  best_reward_seen: {best_reward:.4f}")
    print(f"  best_goal_distance: {best_goal:.4f}")
    print(f"  best_threat_distance: {best_threat:.4f}")
    print(f"  best_energy: {best_energy:.4f}")
    print(f"  best_policy_entropy: {best_entropy:.4f}")
    if args.save_best:
        save_genome(args.save_best, best_single, best_metrics, args)
        print(f"  saved_best: {args.save_best}")
    print("  best_action_bias:")
    for name, value in top_action_bias(best_single):
        print(f"    {name:>7}: {value:+.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolve JAX SoA/SSM agent rules with population selection."
    )
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--agents", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--generations", type=int, default=18)
    parser.add_argument("--mutation-sigma", type=float, default=0.07)
    parser.add_argument("--speed", type=float, default=0.028)
    parser.add_argument("--log-every", type=int, default=3)
    parser.add_argument("--save-best", default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
