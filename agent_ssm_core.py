"""Shared SoA/SSM agent-world primitives.

This module holds the common math used by the hand-coded agent runtime,
evolutionary search, and replay harness. Scripts still own their CLI and
benchmark/reporting behavior; this file owns the world dynamics.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


INPUT_DIM = 8
STATE_DIM = 8
ACTION_DIM = 4

ACTION_NAMES = ("forage", "flee", "rest", "explore")
GOAL_POS = jnp.array([0.72, 0.58], dtype=jnp.float32)
THREAT_POS = jnp.array([-0.35, -0.18], dtype=jnp.float32)


class World(NamedTuple):
    positions_xy: jax.Array  # [agents, 2]
    energy: jax.Array  # [agents]
    states_h: jax.Array  # [agents, STATE_DIM]


class Weights(NamedTuple):
    decay_a: jax.Array  # [STATE_DIM]
    input_b: jax.Array  # [INPUT_DIM, STATE_DIM]
    output_c: jax.Array  # [STATE_DIM, ACTION_DIM]
    action_bias: jax.Array  # [ACTION_DIM]


class Genome(NamedTuple):
    decay_raw: jax.Array  # [STATE_DIM] or [population, STATE_DIM]
    input_b: jax.Array  # [INPUT_DIM, STATE_DIM] or [population, INPUT_DIM, STATE_DIM]
    output_c: jax.Array  # [STATE_DIM, ACTION_DIM] or [population, STATE_DIM, ACTION_DIM]
    action_bias: jax.Array  # [ACTION_DIM] or [population, ACTION_DIM]


class StepMetrics(NamedTuple):
    action_counts: jax.Array  # [ACTION_DIM]
    mean_energy: jax.Array
    mean_goal_distance: jax.Array
    mean_threat_distance: jax.Array


def norm2(x: jax.Array, eps: float = 1.0e-6) -> jax.Array:
    return jnp.sqrt(jnp.sum(x * x, axis=-1) + eps)


def unit(x: jax.Array) -> jax.Array:
    return x / norm2(x)[..., None]


def decay_from_raw(raw: jax.Array) -> jax.Array:
    return 0.03 + 0.95 * jax.nn.sigmoid(raw)


def observe(positions_xy: jax.Array, energy: jax.Array) -> jax.Array:
    """Build TENSORS_X from independent SoA rails."""
    to_goal = GOAL_POS - positions_xy
    to_threat = THREAT_POS - positions_xy
    goal_distance = norm2(to_goal)
    threat_distance = norm2(to_threat)

    food_signal = jnp.exp(-3.0 * goal_distance * goal_distance)
    threat_signal = jnp.exp(-12.0 * threat_distance * threat_distance)
    low_energy = 1.0 - energy
    bias = jnp.ones_like(energy)

    return jnp.stack(
        (
            to_goal[:, 0],
            to_goal[:, 1],
            food_signal,
            to_threat[:, 0],
            to_threat[:, 1],
            threat_signal,
            low_energy,
            bias,
        ),
        axis=1,
    ).astype(jnp.float32)


def init_positions_energy(num_agents: int, seed: int) -> tuple[jax.Array, jax.Array]:
    key = jax.random.PRNGKey(seed)
    pos_key, energy_key = jax.random.split(key)
    positions_xy = jax.random.uniform(
        pos_key, (num_agents, 2), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    energy = jax.random.uniform(
        energy_key, (num_agents,), minval=0.45, maxval=1.0, dtype=jnp.float32
    )
    return positions_xy, energy


def init_world(num_agents: int, seed: int) -> World:
    positions_xy, energy = init_positions_energy(num_agents, seed)
    states_h = jnp.zeros((num_agents, STATE_DIM), dtype=jnp.float32)
    return World(positions_xy=positions_xy, energy=energy, states_h=states_h)


def init_handcoded_weights() -> Weights:
    """Hand-coded rules that make the first prototype interpretable."""
    decay_a = jnp.array([0.86, 0.86, 0.74, 0.74, 0.90, 0.82, 0.94, 0.99])

    input_b = jnp.zeros((INPUT_DIM, STATE_DIM), dtype=jnp.float32)
    input_b = input_b.at[0, 0].set(1.15)  # remembered goal x
    input_b = input_b.at[1, 1].set(1.15)  # remembered goal y
    input_b = input_b.at[3, 2].set(1.05)  # remembered threat x
    input_b = input_b.at[4, 3].set(1.05)  # remembered threat y
    input_b = input_b.at[2, 4].set(2.00)  # food/goal signal
    input_b = input_b.at[5, 5].set(2.70)  # danger signal
    input_b = input_b.at[6, 6].set(2.20)  # fatigue
    input_b = input_b.at[7, 7].set(0.35)  # base drive

    output_c = jnp.zeros((STATE_DIM, ACTION_DIM), dtype=jnp.float32)
    output_c = output_c.at[4, 0].set(0.80)  # forage likes food memory
    output_c = output_c.at[5, 0].set(-1.20)
    output_c = output_c.at[6, 0].set(-1.20)
    output_c = output_c.at[7, 0].set(0.38)

    output_c = output_c.at[5, 1].set(2.20)  # flee when danger is remembered
    output_c = output_c.at[6, 1].set(-0.25)

    output_c = output_c.at[5, 2].set(-0.30)
    output_c = output_c.at[6, 2].set(2.35)  # rest when fatigue is remembered
    output_c = output_c.at[7, 2].set(-0.08)

    output_c = output_c.at[4, 3].set(-0.25)
    output_c = output_c.at[5, 3].set(-0.45)
    output_c = output_c.at[6, 3].set(-0.30)
    output_c = output_c.at[7, 3].set(0.28)  # explore by default

    action_bias = jnp.array([0.24, -0.10, -0.24, 0.12], dtype=jnp.float32)

    return Weights(
        decay_a=decay_a.astype(jnp.float32),
        input_b=input_b,
        output_c=output_c,
        action_bias=action_bias,
    )


def count_actions(actions: jax.Array) -> jax.Array:
    return jnp.sum(actions[:, None] == jnp.arange(ACTION_DIM), axis=0)


def normalized_policy_entropy(action_counts: jax.Array) -> jax.Array:
    probs = action_counts / jnp.maximum(jnp.sum(action_counts), 1)
    entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0.0))
    return entropy / jnp.log(jnp.asarray(ACTION_DIM, dtype=jnp.float32))


def reward_score(
    mean_goal_distance: jax.Array,
    mean_threat_distance: jax.Array,
    mean_energy: jax.Array,
    normalized_entropy: jax.Array,
) -> jax.Array:
    return (
        2.2 * (1.25 - mean_goal_distance)
        + 1.1 * mean_threat_distance
        + 0.9 * mean_energy
        + 0.08 * normalized_entropy
    )


def step_arrays(
    positions_xy: jax.Array,
    energy: jax.Array,
    states_h: jax.Array,
    decay_a: jax.Array,
    input_b: jax.Array,
    output_c: jax.Array,
    action_bias: jax.Array,
    key: jax.Array,
    speed: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    tensors_x = observe(positions_xy, energy)
    states_h = jnp.tanh(states_h * decay_a + tensors_x @ input_b)
    logits_y = states_h @ output_c + action_bias
    actions = jnp.argmax(logits_y, axis=1)

    to_goal = GOAL_POS - positions_xy
    to_threat = THREAT_POS - positions_xy
    goal_dir = unit(to_goal)
    flee_dir = -unit(to_threat)
    rest_dir = jnp.zeros_like(positions_xy)
    explore_dir = unit(jax.random.normal(key, positions_xy.shape, dtype=jnp.float32))

    action_dirs = jnp.stack((goal_dir, flee_dir, rest_dir, explore_dir), axis=1)
    chosen_dirs = jnp.take_along_axis(action_dirs, actions[:, None, None], axis=1)[:, 0, :]

    moved = actions != 2
    positions_xy = jnp.clip(positions_xy + speed * chosen_dirs, -1.0, 1.0)

    goal_distance = norm2(positions_xy - GOAL_POS)
    threat_distance = norm2(positions_xy - THREAT_POS)
    food_gain = 0.026 * jnp.exp(-10.0 * goal_distance * goal_distance) * (actions == 0)
    rest_gain = 0.018 * (actions == 2)
    move_cost = 0.006 * moved
    danger_cost = 0.018 * jnp.exp(-18.0 * threat_distance * threat_distance)
    energy = jnp.clip(energy + food_gain + rest_gain - move_cost - danger_cost, 0.0, 1.0)
    return positions_xy, energy, states_h, actions


def step_world(world: World, weights: Weights, key: jax.Array, speed: float) -> tuple[World, StepMetrics]:
    world, metrics, _ = step_world_with_actions(world, weights, key, speed)
    return world, metrics


def step_world_with_actions(
    world: World, weights: Weights, key: jax.Array, speed: float
) -> tuple[World, StepMetrics, jax.Array]:
    positions_xy, energy, states_h, actions = step_arrays(
        world.positions_xy,
        world.energy,
        world.states_h,
        weights.decay_a,
        weights.input_b,
        weights.output_c,
        weights.action_bias,
        key,
        speed,
    )
    goal_distance = norm2(positions_xy - GOAL_POS)
    threat_distance = norm2(positions_xy - THREAT_POS)
    metrics = StepMetrics(
        action_counts=count_actions(actions),
        mean_energy=jnp.mean(energy),
        mean_goal_distance=jnp.mean(goal_distance),
        mean_threat_distance=jnp.mean(threat_distance),
    )
    return World(positions_xy=positions_xy, energy=energy, states_h=states_h), metrics, actions


def step_genome(
    positions_xy: jax.Array,
    energy: jax.Array,
    states_h: jax.Array,
    genome: Genome,
    key: jax.Array,
    speed: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return step_arrays(
        positions_xy,
        energy,
        states_h,
        decay_from_raw(genome.decay_raw),
        genome.input_b,
        genome.output_c,
        genome.action_bias,
        key,
        speed,
    )
