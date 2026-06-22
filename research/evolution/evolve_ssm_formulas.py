#!/usr/bin/env python3
"""Evolve SSM update formulas, not only SSM weights.

This is a small NAS-style layer for the agent runtime. Each architecture is a
discrete formula for the hidden-state update. For every candidate formula, the
script runs a short inner weight evolution loop, measures reward/throughput in
the existing agent world, then mutates the best formulas into the next
architecture generation.

It is deliberately a constrained math DSL, not arbitrary self-modifying code.
That keeps the loop reproducible while still letting the system discover update
rules beyond the hand-written baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
    GOAL_POS,
    INPUT_DIM,
    STATE_DIM,
    THREAT_POS,
    Genome,
    count_actions,
    decay_from_raw,
    init_positions_energy,
    norm2,
    normalized_policy_entropy,
    observe,
    reward_score,
)


MEMORY_OPS = ("identity", "tanh", "softsign", "sin", "relu")
INPUT_OPS = ("identity", "tanh", "softsign", "sin", "relu")
COMBINE_OPS = ("add", "add_mul", "gated_input", "gated_mix", "max")
FINAL_OPS = ("tanh", "softsign", "clip", "tanh_plus_sin")

BASELINE_FORMULA = ("identity", "identity", "add", "tanh")


class FormulaSpec(NamedTuple):
    memory_op: str
    input_op: str
    combine_op: str
    final_op: str


class EvalMetrics(NamedTuple):
    reward: jax.Array  # [population]
    mean_energy: jax.Array  # [population]
    mean_goal_distance: jax.Array  # [population]
    mean_threat_distance: jax.Array  # [population]
    policy_entropy: jax.Array  # [population]


def formula_id(formula: FormulaSpec) -> str:
    return (
        f"mem={formula.memory_op}|inp={formula.input_op}|"
        f"combine={formula.combine_op}|final={formula.final_op}"
    )


def formula_payload(formula: FormulaSpec) -> dict[str, str]:
    return {
        "memory_op": formula.memory_op,
        "input_op": formula.input_op,
        "combine_op": formula.combine_op,
        "final_op": formula.final_op,
        "id": formula_id(formula),
    }


def safe_softsign(x: jax.Array) -> jax.Array:
    return x / (1.0 + jnp.abs(x))


def apply_unary(x: jax.Array, op: str) -> jax.Array:
    if op == "identity":
        return x
    if op == "tanh":
        return jnp.tanh(x)
    if op == "softsign":
        return safe_softsign(x)
    if op == "sin":
        return jnp.sin(x)
    if op == "relu":
        return jax.nn.relu(x)
    raise ValueError(f"Unknown unary op: {op}")


def combine_terms(memory_term: jax.Array, input_term: jax.Array, op: str) -> jax.Array:
    if op == "add":
        return memory_term + input_term
    if op == "add_mul":
        return memory_term + input_term + 0.30 * memory_term * input_term
    if op == "gated_input":
        return memory_term * jax.nn.sigmoid(input_term) + input_term
    if op == "gated_mix":
        return memory_term * jax.nn.sigmoid(input_term) + input_term * jax.nn.sigmoid(memory_term)
    if op == "max":
        return jnp.maximum(memory_term, input_term)
    raise ValueError(f"Unknown combine op: {op}")


def apply_final(x: jax.Array, op: str) -> jax.Array:
    if op == "tanh":
        return jnp.tanh(x)
    if op == "softsign":
        return safe_softsign(x)
    if op == "clip":
        return jnp.clip(x, -1.0, 1.0)
    if op == "tanh_plus_sin":
        return jnp.tanh(x) + 0.10 * jnp.sin(x)
    raise ValueError(f"Unknown final op: {op}")


def step_formula(
    positions_xy: jax.Array,
    energy: jax.Array,
    states_h: jax.Array,
    genome: Genome,
    key: jax.Array,
    speed: float,
    *,
    formula: FormulaSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    tensors_x = observe(positions_xy, energy)
    memory_term = apply_unary(states_h * decay_from_raw(genome.decay_raw), formula.memory_op)
    input_term = apply_unary(tensors_x @ genome.input_b, formula.input_op)
    states_h = apply_final(combine_terms(memory_term, input_term, formula.combine_op), formula.final_op)
    logits_y = states_h @ genome.output_c + genome.action_bias
    actions = jnp.argmax(logits_y, axis=1)

    to_goal = GOAL_POS - positions_xy
    to_threat = THREAT_POS - positions_xy
    goal_dir = to_goal / norm2(to_goal)[:, None]
    flee_dir = -to_threat / norm2(to_threat)[:, None]
    rest_dir = jnp.zeros_like(positions_xy)
    explore_dir = jax.random.normal(key, positions_xy.shape, dtype=jnp.float32)
    explore_dir = explore_dir / norm2(explore_dir)[:, None]

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


def evaluate_one(
    genome: Genome,
    init_positions_xy: jax.Array,
    init_energy: jax.Array,
    key: jax.Array,
    *,
    formula: FormulaSpec,
    steps: int,
    speed: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    states_h = jnp.zeros((init_positions_xy.shape[0], STATE_DIM), dtype=jnp.float32)
    keys = jax.random.split(key, steps)

    def body(carry: tuple[jax.Array, jax.Array, jax.Array], step_key: jax.Array):
        positions_xy, energy, states_h = carry
        positions_xy, energy, states_h, actions = step_formula(
            positions_xy,
            energy,
            states_h,
            genome,
            step_key,
            speed,
            formula=formula,
        )
        return (positions_xy, energy, states_h), count_actions(actions)

    (positions_xy, energy, _), action_counts = jax.lax.scan(
        body, (init_positions_xy, init_energy, states_h), keys
    )
    goal_distance = norm2(positions_xy - GOAL_POS)
    threat_distance = norm2(positions_xy - THREAT_POS)
    mean_energy = jnp.mean(energy)
    mean_goal_distance = jnp.mean(goal_distance)
    mean_threat_distance = jnp.mean(threat_distance)
    entropy = normalized_policy_entropy(jnp.sum(action_counts, axis=0))
    reward = reward_score(mean_goal_distance, mean_threat_distance, mean_energy, entropy)
    return reward, mean_energy, mean_goal_distance, mean_threat_distance, entropy


@partial(jax.jit, static_argnames=("formula", "steps"))
def evaluate_population(
    population: Genome,
    init_positions_xy: jax.Array,
    init_energy: jax.Array,
    keys: jax.Array,
    *,
    formula: FormulaSpec,
    steps: int,
    speed: float,
) -> EvalMetrics:
    def mapped_eval(genome: Genome, key: jax.Array):
        return evaluate_one(
            genome,
            init_positions_xy,
            init_energy,
            key,
            formula=formula,
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


def init_weight_population(key: jax.Array, population: int) -> Genome:
    decay_key, b_key, c_key, bias_key = jax.random.split(key, 4)
    return Genome(
        decay_raw=0.35 * jax.random.normal(decay_key, (population, STATE_DIM), dtype=jnp.float32),
        input_b=0.75 * jax.random.normal(b_key, (population, INPUT_DIM, STATE_DIM), dtype=jnp.float32),
        output_c=0.65 * jax.random.normal(c_key, (population, STATE_DIM, ACTION_DIM), dtype=jnp.float32),
        action_bias=0.18 * jax.random.normal(bias_key, (population, ACTION_DIM), dtype=jnp.float32),
    )


def take_tree(tree: Genome, indices: jax.Array) -> Genome:
    return jax.tree_util.tree_map(lambda x: x[indices], tree)


def mutate_weight_elites(key: jax.Array, elites: Genome, population: int, mutation_sigma: float) -> Genome:
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


def scalar(x: jax.Array) -> float:
    return float(jax.device_get(x))


def random_formula(rng: random.Random) -> FormulaSpec:
    return FormulaSpec(
        memory_op=rng.choice(MEMORY_OPS),
        input_op=rng.choice(INPUT_OPS),
        combine_op=rng.choice(COMBINE_OPS),
        final_op=rng.choice(FINAL_OPS),
    )


def mutate_formula(rng: random.Random, formula: FormulaSpec) -> FormulaSpec:
    fields = list(formula)
    choices = [MEMORY_OPS, INPUT_OPS, COMBINE_OPS, FINAL_OPS]
    index = rng.randrange(len(fields))
    alternatives = [value for value in choices[index] if value != fields[index]]
    fields[index] = rng.choice(alternatives)
    return FormulaSpec(*fields)


def unique_formulas(formulas: list[FormulaSpec]) -> list[FormulaSpec]:
    seen: set[str] = set()
    unique: list[FormulaSpec] = []
    for formula in formulas:
        key = formula_id(formula)
        if key not in seen:
            seen.add(key)
            unique.append(formula)
    return unique


def initial_formula_population(rng: random.Random, size: int) -> list[FormulaSpec]:
    formulas = [FormulaSpec(*BASELINE_FORMULA)]
    while len(unique_formulas(formulas)) < size:
        formulas.append(random_formula(rng))
    return unique_formulas(formulas)[:size]


def next_formula_population(
    rng: random.Random,
    elite_formulas: list[FormulaSpec],
    size: int,
) -> list[FormulaSpec]:
    formulas = list(elite_formulas)
    attempts = 0
    while len(unique_formulas(formulas)) < size and attempts < size * 20:
        parent = rng.choice(elite_formulas)
        formulas.append(mutate_formula(rng, parent))
        attempts += 1
    while len(unique_formulas(formulas)) < size:
        formulas.append(random_formula(rng))
    return unique_formulas(formulas)[:size]


def evaluate_formula_architecture(
    formula: FormulaSpec,
    *,
    architecture_generation: int,
    architecture_index: int,
    init_positions_xy: jax.Array,
    init_energy: jax.Array,
    args: argparse.Namespace,
) -> dict[str, object]:
    base_seed = args.seed + 10_000 * architecture_generation + 97 * architecture_index
    key = jax.random.PRNGKey(base_seed)
    population = init_weight_population(key, args.weight_population)

    compile_keys = jax.random.split(jax.random.PRNGKey(args.seed + 777), args.weight_population)
    compile_start = time.perf_counter()
    warmup = evaluate_population(
        population,
        init_positions_xy,
        init_energy,
        compile_keys,
        formula=formula,
        steps=args.steps,
        speed=args.speed,
    )
    jax.block_until_ready(warmup.reward)
    compile_s = time.perf_counter() - compile_start

    best: dict[str, object] | None = None
    start = time.perf_counter()
    for weight_generation in range(args.weight_generations + 1):
        key, eval_key, mutate_key = jax.random.split(key, 3)
        eval_keys = jax.random.split(eval_key, args.weight_population)
        metrics = evaluate_population(
            population,
            init_positions_xy,
            init_energy,
            eval_keys,
            formula=formula,
            steps=args.steps,
            speed=args.speed,
        )
        jax.block_until_ready(metrics.reward)
        elite_indices = jnp.argsort(metrics.reward)[-args.weight_elites :][::-1]
        best_idx = int(elite_indices[0])
        candidate = {
            "weight_generation": weight_generation,
            "reward": scalar(metrics.reward[best_idx]),
            "mean_reward": scalar(jnp.mean(metrics.reward)),
            "goal_distance": scalar(metrics.mean_goal_distance[best_idx]),
            "threat_distance": scalar(metrics.mean_threat_distance[best_idx]),
            "energy": scalar(metrics.mean_energy[best_idx]),
            "policy_entropy": scalar(metrics.policy_entropy[best_idx]),
        }
        if best is None or float(candidate["reward"]) > float(best["reward"]):
            best = candidate
        if weight_generation < args.weight_generations:
            elites = take_tree(population, elite_indices)
            population = mutate_weight_elites(
                mutate_key,
                elites,
                args.weight_population,
                args.weight_mutation_sigma,
            )

    elapsed = time.perf_counter() - start
    evaluated_agent_steps = (args.weight_generations + 1) * args.weight_population * args.agents * args.steps
    assert best is not None
    return {
        "architecture_generation": architecture_generation,
        "architecture_index": architecture_index,
        "formula": formula_payload(formula),
        "best": best,
        "compile_s": compile_s,
        "weight_evolution_s": elapsed,
        "evaluated_agent_steps": evaluated_agent_steps,
        "throughput_agent_steps_s": evaluated_agent_steps / max(elapsed, 1.0e-9),
    }


def save_json(path: str, payload: dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace) -> None:
    if args.architecture_elites < 1 or args.architecture_elites > args.architecture_population:
        raise SystemExit("--architecture-elites must be between 1 and --architecture-population.")
    if args.weight_elites < 1 or args.weight_elites > args.weight_population:
        raise SystemExit("--weight-elites must be between 1 and --weight-population.")

    rng = random.Random(args.seed)
    init_positions_xy, init_energy = init_positions_energy(args.agents, args.seed)
    formulas = initial_formula_population(rng, args.architecture_population)
    all_records: list[dict[str, object]] = []

    print("TextPy/SoA formula evolution prototype")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("Search space")
    print(f"  memory_ops: {', '.join(MEMORY_OPS)}")
    print(f"  input_ops: {', '.join(INPUT_OPS)}")
    print(f"  combine_ops: {', '.join(COMBINE_OPS)}")
    print(f"  final_ops: {', '.join(FINAL_OPS)}")
    print("")
    print("Protocol")
    print(f"  architecture_population: {args.architecture_population}")
    print(f"  architecture_generations: {args.architecture_generations}")
    print(f"  weight_population: {args.weight_population}")
    print(f"  weight_generations_per_formula: {args.weight_generations}")
    print(f"  agents: {args.agents}")
    print(f"  steps: {args.steps}")
    print("")

    total_start = time.perf_counter()
    for architecture_generation in range(args.architecture_generations + 1):
        print(f"Architecture generation {architecture_generation}")
        generation_records: list[dict[str, object]] = []
        for architecture_index, formula in enumerate(formulas):
            record = evaluate_formula_architecture(
                formula,
                architecture_generation=architecture_generation,
                architecture_index=architecture_index,
                init_positions_xy=init_positions_xy,
                init_energy=init_energy,
                args=args,
            )
            generation_records.append(record)
            all_records.append(record)
            best = record["best"]
            print(
                f"  idx={architecture_index:02d} "
                f"reward={best['reward']:.4f} "
                f"goal={best['goal_distance']:.4f} "
                f"entropy={best['policy_entropy']:.3f} "
                f"formula={record['formula']['id']}"
            )

        ranked = sorted(generation_records, key=lambda item: float(item["best"]["reward"]), reverse=True)
        elites = [
            FormulaSpec(
                record["formula"]["memory_op"],
                record["formula"]["input_op"],
                record["formula"]["combine_op"],
                record["formula"]["final_op"],
            )
            for record in ranked[: args.architecture_elites]
        ]
        best_record = ranked[0]
        print(
            f"  best_formula: {best_record['formula']['id']} "
            f"reward={best_record['best']['reward']:.4f}"
        )
        if architecture_generation < args.architecture_generations:
            formulas = next_formula_population(rng, elites, args.architecture_population)
        print("")

    elapsed = time.perf_counter() - total_start
    leaderboard = sorted(all_records, key=lambda item: float(item["best"]["reward"]), reverse=True)
    best = leaderboard[0]
    baseline_records = [
        record
        for record in all_records
        if record["formula"]["id"] == formula_id(FormulaSpec(*BASELINE_FORMULA))
    ]
    baseline = max(baseline_records, key=lambda item: float(item["best"]["reward"])) if baseline_records else None
    best_minus_baseline = None
    if baseline is not None:
        best_minus_baseline = float(best["best"]["reward"]) - float(baseline["best"]["reward"])

    payload = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "protocol": {
            "architecture_population": args.architecture_population,
            "architecture_elites": args.architecture_elites,
            "architecture_generations": args.architecture_generations,
            "weight_population": args.weight_population,
            "weight_elites": args.weight_elites,
            "weight_generations": args.weight_generations,
            "weight_mutation_sigma": args.weight_mutation_sigma,
            "agents": args.agents,
            "steps": args.steps,
            "speed": args.speed,
            "seed": args.seed,
            "search_space_size": len(MEMORY_OPS) * len(INPUT_OPS) * len(COMBINE_OPS) * len(FINAL_OPS),
        },
        "baseline_formula": formula_payload(FormulaSpec(*BASELINE_FORMULA)),
        "baseline_record": baseline,
        "best_record": best,
        "best_minus_baseline_reward": best_minus_baseline,
        "leaderboard": leaderboard,
        "elapsed_s": elapsed,
        "claim_guardrail": (
            "This is constrained formula search over a small DSL. It is a NAS/meta-learning "
            "experiment, not evidence of autonomous AGI."
        ),
    }

    print("Best formula search result")
    print(f"  formula: {best['formula']['id']}")
    print(f"  reward: {best['best']['reward']:.4f}")
    print(f"  goal_distance: {best['best']['goal_distance']:.4f}")
    print(f"  threat_distance: {best['best']['threat_distance']:.4f}")
    print(f"  energy: {best['best']['energy']:.4f}")
    print(f"  policy_entropy: {best['best']['policy_entropy']:.4f}")
    if best_minus_baseline is not None:
        print(f"  best_minus_baseline_reward: {best_minus_baseline:+.4f}")
    print(f"  elapsed_s: {elapsed:.4f}")
    print("  guardrail: constrained formula search, not autonomous AGI")

    if args.output_json:
        save_json(args.output_json, payload)
        print("")
        print(f"Saved JSON: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve SSM hidden-state formulas in the agent world.")
    parser.add_argument("--architecture-population", type=int, default=5)
    parser.add_argument("--architecture-elites", type=int, default=2)
    parser.add_argument("--architecture-generations", type=int, default=1)
    parser.add_argument("--weight-population", type=int, default=16)
    parser.add_argument("--weight-elites", type=int, default=4)
    parser.add_argument("--weight-generations", type=int, default=2)
    parser.add_argument("--weight-mutation-sigma", type=float, default=0.08)
    parser.add_argument("--agents", type=int, default=512)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--speed", type=float, default=0.028)
    parser.add_argument("--seed", type=int, default=37)
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
