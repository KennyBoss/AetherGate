#!/usr/bin/env python3
"""Evolve bounded-loop CodePy programs with a tape and pointer.

This is the first stateful CodePy prototype. Unlike `evolve_code_blocks.py`,
which executes a straight-line register DSL, this evaluator has:

* fixed tape memory (`tape[8]` by default),
* a pointer (`ptr`),
* a program counter (`pc`),
* bounded loop execution via `jax.lax.scan`.

It remains safe DSL search. It never executes generated Python source and it
does not run unbounded loops.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from code_tape_prior import TRACE_FEATURE_NAMES, trace_embedding_from_metrics


CLIP_VALUE = 1.0e6

TAPE_BLOCKS = (
    "noop",
    "r0 = 0",
    "r1 = 0",
    "r0 = tape[ptr]",
    "r1 = tape[ptr]",
    "tape[ptr] = r0",
    "tape[ptr] = r1",
    "ptr += 1",
    "ptr -= 1",
    "ptr = 0",
    "r0 += r1",
    "r0 += 1",
    "r0 -= 1",
    "p = r0 > 0",
    "p = r1 > 0",
    "p = ptr < input_len",
    "r0 += where(p, 1, 0)",
    "r0 = where(p, r1, r0)",
    "pc = 0",
    "pc = where(p, 0, pc + 1)",
    "p = tape[ptr] > r0",
    "p = tape[ptr] > r1",
    "p = ptr < input_len - 1",
    "r0 = ptr",
    "r1 = ptr",
    "r0 = where(p, tape[ptr], r0)",
    "r1 = where(p, tape[ptr], r1)",
    "r0 = where(p, ptr, r0)",
    "r1 = where(p, ptr, r1)",
    "pc = 3",
    "pc = where(p, 3, pc + 1)",
    "halt",
    "r0 = r1",
    "r1 = r0",
    "r1 = where(p, r0, r1)",
)

TARGET_DESCRIPTIONS = {
    "sum4": "sum of a length-4 vector",
    "count_positive4": "number of positive values in a length-4 vector",
    "argmax_index4": "index of the first maximum value in a length-4 vector",
}

REFERENCE_PROGRAMS = {
    "sum4": [4, 10, 7, 18],
    "count_positive4": [4, 14, 16, 7, 18],
    "argmax_index4": [3, 24, 7, 20, 25, 28, 7, 15, 30, 32, 31],
}


class EvalMetrics(NamedTuple):
    reward: jax.Array
    train_mse: jax.Array
    holdout_mse: jax.Array
    max_abs_error: jax.Array
    holdout_max_abs_error: jax.Array
    active_blocks: jax.Array
    shaping_bonus: jax.Array
    read_steps: jax.Array
    write_steps: jax.Array
    move_steps: jax.Array
    loop_steps: jax.Array
    accumulate_steps: jax.Array
    predicate_steps: jax.Array
    conditional_steps: jax.Array
    max_ptr: jax.Array


class TapeExecution(NamedTuple):
    output: jax.Array
    read_steps: jax.Array
    write_steps: jax.Array
    move_steps: jax.Array
    loop_steps: jax.Array
    accumulate_steps: jax.Array
    predicate_steps: jax.Array
    conditional_steps: jax.Array
    max_ptr: jax.Array


def target_values(vectors: jax.Array, target: str) -> jax.Array:
    if target == "sum4":
        return jnp.sum(vectors, axis=1)
    if target == "count_positive4":
        return jnp.sum(vectors > 0.0, axis=1).astype(jnp.float32)
    if target == "argmax_index4":
        return jnp.argmax(vectors, axis=1).astype(jnp.float32)
    raise ValueError(f"Unknown target: {target}")


def make_dataset(
    target: str, cases: int, input_len: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    values = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    train_rows = []
    holdout_rows = []
    for i in range(cases):
        train_rows.append([values[(i * (j + 2) + j) % len(values)] for j in range(input_len)])
        holdout_rows.append(
            [values[((i + 1) * (j + 3) + 2 * j + 1) % len(values)] for j in range(input_len)]
        )
    train_x = jnp.asarray(train_rows, dtype=jnp.float32)
    holdout_x = jnp.asarray(holdout_rows, dtype=jnp.float32)
    return train_x, target_values(train_x, target), holdout_x, target_values(holdout_x, target)


def select_op(op_ids: jax.Array, op_id: int, value: jax.Array, fallback: jax.Array) -> jax.Array:
    return jnp.where(op_ids == op_id, value, fallback)


def select_op_expanded(
    op_ids: jax.Array, op_id: int, value: jax.Array, fallback: jax.Array
) -> jax.Array:
    return jnp.where(op_ids[..., None] == op_id, value, fallback)


def execute_programs(
    programs: jax.Array,
    vectors: jax.Array,
    *,
    tape_len: int,
    input_len: int,
    max_steps: int,
) -> TapeExecution:
    population, program_length = programs.shape
    cases = vectors.shape[0]
    padded = jnp.zeros((vectors.shape[0], tape_len), dtype=vectors.dtype)
    padded = padded.at[:, :input_len].set(vectors)
    tape = jnp.broadcast_to(padded[None, :, :], (population, cases, tape_len))
    r0 = jnp.zeros((population, cases), dtype=jnp.float32)
    r1 = jnp.zeros((population, cases), dtype=jnp.float32)
    ptr = jnp.zeros((population, cases), dtype=jnp.int32)
    pc = jnp.zeros((population, cases), dtype=jnp.int32)
    predicate = jnp.zeros((population, cases), dtype=jnp.bool_)
    read_steps = jnp.zeros((population, cases), dtype=jnp.float32)
    write_steps = jnp.zeros((population, cases), dtype=jnp.float32)
    move_steps = jnp.zeros((population, cases), dtype=jnp.float32)
    loop_steps = jnp.zeros((population, cases), dtype=jnp.float32)
    accumulate_steps = jnp.zeros((population, cases), dtype=jnp.float32)
    predicate_steps = jnp.zeros((population, cases), dtype=jnp.float32)
    conditional_steps = jnp.zeros((population, cases), dtype=jnp.float32)
    max_ptr_seen = jnp.zeros((population, cases), dtype=jnp.int32)

    program_by_case = jnp.broadcast_to(programs[:, None, :], (population, cases, program_length))

    def body(carry, _):
        (
            r0,
            r1,
            tape,
            ptr,
            pc,
            predicate,
            read_steps,
            write_steps,
            move_steps,
            loop_steps,
            accumulate_steps,
            predicate_steps,
            conditional_steps,
            max_ptr_seen,
        ) = carry
        op_ids = jnp.take_along_axis(program_by_case, pc[..., None], axis=2)[..., 0]
        cell = jnp.take_along_axis(tape, ptr[..., None], axis=2)[..., 0]

        old_r0 = r0
        old_r1 = r1
        old_ptr = ptr
        old_pc = pc
        old_predicate = predicate
        old_tape = tape

        r0 = select_op(op_ids, 1, jnp.zeros_like(r0), r0)
        r1 = select_op(op_ids, 2, jnp.zeros_like(r1), r1)
        r0 = select_op(op_ids, 3, cell, r0)
        r1 = select_op(op_ids, 4, cell, r1)
        read_steps = read_steps + ((op_ids == 3) | (op_ids == 4)).astype(jnp.float32)

        one_hot_ptr = jax.nn.one_hot(old_ptr, tape_len, dtype=tape.dtype)
        tape_r0 = old_tape * (1.0 - one_hot_ptr) + old_r0[..., None] * one_hot_ptr
        tape_r1 = old_tape * (1.0 - one_hot_ptr) + old_r1[..., None] * one_hot_ptr
        tape = select_op_expanded(op_ids, 5, tape_r0, tape)
        tape = select_op_expanded(op_ids, 6, tape_r1, tape)
        write_steps = write_steps + ((op_ids == 5) | (op_ids == 6)).astype(jnp.float32)

        ptr = select_op(op_ids, 7, old_ptr + 1, ptr)
        ptr = select_op(op_ids, 8, old_ptr - 1, ptr)
        ptr = select_op(op_ids, 9, jnp.zeros_like(ptr), ptr)
        ptr = jnp.clip(ptr, 0, tape_len - 1)
        move_steps = move_steps + ((op_ids == 7) | (op_ids == 8) | (op_ids == 9)).astype(jnp.float32)
        max_ptr_seen = jnp.maximum(max_ptr_seen, ptr)

        r0 = select_op(op_ids, 10, old_r0 + old_r1, r0)
        r0 = select_op(op_ids, 11, old_r0 + 1.0, r0)
        r0 = select_op(op_ids, 12, old_r0 - 1.0, r0)
        r0 = select_op(op_ids, 23, old_ptr.astype(jnp.float32), r0)
        r1 = select_op(op_ids, 24, old_ptr.astype(jnp.float32), r1)
        r0 = select_op(op_ids, 32, old_r1, r0)
        r1 = select_op(op_ids, 33, old_r0, r1)
        accumulate_steps = accumulate_steps + (op_ids == 10).astype(jnp.float32)

        predicate = jnp.where(op_ids == 13, old_r0 > 0.0, predicate)
        predicate = jnp.where(op_ids == 14, old_r1 > 0.0, predicate)
        predicate = jnp.where(op_ids == 15, old_ptr < input_len, predicate)
        predicate = jnp.where(op_ids == 20, cell > old_r0, predicate)
        predicate = jnp.where(op_ids == 21, cell > old_r1, predicate)
        predicate = jnp.where(op_ids == 22, old_ptr < input_len - 1, predicate)
        predicate_steps = predicate_steps + (
            (op_ids == 13)
            | (op_ids == 14)
            | (op_ids == 15)
            | (op_ids == 20)
            | (op_ids == 21)
            | (op_ids == 22)
        ).astype(jnp.float32)

        r0 = select_op(op_ids, 16, old_r0 + old_predicate.astype(jnp.float32), r0)
        r0 = select_op(op_ids, 17, jnp.where(old_predicate, old_r1, old_r0), r0)
        r0 = select_op(op_ids, 25, jnp.where(old_predicate, cell, old_r0), r0)
        r1 = select_op(op_ids, 26, jnp.where(old_predicate, cell, old_r1), r1)
        r0 = select_op(op_ids, 27, jnp.where(old_predicate, old_ptr.astype(jnp.float32), old_r0), r0)
        r1 = select_op(op_ids, 28, jnp.where(old_predicate, old_ptr.astype(jnp.float32), old_r1), r1)
        r1 = select_op(op_ids, 34, jnp.where(old_predicate, old_r0, old_r1), r1)
        conditional_steps = conditional_steps + (
            (op_ids == 16)
            | (op_ids == 17)
            | (op_ids == 25)
            | (op_ids == 26)
            | (op_ids == 27)
            | (op_ids == 28)
            | (op_ids == 34)
        ).astype(jnp.float32)

        default_pc = jnp.minimum(old_pc + 1, program_length - 1)
        pc = default_pc
        pc = select_op(op_ids, 18, jnp.zeros_like(pc), pc)
        pc = select_op(op_ids, 19, jnp.where(old_predicate, 0, default_pc), pc)
        pc = select_op(op_ids, 29, jnp.full_like(pc, jnp.minimum(3, program_length - 1)), pc)
        pc = select_op(
            op_ids,
            30,
            jnp.where(old_predicate, jnp.minimum(3, program_length - 1), default_pc),
            pc,
        )
        pc = select_op(op_ids, 31, old_pc, pc)
        pc = jnp.clip(pc, 0, program_length - 1)
        loop_steps = loop_steps + (
            (op_ids == 18) | (op_ids == 19) | (op_ids == 29) | (op_ids == 30)
        ).astype(jnp.float32)

        r0 = jnp.clip(r0, -CLIP_VALUE, CLIP_VALUE)
        r1 = jnp.clip(r1, -CLIP_VALUE, CLIP_VALUE)
        tape = jnp.clip(tape, -CLIP_VALUE, CLIP_VALUE)
        return (
            r0,
            r1,
            tape,
            ptr,
            pc,
            predicate,
            read_steps,
            write_steps,
            move_steps,
            loop_steps,
            accumulate_steps,
            predicate_steps,
            conditional_steps,
            max_ptr_seen,
        ), None

    (
        r0,
        _,
        _,
        _,
        _,
        _,
        read_steps,
        write_steps,
        move_steps,
        loop_steps,
        accumulate_steps,
        predicate_steps,
        conditional_steps,
        max_ptr_seen,
    ), _ = jax.lax.scan(
        body,
        (
            r0,
            r1,
            tape,
            ptr,
            pc,
            predicate,
            read_steps,
            write_steps,
            move_steps,
            loop_steps,
            accumulate_steps,
            predicate_steps,
            conditional_steps,
            max_ptr_seen,
        ),
        None,
        length=max_steps,
    )
    return TapeExecution(
        output=r0,
        read_steps=jnp.mean(read_steps, axis=1),
        write_steps=jnp.mean(write_steps, axis=1),
        move_steps=jnp.mean(move_steps, axis=1),
        loop_steps=jnp.mean(loop_steps, axis=1),
        accumulate_steps=jnp.mean(accumulate_steps, axis=1),
        predicate_steps=jnp.mean(predicate_steps, axis=1),
        conditional_steps=jnp.mean(conditional_steps, axis=1),
        max_ptr=jnp.mean(max_ptr_seen.astype(jnp.float32), axis=1),
    )


@jax.jit(static_argnames=("tape_len", "input_len", "max_steps"))
def evaluate_population(
    programs: jax.Array,
    train_x: jax.Array,
    train_y: jax.Array,
    holdout_x: jax.Array,
    holdout_y: jax.Array,
    length_penalty: float,
    shape_read: float,
    shape_move: float,
    shape_loop: float,
    shape_accumulate: float,
    shape_predicate: float,
    shape_conditional: float,
    shape_write: float,
    shape_coverage: float,
    tape_len: int,
    input_len: int,
    max_steps: int,
) -> EvalMetrics:
    train_exec = execute_programs(
        programs, train_x, tape_len=tape_len, input_len=input_len, max_steps=max_steps
    )
    holdout_exec = execute_programs(
        programs, holdout_x, tape_len=tape_len, input_len=input_len, max_steps=max_steps
    )
    train_pred = train_exec.output
    holdout_pred = holdout_exec.output
    train_error = train_pred - train_y[None, :]
    holdout_error = holdout_pred - holdout_y[None, :]
    train_mse = jnp.mean(train_error * train_error, axis=1)
    holdout_mse = jnp.mean(holdout_error * holdout_error, axis=1)
    max_abs_error = jnp.max(jnp.abs(train_error), axis=1)
    holdout_max_abs_error = jnp.max(jnp.abs(holdout_error), axis=1)
    active_blocks = jnp.sum(programs != 0, axis=1)
    read_score = jnp.clip(train_exec.read_steps / max(input_len, 1), 0.0, 1.0)
    move_score = jnp.clip(train_exec.move_steps / max(input_len, 1), 0.0, 1.0)
    loop_score = jnp.clip(train_exec.loop_steps / max(input_len, 1), 0.0, 1.0)
    accumulate_score = jnp.clip(train_exec.accumulate_steps / max(input_len, 1), 0.0, 1.0)
    predicate_score = jnp.clip(train_exec.predicate_steps / max(input_len, 1), 0.0, 1.0)
    conditional_score = jnp.clip(train_exec.conditional_steps / max(input_len, 1), 0.0, 1.0)
    write_score = jnp.clip(train_exec.write_steps / max(input_len, 1), 0.0, 1.0)
    coverage_score = jnp.clip((train_exec.max_ptr + 1.0) / max(input_len, 1), 0.0, 1.0)
    shaping_bonus = (
        shape_read * read_score
        + shape_move * move_score
        + shape_loop * loop_score
        + shape_accumulate * accumulate_score
        + shape_predicate * predicate_score
        + shape_conditional * conditional_score
        + shape_write * write_score
        + shape_coverage * coverage_score
    )
    reward = -jnp.log1p(train_mse) + shaping_bonus - length_penalty * active_blocks
    return EvalMetrics(
        reward=reward,
        train_mse=train_mse,
        holdout_mse=holdout_mse,
        max_abs_error=max_abs_error,
        holdout_max_abs_error=holdout_max_abs_error,
        active_blocks=active_blocks,
        shaping_bonus=shaping_bonus,
        read_steps=train_exec.read_steps,
        write_steps=train_exec.write_steps,
        move_steps=train_exec.move_steps,
        loop_steps=train_exec.loop_steps,
        accumulate_steps=train_exec.accumulate_steps,
        predicate_steps=train_exec.predicate_steps,
        conditional_steps=train_exec.conditional_steps,
        max_ptr=train_exec.max_ptr,
    )


def init_programs(key: jax.Array, population: int, program_length: int, vocab_size: int) -> jax.Array:
    return jax.random.randint(
        key,
        (population, program_length),
        minval=0,
        maxval=vocab_size,
        dtype=jnp.int32,
    )


def pad_program(program_ids: list[int], program_length: int) -> np.ndarray:
    padded = list(program_ids[:program_length])
    padded.extend([0] * max(program_length - len(padded), 0))
    return np.asarray(padded, dtype=np.int32)


def inject_reference_programs(programs: jax.Array, target: str, program_length: int) -> jax.Array:
    reference = pad_program(REFERENCE_PROGRAMS[target], program_length)
    return programs.at[0].set(jnp.asarray(reference, dtype=jnp.int32))


def mutate_elites(
    key: jax.Array,
    elites: jax.Array,
    *,
    population: int,
    mutation_rate: float,
    random_reset_count: int,
    vocab_size: int,
) -> jax.Array:
    elite_count, program_length = elites.shape
    keys = jax.random.split(key, 6)
    parent_a = jax.random.randint(keys[0], (population,), 0, elite_count)
    parent_b = jax.random.randint(keys[1], (population,), 0, elite_count)
    left = elites[parent_a]
    right = elites[parent_b]
    if program_length > 1:
        cuts = jax.random.randint(keys[2], (population, 1), 1, program_length)
        positions = jnp.arange(program_length)[None, :]
        children = jnp.where(positions < cuts, left, right)
    else:
        children = left
    mutation_mask = jax.random.bernoulli(keys[3], mutation_rate, children.shape)
    new_ops = jax.random.randint(keys[4], children.shape, 0, vocab_size, dtype=jnp.int32)
    mutated = jnp.where(mutation_mask, new_ops, children)
    if random_reset_count > 0:
        reset_count = min(random_reset_count, population - elite_count)
        random_programs = init_programs(keys[5], reset_count, program_length, vocab_size)
        mutated = mutated.at[-reset_count:].set(random_programs)
    return mutated.at[:elite_count].set(elites)


def scalar(x: jax.Array) -> float:
    return float(jax.device_get(x))


def compact_program(program: np.ndarray) -> list[int]:
    return [int(op) for op in program if int(op) != 0]


def render_program(program: np.ndarray) -> list[str]:
    lines = ["# bounded CodePy tape program"]
    for i, op in enumerate(program):
        lines.append(f"{i:02d}: [{int(op):02d}] {TAPE_BLOCKS[int(op)]}")
    return lines


def save_json(path: str, payload: dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, records: list[dict[str, object]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def shaping_args(args: argparse.Namespace) -> tuple[float, float, float, float, float, float, float, float]:
    return (
        args.shape_read,
        args.shape_move,
        args.shape_loop,
        args.shape_accumulate,
        args.shape_predicate,
        args.shape_conditional,
        args.shape_write,
        args.shape_coverage,
    )


def evaluate_single_np(
    program: np.ndarray,
    train_x: jax.Array,
    train_y: jax.Array,
    holdout_x: jax.Array,
    holdout_y: jax.Array,
    args: argparse.Namespace,
) -> dict[str, object]:
    metrics = evaluate_population(
        jnp.asarray(program[None, :], dtype=jnp.int32),
        train_x,
        train_y,
        holdout_x,
        holdout_y,
        args.length_penalty,
        *shaping_args(args),
        args.tape_len,
        args.input_len,
        args.max_steps,
    )
    jax.block_until_ready(metrics.reward)
    return {
        "reward": scalar(metrics.reward[0]),
        "train_mse": scalar(metrics.train_mse[0]),
        "holdout_mse": scalar(metrics.holdout_mse[0]),
        "max_abs_error": scalar(metrics.max_abs_error[0]),
        "holdout_max_abs_error": scalar(metrics.holdout_max_abs_error[0]),
        "active_blocks": int(jax.device_get(metrics.active_blocks[0])),
        "shaping_bonus": scalar(metrics.shaping_bonus[0]),
        "read_steps": scalar(metrics.read_steps[0]),
        "write_steps": scalar(metrics.write_steps[0]),
        "move_steps": scalar(metrics.move_steps[0]),
        "loop_steps": scalar(metrics.loop_steps[0]),
        "accumulate_steps": scalar(metrics.accumulate_steps[0]),
        "predicate_steps": scalar(metrics.predicate_steps[0]),
        "conditional_steps": scalar(metrics.conditional_steps[0]),
        "max_ptr": scalar(metrics.max_ptr[0]),
    }


def record_from_metrics(
    *,
    args: argparse.Namespace,
    generation: int,
    population_index: int,
    program: np.ndarray,
    metrics: EvalMetrics,
    metric_index: int,
    rank: int,
    source: str,
) -> dict[str, object]:
    record = {
        "generation": generation,
        "population_index": population_index,
        "rank": rank,
        "source": source,
        "program_ids": [int(op) for op in program],
        "compact_program_ids": compact_program(program),
        "reward": scalar(metrics.reward[metric_index]),
        "train_mse": scalar(metrics.train_mse[metric_index]),
        "holdout_mse": scalar(metrics.holdout_mse[metric_index]),
        "max_abs_error": scalar(metrics.max_abs_error[metric_index]),
        "holdout_max_abs_error": scalar(metrics.holdout_max_abs_error[metric_index]),
        "active_blocks": int(jax.device_get(metrics.active_blocks[metric_index])),
        "shaping_bonus": scalar(metrics.shaping_bonus[metric_index]),
        "read_steps": scalar(metrics.read_steps[metric_index]),
        "write_steps": scalar(metrics.write_steps[metric_index]),
        "move_steps": scalar(metrics.move_steps[metric_index]),
        "loop_steps": scalar(metrics.loop_steps[metric_index]),
        "accumulate_steps": scalar(metrics.accumulate_steps[metric_index]),
        "predicate_steps": scalar(metrics.predicate_steps[metric_index]),
        "conditional_steps": scalar(metrics.conditional_steps[metric_index]),
        "max_ptr": scalar(metrics.max_ptr[metric_index]),
        "args": {
            "target": args.target,
            "program_length": args.program_length,
            "max_steps": args.max_steps,
            "tape_len": args.tape_len,
            "input_len": args.input_len,
            "cases": args.cases,
            "seed": args.seed,
            "backend": args.backend,
        },
    }
    record["trace_embedding"] = {
        "names": TRACE_FEATURE_NAMES,
        "values": trace_embedding_from_metrics(
            record,
            program_length=args.program_length,
            max_steps=args.max_steps,
            tape_len=args.tape_len,
        ),
    }
    return record


def run(args: argparse.Namespace) -> None:
    if args.input_len > args.tape_len:
        raise SystemExit("--input-len must be <= --tape-len")
    if args.program_length < 1:
        raise SystemExit("--program-length must be >= 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")
    if args.elites < 1 or args.elites > args.population:
        raise SystemExit("--elites must be between 1 and --population")
    if args.trace_top_k < 0 or args.trace_sample_k < 0:
        raise SystemExit("--trace-top-k and --trace-sample-k must be >= 0")

    train_x, train_y, holdout_x, holdout_y = make_dataset(args.target, args.cases, args.input_len)
    vocab_size = len(TAPE_BLOCKS)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    programs = init_programs(init_key, args.population, args.program_length, vocab_size)
    if args.inject_reference:
        programs = inject_reference_programs(programs, args.target, args.program_length)

    print("CodePy bounded tape evolution")
    print(f"backend: {jax.default_backend()}")
    print(f"target: {args.target} ({TARGET_DESCRIPTIONS[args.target]})")
    print(f"population: {args.population:,}")
    print(f"program_length: {args.program_length}")
    print(f"max_steps: {args.max_steps}")
    print(f"tape_len: {args.tape_len}")
    print(f"input_len: {args.input_len}")
    print(f"inject_reference: {args.inject_reference}")

    compile_start = time.perf_counter()
    warmup = evaluate_population(
        programs,
        train_x,
        train_y,
        holdout_x,
        holdout_y,
        args.length_penalty,
        *shaping_args(args),
        args.tape_len,
        args.input_len,
        args.max_steps,
    )
    jax.block_until_ready(warmup.reward)
    compile_s = time.perf_counter() - compile_start

    best_record: dict[str, object] | None = None
    best_program: np.ndarray | None = None
    first_solution: dict[str, object] | None = None
    generations_run = 0
    trace_records_written = 0
    random_reset_count = int(args.population * args.random_reset_fraction)
    if args.trace_jsonl:
        trace_dir = os.path.dirname(args.trace_jsonl)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        open(args.trace_jsonl, "w", encoding="utf-8").close()
    start = time.perf_counter()
    for generation in range(args.generations + 1):
        generations_run = generation + 1
        key, mutate_key = jax.random.split(key)
        metrics = evaluate_population(
            programs,
            train_x,
            train_y,
            holdout_x,
            holdout_y,
            args.length_penalty,
            *shaping_args(args),
            args.tape_len,
            args.input_len,
            args.max_steps,
        )
        jax.block_until_ready(metrics.reward)
        elite_indices = jnp.argsort(metrics.reward)[-args.elites :][::-1]
        elites = programs[elite_indices]
        best_idx = int(jax.device_get(elite_indices[0]))
        if args.trace_jsonl:
            trace_indices: list[tuple[int, int, str]] = []
            if args.trace_all:
                trace_indices.extend((idx, idx, "all") for idx in range(args.population))
                top_count = args.population
            else:
                top_count = min(args.trace_top_k, args.population)
                if top_count > 0:
                    top_indices = [int(i) for i in jax.device_get(elite_indices[:top_count])]
                    trace_indices.extend((idx, rank, "top") for rank, idx in enumerate(top_indices))
            sample_count = 0 if args.trace_all else min(args.trace_sample_k, args.population)
            if sample_count > 0:
                sample_key = jax.random.fold_in(mutate_key, generation + 10_000)
                sample_indices = [
                    int(i)
                    for i in jax.device_get(
                        jax.random.choice(
                            sample_key,
                            args.population,
                            shape=(sample_count,),
                            replace=False,
                        )
                    )
                ]
                trace_indices.extend((idx, top_count + rank, "sample") for rank, idx in enumerate(sample_indices))
            if trace_indices:
                host_programs = np.asarray(jax.device_get(programs), dtype=np.int32)
                seen_indices: set[int] = set()
                trace_records: list[dict[str, object]] = []
                for idx, rank, source in trace_indices:
                    if idx in seen_indices:
                        continue
                    seen_indices.add(idx)
                    trace_records.append(
                        record_from_metrics(
                            args=args,
                            generation=generation,
                            population_index=idx,
                            program=host_programs[idx],
                            metrics=metrics,
                            metric_index=idx,
                            rank=rank,
                            source=source,
                        )
                    )
                append_jsonl(args.trace_jsonl, trace_records)
                trace_records_written += len(trace_records)
        current = {
            "generation": generation,
            "reward": scalar(metrics.reward[best_idx]),
            "mean_reward": scalar(jnp.mean(metrics.reward)),
            "train_mse": scalar(metrics.train_mse[best_idx]),
            "holdout_mse": scalar(metrics.holdout_mse[best_idx]),
            "max_abs_error": scalar(metrics.max_abs_error[best_idx]),
            "holdout_max_abs_error": scalar(metrics.holdout_max_abs_error[best_idx]),
            "active_blocks": int(jax.device_get(metrics.active_blocks[best_idx])),
            "shaping_bonus": scalar(metrics.shaping_bonus[best_idx]),
            "read_steps": scalar(metrics.read_steps[best_idx]),
            "write_steps": scalar(metrics.write_steps[best_idx]),
            "move_steps": scalar(metrics.move_steps[best_idx]),
            "loop_steps": scalar(metrics.loop_steps[best_idx]),
            "accumulate_steps": scalar(metrics.accumulate_steps[best_idx]),
            "predicate_steps": scalar(metrics.predicate_steps[best_idx]),
            "conditional_steps": scalar(metrics.conditional_steps[best_idx]),
            "max_ptr": scalar(metrics.max_ptr[best_idx]),
        }
        if best_record is None or float(current["reward"]) > float(best_record["reward"]):
            best_record = current
            best_program = np.asarray(jax.device_get(programs[best_idx]), dtype=np.int32)
        exact = (
            float(current["train_mse"]) <= args.stop_mse
            and float(current["holdout_mse"]) <= args.stop_mse
        )
        if exact and first_solution is None:
            first_solution = {
                **current,
                "program_ids": [int(op) for op in jax.device_get(programs[best_idx])],
            }
        if generation == 0 or generation == args.generations or generation % args.log_every == 0:
            print(
                f"gen={generation:03d} reward={current['reward']:.6f} "
                f"train_mse={current['train_mse']:.6g} "
                f"holdout_mse={current['holdout_mse']:.6g} "
                f"active={current['active_blocks']} "
                f"shape={current['shaping_bonus']:.4f}"
            )
        if exact:
            print(f"early_stop: train/holdout MSE <= {args.stop_mse:g}")
            break
        if generation < args.generations:
            programs = mutate_elites(
                mutate_key,
                elites,
                population=args.population,
                mutation_rate=args.mutation_rate,
                random_reset_count=random_reset_count,
                vocab_size=vocab_size,
            )

    evolution_s = time.perf_counter() - start
    assert best_record is not None
    assert best_program is not None
    reference_program = pad_program(REFERENCE_PROGRAMS[args.target], args.program_length)
    reference_metrics = evaluate_single_np(
        reference_program, train_x, train_y, holdout_x, holdout_y, args
    )
    best_metrics = evaluate_single_np(best_program, train_x, train_y, holdout_x, holdout_y, args)
    payload = {
        "format_version": 1,
        "prototype": "CodePy bounded tape evolution",
        "guardrail": "Constrained tape DSL with bounded jax.lax.scan loop; no arbitrary Python source is executed.",
        "target": {"name": args.target, "description": TARGET_DESCRIPTIONS[args.target]},
        "state_geometry": {
            "registers": ["r0", "r1"],
            "tape_len": args.tape_len,
            "input_len": args.input_len,
            "pointer": "ptr",
            "program_counter": "pc",
            "max_steps": args.max_steps,
        },
        "block_vocabulary": [
            {"id": i, "block": block}
            for i, block in enumerate(TAPE_BLOCKS)
        ],
        "best": {
            **best_record,
            **best_metrics,
            "program_ids": [int(op) for op in best_program],
            "compact_program_ids": compact_program(best_program),
            "compact_program_blocks": [TAPE_BLOCKS[op] for op in compact_program(best_program)],
            "rendered_program": render_program(best_program),
        },
        "first_solution": first_solution,
        "reference_program": {
            "program_ids": [int(op) for op in reference_program],
            "compact_program_ids": compact_program(reference_program),
            "compact_program_blocks": [TAPE_BLOCKS[op] for op in compact_program(reference_program)],
            "metrics": reference_metrics,
            "rendered_program": render_program(reference_program),
        },
        "run": {
            "backend": jax.default_backend(),
            "compile_s": compile_s,
            "evolution_s": evolution_s,
            "generations_run": generations_run,
            "evaluated_programs": generations_run * args.population,
            "trace_records_written": trace_records_written,
            "trace_mode": "all" if args.trace_all else "top_sample",
        },
        "reward_shaping": {
            "shape_read": args.shape_read,
            "shape_move": args.shape_move,
            "shape_loop": args.shape_loop,
            "shape_accumulate": args.shape_accumulate,
            "shape_predicate": args.shape_predicate,
            "shape_conditional": args.shape_conditional,
            "shape_write": args.shape_write,
            "shape_coverage": args.shape_coverage,
            "note": "Trace-based shaping rewards machine behavior only; it does not inject target programs.",
        },
        "args": vars(args),
    }
    if args.output_json:
        save_json(args.output_json, payload)

    print("")
    print("Run")
    print(f"  compile_s: {compile_s:.4f}")
    print(f"  evolution_s: {evolution_s:.4f}")
    print(f"  best_generation: {best_record['generation']}")
    print(f"  best_train_mse: {best_metrics['train_mse']:.8g}")
    print(f"  best_holdout_mse: {best_metrics['holdout_mse']:.8g}")
    print(f"  best_shaping_bonus: {best_metrics['shaping_bonus']:.6g}")
    print(f"  best_read_steps: {best_metrics['read_steps']:.6g}")
    print(f"  best_move_steps: {best_metrics['move_steps']:.6g}")
    print(f"  best_loop_steps: {best_metrics['loop_steps']:.6g}")
    print(f"  best_accumulate_steps: {best_metrics['accumulate_steps']:.6g}")
    print(f"  best_max_ptr: {best_metrics['max_ptr']:.6g}")
    print(f"  reference_train_mse: {reference_metrics['train_mse']:.8g}")
    print(f"  reference_holdout_mse: {reference_metrics['holdout_mse']:.8g}")
    if args.output_json:
        print(f"  output_json: {args.output_json}")
    if args.trace_jsonl:
        print(f"  trace_jsonl: {args.trace_jsonl}")
        print(f"  trace_records_written: {trace_records_written}")
    print("  best_program:")
    for line in render_program(best_program):
        print(f"    {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve bounded-loop CodePy tape programs.")
    parser.add_argument("--target", choices=tuple(TARGET_DESCRIPTIONS), default="sum4")
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--elites", type=int, default=128)
    parser.add_argument("--program-length", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--tape-len", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=4)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--mutation-rate", type=float, default=0.16)
    parser.add_argument("--random-reset-fraction", type=float, default=0.04)
    parser.add_argument("--length-penalty", type=float, default=1.0e-6)
    parser.add_argument("--shape-read", type=float, default=0.0)
    parser.add_argument("--shape-move", type=float, default=0.0)
    parser.add_argument("--shape-loop", type=float, default=0.0)
    parser.add_argument("--shape-accumulate", type=float, default=0.0)
    parser.add_argument("--shape-predicate", type=float, default=0.0)
    parser.add_argument("--shape-conditional", type=float, default=0.0)
    parser.add_argument("--shape-write", type=float, default=0.0)
    parser.add_argument("--shape-coverage", type=float, default=0.0)
    parser.add_argument("--stop-mse", type=float, default=1.0e-8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--inject-reference", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--trace-jsonl", default=None)
    parser.add_argument("--trace-all", action="store_true")
    parser.add_argument("--trace-top-k", type=int, default=0)
    parser.add_argument("--trace-sample-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
