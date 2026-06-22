#!/usr/bin/env python3
"""Evolve programs made from cached CodePy blocks.

This prototype tests the "programmer calculator" idea directly: a candidate is
not generated character by character. It is a fixed-length array of small
integer opcodes. Each opcode selects one reusable CodePy block, and a JAX loop
executes many candidate programs over a whole test batch at once.

The DSL is deliberately tiny and safe. It manipulates two scalar registers
(`r0`, `r1`) and returns `r0`; it never executes arbitrary Python source.
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


CLIP_VALUE = 1.0e6

CODEPY_BLOCKS = (
    "noop",
    "r0 = x",
    "r1 = x",
    "r0 = 0",
    "r1 = 0",
    "r0 = 1",
    "r1 = 1",
    "r0 = -1",
    "r1 = -1",
    "r0 += 1",
    "r1 += 1",
    "r0 -= 1",
    "r1 -= 1",
    "r0 += x",
    "r1 += x",
    "r0 *= x",
    "r1 *= x",
    "r0 += r1",
    "r0 -= r1",
    "r0 *= r1",
    "r0 = abs(r0)",
    "r1 = abs(r1)",
    "r0 = r1",
    "r1 = r0",
    "r0 = y",
    "r1 = y",
    "r0 += y",
    "r1 += y",
    "r0 *= y",
    "r1 *= y",
    "p = r0 > 0",
    "p = r0 <= 0",
    "p = r1 > 0",
    "p = r1 <= 0",
    "p = r1 > r0",
    "p = r0 > r1",
    "p = r0 == 0",
    "p = r1 == 0",
    "r0 = where(p, r1, r0)",
    "r1 = where(p, r0, r1)",
    "r0 = where(p, 0, r0)",
    "r1 = where(p, 0, r1)",
    "r0 = where(p, -r0, r0)",
    "r1 = where(p, -r1, r1)",
    "r0 = where(p, 1, r0)",
    "r0 = where(p, -1, r0)",
    "r1 = where(p, 1, r1)",
    "r1 = where(p, -1, r1)",
    "r0 = square(r0)",
    "r1 = square(r1)",
    "r0 = relu(r0)",
    "r1 = relu(r1)",
    "r0 = max(r0, r1)",
    "r1 = max(r1, r0)",
    "r0 = min(r0, r1)",
    "r1 = min(r1, r0)",
    "r0 = sign(r0)",
    "r1 = sign(r1)",
    "r0 = clamp01(r0)",
    "r1 = clamp01(r1)",
    "r0 = max(abs(r0), r1)",
    "r1 = max(abs(r1), r0)",
    "r0 = clamp01(abs(r0))",
    "r1 = clamp01(abs(r1))",
)

BASE_BLOCK_IDS = tuple(i for i in range(48) if i not in (20, 21))
SQUARE_MACRO_BLOCK_IDS = (48, 49)
ABS_MACRO_BLOCK_IDS = (20, 21)
MAX_MACRO_BLOCK_IDS = (52, 53)
PHASE_A_MACRO_BLOCK_IDS = (20, 21, 48, 49, 50, 51, 52, 53, 56, 57)
CLAMP_MACRO_BLOCK_IDS = (58, 59)
MAX_ABS_MACRO_BLOCK_IDS = (60, 61)
CLAMP_ABS_MACRO_BLOCK_IDS = (62, 63)
LEARNED_BLOCK_IDS = BASE_BLOCK_IDS + PHASE_A_MACRO_BLOCK_IDS
STAGE4_AFTER_SQUARE_BLOCK_IDS = BASE_BLOCK_IDS + SQUARE_MACRO_BLOCK_IDS
STAGE4_AFTER_ABS_BLOCK_IDS = STAGE4_AFTER_SQUARE_BLOCK_IDS + ABS_MACRO_BLOCK_IDS
STAGE4_AFTER_MAX_BLOCK_IDS = STAGE4_AFTER_ABS_BLOCK_IDS + MAX_MACRO_BLOCK_IDS
STAGE4_AFTER_CLAMP_BLOCK_IDS = STAGE4_AFTER_MAX_BLOCK_IDS + CLAMP_MACRO_BLOCK_IDS
STAGE4_AFTER_MAX_ABS_BLOCK_IDS = STAGE4_AFTER_CLAMP_BLOCK_IDS + MAX_ABS_MACRO_BLOCK_IDS
STAGE4_AFTER_CLAMP_ABS_BLOCK_IDS = STAGE4_AFTER_MAX_ABS_BLOCK_IDS + CLAMP_ABS_MACRO_BLOCK_IDS

BLOCK_PROFILES = {
    "base": BASE_BLOCK_IDS,
    "learned": LEARNED_BLOCK_IDS,
    "stage4_after_square": STAGE4_AFTER_SQUARE_BLOCK_IDS,
    "stage4_after_abs": STAGE4_AFTER_ABS_BLOCK_IDS,
    "stage4_after_max": STAGE4_AFTER_MAX_BLOCK_IDS,
    "stage4_after_clamp": STAGE4_AFTER_CLAMP_BLOCK_IDS,
    "stage4_after_max_abs": STAGE4_AFTER_MAX_ABS_BLOCK_IDS,
    "stage4_after_clamp_abs": STAGE4_AFTER_CLAMP_ABS_BLOCK_IDS,
}

TARGET_DESCRIPTIONS = {
    "quadratic": "y = x*x + 2*x + 1",
    "affine": "y = 3*x - 2",
    "square": "y = x*x",
    "square_minus_one": "y = x*x - 1",
    "cubic_minus_x": "y = x*x*x - x",
    "abs": "y = abs(x)",
    "relu": "y = max(x, 0)",
    "sign": "y = sign(x)",
    "max_xy": "z = max(x, y)",
    "piecewise_square_neg": "y = x*x if x > 0 else -x",
    "clamp01": "y = min(max(x, 0), 1)",
    "max_abs_x_y": "z = max(abs(x), y)",
    "clamp_abs_01": "y = min(max(abs(x), 0), 1)",
    "piecewise_max_abs_x_y": "z = max(abs(x), y) if x > 0 else -max(abs(x), y)",
}

TWO_INPUT_TARGETS = {"max_xy", "max_abs_x_y", "piecewise_max_abs_x_y"}


class EvalMetrics(NamedTuple):
    reward: jax.Array  # [population]
    train_mse: jax.Array  # [population]
    holdout_mse: jax.Array  # [population]
    max_abs_error: jax.Array  # [population]
    holdout_max_abs_error: jax.Array  # [population]
    active_blocks: jax.Array  # [population]


def target_values(xs: jax.Array, ys: jax.Array, target: str) -> jax.Array:
    if target == "quadratic":
        return xs * xs + 2.0 * xs + 1.0
    if target == "affine":
        return 3.0 * xs - 2.0
    if target == "square":
        return xs * xs
    if target == "square_minus_one":
        return xs * xs - 1.0
    if target == "cubic_minus_x":
        return xs * xs * xs - xs
    if target == "abs":
        return jnp.abs(xs)
    if target == "relu":
        return jnp.maximum(xs, 0.0)
    if target == "sign":
        return jnp.sign(xs)
    if target == "max_xy":
        return jnp.maximum(xs, ys)
    if target == "piecewise_square_neg":
        return jnp.where(xs > 0.0, xs * xs, -xs)
    if target == "clamp01":
        return jnp.clip(xs, 0.0, 1.0)
    if target == "max_abs_x_y":
        return jnp.maximum(jnp.abs(xs), ys)
    if target == "clamp_abs_01":
        return jnp.clip(jnp.abs(xs), 0.0, 1.0)
    if target == "piecewise_max_abs_x_y":
        base = jnp.maximum(jnp.abs(xs), ys)
        return jnp.where(xs > 0.0, base, -base)
    raise ValueError(f"Unknown target: {target}")


def make_dataset(
    target: str, cases: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if target in TWO_INPUT_TARGETS:
        side = int(np.ceil(np.sqrt(cases)))
        grid_x = jnp.linspace(-2.0, 2.0, side, dtype=jnp.float32)
        grid_y = jnp.linspace(-1.65, 2.35, side, dtype=jnp.float32)
        mesh_x, mesh_y = jnp.meshgrid(grid_x, grid_y, indexing="ij")
        train_x = mesh_x.reshape(-1)[:cases]
        train_y_input = mesh_y.reshape(-1)[:cases]

        hold_x = jnp.linspace(-1.93, 2.07, side, dtype=jnp.float32)
        hold_y = jnp.linspace(-1.41, 2.59, side, dtype=jnp.float32)
        hold_mesh_x, hold_mesh_y = jnp.meshgrid(hold_x, hold_y, indexing="ij")
        holdout_x = hold_mesh_x.reshape(-1)[:cases]
        holdout_y_input = hold_mesh_y.reshape(-1)[:cases]
    else:
        train_x = jnp.linspace(-2.0, 2.0, cases, dtype=jnp.float32)
        holdout_x = jnp.linspace(-1.93, 2.07, cases, dtype=jnp.float32)
        if target in {"relu", "sign", "piecewise_square_neg", "clamp01", "clamp_abs_01"} and cases >= 3:
            midpoint = cases // 2
            train_x = train_x.at[midpoint].set(0.0)
            holdout_x = holdout_x.at[midpoint].set(0.0)
        if target in {"clamp01", "clamp_abs_01"} and cases >= 5:
            train_x = train_x.at[cases // 2 + 1].set(1.0)
            holdout_x = holdout_x.at[cases // 2 + 1].set(1.0)
        train_y_input = jnp.zeros_like(train_x)
        holdout_y_input = jnp.zeros_like(holdout_x)

    return (
        train_x,
        train_y_input,
        target_values(train_x, train_y_input, target),
        holdout_x,
        holdout_y_input,
        target_values(holdout_x, holdout_y_input, target),
    )


def pack_registers(r0: jax.Array, r1: jax.Array) -> jax.Array:
    r0 = jnp.clip(r0, -CLIP_VALUE, CLIP_VALUE)
    r1 = jnp.clip(r1, -CLIP_VALUE, CLIP_VALUE)
    return jnp.stack((r0, r1), axis=1)


def apply_block(
    op_id: jax.Array, registers: jax.Array, predicate: jax.Array, xs: jax.Array, ys: jax.Array
) -> tuple[jax.Array, jax.Array]:
    def noop(operand):
        registers, predicate, _, _ = operand
        return registers, predicate

    def r0_x(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(xs, registers[:, 1]), predicate

    def r1_x(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(registers[:, 0], xs), predicate

    def r0_zero(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(jnp.zeros_like(xs), registers[:, 1]), predicate

    def r1_zero(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(registers[:, 0], jnp.zeros_like(xs)), predicate

    def r0_one(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(jnp.ones_like(xs), registers[:, 1]), predicate

    def r1_one(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(registers[:, 0], jnp.ones_like(xs)), predicate

    def r0_neg_one(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(-jnp.ones_like(xs), registers[:, 1]), predicate

    def r1_neg_one(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(registers[:, 0], -jnp.ones_like(xs)), predicate

    def r0_plus_one(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0] + 1.0, registers[:, 1]), predicate

    def r1_plus_one(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], registers[:, 1] + 1.0), predicate

    def r0_minus_one(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0] - 1.0, registers[:, 1]), predicate

    def r1_minus_one(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], registers[:, 1] - 1.0), predicate

    def r0_plus_x(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(registers[:, 0] + xs, registers[:, 1]), predicate

    def r1_plus_x(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(registers[:, 0], registers[:, 1] + xs), predicate

    def r0_times_x(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(registers[:, 0] * xs, registers[:, 1]), predicate

    def r1_times_x(operand):
        registers, predicate, xs, _ = operand
        return pack_registers(registers[:, 0], registers[:, 1] * xs), predicate

    def r0_plus_r1(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0] + registers[:, 1], registers[:, 1]), predicate

    def r0_minus_r1(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0] - registers[:, 1], registers[:, 1]), predicate

    def r0_times_r1(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0] * registers[:, 1], registers[:, 1]), predicate

    def r0_abs(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.abs(registers[:, 0]), registers[:, 1]), predicate

    def r1_abs(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.abs(registers[:, 1])), predicate

    def r0_from_r1(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 1], registers[:, 1]), predicate

    def r1_from_r0(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], registers[:, 0]), predicate

    def r0_y(operand):
        registers, predicate, _, ys = operand
        return pack_registers(ys, registers[:, 1]), predicate

    def r1_y(operand):
        registers, predicate, _, ys = operand
        return pack_registers(registers[:, 0], ys), predicate

    def r0_plus_y(operand):
        registers, predicate, _, ys = operand
        return pack_registers(registers[:, 0] + ys, registers[:, 1]), predicate

    def r1_plus_y(operand):
        registers, predicate, _, ys = operand
        return pack_registers(registers[:, 0], registers[:, 1] + ys), predicate

    def r0_times_y(operand):
        registers, predicate, _, ys = operand
        return pack_registers(registers[:, 0] * ys, registers[:, 1]), predicate

    def r1_times_y(operand):
        registers, predicate, _, ys = operand
        return pack_registers(registers[:, 0], registers[:, 1] * ys), predicate

    def pred_r0_positive(operand):
        registers, _, _, _ = operand
        return registers, registers[:, 0] > 0.0

    def pred_r0_nonpositive(operand):
        registers, _, _, _ = operand
        return registers, registers[:, 0] <= 0.0

    def pred_r1_positive(operand):
        registers, _, _, _ = operand
        return registers, registers[:, 1] > 0.0

    def pred_r1_nonpositive(operand):
        registers, _, _, _ = operand
        return registers, registers[:, 1] <= 0.0

    def pred_r1_gt_r0(operand):
        registers, _, _, _ = operand
        return registers, registers[:, 1] > registers[:, 0]

    def pred_r0_gt_r1(operand):
        registers, _, _, _ = operand
        return registers, registers[:, 0] > registers[:, 1]

    def pred_r0_zero(operand):
        registers, _, _, _ = operand
        return registers, registers[:, 0] == 0.0

    def pred_r1_zero(operand):
        registers, _, _, _ = operand
        return registers, registers[:, 1] == 0.0

    def r0_where_r1(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.where(predicate, registers[:, 1], registers[:, 0]), registers[:, 1]), predicate

    def r1_where_r0(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.where(predicate, registers[:, 0], registers[:, 1])), predicate

    def r0_where_zero(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.where(predicate, 0.0, registers[:, 0]), registers[:, 1]), predicate

    def r1_where_zero(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.where(predicate, 0.0, registers[:, 1])), predicate

    def r0_where_neg(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.where(predicate, -registers[:, 0], registers[:, 0]), registers[:, 1]), predicate

    def r1_where_neg(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.where(predicate, -registers[:, 1], registers[:, 1])), predicate

    def r0_where_one(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.where(predicate, 1.0, registers[:, 0]), registers[:, 1]), predicate

    def r0_where_neg_one(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.where(predicate, -1.0, registers[:, 0]), registers[:, 1]), predicate

    def r1_where_one(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.where(predicate, 1.0, registers[:, 1])), predicate

    def r1_where_neg_one(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.where(predicate, -1.0, registers[:, 1])), predicate

    def r0_square(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0] * registers[:, 0], registers[:, 1]), predicate

    def r1_square(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], registers[:, 1] * registers[:, 1]), predicate

    def r0_relu(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.maximum(registers[:, 0], 0.0), registers[:, 1]), predicate

    def r1_relu(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.maximum(registers[:, 1], 0.0)), predicate

    def r0_max_r1(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.maximum(registers[:, 0], registers[:, 1]), registers[:, 1]), predicate

    def r1_max_r0(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.maximum(registers[:, 1], registers[:, 0])), predicate

    def r0_sign(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.sign(registers[:, 0]), registers[:, 1]), predicate

    def r1_sign(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.sign(registers[:, 1])), predicate

    def r0_min_r1(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.minimum(registers[:, 0], registers[:, 1]), registers[:, 1]), predicate

    def r1_min_r0(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.minimum(registers[:, 1], registers[:, 0])), predicate

    def r0_clamp01(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.clip(registers[:, 0], 0.0, 1.0), registers[:, 1]), predicate

    def r1_clamp01(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.clip(registers[:, 1], 0.0, 1.0)), predicate

    def r0_max_abs_r1(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.maximum(jnp.abs(registers[:, 0]), registers[:, 1]), registers[:, 1]), predicate

    def r1_max_abs_r0(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.maximum(jnp.abs(registers[:, 1]), registers[:, 0])), predicate

    def r0_clamp_abs01(operand):
        registers, predicate, _, _ = operand
        return pack_registers(jnp.clip(jnp.abs(registers[:, 0]), 0.0, 1.0), registers[:, 1]), predicate

    def r1_clamp_abs01(operand):
        registers, predicate, _, _ = operand
        return pack_registers(registers[:, 0], jnp.clip(jnp.abs(registers[:, 1]), 0.0, 1.0)), predicate

    return jax.lax.switch(
        op_id,
        (
            noop,
            r0_x,
            r1_x,
            r0_zero,
            r1_zero,
            r0_one,
            r1_one,
            r0_neg_one,
            r1_neg_one,
            r0_plus_one,
            r1_plus_one,
            r0_minus_one,
            r1_minus_one,
            r0_plus_x,
            r1_plus_x,
            r0_times_x,
            r1_times_x,
            r0_plus_r1,
            r0_minus_r1,
            r0_times_r1,
            r0_abs,
            r1_abs,
            r0_from_r1,
            r1_from_r0,
            r0_y,
            r1_y,
            r0_plus_y,
            r1_plus_y,
            r0_times_y,
            r1_times_y,
            pred_r0_positive,
            pred_r0_nonpositive,
            pred_r1_positive,
            pred_r1_nonpositive,
            pred_r1_gt_r0,
            pred_r0_gt_r1,
            pred_r0_zero,
            pred_r1_zero,
            r0_where_r1,
            r1_where_r0,
            r0_where_zero,
            r1_where_zero,
            r0_where_neg,
            r1_where_neg,
            r0_where_one,
            r0_where_neg_one,
            r1_where_one,
            r1_where_neg_one,
            r0_square,
            r1_square,
            r0_relu,
            r1_relu,
            r0_max_r1,
            r1_max_r0,
            r0_min_r1,
            r1_min_r0,
            r0_sign,
            r1_sign,
            r0_clamp01,
            r1_clamp01,
            r0_max_abs_r1,
            r1_max_abs_r0,
            r0_clamp_abs01,
            r1_clamp_abs01,
        ),
        (registers, predicate, xs, ys),
    )


def execute_program(program: jax.Array, xs: jax.Array, ys: jax.Array) -> jax.Array:
    registers = pack_registers(xs, jnp.zeros_like(xs))
    predicate = jnp.zeros_like(xs, dtype=jnp.bool_)

    def body(carry: tuple[jax.Array, jax.Array], op_id: jax.Array):
        registers, predicate = carry
        return apply_block(op_id, registers, predicate, xs, ys), None

    (registers, _), _ = jax.lax.scan(body, (registers, predicate), program)
    return registers[:, 0]


def select_op(op_ids: jax.Array, op_id: int, value: jax.Array, fallback: jax.Array) -> jax.Array:
    return jnp.where(op_ids[:, None] == op_id, value, fallback)


def execute_programs(programs: jax.Array, xs: jax.Array, ys: jax.Array) -> jax.Array:
    """Execute all candidate programs as SoA register rails [population, cases]."""
    xs_by_program = jnp.broadcast_to(xs[None, :], (programs.shape[0], xs.shape[0]))
    ys_by_program = jnp.broadcast_to(ys[None, :], (programs.shape[0], ys.shape[0]))
    r0 = xs_by_program
    r1 = jnp.zeros_like(xs_by_program)
    predicate = jnp.zeros_like(xs_by_program, dtype=jnp.bool_)

    def body(carry: tuple[jax.Array, jax.Array, jax.Array], op_ids: jax.Array):
        old_r0, old_r1, old_predicate = carry
        new_r0 = old_r0
        new_r1 = old_r1
        new_predicate = old_predicate

        zeros = jnp.zeros_like(old_r0)
        ones = jnp.ones_like(old_r0)
        neg_ones = -ones

        new_r0 = select_op(op_ids, 1, xs_by_program, new_r0)
        new_r1 = select_op(op_ids, 2, xs_by_program, new_r1)
        new_r0 = select_op(op_ids, 3, zeros, new_r0)
        new_r1 = select_op(op_ids, 4, zeros, new_r1)
        new_r0 = select_op(op_ids, 5, ones, new_r0)
        new_r1 = select_op(op_ids, 6, ones, new_r1)
        new_r0 = select_op(op_ids, 7, neg_ones, new_r0)
        new_r1 = select_op(op_ids, 8, neg_ones, new_r1)
        new_r0 = select_op(op_ids, 9, old_r0 + 1.0, new_r0)
        new_r1 = select_op(op_ids, 10, old_r1 + 1.0, new_r1)
        new_r0 = select_op(op_ids, 11, old_r0 - 1.0, new_r0)
        new_r1 = select_op(op_ids, 12, old_r1 - 1.0, new_r1)
        new_r0 = select_op(op_ids, 13, old_r0 + xs_by_program, new_r0)
        new_r1 = select_op(op_ids, 14, old_r1 + xs_by_program, new_r1)
        new_r0 = select_op(op_ids, 15, old_r0 * xs_by_program, new_r0)
        new_r1 = select_op(op_ids, 16, old_r1 * xs_by_program, new_r1)
        new_r0 = select_op(op_ids, 17, old_r0 + old_r1, new_r0)
        new_r0 = select_op(op_ids, 18, old_r0 - old_r1, new_r0)
        new_r0 = select_op(op_ids, 19, old_r0 * old_r1, new_r0)
        new_r0 = select_op(op_ids, 20, jnp.abs(old_r0), new_r0)
        new_r1 = select_op(op_ids, 21, jnp.abs(old_r1), new_r1)
        new_r0 = select_op(op_ids, 22, old_r1, new_r0)
        new_r1 = select_op(op_ids, 23, old_r0, new_r1)
        new_r0 = select_op(op_ids, 24, ys_by_program, new_r0)
        new_r1 = select_op(op_ids, 25, ys_by_program, new_r1)
        new_r0 = select_op(op_ids, 26, old_r0 + ys_by_program, new_r0)
        new_r1 = select_op(op_ids, 27, old_r1 + ys_by_program, new_r1)
        new_r0 = select_op(op_ids, 28, old_r0 * ys_by_program, new_r0)
        new_r1 = select_op(op_ids, 29, old_r1 * ys_by_program, new_r1)

        new_predicate = jnp.where(op_ids[:, None] == 30, old_r0 > 0.0, new_predicate)
        new_predicate = jnp.where(op_ids[:, None] == 31, old_r0 <= 0.0, new_predicate)
        new_predicate = jnp.where(op_ids[:, None] == 32, old_r1 > 0.0, new_predicate)
        new_predicate = jnp.where(op_ids[:, None] == 33, old_r1 <= 0.0, new_predicate)
        new_predicate = jnp.where(op_ids[:, None] == 34, old_r1 > old_r0, new_predicate)
        new_predicate = jnp.where(op_ids[:, None] == 35, old_r0 > old_r1, new_predicate)
        new_predicate = jnp.where(op_ids[:, None] == 36, old_r0 == 0.0, new_predicate)
        new_predicate = jnp.where(op_ids[:, None] == 37, old_r1 == 0.0, new_predicate)

        new_r0 = select_op(op_ids, 38, jnp.where(old_predicate, old_r1, old_r0), new_r0)
        new_r1 = select_op(op_ids, 39, jnp.where(old_predicate, old_r0, old_r1), new_r1)
        new_r0 = select_op(op_ids, 40, jnp.where(old_predicate, 0.0, old_r0), new_r0)
        new_r1 = select_op(op_ids, 41, jnp.where(old_predicate, 0.0, old_r1), new_r1)
        new_r0 = select_op(op_ids, 42, jnp.where(old_predicate, -old_r0, old_r0), new_r0)
        new_r1 = select_op(op_ids, 43, jnp.where(old_predicate, -old_r1, old_r1), new_r1)
        new_r0 = select_op(op_ids, 44, jnp.where(old_predicate, 1.0, old_r0), new_r0)
        new_r0 = select_op(op_ids, 45, jnp.where(old_predicate, -1.0, old_r0), new_r0)
        new_r1 = select_op(op_ids, 46, jnp.where(old_predicate, 1.0, old_r1), new_r1)
        new_r1 = select_op(op_ids, 47, jnp.where(old_predicate, -1.0, old_r1), new_r1)
        new_r0 = select_op(op_ids, 48, old_r0 * old_r0, new_r0)
        new_r1 = select_op(op_ids, 49, old_r1 * old_r1, new_r1)
        new_r0 = select_op(op_ids, 50, jnp.maximum(old_r0, 0.0), new_r0)
        new_r1 = select_op(op_ids, 51, jnp.maximum(old_r1, 0.0), new_r1)
        new_r0 = select_op(op_ids, 52, jnp.maximum(old_r0, old_r1), new_r0)
        new_r1 = select_op(op_ids, 53, jnp.maximum(old_r1, old_r0), new_r1)
        new_r0 = select_op(op_ids, 54, jnp.minimum(old_r0, old_r1), new_r0)
        new_r1 = select_op(op_ids, 55, jnp.minimum(old_r1, old_r0), new_r1)
        new_r0 = select_op(op_ids, 56, jnp.sign(old_r0), new_r0)
        new_r1 = select_op(op_ids, 57, jnp.sign(old_r1), new_r1)
        new_r0 = select_op(op_ids, 58, jnp.clip(old_r0, 0.0, 1.0), new_r0)
        new_r1 = select_op(op_ids, 59, jnp.clip(old_r1, 0.0, 1.0), new_r1)
        new_r0 = select_op(op_ids, 60, jnp.maximum(jnp.abs(old_r0), old_r1), new_r0)
        new_r1 = select_op(op_ids, 61, jnp.maximum(jnp.abs(old_r1), old_r0), new_r1)
        new_r0 = select_op(op_ids, 62, jnp.clip(jnp.abs(old_r0), 0.0, 1.0), new_r0)
        new_r1 = select_op(op_ids, 63, jnp.clip(jnp.abs(old_r1), 0.0, 1.0), new_r1)

        new_r0 = jnp.clip(new_r0, -CLIP_VALUE, CLIP_VALUE)
        new_r1 = jnp.clip(new_r1, -CLIP_VALUE, CLIP_VALUE)
        return (new_r0, new_r1, new_predicate), None

    (r0, _, _), _ = jax.lax.scan(body, (r0, r1, predicate), programs.T)
    return r0


def map_profile_programs(programs: jax.Array, block_ids: jax.Array) -> jax.Array:
    return jnp.take(block_ids, programs)


@jax.jit
def evaluate_population(
    programs: jax.Array,
    train_x: jax.Array,
    train_y_input: jax.Array,
    train_y: jax.Array,
    holdout_x: jax.Array,
    holdout_y_input: jax.Array,
    holdout_y: jax.Array,
    length_penalty: float,
    block_ids: jax.Array,
) -> EvalMetrics:
    op_programs = map_profile_programs(programs, block_ids)
    train_pred = execute_programs(op_programs, train_x, train_y_input)
    holdout_pred = execute_programs(op_programs, holdout_x, holdout_y_input)

    train_error = train_pred - train_y[None, :]
    holdout_error = holdout_pred - holdout_y[None, :]
    train_mse = jnp.mean(train_error * train_error, axis=1)
    holdout_mse = jnp.mean(holdout_error * holdout_error, axis=1)
    max_abs_error = jnp.max(jnp.abs(train_error), axis=1)
    holdout_max_abs_error = jnp.max(jnp.abs(holdout_error), axis=1)
    active_blocks = jnp.sum(programs != 0, axis=1)

    finite = jnp.all(jnp.isfinite(train_pred), axis=1) & jnp.all(jnp.isfinite(holdout_pred), axis=1)
    reward = -jnp.log1p(train_mse) - length_penalty * active_blocks - jnp.where(finite, 0.0, 100.0)
    return EvalMetrics(
        reward=reward,
        train_mse=train_mse,
        holdout_mse=holdout_mse,
        max_abs_error=max_abs_error,
        holdout_max_abs_error=holdout_max_abs_error,
        active_blocks=active_blocks,
    )


def evaluate_program_np(
    program: np.ndarray,
    train_x: jax.Array,
    train_y_input: jax.Array,
    train_y: jax.Array,
    holdout_x: jax.Array,
    holdout_y_input: jax.Array,
    holdout_y: jax.Array,
    length_penalty: float,
    block_ids: jax.Array,
) -> dict[str, object]:
    programs = jnp.asarray(program[None, :], dtype=jnp.int32)
    metrics = evaluate_population(
        programs,
        train_x,
        train_y_input,
        train_y,
        holdout_x,
        holdout_y_input,
        holdout_y,
        length_penalty,
        block_ids,
    )
    jax.block_until_ready(metrics.reward)
    active_blocks = int(jax.device_get(metrics.active_blocks[0]))
    return {
        "reward": scalar(metrics.reward[0]),
        "train_mse": scalar(metrics.train_mse[0]),
        "holdout_mse": scalar(metrics.holdout_mse[0]),
        "max_abs_error": scalar(metrics.max_abs_error[0]),
        "holdout_max_abs_error": scalar(metrics.holdout_max_abs_error[0]),
        "active_blocks": active_blocks,
    }


def init_programs(key: jax.Array, population: int, program_length: int, vocab_size: int) -> jax.Array:
    return jax.random.randint(
        key,
        (population, program_length),
        minval=0,
        maxval=vocab_size,
        dtype=jnp.int32,
    )


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


def compact_program(program: np.ndarray) -> list[int]:
    return [int(op) for op in program if int(op) != 0]


def program_op_ids(program: np.ndarray, active_block_ids: np.ndarray) -> np.ndarray:
    return np.asarray([int(active_block_ids[int(op)]) for op in program], dtype=np.int32)


def compact_op_ids(program: np.ndarray, active_block_ids: np.ndarray) -> list[int]:
    return [int(active_block_ids[int(op)]) for op in program if int(op) != 0]


def block_to_python(op: int) -> str:
    block = CODEPY_BLOCKS[op]
    translations = {
        "p = r0 > 0": "p = r0 > 0",
        "p = r0 <= 0": "p = r0 <= 0",
        "p = r1 > 0": "p = r1 > 0",
        "p = r1 <= 0": "p = r1 <= 0",
        "p = r1 > r0": "p = r1 > r0",
        "p = r0 > r1": "p = r0 > r1",
        "p = r0 == 0": "p = r0 == 0",
        "p = r1 == 0": "p = r1 == 0",
        "r0 = where(p, r1, r0)": "r0 = r1 if p else r0",
        "r1 = where(p, r0, r1)": "r1 = r0 if p else r1",
        "r0 = where(p, 0, r0)": "r0 = 0 if p else r0",
        "r1 = where(p, 0, r1)": "r1 = 0 if p else r1",
        "r0 = where(p, -r0, r0)": "r0 = -r0 if p else r0",
        "r1 = where(p, -r1, r1)": "r1 = -r1 if p else r1",
        "r0 = where(p, 1, r0)": "r0 = 1 if p else r0",
        "r0 = where(p, -1, r0)": "r0 = -1 if p else r0",
        "r1 = where(p, 1, r1)": "r1 = 1 if p else r1",
        "r1 = where(p, -1, r1)": "r1 = -1 if p else r1",
        "r0 = square(r0)": "r0 = r0 * r0",
        "r1 = square(r1)": "r1 = r1 * r1",
        "r0 = relu(r0)": "r0 = max(r0, 0)",
        "r1 = relu(r1)": "r1 = max(r1, 0)",
        "r0 = max(r0, r1)": "r0 = max(r0, r1)",
        "r1 = max(r1, r0)": "r1 = max(r1, r0)",
        "r0 = min(r0, r1)": "r0 = min(r0, r1)",
        "r1 = min(r1, r0)": "r1 = min(r1, r0)",
        "r0 = sign(r0)": "r0 = 1 if r0 > 0 else (-1 if r0 < 0 else 0)",
        "r1 = sign(r1)": "r1 = 1 if r1 > 0 else (-1 if r1 < 0 else 0)",
        "r0 = clamp01(r0)": "r0 = min(max(r0, 0), 1)",
        "r1 = clamp01(r1)": "r1 = min(max(r1, 0), 1)",
        "r0 = max(abs(r0), r1)": "r0 = max(abs(r0), r1)",
        "r1 = max(abs(r1), r0)": "r1 = max(abs(r1), r0)",
        "r0 = clamp01(abs(r0))": "r0 = min(max(abs(r0), 0), 1)",
        "r1 = clamp01(abs(r1))": "r1 = min(max(abs(r1), 0), 1)",
    }
    return translations.get(block, block)


def pad_program(program_ids: list[int], program_length: int) -> np.ndarray:
    padded = list(program_ids[:program_length])
    padded.extend([0] * (program_length - len(padded)))
    return np.asarray(padded, dtype=np.int32)


def minimize_program(
    program: np.ndarray,
    train_x: jax.Array,
    train_y_input: jax.Array,
    train_y: jax.Array,
    holdout_x: jax.Array,
    holdout_y_input: jax.Array,
    holdout_y: jax.Array,
    *,
    tolerance: float,
    length_penalty: float,
    block_ids: jax.Array,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    program_length = int(program.shape[0])
    current_ids = compact_program(program)
    removals: list[dict[str, object]] = []
    index = 0

    while index < len(current_ids):
        candidate_ids = current_ids[:index] + current_ids[index + 1 :]
        candidate_program = pad_program(candidate_ids, program_length)
        candidate_metrics = evaluate_program_np(
            candidate_program,
            train_x,
            train_y_input,
            train_y,
            holdout_x,
            holdout_y_input,
            holdout_y,
            length_penalty,
            block_ids,
        )
        if (
            float(candidate_metrics["train_mse"]) <= tolerance
            and float(candidate_metrics["holdout_mse"]) <= tolerance
        ):
            removed_op = current_ids.pop(index)
            removed_real_op = int(jax.device_get(block_ids[removed_op]))
            removals.append(
                {
                    "index": index,
                    "removed_profile_id": removed_op,
                    "removed_op_id": removed_real_op,
                    "removed_block": CODEPY_BLOCKS[removed_real_op],
                    "remaining_blocks": len(current_ids),
                    "train_mse": candidate_metrics["train_mse"],
                    "holdout_mse": candidate_metrics["holdout_mse"],
                }
            )
        else:
            index += 1

    minimized = pad_program(current_ids, program_length)
    minimized_metrics = evaluate_program_np(
        minimized,
        train_x,
        train_y_input,
        train_y,
        holdout_x,
        holdout_y_input,
        holdout_y,
        length_penalty,
        block_ids,
    )
    return minimized, minimized_metrics, removals


def render_program(program: np.ndarray, active_block_ids: np.ndarray) -> list[str]:
    lines = ["def codepy_program(x, y=0):", "    r0 = x", "    r1 = 0", "    p = False"]
    active_index = 0
    for op_id in program:
        profile_op = int(op_id)
        if profile_op == 0:
            continue
        op = int(active_block_ids[profile_op])
        active_index += 1
        lines.append(f"    # block {active_index}: [{op:02d}] {CODEPY_BLOCKS[op]}")
        lines.append(f"    {block_to_python(op)}")
    lines.append("    return r0")
    return lines


def block_payload(active_block_ids: np.ndarray) -> list[dict[str, object]]:
    return [
        {"profile_id": i, "op_id": int(op_id), "block": CODEPY_BLOCKS[int(op_id)]}
        for i, op_id in enumerate(active_block_ids)
    ]


def save_json(path: str, payload: dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def scalar(x: jax.Array) -> float:
    return float(jax.device_get(x))


def run(args: argparse.Namespace) -> None:
    if args.elites < 1 or args.elites > args.population:
        raise SystemExit("--elites must be between 1 and --population.")
    if args.program_length < 1:
        raise SystemExit("--program-length must be at least 1.")
    if args.cases < 3:
        raise SystemExit("--cases must be at least 3.")
    if not 0.0 <= args.mutation_rate <= 1.0:
        raise SystemExit("--mutation-rate must be between 0 and 1.")
    if not 0.0 <= args.random_reset_fraction < 1.0:
        raise SystemExit("--random-reset-fraction must be in [0, 1).")

    train_x, train_y_input, train_y, holdout_x, holdout_y_input, holdout_y = make_dataset(
        args.target, args.cases
    )
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    active_block_ids_np = np.asarray(BLOCK_PROFILES[args.block_profile], dtype=np.int32)
    active_block_ids = jnp.asarray(active_block_ids_np, dtype=jnp.int32)
    vocab_size = int(active_block_ids_np.shape[0])
    programs = init_programs(init_key, args.population, args.program_length, vocab_size)
    random_reset_count = int(args.population * args.random_reset_fraction)

    print("TextPy/SoA CodePy block evolution prototype")
    print(f"backend: {jax.default_backend()}")
    print(f"devices: {jax.devices()}")
    print("")
    print("CodePy block vocabulary")
    print(f"  profile: {args.block_profile}")
    print(f"  blocks: {vocab_size}")
    for profile_id, op_id in enumerate(active_block_ids_np):
        print(f"  [{profile_id:02d}] op={int(op_id):02d} {CODEPY_BLOCKS[int(op_id)]}")
    print("")
    print("Target")
    print(f"  name: {args.target}")
    print(f"  formula: {TARGET_DESCRIPTIONS[args.target]}")
    print("")
    print("Protocol")
    print(f"  population: {args.population:,}")
    print(f"  program_length: {args.program_length}")
    print(f"  generations: {args.generations}")
    print(f"  elites: {args.elites}")
    print(f"  mutation_rate: {args.mutation_rate:.3f}")
    print(f"  random_reset_count: {random_reset_count}")
    print(f"  train_cases: {args.cases}")
    print("")

    compile_start = time.perf_counter()
    warmup = evaluate_population(
        programs,
        train_x,
        train_y_input,
        train_y,
        holdout_x,
        holdout_y_input,
        holdout_y,
        args.length_penalty,
        active_block_ids,
    )
    jax.block_until_ready(warmup.reward)
    compile_s = time.perf_counter() - compile_start

    best_record: dict[str, object] | None = None
    best_program: np.ndarray | None = None
    first_solution_record: dict[str, object] | None = None
    first_solution_program: np.ndarray | None = None
    generations_run = 0
    start = time.perf_counter()
    for generation in range(args.generations + 1):
        generations_run = generation + 1
        key, eval_key, mutate_key = jax.random.split(key, 3)
        metrics = evaluate_population(
            programs,
            train_x,
            train_y_input,
            train_y,
            holdout_x,
            holdout_y_input,
            holdout_y,
            args.length_penalty,
            active_block_ids,
        )
        jax.block_until_ready(metrics.reward)

        elite_indices = jnp.argsort(metrics.reward)[-args.elites :][::-1]
        elites = programs[elite_indices]
        best_idx = int(jax.device_get(elite_indices[0]))
        current = {
            "generation": generation,
            "reward": scalar(metrics.reward[best_idx]),
            "mean_reward": scalar(jnp.mean(metrics.reward)),
            "train_mse": scalar(metrics.train_mse[best_idx]),
            "holdout_mse": scalar(metrics.holdout_mse[best_idx]),
            "max_abs_error": scalar(metrics.max_abs_error[best_idx]),
            "holdout_max_abs_error": scalar(metrics.holdout_max_abs_error[best_idx]),
            "active_blocks": int(jax.device_get(metrics.active_blocks[best_idx])),
        }
        if best_record is None or float(current["reward"]) > float(best_record["reward"]):
            best_record = current
            best_program = np.asarray(jax.device_get(programs[best_idx]), dtype=np.int32)

        exact_solution = (
            float(current["train_mse"]) <= args.stop_mse
            and float(current["holdout_mse"]) <= args.stop_mse
        )
        if exact_solution and first_solution_record is None:
            first_solution_record = current.copy()
            first_solution_program = np.asarray(jax.device_get(programs[best_idx]), dtype=np.int32)

        if generation == 0 or generation == args.generations or generation % args.log_every == 0:
            print(
                f"gen={generation:03d} "
                f"best_reward={current['reward']:.6f} "
                f"mean_reward={current['mean_reward']:.6f} "
                f"train_mse={current['train_mse']:.6g} "
                f"holdout_mse={current['holdout_mse']:.6g} "
                f"active={current['active_blocks']}"
            )

        if exact_solution:
            if args.stop_active_blocks is None:
                print(f"early_stop: train/holdout MSE <= {args.stop_mse:g}")
                break

            candidate_program = np.asarray(jax.device_get(programs[best_idx]), dtype=np.int32)
            _, candidate_min_metrics, _ = minimize_program(
                candidate_program,
                train_x,
                train_y_input,
                train_y,
                holdout_x,
                holdout_y_input,
                holdout_y,
                tolerance=args.minimize_tolerance,
                length_penalty=args.length_penalty,
                block_ids=active_block_ids,
            )
            candidate_min_blocks = int(candidate_min_metrics["active_blocks"])
            if candidate_min_blocks <= args.stop_active_blocks:
                best_record = current
                best_program = candidate_program
                print(
                    "early_stop: "
                    f"MSE <= {args.stop_mse:g} and minimized active blocks "
                    f"{candidate_min_blocks} <= {args.stop_active_blocks}"
                )
                break
            if generation == 0 or generation % args.log_every == 0:
                print(
                    "continue_search: "
                    f"MSE <= {args.stop_mse:g}, minimized_active={candidate_min_blocks}, "
                    f"target_active<={args.stop_active_blocks}"
                )

        if generation < args.generations:
            programs = mutate_elites(
                mutate_key,
                elites,
                population=args.population,
                mutation_rate=args.mutation_rate,
                random_reset_count=random_reset_count,
                vocab_size=vocab_size,
            )

    elapsed = time.perf_counter() - start
    evaluated_programs = generations_run * args.population
    evaluated_block_ops = evaluated_programs * args.program_length * args.cases * 2
    throughput = evaluated_block_ops / max(elapsed, 1.0e-9)

    assert best_record is not None
    assert best_program is not None
    first_solution_payload: dict[str, object] | None = None
    if first_solution_record is not None and first_solution_program is not None:
        first_solution_payload = {
            **first_solution_record,
            "program_ids": [int(op) for op in first_solution_program],
            "op_ids": [int(op) for op in program_op_ids(first_solution_program, active_block_ids_np)],
            "compact_program_ids": compact_program(first_solution_program),
            "compact_op_ids": compact_op_ids(first_solution_program, active_block_ids_np),
            "compact_program_blocks": [
                CODEPY_BLOCKS[op] for op in compact_op_ids(first_solution_program, active_block_ids_np)
            ],
        }
    program_ids = [int(op) for op in best_program]
    op_ids = [int(op) for op in program_op_ids(best_program, active_block_ids_np)]
    compact_ids = compact_program(best_program)
    compact_op_id_list = compact_op_ids(best_program, active_block_ids_np)
    compact_blocks = [CODEPY_BLOCKS[op] for op in compact_op_id_list]
    rendered = render_program(best_program, active_block_ids_np)
    minimized_program, minimized_metrics, minimizer_removals = minimize_program(
        best_program,
        train_x,
        train_y_input,
        train_y,
        holdout_x,
        holdout_y_input,
        holdout_y,
        tolerance=args.minimize_tolerance,
        length_penalty=args.length_penalty,
        block_ids=active_block_ids,
    )
    minimized_ids = [int(op) for op in minimized_program]
    minimized_op_ids = [int(op) for op in program_op_ids(minimized_program, active_block_ids_np)]
    minimized_compact_ids = compact_program(minimized_program)
    minimized_compact_op_ids = compact_op_ids(minimized_program, active_block_ids_np)
    minimized_blocks = [CODEPY_BLOCKS[op] for op in minimized_compact_op_ids]
    minimized_rendered = render_program(minimized_program, active_block_ids_np)
    search_blocks = int(best_record["active_blocks"])
    minimal_blocks = int(minimized_metrics["active_blocks"])
    compression_ratio = search_blocks / max(minimal_blocks, 1)
    compression_metrics = {
        "search_blocks": search_blocks,
        "minimal_blocks": minimal_blocks,
        "removed_blocks": search_blocks - minimal_blocks,
        "compression_ratio": compression_ratio,
    }

    payload = {
        "format_version": 1,
        "prototype": "CodePy block evolution",
        "guardrail": "Constrained register DSL search; no arbitrary Python source is executed.",
        "target": {"name": args.target, "formula": TARGET_DESCRIPTIONS[args.target]},
        "inputs": {
            "arity": 2 if args.target in TWO_INPUT_TARGETS else 1,
            "train_cases": int(train_x.shape[0]),
            "holdout_cases": int(holdout_x.shape[0]),
        },
        "block_vocabulary": block_payload(active_block_ids_np),
        "block_profile": {
            "name": args.block_profile,
            "vocab_size": vocab_size,
            "base_block_count": len(BASE_BLOCK_IDS),
            "learned_block_count": len(LEARNED_BLOCK_IDS) - len(BASE_BLOCK_IDS),
        },
        "best": {
            **best_record,
            "program_ids": program_ids,
            "op_ids": op_ids,
            "compact_program_ids": compact_ids,
            "compact_op_ids": compact_op_id_list,
            "compact_program_blocks": compact_blocks,
            "rendered_python": rendered,
        },
        "first_solution": first_solution_payload,
        "minimized": {
            **minimized_metrics,
            "method": "semantic_block_deletion",
            "tolerance": args.minimize_tolerance,
            "removed_blocks": len(minimizer_removals),
            "removals": minimizer_removals,
            "program_ids": minimized_ids,
            "op_ids": minimized_op_ids,
            "compact_program_ids": minimized_compact_ids,
            "compact_op_ids": minimized_compact_op_ids,
            "compact_program_blocks": minimized_blocks,
            "rendered_python": minimized_rendered,
        },
        "compression": compression_metrics,
        "run": {
            "backend": jax.default_backend(),
            "compile_s": compile_s,
            "evolution_s": elapsed,
            "evaluated_programs": evaluated_programs,
            "evaluated_block_ops": evaluated_block_ops,
            "throughput_block_ops_s": throughput,
        },
        "args": vars(args),
    }

    if args.output_json:
        save_json(args.output_json, payload)

    print("")
    print("Run")
    print(f"  compile_s: {compile_s:.4f}")
    print(f"  evolution_s: {elapsed:.4f}")
    print(f"  evaluated_programs: {evaluated_programs:,}")
    print(f"  evaluated_block_ops: {evaluated_block_ops:,}")
    print(f"  throughput_block_ops_s: {throughput:,.0f}")
    print(f"  best_generation: {best_record['generation']}")
    print(f"  best_reward_seen: {best_record['reward']:.6f}")
    print(f"  best_train_mse: {best_record['train_mse']:.8g}")
    print(f"  best_holdout_mse: {best_record['holdout_mse']:.8g}")
    print(f"  best_max_abs_error: {best_record['max_abs_error']:.8g}")
    print(f"  best_holdout_max_abs_error: {best_record['holdout_max_abs_error']:.8g}")
    print(f"  best_active_blocks: {best_record['active_blocks']}")
    print(f"  minimized_active_blocks: {minimized_metrics['active_blocks']}")
    print(f"  minimized_train_mse: {minimized_metrics['train_mse']:.8g}")
    print(f"  minimized_holdout_mse: {minimized_metrics['holdout_mse']:.8g}")
    print(f"  minimizer_removed_blocks: {len(minimizer_removals)}")
    print("  compression:")
    print(f"    search_blocks: {search_blocks}")
    print(f"    minimal_blocks: {minimal_blocks}")
    print(f"    compression_ratio: {compression_ratio:.3f}")
    if args.output_json:
        print(f"  output_json: {args.output_json}")
    print("  best_program_ids:")
    print(f"    {program_ids}")
    print("  compact_blocks:")
    for op_id, block in zip(compact_ids, compact_blocks):
        print(f"    [{op_id:02d}] {block}")
    print("  rendered_python:")
    for line in rendered:
        print(f"    {line}")
    print("  minimized_blocks:")
    for op_id, block in zip(minimized_compact_ids, minimized_blocks):
        print(f"    [{op_id:02d}] {block}")
    print("  minimized_rendered_python:")
    for line in minimized_rendered:
        print(f"    {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolve safe CodePy register programs from cached command blocks."
    )
    parser.add_argument("--target", choices=tuple(TARGET_DESCRIPTIONS), default="quadratic")
    parser.add_argument("--block-profile", choices=tuple(BLOCK_PROFILES), default="base")
    parser.add_argument("--population", type=int, default=4096)
    parser.add_argument("--program-length", type=int, default=8)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--elites", type=int, default=128)
    parser.add_argument("--mutation-rate", type=float, default=0.16)
    parser.add_argument("--random-reset-fraction", type=float, default=0.04)
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0e-6,
        help="Reward penalty per active block; increase to bias evolution toward shorter programs.",
    )
    parser.add_argument("--cases", type=int, default=33)
    parser.add_argument("--stop-mse", type=float, default=1.0e-8)
    parser.add_argument(
        "--stop-active-blocks",
        type=int,
        default=None,
        help="If set, exact solutions keep evolving until minimization reaches this active block count.",
    )
    parser.add_argument(
        "--minimize-tolerance",
        type=float,
        default=1.0e-8,
        help="Train and holdout MSE threshold used by semantic block deletion.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
        help="JAX backend. Default is cpu because Apple Metal JAX is experimental.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
