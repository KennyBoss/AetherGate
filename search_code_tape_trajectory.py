#!/usr/bin/env python3
"""Trajectory-guided beam search for bounded CodePy tape programs.

This is the Phase 4 path for hard stateful tasks such as `argmax_index4`.
Instead of ranking candidates by a token prior, the beam samples execution
signatures: trajectories of `r0`, `r1`, `ptr`, `pc`, and predicate `p`.

The interpreter mirrors `evolve_code_tape.py` and stays inside the safe DSL.
No generated Python source is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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

from evolve_code_tape import (  # noqa: E402
    REFERENCE_PROGRAMS,
    TAPE_BLOCKS,
    TARGET_DESCRIPTIONS,
    compact_program,
    render_program,
)
from code_tape_trace_prior import (  # noqa: E402
    append_jsonl,
    classify_trace_record,
    load_json,
    predict_trace_probability,
    safe_log,
)
from code_tape_late_game_features import predict_late_game_score  # noqa: E402


CLIP_VALUE = 1.0e6
ARGMAX_ROLLOUT_SUFFIXES: tuple[tuple[int, ...], ...] = (
    (),
    (7,),
    (20,),
    (25,),
    (28,),
    (15,),
    (30,),
    (32,),
    (31,),
    (20, 25),
    (20, 28),
    (25, 28),
    (7, 20),
    (15, 30),
    (32, 31),
    (20, 25, 28),
    (7, 20, 25),
    (7, 20, 28),
    (20, 25, 28, 7),
    (20, 25, 28, 7, 15),
    (20, 25, 28, 7, 15, 30),
    (7, 20, 25, 28, 7, 15, 30),
    (20, 25, 28, 7, 15, 30, 32),
    (20, 25, 28, 7, 15, 30, 32, 31),
    (7, 20, 25, 28, 7, 15, 30, 32, 31),
)

ARGMAX_PREFIX_STAGE_DEFS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("read_max", (3,)),
    ("index_init", (24, 2)),
    ("first_move", (7,)),
    ("compare_current", (20,)),
    ("conditional_value_update", (25,)),
    ("conditional_index_update", (28,)),
    ("second_move", (7,)),
    ("loop_test", (15,)),
    ("loop_jump", (30,)),
    ("return_index", (32,)),
    ("halt", (31,)),
)


@dataclass(frozen=True)
class BeamItem:
    prefix: tuple[int, ...]
    signature: str
    parent_signature: str | None
    score: float
    raw_score: float
    program_ids: tuple[int, ...]
    survival_ttl: int = 0
    survival_score: float = 0.0
    survival_root: tuple[int, ...] = ()


def target_values_np(vectors: np.ndarray, target: str) -> np.ndarray:
    if target == "sum4":
        return np.sum(vectors, axis=1).astype(np.float32)
    if target == "count_positive4":
        return np.sum(vectors > 0.0, axis=1).astype(np.float32)
    if target == "argmax_index4":
        return np.argmax(vectors, axis=1).astype(np.float32)
    raise ValueError(f"Unknown target: {target}")


def make_dataset_np(target: str, cases: int, input_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    train_rows = []
    holdout_rows = []
    for i in range(cases):
        train_rows.append([values[(i * (j + 2) + j) % len(values)] for j in range(input_len)])
        holdout_rows.append(
            [values[((i + 1) * (j + 3) + 2 * j + 1) % len(values)] for j in range(input_len)]
        )
    train_x = np.asarray(train_rows, dtype=np.float32)
    holdout_x = np.asarray(holdout_rows, dtype=np.float32)
    return train_x, target_values_np(train_x, target), holdout_x, target_values_np(holdout_x, target)


def pad_program(prefix: tuple[int, ...], program_length: int) -> np.ndarray:
    program = list(prefix[:program_length])
    program.extend([0] * max(program_length - len(program), 0))
    return np.asarray(program, dtype=np.int32)


def prefix_to_program(prefix: tuple[int, ...], suffix: tuple[int, ...], program_length: int) -> np.ndarray:
    combined = tuple(prefix[:program_length]) + tuple(suffix)
    return pad_program(combined[:program_length], program_length)


def execute_program_np(
    program: np.ndarray,
    vectors: np.ndarray,
    *,
    tape_len: int,
    input_len: int,
    max_steps: int,
) -> dict[str, np.ndarray]:
    cases = vectors.shape[0]
    program_length = len(program)
    tape = np.zeros((cases, tape_len), dtype=np.float32)
    tape[:, :input_len] = vectors[:, :input_len]
    r0 = np.zeros(cases, dtype=np.float32)
    r1 = np.zeros(cases, dtype=np.float32)
    ptr = np.zeros(cases, dtype=np.int32)
    pc = np.zeros(cases, dtype=np.int32)
    predicate = np.zeros(cases, dtype=bool)

    r0_trace = np.zeros((max_steps, cases), dtype=np.float32)
    r1_trace = np.zeros((max_steps, cases), dtype=np.float32)
    ptr_trace = np.zeros((max_steps, cases), dtype=np.int32)
    pc_trace = np.zeros((max_steps, cases), dtype=np.int32)
    p_trace = np.zeros((max_steps, cases), dtype=np.int8)
    op_trace = np.zeros((max_steps, cases), dtype=np.int32)

    for step in range(max_steps):
        old_r0 = r0.copy()
        old_r1 = r1.copy()
        old_ptr = ptr.copy()
        old_pc = pc.copy()
        old_predicate = predicate.copy()
        old_tape = tape.copy()
        op_ids = program[np.clip(old_pc, 0, program_length - 1)]
        cell = old_tape[np.arange(cases), old_ptr]

        r0 = np.where(op_ids == 1, 0.0, r0)
        r1 = np.where(op_ids == 2, 0.0, r1)
        r0 = np.where(op_ids == 3, cell, r0)
        r1 = np.where(op_ids == 4, cell, r1)

        write_r0 = op_ids == 5
        write_r1 = op_ids == 6
        if np.any(write_r0):
            tape[write_r0, old_ptr[write_r0]] = old_r0[write_r0]
        if np.any(write_r1):
            tape[write_r1, old_ptr[write_r1]] = old_r1[write_r1]

        ptr = np.where(op_ids == 7, old_ptr + 1, ptr)
        ptr = np.where(op_ids == 8, old_ptr - 1, ptr)
        ptr = np.where(op_ids == 9, 0, ptr)
        ptr = np.clip(ptr, 0, tape_len - 1).astype(np.int32)

        r0 = np.where(op_ids == 10, old_r0 + old_r1, r0)
        r0 = np.where(op_ids == 11, old_r0 + 1.0, r0)
        r0 = np.where(op_ids == 12, old_r0 - 1.0, r0)
        r0 = np.where(op_ids == 23, old_ptr.astype(np.float32), r0)
        r1 = np.where(op_ids == 24, old_ptr.astype(np.float32), r1)
        r0 = np.where(op_ids == 32, old_r1, r0)
        r1 = np.where(op_ids == 33, old_r0, r1)

        predicate = np.where(op_ids == 13, old_r0 > 0.0, predicate)
        predicate = np.where(op_ids == 14, old_r1 > 0.0, predicate)
        predicate = np.where(op_ids == 15, old_ptr < input_len, predicate)
        predicate = np.where(op_ids == 20, cell > old_r0, predicate)
        predicate = np.where(op_ids == 21, cell > old_r1, predicate)
        predicate = np.where(op_ids == 22, old_ptr < input_len - 1, predicate)

        r0 = np.where(op_ids == 16, old_r0 + old_predicate.astype(np.float32), r0)
        r0 = np.where(op_ids == 17, np.where(old_predicate, old_r1, old_r0), r0)
        r0 = np.where(op_ids == 25, np.where(old_predicate, cell, old_r0), r0)
        r1 = np.where(op_ids == 26, np.where(old_predicate, cell, old_r1), r1)
        r0 = np.where(op_ids == 27, np.where(old_predicate, old_ptr.astype(np.float32), old_r0), r0)
        r1 = np.where(op_ids == 28, np.where(old_predicate, old_ptr.astype(np.float32), old_r1), r1)
        r1 = np.where(op_ids == 34, np.where(old_predicate, old_r0, old_r1), r1)

        default_pc = np.minimum(old_pc + 1, program_length - 1)
        pc = default_pc
        pc = np.where(op_ids == 18, 0, pc)
        pc = np.where(op_ids == 19, np.where(old_predicate, 0, default_pc), pc)
        pc = np.where(op_ids == 29, min(3, program_length - 1), pc)
        pc = np.where(op_ids == 30, np.where(old_predicate, min(3, program_length - 1), default_pc), pc)
        pc = np.where(op_ids == 31, old_pc, pc)
        pc = np.clip(pc, 0, program_length - 1).astype(np.int32)

        r0 = np.clip(r0, -CLIP_VALUE, CLIP_VALUE)
        r1 = np.clip(r1, -CLIP_VALUE, CLIP_VALUE)
        tape = np.clip(tape, -CLIP_VALUE, CLIP_VALUE)

        r0_trace[step] = r0
        r1_trace[step] = r1
        ptr_trace[step] = ptr
        pc_trace[step] = pc
        p_trace[step] = predicate.astype(np.int8)
        op_trace[step] = op_ids

    return {
        "output": r0.copy(),
        "r0": r0_trace,
        "r1": r1_trace,
        "ptr": ptr_trace,
        "pc": pc_trace,
        "p": p_trace,
        "op": op_trace,
    }


def signature_hash(trace: dict[str, np.ndarray], *, signature_cases: int, signature_steps: int) -> tuple[str, list[int]]:
    cases = min(signature_cases, trace["r0"].shape[1])
    steps = min(signature_steps, trace["r0"].shape[0])
    parts = [
        np.rint(trace["r0"][:steps, :cases]).clip(-9, 9).astype(np.int8),
        np.rint(trace["r1"][:steps, :cases]).clip(-9, 9).astype(np.int8),
        trace["ptr"][:steps, :cases].astype(np.int8),
        trace["pc"][:steps, :cases].astype(np.int8),
        trace["p"][:steps, :cases].astype(np.int8),
    ]
    packed = np.concatenate([part.reshape(-1) for part in parts]).astype(np.int8)
    digest = hashlib.sha1(packed.tobytes()).hexdigest()[:16]
    preview = [int(x) for x in packed[: min(48, packed.size)]]
    return digest, preview


def signature_entropy(signatures: list[str]) -> float:
    if not signatures:
        return 0.0
    counts: dict[str, int] = {}
    for sig in signatures:
        counts[sig] = counts.get(sig, 0) + 1
    probs = np.asarray(list(counts.values()), dtype=np.float64) / len(signatures)
    return float(-np.sum(probs * np.log2(probs)))


def argmax_trace_score(trace: dict[str, np.ndarray], vectors: np.ndarray, target: np.ndarray, input_len: int) -> dict[str, float]:
    output = trace["output"]
    train_mse = float(np.mean((output - target) ** 2))
    max_values = np.max(vectors[:, :input_len], axis=1)
    max_indices = np.argmax(vectors[:, :input_len], axis=1).astype(np.float32)
    r0 = trace["r0"]
    r1 = trace["r1"]
    ptr = trace["ptr"]
    p = trace["p"]
    pc = trace["pc"]

    index_hit = float(np.mean(np.any(np.abs(r1 - max_indices[None, :]) < 0.1, axis=0)))
    max_hit = float(np.mean(np.any(np.abs(r0 - max_values[None, :]) < 0.1, axis=0)))
    output_hit = float(np.mean(np.abs(output - target) < 0.1))
    coverage = float(np.mean(np.max(ptr, axis=0) >= input_len))
    predicate_rate = float(np.mean(p))
    predicate_entropy = 1.0 - abs(predicate_rate - 0.5) * 2.0
    loop_hit = float(np.mean(np.any(pc == 3, axis=0)))
    halt_hit = float(np.mean(pc[-1] == pc[-2])) if pc.shape[0] >= 2 else 0.0
    running_values = []
    running_indices = []
    for k in range(1, input_len + 1):
        prefix = vectors[:, :k]
        running_values.append(np.max(prefix, axis=1))
        running_indices.append(np.argmax(prefix, axis=1).astype(np.float32))
    value_state_hits = []
    index_state_hits = []
    paired_state_hits = []
    for value, index in zip(running_values, running_indices):
        value_matches = np.abs(r0 - value[None, :]) < 0.1
        index_matches = np.abs(r1 - index[None, :]) < 0.1
        value_state_hits.append(np.mean(np.any(value_matches, axis=0)))
        index_state_hits.append(np.mean(np.any(index_matches, axis=0)))
        paired_state_hits.append(np.mean(np.any(value_matches & index_matches, axis=0)))
    running_value_hit = float(np.mean(value_state_hits))
    running_index_hit = float(np.mean(index_state_hits))
    paired_running_hit = float(np.mean(paired_state_hits))
    trajectory_score = (
        2.5 * output_hit
        + 1.5 * index_hit
        + 1.0 * max_hit
        + 1.3 * paired_running_hit
        + 0.8 * running_value_hit
        + 0.8 * running_index_hit
        + 0.7 * coverage
        + 0.5 * max(predicate_entropy, 0.0)
        + 0.4 * loop_hit
        + 0.2 * halt_hit
    )
    return {
        "train_mse": train_mse,
        "output_hit": output_hit,
        "index_hit": index_hit,
        "max_hit": max_hit,
        "coverage": coverage,
        "predicate_entropy": max(predicate_entropy, 0.0),
        "loop_hit": loop_hit,
        "halt_hit": halt_hit,
        "running_value_hit": running_value_hit,
        "running_index_hit": running_index_hit,
        "paired_running_hit": paired_running_hit,
        "trajectory_score": trajectory_score,
    }


def generic_trace_score(trace: dict[str, np.ndarray], target: np.ndarray, input_len: int) -> dict[str, float]:
    output = trace["output"]
    train_mse = float(np.mean((output - target) ** 2))
    output_hit = float(np.mean(np.abs(output - target) < 0.1))
    coverage = float(np.mean(np.max(trace["ptr"], axis=0) >= input_len))
    predicate_rate = float(np.mean(trace["p"]))
    predicate_entropy = max(1.0 - abs(predicate_rate - 0.5) * 2.0, 0.0)
    trajectory_score = 2.5 * output_hit + 0.7 * coverage + 0.5 * predicate_entropy
    return {
        "train_mse": train_mse,
        "output_hit": output_hit,
        "index_hit": 0.0,
        "max_hit": 0.0,
        "coverage": coverage,
        "predicate_entropy": predicate_entropy,
        "loop_hit": float(np.mean(np.any(trace["pc"] == 3, axis=0))),
        "halt_hit": float(np.mean(trace["pc"][-1] == trace["pc"][-2])) if trace["pc"].shape[0] >= 2 else 0.0,
        "trajectory_score": trajectory_score,
    }


def score_candidate(
    program: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_trace = execute_program_np(
        program,
        train_x,
        tape_len=args.tape_len,
        input_len=args.input_len,
        max_steps=args.max_steps,
    )
    holdout_trace = execute_program_np(
        program,
        holdout_x,
        tape_len=args.tape_len,
        input_len=args.input_len,
        max_steps=args.max_steps,
    )
    if args.target == "argmax_index4":
        train_scores = argmax_trace_score(train_trace, train_x, train_y, args.input_len)
    else:
        train_scores = generic_trace_score(train_trace, train_y, args.input_len)
    holdout_mse = float(np.mean((holdout_trace["output"] - holdout_y) ** 2))
    sig, preview = signature_hash(
        train_trace,
        signature_cases=args.signature_cases,
        signature_steps=args.signature_steps,
    )
    raw_score = (
        train_scores["trajectory_score"]
        - args.mse_weight * math.log1p(train_scores["train_mse"])
        - args.length_penalty * float(np.sum(program != 0))
    )
    return {
        **train_scores,
        "holdout_mse": holdout_mse,
        "signature": sig,
        "signature_preview": preview,
        "raw_score": raw_score,
        "program_ids": [int(op) for op in program],
        "compact_program_ids": compact_program(program),
    }


def candidate_score_value(row: dict[str, Any], args: argparse.Namespace) -> float:
    return (
        float(row["trajectory_score"])
        - args.mse_weight * math.log1p(float(row["train_mse"]))
        - args.length_penalty * float(len(row.get("compact_program_ids", [])))
    )


def rollout_suffixes_for_prefix(prefix: tuple[int, ...], args: argparse.Namespace) -> list[tuple[int, ...]]:
    remaining = max(args.program_length - len(prefix), 0)
    suffixes = [suffix[:remaining] for suffix in ARGMAX_ROLLOUT_SUFFIXES if len(suffix) <= remaining]
    if remaining > 0:
        suffixes.extend(
            [
                tuple([op] * min(remaining, args.value_rollout_repeat_len))
                for op in args.value_rollout_ops
            ]
        )
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for suffix in suffixes:
        if suffix not in seen:
            seen.add(suffix)
            unique.append(suffix)
        if len(unique) >= args.value_rollouts:
            break
    return unique


def estimate_prefix_value(
    prefix: tuple[int, ...],
    train_x: np.ndarray,
    train_y: np.ndarray,
    holdout_x: np.ndarray,
    holdout_y: np.ndarray,
    args: argparse.Namespace,
    trace_prior: dict[str, Any] | None,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_suffix: tuple[int, ...] = ()
    evaluated = 0
    for suffix in rollout_suffixes_for_prefix(prefix, args):
        program = prefix_to_program(prefix, suffix, args.program_length)
        row = score_candidate(program, train_x, train_y, holdout_x, holdout_y, args)
        row["argmax_role_score"] = argmax_role_score([int(op) for op in prefix + suffix])
        if trace_prior is not None:
            row["trace_prior_probability"] = predict_trace_probability(trace_prior, row)
        value = candidate_score_value(row, args)
        value += args.value_rollout_pair_weight * float(row.get("paired_running_hit", 0.0))
        value += args.value_rollout_output_weight * float(row.get("output_hit", 0.0))
        value += args.value_rollout_role_weight * float(row.get("argmax_role_score", 0.0))
        if trace_prior is not None:
            value += args.value_rollout_prior_weight * safe_log(float(row.get("trace_prior_probability", 0.0)))
        evaluated += 1
        if best is None or value > float(best["prefix_value_score"]):
            best = {
                "prefix_value_score": value,
                "prefix_value_suffix": [int(op) for op in suffix],
                "prefix_value_train_mse": row["train_mse"],
                "prefix_value_holdout_mse": row["holdout_mse"],
                "prefix_value_output_hit": row.get("output_hit", 0.0),
                "prefix_value_paired_running_hit": row.get("paired_running_hit", 0.0),
                "prefix_value_trace_prior_probability": row.get("trace_prior_probability"),
            }
            best_suffix = suffix
    if best is None:
        return {
            "prefix_value_score": 0.0,
            "prefix_value_suffix": [],
            "prefix_value_train_mse": math.inf,
            "prefix_value_holdout_mse": math.inf,
            "prefix_value_output_hit": 0.0,
            "prefix_value_paired_running_hit": 0.0,
            "prefix_value_trace_prior_probability": None,
            "prefix_value_evaluations": 0,
        }
    best["prefix_value_evaluations"] = evaluated
    best["prefix_value_suffix_len"] = len(best_suffix)
    return best


def first_index(prefix: list[int], ops: set[int]) -> int | None:
    for i, op in enumerate(prefix):
        if op in ops:
            return i
    return None


def argmax_role_score(prefix_ids: list[int]) -> float:
    """Score whether a prefix contains argmax-like state roles, not a reference sequence."""
    prefix = [int(op) for op in prefix_ids]
    value_init = first_index(prefix, {3})
    weak_value_init = first_index(prefix, {4})
    index_init = first_index(prefix, {24, 2})
    weak_index_init = first_index(prefix, {23, 1})
    move = first_index(prefix, {7})
    compare = first_index(prefix, {20})
    weak_compare = first_index(prefix, {21})
    cond_value = first_index(prefix, {25})
    weak_cond_value = first_index(prefix, {26, 17})
    cond_index = first_index(prefix, {28})
    weak_cond_index = first_index(prefix, {27, 34})
    loop_test = first_index(prefix, {15})
    weak_loop_test = first_index(prefix, {22})
    loop_jump = first_index(prefix, {30})
    weak_loop_jump = first_index(prefix, {18, 19, 29})
    return_index = first_index(prefix, {32})
    halt = first_index(prefix, {31})

    score = 0.0
    score += 2.0 if value_init is not None else (0.8 if weak_value_init is not None else 0.0)
    score += 2.0 if index_init is not None else (0.8 if weak_index_init is not None else 0.0)
    score += 1.6 if move is not None else 0.0
    score += 2.0 if compare is not None else (0.9 if weak_compare is not None else 0.0)
    score += 2.0 if cond_value is not None else (0.7 if weak_cond_value is not None else 0.0)
    score += 2.0 if cond_index is not None else (0.7 if weak_cond_index is not None else 0.0)
    score += 1.4 if loop_test is not None else (0.5 if weak_loop_test is not None else 0.0)
    score += 1.6 if loop_jump is not None else (0.5 if weak_loop_jump is not None else 0.0)
    score += 1.8 if return_index is not None else 0.0
    score += 0.8 if halt is not None else 0.0

    if value_init is not None and move is not None and value_init < move:
        score += 1.0
    if index_init is not None and move is not None and index_init < move:
        score += 1.0
    if move is not None and compare is not None and move < compare:
        score += 1.0
    if compare is not None and cond_value is not None and compare < cond_value:
        score += 1.0
    if compare is not None and cond_index is not None and compare < cond_index:
        score += 1.0
    if cond_value is not None and cond_index is not None:
        score += 1.0
    if loop_test is not None and loop_jump is not None and loop_test < loop_jump:
        score += 0.8
    if loop_jump is not None and return_index is not None and loop_jump < return_index:
        score += 0.5
    if return_index is not None and halt is not None and return_index < halt:
        score += 0.3
    return score / 20.0


def ordered_stage_progress(prefix_ids: list[int], stage_defs: tuple[tuple[str, tuple[int, ...]], ...]) -> int:
    """Count the ordered role stages matched from the start of a prefix."""
    prefix = [int(op) for op in prefix_ids]
    start = 0
    matched = 0
    for _, choices in stage_defs:
        found = None
        for i in range(start, len(prefix)):
            if prefix[i] in choices:
                found = i
                break
        if found is None:
            break
        matched += 1
        start = found + 1
    return matched


def ordered_relation(prefix: list[int], before: set[int], after: set[int]) -> int:
    before_index = first_index(prefix, before)
    after_index = first_index(prefix, after)
    if before_index is None or after_index is None:
        return 0
    return 1 if before_index < after_index else -1


def argmax_prefix_stage_score(prefix_ids: list[int]) -> dict[str, Any]:
    """Order-aware prefix survival score for argmax-like program skeletons.

    This is deliberately weaker than the late-game ranker: it does not require a
    complete terminal program and it does not accept/reject candidates. It only
    gives the selector an early lane for prefixes that advance through the ordered
    read/index/move/compare/update/loop roles before those roles reduce MSE.
    """
    prefix = [int(op) for op in prefix_ids]
    matched = ordered_stage_progress(prefix, ARGMAX_PREFIX_STAGE_DEFS)
    ordered_hits = 0
    order_violations = 0
    relations = (
        ({3}, {24, 2}),
        ({24, 2}, {7}),
        ({7}, {20}),
        ({20}, {25}),
        ({20}, {28}),
        ({25}, {28}),
        ({28}, {15}),
        ({15}, {30}),
        ({30}, {32}),
        ({32}, {31}),
    )
    for before, after in relations:
        relation = ordered_relation(prefix, before, after)
        if relation > 0:
            ordered_hits += 1
        elif relation < 0:
            order_violations += 1

    first_move = first_index(prefix, {7})
    loop_test = first_index(prefix, {15})
    loop_jump = first_index(prefix, {30})
    return_index = first_index(prefix, {32})
    halt = first_index(prefix, {31})
    compare = first_index(prefix, {20})
    cond_value = first_index(prefix, {25})
    cond_index = first_index(prefix, {28})

    if loop_test is not None and (first_move is None or loop_test < first_move):
        order_violations += 2
    if loop_jump is not None and (loop_test is None or loop_jump < loop_test):
        order_violations += 1
    if return_index is not None and (loop_jump is None or return_index < loop_jump):
        order_violations += 1
    if halt is not None and (return_index is None or halt < return_index):
        order_violations += 1
    if cond_value is not None and (compare is None or cond_value < compare):
        order_violations += 1
    if cond_index is not None and (compare is None or cond_index < compare):
        order_violations += 1

    score = float(matched) + 0.25 * float(ordered_hits) - 0.75 * float(order_violations)
    max_score = float(len(ARGMAX_PREFIX_STAGE_DEFS)) + 0.25 * float(len(relations))
    normalized = max(0.0, min(score / max_score, 1.0))
    return {
        "argmax_prefix_stage_score": score,
        "argmax_prefix_stage_score_norm": normalized,
        "argmax_prefix_stage_matched": matched,
        "argmax_prefix_ordered_hits": ordered_hits,
        "argmax_prefix_order_violations": order_violations,
    }


def select_diverse(scored: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_prefixes: set[tuple[int, ...]] = set()
    signature_counts: dict[str, int] = {}
    remaining = list(scored)

    def add_row(
        row: dict[str, Any],
        score: float,
        *,
        lane: str,
        ignore_signature_limit: bool = False,
    ) -> bool:
        prefix = tuple(int(op) for op in row["prefix_ids"])
        if prefix in selected_prefixes:
            return False
        duplicate_count = signature_counts.get(row["signature"], 0)
        if not ignore_signature_limit and duplicate_count >= args.max_same_signature:
            return False
        row["diverse_score"] = score
        row["selection_lane"] = lane
        row["selection_ignored_signature_limit"] = bool(ignore_signature_limit)
        selected.append(row)
        selected_prefixes.add(prefix)
        signature_counts[row["signature"]] = duplicate_count + 1
        return True

    def fill_lane(
        rows: list[dict[str, Any]],
        quota: int,
        key,
        *,
        lane: str,
        ignore_signature_limit: bool = False,
    ) -> None:
        if quota <= 0 or len(selected) >= args.beam_width:
            return
        added = 0
        for row in sorted(rows, key=key, reverse=True):
            duplicate_count = signature_counts.get(row["signature"], 0)
            score = float(row["raw_score"]) - args.diversity_penalty * duplicate_count
            if add_row(row, score, lane=lane, ignore_signature_limit=ignore_signature_limit):
                added += 1
            if len(selected) >= args.beam_width or added >= quota:
                break

    def fill_survival_lane(rows: list[dict[str, Any]], quota: int, key) -> None:
        if quota <= 0 or len(selected) >= args.beam_width:
            return
        added = 0
        root_counts: dict[tuple[int, ...], int] = {}
        for row in sorted(rows, key=key, reverse=True):
            root = tuple(int(op) for op in (row.get("survival_root") or row["prefix_ids"]))
            if args.survival_max_per_root > 0 and root_counts.get(root, 0) >= args.survival_max_per_root:
                continue
            duplicate_count = signature_counts.get(row["signature"], 0)
            score = float(row["raw_score"]) - args.diversity_penalty * duplicate_count
            if add_row(
                row,
                score,
                lane="survival",
                ignore_signature_limit=args.survival_ignore_signature_limit,
            ):
                root_counts[root] = root_counts.get(root, 0) + 1
                added += 1
            if len(selected) >= args.beam_width or added >= quota:
                break

    if args.prefix_stage_lane_fraction > 0.0:
        prefix_stage_quota = int(round(args.beam_width * args.prefix_stage_lane_fraction))
        fill_lane(
            remaining,
            prefix_stage_quota,
            lambda row: (
                float(row.get("argmax_prefix_stage_score", 0.0)),
                int(row.get("argmax_prefix_stage_matched", 0)),
                -int(row.get("argmax_prefix_order_violations", 0)),
                float(row.get("prefix_value_score") or -1.0e30),
                float(row.get("survival_score", 0.0)),
                float(row.get("argmax_role_score", 0.0)),
                -float(row["train_mse"]),
            ),
            lane="prefix_stage",
            ignore_signature_limit=args.prefix_stage_ignore_signature_limit,
        )

    if args.selection_mode == "multilane":
        raw_quota = max(1, int(round(args.beam_width * args.raw_lane_fraction)))
        prior_quota = int(round(args.beam_width * args.prior_lane_fraction))
        role_quota = int(round(args.beam_width * args.role_lane_fraction))
        behavior_quota = int(round(args.beam_width * args.behavior_lane_fraction))
        fill_lane(
            remaining,
            raw_quota,
            lambda row: (float(row["raw_score"]), -float(row["train_mse"])),
            lane="raw",
        )
        fill_lane(
            remaining,
            prior_quota,
            lambda row: (
                float(row.get("trace_prior_probability", 0.0)),
                float(row["trajectory_score"]),
                -float(row["train_mse"]),
            ),
            lane="prior",
        )
        fill_lane(
            remaining,
            role_quota,
            lambda row: (
                float(row.get("argmax_role_score", 0.0)),
                float(row.get("paired_running_hit", 0.0)),
                float(row.get("running_value_hit", 0.0)),
                float(row.get("coverage", 0.0)),
            ),
            lane="role",
        )
        fill_lane(
            remaining,
            behavior_quota,
            lambda row: (
                float(row.get("paired_running_hit", 0.0)),
                float(row.get("running_value_hit", 0.0)),
                float(row.get("running_index_hit", 0.0)),
                float(row.get("coverage", 0.0)),
                float(row.get("loop_hit", 0.0)),
            ),
            lane="behavior",
        )

    if args.late_game_lane_fraction > 0.0:
        late_game_quota = int(round(args.beam_width * args.late_game_lane_fraction))
        fill_lane(
            [row for row in remaining if row.get("late_game_score") is not None],
            late_game_quota,
            lambda row: (
                float(row.get("late_game_score") or 0.0),
                float(row.get("argmax_role_score", 0.0)),
                float(row.get("paired_running_hit", 0.0)),
                -float(row["train_mse"]),
            ),
            lane="late_game",
        )

    if args.value_lane_fraction > 0.0:
        value_quota = int(round(args.beam_width * args.value_lane_fraction))
        fill_lane(
            remaining,
            value_quota,
            lambda row: (
                float(row.get("prefix_value_score") or -1.0e30),
                float(row.get("argmax_role_score", 0.0)),
                float(row.get("trace_prior_probability", 0.0)),
                -float(row["train_mse"]),
            ),
            lane="value",
        )

    if args.survival_lane_fraction > 0.0:
        survival_quota = int(round(args.beam_width * args.survival_lane_fraction))
        fill_survival_lane(
            [row for row in remaining if int(row.get("survival_ttl", 0)) > 0],
            survival_quota,
            lambda row: (
                float(row.get("survival_score", 0.0)),
                int(row.get("survival_ttl", 0)),
                float(row.get("prefix_value_score") or -1.0e30),
                float(row.get("argmax_role_score", 0.0)),
                -float(row["train_mse"]),
            ),
        )

    while remaining and len(selected) < args.beam_width:
        best_i = 0
        best_score = -1.0e30
        for i, row in enumerate(remaining):
            if tuple(int(op) for op in row["prefix_ids"]) in selected_prefixes:
                continue
            duplicate_count = signature_counts.get(row["signature"], 0)
            if duplicate_count >= args.max_same_signature:
                continue
            diversity_penalty = args.diversity_penalty * duplicate_count
            score = float(row["raw_score"]) - diversity_penalty
            if score > best_score:
                best_i = i
                best_score = score
        row = remaining.pop(best_i)
        if best_score <= -1.0e29:
            break
        add_row(row, best_score, lane="fallback")
    return selected


def top_unique_by(scored: list[dict[str, Any]], limit: int, key) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for row in sorted(scored, key=key, reverse=True):
        prefix = tuple(int(op) for op in row["prefix_ids"])
        if prefix in seen:
            continue
        seen.add(prefix)
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def propagate_survival(parent: BeamItem) -> tuple[int, float, tuple[int, ...]]:
    if parent.survival_ttl <= 0:
        return 0, 0.0, ()
    return parent.survival_ttl - 1, parent.survival_score, parent.survival_root


def maybe_activate_survival(row: dict[str, Any], args: argparse.Namespace) -> None:
    if args.survival_ttl <= 0:
        return
    value_score = row.get("prefix_value_score")
    if value_score is None:
        return
    value_score = float(value_score)
    if value_score < args.survival_min_value:
        return
    current_ttl = int(row.get("survival_ttl", 0))
    current_score = float(row.get("survival_score", 0.0))
    if value_score >= current_score or current_ttl <= 0:
        row["survival_score"] = value_score
        row["survival_root"] = list(row["prefix_ids"])
    else:
        row["survival_score"] = current_score
        row["survival_root"] = row.get("survival_root", [])
    row["survival_ttl"] = max(current_ttl, args.survival_ttl)
    row["survival_activated"] = True


def build_trace_graph(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    selected_signatures: list[str],
) -> dict[str, Any]:
    return {
        "nodes": sorted(nodes.values(), key=lambda row: (row["depth"], row["signature"])),
        "edges": edges,
        "selected_signatures": selected_signatures,
        "signature_entropy": signature_entropy(selected_signatures),
    }


def run(args: argparse.Namespace) -> None:
    if args.program_length < 1:
        raise SystemExit("--program-length must be >= 1")
    if args.beam_width < 1:
        raise SystemExit("--beam-width must be >= 1")
    if args.max_same_signature < 1:
        raise SystemExit("--max-same-signature must be >= 1")
    trace_prior = load_json(args.trace_prior_json) if args.trace_prior_json else None
    late_game_model = load_json(args.late_game_ranker_json) if args.late_game_ranker_json else None
    # The late-game ranker expects terminal control (halt/return) to exist, so it
    # only fires on the final stages of the program; earlier depths leave it None.
    late_game_gate_depth = args.program_length - max(args.late_game_min_remaining, 0)
    if args.candidate_jsonl:
        candidate_path = Path(args.candidate_jsonl)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text("", encoding="utf-8")
    train_x, train_y, holdout_x, holdout_y = make_dataset_np(args.target, args.cases, args.input_len)
    vocab_size = len(TAPE_BLOCKS)
    start = time.perf_counter()
    beam = [
        BeamItem(prefix=(), signature="root", parent_signature=None, score=0.0, raw_score=0.0, program_ids=tuple())
    ]
    reference_prefixes = [
        tuple(REFERENCE_PROGRAMS[args.target][:depth])
        for depth in range(1, min(args.program_length, len(REFERENCE_PROGRAMS[args.target])) + 1)
    ]
    best: dict[str, Any] | None = None
    first_solution: dict[str, Any] | None = None
    layers: list[dict[str, Any]] = []
    graph_nodes: dict[str, dict[str, Any]] = {
        "root": {"signature": "root", "depth": 0, "best_score": 0.0, "best_program_ids": []}
    }
    graph_edges: list[dict[str, Any]] = []
    evaluated_programs = 0
    candidate_records_written = 0
    value_rollout_evaluations = 0
    survival_activations = 0
    survival_selected_total = 0
    late_game_lane_selected_total = 0
    prefix_stage_lane_selected_total = 0

    for depth in range(1, args.program_length + 1):
        candidates: list[tuple[tuple[int, ...], BeamItem, int]] = []
        for item in beam:
            for op_id in range(vocab_size):
                candidates.append(((*item.prefix, op_id), item, op_id))
        if args.seed_reference_prefixes and depth <= len(reference_prefixes):
            prefix = reference_prefixes[depth - 1]
            candidates.append((prefix, beam[0], prefix[-1]))
        seen_prefixes: set[tuple[int, ...]] = set()
        deduped_candidates: list[tuple[tuple[int, ...], BeamItem, int]] = []
        for prefix, parent, op_id in candidates:
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            deduped_candidates.append((prefix, parent, op_id))
        candidates = deduped_candidates
        scored: list[dict[str, Any]] = []
        for prefix, parent, op_id in candidates:
            survival_ttl, survival_score, survival_root = propagate_survival(parent)
            program = pad_program(prefix, args.program_length)
            row = score_candidate(program, train_x, train_y, holdout_x, holdout_y, args)
            evaluated_programs += 1
            row.update(
                {
                    "depth": depth,
                    "prefix_ids": [int(op) for op in prefix],
                    "parent_signature": parent.signature,
                    "op_id": int(op_id),
                    "op_block": TAPE_BLOCKS[op_id],
                    "parent_survival_ttl": parent.survival_ttl,
                    "parent_survival_score": parent.survival_score,
                    "parent_survival_root": [int(op) for op in parent.survival_root],
                    "survival_ttl": survival_ttl,
                    "survival_score": survival_score,
                    "survival_root": [int(op) for op in survival_root],
                    "survival_activated": False,
                    "seeded_reference_prefix": args.seed_reference_prefixes
                    and depth <= len(reference_prefixes)
                    and tuple(prefix) == reference_prefixes[depth - 1],
                    "source": "trajectory_beam_candidate",
                    "target": args.target,
                    "args": {
                        "target": args.target,
                        "program_length": args.program_length,
                        "max_steps": args.max_steps,
                        "tape_len": args.tape_len,
                        "input_len": args.input_len,
                        "cases": args.cases,
                        "beam_width": args.beam_width,
                        "signature_cases": args.signature_cases,
                        "signature_steps": args.signature_steps,
                        "survival_lane_fraction": args.survival_lane_fraction,
                        "survival_ttl": args.survival_ttl,
                        "survival_min_value": args.survival_min_value,
                        "survival_max_per_root": args.survival_max_per_root,
                        "backend": args.backend,
                    },
                }
            )
            row["argmax_role_score"] = argmax_role_score(row["prefix_ids"]) if args.target == "argmax_index4" else 0.0
            if args.target == "argmax_index4":
                row.update(argmax_prefix_stage_score(row["prefix_ids"]))
            else:
                row.update(
                    {
                        "argmax_prefix_stage_score": 0.0,
                        "argmax_prefix_stage_score_norm": 0.0,
                        "argmax_prefix_stage_matched": 0,
                        "argmax_prefix_ordered_hits": 0,
                        "argmax_prefix_order_violations": 0,
                    }
                )
            row.update(classify_trace_record(row, success_mse=args.stop_mse))
            if trace_prior is not None:
                probability = predict_trace_probability(trace_prior, row)
                row["trace_prior_probability"] = probability
                row["trace_prior_log_probability"] = safe_log(probability)
                row["raw_score_without_trace_prior"] = row["raw_score"]
                row["raw_score"] = row["raw_score"] + args.trace_prior_weight * safe_log(probability)
            row["late_game_score"] = None
            if late_game_model is not None and depth >= late_game_gate_depth:
                late_game_score = predict_late_game_score(late_game_model, row["compact_program_ids"])
                row["late_game_score"] = late_game_score
                if args.late_game_weight != 0.0:
                    row["raw_score_without_late_game"] = row["raw_score"]
                    row["raw_score"] = row["raw_score"] + args.late_game_weight * safe_log(late_game_score)
            scored.append(row)

        if args.value_rollouts > 0 and depth <= args.value_rollout_max_depth:
            preselected_rows: list[dict[str, Any]] = []
            preselected_rows.extend(
                top_unique_by(
                    scored,
                    args.value_rollout_top_k,
                    lambda row: (
                        float(row["raw_score"]),
                        float(row.get("argmax_role_score", 0.0)),
                        float(row.get("paired_running_hit", 0.0)),
                        -float(row["train_mse"]),
                    ),
                )
            )
            role_top_k = min(args.value_rollout_role_top_k, len(scored))
            if role_top_k > 0:
                preselected_rows.extend(
                    top_unique_by(
                        scored,
                        role_top_k,
                        lambda row: (
                            float(row.get("argmax_role_score", 0.0)),
                            float(row.get("trace_prior_probability", 0.0)),
                            float(row.get("running_value_hit", 0.0)),
                            float(row.get("running_index_hit", 0.0)),
                        ),
                    )
                )
            preselected_map = {
                tuple(int(op) for op in row["prefix_ids"]): row
                for row in preselected_rows
            }
            preselected = list(preselected_map.values())
            for row in preselected:
                value = estimate_prefix_value(
                    tuple(int(op) for op in row["prefix_ids"]),
                    train_x,
                    train_y,
                    holdout_x,
                    holdout_y,
                    args,
                    trace_prior,
                )
                row.update(value)
                value_rollout_evaluations += int(value.get("prefix_value_evaluations", 0))
                row["raw_score_without_prefix_value"] = row["raw_score"]
                row["raw_score"] = row["raw_score"] + args.value_weight * float(value["prefix_value_score"])
                was_active = bool(row.get("survival_activated", False))
                maybe_activate_survival(row, args)
                if bool(row.get("survival_activated", False)) and not was_active:
                    survival_activations += 1
            for row in scored:
                if "prefix_value_score" not in row:
                    row["prefix_value_score"] = None
                    row["prefix_value_evaluations"] = 0

        if args.survival_score_weight != 0.0:
            for row in scored:
                if int(row.get("survival_ttl", 0)) > 0:
                    row["raw_score_without_survival_score"] = row["raw_score"]
                    row["raw_score"] = row["raw_score"] + (
                        args.survival_score_weight * float(row.get("survival_score", 0.0))
                    )

        for row in scored:
            node = graph_nodes.get(row["signature"])
            if node is None or float(row["raw_score"]) > float(node["best_score"]):
                graph_nodes[row["signature"]] = {
                    "signature": row["signature"],
                    "depth": depth,
                    "best_score": row["raw_score"],
                    "best_program_ids": row["program_ids"],
                    "best_prefix_ids": row["prefix_ids"],
                    "train_mse": row["train_mse"],
                    "holdout_mse": row["holdout_mse"],
                    "trajectory_score": row["trajectory_score"],
                    "signature_preview": row["signature_preview"],
                }
            graph_edges.append(
                {
                    "from": row["parent_signature"],
                    "to": row["signature"],
                    "op_id": int(row["op_id"]),
                    "op_block": row["op_block"],
                    "depth": depth,
                }
            )
        scored.sort(key=lambda row: (row["raw_score"], -row["train_mse"]), reverse=True)
        selected = select_diverse(scored, args)
        selected_prefixes = {tuple(int(op) for op in row["prefix_ids"]) for row in selected}
        for row in scored:
            row["selected_next_beam"] = tuple(int(op) for op in row["prefix_ids"]) in selected_prefixes
            if "selection_lane" not in row:
                row["selection_lane"] = None
                row["selection_ignored_signature_limit"] = False
        if args.candidate_jsonl:
            append_jsonl(args.candidate_jsonl, scored)
            candidate_records_written += len(scored)
        beam = [
            BeamItem(
                prefix=tuple(row["prefix_ids"]),
                signature=row["signature"],
                parent_signature=row["parent_signature"],
                score=float(row["diverse_score"]),
                raw_score=float(row["raw_score"]),
                program_ids=tuple(row["program_ids"]),
                survival_ttl=int(row.get("survival_ttl", 0)),
                survival_score=float(row.get("survival_score", 0.0)),
                survival_root=tuple(int(op) for op in row.get("survival_root", [])),
            )
            for row in selected
        ]
        survival_selected = sum(1 for row in selected if int(row.get("survival_ttl", 0)) > 0)
        survival_lane_selected = sum(1 for row in selected if row.get("selection_lane") == "survival")
        survival_selected_total += survival_selected
        late_game_lane_selected = sum(1 for row in selected if row.get("selection_lane") == "late_game")
        late_game_lane_selected_total += late_game_lane_selected
        prefix_stage_lane_selected = sum(1 for row in selected if row.get("selection_lane") == "prefix_stage")
        prefix_stage_lane_selected_total += prefix_stage_lane_selected
        for row in scored:
            exact = row["train_mse"] <= args.stop_mse and row["holdout_mse"] <= args.stop_mse
            if best is None or (row["raw_score"], -row["train_mse"]) > (
                float(best["raw_score"]),
                -float(best["train_mse"]),
            ):
                best = row
            if exact and first_solution is None:
                first_solution = row
        selected_signatures = [row["signature"] for row in selected]
        layer_best = selected[0]
        layers.append(
            {
                "depth": depth,
                "candidates": len(scored),
                "beam_width": len(selected),
                "unique_candidate_signatures": len({row["signature"] for row in scored}),
                "selected_signature_entropy": signature_entropy(selected_signatures),
                "best_raw_score": layer_best["raw_score"],
                "best_diverse_score": layer_best["diverse_score"],
                "best_train_mse": layer_best["train_mse"],
                "best_holdout_mse": layer_best["holdout_mse"],
                "best_signature": layer_best["signature"],
                "best_prefix_ids": layer_best["prefix_ids"],
                "survival_candidates": sum(1 for row in scored if int(row.get("survival_ttl", 0)) > 0),
                "survival_selected": survival_selected,
                "survival_lane_selected": survival_lane_selected,
                "late_game_lane_selected": late_game_lane_selected,
                "prefix_stage_lane_selected": prefix_stage_lane_selected,
                "survival_activated": sum(1 for row in scored if row.get("survival_activated")),
                "selection_lane_counts": {
                    lane: sum(1 for row in selected if row.get("selection_lane") == lane)
                    for lane in sorted({str(row.get("selection_lane", "unknown")) for row in selected})
                },
            }
        )
        print(
            f"depth={depth:02d} candidates={len(scored):04d} "
            f"unique_sig={layers[-1]['unique_candidate_signatures']:04d} "
            f"H={layers[-1]['selected_signature_entropy']:.3f} "
            f"score={layer_best['raw_score']:.4f} "
            f"mse={layer_best['train_mse']:.4g}/{layer_best['holdout_mse']:.4g} "
            f"surv={survival_selected}/{survival_lane_selected} "
            f"prefix_stage={prefix_stage_lane_selected}"
        )
        if first_solution is not None and args.stop_on_solution:
            break

    elapsed = time.perf_counter() - start
    if best is None:
        raise SystemExit("Trajectory search produced no candidates.")
    reference_program = pad_program(tuple(REFERENCE_PROGRAMS[args.target]), args.program_length)
    reference = score_candidate(reference_program, train_x, train_y, holdout_x, holdout_y, args)
    trace_graph = build_trace_graph(graph_nodes, graph_edges, [item.signature for item in beam])
    reconstructed = first_solution or best
    payload = {
        "format_version": 1,
        "prototype": "CodePy tape trajectory-guided beam search",
        "guardrail": "Samples safe DSL execution signatures; no arbitrary Python source is executed.",
        "target": {"name": args.target, "description": TARGET_DESCRIPTIONS[args.target]},
        "trajectory_inference": {
            "objective": "rank P(trace | task) using execution signatures before selecting programs",
            "signature_fields": ["r0", "r1", "ptr", "pc", "p"],
            "seeded_reference_prefixes": args.seed_reference_prefixes,
            "seeded_mode_note": (
                "When enabled, one target/reference prefix is inserted per depth as a reconstruction "
                "diagnostic. It is not evidence of autonomous discovery."
            ),
            "diversity_constraint": {
                "diversity_penalty": args.diversity_penalty,
                "max_same_signature": args.max_same_signature,
                "selection_mode": args.selection_mode,
            },
            "latent_trace_graph": {
                "nodes": len(trace_graph["nodes"]),
                "edges": len(trace_graph["edges"]),
                "selected_signature_entropy": trace_graph["signature_entropy"],
            },
        },
        "search": {
            "method": "trajectory_guided_beam",
            "beam_width": args.beam_width,
            "evaluated_programs": evaluated_programs,
            "value_rollout_evaluations": value_rollout_evaluations,
            "total_evaluated_programs": evaluated_programs + value_rollout_evaluations,
            "candidate_jsonl": args.candidate_jsonl,
            "candidate_records_written": candidate_records_written,
            "trace_prior_json": args.trace_prior_json,
            "trace_prior_weight": args.trace_prior_weight,
            "late_game_ranker_json": args.late_game_ranker_json,
            "late_game_weight": args.late_game_weight,
            "late_game_lane_fraction": args.late_game_lane_fraction,
            "late_game_min_remaining": args.late_game_min_remaining,
            "late_game_gate_depth": late_game_gate_depth,
            "late_game_lane_selected_total": late_game_lane_selected_total,
            "prefix_stage_lane_fraction": args.prefix_stage_lane_fraction,
            "prefix_stage_ignore_signature_limit": args.prefix_stage_ignore_signature_limit,
            "prefix_stage_lane_selected_total": prefix_stage_lane_selected_total,
            "value_rollouts": args.value_rollouts,
            "value_weight": args.value_weight,
            "survival_lane_fraction": args.survival_lane_fraction,
            "survival_ttl": args.survival_ttl,
            "survival_min_value": args.survival_min_value,
            "survival_score_weight": args.survival_score_weight,
            "survival_max_per_root": args.survival_max_per_root,
            "survival_ignore_signature_limit": args.survival_ignore_signature_limit,
            "survival_activations": survival_activations,
            "survival_selected_total": survival_selected_total,
            "elapsed_s": elapsed,
        },
        "layers": layers,
        "best": best,
        "first_solution": first_solution,
        "reconstructed_program": {
            "source": "first_solution_trace" if first_solution is not None else "best_trace_node",
            "signature": reconstructed["signature"],
            "program_ids": reconstructed["program_ids"],
            "compact_program_ids": reconstructed["compact_program_ids"],
            "rendered_program": render_program(np.asarray(reconstructed["program_ids"], dtype=np.int32)),
            "train_mse": reconstructed["train_mse"],
            "holdout_mse": reconstructed["holdout_mse"],
            "trajectory_score": reconstructed["trajectory_score"],
        },
        "reference_program": {
            **reference,
            "rendered_program": render_program(reference_program),
        },
        "trace_graph": trace_graph,
        "block_vocabulary": [{"id": i, "block": block} for i, block in enumerate(TAPE_BLOCKS)],
        "args": vars(args),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CodePy tape trajectory-guided search")
    print(f"  output_json: {args.output_json}")
    print(f"  evaluated_programs: {evaluated_programs}")
    if value_rollout_evaluations:
        print(f"  value_rollout_evaluations: {value_rollout_evaluations}")
        print(f"  total_evaluated_programs: {evaluated_programs + value_rollout_evaluations}")
    if args.survival_ttl > 0 or args.survival_lane_fraction > 0.0:
        print(f"  survival_activations: {survival_activations}")
        print(f"  survival_selected_total: {survival_selected_total}")
    if args.prefix_stage_lane_fraction > 0.0:
        print(f"  prefix_stage_lane_selected_total: {prefix_stage_lane_selected_total}")
    if args.candidate_jsonl:
        print(f"  candidate_jsonl: {args.candidate_jsonl}")
        print(f"  candidate_records_written: {candidate_records_written}")
    print(f"  graph_nodes: {len(trace_graph['nodes'])}")
    print(f"  graph_edges: {len(trace_graph['edges'])}")
    print(f"  best_train_mse: {best['train_mse']:.8g}")
    print(f"  best_holdout_mse: {best['holdout_mse']:.8g}")
    print(f"  first_solution: {first_solution is not None}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory-guided CodePy tape beam search.")
    parser.add_argument("--target", choices=tuple(TARGET_DESCRIPTIONS), default="argmax_index4")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--program-length", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--tape-len", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=4)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--signature-cases", type=int, default=8)
    parser.add_argument("--signature-steps", type=int, default=18)
    parser.add_argument("--diversity-penalty", type=float, default=0.65)
    parser.add_argument("--max-same-signature", type=int, default=1)
    parser.add_argument("--selection-mode", choices=("diverse", "multilane"), default="diverse")
    parser.add_argument("--raw-lane-fraction", type=float, default=0.45)
    parser.add_argument("--prior-lane-fraction", type=float, default=0.2)
    parser.add_argument("--role-lane-fraction", type=float, default=0.25)
    parser.add_argument("--behavior-lane-fraction", type=float, default=0.1)
    parser.add_argument("--mse-weight", type=float, default=0.45)
    parser.add_argument("--length-penalty", type=float, default=1.0e-6)
    parser.add_argument("--value-rollouts", type=int, default=0)
    parser.add_argument("--value-rollout-top-k", type=int, default=64)
    parser.add_argument("--value-rollout-role-top-k", type=int, default=64)
    parser.add_argument("--value-rollout-max-depth", type=int, default=4)
    parser.add_argument("--value-weight", type=float, default=0.15)
    parser.add_argument("--value-lane-fraction", type=float, default=0.0)
    parser.add_argument("--value-rollout-pair-weight", type=float, default=1.0)
    parser.add_argument("--value-rollout-output-weight", type=float, default=1.0)
    parser.add_argument("--value-rollout-role-weight", type=float, default=0.5)
    parser.add_argument("--value-rollout-prior-weight", type=float, default=0.1)
    parser.add_argument("--value-rollout-repeat-len", type=int, default=3)
    parser.add_argument("--value-rollout-ops", type=int, nargs="*", default=[7, 20, 25, 28, 15, 30, 32, 31])
    parser.add_argument("--prefix-stage-lane-fraction", type=float, default=0.0)
    parser.add_argument("--prefix-stage-ignore-signature-limit", action="store_true")
    parser.add_argument("--survival-lane-fraction", type=float, default=0.0)
    parser.add_argument("--survival-ttl", type=int, default=0)
    parser.add_argument("--survival-min-value", type=float, default=0.0)
    parser.add_argument("--survival-score-weight", type=float, default=0.0)
    parser.add_argument("--survival-max-per-root", type=int, default=0)
    parser.add_argument("--survival-ignore-signature-limit", action="store_true")
    parser.add_argument("--stop-mse", type=float, default=1.0e-8)
    parser.add_argument("--stop-on-solution", action="store_true")
    parser.add_argument("--seed-reference-prefixes", action="store_true")
    parser.add_argument("--candidate-jsonl", default=None)
    parser.add_argument("--trace-prior-json", default=None)
    parser.add_argument("--trace-prior-weight", type=float, default=1.0)
    parser.add_argument("--late-game-ranker-json", default=None)
    parser.add_argument("--late-game-weight", type=float, default=0.25)
    parser.add_argument("--late-game-lane-fraction", type=float, default=0.25)
    parser.add_argument("--late-game-min-remaining", type=int, default=2)
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
