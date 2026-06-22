#!/usr/bin/env python3
"""Use a lightweight prior inside guided beam search for CodePy tape programs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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

import jax
import jax.numpy as jnp

from code_tape_prior import (
    load_json,
    predict_probability,
    save_json,
    trace_embedding_from_metrics,
    vectorize_program,
)
from evolve_code_tape import (
    REFERENCE_PROGRAMS,
    TAPE_BLOCKS,
    TARGET_DESCRIPTIONS,
    compact_program,
    evaluate_population,
    make_dataset,
    render_program,
    shaping_args,
)


def safe_log(probability: float) -> float:
    return math.log(max(probability, 1.0e-12))


def scalar(x: jax.Array) -> float:
    return float(jax.device_get(x))


def metric_record(
    program: np.ndarray,
    metrics: Any,
    idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record = {
        "program_ids": [int(op) for op in program],
        "compact_program_ids": compact_program(program),
        "reward": scalar(metrics.reward[idx]),
        "train_mse": scalar(metrics.train_mse[idx]),
        "holdout_mse": scalar(metrics.holdout_mse[idx]),
        "max_abs_error": scalar(metrics.max_abs_error[idx]),
        "holdout_max_abs_error": scalar(metrics.holdout_max_abs_error[idx]),
        "active_blocks": int(jax.device_get(metrics.active_blocks[idx])),
        "shaping_bonus": scalar(metrics.shaping_bonus[idx]),
        "read_steps": scalar(metrics.read_steps[idx]),
        "write_steps": scalar(metrics.write_steps[idx]),
        "move_steps": scalar(metrics.move_steps[idx]),
        "loop_steps": scalar(metrics.loop_steps[idx]),
        "accumulate_steps": scalar(metrics.accumulate_steps[idx]),
        "predicate_steps": scalar(metrics.predicate_steps[idx]),
        "conditional_steps": scalar(metrics.conditional_steps[idx]),
        "max_ptr": scalar(metrics.max_ptr[idx]),
    }
    record["trace_embedding"] = trace_embedding_from_metrics(
        record,
        program_length=args.program_length,
        max_steps=args.max_steps,
        tape_len=args.tape_len,
    )
    return record


def evaluate_programs_np(
    programs: list[np.ndarray],
    train_x: jax.Array,
    train_y: jax.Array,
    holdout_x: jax.Array,
    holdout_y: jax.Array,
    args: argparse.Namespace,
) -> tuple[Any, list[dict[str, Any]]]:
    metrics = evaluate_population(
        jnp.asarray(np.vstack(programs), dtype=jnp.int32),
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
    return metrics, [metric_record(program, metrics, i, args) for i, program in enumerate(programs)]


def score_record(
    record: dict[str, Any],
    model: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    features = vectorize_program(
        record["program_ids"],
        record["trace_embedding"],
        program_length=args.program_length,
        vocab_size=len(TAPE_BLOCKS),
    )
    prior_probability = predict_probability(model, features)
    reward = float(record["reward"])
    train_mse = float(record["train_mse"])
    score = (
        args.prior_weight * safe_log(prior_probability)
        + args.reward_weight * reward
        - args.mse_weight * math.log1p(train_mse)
    )
    return {
        **record,
        "prior_probability": prior_probability,
        "guided_score": score,
        "rendered_program": render_program(np.asarray(record["program_ids"], dtype=np.int32)),
    }


def prefix_to_program(prefix: tuple[int, ...], program_length: int) -> np.ndarray:
    program = list(prefix[:program_length])
    program.extend([0] * max(program_length - len(program), 0))
    return np.asarray(program, dtype=np.int32)


def expansion_ops(prefix: tuple[int, ...], model: dict[str, Any], args: argparse.Namespace) -> list[int]:
    if args.expand_top_k >= len(TAPE_BLOCKS):
        return list(range(len(TAPE_BLOCKS)))
    scored_ops = []
    zero_trace = [0.0] * 10
    for op_id in range(len(TAPE_BLOCKS)):
        program = prefix_to_program((*prefix, op_id), args.program_length)
        features = vectorize_program(
            program.tolist(),
            zero_trace,
            program_length=args.program_length,
            vocab_size=len(TAPE_BLOCKS),
        )
        scored_ops.append((predict_probability(model, features), op_id))
    scored_ops.sort(reverse=True)
    return [op_id for _, op_id in scored_ops[: args.expand_top_k]]


def seed_prefixes(args: argparse.Namespace) -> list[tuple[int, ...]]:
    if args.seed_empty:
        return [()]
    return [(op_id,) for op_id in range(len(TAPE_BLOCKS))]


def run(args: argparse.Namespace) -> None:
    if args.program_length < 1:
        raise SystemExit("--program-length must be >= 1")
    if args.beam_width < 1 or args.expand_top_k < 1:
        raise SystemExit("--beam-width and --expand-top-k must be >= 1")
    prior_payload = load_json(args.prior_json)
    model = prior_payload["model"]
    if int(model["program_length"]) != args.program_length:
        raise SystemExit("Prior program_length does not match search --program-length.")
    if int(model["vocab_size"]) != len(TAPE_BLOCKS):
        raise SystemExit("Prior vocab_size does not match tape block vocabulary.")

    train_x, train_y, holdout_x, holdout_y = make_dataset(args.target, args.cases, args.input_len)
    start = time.perf_counter()
    beam = seed_prefixes(args)
    best: dict[str, Any] | None = None
    first_solution: dict[str, Any] | None = None
    layers: list[dict[str, Any]] = []
    evaluated_programs = 0
    seen_full_programs: set[tuple[int, ...]] = set()

    for depth in range(1 if args.seed_empty else 2, args.program_length + 1):
        candidates: list[tuple[int, ...]] = []
        for prefix in beam:
            if len(prefix) >= depth:
                candidates.append(prefix[:depth])
                continue
            for op_id in expansion_ops(prefix, model, args):
                candidates.append((*prefix, op_id))
        unique_candidates = list(dict.fromkeys(candidates))
        programs = [prefix_to_program(prefix, args.program_length) for prefix in unique_candidates]
        metrics, records = evaluate_programs_np(programs, train_x, train_y, holdout_x, holdout_y, args)
        evaluated_programs += len(programs)
        scored = [
            score_record(
                {
                    **record,
                    "prefix_ids": list(unique_candidates[i]),
                    "depth": depth,
                },
                model,
                args,
            )
            for i, record in enumerate(records)
        ]
        scored.sort(key=lambda row: (row["guided_score"], row["reward"]), reverse=True)
        beam = [tuple(row["prefix_ids"]) for row in scored[: args.beam_width]]

        for row in scored:
            full = tuple(row["program_ids"])
            if full in seen_full_programs:
                continue
            seen_full_programs.add(full)
            if best is None or float(row["reward"]) > float(best["reward"]):
                best = row
            exact = (
                float(row["train_mse"]) <= args.stop_mse
                and float(row["holdout_mse"]) <= args.stop_mse
            )
            if exact and first_solution is None:
                first_solution = row
        layer_best = scored[0]
        layers.append(
            {
                "depth": depth,
                "candidates": len(scored),
                "beam_width": len(beam),
                "best_guided_score": layer_best["guided_score"],
                "best_reward": layer_best["reward"],
                "best_prior_probability": layer_best["prior_probability"],
                "best_train_mse": layer_best["train_mse"],
                "best_holdout_mse": layer_best["holdout_mse"],
                "best_prefix_ids": layer_best["prefix_ids"],
            }
        )
        print(
            f"depth={depth:02d} candidates={len(scored):04d} "
            f"score={layer_best['guided_score']:.4f} "
            f"p={layer_best['prior_probability']:.4f} "
            f"mse={layer_best['train_mse']:.4g}/{layer_best['holdout_mse']:.4g}"
        )
        if first_solution is not None and args.stop_on_solution:
            break

    elapsed = time.perf_counter() - start
    if best is None:
        raise SystemExit("Search produced no candidates.")
    reference = prefix_to_program(tuple(REFERENCE_PROGRAMS[args.target]), args.program_length)
    _, reference_records = evaluate_programs_np(
        [reference], train_x, train_y, holdout_x, holdout_y, args
    )
    reference_record = score_record(reference_records[0], model, args)
    payload = {
        "format_version": 1,
        "prototype": "CodePy tape prior-guided beam search",
        "guardrail": "Prior ranks safe DSL token candidates; every candidate is verified by the bounded tape evaluator.",
        "target": {"name": args.target, "description": TARGET_DESCRIPTIONS[args.target]},
        "prior_json": args.prior_json,
        "search": {
            "method": "prior_guided_beam",
            "beam_width": args.beam_width,
            "expand_top_k": args.expand_top_k,
            "prior_weight": args.prior_weight,
            "reward_weight": args.reward_weight,
            "mse_weight": args.mse_weight,
            "evaluated_programs": evaluated_programs,
            "elapsed_s": elapsed,
        },
        "layers": layers,
        "best": best,
        "first_solution": first_solution,
        "reference_program": reference_record,
        "block_vocabulary": [{"id": i, "block": block} for i, block in enumerate(TAPE_BLOCKS)],
        "args": vars(args),
    }
    save_json(args.output_json, payload)
    print("CodePy tape prior-guided search")
    print(f"  output_json: {args.output_json}")
    print(f"  evaluated_programs: {evaluated_programs}")
    print(f"  best_train_mse: {best['train_mse']:.8g}")
    print(f"  best_holdout_mse: {best['holdout_mse']:.8g}")
    print(f"  first_solution: {first_solution is not None}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodePy tape prior-guided beam search.")
    parser.add_argument("--prior-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--target", choices=tuple(TARGET_DESCRIPTIONS), default="sum4")
    parser.add_argument("--program-length", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--tape-len", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=4)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--expand-top-k", type=int, default=20)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.25)
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
    parser.add_argument("--stop-on-solution", action="store_true")
    parser.add_argument("--seed-empty", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("cpu", "metal", "gpu", "tpu", "auto"),
        default=REQUESTED_BACKEND,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
