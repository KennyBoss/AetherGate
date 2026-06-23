#!/usr/bin/env python3
"""Stage-5A Option B, step 1 — tape-OUTPUT evolution (the executor upgrade).

Every prior CodePy target read the tape and produced a SCALAR in r0 (sum, count,
argmax index). `swap_if_gt` is the first target whose result is a *transformed
tape* — the data itself is rearranged. This needs the executor to be scored on
its final tape, not on r0.

The executor already maintained the full tape in its scan carry (write-back ops
`tape[ptr] = r0/r1` exist); `evolve_code_tape.execute_programs` now also returns
`final_tape`. This driver reuses that verbatim and only changes the FITNESS to be
MSE over the first `input_len` tape cells against a vector target.

Targets (tape -> tape):
  swap_if_gt2 : sort 2 elements ascending (tape[0] <= tape[1])
  sort3       : sort 3 elements ascending (deferred; needs multi-pass / halt)
  reverse2/3  : reverse the first input_len cells (pure data movement control)

This is a single-pass-friendly step: swap_if_gt2 needs no halt, only the
self-stabilizing semantics already verified. Multi-pass targets (sort3) are
listed but expected to expose the phase-boundary gap — that is where a real
`halt` gets EARNED from a measured failure, not predicted.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp

_spec = importlib.util.spec_from_file_location(
    "evolve_code_tape", os.path.join(os.path.dirname(__file__), "evolve_code_tape.py")
)
ect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ect)


TAPE_TARGETS = {
    "swap_if_gt2": "sort 2 elements ascending (swap if tape[0] > tape[1])",
    "sort3": "sort 3 elements ascending (multi-pass; expected to need halt)",
    "reverse3": "reverse the first input_len cells (pure data movement)",
}


def tape_target(vectors: np.ndarray, target: str) -> np.ndarray:
    if target in ("swap_if_gt2", "sort3"):
        return np.sort(vectors, axis=1)
    if target == "reverse3":
        return vectors[:, ::-1]
    raise ValueError(target)


def rich_cases(target, cases, input_len, seed, values=(-2., -1., 0., 1., 2.)):
    rng = np.random.default_rng(seed)
    arr = rng.choice(np.asarray(values, np.float32), size=(cases, input_len))
    y = tape_target(arr, target)
    return jnp.asarray(arr, jnp.float32), jnp.asarray(y, jnp.float32)


def evaluate_tape(programs, train_x, train_y, holdout_x, holdout_y,
                  *, tape_len, input_len, max_steps, length_penalty, shaping):
    tr = ect.execute_programs(programs, train_x, tape_len=tape_len,
                              input_len=input_len, max_steps=max_steps)
    ho = ect.execute_programs(programs, holdout_x, tape_len=tape_len,
                              input_len=input_len, max_steps=max_steps)
    # score only the first input_len cells of the final tape
    tr_pred = tr.final_tape[:, :, :input_len]              # (pop, cases, L)
    ho_pred = ho.final_tape[:, :, :input_len]
    tr_err = tr_pred - train_y[None, :, :]
    ho_err = ho_pred - holdout_y[None, :, :]
    train_mse = jnp.mean(tr_err * tr_err, axis=(1, 2))
    holdout_mse = jnp.mean(ho_err * ho_err, axis=(1, 2))
    active = jnp.sum(programs != 0, axis=1)
    # light behavior shaping: reward actually writing to the tape + moving
    write_score = jnp.clip(tr.write_steps / max(input_len, 1), 0.0, 1.0)
    move_score = jnp.clip(tr.move_steps / max(input_len, 1), 0.0, 1.0)
    read_score = jnp.clip(tr.read_steps / max(input_len, 1), 0.0, 1.0)
    cond_score = jnp.clip(tr.conditional_steps / max(input_len, 1), 0.0, 1.0)
    shaping_bonus = (shaping["write"] * write_score + shaping["move"] * move_score
                     + shaping["read"] * read_score + shaping["cond"] * cond_score)
    reward = -jnp.log1p(train_mse) + shaping_bonus - length_penalty * active
    return reward, train_mse, holdout_mse


def run(args):
    vocab = len(ect.TAPE_BLOCKS)
    shaping = {"write": args.shape_write, "move": args.shape_move,
               "read": args.shape_read, "cond": args.shape_conditional}
    key = jax.random.PRNGKey(args.seed)
    key, ik = jax.random.split(key)
    programs = ect.init_programs(ik, args.population, args.program_length, vocab)

    tx, ty = rich_cases(args.target, args.cases, args.input_len, args.seed + 1)
    hx, hy = rich_cases(args.target, args.cases, args.input_len, args.seed + 9973)

    print("CodePy tape-OUTPUT evolution")
    print(f"target: {args.target} ({TAPE_TARGETS[args.target]})")
    print(f"population: {args.population:,}  program_length: {args.program_length}  "
          f"input_len: {args.input_len}  tape_len: {args.tape_len}  max_steps: {args.max_steps}")

    reset_count = int(args.population * args.random_reset_fraction)
    best_h = None
    best_prog = None
    start = time.perf_counter()
    for gen in range(args.generations + 1):
        key, mk = jax.random.split(key)
        reward, tmse, hmse = evaluate_tape(
            programs, tx, ty, hx, hy, tape_len=args.tape_len, input_len=args.input_len,
            max_steps=args.max_steps, length_penalty=args.length_penalty, shaping=shaping)
        jax.block_until_ready(reward)
        elite_idx = jnp.argsort(reward)[-args.elites:][::-1]
        elites = programs[elite_idx]
        bi = int(jax.device_get(elite_idx[0]))
        ct, ch = float(tmse[bi]), float(hmse[bi])
        if best_h is None or ch < best_h:
            best_h = ch
            best_prog = np.asarray(jax.device_get(programs[bi]), dtype=np.int32)
        if gen == 0 or gen == args.generations or gen % args.log_every == 0:
            print(f"gen={gen:03d} train_mse={ct:.6g} holdout_mse={ch:.6g}")
        if ct <= args.stop_mse and ch <= args.stop_mse:
            print(f"early_stop: train/holdout tape-MSE <= {args.stop_mse:g}")
            best_prog = np.asarray(jax.device_get(programs[bi]), dtype=np.int32)
            break
        if gen < args.generations:
            programs = ect.mutate_elites(mk, elites, population=args.population,
                                         mutation_rate=args.mutation_rate,
                                         random_reset_count=reset_count, vocab_size=vocab)
    dur = time.perf_counter() - start

    ids = [int(o) for o in best_prog if int(o) != 0]
    print(f"\nelapsed_s: {dur:.2f}")
    print(f"best holdout tape-MSE: {best_h:.6g}")
    print(f"compact_ids: {','.join(str(i) for i in ids)}")
    print("blocks:")
    for o in best_prog:
        if int(o) != 0:
            print(f"  [{int(o):02d}] {ect.TAPE_BLOCKS[int(o)]}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=tuple(TAPE_TARGETS), default="swap_if_gt2")
    ap.add_argument("--population", type=int, default=16384)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--elites", type=int, default=256)
    ap.add_argument("--program-length", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--tape-len", type=int, default=8)
    ap.add_argument("--input-len", type=int, default=2)
    ap.add_argument("--cases", type=int, default=128)
    ap.add_argument("--mutation-rate", type=float, default=0.16)
    ap.add_argument("--random-reset-fraction", type=float, default=0.05)
    ap.add_argument("--length-penalty", type=float, default=1.0e-6)
    ap.add_argument("--shape-read", type=float, default=0.15)
    ap.add_argument("--shape-move", type=float, default=0.2)
    ap.add_argument("--shape-write", type=float, default=0.3)
    ap.add_argument("--shape-conditional", type=float, default=0.25)
    ap.add_argument("--stop-mse", type=float, default=1.0e-8)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=41)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
