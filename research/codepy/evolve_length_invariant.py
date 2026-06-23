#!/usr/bin/env python3
"""Stage-5A length-invariance training (Option A fix).

`stability_stress.py` showed the "certified" tape programs hold values and
structure but NOT length: they were each evolved on a single input_len (=4) and
break past it. The honest fix is to make evolution face MANY lengths at once, so
a candidate cannot lock onto a single length — fitness is the WORST-length
reward, which only a genuine bounded loop maximizes.

This reuses the validated executor / reward / mutation from `evolve_code_tape.py`
verbatim; the only new thing is the multi-length fitness wrapper and geometry
sized so every trained length fits (tape_len and max_steps large enough for the
longest length's loop). Output programs are then re-checked with
`stability_stress.py` across held-out lengths.
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


def rich_cases(target, cases, input_len, seed, values=(-2., -1., 0., 1., 2.)):
    rng = np.random.default_rng(seed)
    arr = rng.choice(np.asarray(values, np.float32), size=(cases, input_len))
    x = jnp.asarray(arr, dtype=jnp.float32)
    return x, ect.target_values(x, target)


def evaluate_multilength(programs, datasets, length_penalty, shaping, tape_len, max_steps):
    """Reward = worst per-length reward; train_mse/holdout_mse aggregated as max
    over lengths so a candidate must solve EVERY trained length."""
    rewards = []
    train_mses = []
    holdout_mses = []
    for (tx, ty, hx, hy, L) in datasets:
        m = ect.evaluate_population(
            programs, tx, ty, hx, hy, length_penalty, *shaping,
            tape_len, L, max_steps,
        )
        rewards.append(m.reward)
        train_mses.append(m.train_mse)
        holdout_mses.append(m.holdout_mse)
    worst_reward = jnp.min(jnp.stack(rewards, axis=0), axis=0)
    worst_train = jnp.max(jnp.stack(train_mses, axis=0), axis=0)
    worst_holdout = jnp.max(jnp.stack(holdout_mses, axis=0), axis=0)
    return worst_reward, worst_train, worst_holdout


def run(args):
    lengths = [int(x) for x in args.lengths.split(",")]
    if max(lengths) > args.tape_len:
        raise SystemExit("max length must be <= --tape-len")
    vocab = len(ect.TAPE_BLOCKS)
    shaping = (args.shape_read, args.shape_move, args.shape_loop, args.shape_accumulate,
               args.shape_predicate, args.shape_conditional, args.shape_write, args.shape_coverage)

    key = jax.random.PRNGKey(args.seed)
    key, ik = jax.random.split(key)
    programs = ect.init_programs(ik, args.population, args.program_length, vocab)

    # one fresh train+holdout set per length; disjoint seeds
    datasets = []
    for L in lengths:
        tx, ty = rich_cases(args.target, args.cases, L, args.seed + L)
        hx, hy = rich_cases(args.target, args.cases, L, args.seed + L + 9973)
        datasets.append((tx, ty, hx, hy, L))

    print("CodePy length-invariant evolution")
    print(f"target: {args.target}  lengths: {lengths}  tape_len: {args.tape_len}  max_steps: {args.max_steps}")
    print(f"population: {args.population:,}  program_length: {args.program_length}")

    reset_count = int(args.population * args.random_reset_fraction)
    best_worst_holdout = None
    best_program = None
    start = time.perf_counter()
    for gen in range(args.generations + 1):
        key, mk = jax.random.split(key)
        wr, wt, wh = evaluate_multilength(programs, datasets, args.length_penalty, shaping,
                                          args.tape_len, args.max_steps)
        jax.block_until_ready(wr)
        elite_idx = jnp.argsort(wr)[-args.elites:][::-1]
        elites = programs[elite_idx]
        bi = int(jax.device_get(elite_idx[0]))
        cur_wt = float(wt[bi]); cur_wh = float(wh[bi])
        if best_worst_holdout is None or cur_wh < best_worst_holdout:
            best_worst_holdout = cur_wh
            best_program = np.asarray(jax.device_get(programs[bi]), dtype=np.int32)
        if gen == 0 or gen == args.generations or gen % args.log_every == 0:
            print(f"gen={gen:03d} worst_train_mse={cur_wt:.6g} worst_holdout_mse={cur_wh:.6g}")
        if cur_wt <= args.stop_mse and cur_wh <= args.stop_mse:
            print(f"early_stop: worst-length train/holdout MSE <= {args.stop_mse:g}")
            best_program = np.asarray(jax.device_get(programs[bi]), dtype=np.int32)
            break
        if gen < args.generations:
            programs = ect.mutate_elites(mk, elites, population=args.population,
                                         mutation_rate=args.mutation_rate,
                                         random_reset_count=reset_count, vocab_size=vocab)
    dur = time.perf_counter() - start

    ids = [int(o) for o in best_program if int(o) != 0]
    print(f"\nelapsed_s: {dur:.2f}")
    print(f"best worst-length holdout_mse: {best_worst_holdout:.6g}")
    print(f"compact_ids: {','.join(str(i) for i in ids)}")
    print("blocks:")
    for o in best_program:
        if int(o) != 0:
            print(f"  [{int(o):02d}] {ect.TAPE_BLOCKS[int(o)]}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=tuple(ect.TARGET_DESCRIPTIONS), default="argmax_index4")
    ap.add_argument("--lengths", default="2,3,4,5,6")
    ap.add_argument("--population", type=int, default=32768)
    ap.add_argument("--generations", type=int, default=400)
    ap.add_argument("--elites", type=int, default=256)
    ap.add_argument("--program-length", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=48)
    ap.add_argument("--tape-len", type=int, default=8)
    ap.add_argument("--cases", type=int, default=192)
    ap.add_argument("--mutation-rate", type=float, default=0.14)
    ap.add_argument("--random-reset-fraction", type=float, default=0.05)
    ap.add_argument("--length-penalty", type=float, default=1.0e-6)
    ap.add_argument("--shape-read", type=float, default=0.15)
    ap.add_argument("--shape-move", type=float, default=0.15)
    ap.add_argument("--shape-loop", type=float, default=0.15)
    ap.add_argument("--shape-accumulate", type=float, default=0.0)
    ap.add_argument("--shape-predicate", type=float, default=0.25)
    ap.add_argument("--shape-conditional", type=float, default=0.4)
    ap.add_argument("--shape-write", type=float, default=0.0)
    ap.add_argument("--shape-coverage", type=float, default=0.15)
    ap.add_argument("--stop-mse", type=float, default=1.0e-8)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=5)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
