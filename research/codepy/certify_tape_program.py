#!/usr/bin/env python3
"""Certify CodePy tape programs on a FRESH random distribution, not the built-in
32-case stride holdout.

Stage-5A anti-self-deception. `make_dataset` builds train/holdout from a
deterministic stride pattern over {-2,-1,0,1,2}^input_len. A program can hit
holdout_mse=0 by fitting that thin 32-row holdout while still being wrong on the
true input distribution (observed for argmax_index4). The honest certifier is
generalization to many freshly sampled vectors. A program PASSES only if its
max-abs-error is 0 across a large fresh sample AND across several sample seeds.

Usage:
    JAX_PLATFORM_NAME=cpu python3 certify_tape_program.py --target count_positive4 \
        --program 21,7,16,19,18,19
"""
from __future__ import annotations

import argparse
import importlib.util
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

_spec = importlib.util.spec_from_file_location(
    "evolve_code_tape", os.path.join(os.path.dirname(__file__), "evolve_code_tape.py")
)
ect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ect)


def certify(target, program_ids, *, input_len=4, tape_len=8, max_steps=32,
            cases=2000, sample_seeds=(0, 1, 2, 3, 4), values=(-2., -1., 0., 1., 2.)):
    padded = ect.pad_program(program_ids, len(program_ids))
    prog = jnp.asarray(padded[None, :], dtype=jnp.int32)
    worst = 0.0
    worst_acc = 1.0
    for s in sample_seeds:
        rng = np.random.default_rng(s)
        arr = rng.choice(np.asarray(values, np.float32), size=(cases, input_len))
        big = jnp.asarray(arr, dtype=jnp.float32)
        y = np.asarray(ect.target_values(big, target))
        ex = ect.execute_programs(prog, big, tape_len=tape_len,
                                  input_len=input_len, max_steps=max_steps)
        pred = np.asarray(ex.output[0])
        worst = max(worst, float(np.max(np.abs(pred - y))))
        worst_acc = min(worst_acc, float(np.mean(np.round(pred) == y)))
    return {"max_abs_error": worst, "min_accuracy": worst_acc,
            "passes": worst == 0.0, "cases": cases, "seeds": list(sample_seeds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=tuple(ect.TARGET_DESCRIPTIONS), required=True)
    ap.add_argument("--program", required=True, help="comma-separated op ids")
    ap.add_argument("--input-len", type=int, default=4)
    ap.add_argument("--tape-len", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--cases", type=int, default=2000)
    args = ap.parse_args()
    ids = [int(x) for x in args.program.split(",") if x.strip() != ""]
    res = certify(args.target, ids, input_len=args.input_len, tape_len=args.tape_len,
                  max_steps=args.max_steps, cases=args.cases)
    print(f"target            : {args.target}")
    print(f"program blocks    : {[ect.TAPE_BLOCKS[i] for i in ids]}")
    print(f"fresh cases/seed  : {res['cases']}  seeds {res['seeds']}")
    print(f"worst max-abs-err : {res['max_abs_error']}")
    print(f"worst accuracy    : {res['min_accuracy']:.4f}")
    print(f"CERTIFIED (err==0): {res['passes']}")


if __name__ == "__main__":
    main()
