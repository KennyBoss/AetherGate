#!/usr/bin/env python3
"""Diagnostic: is sort3 expressible in the CURRENT tape DSL, with no new primitive?

The plan assumed sort3 is "multi-pass; expected to need halt." But a FIXED-length
sort3 is a straight-line sorting network — CE(0,1), CE(1,2), CE(0,1) — and the
compare-exchange atom `sort2(r0,r1)` (block 35) plus tape write-back already exist.
The only thing the failed evolution probe lacked was PROGRAM LENGTH: the network is
~22 ops, but the probe ran program_length=12, so it could not fit.

This hand-builds the 3-comparator network and evaluates it directly. If holdout
tape-MSE == 0, sort3 is expressible NOW (no halt, no new block) and the wall the
evolution hit was search/length, not representation.
"""
from __future__ import annotations

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

# Block ids (see evolve_code_tape.TAPE_BLOCKS):
#   3 r0=tape[ptr]   4 r1=tape[ptr]   5 tape[ptr]=r0   6 tape[ptr]=r1
#   7 ptr+=1         8 ptr-=1         9 ptr=0          35 sort2(r0,r1)
# Compare-exchange CE(i,j) on adjacent cells with ptr at i:
#   load both, sort2, write min->i, max->j.

SORT3_NETWORK = [
    # CE(0,1): ptr starts at 0
    9,            # ptr = 0
    3,            # r0 = tape[0]
    7,            # ptr = 1
    4,            # r1 = tape[1]
    35,           # (r0,r1) = (min,max)
    6,            # tape[1] = max
    8,            # ptr = 0
    5,            # tape[0] = min
    # CE(1,2): move ptr to 1
    7,            # ptr = 1
    3,            # r0 = tape[1]
    7,            # ptr = 2
    4,            # r1 = tape[2]
    35,           # (r0,r1) = (min,max)
    6,            # tape[2] = max
    8,            # ptr = 1
    5,            # tape[1] = min
    # CE(0,1) again: ptr at 1
    4,            # r1 = tape[1]
    8,            # ptr = 0
    3,            # r0 = tape[0]
    35,           # (r0,r1) = (min,max)
    5,            # tape[0] = min
    7,            # ptr = 1
    6,            # tape[1] = max
]


def main():
    prog_len = len(SORT3_NETWORK)
    max_steps = prog_len + 2  # straight-line: one step per op, pc auto-increments
    tape_len = 8
    input_len = 3

    # fresh distribution, including ties / negatives / fractional
    rng = np.random.default_rng(20260623)
    grids = {
        "small_int": (-2., -1., 0., 1., 2.),
        "wide_int": tuple(float(v) for v in range(-50, 51)),
        "fractional": tuple(np.round(np.linspace(-5, 5, 21), 2).tolist()),
    }
    program = jnp.asarray(ect.pad_program(SORT3_NETWORK, prog_len)[None, :], dtype=jnp.int32)

    print(f"sort3 hand-built network: {prog_len} ops, max_steps={max_steps}")
    all_pass = True
    for name, values in grids.items():
        arr = rng.choice(np.asarray(values, np.float32), size=(2000, input_len))
        x = jnp.asarray(arr, jnp.float32)
        y = np.sort(arr, axis=1)
        out = ect.execute_programs(program, x, tape_len=tape_len,
                                   input_len=input_len, max_steps=max_steps)
        pred = np.asarray(out.final_tape[0, :, :input_len])
        max_abs = float(np.max(np.abs(pred - y)))
        ok = max_abs == 0.0
        all_pass = all_pass and ok
        print(f"  grid={name:10s} cases=2000  max_abs_error={max_abs:.6g}  {'PASS' if ok else 'FAIL'}")

    print()
    print("VERDICT:", "sort3 IS expressible in the current DSL (no halt, no new block)"
          if all_pass else "sort3 NOT exactly expressible — representational gap remains")


if __name__ == "__main__":
    main()
