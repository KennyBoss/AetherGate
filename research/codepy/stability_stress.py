#!/usr/bin/env python3
"""Stage-5A Stability Layer (Option A) — does a "certified" tape program survive
distribution shift, or did it only learn the training grid?

`certify_tape_program.py` samples from ONE fixed distribution: choice([-2..2],
size=(N, 4)). That extends certification by SAMPLE but not by STRUCTURE. A
program can score acc 1.0 there while secretly depending on:
  * a fixed length (input_len = 4 baked into `p = ptr < input_len`),
  * the value grid {-2..2} (no large / fractional / out-of-range inputs),
  * benign structure (no all-equal ties, no monotone, no single-peak edges).

This harness stresses three invariance axes and reports, per axis, whether the
program's max-abs-error stays exactly 0. It is the honest extension of the
certifier: "fresh" must mean a different distribution, not new points on the
same lattice.

Axes
----
* LENGTH    : evaluate at input_len in {2,3,4,5,6,7,8} (tape_len=8). The decisive
              test of "real bounded loop" vs "unrolled-for-4". For length, the
              program is re-run with that input_len passed to the executor, so a
              genuine `ptr < input_len` loop adapts; a baked constant does not.
* VALUES    : wider grids — large ints, fractional, shifted/negative ranges.
* STRUCTURE : adversarial rows — all-equal (ties), monotone up/down, single-peak
              at first/last index, all-negative, all-positive.

Usage:
    JAX_PLATFORM_NAME=cpu python3 stability_stress.py --target argmax_index4 \
        --program 3,23,4,21,27,26,0,7,15,30,4,0
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


def _run(prog, arr, target, *, tape_len, input_len, max_steps):
    big = jnp.asarray(arr, dtype=jnp.float32)
    y = np.asarray(ect.target_values(big, target))
    ex = ect.execute_programs(prog, big, tape_len=tape_len, input_len=input_len,
                              max_steps=max_steps)
    pred = np.asarray(ex.output[0])
    return float(np.max(np.abs(pred - y))), float(np.mean(np.round(pred) == y))


def axis_length(prog, target, *, tape_len, max_steps, cases, lengths, seeds):
    rows = []
    for L in lengths:
        if L > tape_len:
            continue
        worst_err, worst_acc = 0.0, 1.0
        for s in seeds:
            rng = np.random.default_rng(1000 + s + L)
            arr = rng.choice(np.asarray([-2., -1., 0., 1., 2.], np.float32), size=(cases, L))
            e, a = _run(prog, arr, target, tape_len=tape_len, input_len=L, max_steps=max_steps)
            worst_err = max(worst_err, e); worst_acc = min(worst_acc, a)
        rows.append((f"len={L}", worst_err, worst_acc))
    return rows


def axis_values(prog, target, *, tape_len, input_len, max_steps, cases, seeds):
    grids = {
        "grid[-2..2]": [-2., -1., 0., 1., 2.],
        "ints[-9..9]": list(range(-9, 10)),
        "shifted[3..7]": [3., 4., 5., 6., 7.],
        "frac[-2..2]": [-2., -1.5, -0.5, 0.5, 1.5, 2.],
        "wide[-50..50]": [-50., -17., 0., 13., 50.],
    }
    rows = []
    for name, vals in grids.items():
        worst_err, worst_acc = 0.0, 1.0
        for s in seeds:
            rng = np.random.default_rng(2000 + s)
            arr = rng.choice(np.asarray(vals, np.float32), size=(cases, input_len))
            e, a = _run(prog, arr, target, tape_len=tape_len, input_len=input_len, max_steps=max_steps)
            worst_err = max(worst_err, e); worst_acc = min(worst_acc, a)
        rows.append((name, worst_err, worst_acc))
    return rows


def axis_structure(prog, target, *, tape_len, input_len, max_steps):
    L = input_len
    vals = [-2., -1., 0., 1., 2.]
    cases = []
    for v in vals:
        cases.append([float(v)] * L)                       # all-equal (ties)
    cases.append([float(i) for i in range(L)])             # monotone up
    cases.append([float(L - 1 - i) for i in range(L)])     # monotone down
    peak_first = [-1.] * L; peak_first[0] = 9.; cases.append(peak_first)
    peak_last = [-1.] * L; peak_last[-1] = 9.; cases.append(peak_last)
    cases.append([-2.] * L)                                # all-negative
    cases.append([2.] * L)                                 # all-positive
    arr = np.asarray(cases, np.float32)
    e, a = _run(prog, arr, target, tape_len=tape_len, input_len=input_len, max_steps=max_steps)
    return [("adversarial-structure", e, a)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=tuple(ect.TARGET_DESCRIPTIONS), required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--tape-len", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--input-len", type=int, default=4)
    ap.add_argument("--cases", type=int, default=2000)
    ap.add_argument("--lengths", default="2,3,4,5,6,7,8")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    ids = [int(x) for x in args.program.split(",") if x.strip() != ""]
    prog = jnp.asarray(ect.pad_program(ids, len(ids))[None, :], dtype=jnp.int32)
    lengths = [int(x) for x in args.lengths.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"target : {args.target}")
    print(f"blocks : {[ect.TAPE_BLOCKS[i] for i in ids]}")
    print(f"{'axis case':>22} {'max_abs_err':>12} {'accuracy':>10} {'ok':>4}")

    all_rows = []
    all_rows += axis_length(prog, args.target, tape_len=args.tape_len, max_steps=args.max_steps,
                            cases=args.cases, lengths=lengths, seeds=seeds)
    all_rows += axis_values(prog, args.target, tape_len=args.tape_len, input_len=args.input_len,
                            max_steps=args.max_steps, cases=args.cases, seeds=seeds)
    all_rows += axis_structure(prog, args.target, tape_len=args.tape_len,
                               input_len=args.input_len, max_steps=args.max_steps)

    length_ok = True
    for name, err, acc in all_rows:
        ok = err == 0.0
        if name.startswith("len=") and name != f"len={args.input_len}" and not ok:
            length_ok = False
        print(f"{name:>22} {err:>12.4g} {acc:>10.4f} {('PASS' if ok else 'FAIL'):>4}")

    full = all(err == 0.0 for _, err, _ in all_rows)
    print(f"\n  LENGTH-INVARIANT (all lengths err==0): "
          f"{all(err == 0.0 for n, err, _ in all_rows if n.startswith('len='))}")
    print(f"  FULLY STABLE (every axis err==0)     : {full}")


if __name__ == "__main__":
    main()
