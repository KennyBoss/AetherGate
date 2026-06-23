#!/usr/bin/env python3
"""Verify: seed 7's program on train-distribution holdout (the holdout the
evolution saw — same value set, disjoint seed). If the trace is right, the
corruption should show as nonzero MSE even here. If MSE is zero, something
else is going on."""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util, numpy as np, jax.numpy as jnp

spec = importlib.util.spec_from_file_location(
    "ect", os.path.join(os.path.dirname(__file__), "evolve_code_tape.py"))
ect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ect)

# Seed 7 compact ids from the cert run
COMPACT = [4, 35, 6, 9, 5, 14, 17, 7, 18]

# Test: program_length=9 (as fresh_cert uses) vs 10 (as evolution uses)
for prog_len in [9, 10]:
    P = ect.pad_program(COMPACT, prog_len)
    print(f"\n=== program_length={prog_len}  padded={list(P)} ===")

    # Rich holdout — same distribution as evolution, just different seed
    rng = np.random.default_rng(7 + 9973)  # seed 7, holdout offset
    values = np.asarray([-2., -1., 0., 1., 2.], dtype=np.float32)
    arr = rng.choice(values, size=(128, 2))
    tgt = np.sort(arr, axis=1)

    ex = ect.execute_programs(
        jnp.asarray(P[None, :], dtype=jnp.int32),
        jnp.asarray(arr, jnp.float32),
        tape_len=8, input_len=2, max_steps=24)
    pred = np.array(ex.final_tape[0])[:, :2]
    errors = pred - tgt
    mse = float(np.mean(errors * errors))
    max_err = float(np.max(np.abs(errors)))
    n_wrong = int(np.sum(np.max(np.abs(errors), axis=1) > 1e-6))

    print(f"  Holdout MSE: {mse:.6g}   max_err: {max_err:.6g}   wrong_rows: {n_wrong}/{len(arr)}")

    # Show a few failing rows
    bad = np.where(np.max(np.abs(errors), axis=1) > 1e-6)[0]
    for idx in bad[:5]:
        print(f"    row {idx}: input={arr[idx].tolist()}  pred={pred[idx].tolist()}  tgt={tgt[idx].tolist()}")

# Also test the EXACT evaluation that the evolution used for seed 7:
# evolve_tape_output.py uses rich_cases(seed+1=8) for train and rich_cases(seed+9973=9980) for holdout
print(f"\n=== Exact evolution holdout (seed 7, holdout_seed=7+9973=9980) ===")
rng_ho = np.random.default_rng(7 + 9973)
arr_ho = rng_ho.choice(values, size=(128, 2))
tgt_ho = np.sort(arr_ho, axis=1)

for prog_len in [9, 10]:
    P = ect.pad_program(COMPACT, prog_len)
    ex = ect.execute_programs(
        jnp.asarray(P[None, :], dtype=jnp.int32),
        jnp.asarray(arr_ho, jnp.float32),
        tape_len=8, input_len=2, max_steps=24)
    pred = np.array(ex.final_tape[0])[:, :2]
    errors = pred - tgt_ho
    mse = float(np.mean(errors * errors))
    max_err = float(np.max(np.abs(errors)))
    n_wrong = int(np.sum(np.max(np.abs(errors), axis=1) > 1e-6))
    print(f"  prog_len={prog_len}: MSE={mse:.6g}  max_err={max_err:.6g}  wrong={n_wrong}/128")
