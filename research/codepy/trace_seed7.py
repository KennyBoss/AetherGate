#!/usr/bin/env python3
"""Step-by-step trace of seed 7's program on counterexample inputs.
Program: [4, 35, 6, 9, 5, 14, 17, 7, 18]
          r1=tape[ptr] | sort2(r0,r1) | tape[ptr]=r1 | ptr=0 | tape[ptr]=r0 |
          p=r1>0 | r0=where(p,r1,r0) | ptr+=1 | pc=0
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
import numpy as np
import jax
import jax.numpy as jnp

spec = importlib.util.spec_from_file_location(
    "ect", os.path.join(os.path.dirname(__file__), "evolve_code_tape.py")
)
ect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ect)

# Seed 7 compact ids
COMPACT_IDS = [4, 35, 6, 9, 5, 14, 17, 7, 18]
PROGRAM_LENGTH = 10
TAPE_LEN = 8
INPUT_LEN = 2
MAX_STEPS = 24

# Counterexamples that fail
TEST_CASES = [
    ("(-50, 50) — should be (-50, 50)", [-50.0, 50.0]),
    ("(50, -50) — should be (-50, 50)", [50.0, -50.0]),
    ("(-50, -1, 0, 1, 50) — long vector", [-50.0, -1.0, 0.0, 1.0, 50.0, 0.0, 0.0, 0.0]),
    # Control: train-distribution case that passes
    ("(2, -1) — should be (-1, 2)", [2.0, -1.0]),
    ("(-2, 0) — should be (-2, 0)", [-2.0, 0.0]),
]

def render_op(op_id):
    if op_id < len(ect.TAPE_BLOCKS):
        return ect.TAPE_BLOCKS[op_id]
    return f"???{op_id}"

# Build padded program
program_ids = list(COMPACT_IDS)
program_ids.extend([0] * (PROGRAM_LENGTH - len(program_ids)))
P = np.asarray(program_ids, dtype=np.int32)

print("=" * 90)
print(f"SEED 7 AUTOPSY: compact_ids={COMPACT_IDS}")
print(f"padded: {program_ids}")
print("Rendered:")
for i, op in enumerate(program_ids):
    print(f"  {i:02d}: [{op:02d}] {render_op(op)}")
print("=" * 90)

def trace_execution(vectors_list, label, max_trace_steps=30):
    """Execute and print step-by-step trace."""
    vectors_np = np.asarray(vectors_list, dtype=np.float32)
    arr = jnp.asarray(vectors_np[None, :], dtype=jnp.float32)
    tgt = np.sort(vectors_np)

    # Manually trace the scan
    population, cases = 1, 1
    padded = jnp.zeros((cases, TAPE_LEN), dtype=jnp.float32)
    padded = padded.at[:, :INPUT_LEN].set(arr)
    tape = padded  # (1, 8)
    r0 = jnp.zeros((1,), dtype=jnp.float32)
    r1 = jnp.zeros((1,), dtype=jnp.float32)
    ptr = jnp.zeros((1,), dtype=jnp.int32)
    pc = jnp.zeros((1,), dtype=jnp.int32)
    predicate = jnp.zeros((1,), dtype=jnp.bool_)

    program = jnp.asarray(P[None, :], dtype=jnp.int32)  # (1, 10)

    print(f"\n{'─' * 80}")
    print(f"TRACE: {label}")
    print(f"Input: {vectors_np}    Target: {tgt.tolist()}")
    print(f"{'─' * 80}")
    print(f"{'Step':<5} {'pc':<4} {'op':<30} {'ptr':<4} {'r0':>10} {'r1':>10} {'p':>6} tape")
    print(f"{'─' * 80}")

    for step in range(max_trace_steps):
        pc_val = int(pc[0])
        op_id = int(program[0, pc_val])
        ptr_val = int(ptr[0])
        r0_val = float(r0[0])
        r1_val = float(r1[0])
        p_val = bool(predicate[0])
        tape_vals = [float(tape[0, i]) for i in range(min(INPUT_LEN + 2, TAPE_LEN))]

        print(f"{step:<5} {pc_val:<4} [{op_id:02d}] {render_op(op_id):<25} {ptr_val:<4} {r0_val:>10.1f} {r1_val:>10.1f} {str(p_val):>6} {tape_vals}")

        # Execute one step (manually replicate body)
        cell = tape[0, ptr_val]
        old_r0 = r0
        old_r1 = r1
        old_ptr = ptr
        old_pc = pc
        old_predicate = predicate
        old_tape = tape

        # Same logic as execute_programs body
        r0 = jnp.where(op_id == 1, jnp.zeros_like(r0), r0)
        r1 = jnp.where(op_id == 2, jnp.zeros_like(r1), r1)
        r0 = jnp.where(op_id == 3, cell, r0)
        r1 = jnp.where(op_id == 4, cell, r1)

        one_hot_ptr = jax.nn.one_hot(old_ptr, TAPE_LEN, dtype=tape.dtype)
        tape_r0 = old_tape * (1.0 - one_hot_ptr) + old_r0[..., None] * one_hot_ptr
        tape_r1 = old_tape * (1.0 - one_hot_ptr) + old_r1[..., None] * one_hot_ptr
        tape = jnp.where(op_id == 5, tape_r0, tape)
        tape = jnp.where(op_id == 6, tape_r1, tape)

        ptr = jnp.where(op_id == 7, old_ptr + 1, ptr)
        ptr = jnp.where(op_id == 8, old_ptr - 1, ptr)
        ptr = jnp.where(op_id == 9, jnp.zeros_like(ptr), ptr)
        ptr = jnp.clip(ptr, 0, TAPE_LEN - 1)

        r0 = jnp.where(op_id == 10, old_r0 + old_r1, r0)
        r0 = jnp.where(op_id == 11, old_r0 + 1.0, r0)
        r0 = jnp.where(op_id == 12, old_r0 - 1.0, r0)
        r0 = jnp.where(op_id == 23, old_ptr.astype(jnp.float32), r0)
        r1 = jnp.where(op_id == 24, old_ptr.astype(jnp.float32), r1)
        r0 = jnp.where(op_id == 32, old_r1, r0)
        r1 = jnp.where(op_id == 33, old_r0, r1)

        predicate = jnp.where(op_id == 13, old_r0 > 0.0, predicate)
        predicate = jnp.where(op_id == 14, old_r1 > 0.0, predicate)
        predicate = jnp.where(op_id == 15, old_ptr < INPUT_LEN, predicate)
        predicate = jnp.where(op_id == 20, cell > old_r0, predicate)
        predicate = jnp.where(op_id == 21, cell > old_r1, predicate)
        predicate = jnp.where(op_id == 22, old_ptr < INPUT_LEN - 1, predicate)

        r0 = jnp.where(op_id == 16, old_r0 + old_predicate.astype(jnp.float32), r0)
        r0 = jnp.where(op_id == 17, jnp.where(old_predicate, old_r1, old_r0), r0)
        r0 = jnp.where(op_id == 25, jnp.where(old_predicate, cell, old_r0), r0)
        r1 = jnp.where(op_id == 26, jnp.where(old_predicate, cell, old_r1), r1)
        r0 = jnp.where(op_id == 27, jnp.where(old_predicate, old_ptr.astype(jnp.float32), old_r0), r0)
        r1 = jnp.where(op_id == 28, jnp.where(old_predicate, old_ptr.astype(jnp.float32), old_r1), r1)
        r1 = jnp.where(op_id == 34, jnp.where(old_predicate, old_r0, old_r1), r1)
        r0 = jnp.where(op_id == 35, jnp.minimum(old_r0, old_r1), r0)
        r1 = jnp.where(op_id == 35, jnp.maximum(old_r0, old_r1), r1)

        default_pc = jnp.minimum(old_pc + 1, PROGRAM_LENGTH - 1)
        pc = default_pc
        pc = jnp.where(op_id == 18, jnp.zeros_like(pc), pc)
        pc = jnp.where(op_id == 19, jnp.where(old_predicate, 0, default_pc), pc)
        pc = jnp.where(op_id == 29, jnp.full_like(pc, jnp.minimum(3, PROGRAM_LENGTH - 1)), pc)
        pc = jnp.where(op_id == 30,
                       jnp.where(old_predicate, jnp.minimum(3, PROGRAM_LENGTH - 1), default_pc), pc)
        pc = jnp.where(op_id == 31, old_pc, pc)
        pc = jnp.clip(pc, 0, PROGRAM_LENGTH - 1)

        r0 = jnp.clip(r0, -ect.CLIP_VALUE, ect.CLIP_VALUE)
        r1 = jnp.clip(r1, -ect.CLIP_VALUE, ect.CLIP_VALUE)
        tape = jnp.clip(tape, -ect.CLIP_VALUE, ect.CLIP_VALUE)

    # Final state
    final_tape = [float(tape[0, i]) for i in range(INPUT_LEN)]
    max_err = max(abs(final_tape[i] - tgt[i]) for i in range(INPUT_LEN))
    print(f"{'─' * 80}")
    print(f"Final tape[:2]: {final_tape}    Target: {tgt.tolist()}    max_err: {max_err:.1f}")

    # Also show the official result via execute_programs for verification
    P_full = ect.pad_program(COMPACT_IDS, PROGRAM_LENGTH)
    ex = ect.execute_programs(jnp.asarray(P_full[None, :], dtype=jnp.int32),
                               arr, tape_len=TAPE_LEN, input_len=INPUT_LEN, max_steps=MAX_STEPS)
    official_pred = np.array(ex.final_tape[0])[:, :INPUT_LEN]
    print(f"Official (via execute_programs): {official_pred.tolist()}")
    return max_err

for label, vec in TEST_CASES:
    trace_execution(vec, label)
