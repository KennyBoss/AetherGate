#!/usr/bin/env python3
"""Measure-first: can the CodePy block set EXPRESS sort3 at all?

Hand-build the 3-comparator sorting network (0,1)(1,2)(0,1) from TAPE_BLOCKS
and check it drives tape-MSE to ~0. If yes -> evolution's plateau is a SEARCH
failure, not an expressibility gap.

Each comparator on tape cells (i,j):  load both into r0/r1, sort2, write back.
"""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import importlib.util, sys
sys.path.insert(0, "/Users/makbuk/000cor/research/codepy")
import numpy as np
import jax.numpy as jnp

spec = importlib.util.spec_from_file_location(
    "ect", "/Users/makbuk/000cor/research/codepy/evolve_code_tape.py")
ect = importlib.util.module_from_spec(spec); spec.loader.exec_module(ect)

# comparator (0,1): ptr=0; r0=tape[0]; ptr+=1; r1=tape[1]; sort2; tape[1]=r1; ptr-=1; tape[0]=r0
C01 = [9, 3, 7, 4, 35, 6, 8, 5]
# comparator (1,2): ptr+=1(->1); r0=tape[1]; ptr+=1(->2); r1=tape[2]; sort2; tape[2]=r1; ptr-=1; tape[1]=r0
C12 = [7, 3, 7, 4, 35, 6, 8, 5]
# comparator (0,1) again, ptr currently 1: ptr-=1(->0); r0=tape[0]; ptr+=1; r1=tape[1]; sort2; tape[1]=r1; ptr-=1; tape[0]=r0
C01b = [8, 3, 7, 4, 35, 6, 8, 5]

prog = C01 + C12 + C01b
print(f"program length: {len(prog)} ops")
for o in prog:
    print(f"  [{o:02d}] {ect.TAPE_BLOCKS[o]}")

programs = jnp.asarray(np.array([prog], dtype=np.int32))

rng = np.random.default_rng(7)
x = rng.choice(np.array([-2., -1., 0., 1., 2.], np.float32), size=(256, 3))
y = np.sort(x, axis=1)
xj = jnp.asarray(x, jnp.float32)

res = ect.execute_programs(programs, xj, tape_len=8, input_len=3, max_steps=len(prog))
pred = np.asarray(res.final_tape[0, :, :3])
mse = float(np.mean((pred - y) ** 2))
exact = int(np.sum(np.all(np.isclose(pred, y, atol=1e-6), axis=1)))
print(f"\ntape-MSE: {mse:.8g}")
print(f"exactly sorted: {exact}/256 triples")
print("sample:")
for k in range(4):
    print(f"  in={x[k]}  ->  out={pred[k]}  (want {y[k]})")
