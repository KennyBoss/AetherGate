# CodePy Turing Completeness Plan

CodePy is not yet a general algorithm synthesis system. Current Stage 4
programs are straight-line register/predicate programs with `where`-style
branching. Stage 4B shows hierarchical reuse, but it also shows macro-library
noise at larger depths. General algorithmic scaling needs two additions:

1. macro pruning or regularization, so the library does not grow into a noisy
   action space;
2. bounded stateful control flow, so programs can express loops and memory
   updates.

## Path A: Macro Pruning

Immediate mechanism:

```text
utility = success_rate - penalty * hierarchy_depth
```

Depth is a proxy for library size. A profile is pruned when a deeper profile
does not improve success rate, generation-found, or minimal block count enough
to justify its extra macro cost.

The current offline analysis is implemented in `analyze_code_block_pruning.py`.
It reads Stage 4B results, finds noisy depths, computes a Pareto frontier, and
selects regularized depths under several macro penalties.

Future online mechanism:

```text
for each promoted macro:
    track reuse_count over recent successful programs
    track marginal_fitness_gain vs parent profile
    if reuse_count == 0 for N tasks or marginal gain < threshold:
        remove macro from active profile
```

This should be evaluated by rerunning Stage 4B with pruned profiles, not by
only analyzing old results.

## Path B: Bounded Loops And Memory

The current JAX engine is ready for bounded loops, but not arbitrary unbounded
`while` or unrestricted dynamic `JUMP`.

Safe JAX-compatible control flow:

```text
for step in range(MAX_STEPS):
    execute one opcode or one micro-program block
    update registers, predicate, memory, and program counter
    mask finished lanes
```

This maps naturally to `jax.lax.scan` with fixed `MAX_STEPS`. The program
counter can be a vector rail, and branch/jump blocks can update it with
clipped offsets. This is bounded-control-flow synthesis, not full unbounded
Turing completeness.

Recommended Stage 5A DSL:

```text
scalar registers: r0, r1, r2
predicate: p
memory: fixed small vector mem[K]
pc: integer program counter
budget: MAX_STEPS

blocks:
    pc += 1
    pc += where(p, offset, 1)
    p = r0 > r1
    mem[i] = r0
    r0 = mem[i]
    r0 += 1
    r0 -= 1
    r0 += r1
    r0 = where(p, r1, r0)
```

First algorithmic targets should be smaller than sorting:

```text
count_positive(vector[4])
sum(vector[4])
prefix_max(vector[4])
argmax_index(vector[4])
swap_if_gt(a,b)
sort2(a,b)
sort3(a,b,c)
```

Sorting should come after `swap_if_gt` and `sort3` are reliable. A full sort is
otherwise likely to measure search explosion more than algorithmic capability.

## Current Verdict

- JAX can support CodePy loops if they are bounded by `MAX_STEPS`.
- Arbitrary unbounded `while` is not a good next step for the current SoA/JAX
  evaluator.
- `JUMP` is feasible as a bounded `pc` update inside `lax.scan`.
- External memory is feasible as a fixed-size vector rail.
- General Algorithmic Scaling remains not proven until bounded-loop programs
  solve multi-step stateful tasks across held-out inputs and seeds.

## Implemented Stage 5A Gate

`evolve_code_tape.py` implements the first bounded stateful executor:

```text
state = r0, r1, tape[8], ptr, pc, predicate
execution = jax.lax.scan(..., length=MAX_STEPS)
```

Implemented action blocks include tape reads/writes, pointer movement,
predicate checks, conditional increment, accumulation, and bounded `pc`
updates. The current demos use reference-injected programs to verify state
geometry:

```text
sum4:
    r1 = tape[ptr]
    r0 += r1
    ptr += 1
    pc = 0

count_positive4:
    r1 = tape[ptr]
    p = r1 > 0
    r0 += where(p, 1, 0)
    ptr += 1
    pc = 0
```

Both injected demos verify zero train/holdout error. In addition,
`code-tape-sum-clean-demo` removes reference injection and uses trace-shaped
reward for read/move/loop/accumulate/coverage behavior. The current clean run
autonomously discovers a zero-error `sum4` loop:

```text
r1 = tape[ptr]
r0 += where(p, 1, 0)
r0 += r1
ptr += 1
pc = 0
r0 = tape[ptr]
```

The extra conditional block is inert because the predicate stays false, and
the trailing read is harmless after the accumulation loop. This is a first
autonomous stateful discovery result, but `count_positive4` remains an injected
state-geometry demo until it is found without reference injection.
