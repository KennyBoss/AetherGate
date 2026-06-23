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

**Now empirically load-bearing, not just hygiene.** The skill-compression line
hit this exact wall from the other side. `skill_compression/ecs_stage1b_selection.py`
(MODEL_THESIS §6d) fixed skill *selection* with best-first search, then found that
an **unpruned library is net-negative**: at fixed everything-else, eval depth 3,
a library cap of `kmax=8` *loses* to a no-skill baseline by +12 forward-passes,
while `kmax=2` *wins* by −4.7 (4/5 seeds). The branching cost of macros grows
~linearly with library size while path-saving saturates, so library-size control
is the difference between compression and a net slowdown. Worse, the promoted
macros there were solution-path n-grams (`cea`, `ceac`) straddling subroutine
boundaries, not the generative subroutines (`cac`, `bbd`, …) — so pruning by size
is necessary but not sufficient; promotion must also recover the *true* procedures.
That is the same "macro-library noise at larger depths" this plan opened with,
now measured as a cost regression. Path A is the shared fix for both lines.

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

## CodePy ↔ MODEL_THESIS: storage emerging, selection unresolved, compression unproven

The tempting one-liner — "CodePy solves storage, ECS must solve selection" —
overstates what is measured. The honest three-line status is:

```text
storage is emerging,
selection is unresolved,
compression remains unproven.
```

**Storage is emerging, not solved.** What is demonstrated:

```text
operations -> macros -> small loops -> sum4
```

What is NOT yet demonstrated is that the library grows into an increasingly
powerful set of procedures:

```text
sum4 -> count_positive -> argmax -> sort3 -> new algorithms
```

The library exists, but its ability to become a more capable procedure set is
unshown. Autonomously (no reference injection) only `sum4` is found, and even
that carries two inert blocks; `count_positive4` is still an injected demo. So:

> CodePy demonstrates executable procedural storage and limited autonomous
> algorithm discovery. However, procedural storage alone does not yield
> compression. The unresolved bottleneck is goal-conditioned selection and
> composition of procedures under distribution shift.

**Selection is the harder half, and Stage 1a already proved it.** Splitting the
roles cleanly:

```text
CodePy answers: "how to store a procedure?"
ECS    answers: "when to invoke it?"
```

`MODEL_THESIS.md` §6c ran exactly the naive build — growing library + trained
per-step gate — and returned **PASS = False**. The diagnosis was not noise: with
unreliable selection the gate either

```text
abstains  -> collapses to dense behavior (skills add nothing)
misfires  -> solve-rate drops (skills hurt reliability)
```

This is a *valuable* negative. It answers, with 6 seeds and `protocol_valid`,
the question "is it enough to just grow the library and train a small gate?" —
answer: **no**. That removes a plausible year of wandering down the
grow-the-library direction. The same residual (goal-conditioned inverse) that
the 3.3–3.4 structural arc isolated is now also the empirical blocker of Stage 1
— the two lines meet on one open problem.

**Compression remains unproven.** Even a perfect selector over a perfect library
would still have to survive the two impostor controls before "compression" is
earned, not "caching":

- `cache-M` — a library indexed by exact inputs is a *function table*. A library
  of named programs without a learned, composing selector is rejected by this
  control by construction.
- Axis-C — the procedure must be invoked inside a *novel composition* unseen at
  storage time, with the active-compute drop holding there. Stage 1a showed no
  Axis-C transfer (ECS 14.44 ≈ dense 14.13).

Until a found procedure is selected and composed on held-out Axis-S/Axis-C
splits with an active-compute drop at matched loss, CodePy is procedural storage,
not compressed experience.

### A radical hypothesis to earn, not assume

If selection is the harder half, the architecture may invert: the dense network
need not store the skills at all, only the machinery to choose, compose, and
adapt them.

```text
LLM    = selector / composer / adapter
CodePy = procedural memory
```

Under G7 this stays a **hypothesis with an earned predictor**, not a premise. It
is admitted only if it makes a falsifiable structural prediction and that test
fires. Candidate test: as the procedure library grows, a model carrying the
*selection* mechanism should show active-compute on held-out Axis-C *decreasing*
while its parameter count stays flat — weights spent on dispatch, not on
re-approximating each procedure. If instead matched performance still requires
the weights to re-encode the procedures (frozen-backbone ablation kills the
drop), the inversion is false for our system and "LLM = selector" stays out of
the conclusions. Do not assume the inversion — measure it.
