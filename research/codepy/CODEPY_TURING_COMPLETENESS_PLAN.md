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
autonomous stateful discovery result.

### Stage 5A ladder — current frontier (autonomous, fresh-distribution certified)

The built-in train/holdout is a thin 32-case stride pattern; a program can hit
`holdout_mse=0` while still being wrong on the true input distribution. So
discoveries are re-certified by `certify_tape_program.py` on **2,000 freshly
sampled vectors × 5 sample seeds** — PASS only if max-abs-error is exactly 0
everywhere. Status:

| target | autonomous (no injection) | built-in holdout | fresh cert (2k×5) | verdict |
|---|---|---|---|---|
| `sum4` | yes | mse 0 | err 0, acc 1.0 | **CERTIFIED** |
| `count_positive4` | yes, 8/8 seeds | mse 0 | err 0, acc 1.0 | **CERTIFIED** |
| `argmax_index4` | yes, 5/5 seeds (rich data) | mse 0 | err 0, acc 1.0 | **CERTIFIED** |

`count_positive4` is now an autonomous, fresh-certified discovery — it is no
longer an injected state-geometry demo. The clean program (seed 41) is a real
predicate-accumulate loop that generalizes to count∈{0..4} including all-negative
(count 0) and all-positive (count 4) cases absent from the training stride:

```text
p = tape[ptr] > r1      # r1 stays 0, so this is tape[ptr] > 0
ptr += 1
r0 += where(p, 1, 0)
pc = where(p, 0, pc + 1)
pc = 0                  # unconditional loop back
```

`argmax_index4` — the first **two-variable** target (track the running max AND
remember its index) — is now also autonomously discovered and fresh-certified,
but only after a dataset fix. It first looked solved on seed 41 (`holdout_mse=0`)
yet scored just **90% on the fresh distribution**: with only 32 fixed train rows,
evolution overfit the holdout instead of learning argmax. The fix is the
`--rich-data` flag, which trains on many freshly sampled vectors so that fitting
train is ≈ generalizing; under it, 5/5 seeds reach fresh-distribution accuracy
1.0. A representative certified program shows the genuine coupled update:

```text
r1 = tape[ptr]
p  = tape[ptr] > r1          # current cell beats running max?
r0 = where(p, ptr, r0)       # capture the INDEX when it does
r1 = where(p, tape[ptr], r1) # advance the running MAX
ptr += 1
p  = ptr < input_len
pc = where(p, 3, pc + 1)     # loop while in-bounds, else fall through
```

This is the cautionary lesson made concrete: the built-in 32-case holdout is a
necessary-not-sufficient gate, and fresh-distribution certification
(`certify_tape_program.py`) is what actually moves a rung from "looks solved" to
"solved." With it, the Stage 5A ladder now stands certified through argmax:
**operations → macros → small loops → sum4 → count_positive4 → argmax_index4**.

### Stability Layer (Option A) — certification was over-claimed; the real axis is LENGTH

`certify_tape_program.py` samples 2k×5 fresh vectors but always from ONE
structure: `input_len = 4`, values in {-2..2}. That extends certification by
*sample*, not by *structure*. `stability_stress.py` stresses three invariance
axes (length ∈ {2..8}, value grids incl. {-50..50} and fractional, adversarial
structure incl. ties / monotone / single-peak). Result on the three "certified"
programs:

| program | VALUES axis | STRUCTURE axis | LENGTH axis | length-invariant? |
|---|---|---|---|---|
| `sum4` | all PASS | all PASS | holds len 2–6, fails ≥7 | **No** |
| `count_positive4` | all PASS | all PASS | holds len 2–6, fails ≥7 | **No** |
| `argmax_index4` | all PASS | all PASS | holds len 2–4, fails ≥5 | **No** |

So the table above is honest only at `input_len = 4`. Values and structure are
robust everywhere (the algorithms are not grid- or tie-fragile); the single
failing axis is **length** — exactly the "real bounded loop vs unrolled-for-4"
discriminator. Two distinct failure modes, separated by re-running with larger
tape geometry (`tape_len=16, max_steps=64`):

- **`count_positive4` is a genuine bounded loop.** Its length ceiling *scales with
  geometry*: at `tape_len=16, max_steps=64` it holds to **len 12** (was 6). The
  bound is `max_steps`, not a baked constant — give it enough steps for the length
  and it generalizes. This is the strongest stateful result so far.
- **`argmax_index4` is the most fragile** (fails at len 5), consistent with the
  selection pressure that produced it: it was evolved only on length-4 vectors,
  so its loop mechanics never had to generalize past 4.
- **`sum4`'s apparent failure under changed geometry is partly a harness artifact:**
  running ONE fixed program (evolved under `tape_len=8/max_steps=32`) inside a
  *different* geometry changes the very termination conditions it was selected for.
  The honest length-invariance test is to *re-train with varying input length*,
  not to replay an old program in new geometry.

**Verdict.** "CERTIFIED ×3" was an over-claim scoped to `input_len=4`. The
corrected status: **value- and structure-robust at length 4; not yet
length-invariant.** Fixing this (the named next step, agreed: bring argmax — and
ideally all three — to true length-invariance) requires training with **mixed
input lengths** so evolution cannot lock onto a single length, then re-certifying
across lengths with geometry sized to the length. Only after that does Option B
(structured outputs: `swap_if_gt` → conditional 3-permutation) begin, on a base
proven to be a loop rather than a length-4 pattern.

### Length-invariance achieved (the Option-A fix, all three rungs)

`evolve_length_invariant.py` trains each candidate against **several lengths at
once** and scores it by its WORST-length reward, so only a genuine bounded loop
maximizes fitness. `stability_stress.py` then certifies on **held-out lengths**
(not in training) plus the value and structure axes. Result — every rung is now
**FULLY STABLE** (max-abs-error 0 on every axis), 5/5 seeds reaching worst-length
holdout MSE 0:

| target | trained lengths | held-out lengths | values | structure | verdict |
|---|---|---|---|---|---|
| `sum4` | {2,4,6} | {3,5,7,8,9,10} | PASS | PASS | **LENGTH-INVARIANT** |
| `count_positive4` | {2,4,6} | {3,5,7,8,9,10} | PASS | PASS | **LENGTH-INVARIANT** |
| `argmax_index4` | {2,3,4,5,6} | {7,8} | PASS | PASS | **LENGTH-INVARIANT** |

The discovered argmax loop is genuinely self-terminating — it uses
`p = ptr < input_len` to drive a conditional `pc`, so it iterates exactly
`input_len` times regardless of length:

```text
p  = ptr < input_len          # in-bounds?
r1 = tape[ptr]
r0 = where(p, ptr, r0)        # provisional index
p  = tape[ptr] > r1
ptr += 1
p  = tape[ptr] > r1           # current beats running max?
r1 = where(p, tape[ptr], r1)  # advance running max
pc = where(p, 0, pc + 1)      # loop control
pc = 3
...
```

**One honest boundary caveat, surfaced by the stress layer.** The `sum`/`count`
loops use an *unconditional* `pc = 0` and lean on `ptr` clipping at `tape_len-1`
for termination, so they fail at exactly `input_len == tape_len` (the tape-edge
clip over-runs by one). Every held-out length **strictly below** `tape_len`
passes; the failure is a DSL-geometry artifact (no predicate-gated `halt`), not
an algorithm error. argmax avoids it by self-terminating on `ptr < input_len`.
This is logged, not hidden: it is the next concrete DSL gap (wire a
predicate-conditional `halt`) and it does not affect the length-invariance result
within `input_len < tape_len`.

**Ladder status (corrected, honest).** `sum4`, `count_positive4`, `argmax_index4`
are autonomous, fresh-distribution **and length-invariant** (value/structure
stable, generalizing to held-out lengths < tape_len). The base is now proven to
be *loops*, not length-4 patterns — Option B (structured outputs) can build on it:
`swap_if_gt` (2 elements; first target whose result is a transformed *tape*, not
a scalar `r0` — requires extending the executor's output from `r0` to the tape)
→ conditional 3-permutation → only then `sort`.

### Self-stabilizing, NOT self-halting — the computation-semantics gap before Option B

A reviewer flagged that the loops might be "bounded loops with an external safety
rail" rather than real computation. Checked empirically (sweep `max_steps` on the
length-invariant programs, all else fixed):

| max_steps | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| argmax LI length-invariant | ✗ | ✗ | ✓ | ✓ | ✓ |
| count LI length-invariant | ✗ | ✗ | ✓ | ✓ | ✓ |

The result is **invariant to surplus steps** (64→256 all pass): once enough steps
exist to finish, extra steps do not corrupt the answer. So the programs are
**self-stabilizing** (they *hold* the result after completing — argmax via
`p = ptr < input_len` ceasing to move `r0/r1` past the bound), which is stronger
than an "external rail." But the reviewer is right on the core point: there is
**no notion of "computation finished."** `jax.lax.scan` always runs the full
`max_steps`; the `halt` block (id 31) is implemented as `pc = old_pc` (freeze the
counter in place), not a real exit. The model is **self-stabilizing, not
self-halting.**

Why this matters precisely where Option B goes:
- A **single-pass** target (incl. `swap_if_gt` on 2 elements) only needs
  self-stabilization, which we already have — surplus steps are harmless.
- A **multi-pass** target (3-permutation, `sort`) needs a **phase boundary**:
  "output of pass 1 becomes input of pass 2." Without a real `halt`/phase marker,
  passes bleed into each other and the stress layer will (correctly) reject it.

**Therefore the next step is an executor upgrade, but a smaller, ordered one than
"build a self-terminating model":**

1. **`output = tape` mode (hard blocker for Option B).** The executor returns the
   tape, not `r0`; `target_values` returns vectors; reward is MSE over the tape.
   Without this, `swap_if_gt` cannot even be *expressed* (nothing to score). Do
   this first.
2. **`swap_if_gt` (2 elements)** under the existing self-stabilizing semantics +
   stress certification. Single pass, so no halt needed yet.
3. **Real predicate-gated `halt` (deferred until perm3/sort demand it).** Add it
   only when the stress test on 3-permutation *proves* passes are interfering —
   i.e. earn the primitive from a measured failure, not from prediction. This is
   the anti-DSL-bloat discipline: the `halt` block already exists as a no-op stub;
   wiring it to a real `pc → length`/done-flag is a localized change, made when
   the data demands it.

### Step 1 done — tape-output works, and it EARNED the compare-exchange primitive

`evolve_code_tape.py` now also returns `final_tape` (additive; the scalar path is
untouched and `sum4` still solves). `evolve_tape_output.py` scores candidates on
the final tape, and `swap_if_gt2` (sort 2 elements ascending) is the first target
whose result is transformed *memory*, not a register scalar.

The first run failed (holdout tape-MSE ≈ 0.5, not 0). Rather than throw compute at
it, a **vectorized expressibility search** asked whether the DSL can represent the
swap at all: **1.5M random programs up to length 10 → zero exact solutions, error
floored at 2.0.** That is a representational gap, not a tuning problem. Diagnosed
precisely: a 2-element conditional swap needs `(a,b) → (min,max)` written back to
two cells, but min/max requires holding both values while swapping two registers —
and with only `r0, r1` and no temp, the swap cannot be expressed across separate
`pc` steps (`r0=where(p,r1,r0)` at one step and `r1=where(p,r0,r1)` at the next
clobber each other).

So the gap **earned one primitive** (the anti-bloat rule: add a block only when a
measured failure demands it). Block 35 `sort2(r0,r1)` is the textbook
compare-exchange — the atom of every sorting network — setting
`(r0,r1) ← (min(r0,r1), max(r0,r1))` simultaneously from old values. With it,
`swap_if_gt2` becomes discoverable: 2/4 seeds reach holdout-MSE 0, and both
solutions are **fresh-distribution certified** (max-abs-err 0 across value grids
incl. {-50..50} and fractional, 4 seeds × 1000 cases). The canonical certified
program:

```text
r0 = tape[ptr]      # load a   (ptr=0)
ptr += 1
r1 = tape[ptr]      # load b   (ptr=1)
sort2(r0, r1)       # r0=min, r1=max  ← the earned compare-exchange atom
tape[ptr] = r1      # cell 1 = max
ptr -= 1
tape[ptr] = r0      # cell 0 = min
```

This is the **first memory-transforming procedure** in CodePy — it rearranges the
data on the tape rather than reducing it to a scalar. Honest status: discovery is
2/4 seeds (not yet the 5/5 robustness bar of the scalar rungs), and it has not yet
been pushed through the length axis (swap generalizing to `swap_if_gt` over an
adjacent pair anywhere in a longer tape). Next: (a) tune shaping for 5/5
robustness, (b) the natural multi-pass target **bubble/3-permutation** built from
chained compare-exchanges — which is exactly where the deferred `halt` / phase
boundary is expected to be *earned* from a measured multi-pass failure.

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
