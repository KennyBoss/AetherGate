# Experience-Compressing SSM — thesis as an experimental CONTRACT (not a manifesto)

**Status: Stage-1 experimental protocol.** This is the bridge from the frozen
six-mechanism decomposition (`skill_compression/DECOMPOSITION.md`) toward a *new model*
whose edge is the property a frozen Transformer lacks: **active inference compute per
task falls with accumulated experience on repeated structure.** It is written as a
contract — definitions, baselines, pass/fail — so Stage 1 is a falsifiable experiment,
not a pretty simulator. Same discipline as `ROADMAP.md` Phase 1 (the vaccine against
self-deception).

## 0. The single hypothesis (a learning-dynamics claim, NOT a physical law)

> There exists an architecture in which **minimizing ordinary task loss by gradient
> descent *implicitly* reduces marginal active-inference-compute on repeated subtasks**,
> at matched task quality, **more than** a parameter-matched dense model trained on the
> same task order (curriculum) — and the reduction **transfers** to held-out instances
> of a learned abstraction (so it is compression, not caching).

If a parameter-matched dense + curriculum baseline matches it, the architecture adds
nothing → the hypothesis is **not supported** in that regime. We say so and stop.

## 1. Definitions (so the claim is measurable)

- **Skill.** An entry in a bounded skill memory `M` that is *invoked* in place of
  recomputing a recurring sub-procedure. A skill is real only if invoking it lowers
  active compute on inputs *not* used to create it.
- **Active compute / task (the cost metric).** Active parameters · steps (≈ FLOPs)
  consumed to solve a task at a fixed loss threshold — i.e. *conditional* compute, the
  quantity a gate/MoE controls. NOT wall-clock, NOT total parameters.
- **Reuse(task).** Fraction of the task's solution expressible by existing `M` entries
  (measured, held out from skill creation).
- **Computation-reuse, two-axis (the operational core).** A drop in cost counts as
  *computation*-reuse only if it survives BOTH:
  - **Axis S (surface):** *same latent function, different surface form* — a bijective
    remap of primitive tokens, distractor padding, and permutation of independent parts.
    Kills "recognized an input n-gram" (reuse = similarity).
  - **Axis C (composition):** the skill is invoked as a *callable subprocedure inside a
    NEW composition* unseen at skill-creation. Kills function-table memorization (the
    model storing `f` as a lookup rather than a reusable procedure) — which Axis S alone
    does not catch.
- **The five impostors** (each gets a control/ablation, §3):
  - *external cache* → cost must drop on **held-out** instances, not memorized pairs (`cache-M`).
  - *curriculum* → must beat a dense model on the **same task order** (`dense+curriculum`),
    which is ALSO the control for *implicit parameter reuse* (weights silently specializing).
  - *compute-redistribution* → total active compute **at matched loss** must decrease.
  - *gating collapse / "gate is just a router"* → must beat **frequency-matched random
    routing** (`random-routing`, see G6).
  - *hidden-state attractors* (SSM silently specializing its state) → cost-drop must
    survive a **frozen backbone** after warmup (`frozen-backbone` ablation).

## 2. The architecture (minimal Stage-1 form)

Reuse what is already validated; add the *least* new differentiable machinery (staging
the differentiability is the core methodological correction — do NOT fuse everything).

| component | source | Stage-1 status |
|---|---|---|
| recurrent backbone (SSM / dynamic-KV-memory) | MQAR-validated (`RESULTS.md`) | reused, frozen design |
| bounded skill memory `M` | promotion machinery (`skill_compression_bench`) | reused (growth rule measured, not yet differentiable) |
| **learned applicability gate** `g_t` | `skill_compression_ssm_gate.py` (BPTT+Adam) | **the ONE differentiable element trained online in Stage 1** |
| goal-conditioned composition | Phase 3.4 policy | deferred to Stage 1.5 (after the gate result) |

**Stage 1 changes exactly one thing into a learning loop** (the gate over a streamed,
growing library), and asks whether the cost law emerges. Everything else is held fixed,
exactly as the 3.x arc isolated one mechanism at a time.

## 3. Stage-1 experiment contract

**Task stream.** MQAR-like recall + synthetic program-stream with a hidden library of
recurring sub-procedures (reuse the `skill_compression` generator) and **controlled
repetition schedule** of that structure; held-out instances reserved for the transfer
test.

**Arms (parameter-matched where applicable):**
1. `ECS` — recurrent backbone + bounded `M` (growing) + learned gate `g_t`.
2. `dense` — parameter-matched recurrent net, no memory, no gate.
3. `dense+curriculum` — `dense` on the **same task order** (isolates curriculum AND implicit parameter reuse).
4. `frozen-M` — `ECS` with library growth OFF (isolates growth).
5. `cache-M` — `ECS` whose memory keys on exact inputs, no abstraction (isolates caching).
6. `random-routing` — `ECS` memory but skill choice **random at the same marginal
   fire-rate** as the learned gate (isolates *learned selection* from *having skills*; G6).

**Ablations on `ECS`:** remove gate; remove memory; **frozen-backbone** (freeze backbone
after warmup, adapt only memory+gate → kills hidden-state-attractor explanation);
(separately) add explicit MDL compression loss — see G2.

**Dissociation evals (computation-reuse, not similarity).** The held-out reuse split is
evaluated three ways: (i) plain held-out instances; (ii) **Axis-S** surface-remapped
(token bijection + distractors + permutation); (iii) **Axis-C** the skill embedded in a
novel composition. The cost-drop must hold on (ii) AND (iii), not only (i).

**The pass/fail metric (formal).** Let `A(e)` = mean active-compute/task on the held-out
reuse split after cumulative exposure `e`, at **matched per-task loss** `L*`. Library
growth is **blind** to all held-out/dissociation splits.
- **PASS** iff: `A_ECS(e)` decreases with `e`; its slope is significantly steeper
  (95% CI, ≥6 seeds) than `dense+curriculum` AND `frozen-M` AND `cache-M` AND
  `random-routing`; the drop **survives Axis-S and Axis-C**; `frozen-backbone` keeps the
  drop; and `ECS` matches `dense` loss at `L*`.
- **FAIL** iff any control matches the slope (`dense+curriculum`→curriculum/param-reuse;
  `cache-M`→caching; `random-routing`→no learned selection), OR the drop vanishes under
  Axis-S (→ similarity) or Axis-C (→ table memorization), OR it vanishes under
  `frozen-backbone` (→ attractors), OR `ECS` cannot reach `L*`.

## 4. What would prove us wrong (kept visible)

1. Cost falls only on memorized inputs, not held-out instances → caching, not compression.
2. `dense+curriculum` reproduces the slope → it was task ordering / silent param reuse.
3. Gain needs the explicit MDL loss to appear → engineered, not emergent (weaker claim; G2).
4. Matched-loss not reachable → conditional compute is buying nothing at this scale.
5. Drop vanishes under Axis-S surface remap → it was input similarity, not computation reuse.
6. Drop vanishes under Axis-C novel composition → it was function-table memorization.
7. `random-routing` matches the slope → the gate selects nothing; gain is just having skills.
8. Drop vanishes under `frozen-backbone` → it was hidden-state specialization, not the memory.

## 5. The ladder small → scale (each stage gated by the previous)

- **Stage 1** — emergence of the cost law in one differentiable gate over a growing
  library (this document). CPU / modest GPU. *Make-or-break.*
- **Stage 1.5** — add goal-conditioned composition (Phase 3.4) once the gate result holds.
- **Stage 2** — real tokens, small LM (10–100M): match dense Transformer at matched
  params on standard eval **and** show the unique cost-drop on a repeated stream.
- **Stage 3** — scaling-law study: competitive loss-vs-compute curve + the
  inference/continual advantage growing with scale. The fundraising artifact.
- **Stage 4** — scale with capital/team. The own model.

(`AetherGate` ships in parallel as the runway — real CI-cost savings fund the research.)

## 6. Anti-self-deception guardrails (the corrections over a naive plan)

- **G1 — stage the differentiability.** Make ONE mechanism learnable per stage with full
  controls. Fusing backbone+memory+soft-gate+MDL-loss at once would make any positive
  uninterpretable — the exact opposite of what made the 3.x results clean.
- **G2 — emergence WITHOUT a compression loss first.** The strong, falsifiable claim is
  that reuse arises from task-loss + a capacity-bounded memory + a compute-penalized
  gate. An explicit MDL/compression auxiliary loss *manufactures* the result and is
  admitted only as an ablation, never in the base model for the emergence claim.
- **G3 — the metric is active compute at matched loss on held-out reuse**, never
  wall-clock, never cost-vs-reuse correlation on the training stream (that is curriculum).
- **G4 — every win survives ≥6 seeds, held-out, `protocol_valid`, with the curriculum,
  cache, and frozen-memory controls present**, exactly as in `skill_compression/`.
- **G5 — reuse means COMPUTATION reuse, on two axes.** A cost-drop is admitted only if it
  survives Axis-S (same function, remapped surface) AND Axis-C (skill in a novel
  composition). One axis is insufficient: S alone misses table memorization; C alone
  misses surface shortcuts. Library growth stays blind to both eval splits.
- **G6 — the random-routing control matches the MARGINAL fire-rate, not per-skill
  frequencies.** Matching the learned gate's per-skill distribution would leak the gate's
  decision into the control. Preserve only *how often any skill fires* (the P* act/think
  ratio) and randomize *which* — so `ECS > random-routing` isolates learned selection
  from mere skill availability / gating collapse.
- **G7 — deeper frames are instruments, not licenses.** The representation-MDL
  order-parameter (frame A) and the one-process re-compilation claim (frame B) are
  admitted only as auxiliary measurements/tests (§6b). The behavioral active-compute
  metric on held-out Axis-S/C remains the sole arbiter of PASS/FAIL. "Phase transition"
  and "self-rewriting computation" are forbidden as conclusions unless their specific
  order-parameter / coherence test fires; otherwise they are seductive, unfalsifiable
  language and stay out of the result. Do not assume unification — earn it.

## 6b. Epistemic frame — auxiliary instruments (earned, not assumed)

Two deeper lenses are admitted ONLY as additional measurements layered on top of the
behavioral contract (§3), which stays primary and load-bearing. They sharpen the *why*;
they never relax a falsifier.

- **Representation-space view (frame A).** If compression is real, tasks sharing a latent
  skill should converge to a stable internal code. **Auxiliary order-parameter**, tracked
  vs exposure `e`: intrinsic dimensionality of the hidden-state manifold per skill-family
  (expect ↓), `MI(hidden; skill_id)` (↑) and `MI(hidden; surface_form)` (↓), skill-cluster
  separability. The deep form of compression is falling **representation-MDL**, not only
  FLOPs. *Guard:* a representation that "stabilizes" while active-compute on Axis-S/C does
  NOT drop is **clustering, not compression** — rejected. Representational evidence is
  corroborating only; no claim is admitted from it alone.
- **Re-compilation view (frame B) — a HYPOTHESIS to earn, not a premise.** The conjecture
  that gate, memory and skill are temporal cross-sections of one behavior-recompilation
  process makes a falsifiable structural prediction: **under a controlled task-distribution
  shift, gate-firing, memory-contents, and representation co-adapt coherently** (and
  intervening on one lawfully shifts the others). **Coherence test:** measure their joint
  vs independent adaptation across a distribution shift. Co-adaptation ⇒ one re-compiler;
  independent drift ⇒ three bolted-on modules (frame B false for our system). Until this
  test passes, gate/memory/skill remain three separately-measured mechanisms — *we earned
  the six-mechanism separation by NOT assuming unification (the 3.4 lesson)*. Bounded
  framing: this is amortized behavioral re-compilation (test-time training / fast-weights /
  amortized-inference family), not self-modifying AGI.

## 6c. Stage 1a — result: the law does NOT emerge from naive training (honest negative)

`ecs_stage1.py` runs the contract with a *trained* per-step gate + growing skill memory
over the task stream (decode-steps = active compute; the Markov form of the backbone,
recurrence deferred to Stage 2). **6 seeds, deterministic, `protocol_valid=True`.**

| arm | active compute (steps) | solve-rate |
|---|---|---|
| **ECS** (trained gate + growing M) | 9.04 | 0.856 |
| dense (shuffled) | 7.80 | 1.00 |
| frozen-M = cache-M (ECS policy, no usable skills) | 10.60 | — |
| random-routing | 22.56 | — |
| ECS slope −0.030 vs **dense+curriculum slope −0.032** | (identical) | |

**PASS = False, and every guard fired as designed:**
- **Matched-loss guard FAILS:** ECS solve-rate 0.856 < dense 1.00 — the gate sometimes
  fires a skill that overshoots and fails to solve within the cap. Skills *hurt*
  reliability, so a cheaper step-count would be uncomparable anyway.
- **Curriculum control fires:** ECS's decline (−0.030) equals `dense+curriculum`
  (−0.032) → the cost-drop over the stream is **task ordering, not skill compression**.
- **No Axis-C transfer:** ECS 14.44 ≈ dense 14.13 on novel compositions.

**Diagnosis (precise, and it closes the loop).** The law fails to emerge for one reason:
**skill *selection* is the unsolved goal-conditioned inverse** — the exact Phase-3.4
wall. `random-routing`'s blow-up (22.6) shows firing is only safe with the *right*
skill; the naive independent per-step head cannot pick it, so the trained gate either
abstains (→ ECS ≈ primitive ablation) or misfires (→ solve-rate drop). Emergence is
**gated by the same residual** the structural arc isolated (3.3–3.4): goal-conditioned
selection of the right parametric skill.

**Oracle-ceiling diagnostic (pre-registered, distinguishes A vs C).** Before building a
stronger selector we measured the perfect-selection ceiling (shortest action path under
primitives vs primitives+skills, pure search): held-out **prim-optimal 3.27 → skill-optimal
2.00** (gain +1.28); Axis-C **4.67 → 2.68** (gain +1.99, composes to depth). So the library
DOES hold useful, composing abstractions → **C (inadequate library) is ruled out**; the
bottleneck is realization. Honest refinement: the learned ECS (9.04) is far from the
skill-ceiling (2.0) — but `dense` (7.80) is equally far from its primitive-ceiling (3.27),
so the deficit is **per-step selection quality broadly (A)**, with skill-selection harder
on top; B (representation) is not separable from weak selection in this greedy setup. Part
of the gap is the **greedy-decode ceiling** (a per-step policy cannot match a non-greedy
search optimum), so 1a''s fair target is ECS-selector vs dense-selector at matched decode
regime — *pre-registered, not to be moved post-hoc*.

**Selection is now high-stakes (the most telling number).** `random-routing` 22.6 vs
`dense` 7.8 means a wrong skill call is *catastrophic*, a right one is a big saving — the
library has stopped being a "safe accelerator" and become a tool-use / MoE / program-
synthesis–like regime where the value lives entirely in the selection policy.

**Named fix → Stage 1a' (a test of hypothesis A, not a claim).** Replace the naive
per-step head with the Phase-3.4b/c selection machinery (autoregressive interaction +
best-first ordering). Pre-registered prediction table:

| if outcome | conclusion |
|---|---|
| ECS→dense solve-rate, compute ↓, beats curriculum + survives Axis-C | A confirmed: selection was the bottleneck |
| solve-rate recovers but compute ≈ dense | greedy-decode ceiling dominates; need search/recurrence |
| slope stays ≈ dense+curriculum | not architectural here; the drop is curriculum |
| no change at all | bottleneck deeper than selection (B); revisit representation |

**Reading.** This is the intended use of the contract: a naive build was *prevented* from
claiming a curriculum artifact as emergence. The structural arc's residual is now also
the empirical blocker of Stage 1 — the two lines meet on the same open problem.

## 6d. Stage 1a' — result: selection fixed, compression still unproven (honest negative)

`ecs_stage1b_selection.py` runs the named fix: the greedy independent-argmax commit is
replaced by **policy-guided best-first search** (Phase-3.4c ordering — a wrong skill is
recoverable by backtracking, not fatal). Metric correction baked in (G3): under search,
active compute is **policy forward-passes (nodes expanded)**, counted identically for
every arm, so search cost cannot hide. **6 seeds, `protocol_valid=True`, solve-rate 1.0
on every seed.**

| arm | active compute (forward-passes) |
|---|---|
| **ECS** (best-first + growing M) | 12.37 |
| dense (shuffled) | 11.17 |
| frozen-M (ECS policy, no usable skills) | 17.08 |
| cache-M (held-out miss → primitives) | 17.08 |
| random-routing (gate fire-rate matched, which-skill randomized) | 22.09 |
| Axis-C: ECS 38.70 vs dense 32.66 (ECS axisC-solve 0.996) | |

**PASS = False — and it lands on the pre-registered row 2** ("solve-rate recovers but
compute ≈ dense → greedy/search ceiling, not selection"):

- **The selection fix worked at its job.** Stage-1a's misfire/abstain collapse is gone:
  solve-rate is fully matched (1.0 vs 1.0, all seeds), and ECS robustly beats *every
  control that shares its policy* — frozen-M / cache-M (17.1) and random-routing (22.1).
  So choosing the *right* skill via best-first is worth ~30% fewer forwards than mis-using
  or ablating the skills. §6c's "value lives in the selection policy" is confirmed in that
  direction: competent selection recovers what the naive head threw away.
- **But the PASS bar — beat the independent dense baseline — is not cleared.** ECS vs
  `dense` is a **seed-dependent wash** (ECS wins 3/6 held-out seeds, loses 3/6; mean 12.37
  vs 11.17 sits inside the seed spread), and **Axis-C transfer does not survive** (dense
  cheaper on 4/6 seeds; the small-config Axis-C win washed out at 6 seeds). A clean
  primitive searcher with *no skill apparatus* is as cheap or cheaper.

**Diagnosis (the new, sharper blocker).** Selection is no longer the wall. The residual is
a **break-even**: the macro-augmented action set adds *search branching* that roughly
cancels the *path-shortening* the macros buy. Skills beat their own misuse (vs
random-routing) and their own ablation (vs frozen-M/cache-M), but not their own absence
(vs dense). Two follow-up sweeps (`sweep_depth_breakeven.py` + a kmax probe) located the
real lever — and *falsified my first guess*:

- **Depth does NOT rescue it (pre-registered prediction falsified, kept visible).** I
  predicted ECS would cross below dense past some composition depth `d*`. It does not: at
  eval depths 2→6 the ECS−dense forward gap is `+1.2, +12.0, +13.7, +15.3, +2.3` — ECS is
  *never* cheaper, `d* = None`, and the gap widens before collapsing only as both hit the
  solve-rate floor. The depth hypothesis is dead; do not revive it.
- **Library SIZE is the lever (the actual finding).** Sweeping the macro cap at fixed
  everything-else, eval depth 3: `kmax=8` (the default) loses by **+12.0** forwards, but
  `kmax=2` *wins* by **−4.7** (4/5 seeds, 23.3 vs 27.9). A small library beats dense; a
  large one loses to it. The branching cost of macros is roughly linear in library size
  while the path-saving saturates, so an unpruned library is net-negative.
- **…but the macros are surface n-grams, not the true subroutines.** Inspecting promotions:
  the library captures solution-path n-grams (`cea`, `ac`, `ceac`) straddling subroutine
  boundaries, **not** the generative hidden subroutines (`cac`, `bbd`, `edd`, `acb`). At
  kmax=8 they are blatantly redundant (`cea`/`ceac`/`acea`/`eac`/`ea`) — pure branching
  noise. So even the kmax=2 win is "less noise," not "right abstraction," and Axis-C stays
  a wash (kmax=2: ECS 23.3 ≈ dense 23.2 at depth 3; only −3.4 at depth 4).

**Verdict, precise.** Compression — a cost-drop that beats the independent baseline AND
transfers under Axis-C — remains **unproven**, but the blocker is now *named and localized*:
not selection (fixed), but **library quality + size control**. This is exactly
`codepy/CODEPY_TURING_COMPLETENESS_PLAN.md` **Path A (macro pruning)** and the abstraction
gap the CodePy↔thesis section flagged ("storage emerging" — the library exists but does not
yet capture the *generative* procedures). The two research lines meet on one mechanism.

**Named next test (pre-registered, do not move post-hoc).** The fix is not more search but
a *better, smaller* library. Falsifiable prediction: with (a) online macro pruning by
marginal forward-pass utility (Path A) AND (b) promotion that recovers the generative
subroutines (measured: promoted-macro ↔ hidden-subroutine match rate ↑ from the current
~0), ECS at its pruned-optimal size should beat dense on held-out AND survive Axis-C at
depth ≥3 across ≥6 seeds. If a library that provably captures the true subroutines still
does not beat dense once branching is priced in, the architectural claim fails at this
scale and we say so.

## 6e. Stage 1a'' — pruning + a recoverability probe (superseded in part by §6f)

`ecs_stage1c_pruning.py` ran the §6d pre-registered test. Both halves came back negative
in a way that, together, point past this DSL — and one half turned out **ill-posed**, so
we report it rather than pretend it was a fair test (G).

- **(b) Generative-subroutine recovery is ILL-POSED here (measured, not assumed).** A
  recoverability probe over 600 sampled generative programs: the shortest-compute solution
  is **strictly shorter than the generative program 89%** of the time and **contains a
  whole hidden subroutine only 2.2%** of the time. The integer state *collapses* (many
  programs hit the same number), so optimal-compute supervision almost never traverses the
  "intended" subroutine. Match-rate ≈ 0 is therefore a property of the **environment**,
  not a weak promoter: "recover the generative procedures" cannot be satisfied under
  shortest-path supervision in a collapsing-state DSL. The only learnable skill here is a
  recurring *optimal-path n-gram*.
- **(a) Utility pruning helps, but cannot manufacture compression that the library does
  not contain.** Marginal-forward-pass pruning (blind to eval) removes almost nothing
  (8 → 7.3 macros) and held-out stays a **wash** (pruned 12.30 vs dense 11.17, 3/6 seeds);
  Axis-C still fails (2/6). PASS_pathA = False. Crucially this **refines §6d's kmax claim**:
  the earlier "kmax=2 wins −4.7" was a *fixed-small-library-from-the-start* effect (it
  changes which macros ever get promoted), NOT the same as pruning a grown library — at the
  trained config the individual macros are near-neutral, not net-harmful, so there is
  nothing for pruning to cut. Library *size policy at promotion time* ≠ *post-hoc pruning*.

**Unified conclusion (the honest end of the symbolic line).** Compression cannot be
demonstrated in this DSL because the DSL's state collapses: the generative structure is
invisible to optimal-compute supervision, so the library can only ever hold path n-grams
that roughly break even against primitive search. This is **not** a failure of selection
(fixed in 1a'), of pruning (1a''), or of scale (capacity-insensitive, 1a') — it is the
**environment lacking observable generative structure**. The pre-registered escape is
therefore not another symbolic tweak but the venue itself: a **rich-state environment
where the generative procedure is observable** (real code / the AetherGate repair loop,
where a "skill" is a reusable patch/procedure visible in the solution, not an emergent
n-gram). That is Stage 2, and 1a'' is what earns the move — a measured boundary, not a
pivot of convenience. Artifacts: `ecs_stage1c_pruning.json`.

> **Correction (see §6f).** §6e concluded the blocker was *observability* (wrong
> environment). The follow-up disentangling experiment shows that was only half right: the
> blocker is **sufficiency**, which a domain change does NOT fix. §6f supersedes the
> "Stage 2 is justified" reading below.

## 6f. Stage 1b — disentangling: the ceiling itself fails (the real end of the symbolic line)

OpenAI's reading of §6e — "you trained shortcuts, not programs" — is correct but raised
the decisive question: is shortcut-dominance a property of the SUPERVISION (fixable) or
the DOMAIN/METHOD (fatal)? `ecs_stage1d_disentangle.py` separates two axes in the SAME
DSL, no domain change (which would beg the question). **6 seeds, deterministic.**

**(A) Identifiability — YES, it was the supervision.** Online promotion under a
*generative* teacher (the true subroutine decomposition) recovers **3.8/4** hidden
subroutines (match-rate 0.48; the rest are legitimate 2-grams within them) vs **0.02**
under the shortest-path teacher. So "ECS learns shortcuts not programs" was an artifact of
optimal-compute supervision, and is fully fixable by aligning the teacher. OpenAI's
diagnosis confirmed.

**(B) Sufficiency — NO, and decisively. This is the result.** Hand ECS the **ORACLE
library = the exact four hidden subroutines** and run the canonical best-first selector
(the ceiling for *any* promotion method). It **loses to dense**: held-out **20.0 vs 11.2
forward-passes (oracle wins 1/6 seeds)**, Axis-C **56.3 vs 32.7 (0/6)**, matched solve.
The *perfect* library, competently selected, is *worse* than no library at all.

| axis | result | meaning |
|---|---|---|
| (A) identifiability | match 0.48 (gen) vs 0.02 (shortest) | program WAS identifiable — supervision artifact |
| (B) sufficiency (oracle lib) | held 20.0 vs dense 11.2; Axis-C 56.3 vs 32.7 | even the true programs do NOT pay |

**Decision (pre-registered branch A=yes, B=no): the compression thesis fails at this
scale regardless of domain.** The mechanism is now exact and general: in discrete
best-first search every macro adds a *branch at every expanded node* (search cost ∝
action-set size) while saving at most a few path steps; the branching tax dominates the
path-saving, so macro-augmentation is **net-negative even with a perfect, fully-aligned
library**. This is NOT observability (a richer domain would not change the search
arithmetic), NOT selection (best-first is competent), NOT promotion quality (oracle is the
ceiling). **A domain change to AetherGate would not rescue it** — §6e's "Stage 2 justified"
reading is retracted.

**What this means for the program (honest, load-bearing).** The "discrete macro library +
search" instantiation of experience-compression does not yield a marginal-compute win at
this scale, even at its ceiling. The win the whole line is chasing — *compressed
experience lowering inference cost* — must come from a substrate where reuse does NOT pay
a per-node branching tax: i.e. **amortized into the policy/weights (parametric skills that
add no branching), not bolted on as extra discrete actions a searcher must branch over.**
That is a genuine redirection named by measurement, not a tweak. The KV-memory / SSM
backbone (recall amortized in state, O(1), no branching) is the substrate that escapes the
tax — which is where Stage 2 should actually go: skills as *learned fast-weights*, not
search actions. Artifacts: `ecs_stage1d_disentangle.json`.

## 6g. Stage 2 (falsification) — parametric skill beats BOTH discrete and clean-dense in-distribution; transfer still open

`ecs_stage2_fastweight.py` runs the single two-sided test §6f named: the SAME true
subroutines expressed as **context-gated fast-weights** (a learned additive shift `Δ_k` on
the 5 primitive-action logits, fired by a per-skill gate) vs the **discrete-macro** form
(the broken class) vs **dense**. The parametric skill never adds an action — the action
set stays size-5 — so it pays no branching tax; cost is forward-passes, counted
identically. Two controls for dense: skills-off-same-policy AND a **cleanly-trained dense**
that never saw skill structure (removes the "param just had more training" confound).
Interference measured explicitly (skills-on minus skills-off on single-skill sets).
**6 seeds, deterministic, solve-rate 1.0 everywhere.**

| arm (forward-passes) | held-out | Axis-C |
|---|---|---|
| **param-skill (Δ on logits)** | **6.51** | 37.35 |
| clean dense (fair control) | 8.32 | 32.30 |
| dense (same policy, skills off) | 6.88 | 36.56 |
| discrete-macro (the §6f broken class) | 7.49 | 50.81 |
| interference (on−off, single-skill) | mean 0.044, max 0.13 | |

**The mechanism claim is CONFIRMED (first skill apparatus in the line to beat no-skill):**
- **param < discrete** (6.51 vs 7.49) — the branching tax is really avoided by moving the
  skill into dynamics-space. The §6f redirection was correct.
- **param < clean dense** (6.51 vs 8.32, **5/6 seeds**, ~22%) — a real in-distribution win
  against a separately-trained baseline that never saw skills. Not the training confound.
- **Interference is negligible** (mean 0.044) — the named hidden tax did not appear. So the
  pre-registered "param wins + low interference" branch fires: **dynamics-space skills work
  where discrete-search skills could not.**

**But the honest limit — compositional transfer still fails (the line's one invariant).**
Axis-C: param **37.35 > clean dense 32.30** (2/6). The fast-weight skills help on familiar
depth and do **not** compose to novel depth. So this is an *in-distribution* compression
win, not yet the transferring compression the thesis ultimately requires. Every stage of
this line has died on Axis-C; the class change fixed the *cost-of-acting* (branching tax)
but not *composition*.

**Precise standing.** For the first time a skill mechanism is net-positive vs no-skill at
its own compute budget, with no branching tax and no interference — the parametric/
fast-weight class is validated as the right substrate (consistent with the KV-memory/SSM
direction). What remains unproven, exactly as before, is **compositional transfer**: skills
that lower cost on *novel compositions*, not just familiar ones. That — not the substrate —
is now the lone open problem, and it is the same Axis-C wall the structural arc (3.3–3.4)
and every Stage-1 arm hit. The substrate question is settled; the composition question is
the whole remaining game. Artifacts: `ecs_stage2_fastweight.json`.

## 7. Honest scope

We are not out-training frontier LLMs and not claiming AGI. We are testing whether one
architectural property — *implicit reduction of marginal inference compute through
compressed experience* — emerges from training and survives controls at small scale. A
positive earns the right to scale; a negative is a real result. Compass:
`NOTES_cognition_map.md`. Evidence: `skill_compression/DECOMPOSITION.md`, `RESULTS.md`.
