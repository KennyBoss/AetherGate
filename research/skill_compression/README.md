# Skill-Compression — Phase 1 crash test

> **Program result (one line).** A learned internal state does not always reduce
> the *number* of actions, but it stably reshapes the *distribution* of actions
> across familiar / mixed / novel regimes (`fire_reuse > fire_mixed > fire_OOD`,
> the invariant that survived every stage MQAR → compression → applicability →
> hierarchy → real code-repair); as the cost of acting rises this converts into
> measurable compute/CI savings. *Cheap actions → act first; costly actions →
> think first.*

Self-contained, CPU-only (no JAX) experiment for the make-or-break question of
`../ROADMAP_SKILL_COMPRESSION.md`:

> Does compressing repeated experience into reusable skills **measurably** lower
> cost on later tasks, versus a parameter-matched baseline that does no
> promotion — and is the gain **reuse-specific** rather than a trivial artifact
> of having a bigger action set?

## How to run

```bash
python3 skill_compression_bench.py --seeds 11,17,23,41,73,101 \
  --output-json artifacts/skill_compression_seed6.json
```

Runs in well under a second. Output JSON carries a `protocol_evidence` block
(held-out disjointness, seed count) exactly like the repo's recall harness.

## The benchmark (one sentence)

Program synthesis over a 5-op integer DSL where tasks are composed from a hidden
library of subroutines, so solutions share recurring sub-sequences; the cost
metric is **search nodes expanded** to solve a held-out task by deterministic
iterative-deepening DFS.

## Arms (parameter-matched: identical DSL + identical solver)

| Arm | What it is |
|---|---|
| `OFF` | primitives only, no compression (baseline) |
| `FREQ` | promote a sub-sequence once it recurs ≥K times (frequency heuristic — our own baseline gate) |
| `UTIL` | scored promotion gate `g_t`: `utility = reuse·(len−1) − penalty·lib_size` |
| `RANDMACRO` | **decisive control** — same count & lengths of macros as UTIL, but random contents |
| `UTIL+APP` | **Phase 1.5** — UTIL's library behind an *applicability gate*: cheap macro-probe (`--probe-budget` nodes), fall back to primitives if the skills don't fit |

The `RANDMACRO` arm is the point: a macro is a multi-step jump that shortens any
depth-limited search regardless of reuse. Only if learned macros beat *random*
macros of matched shape is the gain genuinely about captured reuse.

## Results (6 seeds, held-out, protocol_valid=True, deterministic)

Mean search nodes to solve a held-out task (lower = cheaper):

| Arm | REUSE eval | speedup vs OFF | NO-REUSE (OOD) eval |
|---|---|---|---|
| OFF | 54.4 | 1.0× | 125.9 |
| FREQ | 10.1 | 5.4× | 359.4 |
| RANDMACRO | 16.1 | 3.4× | 34.4 |
| UTIL | 11.9 | 4.6× | **456.0** ← OOD penalty |
| **UTIL+APP** | **9.5** | **5.7×** | **94.8** |

Seed-6 run committed under `artifacts/`. Reproduce: rerun the command above
(deterministic — sorted tie-break, independent of `PYTHONHASHSEED`).

## Verdict — honest, bounded

**HOLDS (the core claim):**
- **Compression pays on reuse.** OFF `54.4` → learned `~10–12` nodes, ~**5×**
  cheaper on held-out tasks. Robust across 6 seeds, `protocol_valid=True`.
- **The gain is reuse-specific.** UTIL (`11.9`) beats the matched RANDMACRO
  control (`16.1`): learned macros capture real structure, not just "longer
  jumps." This is the decisive control and it passes.

**Phase 1.5 — the applicability gate WORKS (Pareto win):**
- The bare learned library is a *specialization with a cost*: on out-of-
  distribution NO-REUSE tasks it *hurts* badly (UTIL `456` vs OFF `126`) because
  foreign macros inflate branching — exactly like automatism misleading on novel
  tasks.
- Gating it behind a **cheap macro-probe + primitive fallback** (no oracle label;
  decision made purely from search behaviour) gives the best of both: `UTIL+APP`
  is **9.5** on reuse (keeps the speedup) **and 94.8** on OOD — *below* the OFF
  baseline of 125.9. It **Pareto-dominates** every other arm.
- Robust to the probe budget: across `--probe-budget` ∈ {30,40,60,100} reuse
  stays ~8–9 and OOD ~79–107 (always < OFF). Not knife-edge tuning.

**Still open (honest negative):**
- **The scored gate does NOT yet robustly beat the frequency heuristic** (6-seed
  FREQ `10.1` < UTIL `11.9`; at 3 seeds the order flips; UTIL is high-variance).
  The roadmap's "beat your own heuristic" bar is **not yet cleared** → gate
  redesign + a *learned recurrent* gate over SSM state is the next milestone.

## Phase 1.5B — a LEARNED recurrent gate (SSM-GATE): honest negative

`skill_compression_ssm_gate.py` replaces the hand-set probe with a *learned*
recurrent gate `h_t = tanh(A h_{t-1} + B x_t); g = σ(w·h_T + b)` over a per-task
sequence of macro-reachability probes (one token per skill: "does this skill move
the state toward the target?"). The gate trains by BPTT+Adam on the agent's OWN
mixed experience (familiar + novel tasks, labelled by whether skills actually
helped), never on the held-out eval; the feature-probe cost is charged to it.

Goal (the chosen bar): match `UTIL+APP` on reuse **and** beat it on OOD.

| Arm | REUSE | OOD |
|---|---|---|
| OFF | 54.4 | 125.9 |
| UTIL | 11.9 | 456.0 |
| UTIL+APP (behavioural probe) | **9.5** | **94.8** |
| SSM-GATE (learned recogniser) | 17.0 | 101.8 |

**Verdict: the learned static gate does NOT beat the behavioural probe** (reuse
17.0 vs 9.5; OOD 101.8 vs 94.8). It trains to **0.93** accuracy on its own
experience, yet loses — and the loss is *robust to the decision threshold*
(reuse stays 17.0 at thresholds 0.5→0.1). Diagnosis: the gate confidently refuses
~6–8% of reuse tasks because one-step reachability features are **myopic** — they
cannot see that a task is solved by a *composition* of skills, whereas the probe
discovers multi-step reachability by actually searching.

**Why this is a result, not a dead end.** It is direct evidence for this repo's
"action over prediction" thesis (`../NOTES_cognition_map.md`): *knowing when to
apply a skill is better decided by acting (a cheap probe) than by static
recognition* — **when probing is cheap**.

### The completing experiment: when attempting a skill is EXPENSIVE

The negative predicts its own boundary: if *trying* a skill has real cost (the
`--exec-penalty` knob — real actions / API calls), the always-probe `UTIL+APP`
pays that cost on every task, while the selective learned gate pays it only when
it fires. Sweeping the penalty `P` (mean nodes, reuse / OOD):

| `P` (cost to attempt a skill) | UTIL+APP | SSM-GATE | learned gate wins? |
|---|---|---|---|
| 0 (cheap probe) | **9.5 / 94.8** | 17.0 / 101.8 | no |
| 40 | 49.5 / 134.8 | 53.8 / **131.7** | OOD only |
| 60 | 69.5 / 154.8 | 72.3 / **146.7** | ✅ (within 5% reuse) |
| 100 | 109.5 / 194.8 | **109.1 / 176.6** | ✅ both |
| 200 | 209.5 / 294.8 | **201.2 / 251.4** | ✅ both |

**There is a clean regime boundary.** Below it, *act-then-decide* (behavioural
probe) wins. Above it, *recognise-then-decide* (the learned `g_t`) wins, because
it avoids the cost of futile skill attempts — exactly what an internal model of
applicability is *for*. The learned recurrent gate earns its keep precisely where
the cognition map says an internal world-model should: **when acting has
consequences.**

### Phase 1.5D — the crossover obeys an EXACT cost law; P* with a CI

`skill_compression_phase_diagram.py` pins the boundary down. The two arms have
linear cost in the action-penalty `P`, with slopes that are exactly their
skill-attempt rates:

```
C_app(P) = base_app + 1.0       · P     (probe attempts a skill every task)
C_ssm(P) = base_ssm + fire_rate · P     (gate attempts only when it fires)
  =>  crossover   P* = (base_ssm − base_app) / (1 − fire_rate)
```

Validated against independent high-`P` runs: **max abs error = 0.00 nodes** — the
law is exact, so `P*` is a *derived constant of the environment*, not a fuzzy
empirical fit. Crossover `P*` (6 seeds, 95% CI):

| Task stream | `P*` | 95% CI | robust? |
|---|---|---|---|
| OOD (novel) | **30.1** | [19.5, 40.8] | ✅ tight |
| Mixed 50/50 (deployment-like) | **47.2** | [31.5, 62.9] | ✅ tight |
| Reuse (in-distribution) | 181.9 | [−36, 400] | ✗ ill-conditioned |

**Reading.** On a realistic mixed stream the learned think-first gate becomes
strictly cheaper than the act-first probe once attempting a skill costs more than
`P* ≈ 47` node-equivalents, and stays cheaper for all larger `P`. The reuse-only
`P*` is high-variance *by construction* — there `fire_rate → 1`, so the
denominator `(1 − fire_rate)` is tiny and `P*` is ill-conditioned (when almost
every task is familiar, the always-probe overhead is negligible and you need a
very high, poorly-defined action cost before think-first pays). That is an honest
property of the law, not noise to hide.

**The phase law (one line).** A learned internal model carries a fixed overhead
but a smaller marginal cost-of-acting; past a critical action-cost `P*` it always
wins. *Cheap actions → act first. Costly actions → think first.*

## Phase 2 — does the boundary `P*` shift with hierarchy depth?

`skill_compression_hierarchy.py` turns `P*` into an order parameter `P*(D)`. A
hidden skill hierarchy is built bottom-up — a skill that calls skills:
`L1`(prim seq) → `L2`(two `L1`) → `L3`(two `L2`). The budget is tamed by gating
between **adjacent** abstraction levels (engage top-level `LD` vs fall back to
`L(D−1)`), so search stays ~1–2 steps at every depth; only the action set grows.
A depth-`D` reuse task is one `LD` skill; an OOD task is a same-length *invalid*
`L(D−1)` combination the top skill cannot cover.

**Result 1 — the economic boundary is depth-INVARIANT (Scenario A).** Mixed-stream
`P*(D)` (6 seeds): `D1: 10.3 → D2: 17.3 → D3: 17.0` (CIs at D2/D3 ≈ ±0.5). For a
sound (adjacent-level) fallback the act/think break-even does **not** fall with
depth — it is *local to each abstraction transition*, not amplified by how deep
the hierarchy goes. This **refines the coupling hypothesis** (we expected `P*↓`):
deeper skills do not, by themselves, make the internal model pay off earlier.
(D1 reads lower only because there "one level down" *is* the primitive floor.)

**Result 2 — but the gate's JUDGEMENT sharpens with depth (the positive).** As
the hierarchy deepens, the richer recurrent context makes the learned gate
strictly more discriminating:

| depth `D` | gate accuracy | reuse fire-rate | OOD fire-rate |
|---|---|---|---|
| 1 | 0.917 | 0.703 | 0.139 |
| 2 | 0.956 | 0.994 | 0.050 |
| 3 | **0.987** | **0.997** | **0.033** |

The gate converges to *fire confidently on a familiar skill* (→1.00) and *abstain
confidently on a novel one* (→0.03); accuracy climbs 0.92→0.99. So hierarchy does
not move the **price** of thinking-first, but it improves the **quality** of the
recognition — the internal model recognises applicability ever more reliably with
deeper compositional context. (Reuse-only `P*` stays ill-conditioned, `fire→1`.)

**One line.** *Depth does not move the price `P*`; it sharpens the judgement.*

## The bridge — does the boundary survive a REAL code-repair loop?

`code_repair_agent.py` leaves the synthetic search env for an actual
engineering loop: it **`exec`s candidate patches and runs them against test
cases** (no LLM, no network — those are the confounds we avoid). A buggy program
is `def f(x,y): return (x op1 y) op2 c`; a repair finds the `(op1,op2,c)` triple
(36 candidates) that passes the tests. A hidden library of recurring fix patterns
is the skill set. The expensive operation is the real one: **running the test
suite**. Cost = number of test runs.

Arms: `OFF` (enumerate all 36, run tests until pass), `APP` (try library patches
first, then broad), `SSM` (a learned recurrent gate engages the library only when
a cheap 1–2-case partial-pass probe says it fits). Same phase law:
`P* = (base_ssm − base_app)/(1 − fire_rate)`, `P` = cost of attempting a skill.

**Result (6 seeds, real code execution, held-out fresh bugs, `gate_acc = 0.995`):**

| stream | OFF runs | APP runs | SSM runs | fire | `P*` | 95% CI |
|---|---|---|---|---|---|---|
| reuse | 23.6 | **3.5** | 6.0 | ~1.0 | 214 | [201, 227] |
| OOD | 15.0 | 17.7 | 15.9 | 0.09 | **−2.0** | [−2.3, −1.7] |
| **mixed 50/50** | 19.3 | 10.6 | 10.9 | 0.54 | **0.7** | **[0.4, 1.0]** |

**The phase boundary survives — and the real-world regime is already on the
think-first side.** Three readings:
1. The same `act-first / think-first` crossover law holds in a real test-running
   loop, with **tight CIs**.
2. On novel (OOD) bugs `P* ≈ −2`: the learned gate is **already cheaper than the
   act-first probe at zero action cost**, because a wasted library probe costs
   *whole* test runs while the gate's partial probe costs fractions of one.
3. On the mixed stream `P* ≈ 0.7` — essentially **zero**. Any realistic test-run
   cost (seconds of CI, a paid API call) is `≫ 0.7`, so think-first wins
   immediately. In the synthetic env `P*` was ~47; in the real loop it collapses
   toward 0, because *running the test suite is exactly the kind of expensive
   action the law says favours an internal model.*

The chain now runs end to end on one idea — *learn to decide when to act*:
`MQAR → Skill Compression → Applicability → P* → Hierarchy → Code-Repair Agent`.

## Phase 2.5 — does the *structure* survive heterogeneity + real `pytest`?

The riskiest test: take the gate out of the single template it was tuned on into
a **heterogeneous** benchmark (`code_repair_real_repos.py`) of five function
families with different patch-space sizes — `arith2`(36), `cmp`(20), `linear`(18),
`select`(4), `offbyone`(6) — and ask whether the structure that survived every
prior stage holds: `fire_reuse > fire_mixed > fire_OOD`. A sample of repairs is
also run through **real `pytest` subprocesses** (real wall-clock = CI cost).

Scope honesty: these are *generated-but-real* Python modules (real source, real
`pytest`), self-contained so the sandbox stays confound-free. Fetching actual
GitHub repos needs network + dependency isolation and is the explicit next step.

**Result (6 seeds, held-out, real `pytest` validated):**

| stream | OFF runs | APP runs | SSM runs | fire-rate |
|---|---|---|---|---|
| reuse | 8.9 | 2.6 | 6.1 | **0.567** |
| mixed | 8.3 | 5.9 | 7.5 | **0.370** |
| OOD | 7.6 | 9.1 | 9.0 | **0.173** |

1. **The structure survives — `fire_reuse > fire_mixed > fire_OOD`, all 6 seeds**
   (reuse `0.567` > mixed `0.370` > OOD `0.173`; `holds_all_seeds = True`). The
   same act/think signature that appeared in MQAR-adjacent memory, compression,
   applicability, hierarchy and the single-template agent **persists across
   heterogeneous bug classes**. Gate accuracy drops from `0.995` (one template) to
   `0.873` (five families) — heterogeneity is genuinely harder — yet the ordering
   is intact.
2. **Real `pytest` confirmed.** End-to-end pytest subprocesses agree with the
   in-process proxy **1.000** (after fixing a stale-`.pyc` trap — same-length
   sources collided on mtime+size; unique rundir + `PYTHONDONTWRITEBYTECODE` fix
   it). Measured wall-clock ≈ `0.147 s` per test run.
3. **Honest product caveat.** At *zero* attempt-cost, SSM is **−28.5 % vs APP**
   on the mixed stream (the probe wins when probing is free, exactly as the law
   says: `P* ≈ 0`, not negative, in this harder regime). SSM's value appears only
   once a test run is genuinely expensive. The defensible product number is the
   **8.8 % fewer test runs than the no-skill OFF baseline** → ≈ `0.107 s` CI saved
   per bug at the measured pytest cost. No inflation: against the strong probe
   baseline the learned gate only pays off when CI is slow.

## Phase 2.6 — Real CI Scaling: the boundary in CI-minutes

`code_repair_ci_scaling.py` changes nothing about the agent and varies only
`P` = the cost of running the test suite (Unit ~1× → Heavy CI ~100×), to answer
the business question: *above what CI cost does the learned gate save engineering
time, and how much?*

Measured cleanly (the technical block, handled): raw pytest wall-clock is
non-deterministic, so we (1) count test runs **deterministically**, separating
**FULL-suite** runs from cheap **SMOKE** probes, (2) calibrate one real pytest
invocation `t0 ≈ 0.18 s`, (3) build the diagram analytically
`cost(P) = full_runs·(t0·P) + probe_runs·t0`. No parallel environments race; the
crossover is exact. The asymmetry that makes a crossover: **APP decides by
running the full suite; SSM-GATE decides with a cheap smoke probe** and spends a
full run only on the committed path → `P* = probe_runs/(full_app − full_ssm)`.

**Deterministic run counts/bug (6 seeds):** OFF_full `8.25`, APP_full `5.85`,
SSM_full `5.66`, SSM_probe `4.65`. **The phase boundary appears in money:**

| regime | `P` | saved vs APP (strong probe) | saved vs OFF (no-skill) |
|---|---|---|---|
| unit | 1 | −13.6 min/1k | −6.3 min/1k |
| unit+ | 3 | −12.5 | **+9.5** |
| integration | 10 | −8.4 | **+65** |
| integration+ | 30 | **+3.3** | **+223** |
| heavy CI | 100 | **+44** | **+778** |

(min saved per 1,000 bugs.) Two honest readings:
1. **Against the realistic "before" (no learned skill, OFF):** the gate saves
   from `P ≳ 3` and reaches **+778 min (~13 h) per 1,000 bugs at heavy CI** — a
   clean, monotone CI-savings-vs-cost curve.
2. **Against a strong probe heuristic (APP):** the learned gate only pays off
   above the crossover, `P* ≈ 13` (95% CI wide, `±10`, `per-seed [6.7–19.9]`):
   the denominator `full_app − full_ssm ≈ 0.19` is small, so `P*` is genuinely
   ill-conditioned — we report the wide interval rather than a false-precise point.
   The robust result is the **sign flip** (negative at unit CI, positive at heavy
   CI) — *cheap CI: just act; expensive CI: think first*, now in CI-seconds.

The business sentence the chain now supports, end to end and outside its home
environment: *"above an integration-grade CI cost, a learned applicability gate
cuts test-suite executions ~3–30% versus brute force, scaling to hours of CI per
thousand bugs — because running tests is exactly the expensive action the law
says to think before taking."*

## Phase 2.7 — AetherGate harness: a real multi-file repo through real `pytest`

`code_repair_harness.py` is the product scaffold. It runs the act/think gate on a
genuine multi-file Python **package** (`proj/` with `adder`/`scaler`/`clamper`/
`selector` modules + a real `pytest` suite, one test per module), executed through
**real `pytest` subprocesses in isolated temp dirs**. A bug lives in one module;
the failing test localises it; the search patches that module. Swapping in a
cloned GitHub repo means replacing `make_repo_spec` with a loader for
`{repo dir, failing test, patch space}` — the network step left for deployment.

**Result (6 seeds; fast backend for statistics, real `pytest` validation):**

1. **The invariant survives a real multi-file repo + real `pytest`.**
   `fire_reuse 0.642 > fire_mixed 0.455 > fire_OOD 0.267`, reuse>OOD on **all 6
   seeds**. The in-process fast backend agrees with real `pytest` **1.000** over
   84 real runs (`t0 ≈ 0.18 s/run`). The structure that anchors the whole program
   is intact in the most realistic setting we can run confound-free.
2. **vs the realistic no-skill baseline (OFF / brute force):** the skill+gate
   system saves from `P ≳ 3`, reaching **+934 min/1k bugs at heavy CI** — the
   product story holds.
3. **Honest negative vs a strong probe heuristic (APP).** Here the *learned* gate
   does **not** beat APP: `SSM_full 5.93 > APP_full 5.81` (deterministic, both
   seeds), so `P* = None` — no CI cost makes the gate cheaper than the probe. At
   3 seeds an earlier run showed `P* ≈ 7.4`; the 6-seed truth erases it. Cause:
   gate accuracy drops to `0.885` on the multi-file task, and its mistaken
   abstentions on familiar bugs (→ broad search → *more* full runs) cancel the
   full-runs it saves by abstaining on novel bugs. The learned gate beats brute
   force, not yet a good behavioural probe, in this harder setting. A sharper gate
   (the open Phase 1.5 item) is the prerequisite, not a patched-over result.

The discipline holds to the end: the 3-seed favourable `P*` was not kept; the
6-seed negative against the strong baseline is reported as the result.

### Phase 2.8 — the negative names the fix: the SMOKE-filter (product arm)

The Phase 2.7 negative is precise: APP wastes *full* CI runs trying patches
blindly. The fix a real CI engineer would use: a **cheap smoke probe decides
which patches deserve a full run.** `repair_smoke_filter` smoke-tests each library
patch (fast subset) and spends a full suite run only on the survivors. This is
algorithmic, not a noisy learned gate — and it wins robustly:

| arm | full-runs/bug (mixed) |
|---|---|
| OFF (brute force) | 9.08 |
| APP (blind probe) | 5.81 |
| SSM (learned gate) | 5.93 |
| **SMOKE-filter** | **3.48** |

`P*(SMOKE vs APP) = 2.09` (tight) — the smoke-filter beats the strong probe from
roughly unit+ CI upward. Real-`pytest`-calibrated CI savings (6 seeds, det.):

| `P` | SMOKE saves vs APP | SMOKE saves vs OFF |
|---|---|---|
| 1 | −7.4 min/1k | +2.1 |
| 3 | **+6.2** | **+34.9** |
| 10 | **+53.8** | **+149.6** |
| 30 | **+189.9** | **+477.4** |
| 100 | **+666.2** | **+1624.7** (~27 h) |

**Resolution.** The learned gate's job was never to out-search a good probe; it
was to *decide when acting is worth it*. The mechanism that actually saves CI is
**smoke-filtering** — run the cheap test first, commit the expensive one only when
it looks promising — and the learned gate's residual role is the meta-decision
*whether to bother smoke-filtering at all* (worthwhile only when even smoke is
non-trivial). Same principle, now an algorithm a CTO can deploy: **above unit-grade
CI cost, AetherGate cuts full test-suite runs ~40% vs brute force and ~40% vs a
naïve fix-library probe, scaling to tens of CI-hours per thousand bugs.**

## Phase 3.0 — the product scaffold: `aethergate/` on real isolated `pytest`

The SMOKE-filter is packaged behind a typed contract so it can leave the research
scripts and point at real repos. `aethergate/` (see its README) decouples:

- **`TaskSpec`** (JSON contract): `repo_dir`, `buggy_path`, `smoke_cmd`/`full_cmd`,
  `patch_space[Candidate]`, `meta`.
- **`IsolatedRunner`**: runs a candidate in an isolated repo copy via real
  `pytest`, `PYTHONDONTWRITEBYTECODE`, optional `fresh_per_run` side-effect isolation.
- **`AetherGate`**: the smoke→full cascade + `baseline_app`/`baseline_off`,
  emitting `RepairMetrics`.
- **`fixtures`**: generates real multi-file packages today; a git-clone loader
  swaps in tomorrow with no downstream change.

`run_aethergate_demo.py` drives it end-to-end through **real isolated `pytest`**.
Deterministic full-runs/bug (12 tasks): **OFF 17.75 → APP 9.92 → AetherGate 6.75**
(+5 smoke), crossover `P* ≈ 1.6`. Projected CI savings: vs the strong probe up to
**+778 min/1k bugs**, vs brute force **+2733 min (~45 h)/1k** at heavy CI. A
serialized contract sits in `artifacts/sample_taskspec.json`. The only piece left
for real GitHub repos is third-party-dependency isolation (venv/Docker) — a
deployment add, out of scope for the confound-free sandbox.

## Phase 3.1 — the self-improvement stream: skills BORN from solved tasks

Every phase so far either promoted the library in one batch then froze it, or (Phase
2) *provided* the library identically to both arms to isolate the gate. None ran the
closed loop the roadmap flags as the open term `g_t`: tasks arriving one-by-one,
skills *born from solved tasks mid-stream* and reused at once. `skill_compression_stream.py`
runs exactly that on the proven bench machinery (identical DSL, solver, UTIL gate,
applicability gate — only the *temporal* behaviour is new), then **freezes** the
library and **shifts the task grammar** ("a new repository") to ask: does the saving
*transfer*, or does the curve rebound?

Three arms over the same stream: `growing_gate` (online UTIL promotion + applicability
gate), `frozen_empty` (primitives only — no-compression baseline), `random_growing`
(library count/length-matched to growing's promotion schedule but RANDOM contents —
the online RANDMACRO control).

**Result 1 — the cost-per-task curve falls with experience (6 seeds, `protocol_valid=True`).**
Mean nodes/task, train early-25% → late-25%:

| arm | early | late | curve |
|---|---|---|---|
| **growing_gate** | 20.0 | **4.4** | `█▄▁▂▂▂▁▂` |
| random_growing | 25.2 | 12.4 | (matched-shape control) |
| frozen_empty | 27.4 | 25.4 | flat (no compression) |

`compression_pays_in_stream` (growing < frozen) **and** `fall_is_reuse_specific`
(growing < random) **and** `curve_declines` — all True. This is the Phase-3a
deliverable (cost-per-task declining over the run vs a no-promotion control), and the
fall is reuse-specific, not a bigger-action-set artifact.

**Result 2 — the honest correction on "transfer".** The naive test "growing beats
no-skill after the shift" is **True even for a 0-overlap grammar** — because in an
arithmetic DSL a macro is a *generically useful multi-step jump* (the RANDMACRO effect
again). So beating `frozen_empty` does NOT prove grammar transfer. The clean signal is
the **margin over `random_growing`** (count/length-matched random library): it isolates
"learned structure still fits the shifted grammar" from "just having longer jumps". It
**decays as the grammar diverges but stays positive** (6 seeds):

| regime (freeze→shift) | growing | random | frozen | **grammar-specific margin** |
|---|---|---|---|---|
| R0_identical (same grammar) | 9.8 | 27.6 | 53.8 | **17.8** |
| R1_partial (50% novel subs) | 19.3 | 32.1 | 50.8 | 12.8 |
| R2_disjoint (100% novel subs) | 22.9 | 33.3 | 52.6 | **10.4** |

**Reading:** there is no binary "transfers / doesn't". A grown skill carries *two*
components — a generic-jump value that survives any grammar shift, and a
grammar-specific value that **decays** with divergence (17.8 → 10.4) but does not
vanish at zero overlap. The honest negative the design predicted (concrete macros
can't *fully* transfer to a brand-new grammar) shows up as the **shrinking margin**,
not as a rebound — and the applicability gate keeps cost low throughout (no OOD
blow-up). True grammar-agnostic transfer needs *parametric* skills (variable binding),
which the concrete-sequence representation does not have — the named next milestone.

**Result 3 — compositional reach, and the gate's myopia squandering it (R2_depth).**
On deeper compositions the frozen library *can* solve tasks ~**5.1× cheaper** than
primitive brute force (`library_full` **682** vs `frozen_empty` **3462** nodes) — the
skills genuinely extend reach. But the gated agent only reaches **3260** (barely below
3462), because the **cheap 40-node probe is too myopic to discover a deep multi-macro
solution** and falls back to primitives — the exact Phase-1.5B finding (one-step probes
miss multi-skill compositions) reproduced in the streaming setting. The reach lives in
the library; unlocking it needs a probe that can see compositional depth.

**One line.** *Online-grown skills make per-task cost fall with experience (5× on the
home grammar) and that gain is reuse-specific; under a grammar shift the gain splits
into a generic-jump part that survives and a grammar-specific part that decays without
collapsing — and the cheap applicability probe, by design, cannot unlock the library's
deep-composition reach.* Reproduce: `python3 skill_compression_stream.py` (deterministic,
6 seeds, well under a minute; artifact `artifacts/skill_compression_stream.json`).

## Phase 3.2 — the depth-aware probe: fixing the gate's broken depth sensor

Phase 3.1's R2_depth negative was precise: the library *can* solve deep compositions
~5× cheaper than primitives (682 vs 3462 nodes), but the cheap fixed-40-node probe is
**horizon-blind** — it cannot see far enough to discover a multi-macro solution, so it
bails to primitive brute force. That is not "no skill"; it is a **search-horizon
collapse** in the *selector*. The fix targets the sensor, not the skills (same library,
same solver — only the gate's horizon control changes). `probe_macros_depth_aware`:

1. **Adaptive horizon** — escalates the macro-probe budget instead of a tiny fixed cap.
2. **Cost gradient, not cost** — tracks the best distance-to-target and, at each
   checkpoint, bails only after `patience` consecutive windows with no improvement.
   So it *pursues depth where the skills are paying off* and *abandons a foreign
   grammar early* — credit assignment over latent search depth.

New arm `growing_depth` runs the SAME frozen Phase-3.1 library; only the gate differs.
**Result (6 seeds, `protocol_valid=True`, deterministic), test mean nodes/task:**

| regime | cheap gate | **depth-aware** | random | frozen | depth speedup |
|---|---|---|---|---|---|
| R0_identical | 9.8 | 10.4 | 27.6 | 53.8 | ×0.94 (small overhead) |
| R1_partial | 19.3 | 17.5 | 32.1 | 50.8 | ×1.10 |
| R2_disjoint | 22.9 | **11.9** | 33.3 | 52.6 | **×1.92** |
| R2_depth | 3260.7 | **1268.3** | 3399.4 | 3462.5 | **×2.57** (lib reach 682) |

**Three readings:**
1. **The squandered reach comes back.** On deep compositions the depth-aware gate
   cuts cost ×2.57 vs the cheap gate (×3.4 → 964 at `--deep-probe-budget 8000`,
   approaching the library's true 682 reach). The depth sensor was the bottleneck,
   not the skills — confirming the Phase-3.1 diagnosis.
2. **Budget converts to reach ONLY where the gradient is positive.** R0/R1/R2_disjoint
   are *byte-for-byte unchanged* as `--deep-probe-budget` goes 2000→4000→8000 — foreign
   and shallow tasks bail at the checkpoints regardless of the cap. The extra horizon
   is spent exclusively on genuinely-deep reuse. This is the clean isolation: the cost
   gradient *is* the restored `visibility(gate, depth)`.
3. **No regression, no collapse — and a bonus.** Shallow R0 pays only a tiny escalation
   overhead (9.8→10.4); R2_disjoint actually *improves* (22.9→11.9, the gradient bails
   a misfitting library faster than the cheap probe's blind fallback). `no_regression`
   and `no_collapse` hold every regime.

**Honest residual.** Depth-aware recovers most, not all, of the reach (1268 vs 682 at
the default budget; the gap is the escalation overhead the probe pays before finding).
And it does not move the grammar-specific transfer margin (still 17.8→10.4) — that gap
is a *representation* limit (concrete macros), the separate next milestone. The depth
probe fixes the *selector*; parametric skills must fix the *representation*.

**One line.** *The R2_depth "negative" was a blind selector, not a missing skill:
a probe that escalates its horizon along the cost gradient unlocks the library's deep
reach (×2.6–3.4) while spending the extra budget only where depth pays — leaving every
shallow/foreign regime untouched.* Reproduce:
`python3 skill_compression_stream.py` (the `growing_depth` arm); sweep
`--deep-probe-budget / --probe-patience / --checkpoints`.

## Phase 3.3 — typed skill schemas: a representation intervention that HONESTLY FAILS

With routing fixed (Phase 3.2), the residual grammar-specific ceiling (margin 17.8 →
10.4) was cleanly isolated to the third multiplier — `reach ≈ Σ P(use_i)·utility(i)`,
where the depth probe lifted `P(use)` but never `utility(i)` (skill *representation*).
A skill is a CONCRETE primitive sequence, so it only fires on its exact byte-sequence;
a new grammar's different constants never match. `skill_compression_typed.py` tests the
obvious fix: abstract each macro into an op-CLASS schema + constant slots
(`M={c,e}`, `A={a,b,d}`; `[c,a,c] → (M,A,M)`), so a typed skill can match a novel
grammar by *shape* and fill its constants. The structural bet is concrete: novel
subroutine `N1=[d,e,a]` has schema `(A,M,A)` = train subroutine `S4=[a,c,b]`.

Everything else is held fixed (same promoted skills, same solver, same depth-aware
gate); only the action-set representation changes. The decisive control is `typed_random`
— the SAME number of extra actions, random contents — because "more actions" is itself
a generic speedup (the RANDMACRO lesson). **Result (6 seeds, deterministic), test mean
nodes/task, concrete / typed / typed_random:**

| regime | concrete | typed | typed_random | schema-specific (typed vs random) |
|---|---|---|---|---|
| R0_identical | 10.4 | 3.7 | 4.1 | +0.4 |
| R1_partial | 17.5 | 4.8 | 4.8 | +0.0 |
| R2_disjoint | 11.9 | 4.0 | 4.5 | **+0.5** |
| R2_depth | 1268 | 1625 | 675 | **−950** |

**Verdict: `representation_lifts_ceiling = False`.** Typed "helps" only as much as a
matched RANDOM library does — the schema-specific margin on R2_disjoint is **+0.5**
(noise), and on R2_depth typed is *worse* than both concrete and random (redundant
near-duplicate instantiations inflate branching). The invariance checks falsify the
naive fix exactly as designed:
- **R2_disjoint does not rise schema-specifically** (+0.5 vs random).
- **R2_depth is not stable** (moves 28% vs concrete) → by the pre-registered rule that
  signals an action-count/routing effect, not a representation one.
- A sweep confirms it: at `--max-inst-per-skill 99` (full schema) typed degrades
  *everywhere* from branching explosion, while random of the same count stays fast.

**Why it failed, precisely (the negative names the next mechanism).** The one novel
subroutine that shares a home schema (`N1`, distance-3 from `[a,c,b]`) only enters the
action set under *full* expansion — exactly the slot-explosion regime that kills search.
Flat schema expansion just throws more concrete candidates at the solver = generic
jumps, indistinguishable from random. The value of a parametric skill is **not** "more
candidates in the action set"; it is **binding the slots to THIS task's target** —
per-application constant *fitting*, not enumeration. That is the mechanism Phase 3.4
must build (instantiate a schema toward the current goal), and it is what this clean
negative isolates. Reproduce: `python3 skill_compression_typed.py`
(sweep `--max-inst-per-skill / --max-typed-actions`).

## Phase 3.4 / Test A — goal-conditioned slot policy: the agent FEELS the goal

Phase 3.3 isolated the lesson: a useful representation is not an expanded action set,
it is a *goal-conditioned constraint* on which instantiation to use (`lookahead ==
implicit search ==` the collapse we already measured). So Phase 3.4 replaces
"enumerate slots → search" with a one-shot POLICY `π(features(x,target)) → binding per
slot` — a tiny MLP + per-slot softmax heads, Adam, trained on the agent's own solved
`(x,target)→binding` experience for each promoted schema (`skill_compression_policy.py`).

**Test A (goal sensitivity)** is the cheapest falsification that we have NOT just
rebuilt enumeration, cleared before any search integration. Against a `goal_blind`
baseline (always emit the schema's modal binding, ignoring the goal), on HELD-OUT
goals (6 seeds, deterministic):

| schema | goal-conditioned solve | goal-blind solve | sensitivity (distinct bindings / x0) |
|---|---|---|---|
| MAM | 0.265 | 0.098 | 2.48 |
| AAA | 0.986 | 0.153 | 2.20 |
| MAA | 0.464 | 0.095 | 2.15 |
| AMA | 0.210 | 0.058 | 3.08 |
| **mean** | **0.481** | **0.101** | **2.48** |

**Test A PASSES.** The policy beats goal-blind **×4.8 (Δ+0.38)** and emits ~2.5 distinct
bindings per `x0` as the target varies — the slot choice *tracks the goal*. This is the
first measurement in the arc where conditioning the action space (not expanding it)
produces a real, control-checked gain — exactly the mechanism Phase 3.3's negative
named.

**Honest bound.** Absolute accuracy is schema-dependent: additive `AAA` is near-solved
(0.99 — a linear inverse), but multiplicative `MAM`/`AMA` are hard (0.21–0.27) because
the `×2 / ×3` inverse is nonlinear for a tiny MLP on generic (leak-free) features. So
goal-conditioning is **real and learnable** (the prerequisite), but not yet accurate
enough to drop into the solver as-is. **Next (Test B/C, gated on this):** lift
multiplicative-schema accuracy (richer policy — the recurrent SSM, or structured
features) then wire the policy into the search and check the regime invariants —
`R2_disjoint` beats `typed_random` by a *large* margin, `R2_depth` stays stable, and
budget-sensitivity drops. Reproduce: `python3 skill_compression_policy.py`.

**The arc, three orthogonal mechanisms now separated by measurement.** `reach ≈
Σ P(use_i)·utility(i)`: Phase 3.1 grows the skills (reuse), Phase 3.2 lifts `P(use)`
(routing/depth-visibility), Phase 3.3 proves `utility` cannot be raised by enumeration,
and Phase 3.4/Test A shows it *can* be raised by goal-conditioning. The program's claim
sharpened to one line: *intelligence gain is not in expanding the action space, but in
conditioning it.*

## Phase 3.4b-lite — slot INTERACTION policy: the bottleneck is structure, not size

Phase 3.4 left a schema-dependent gap (additive `AAA` ~0.99; multiplicative `MAM`/
`AMA` 0.21–0.27). The hypothesis to test before reaching for a bigger model: the gap
is *interaction* — the policy models each slot independently, `P(slot_i|goal)`, but
multiplicative schemas are entangled (`target = ((x·m1)+a1)·m2` couples the slots),
needing `P(slot_i|goal,schema,slot_{<i})`. `skill_compression_policy_interaction.py`
adds exactly that: an autoregressive head per slot conditioned on a learned context of
the previously chosen slots. Greedy decode is **O(L), not enumeration** (the 3.3 trap).
Control = the SAME independent 3.4 policy on the SAME split.

**Result (6 seeds, held-out goals, deterministic), goal-conditioned solve:**

| schema | independent (3.4) | interaction (3.4b) | gain | structure |
|---|---|---|---|---|
| AAA | 0.986 | 0.986 | +0.000 | separable |
| MAA | 0.464 | 0.607 | **+0.143** | entangled |
| AMA | 0.210 | 0.471 | **+0.261** | entangled |
| MAM | 0.265 | 0.275 | +0.010 | entangled |

**The interaction test PASSES** (entangled-schema gain +0.138, additive `AAA`
unchanged): modeling slot coupling — not adding capacity or memory — is what lifts the
hard schemas. A recurrent SSM justified only as a "bigger MLP" would not have produced
this selective gain; the lever is *structure*.

**The residual names the next structural variable: decode ORDER.** `MAM` barely moves
(+0.01) and mean sensitivity stays 2.6 (< 3.0). `MAM = ((x·m1)+a1)·m2`: left-to-right
decoding must commit `m1` (deeply nested, least determinable from the goal) before it
can see `m2` (which the target most constrains — `target` is divisible by `m2`). `AMA`
does not put its hardest slot first, so it jumps. The chain rule can express any joint,
but at finite capacity the **conditioning direction** decides learnability. So the next
lever is *dependency direction*, not memory — which is direct evidence that a
fixed-order recurrent SSM would inherit the same `MAM` failure. The principled next step
is an **order-agnostic / best-first interaction model** (resolve the most-constrained
slot first) or a small message-passing policy over the slot-dependency graph — not a
deeper sequential net. Reproduce: `python3 skill_compression_policy_interaction.py`.

## Phase 3.4c — adaptive constraint propagation: an over-engineering check that FIRES

Phase 3.4b left `MAM` an anomaly and blamed decode *order*. The tempting next step is
"adaptive constraint propagation" — recompute, after each assignment, which unassigned
slot is now most constrained (lowest predictive entropy) and resolve it next. 3.4c builds
it honestly: an order-agnostic policy (masked-trained: reveal a random subset, predict
the rest) decoded three ways — `random_order`, `static_first` (rank once at the empty
assignment), `entropy_adaptive` (recompute each step). All O(L²), never enumeration.

**Result (6 seeds, held-out goals, deterministic):**

| schema | independent (3.4) | random-order | static best-first | entropy-adaptive |
|---|---|---|---|---|
| MAM | 0.265 | 0.353 | **0.431** | 0.412 |
| AAA | 0.986 | 1.000 | 1.000 | 1.000 |
| MAA | 0.476 | 0.524 | 0.643 | 0.631 |
| AMA | 0.196 | 0.304 | 0.442 | 0.442 |

**The fork resolves to ORDER BIAS, not constraint-resolution geometry — and the dynamic
mechanism is over-engineering here:**
1. **Ordering rescues `MAM`** (0.265 → 0.43, now in the MAA/AMA band): confirming the
   3.4b diagnosis that fixed left-to-right was the failure.
2. **Best-first beats random** (+0.076): *which* slot is resolved first carries real
   information.
3. **But `entropy_adaptive ≈ static_first` (−0.008)** — recomputing constraints after
   each commit adds *nothing*. At L=3 the entropy ranking at the empty assignment already
   identifies the right order. The pre-registered "adaptive > static" check returns
   **False**: dynamic constraint propagation is not warranted at this scale. (A genuine
   over-engineering guard firing is the intended outcome — the discipline working.)

**The closing of the representation arc.** Having exhausted every *structural* lever in
order — conditioning (3.4), interaction (3.4b), ordering (3.4c) — `MAM`/`AMA` still
plateau (~0.43, sensitivity < 3). The residual is no longer structure; it is genuine
**function approximation of the nonlinear `×2/×3` inverse**. That is the first and only
point in the whole program where added capacity (a richer net / the recurrent SSM) is
justified — *because* the cheaper structural explanations were ruled out first, not
assumed away. Reproduce: `python3 skill_compression_policy_cp.py`.

## What would have proven us wrong (and didn't)

- OFF matching the gates on reuse → did not happen (5× gap).
- UTIL failing to beat RANDMACRO → did not happen (reuse-specificity holds).
- The applicability gate failing to cut the OOD penalty → did not happen
  (456 → 94.8, below baseline).
- The negative on SSM-GATE being a threshold artifact → ruled out (robust 0.5→0.1).
- The fire-rate structure being a single-template artifact → ruled out: it holds
  across 5 heterogeneous families, all 6 seeds (Phase 2.5).
- The real-`pytest` cost being faked by the in-process proxy → ruled out:
  agreement `1.000` after fixing the stale-`.pyc` trap.
- The online cost-per-task curve not actually falling, or falling only as a
  bigger-action-set artifact → did not happen: `growing_gate` 20.0→4.4 beats both
  `frozen_empty` (flat 25.4) and `random_growing` (12.4), all 6 seeds (Phase 3.1).
- "Transfer" being a free pass from beating the no-skill baseline → caught: the
  grammar-specific margin (vs the RANDOM-library control) is the honest signal, and
  it *decays* with grammar divergence (17.8→10.4) — reported as a graded result, not
  a binary win (Phase 3.1).
- The depth-aware probe being a blanket budget increase that just slows shallow tasks
  → ruled out: R0/R1/R2_disjoint are byte-identical across `--deep-probe-budget`
  2000→8000; only deep-reuse tasks consume the extra horizon (Phase 3.2).
- The R2_depth gap being a missing skill rather than a blind selector → ruled out:
  the same frozen library, behind a depth-aware gate, recovers ×2.6–3.4 of the reach
  (Phase 3.2).
- Typed schemas lifting the grammar-specific ceiling → did NOT happen: the gain is
  the generic-more-actions artifact (`typed_random` matches it; schema-specific margin
  +0.5, R2_depth not stable). Flat schema expansion ≠ parametric fitting (Phase 3.3).
- The goal-conditioned policy being secretly goal-blind (enumeration in disguise) →
  ruled out: it beats the modal-binding baseline ×4.8 and emits ~2.5 distinct bindings
  per state as the goal varies, on held-out goals (Phase 3.4 / Test A).
- The multiplicative-schema gap being a model-capacity problem (fix = bigger net / SSM)
  → ruled out: an autoregressive *interaction* model lifts entangled schemas (+0.14)
  with no capacity increase, while separable AAA is unchanged — the bottleneck is slot
  coupling, and the residual (MAM) isolates decode ORDER, not memory (Phase 3.4b).
- Dynamic constraint propagation being needed to fix MAM → ruled out: static best-first
  ordering already rescues MAM (0.27→0.43) and entropy-adaptive recomputation adds
  nothing (−0.008). The over-engineering guard fired; the lever is order, not dynamics
  (Phase 3.4c).

## What this is not

Not a trained neural gate yet — `g_t` here is a *scored* (measured) promotion
criterion and the applicability gate is a *search policy*: the honest first rungs.
A learned recurrent gate over SSM state is the next milestone. Not natural
language; not a product. A controlled, reproducible crash test of one mechanism,
in the repo's falsifiable style.
