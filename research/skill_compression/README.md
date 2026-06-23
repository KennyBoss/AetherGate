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

## What this is not

Not a trained neural gate yet — `g_t` here is a *scored* (measured) promotion
criterion and the applicability gate is a *search policy*: the honest first rungs.
A learned recurrent gate over SSM state is the next milestone. Not natural
language; not a product. A controlled, reproducible crash test of one mechanism,
in the repo's falsifiable style.
