# Roadmap — Intelligence as Experience Compression (research → product)

**Status: research roadmap (hypothesis-driven), not a results claim.** Companion
to `ROADMAP.md` (the narrow KV-memory tool) and `NOTES_cognition_map.md` (the
compass). Same discipline applies: every milestone must name a *measurable*
quantity on our actual JAX/SoA pipeline, every claim stays bounded by evidence in
`../RESULTS.md`. Metaphors that are not yet falsifiable are flagged, not sold.

---

## 0. The hypothesis (stated so it can be killed)

> Intelligence is **not** a large model that re-derives everything from scratch.
> It is a machine that **continuously compresses repeated experience into larger,
> faster, reusable skills** — and then reasons *with* those skills.

Formally, a multi-timescale dynamical system rather than one attention formula:

```text
h_t   = F(h_{t-1}, x_t, M)          # fast working state (≈ SSM)
g_t   = Importance(h_t, x_t)        # is this event worth keeping?
s_t   = s_{t-1} + g_t · h_t         # skill seed accumulator
M_k   = Compress(s_t)  if |s_t|>θ   # promote a stable pattern into a skill
h_t+1 = F(h_t, x_t, M)              # reason *using* the skill library M
```

**The single empirical question that gates this whole roadmap:**

> Does compressing repeated experience into reusable skills **measurably** lower
> cost (compute / steps) and/or raise held-out accuracy on later tasks, versus a
> parameter-matched baseline that does **no** promotion — and does the gain
> *survive* on a benchmark we did not design?

If a no-compression baseline matches the compressing system at matched budget,
the hypothesis is **not supported in that regime**. We say so and move on. This
is the vaccine against self-deception, copied from `ROADMAP.md` Phase 1.

---

## 1. Why this is the right bet now (the gap, not the hype)

Mapping the formula onto what the repo *already* measures:

| Term | Mechanism in repo | Evidence / file | Status |
|---|---|---|---|
| `F` — fast memory | KV-Memory SSM | `RESULTS.md` §2 (recall 0.99 vs 0.20) | ✅ measured |
| `Compress` | macroblocks | `research/codepy/promote_code_tape_macros.py` | ✅ built |
| `s_t` — skill formation | skill ladder | `research/codepy/code_block_skill_ladder.py` | ✅ built |
| `M` — hierarchy | macros-of-macros | `research/codepy/compare_code_block_hierarchy.py` | ⚠️ noisy at depth |
| **`g_t` — importance / promotion** | **the decision "what becomes a skill"** | only a heuristic seed: `utility = success_rate − penalty·depth` (`CODEPY_TURING_COMPLETENESS_PLAN.md`) | ❌ **the open problem** |

**The thesis of this roadmap:** the frontier is *not* another memory block or
another attention variant. It is the **promotion criterion `g_t`** — the rule
that decides which slice of experience is worth compiling into a permanent skill.
That is the one term we have only as a hand-tuned heuristic, and it is the term
the hypothesis lives or dies on.

---

## 2. Phases

### Phase 0 — Ground the formula on existing evidence (cheap, ≈ days) → ✅ DONE
Make the hypothesis auditable before spending compute. No new training.
- [x] Write each of the four solved terms as a one-line *measured* statement with
      its artifact path (table in §1). Reuse committed numbers; invent nothing.
- [x] Pin the **cost metric**: primary = **search nodes expanded to solve a
      held-out task** (`research/skill_compression/`). One number to move.
- [x] Define the **no-promotion baseline** precisely (`OFF` arm: same DSL, same
      solver, promotion disabled) so Phase 1 is a fair A/B.

**Exit:** the hypothesis is a falsifiable, instrumented claim, not a manifesto.

### Phase 1 — Does compression measurably pay? (MAKE-OR-BREAK) → 🟡 CORE HOLDS
First crash test implemented as a self-contained controlled benchmark in
`research/skill_compression/` (program synthesis with recurring subroutines; cost
= search nodes to solve a held-out task; arms `OFF / FREQ / UTIL / RANDMACRO`,
6 seeds, `protocol_valid=True`). Full write-up: `skill_compression/README.md`.
- [x] **Benchmark design.** Reusable-subroutine task family, held-out eval
      explicitly disjoint from train on `(x0,target)`.
- [x] **Promotion gate `g_t` (scored, not yet neural).** `utility =
      reuse·(len−1) − penalty·lib_size`, compared against the `FREQ` heuristic.
- [x] **The A/B crash test + decisive control.** Added a `RANDMACRO` control
      (matched count/length, random contents) to separate "captured reuse" from
      "just having longer jumps."
- [x] **Verdict recorded** in `skill_compression/README.md`.

**Result (6 seeds, mean search nodes, held-out):** OFF `54.4` → learned `~9–12`
(~**5× cheaper**); UTIL `11.6` **beats** RANDMACRO `18.0` → the gain is
**reuse-specific**, not a bigger-action-set artifact.

**Two honest negatives (open Phase-1 follow-ups):**
1. The scored gate does **not** yet robustly beat the simple frequency heuristic
   (6-seed FREQ `8.9` < UTIL `11.6`; UTIL high-variance). The roadmap's "first
   bar" is **not yet cleared** → gate design/tuning, then a *learned recurrent*
   gate over SSM state, is the next milestone.
2. Compression is a **specialization with a cost**: on out-of-distribution
   (no-reuse) tasks the learned library *hurts* (UTIL `458` vs OFF `126`). →
   a deployable skill cache needs an **applicability gate** (when NOT to fire a
   skill), promoted to a first-class Phase-2 item.

**Exit decision:** core claim HOLDS (compression pays, reuse-specific) → proceed
to Phase 1.5 (learned gate + applicability gate) then Phase 2. The *secondary*
bar (beat own heuristic) is open, not failed — no pivot to fallback yet.

### Phase 1.5 — Make the gate real (gating Phase 2) → 🟢 core done
Headline: applicability is the binding constraint, and **how** you decide it
depends on the cost of acting. Behavioural probe wins when trying a skill is
cheap; a learned recurrent `g_t` wins when trying a skill is expensive (clean
crossover at `exec-penalty ≈ 40–60`). This is the seam to the memory line: the
same recurrent-state signal that wins on MQAR now governs skill application.
- [x] **Applicability gate** — a per-task signal for *whether* the skill library
      should fire, to kill the out-of-distribution penalty. Implemented as a cheap
      macro-probe + primitive fallback (no oracle label). **Result: Pareto win** —
      `UTIL+APP` keeps the reuse speedup (`9.5`, ~5.7×) *and* cuts the OOD cost
      from `456` to `94.8`, *below* the OFF baseline `126`. Robust across probe
      budgets. See `skill_compression/README.md`.
- [x] **Phase 1.5B — learned recurrent gate `g_t = σ(A h_{t-1}+B x_t → w·h_T+b)`**
      over per-task skill-reachability probes, trained by BPTT+Adam on the agent's
      own mixed experience (`skill_compression_ssm_gate.py`). **Result: honest
      negative.** The learned static recogniser (`17.0` reuse / `101.8` OOD) does
      NOT beat the behavioural probe `UTIL+APP` (`9.5` / `94.8`), despite `0.93`
      train accuracy; robust to threshold. Diagnosis: one-step features are
      *myopic* (miss multi-skill compositions the probe finds by searching).
      **Reading:** evidence for the repo's *action > prediction* thesis —
      applicability is better decided by *acting* than by static recognition,
      **when probing is cheap**.
- [x] **Phase 1.5C — expensive-execution regime (the completing experiment).**
      Added `--exec-penalty` = real cost of *attempting* a skill. Below the
      boundary the behavioural probe wins (act-then-decide); above it the
      **learned `g_t` wins** (recognise-then-decide). Artifacts:
      `ssm_gate_summary.json` (cheap, negative) + `ssm_gate_expensive_exec.json`
      (P=100, positive).
- [x] **Phase 1.5D — phase diagram: P* with a CI, and an EXACT cost law.**
      `skill_compression_phase_diagram.py` shows both arms are linear in `P` with
      slopes = their skill-attempt rates, so `P* = (base_ssm−base_app)/(1−fire)`.
      Validated to **0.00-node error** (exact). Crossover (6 seeds, 95% CI):
      OOD `P* = 30 [19.5, 40.8]`; mixed 50/50 `P* = 47 [31.5, 62.9]` — both tight;
      reuse-only is ill-conditioned (`fire→1`). **Result: the act-first/think-first
      boundary is a measured constant of the environment.** Artifact:
      `phase_diagram.json`.
- [ ] Beat the frequency heuristic with a *measured* gate (lower-priority now;
      FREQ still wins at 6 seeds — secondary bar, not blocking).

### Phase 2 — Hierarchy & recursion → 🟡 measured (coupling hypothesis refined)
Turned `P*` into an order parameter `P*(D)` over a recursive skill hierarchy
(`skill_compression_hierarchy.py`; `LD` = two `L(D−1)`; adjacent-level gating
keeps search ~1–2 steps so the node budget never explodes).
- [x] **Hierarchical, recursive skills** (`primitive → L1 → L2 → L3`), library
      provided identically to both arms (isolates `P*(D)` from promotion noise).
- [x] **`P*(D)` phase diagram, 6 seeds.** Mixed-stream `P*`: `D1:10.3 → D2:17.3
      → D3:17.0` (tight CIs at D2/3). **Result: Scenario A — the act/think
      boundary is depth-INVARIANT** for a sound adjacent-level fallback. The
      simple coupling hypothesis (`P*↓` with depth) is **refined/refuted**: the
      boundary is *local to each abstraction transition*, not amplified by depth.
- [x] **`fire_rate(D)` + gate accuracy (the positive).** The gate sharpens with
      depth: accuracy `0.92→0.96→0.99`; reuse fire `0.70→0.99→1.00`; OOD fire
      `0.14→0.05→0.03`. Deeper compositional context → strictly more reliable
      applicability recognition. *Depth does not move the price `P*`; it sharpens
      the judgement.* Artifact: `hierarchy_phase_diagram.json`.
- [ ] **Open follow-up:** catastrophic-floor fallback (`L0` floor instead of
      adjacent) and a bottom-up *promotion ladder* (vs provided library) — to test
      whether a sharper, nonlinear `P*(D)` break (Scenario C) exists in a
      different fallback regime.

**Exit:** `P*(D)` is flat under adjacent fallback (boundary is local), but gate
discrimination improves with depth — two separable phenomena, both measured.

### Phase 2.5 — The bridge: phase boundary in a REAL code-repair loop → ✅ HOLDS
Left the synthetic search env for an actual engineering loop (`code_repair_agent.py`):
**real code execution** (`exec` candidate patches + run tests), no LLM/network.
Cost = test runs; `P` = cost of attempting a skill (slow CI / paid call).
- [x] Three arms (OFF broad / APP probe / SSM learned gate) on real test-running.
- [x] **The act-first/think-first law survives**, with tight CIs (6 seeds,
      `gate_acc 0.995`): mixed `P* = 0.7 [0.4, 1.0]`; OOD `P* = −2.0` (gate already
      cheaper at zero cost — a wasted probe is *whole* test runs, the gate's probe
      is fractions). Synthetic `P*≈47` collapses toward 0 in the real loop —
      *running tests is exactly the expensive action the law says favours an
      internal model.* Artifact: `code_repair_summary.json`.
- [x] **Heterogeneity + real `pytest` (`code_repair_real_repos.py`).** Five
      function families (patch-spaces 36/20/18/4/6); a sample run end-to-end
      through real `pytest` subprocesses. **The structure survives:**
      `fire_reuse 0.567 > fire_mixed 0.370 > fire_OOD 0.173`, all 6 seeds
      (`holds_all_seeds=True`); gate_acc `0.873` (heterogeneity is harder, order
      intact). Real `pytest` agrees with the in-process proxy **1.000** (after
      fixing a stale-`.pyc` trap), `0.147 s`/run. Honest product caveat: at zero
      cost SSM is −28.5% vs the probe; the defensible number is **8.8% fewer test
      runs than no-skill OFF**. Artifact: `code_repair_real_repos.json`.
- [x] **Phase 2.6 — Real CI Scaling (`code_repair_ci_scaling.py`).** Vary only
      `P` = test-suite cost (Unit 1× → Heavy 100×); cost computed analytically from
      **deterministic** FULL-run / SMOKE-probe counts × calibrated real pytest
      `t0≈0.18s` (no wall-clock races). **Phase boundary appears in CI-minutes:**
      vs no-skill OFF the gate saves from `P≳3`, up to **+778 min/1k bugs at heavy
      CI**; vs a strong probe it wins above `P*≈13` (95% CI wide ±10 — denominator
      `full_app−full_ssm≈0.19` is ill-conditioned, reported honestly). Robust
      result = the **sign flip**: cheap CI → act; expensive CI → think first.
      Artifact: `ci_scaling.json`.
- [x] **Phase 2.7 — AetherGate harness (`code_repair_harness.py`).** Real
      multi-file Python package + real `pytest` (isolated temp dirs), structured so
      a cloned repo is a drop-in. **Invariant survives:** `fire_reuse 0.642 >
      fire_mixed 0.455 > fire_OOD 0.267`, all 6 seeds; fast↔real-`pytest`
      agreement `1.000`. vs no-skill OFF: **+934 min/1k bugs at heavy CI**.
      **Honest negative vs the strong probe:** `SSM_full 5.93 > APP_full 5.81` →
      `P*=None` (the favourable 3-seed `P*≈7.4` did not survive 6 seeds); gate acc
      drops to `0.885` and reuse misfires cancel OOD savings. The learned gate
      beats brute force, not yet a good probe, here. Artifact:
      `aethergate_harness.json`.
- [x] **Phase 2.8 — SMOKE-filter (the fix the negative named).** Instead of a
      noisy learned gate out-searching the probe, the product mechanism is a cheap
      **smoke probe deciding which patches get a full run** (`repair_smoke_filter`).
      Robust win: full-runs/bug `OFF 9.08 / APP 5.81 / SSM 5.93 / SMOKE 3.48`;
      `P*(SMOKE vs APP)=2.09`. CI savings vs APP up to **+666 min/1k** and vs OFF
      **+1625 min/1k (~27 h)** at heavy CI. Learned gate's residual role: the
      meta-decision *whether to smoke-filter at all*. Artifact:
      `aethergate_harness.json`.
- [x] **Phase 3.0 — product scaffold `aethergate/`.** SMOKE-filter packaged behind
      a typed `TaskSpec` contract + `IsolatedRunner` (real isolated `pytest`,
      `fresh_per_run` option) + `AetherGate` cascade + `fixtures` (real multi-file
      repos; git-clone loader is a drop-in). Driven end-to-end by
      `run_aethergate_demo.py` through **real isolated pytest**: deterministic
      full-runs/bug `OFF 17.75 → APP 9.92 → AetherGate 6.75`, `P*≈1.6`; vs probe
      up to **+778 min/1k**, vs brute force **+2733 min (~45 h)/1k** at heavy CI.
      Contract sample: `artifacts/sample_taskspec.json`.
- [ ] Open (deployment): real GitHub repos + third-party-dependency isolation
      (venv/Docker) — a one-module add; the cascade, contract and metrics are done.

**The chain now runs end-to-end on one idea — *learn to decide when to act*:**
`MQAR → Skill Compression → Applicability → P* → Hierarchy → Code-Repair → CI-$`,
the act/think *structure* persists across heterogeneous real-`pytest` tasks, and
the boundary is now expressed in the unit a CTO pays for (CI-minutes).

### Phase 3 — Product fork
Decide the product *from the Phase-1/2 verdict*, never before.

**3a. If compression pays → "JIT for agents" / adaptive skill cache.**
The wedge: an agent-memory layer that not only *stores* facts (the KV-memory
tool) but *compiles recurring procedures into fast skills*, so per-task cost
**drops as the agent gains experience** — something a frozen LLM cannot do.
- [x] **Phase 3.1 — the self-improvement stream (`skill_compression_stream.py`).**
      The closed loop the roadmap flags as the open `g_t` term: skills BORN online
      from solved tasks, used at once. Cost-per-task **falls 20.0→4.4** over the
      stream (6 seeds, `protocol_valid=True`), beating both `frozen_empty` (flat
      25.4, the no-promotion control) and `random_growing` (12.4) → **the fall is
      reuse-specific.** Then **freeze + grammar shift**: the honest finding is that
      "beats no-skill" is true even at 0-overlap (a macro is a generic jump), so
      transfer is measured as the **margin over a matched RANDOM library** — it
      *decays* with grammar divergence (R0 17.8 → R1 12.8 → R2_disjoint 10.4) but
      stays positive (no rebound; gate prevents OOD blow-up). On deeper tasks the
      library reaches **5.1× cheaper** than primitives (682 vs 3462), but the cheap
      probe is too myopic to unlock it (Phase-1.5B reproduced). Artifact:
      `skill_compression_stream.json`.
- [x] **Phase 3.2 — depth-aware probe (same script, `growing_depth` arm).** The
      R2_depth negative was a *blind selector*, not a missing skill: the cheap
      fixed probe is horizon-blind. `probe_macros_depth_aware` escalates its budget
      and bails on the COST GRADIENT (best-distance stall + patience), pursuing
      depth where skills pay and abandoning a foreign grammar early. Same frozen
      library, only the gate changes. **Result:** R2_depth 3260→**1268** (×2.57;
      ×3.4→964 at budget 8000, vs library reach 682) with **R0/R1/R2_disjoint
      byte-identical across budgets** (extra horizon spent only where the gradient
      is positive) — `no_regression`/`no_collapse` everywhere, R2_disjoint even
      improves (22.9→11.9). The cost gradient *is* the restored `visibility(gate,
      depth)`. **Open:** parametric/variable-binding skills for grammar-agnostic
      transfer (the margin 17.8→10.4 is a representation limit, not a selector one).
- [x] **Phase 3.3 — typed skill schemas (`skill_compression_typed.py`): honest
      negative.** Abstract each macro into an op-class schema + constant slots
      (`M={c,e}`, `A={a,b,d}`) so a skill can match a new grammar by shape. Same
      skills/solver/depth-gate; only the representation changes; control is
      `typed_random` (matched action count, random contents). **`representation_
      lifts_ceiling = False`:** typed beats RANDOM by only +0.5 on R2_disjoint
      (noise), and R2_depth moves 28% (action-count effect, not representation) —
      flat schema expansion is just generic more-jumps, and full expansion explodes
      branching. **The negative names the fix:** a parametric skill's value is
      *binding slots to the current target* (per-application constant FITTING), not
      adding candidates to the action set. Artifact: `skill_compression_typed.json`.
- [x] **Phase 3.4 / Test A — goal-conditioned slot policy (`skill_compression_policy.py`).**
      The fix 3.3 named, done as a POLICY not a search: `π(features(x,target)) →
      slot binding`, one-shot (no lookahead = no enumeration), tiny MLP+softmax+Adam
      trained on own experience. **Test A PASSES** (held-out goals, 6 seeds): beats
      the goal-blind modal baseline ×4.8 (0.481 vs 0.101) and emits ~2.5 distinct
      bindings per state as the goal varies → the slot choice *tracks the goal*, the
      first control-checked representation gain. **Honest bound:** accuracy is
      schema-dependent (additive AAA 0.99; multiplicative MAM/AMA 0.21–0.27 — the
      nonlinear inverse is hard for a tiny MLP). Artifact: `skill_compression_policy.json`.
- [x] **Phase 3.4b-lite — slot interaction policy (`skill_compression_policy_interaction.py`).**
      Tested the right hypothesis BEFORE upgrading the model: the 3.4 gap is *slot
      coupling*, not capacity. An autoregressive head per slot (conditioned on the
      previously chosen slots; greedy O(L), no enumeration) vs the independent 3.4
      control on the same split. **PASSES:** entangled multiplicative schemas rise
      (MAA +0.14, AMA +0.26; mean +0.138) while separable AAA is unchanged — so the
      lever is structure, not a bigger net / SSM. **Residual names the next variable:**
      MAM barely moves (+0.01) — left-to-right decode commits the least-determinable
      slot first; the bottleneck is dependency DIRECTION, not memory (a fixed-order
      SSM would inherit it). Artifact: `skill_compression_policy_interaction.json`.
- [x] **Phase 3.4c — adaptive constraint propagation (`skill_compression_policy_cp.py`):
      over-engineering guard fires.** Order-agnostic masked-trained policy decoded three
      ways (random / static best-first / entropy-adaptive), O(L²), no enumeration.
      **Order is the lever, not dynamics:** best-first rescues MAM (0.27→0.43, into the
      MAA/AMA band) and beats random (+0.076), but `entropy_adaptive ≈ static_first`
      (−0.008) → dynamic constraint propagation adds nothing at L=3. The pre-registered
      "adaptive>static" check returns False (the guard working). Artifact:
      `skill_compression_policy_cp.json`.
- [ ] **Phase 3.4d (now the FIRST justified capacity step):** structural levers
      (conditioning/interaction/ordering) are exhausted; MAM/AMA plateau ~0.43 is genuine
      function-approx of the nonlinear ×2/×3 inverse. Only here is a richer net / the
      recurrent SSM warranted — to raise raw inverse accuracy, not structure.
- [ ] **Phase 3.4 / Test B+C (gated on the inverse-accuracy step):** wire the policy
      into the search; check invariants — R2_disjoint ≫ typed_random, R2_depth stable,
      budget-sensitivity ↓.
- [ ] Cost benchmark vs an attention baseline at matched quality.
- [ ] GPU run; confirm the gain survives scale.

**3b. If only memory pays (Phase 1 breaks) → ship the narrow tool.**
Fall back to `ROADMAP.md` Phase 2a: explicit KV-memory as a cheap
long-context / agent-memory primitive. Honest, smaller, real.

---

## 3. External validation (non-negotiable, copied from Phase-1 discipline)

Every win above is, by default, on a benchmark *we* built. Before any product
claim, the compression result must survive a task family we did **not** design
(an external code/skill-reuse benchmark or a recognized continual-learning
setup), parameter-matched and held-out — exactly as canonical MQAR was the crash
test for the memory claim (`RESULTS.md` §2.4). No external survival → no product.

## 4. What would prove us wrong (kept visible on purpose)

1. No-promotion baseline matches promotion at matched budget → compression is not
   the lever here.
2. The learned `g_t` cannot beat the simple `success_rate − penalty·depth`
   heuristic → no "discovery", just a fancier knob.
3. Hierarchy depth-2 only adds library noise (Stage-4B failure recurs) → the
   matryoshka does not stack in our regime.
4. Gains vanish on an external benchmark → it was task-design, like the objection
   Phase 1 of `ROADMAP.md` was built to answer.

## 5. Guardrail (the project's standing discipline)

We are **not** claiming "the formula for AGI." We are testing one falsifiable
mechanism — *does compressing experience into skills measurably pay?* — on the
infrastructure we already have. Over-claiming is the standing risk; bounded
evidence is the rule. Compass: `NOTES_cognition_map.md`. Evidence:
`../RESULTS.md`. Narrow-tool fallback: `ROADMAP.md`. Frontier: `research/codepy/`.
