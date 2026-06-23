# From Law to Scale — the mini-house → castle plan

**Goal:** turn the Law of Routing Existence (`LAW_OF_ROUTING_EXISTENCE.md`, tag
`hybrid-router-v2.0`) from a law proven on *stand-in* experts into a demonstration
on *real trained* components, ending in an **efficiency frontier** (accuracy vs
compute) where the law *predicted* a real compute win — and honestly marks where
it doesn't.

**Discipline (unchanged):** every stage is falsifiable, deterministic where
possible, multi-seed, and carries a **kill-condition**. We keep the honest
negatives — they are the credibility. We do **not** claim "transformers are a dead
end"; we claim "this law predicts, before training, when a hybrid wins."

The single thread: **measure `headroom` first; routing pays iff errors orthogonal.**

---

## Stage 0 — First contact with real weights → ✅ DONE (instructive)
`stage0_headroom_real_models.py` trains the harness's real Transformer
(context-capped) + real KV-memory SSM on delayed-binding recall and measures
`headroom = oracle − best_single` over query positions.

**Result (real models, CPU):** `ssm = 1.00`, `transformer = 0.0625` (chance),
`oracle = 1.00`, **`headroom = 0`** — but **not** from correlated errors: from
**single-expert dominance**. The KV-memory SSM is *perfect* on its home task, so
it covers everything. The Law's correct verdict: *don't route — deploy the
dominant expert* (exactly the `RESULTS.md` story).

**Lesson (refines the plan):** real-model headroom needs a task where **neither**
architecture dominates — each must own an *exclusive* win region. → Stage 0.5.

## Stage 0.5 — Headroom hunt on real models → ✅ DONE (honest negative)
`stage05_mixed_headroom.py` builds a mixed stream (RECALL = far binding;
LOCAL-COPY = copy a token `offset` back, in the Transformer window) and trains the
real context-capped Transformer + real KV-memory SSM on it.

**Result (real models):**

| | RECALL | LOCAL | ALL |
|---|---|---|---|
| SSM | 0.317 | **1.000** | 0.649 |
| Transformer | 0.117 | **1.000** | 0.546 |

`oracle = 0.684`, `best_single = 0.649`, **`headroom = 0.035`** (noise-level).

**The transformer has NO exclusive region.** The recurrent SSM does positional
LOCAL-copy *as well* as attention (both `1.000`) and beats it on RECALL — so the
KV-memory SSM is a functional **superset** of the context-capped Transformer on
this family. The tiny headroom is just two recall-weak models making slightly
different errors, not real complementarity.

**Kill-condition fired honestly:** no complementary regime for THIS real pair.
Per the Law that is the correct outcome (no orthogonal failure → no routing
value) — but it means the *router/hybrid* demo is **not supported on these
architectures+tasks**. The supported story is the **efficiency frontier**: the
memory-SSM matches/beats the Transformer at far lower compute (`RESULTS.md`).

**Decision:** skip Stages 1–2 for this pair (nothing to route). Re-route the plan
to **Stage 3 (efficiency frontier)** as the honest facade, OR seek a genuinely
transformer-exclusive capability (long-range *compositional reasoning*, not
retrieval or local copy) — which needs a real LM-grade task, not available cheaply
here. The Law stands; this pair simply isn't complementary.

## Stage 1 — Mini-router on real experts → capture the headroom
Wire the router over the two real experts' per-query outputs, trained on cheap
cues only. Run the **3 anti-illusion guards** (efficiency/headroom, min-regime,
shuffle).
- **Exit:** router ≈ oracle, beats best_single, all guards pass.
- **Kill:** router can't capture headroom from cheap cues → identifiability too
  low on real signals (an honest finding about real cues).

## Stage 2 — The prediction test (the science punchline)
Use Stage-0 `headroom` + a measured `efficiency` to **predict**
`routing_advantage = headroom × efficiency` *before* building the router, then
confirm measured ≈ predicted (within CI).
- **Exit:** predicted ≈ measured → the law is **predictive on real weights**.
- **Kill:** large mismatch → the law is descriptive only at this scale.

## Stage 3 — Efficiency frontier (the facade) → accuracy vs compute
Sweep pure Transformer with growing context (recall needs O(D) context → O(D²)
attention) against router+memory (O(1) lookup). Plot **accuracy vs FLOPs**.
- **Exit:** a real compute crossover — hybrid matches a far-larger-context
  Transformer at constant cost on delayed recall; on local LM the Transformer
  wins (law-consistent). One honest figure.
- **Kill:** no crossover at this scale → frontier is a scale-only effect; report.

## Stage 4 — The Headroom Meter (productization) → a tool
Package: given two models + a task → output `headroom` + a routing recommendation
in seconds, **no training**. The law as a diagnostic CTOs can run.
- **Exit:** a CLI/lib that ingests two model-output sets and emits the verdict.

## Stage 5 — Scale (the castle) → real models, real benchmarks
Swap the tiny experts for a small pretrained Transformer + a real retrieval memory;
run a recognized long-context benchmark; confirm the frontier holds.
- **Exit:** the law's prediction + frontier survive at a recognized scale.
- **Kill:** breaks at scale → bounded result, documented (still a real law at
  small scale).
- *Note:* Stages 3–5 may need GPU / network / deps beyond the offline sandbox;
  marked as deployment, not blockers for Stages 0–2.

---

## Order of build
`Stage 0 (headroom on real models)` → `Stage 1 (router captures it)` →
`Stage 2 (law predicts it)` → `Stage 3 (efficiency frontier facade)` →
`Stage 4 (Headroom Meter)` → `Stage 5 (scale)`.

Smallest first. Each stage is a freeze point (tag) before the next.
