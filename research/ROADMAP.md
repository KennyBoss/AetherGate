# Roadmap — from honest result to niche tool

**Decided goal:** a narrow, dependable *tool* (Variant 3) — explicit KV-memory as
a cheap long-context / agent-memory primitive — **reached strictly through an
uncompromising Phase 1 (truth).** No productization before external validation.

**One question gates everything:** is our result real, or did we build tasks our
architecture is uniquely good at? Today every win is on a benchmark *we
generated*. Phase 1 is the vaccine against self-deception.

---

## Phase 0 — Lock in what exists (≈80% done)

Convert the work into a citable, clean artifact. Cheap, almost finished.

- [x] Harden `make verify` (`--timeout 600`, `SYSTEM OK`, 22 checks).
- [x] Rewrite `RESULTS.md` into arXiv structure.
- [x] Add `research/NOTES_cognition_map.md` (strategy compass).
- [ ] Commit the above (two commits: tooling+results, then research notes).
- [ ] Final pass on `RESULTS.md` so it is not embarrassing to show a reviewer.

**Exit:** repo is a self-contained, reproducible, citable artifact.

## Phase 1 — External validation (MAKE-OR-BREAK) → ✅ HELD

Ran KV-Memory SSM vs parameter-matched Transformer on canonical **MQAR**, a
benchmark we did **not** design. **Verdict: the result holds.**

- [x] Implement the external benchmark adapter — `make_mqar_recall` (shared
      vocab, scattered pairs, no query marker) + marker-free maskless scoring.
- [x] Extend the architecture — `dynamic-kv-memory` (context-routed key/value/
      read gates), since id-based gating cannot represent shared-vocab MQAR.
- [x] Keep protocol honesty: parameter-matched (`0.96`), held-out, 3 seeds,
      `protocol_evidence.valid = true`, no privileged init.
- [x] Write the verdict into `RESULTS.md` §2.4.

**Result (3 seeds, `make compare-mqar-recall-multiseed`):**
SSM recall **0.9309** vs Transformer **0.0630** (chance `0.0625`), SSM wins 3/3.

**Exit decision:** HOLDS → the advantage is **not** a task-design artifact →
proceed to **Phase 2a**. (Open follow-ups before scaling: full-context
Transformer control on MQAR; sweep vocab/bindings/delay; report the cost story —
throughput + recurrent-state footprint vs attention.)

## Phase 2 — Fork on the Phase 1 verdict

### 2a. If it holds → niche tool
The market already burns budget on huge Transformer context windows (RAG,
needle-in-haystack, long-memory agents). If explicit KV-memory does the same
*cheaper/faster* on recognized benchmarks, that is the wedge.
- [ ] GPU training run; scale parameters; confirm the gap survives scale.
- [ ] Prototype against ONE concrete use-case: agent long-term memory / cache.
- [ ] Cost benchmark vs an attention baseline at matched quality.

### 2b. If it breaks → pivot to stateful synthesis
Reorient to `research/codepy/` (autonomous stateful program synthesis) as the
primary research line — a second potential B2B seam.
- [ ] Promote `research/codepy/CODEPY_TURING_COMPLETENESS_PLAN.md` to active.
- [ ] Define env metrics: state-consistency, write→read fidelity, trajectory
      length before divergence (from the cognition map).

---

## Benchmark choice for Phase 1

Recommendation: **MQAR (Multi-Query Associative Recall)** — the standard synthetic
associative-recall benchmark from the SSM/attention literature (Zoology /
"Based"). Rationale:
- It tests *exactly* our claim (key→value binding recall under length), so a win
  is directly comparable to published attention/SSM numbers.
- It is small enough to run on CPU at our scale — no pretrained LM required.
- It is external and recognized, so it answers the "you built your own task"
  objection head-on.

Why not the alternatives **for Phase 1**:
- **LongBench / needle-in-haystack** — designed for large *pretrained* LMs;
  meaningless at our 8k-param scale. Reserve for Phase 2a after scaling.
- **bAbI** — reasoning-task bundle, broader than our binding claim; noisier
  signal for the specific question we must answer first.

**Plan:** MQAR in Phase 1 as the crash test; LongBench/needle later (Phase 2a)
*only if* MQAR holds and we have scaled.

---

## Guardrail (the project's discipline)

We are **not** competing with mature Transformer LLMs as products. The standing
risk is **over-claiming** — inflating a narrow recall result into "beats LLMs."
Every claim stays bounded by evidence in `RESULTS.md`. Compass:
`NOTES_cognition_map.md`. Evidence: `RESULTS.md`. Frontier: `research/codepy/`.
