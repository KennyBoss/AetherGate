# Hybrid-router — does routing between attention-like and KV-memory experts pay?

New branch (`hybrid-router`). First falsifiable crash test of the hybrid idea —
*Transformer as a processor over external memory* — without the confounds the
whole line avoided (no LLM, no trained Transformer).

## Why this is not a new postulate

`../../RESULTS.md` already shows two regimes with a crossover: KV-memory beats
attention on long-delay binding (`0.99` vs `0.20`); attention beats memory on
ordinary local prediction. So a hybrid does not need a new hypothesis — the data
say a crossover exists. The open question is whether a **router can find it from
cheap cues** — the act/think gate raised to expert selection.

## Setup

Two analytic experts calibrated to the published regimes:
- **LOCAL** (attention-like): correct within a window `W`, chance beyond it.
- **MEMORY** (KV-store-like): correct on settled bindings regardless of delay,
  but wrong on *stale* (just-overwritten) reads — its realistic failure mode.

A mixed query stream (LOCAL within `W`, RECALL far beyond `W`, some stale). A
learned recurrent router (the reused `SSMGate`) sees two **cheap, noisy** features
— a delay proxy and a noisy staleness flag — and picks an expert per query.

## Result (6 seeds, held-out, deterministic)

| arm | accuracy |
|---|---|
| LOCAL only | 0.527 |
| MEMORY only | 0.757 |
| **ROUTER (learned)** | **0.977** |
| ORACLE (upper bound) | 0.979 |

- **Routing pays, decisively.** Neither single expert exceeds `0.76` — each has a
  real failure regime — yet the router reaches `0.977`, beating the best single
  expert by **+0.22** on every seed.
- **The learned router nearly matches the oracle** (gap `0.002`), from two cheap
  noisy cues, and genuinely switches (memory-pick fraction `0.51`, not "always
  memory").

So the hybrid's value is real *and* recoverable by a small learned router — the
same "decide which mechanism to use" gate, now choosing attention vs retrieval.

## Honest scope

Analytic experts stand in for attention/KV-memory (calibrated to RESULTS, not
trained end-to-end). The next step is to replace the experts with the actual
parameter-matched Transformer and KV-memory SSM from the recall harness and re-run
the same router — turning a calibrated stand-in into a measured hybrid. The
verdict bar stays: `router ≥ max(single experts)` and `router ≈ oracle`.

Run: `python3 hybrid_router_bench.py`

## Phase diagram of computation regimes (`hybrid_router_phase_diagram.py`)

The bench answers *can* a router recover oracle selection. This answers *when* —
decomposing routing value into two independent knobs:

- **complementarity** `(1 − overlap)` — orthogonality of the experts' errors →
  sets the **headroom** `oracle − best_single`.
- **identifiability** `(1 / cue_noise)` — separability of the latent regime from
  the cheap signal → sets how much headroom the router **captures**.

Order parameter: `routing_efficiency = (router − best_single) / headroom ∈ [0,1]`.

**Result (6 seeds, deterministic). The two factors are cleanly orthogonal:**

efficiency grid (rows = overlap, cols = cue_noise):

| overlap \ cue_noise | 0.0 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 1.00 | 0.95 | 0.67 | 0.37 | 0.11 |
| 0.25 | 1.00 | 1.00 | 0.96 | 0.69 | 0.32 | 0.15 |
| 0.50 | 1.00 | 1.00 | 0.95 | 0.62 | 0.38 | 0.16 |
| 0.75 | 1.00 | 1.00 | 0.91 | 0.76 | 0.33 | 0.10 |

- **Efficiency depends only on cue noise, not on overlap** — flat down every
  column, and the critical collapse noise is `N* = 2.0` for *all* overlaps.
  *Regime identifiability alone sets the phase boundary.*
- **Headroom depends only on overlap** (`0.48 → 0.35 → 0.25 → 0.12`), independent
  of cue noise. *Error orthogonality alone sets the magnitude.*

So `routing_value ≈ complementarity(headroom) × identifiability(efficiency)` —
separable and measured. The deep claim made precise: **a computational regime is
recoverable from cheap cues ("the regime is more predictable than the task")
exactly while `cue_noise < N* ≈ 2`; above it routing collapses to the best single
expert no matter how complementary the experts are.**

Run: `python3 hybrid_router_phase_diagram.py`
