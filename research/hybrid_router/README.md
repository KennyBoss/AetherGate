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
| 0.00 | 1.00 | 1.00 | 0.96 | 0.67 | 0.34 | 0.13 |
| 0.25 | 1.00 | 1.00 | 0.95 | 0.67 | 0.35 | 0.15 |
| 0.50 | 1.00 | 1.00 | 0.94 | 0.68 | 0.33 | 0.15 |
| 0.75 | 1.00 | 1.00 | 0.93 | 0.70 | 0.33 | 0.13 |

- **Efficiency depends only on cue noise, not on overlap** — flat down every
  column (e.g. cue=2.0 → `0.33–0.35` across all overlaps), and the critical
  collapse noise is `N* = 2.0` for *all* overlaps. *Regime identifiability alone
  sets the phase boundary.*
- **Headroom depends only on overlap** (`0.49 → 0.35 → 0.25 → 0.11`), independent
  of cue noise. *Error orthogonality alone sets the magnitude.*

So `routing_value ≈ complementarity(headroom) × identifiability(efficiency)` —
separable and measured. The deep claim made precise: **a computational regime is
recoverable from cheap cues ("the regime is more predictable than the task")
exactly while `cue_noise < N* ≈ 2`; above it routing collapses to the best single
expert no matter how complementary the experts are.**

Run: `python3 hybrid_router_phase_diagram.py`

## Expert replacement: analytic → REAL mechanistic expert (`hybrid_router_real_expert.py`)

Staged from the v1.0 baseline tag: keep the router architecture frozen, replace
the **analytic** memory expert (correct-by-fiat) with a **mechanistic** one whose
errors *emerge*, and re-measure with three built-in anti-illusion guards —
because this is exactly where systems start to "lie beautifully".

**Guards (each caught something real):**
- **G1 headroom = oracle − best_single** + `efficiency = advantage/headroom`.
- **G2 min-regime advantage** — router accuracy in the *weaker* regime (catches
  "always pick the dominant expert" collapse).
- **G3 shuffle control** — router on permuted features must collapse to
  best_single; `shuffle_gap` ≈ the real advantage iff the win is regime inference.

**Two honest failures the guards exposed before any "win" was claimed:**
1. Naïve real memory = an **LRU** store evicts *old* bindings — the *same* axis
   the local expert fails on → **correlated errors → headroom ≈ 0** → routing is
   pointless (`efficiency = nan`, all guards zero). A high-looking accuracy was
   *not* routing.
2. With a complementary mechanism but the *natural* query distribution, memory
   dominates (`0.96`), leaving **headroom ≈ 0.045**: `efficiency = 1.0` looked
   perfect, but `shuffle_gap ≈ 0.04` revealed the absolute stakes were tiny — the
   classic "small oracle-gap ≠ good routing" trap, caught by reporting headroom.

**The honest win.** Memory remodelled as a **write-commit-lag** store (fresh
writes read stale, settled reads correct — eventual consistency) gives errors
*orthogonal* to the local window; balancing the recency bands restores headroom:

| cue_noise | LOC | MEM | best | ORACLE | ROUTER | efficiency | min-regime | shuffle_gap |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.64 | 0.76 | 0.76 | 1.00 | 1.00 | 1.00 | 1.00 | 0.36 |
| 4 | 0.63 | 0.76 | 0.76 | 1.00 | 0.96 | 0.85 | 0.91 | 0.33 |
| 8 | 0.63 | 0.77 | 0.77 | 1.00 | 0.93 | 0.70 | 0.83 | 0.30 |
| 16 | 0.62 | 0.76 | 0.76 | 1.00 | 0.89 | 0.54 | 0.80 | 0.27 |
| 32 | 0.63 | 0.77 | 0.77 | 1.00 | 0.83 | 0.23 | 0.71 | 0.19 |

- Routing over a **real** expert beats best-single by **+0.24**, near oracle, and
  **all three guards pass**: `headroom ≈ 0.25`, `min-regime ≈ 1.0` (bilateral, not
  dominance), `shuffle_gap ≈ advantage` (regime inference, not leakage).
- The **phase structure reproduces** — efficiency collapses monotonically with cue
  noise, crossing 0.5 at `N* ≈ 32` *recency-units* (not comparable to the analytic
  `±1`-unit `N* ≈ 2`; the law, not the constant, transfers).

**Takeaway.** Routing value survives the move from fiat to mechanism — but only
when errors are *orthogonal* (not LRU) and the task distribution gives *headroom*.
The guards, not the headline accuracy, are what make that statement trustworthy.

Run: `python3 hybrid_router_real_expert.py`
