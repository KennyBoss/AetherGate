# The Law of Routing Existence

> **Routing is a property of error geometry, not of model capacity.
> A router creates value if and only if its experts fail on orthogonal regions.**

## The law (one equation)

```
routing_advantage = headroom(error_correlation) × efficiency(cue_noise)
```

Two **independent, orthogonal** factors — measured, not asserted:

| factor | what it is | controlled by | behaviour |
|---|---|---|---|
| **headroom** | the value available to route (`oracle − best_single`) | **error correlation** between experts | `≈ 0.24·(1 − corr)` → **0** as errors correlate; flat in cue noise |
| **efficiency** | the fraction of headroom a router captures | **cue identifiability** (`1/noise`) | `1.0 → 0` as cues blur; flat in error correlation |

## The three corollaries (each measured, 6 seeds, deterministic)

1. **No orthogonality → no routing.** At `corr = 1` (experts fail together)
   `headroom = 0` and `routing_advantage = 0` — **even with a perfect signal**
   (`cue = 0`, efficiency would be `1.0`). No amount of cleverness recovers value
   when the experts are wrong *together*.

2. **Identifiability only sets capture, never existence.** Cue noise scales how
   much headroom is captured (`efficiency`), but cannot create headroom. A loud,
   clean signal over correlated experts still routes nothing.

3. **Capacity is irrelevant.** The law contains no term for model size, training,
   or expert strength — only the *geometry* of where experts fail. A weaker pair
   of experts with orthogonal errors routes better than a strong pair with
   correlated errors.

## Why it is trustworthy (anti-illusion guards)

The law was extracted *through* three guards that kill "beautiful but empty" wins:
- **headroom** — exposes the "small oracle-gap ≠ good routing" trap (efficiency
  was `1.0` at headroom `0.045`: perfect capture of nothing).
- **min-regime advantage** — kills the "always pick the dominant expert" collapse
  (router must win in *both* regimes, not one).
- **shuffle control** — kills leakage (router on permuted cues must fall to
  best_single; `shuffle_gap ≈ advantage` proves the win is regime inference).

It also survived the move from **fiat** experts (correct-by-fiat) to **mechanistic**
ones (window-scan vs commit-lag, errors *emergent*) — and exposed a failure
(LRU memory → errors *correlated* with the local expert → headroom `0` → routing
pointless), which is the law predicting its own boundary.

## The testable prediction (for any real hybrid)

> A trained Transformer + external-memory hybrid improves over its best single
> component **iff** their errors are orthogonal. Measure `headroom` first; if it is
> ~0, no router, attention, or fusion layer will help — the problem is geometry,
> not architecture.

## Provenance

`hybrid_router_bench.py` (router beats both, near-oracle) →
`hybrid_router_phase_diagram.py` (complementarity × identifiability) →
`hybrid_router_real_expert.py` (mechanistic experts + guards) →
`hybrid_router_error_geometry.py` (the `corr × cue` factorisation).
Tags: `hybrid-router-v1.0-baseline`, `hybrid-router-v1.1-mechanistic-expert`.
Artifacts: `artifacts/*.json` (deterministic, committed).
