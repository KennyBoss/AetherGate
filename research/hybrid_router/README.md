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
