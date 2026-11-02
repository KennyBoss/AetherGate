# Headroom Meter

**Should you combine two systems? Answer in seconds — no training, no weights.**

A diagnostic built on the **Law of Routing Existence**
(`../LAW_OF_ROUTING_EXISTENCE.md`, tag `hybrid-router-v2.0`):

> `routing_advantage = headroom(error_orthogonality) × efficiency(identifiability)`
> A router / ensemble / hybrid creates value **iff** the two systems fail on
> orthogonal inputs. If `headroom ≈ 0`, no router, attention, or fusion will help —
> the problem is geometry, not architecture.

The Meter takes the one thing every team already has — **per-item correctness of
two systems on the same eval set** — and returns a plain verdict before you spend
a dollar building a hybrid.

## Why it exists

Combining models (ensembles, MoE routers, retrieval+LLM hybrids) is expensive and
usually decided by vibes. This law says the outcome is **predictable from the
error geometry alone**. The Meter makes that prediction in one command.

## Install / run

Pure Python, no dependencies.

```bash
# two files of 0/1 correctness (json list, or one-per-line csv/txt)
python3 headroom_meter.py --a a_correct.json --b b_correct.json

# optional per-item cue -> also estimates how much headroom a cheap router captures
python3 headroom_meter.py --a a.csv --b b.csv --cue cue.csv

# pipeline mode
echo '{"a":[1,0,1],"b":[0,1,1]}' | python3 headroom_meter.py --stdin --json
```

## The three verdicts

| verdict | when | meaning |
|---|---|---|
| **COMBINE** | `headroom > 0.03`, errors orthogonal | a router can lift accuracy toward the oracle — build it |
| **DEPLOY-DOMINANT** | one system ≈ perfect (`best > 0.97`) | combining can't help; ship the strong one alone |
| **DON'T-COMBINE** | `headroom ≈ 0`, errors correlated | both fail on the same inputs; no architecture fixes geometry |

## What it reports

- `headroom = oracle − best_single` — the value a perfect router could add.
- failure overlap (`both / onlyA / onlyB / neither`) — where the systems disagree.
- `error_orthogonality` — fraction of solved items exclusive to one system
  (1.0 = fully complementary, 0 = fully redundant).
- `cue_captured_fraction` (if a cue is given) — how much headroom a cheap,
  threshold-on-the-cue router actually recovers (identifiability).

## Validated against this project's real results

- Stage 0 (real KV-memory SSM vs context-capped Transformer on recall):
  `best = 1.0` → **DEPLOY-DOMINANT** — exactly the Law's call (memory dominates,
  don't route). See `../stage0_headroom_real_models.py`.
- Orthogonal-failure synthetic → **COMBINE**; correlated-failure synthetic →
  **DON'T-COMBINE**. The Meter reproduces all three regimes.

## Honest framing (how to sell it)

Not "SSM beats Transformers." The defensible, measured claim:

> *For long-delay retrieval, a memory-augmented SSM holds constant state cost,
> while a Transformer's cost grows with the required context length.* (Stage 3
> efficiency frontier: memory matches recall at far lower state cost; the gap
> widens with the dependency length.)

The Meter is the general tool behind that finding: **measure headroom first;
combine only when the geometry says it pays.**
