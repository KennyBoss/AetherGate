# The Six-Mechanism Decomposition of Skill-Based Self-Improvement

**Status: a closed causal map, frozen at structural completeness.** This document is the
capstone of the Phase-3 line in `README.md`. It states, as one result, what the
sequence of controlled experiments established: a self-improving skill system's gain
decomposes into **six measurable, separately-controlled mechanisms**, each isolated by
its own falsification, with diminishing marginal return — until only irreducible
function approximation remains. It is frozen *here on purpose*: every structural lever
has been tested, so anything further (capacity fitting) is a different problem and would
contaminate this decomposition.

## The governing identity

```
reach  ≈  Σ_i  P(use_i) · utility(i)
                  │            │
            routing/visibility   representation quality
```

The program's one-line law: **intelligence gain is not in expanding the action space,
but in conditioning it** — and that conditioning factors into the six mechanisms below.

## The six mechanisms (each with its control and verdict)

| # | Mechanism | Phase | What was isolated | Decisive control | Verdict (6 seeds, held-out, deterministic) |
|---|---|---|---|---|---|
| 1 | **Reuse / compression** | 3.1 | skills BORN online from solved tasks lower per-task cost | `frozen_empty` (no promotion) + `random_growing` (matched-count random) | cost/task **20.0→4.4**; beats frozen (flat 25.4) **and** random (12.4) → reuse-specific. Grammar-specific transfer **decays** (17.8→10.4) but does not collapse. |
| 2 | **Routing / depth-visibility** | 3.2 | *when/how far* to engage skills (gate horizon) | cheap fixed probe vs depth-aware; budget sweep | depth-aware **×2.6–3.4** on deep reuse; R0/R1/R2_disjoint *byte-identical across budget* → budget converts to reach only where the cost gradient is positive. `visibility(gate,depth)` restored. |
| 3 | **Representation ≠ enumeration** | 3.3 *(negative)* | does typing the skill (schema expansion) help? | `typed_random` (matched action count, random contents) | **No.** typed ≈ typed_random (schema-specific margin **+0.5**, noise); full expansion explodes branching. Flat enumeration uniformises search. |
| 4 | **Representation via conditioning** | 3.4 / Test A | a one-shot policy `π(x,target,schema)→binding` (no search) | `goal_blind` (modal binding, ignores goal) | beats goal-blind **×4.8** (0.481 vs 0.101), ~2.5 distinct bindings/state as goal varies → the agent *feels the goal*. |
| 5 | **Interaction / slot coupling** | 3.4b | autoregressive `P(slot_i \| …, slot_{<i})` | independent 3.4 policy, same split | entangled schemas rise (**MAA +0.14, AMA +0.26**), separable AAA flat → the lever is *structure*, not capacity/memory. Residual: `MAM` stuck (decode order). |
| 6 | **Ordering** | 3.4c | dynamic vs static slot resolution | random / static-first / entropy-adaptive (same model) | best-first **rescues MAM** (0.27→0.43) and beats random (+0.076); but `adaptive ≈ static` (**−0.008**) → dynamic constraint propagation is **over-engineering** at this scale. Order matters; dynamics do not. |

## The boundary of the structural theory (why freeze here)

After mechanisms 4–6, `MAM`/`AMA` plateau (~0.43, sensitivity < 3) **with every
structural lever exhausted**: conditioning (4), interaction (5), ordering (6) each gave
a diminishing or null marginal gain. The remaining residual is therefore **not
structure** — it is irreducible **function approximation of the nonlinear `×2/×3`
inverse**. That is the *first and only* point in the program where added capacity (a
recurrent SSM / richer features) is justified, precisely because the cheaper structural
explanations were ruled out first.

→ Capacity fitting (`Phase 3.4d`) is a **different problem** and a **separate branch**.
Done here, it would entangle expressivity with structure and destroy the clean causal
map this document records. Hence the freeze.

## What would have falsified the decomposition (and didn't)

- A no-promotion baseline matching growth → didn't (×5, reuse-specific). [1]
- The depth fix being a blanket slowdown → didn't (shallow regimes byte-identical). [2]
- Typing lifting the ceiling → didn't (matched by random). [3]
- The policy being secretly goal-blind → didn't (×4.8, goal-tracking). [4]
- The multiplicative gap being capacity → didn't (interaction fixed it, no capacity). [5]
- Dynamic propagation being necessary → didn't (static best-first suffices). [6]

## Reproduce

Each mechanism is a standalone, deterministic, CPU-only script (6 seeds, well under a
minute each); all emit `protocol_evidence.valid=True`.

```
python3 skill_compression_stream.py              # 1 reuse + 2 routing
python3 skill_compression_typed.py               # 3 representation ≠ enumeration
python3 skill_compression_policy.py              # 4 conditioning (Test A)
python3 skill_compression_policy_interaction.py  # 5 interaction
python3 skill_compression_policy_cp.py           # 6 ordering (over-engineering guard)
```

Artifacts in `artifacts/`. Full narrative and tables: `README.md` (Phases 3.1–3.4c).
