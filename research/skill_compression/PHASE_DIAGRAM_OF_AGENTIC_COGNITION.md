# A Phase Diagram of Agentic Cognition — from associative memory to CI economics

**Status: research report over committed, reproducible evidence.** Every number
below comes from a deterministic script in this directory and its JSON artifact.
Companion running log: `README.md`. Strategy: `../ROADMAP_SKILL_COMPRESSION.md`.

## Thesis

One mechanism — *learn to decide **when to act*** — carried, falsifiably, across
eight task levels from synthetic associative recall to a real isolated-`pytest`
code-repair agent. The durable invariant is not a single constant but a
**structure**:

> A learned internal state does not always reduce the *number* of actions, but it
> stably reshapes their *distribution* across familiar / mixed / novel regimes
> (`fire_reuse > fire_mixed > fire_OOD`); as the cost of acting rises this becomes
> measurable compute/CI savings. **Cheap actions → act first; costly actions →
> think first.**

## The chain (each link reproducible)

| # | Stage | Question | Headline result | Script / artifact |
|---|---|---|---|---|
| 1 | Memory (KV/MQAR) | does dynamic state beat static attention on delayed recall? | recall **0.99 vs 0.20** | `../../RESULTS.md` |
| 2 | Skill compression | does compressing experience pay, reuse-specifically? | ~**5×** fewer search nodes; beats matched random macros | `skill_compression_bench.py` |
| 3 | Applicability gate | can a probe say *when* a skill applies? | OOD penalty `456→95`, Pareto win | `skill_compression_bench.py` |
| 4 | Cost boundary `P*` | when does think-first beat act-first? | exact cost law, `P*` with CIs | `skill_compression_phase_diagram.py` |
| 4b | Learned recurrent gate | can state learn applicability? | honest negative when probing cheap; **wins when acting is costly** (crossover) | `skill_compression_ssm_gate.py` |
| 5 | Hierarchy `P*(D)` | does depth move the boundary? | `P*(D)≈const` (Scenario A); gate **judgement sharpens** acc `0.92→0.99` | `skill_compression_hierarchy.py` |
| 6 | Real code-repair | does the structure survive real `exec`/`pytest`? | structure holds; `P*` collapses toward 0 in real loop | `code_repair_agent.py`, `code_repair_real_repos.py` |
| 7 | CI economics | the boundary in money? | sign-flip in CI-minutes; `+778`/`+934` min/1k at heavy CI | `code_repair_ci_scaling.py`, `code_repair_harness.py` |
| 8 | Product (AetherGate) | deployable on real isolated `pytest`? | OFF 17.75 → APP 9.92 → **AetherGate 6.75** full-runs; up to **+2733 min (~45 h)/1k** | `aethergate/`, `run_aethergate_demo.py` |

## The two laws

**1. Cost law (exact).** Both arms are linear in the per-attempt cost `P`:
`C_app = base_app + P`, `C_ssm = base_ssm + fire·P`, so
`P* = (base_ssm − base_app)/(1 − fire)`. Validated to **0.00-node error** (§4).
The act/think boundary is a *derived constant of the environment*, not a fit.

**2. Filter-cascade law (engineering).** When probing is cheap, *act first*
(behavioural probe beats a learned recogniser, §4b negative). When acting is
costly, *think first*. The general form is a **cascade**: `cheap filter → model
→ expensive action` — realised as SMOKE→(gate)→full CI, which beats both brute
force and a blind probe (Stage 8).

## The invariant that survived everything

`fire_reuse > fire_mixed > fire_OOD`, held on all seeds at every stage where it
was measurable — synthetic gate, hierarchy (and the gate *sharpened* with depth:
acc `0.92→0.96→0.99`), heterogeneous families (5 patch-spaces), and a real
multi-file repo through real `pytest`.

## Honest negatives (kept, not hidden)

- A learned static gate does **not** beat a behavioural probe when probing is
  cheap (Stage 4b) — evidence for *action over prediction*.
- `P*(D)` does **not** fall with hierarchy depth under a sound fallback (Stage 5)
  — the coupling hypothesis was refined to Scenario A.
- On multi-file repos the learned gate does **not** beat the strong probe
  (`P*=None` at 6 seeds); the favourable 3-seed `P*` was discarded. The win comes
  from the *SMOKE-filter*, an algorithm, not a noisy gate (Stages 7–8).
- Reuse-only `P*` is ill-conditioned (`fire→1`); reported as a wide interval.

## Product (AetherGate)

`aethergate/` packages the SMOKE→full cascade behind a typed `TaskSpec` contract
with an `IsolatedRunner` (real `pytest`), validated end-to-end on real multi-file
repos. A `repo_loader` builds a `TaskSpec` from any local checkout (auto patch
space by single-token mutation) and from a git clone (network-gated). Deployment
gap: third-party-dependency isolation (venv/Docker) for arbitrary GitHub repos.

## Reproduce

```bash
pip install -r requirements.txt          # numpy, pytest
python3 skill_compression_bench.py --seeds 11,17,23,41,73,101
python3 skill_compression_phase_diagram.py
python3 skill_compression_hierarchy.py
python3 code_repair_ci_scaling.py
python3 run_aethergate_demo.py --tasks 12        # real isolated pytest
```

## Limitations

Synthetic-to-small-real scale; arithmetic/operator bug class; the cognitive
reading (`fire`-structure ≈ familiar/novel discrimination) is an interpretation
of a measured gate, not a claim about biological cognition. The single missing
external test is real GitHub repositories with their dependencies.
