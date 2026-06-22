# research/ — documented experiments (not the product)

These are self-contained research tracks kept for the record. They are **not**
part of the KV-Memory SSM benchmark product in the repo root. Each has honest,
known limits documented below. Full earlier history (including the deleted
release/evidence tooling) lives in the git freeze commit.

## How to run

The evolution scripts import `agent_ssm_core` from the repo root, so run them
from the repo root with the root on the path:

```bash
PYTHONPATH=. python3 research/evolution/evolve_ssm_agents.py
PYTHONPATH=. python3 research/codepy/evolve_code_tape.py --target sum4 --help
```

## codepy/ — safe program synthesis over a constrained DSL

Evolutionary search over fixed-length opcode programs (`evolve_code_blocks.py`,
`evolve_code_tape.py`) plus trace-prior / beam-search / macro-promotion layers.

**Status (honest):**
- PASS: program synthesis, program compression, skill promotion, hierarchical
  reuse, and an early hierarchical-scaling result (Stage 4B success-rate curves).
- NOT solved: **General Algorithmic Scaling.** Autonomous discovery of
  `argmax_index4` (a stateful loop) was never achieved despite trace priors,
  trajectory beams, prefix-value rollouts, survival policies, and a late-game
  structural re-ranker. Zero-error programs only exist via injected/seeded
  references. This is a documented negative result.

## evolution/ — agent/formula evolution and self-play

`evolve_ssm_agents.py`, `evolve_ssm_formulas.py` (constrained formula/NAS
search), `coevolve_ssm_selfplay.py`, plus `run_evolved_agent.py` /
`export_agent_trajectory.py` (replay/export) and `run_hypothesis_loop.py`
(LLM-style formula-hypothesis feedback). These run and produce artifacts; they
are exploratory, not a shipped capability.
