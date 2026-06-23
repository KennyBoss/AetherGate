# NeoCortex-X Stage 4B Learned World-Model Plan

Дата: 2026-06-24.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.

## Цель

Replace the Stage 4 exact deterministic predictor with a learned world-model
predictor inside the planning smoke.

Stage 4 already proves that lookahead over exact predicted deltas can beat greedy.
Stage 4B must learn transition deltas from experience and then use that learned
predictor for planning.

## Approach

Use the Stage 4 corridor world:

1. Generate transition examples `(state, action) -> delta`.
2. Train a compact table predictor from observed transitions.
3. Plan with the learned predictor.
4. Evaluate by executing chosen actions in the true environment.

## Agents And Controls

- `learned_world_model_planner`: lookahead over learned transition deltas.
- `exact_world_model_planner`: Stage 4 upper-bound planner.
- `greedy_action`: one-step reward baseline.
- `no_model_random`: random action baseline.

## Metrics

- `prediction_exact_rate`: learned predicted delta equals true delta on held-out transitions.
- `task_success`.
- `return`.
- `planner_minus_greedy_return`.
- `learned_minus_random_return`.
- `learned_to_exact_return_ratio`.

## Pass/Fail

- `protocol_evidence.valid = true`;
- `learned_world_model_planner.prediction_exact_rate_mean >= 0.95`;
- `learned_world_model_planner.task_success_mean >= 0.85`;
- `learned_minus_greedy_return_mean >= 2.0`;
- `learned_minus_random_return_mean >= 2.0`;
- `learned_to_exact_return_ratio_mean >= 0.90`;
- `heldout_seed_count >= 3`;
- `train_transition_count > 0`.

## Files

- `research/neocortex_x/stage4b_learned_world_model.py`
- `Makefile`
  - add `neocortex-x-stage4b-world-model-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage4b_learned_world_model.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage4b_learned_world_model.py
make neocortex-x-stage4b-world-model-smoke
```
