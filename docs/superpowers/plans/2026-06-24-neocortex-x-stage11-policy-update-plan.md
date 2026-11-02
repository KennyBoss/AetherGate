# NeoCortex-X Stage 11 Policy Update Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage10-feedback-loop-plan.md`
- `research/neocortex_x/stage10_feedback_loop.py`

## Цель

Stage 11 превращает feedback ledger в следующий policy snapshot:

```text
feedback events -> action weight update -> next policy
```

Это маленький supervised/RL-like update loop. Он не заявляет infinite RL, но
создает воспроизводимый механизм, где успешные действия получают больший вес, а
future policy snapshot можно сохранить и сравнить.

## Update Rule

Для каждого feedback event:

```text
weight[action] += learning_rate * reward
```

Weights ограничены снизу нулем и затем нормализуются в probabilities.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 10 artifact valid and non-empty;
- every reinforced action receives positive weight gain;
- policy probabilities sum to `1.0`;
- replay is deterministic;
- mean reinforced weight gain >= `0.05`.

## Files

- `research/neocortex_x/stage11_policy_update.py`
- `Makefile`
  - add `neocortex-x-stage11-policy-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage11_policy_update.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage11_policy_update.py
make neocortex-x-stage11-policy-smoke
```
