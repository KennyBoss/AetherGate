# NeoCortex-X Stage 13 Policy Promotion Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage12-policy-replay-plan.md`
- `research/neocortex_x/stage12_policy_replay.py`

## Цель

Stage 13 превращает успешный replay в promoted runtime policy manifest:

```text
policy replay -> promotion gate -> runtime policy manifest
```

Это операторский слой: успешная policy перестает быть просто экспериментальным
JSON и получает manifest с thresholds, provenance, rollback pointer и runtime
probabilities.

## Promotion Rules

- Stage 12 replay artifact must pass.
- guided action accuracy must be >= `0.95`.
- guided vs state-blind margin must be >= `0.25`.
- all executive actions must be covered.
- manifest must include rollback artifact references.

## Files

- `research/neocortex_x/stage13_policy_promotion.py`
- `Makefile`
  - add `neocortex-x-stage13-promote-policy-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage13_policy_promotion.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage13_policy_promotion.py
make neocortex-x-stage13-promote-policy-smoke
```
