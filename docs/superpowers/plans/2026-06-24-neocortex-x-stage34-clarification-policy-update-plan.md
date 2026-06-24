# NeoCortex-X Stage 34 Clarification Policy Update Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage33-clarification-feedback-ledger-plan.md`
- `research/neocortex_x/stage33_clarification_feedback_ledger.py`

## Цель

Stage 34 превращает feedback по уточняющим context keys в policy snapshot:

```text
clarification feedback -> key weights -> normalized key probabilities
```

Это bridge от ручного выбора informative missing key к обучаемой политике,
которую можно replay/promote в следующих stages.

## Policy Contract

- policy содержит все context dimensions из Stage 30;
- successful `reinforce_key` events дают положительный gain;
- non-reinforced keys не меняются;
- probabilities нормализуются в сумму `1.0`;
- top/ranked keys выводятся детерминированно;
- replay deterministic.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 33 artifact valid and passed;
- updates present;
- reinforced keys present;
- reinforced key gains positive;
- non-reinforced keys unchanged;
- mean reinforced gain >= configured threshold;
- probability sum = `1.0`;
- top key is reinforced;
- reinforced keys rank before non-reinforced keys;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage34_clarification_policy_update.py`
- `Makefile`
  - add `neocortex-x-stage34-clarification-policy-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage34_clarification_policy_update.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage34_clarification_policy_update.py
make neocortex-x-stage34-clarification-policy-smoke
```
