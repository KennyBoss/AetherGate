# NeoCortex-X Stage 36 Clarification Policy Promotion Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage34-clarification-policy-update-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage35-clarification-policy-replay-plan.md`
- `research/neocortex_x/stage34_clarification_policy_update.py`
- `research/neocortex_x/stage35_clarification_policy_replay.py`

## Цель

Stage 36 промотит replay-validated clarification-key policy в runtime manifest:

```text
Stage 34 policy + Stage 35 replay evidence -> promoted clarification runtime manifest
```

Это не обучает новую policy. Это фиксирует проверенную policy snapshot как
runtime-ready artifact с explicit rollback metadata.

## Promotion Contract

- Stage 35 replay artifact должен быть valid/pass;
- Stage 34 policy artifact должен быть valid/pass;
- `policy_id`, `ranked_keys` и `top_key` должны совпадать между replay и policy;
- runtime manifest должен содержать:
  - `runtime_ranked_keys`;
  - `runtime_top_key`;
  - `runtime_probabilities`;
  - runtime selection contract;
  - replay evidence metrics;
  - rollback policy/command.

## Pass/Fail

- `protocol_evidence.valid = true`;
- replay artifact valid and passed;
- policy artifact valid and passed;
- policy ids match;
- ranked keys match;
- top key matches;
- policy resolution accuracy >= `0.95`;
- policy ambiguous resolution accuracy >= `0.95`;
- target preservation = `1.0`;
- max steps <= number of context dimensions;
- policy mean ambiguous steps beats fixed-order control by configured margin;
- policy accuracy beats always-top-key control by configured margin;
- selected keys are from ranked list;
- runtime contract present;
- rollback present.

## Files

- `research/neocortex_x/stage36_clarification_policy_promotion.py`
- `Makefile`
  - add `neocortex-x-stage36-clarification-policy-promotion-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage36_clarification_policy_promotion.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage36_clarification_policy_promotion.py
make neocortex-x-stage36-clarification-policy-promotion-smoke
```
