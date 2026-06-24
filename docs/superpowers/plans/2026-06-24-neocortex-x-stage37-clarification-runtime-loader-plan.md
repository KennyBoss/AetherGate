# NeoCortex-X Stage 37 Clarification Runtime Loader Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage36-clarification-policy-promotion-plan.md`
- `research/neocortex_x/stage36_clarification_policy_promotion.py`

## Цель

Stage 37 загружает promoted clarification-key manifest как runtime snapshot:

```text
promoted clarification manifest -> runtime key-selection policy + smoke decisions
```

Это runtime analog для Stage 36: policy уже replay-validated и promoted, теперь
нужно проверить, что runtime использует `runtime_ranked_keys` для выбора
уточняющего context key.

## Runtime Contract

- загрузить Stage 36 manifest;
- проверить `runtime_ranked_keys`, `runtime_top_key`, probabilities, rollback;
- создать deterministic `runtime_policy_id`;
- выбирать первый missing key из `runtime_ranked_keys`;
- если все context dimensions известны, не выбирать clarification key;
- для unknown context shape использовать fallback action.

## Pass/Fail

- `protocol_evidence.valid = true`;
- promotion artifact valid and passed;
- ranked keys cover all context dimensions;
- runtime probabilities sum to `1.0`;
- top key equals first ranked key;
- runtime contract present;
- rollback present;
- smoke decisions choose expected missing keys;
- full context produces no clarification key;
- unknown context fallback present;
- selected keys are from ranked list;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage37_clarification_runtime_loader.py`
- `Makefile`
  - add `neocortex-x-stage37-clarification-runtime-loader-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage37_clarification_runtime_loader.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage37_clarification_runtime_loader.py
make neocortex-x-stage37-clarification-runtime-loader-smoke
```
