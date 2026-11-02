# NeoCortex-X Stage 14 Runtime Policy Loader Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage13-policy-promotion-plan.md`
- `research/neocortex_x/stage13_policy_promotion.py`

## Цель

Stage 14 загружает promoted policy manifest как runtime policy snapshot:

```text
promoted manifest -> runtime policy table -> smoke decisions
```

Stage 13 создал promotion manifest. Stage 14 проверяет, что manifest можно
использовать как рабочую конфигурацию: probabilities валидны, actions доступны,
rollback metadata присутствует, smoke decisions воспроизводимы.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 13 promotion artifact valid and passed;
- runtime probabilities sum to `1.0`;
- all required executive actions are available;
- rollback metadata exists;
- deterministic smoke decisions are replayable.

## Files

- `research/neocortex_x/stage14_runtime_loader.py`
- `Makefile`
  - add `neocortex-x-stage14-runtime-loader-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage14_runtime_loader.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage14_runtime_loader.py
make neocortex-x-stage14-runtime-loader-smoke
```
