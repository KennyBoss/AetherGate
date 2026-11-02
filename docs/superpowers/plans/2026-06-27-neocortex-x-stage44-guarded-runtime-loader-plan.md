# NeoCortex-X Stage 44 Guarded Runtime Loader Plan

Дата: 2026-06-27.
Builds on:
- `docs/superpowers/plans/2026-06-27-neocortex-x-stage43-guarded-policy-promotion-plan.md`
- `research/neocortex_x/stage43_guarded_policy_promotion.py`
- `research/neocortex_x/stage44_guarded_runtime_loader.py`

## Цель

Stage 44 загружает guarded promotion manifest из Stage 43 в runtime snapshot:

```text
Stage 43 guarded promotion manifest
  -> validate guard rules + rollback + claim scope
  -> guarded runtime snapshot
  -> smoke decisions for clean/conflict/noisy/drift/unknown context
```

Это не production deployment. Loader должен доказать, что guard реально
активен в runtime snapshot, а не остался только текстом в manifest.

## Loader Contract

Stage 44 должен:

- загрузить Stage 43 guarded promotion artifact;
- проверить manifest id, candidate id, guard id;
- проверить ranked keys, probabilities и context dimensions;
- проверить guard rules для same-context contradiction, noisy context и
  temporal drift;
- проверить rollback metadata;
- проверить claim guardrails и scope `guarded_cached_replay_runtime_candidate`;
- создать runtime snapshot с `guard_active = true`;
- smoke-test:
  - missing context -> ask disambiguating context;
  - clean unique target -> allow contextual answer;
  - same-context contradiction -> request evidence;
  - noisy context -> request context integrity check;
  - temporal drift -> request validity window check;
  - unknown context shape -> unknown context fallback;
- сохранить deterministic replay evidence.

## Pass/Fail

- Stage 43 artifact valid and promotion allowed;
- guard active;
- claim guardrails active;
- rollback present;
- runtime scope limited;
- smoke actions correct;
- smoke keys correct;
- answer permissions correct;
- guard applied to all smoke cases;
- conflict answers blocked;
- clean answer allowed;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage44_guarded_runtime_loader.py`
- `Makefile`
  - add `neocortex-x-stage44-guarded-runtime-loader-smoke`
- `research/neocortex_x/README.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-risk-register-and-hard-gates.md`
- `🧠 NEOCORTEX-X.md`

## Artifact

- `artifacts/neocortex_x/stage44_guarded_runtime_loader.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage44_guarded_runtime_loader.py
make neocortex-x-stage44-guarded-runtime-loader-smoke
make compile
```

## Guardrail

Stage 44 loads only a guarded cached-replay runtime snapshot. It is not live
internet memory, not production-scale memory, not AGI, and not consciousness
evidence.
