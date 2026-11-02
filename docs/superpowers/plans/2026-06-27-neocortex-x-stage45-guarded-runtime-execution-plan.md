# NeoCortex-X Stage 45 Guarded Runtime Execution Plan

Дата: 2026-06-27.
Builds on:
- `docs/superpowers/plans/2026-06-27-neocortex-x-stage44-guarded-runtime-loader-plan.md`
- `research/neocortex_x/stage44_guarded_runtime_loader.py`
- `research/neocortex_x/stage42_clean_adversarial_replay.py`

## Цель

Stage 45 выполняет guarded runtime snapshot из Stage 44 на clean + adversarial
records из Stage 42:

```text
Stage 44 guarded runtime snapshot
  + Stage 42 clean/adversarial records
  -> guarded runtime traces
  -> answers only when safe
  -> conflict/noisy/drift routed to guard actions
```

Это первый шаг после loader, где guard проверяется на полном replay наборе, а
не только на hand-written smoke cases.

## Execution Contract

Stage 45 должен:

- загрузить Stage 44 runtime snapshot;
- загрузить Stage 42 candidate replay records;
- проверить, что candidate ids совпадают;
- выполнить все clean и adversarial records;
- сохранить clean answers;
- блокировать answers для same-context contradiction, noisy context и temporal
  drift;
- сохранить target на safe-context adversarial cases;
- записать output type, guard route, selected action и active compute;
- показать `unsafe_answer_count = 0`;
- сохранить deterministic replay evidence.

## Pass/Fail

- Stage 44 artifact valid;
- Stage 42 artifact valid;
- candidate ids match;
- runtime guard active;
- trace count matches Stage 42 records;
- clean answers preserved;
- clean cases safe;
- adversarial cases safe;
- no-answer risks blocked;
- unsafe answers = `0`;
- safe-context target preserved;
- guard applied to all traces;
- evidence/context-integrity/validity routes present;
- answers emitted for clean or safe records;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage45_guarded_runtime_execution.py`
- `Makefile`
  - add `neocortex-x-stage45-guarded-runtime-execution-smoke`
- `research/neocortex_x/README.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-risk-register-and-hard-gates.md`
- `🧠 NEOCORTEX-X.md`

## Artifact

- `artifacts/neocortex_x/stage45_guarded_runtime_execution.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage45_guarded_runtime_execution.py
make neocortex-x-stage45-guarded-runtime-execution-smoke
make compile
```

## Guardrail

Stage 45 executes a guarded cached-replay runtime snapshot. It is not live
internet memory, not production-scale memory, not AGI, and not consciousness
evidence.
