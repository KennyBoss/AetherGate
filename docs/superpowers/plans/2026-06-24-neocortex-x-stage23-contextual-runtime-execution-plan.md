# NeoCortex-X Stage 23 Contextual Runtime Execution Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage16-contextual-memory-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage22-contextual-runtime-loader-plan.md`
- `research/neocortex_x/stage16_contextual_memory.py`
- `research/neocortex_x/stage22_contextual_runtime_loader.py`

## Цель

Stage 23 выполняет загруженную contextual runtime policy на cached contextual
episodes:

```text
runtime contextual policy + context episodes -> execution traces
```

Это контекстный аналог Stage 15. Система должна не только выбрать action, но и
сформировать правильный bounded output:

- `context_match` -> обычный answer;
- `same_subject_different_context` -> scoped answer;
- `request_context` -> clarification question;
- `flag_conflict` -> evidence request.

## Trace Contract

Каждый trace содержит:

- `observation_id`;
- `context_decision`;
- selected and expected action;
- output payload;
- success flag;
- active compute summary;
- runtime policy id.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 22 runtime artifact valid and passed;
- Stage 16 contextual memory artifact valid and passed;
- execution success >= `0.95`;
- trace count equals contextual episode count;
- all four output types present;
- no false evidence request for `same_subject_different_context`;
- missing context produces clarification output;
- active compute recorded;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage23_contextual_runtime_execution.py`
- `Makefile`
  - add `neocortex-x-stage23-contextual-runtime-execution-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage23_contextual_runtime_execution.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage23_contextual_runtime_execution.py
make neocortex-x-stage23-contextual-runtime-execution-smoke
```
