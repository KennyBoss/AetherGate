# NeoCortex-X Stage 15 Runtime Execution Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage14-runtime-loader-plan.md`
- `research/neocortex_x/stage14_runtime_loader.py`

## Цель

Stage 15 выполняет загруженную runtime policy в маленьком end-to-end цикле:

```text
runtime policy + temporal memory state -> action output -> execution trace
```

Это первый слой, где promoted policy реально используется для выдачи answer,
memory update answer или request-more-evidence output.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 14 runtime artifact valid and passed;
- Stage 8 temporal artifact valid and passed;
- execution success >= `0.95`;
- trace count equals temporal episode count;
- active compute is recorded for every trace;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage15_runtime_execution.py`
- `Makefile`
  - add `neocortex-x-stage15-runtime-execution-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage15_runtime_execution.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage15_runtime_execution.py
make neocortex-x-stage15-runtime-execution-smoke
```
