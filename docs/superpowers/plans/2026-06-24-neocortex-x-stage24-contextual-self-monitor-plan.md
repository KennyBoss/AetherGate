# NeoCortex-X Stage 24 Contextual Self-Monitor Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage23-contextual-runtime-execution-plan.md`
- `research/neocortex_x/stage23_contextual_runtime_execution.py`

## Цель

Stage 24 добавляет self-monitor probe поверх contextual runtime execution:

```text
contextual runtime trace -> monitor record -> reconstruction metrics
```

Это правильное место для self-monitor: после того как NeoCortex-X уже умеет
контекстную память, contextual action gate, feedback, policy replay, promotion,
runtime loader и execution traces.

## Monitor Contract

Каждый monitor record должен восстановить:

- `selected_action`;
- `context_decision`;
- output type;
- compute flags: memory read, context question, evidence request, policy read;
- context class:
  - `direct_context_match`;
  - `scoped_different_context`;
  - `missing_context`;
  - `same_context_conflict`.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 23 execution artifact valid and passed;
- monitor record count equals trace count;
- action reconstruction accuracy >= `0.95`;
- context decision reconstruction accuracy >= `0.95`;
- output type reconstruction accuracy >= `0.95`;
- compute reconstruction accuracy >= `0.95`;
- all four context classes present;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage24_contextual_self_monitor.py`
- `Makefile`
  - add `neocortex-x-stage24-contextual-self-monitor-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage24_contextual_self_monitor.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage24_contextual_self_monitor.py
make neocortex-x-stage24-contextual-self-monitor-smoke
```
