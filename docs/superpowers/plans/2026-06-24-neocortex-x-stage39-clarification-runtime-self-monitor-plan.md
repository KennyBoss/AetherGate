# NeoCortex-X Stage 39 Clarification Runtime Self-Monitor Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage38-clarification-runtime-execution-plan.md`
- `research/neocortex_x/stage38_clarification_runtime_execution.py`

## Цель

Stage 39 добавляет self-monitor probe поверх clarification runtime execution:

```text
clarification runtime trace -> monitor record -> reconstruction metrics
```

Это operational self-monitor: система проверяет, что runtime trace можно
восстановить как direct answer или clarify-then-answer path, что selected keys,
output, target preservation и active compute согласованы с фактическим trace.

## Monitor Contract

Каждый monitor record должен восстановить:

- runtime path: `direct_answer` или `clarify_then_answer`;
- selected clarification keys из step sequence;
- output consistency: answer slot, context, clarification step count;
- target preservation;
- active compute: policy reads, context reads, candidate filters, memory read,
  context questions, emitted answers и total ticks.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 38 execution artifact valid and passed;
- monitor record count equals trace count;
- direct-answer and clarify-then-answer paths are present;
- runtime path reconstruction accuracy >= `0.95`;
- output consistency accuracy >= `0.95`;
- compute reconstruction accuracy >= `0.95`;
- selected key reconstruction accuracy >= `0.95`;
- clarification trace progress accuracy >= `0.95`;
- target preservation = `1.0`;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage39_clarification_runtime_self_monitor.py`
- `Makefile`
  - add `neocortex-x-stage39-clarification-runtime-self-monitor-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage39_clarification_runtime_self_monitor.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage39_clarification_runtime_self_monitor.py
make neocortex-x-stage39-clarification-runtime-self-monitor-smoke
make compile
```

## Guardrail

Stage 39 is a deterministic operational monitor over cached runtime traces.
It is not a consciousness, Phi, live deployment, or production memory-scale
claim.
