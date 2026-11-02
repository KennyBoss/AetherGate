# NeoCortex-X Stage 9 Memory Action Gate Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage8-temporal-freshness-plan.md`
- `research/neocortex_x/stage8_temporal_freshness.py`

## Цель

Stage 9 связывает temporal memory с действием агента.

Stage 8 решает, что происходит с фактом во времени: `keep`, `refresh`,
`flag_conflict`. Stage 9 превращает это в executive action:

```text
memory freshness action -> agent action gate
```

## Gate Mapping

- `keep` -> `answer_from_memory`;
- `refresh` -> `update_memory_and_answer`;
- `flag_conflict` -> `request_more_evidence`.

## Controls

- `memory_action_gate`: uses Stage 8 `future_action`.
- `always_answer_control`: always answers from existing memory.
- `always_search_control`: always requests more evidence.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 8 artifact valid and non-empty;
- all three executive actions are present;
- gate accuracy >= `0.95`;
- gate beats always-answer by >= `0.25`;
- gate beats always-search by >= `0.25`.

## Files

- `research/neocortex_x/stage9_action_gate.py`
- `Makefile`
  - add `neocortex-x-stage9-action-gate-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage9_action_gate.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage9_action_gate.py
make neocortex-x-stage9-action-gate-smoke
```
