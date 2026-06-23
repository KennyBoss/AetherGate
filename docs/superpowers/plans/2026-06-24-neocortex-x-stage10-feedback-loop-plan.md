# NeoCortex-X Stage 10 Feedback Loop Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage9-action-gate-plan.md`
- `research/neocortex_x/stage9_action_gate.py`

## Цель

Stage 10 замыкает маленький learning-style контур поверх памяти:

```text
memory action -> outcome -> reward -> feedback event
```

Это не infinite RL claim. Это минимальный воспроизводимый feedback ledger,
который говорит, была ли память полезной и какое действие следует усилить или
пометить для дополнительного evidence.

## Outcome Rules

- `answer_from_memory` succeeds when answer equals the past value.
- `update_memory_and_answer` succeeds when answer equals the refreshed value.
- `request_more_evidence` succeeds when no answer is emitted and
  `needs_more_evidence = true`.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 9 artifact valid and non-empty;
- feedback event count equals decision count;
- outcome success >= `0.95`;
- mean reward >= `0.80`;
- all feedback events have replay hashes.

## Files

- `research/neocortex_x/stage10_feedback_loop.py`
- `Makefile`
  - add `neocortex-x-stage10-feedback-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage10_feedback_loop.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage10_feedback_loop.py
make neocortex-x-stage10-feedback-smoke
```
