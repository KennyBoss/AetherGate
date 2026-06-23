# NeoCortex-X Stage 18 Contextual Feedback Loop Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage17-contextual-action-gate-plan.md`
- `research/neocortex_x/stage17_contextual_action_gate.py`

## Цель

Stage 18 добавляет feedback ledger для contextual actions:

```text
contextual action -> outcome score -> feedback event -> future policy evidence
```

Это контекстный аналог Stage 10. Главное отличие: scoped answer для другого
контекста должен усиливаться, а не наказываться как конфликт. Если `sky black`
при `night` обработан через `answer_with_context_scope`, это успешное поведение.

## Reward Contract

- `answer_from_context_memory` успешен, если ответ совпадает с contextual
  memory slot и не просит evidence/context.
- `answer_with_context_scope` успешен, если ответ явно ограничен наблюдаемым
  context scope и не просит evidence.
- `ask_context_question` успешен, если missing context действительно приводит к
  clarification question.
- `request_more_evidence` успешен, если используется только для реального
  `flag_conflict`.

## Feedback Event Contract

Каждый feedback event содержит:

- stable `feedback_id`;
- `observation_id`;
- `context_decision`;
- `agent_action`;
- `success`;
- numeric `reward`;
- `reinforce_action`;
- `needs_policy_review`;
- replay hash.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 17 artifact valid and passed;
- feedback count equals contextual action decisions;
- outcome success >= `0.95`;
- mean reward >= `0.90`;
- contextual scoped answers are reinforced;
- missing context clarifications are reinforced;
- real conflicts are routed to evidence request;
- no false evidence penalties for `same_subject_different_context`;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage18_contextual_feedback.py`
- `Makefile`
  - add `neocortex-x-stage18-contextual-feedback-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage18_contextual_feedback.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage18_contextual_feedback.py
make neocortex-x-stage18-contextual-feedback-smoke
```
