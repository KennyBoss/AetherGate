# NeoCortex-X Stage 19 Contextual Policy Update Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage18-contextual-feedback-plan.md`
- `research/neocortex_x/stage18_contextual_feedback.py`

## Цель

Stage 19 превращает contextual feedback events в policy snapshot:

```text
contextual feedback -> decision/action weights -> contextual policy snapshot
```

Это контекстный аналог Stage 11. Главное отличие: policy не должна иметь один
общий вес на действие. `request_more_evidence` хорошо для `flag_conflict`, но
плохо для `same_subject_different_context`; `ask_context_question` хорошо для
`request_context`, но плохо для `context_match`.

## Policy Contract

Policy хранит таблицу:

```text
context_decision -> action -> weight/probability
```

Поддерживаемые decisions:

- `context_match`;
- `same_subject_different_context`;
- `request_context`;
- `flag_conflict`.

Поддерживаемые actions:

- `answer_from_context_memory`;
- `answer_with_context_scope`;
- `ask_context_question`;
- `request_more_evidence`.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 18 artifact valid and passed;
- every reinforced contextual action gains positive weight in its own decision
  bucket;
- probability sum for every decision bucket equals `1.0`;
- each decision bucket's top action equals the reinforced action;
- no cross-context leakage: `request_more_evidence` must not become top action
  for `same_subject_different_context`;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage19_contextual_policy_update.py`
- `Makefile`
  - add `neocortex-x-stage19-contextual-policy-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage19_contextual_policy_update.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage19_contextual_policy_update.py
make neocortex-x-stage19-contextual-policy-smoke
```
