# NeoCortex-X Stage 20 Contextual Policy Replay Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage17-contextual-action-gate-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage19-contextual-policy-update-plan.md`
- `research/neocortex_x/stage17_contextual_action_gate.py`
- `research/neocortex_x/stage19_contextual_policy_update.py`

## Цель

Stage 20 применяет contextual policy snapshot в replay:

```text
context decisions + contextual policy -> replayed agent actions
```

Это контекстный аналог Stage 12. Система должна выбирать действие из policy
bucket для конкретного `context_decision`, а не из глобального action prior.

## Replay Contract

- `context_match` должен выбирать `answer_from_context_memory`;
- `same_subject_different_context` должен выбирать `answer_with_context_scope`;
- `request_context` должен выбирать `ask_context_question`;
- `flag_conflict` должен выбирать `request_more_evidence`.

## Controls

- `contextual_policy_guided`: выбирает top action из bucket
  `policy.probabilities[context_decision]`.
- `global_prior_control`: выбирает глобально самый вероятный action, игнорируя
  context decision.
- `always_evidence_control`: всегда выбирает `request_more_evidence`.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 17 and Stage 19 artifacts valid and passed;
- guided action accuracy >= `0.95`;
- guided replay covers all four contextual actions;
- guided replay beats global prior control by >= `0.25`;
- guided replay has no false evidence request for
  `same_subject_different_context`;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage20_contextual_policy_replay.py`
- `Makefile`
  - add `neocortex-x-stage20-contextual-policy-replay-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage20_contextual_policy_replay.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage20_contextual_policy_replay.py
make neocortex-x-stage20-contextual-policy-replay-smoke
```
