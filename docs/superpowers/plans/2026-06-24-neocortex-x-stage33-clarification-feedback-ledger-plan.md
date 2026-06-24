# NeoCortex-X Stage 33 Clarification Feedback Ledger Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage32-clarification-resolution-loop-plan.md`
- `research/neocortex_x/stage32_clarification_resolution_loop.py`

## Цель

Stage 33 превращает clarification-resolution episodes в feedback ledger:

```text
clarification steps + resolved slot -> reward events -> key-level feedback
```

После Stage 32 система умеет уточнять контекст и сужать candidates. Stage 33
фиксирует, какие context keys действительно помогли, какой reward они получили
и нужно ли review политики.

## Feedback Contract

- каждый clarification step получает `clarification_step_feedback`;
- каждый episode получает resolution feedback;
- successful step reward зависит от reduction ratio, target preservation и
  достижения unique candidate;
- failed step получает negative reward и policy review;
- rewards агрегируются по `reinforce_key`;
- replay deterministic.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 32 artifact valid and passed;
- feedback events present;
- episode feedback count equals episode count;
- clarification step events present;
- success/step/episode success rates >= configured threshold;
- mean reward >= configured threshold;
- no target-loss penalties;
- no policy review needed;
- key rewards present;
- replay hashes present;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage33_clarification_feedback_ledger.py`
- `Makefile`
  - add `neocortex-x-stage33-clarification-feedback-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage33_clarification_feedback_ledger.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage33_clarification_feedback_ledger.py
make neocortex-x-stage33-clarification-feedback-smoke
```
