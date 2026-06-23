# NeoCortex-X Stage 17 Contextual Action Gate Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage16-contextual-memory-plan.md`
- `research/neocortex_x/stage16_contextual_memory.py`

## Цель

Stage 16 научил память отличать настоящий конфликт от другого контекста.
Stage 17 превращает contextual memory decision в исполнительное действие
агента:

```text
context decision -> agent action -> answer / clarification / evidence request
```

Это contextual analog Stage 9 action gate. Если система видит `sky black` при
`night`, она не должна запрашивать evidence как при конфликте. Она должна
ответить с явным context scope. Если context отсутствует, она должна спросить
context, а не гадать.

## Action Contract

Stage 17 мапит решения Stage 16 так:

- `context_match` -> `answer_from_context_memory`;
- `same_subject_different_context` -> `answer_with_context_scope`;
- `request_context` -> `ask_context_question`;
- `flag_conflict` -> `request_more_evidence`.

## Controls

- `contextual_action_gate`: использует Stage 16 decision.
- `unconditional_fact_control`: отвечает из первого known subject slot и
  игнорирует context.
- `always_clarify_control`: всегда просит context.
- `always_evidence_control`: всегда просит дополнительные evidence.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 16 artifact valid and passed;
- all four contextual actions are present;
- gate action accuracy >= `0.95`;
- no false evidence request for contextual alias cases;
- missing context produces `ask_context_question`;
- conflict produces `request_more_evidence`;
- contextual gate beats unconditional fact control by >= `0.25`;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage17_contextual_action_gate.py`
- `Makefile`
  - add `neocortex-x-stage17-contextual-action-gate-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage17_contextual_action_gate.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage17_contextual_action_gate.py
make neocortex-x-stage17-contextual-action-gate-smoke
```
