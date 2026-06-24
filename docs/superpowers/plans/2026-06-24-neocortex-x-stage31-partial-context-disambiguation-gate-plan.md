# NeoCortex-X Stage 31 Partial-Context Disambiguation Gate Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage30-partial-context-scale-retrieval-plan.md`
- `research/neocortex_x/stage30_partial_context_scale_retrieval.py`

## Цель

Stage 31 превращает partial-context candidate set в bounded agent action:

```text
partial-context candidates -> answer if unique / ask best missing context if ambiguous
```

Это следующий шаг после `request_context`: система не просто говорит “нужен
контекст”, а выбирает наиболее полезный недостающий context key для уточнения.

## Action Contract

- one candidate -> `answer_from_unique_context`;
- multiple candidates -> `ask_disambiguating_context`;
- disambiguation key выбирается среди missing context dimensions с максимальным
  разнообразием значений в candidate set.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 30 artifact valid and passed;
- both action types are present;
- gate action accuracy >= `0.95`;
- no false answer on ambiguous candidate set;
- no false question on unique candidate set;
- ambiguous and unique queries are both present;
- every ambiguous query receives a disambiguation key;
- gate beats always-answer and always-ask controls;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage31_partial_context_disambiguation_gate.py`
- `research/neocortex_x/stage30_partial_context_scale_retrieval.py`
  - include evaluated query details in artifact
- `Makefile`
  - add `neocortex-x-stage31-disambiguation-gate-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage31_partial_context_disambiguation_gate.json`

## Verification

```bash
python3 -m py_compile \
  research/neocortex_x/stage30_partial_context_scale_retrieval.py \
  research/neocortex_x/stage31_partial_context_disambiguation_gate.py
make neocortex-x-stage31-disambiguation-gate-smoke
```
