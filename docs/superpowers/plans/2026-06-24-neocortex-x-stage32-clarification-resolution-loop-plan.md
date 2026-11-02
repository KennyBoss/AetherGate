# NeoCortex-X Stage 32 Clarification Resolution Loop Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage31-partial-context-disambiguation-gate-plan.md`
- `research/neocortex_x/stage31_partial_context_disambiguation_gate.py`

## Цель

Stage 32 замыкает partial-context clarification loop:

```text
ambiguous candidate set -> ask context key -> receive answer -> narrow candidates -> resolve slot
```

Stage 31 выбирает, какой context key спросить. Stage 32 симулирует ответ
пользователя из target full context, добавляет его к partial context и
проверяет, что candidate set сужается до правильного slot. Если одного ключа
мало, loop продолжает выбирать следующий informative missing key в bounded
режиме.

## Resolution Contract

- unique candidate queries resolve без лишнего уточнения;
- ambiguous queries используют первый key из Stage 31;
- каждый clarification answer сохраняет target slot в candidate set;
- каждый clarification step уменьшает candidate set;
- final candidate set становится unique и содержит target slot;
- replay deterministic.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 30 and Stage 31 artifacts valid and passed;
- query count equals decision count;
- ambiguous and unique episodes are present;
- resolution accuracy >= configured threshold;
- ambiguous resolution accuracy >= configured threshold;
- target preservation rate = `1.0`;
- first clarification key matches Stage 31;
- every clarification step reduces candidates;
- max steps bounded by context dimension count;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage32_clarification_resolution_loop.py`
- `Makefile`
  - add `neocortex-x-stage32-clarification-resolution-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage32_clarification_resolution_loop.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage32_clarification_resolution_loop.py
make neocortex-x-stage32-clarification-resolution-smoke
```
