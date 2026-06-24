# NeoCortex-X Stage 35 Clarification Policy Replay Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage30-partial-context-scale-retrieval-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage34-clarification-policy-update-plan.md`
- `research/neocortex_x/stage30_partial_context_scale_retrieval.py`
- `research/neocortex_x/stage34_clarification_policy_update.py`

## Цель

Stage 35 применяет learned clarification-key policy к Stage 30 partial-context
episodes:

```text
partial context candidates + ranked clarification keys -> resolved conditional slot
```

Это проверяет, что Stage 34 policy snapshot не просто сохраняется в отчете, а
может детерминированно вести ambiguity resolution loop.

## Replay Contract

- для unique candidate query не задавать лишних уточнений;
- для ambiguous query выбирать первый доступный missing key из
  `policy.ranked_keys`;
- симулировать ответ из `full_context`;
- фильтровать candidate set после каждого уточнения;
- завершать, когда найден unique target slot или исчерпаны context dimensions;
- сохранять target slot на каждом шаге.

## Controls

- `policy_guided_replay`: использует `policy.ranked_keys`.
- `fixed_order_control`: использует статический порядок `CONTEXT_KEYS`.
- `static_entropy_control`: выбирает key с максимальным разнообразием значений
  среди текущих candidates.
- `always_top_key_control`: пробует только `policy.top_key`.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 30 artifact valid and passed;
- Stage 34 policy artifact valid and passed;
- ambiguous and unique queries present;
- policy resolution accuracy >= `0.95`;
- policy ambiguous resolution accuracy >= `0.95`;
- target preservation = `1.0`;
- max policy steps <= number of context dimensions;
- no unnecessary clarification for unique queries;
- selected policy keys come from `policy.ranked_keys`;
- policy mean ambiguous steps beats fixed-order control by configured margin;
- policy accuracy beats always-top-key control by configured margin;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage35_clarification_policy_replay.py`
- `Makefile`
  - add `neocortex-x-stage35-clarification-policy-replay-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage35_clarification_policy_replay.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage35_clarification_policy_replay.py
make neocortex-x-stage35-clarification-policy-replay-smoke
```
