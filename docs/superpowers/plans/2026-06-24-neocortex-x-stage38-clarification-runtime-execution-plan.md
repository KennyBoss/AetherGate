# NeoCortex-X Stage 38 Clarification Runtime Execution Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage30-partial-context-scale-retrieval-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage37-clarification-runtime-loader-plan.md`
- `research/neocortex_x/stage30_partial_context_scale_retrieval.py`
- `research/neocortex_x/stage37_clarification_runtime_loader.py`

## Цель

Stage 38 выполняет promoted clarification runtime policy на cached
partial-context episodes:

```text
runtime clarification policy + partial-context candidates -> execution traces
```

Это runtime analog для Stage 35 replay: Stage 37 policy уже загружена как
runtime snapshot, теперь нужно проверить, что она не просто выбирает ключ в
smoke decisions, а реально ведет bounded clarification loop до answer output.

## Runtime Execution Contract

- загрузить Stage 37 runtime artifact;
- загрузить Stage 30 evaluated partial-context queries;
- для unique candidate query отвечать без лишнего уточнения;
- для ambiguous query выбирать первый missing key из
  `runtime_policy.ranked_keys`;
- симулировать answer value из `full_context`;
- фильтровать candidate set после каждого уточнения;
- завершать, когда найден unique target slot или исчерпаны context dimensions;
- сохранять target slot на каждом шаге;
- писать output payload и active compute summary для каждого trace.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 37 runtime artifact valid and passed;
- Stage 30 artifact valid and passed;
- ambiguous and unique queries present;
- runtime execution success >= `0.95`;
- ambiguous execution success >= `0.95`;
- target preservation = `1.0`;
- trace count equals Stage 30 query count;
- max steps <= number of context dimensions;
- no unnecessary clarification for unique queries;
- selected keys come from `runtime_policy.ranked_keys`;
- context questions are present for ambiguous queries;
- answer outputs are present;
- active compute recorded;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage38_clarification_runtime_execution.py`
- `Makefile`
  - add `neocortex-x-stage38-clarification-runtime-execution-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage38_clarification_runtime_execution.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage38_clarification_runtime_execution.py
make neocortex-x-stage38-clarification-runtime-execution-smoke
make compile
```

## Guardrail

Stage 38 доказывает только deterministic runtime execution поверх cached
partial-context episodes. Это не live deployment, не production storage
benchmark и не claim о 10M+ primary memory.
