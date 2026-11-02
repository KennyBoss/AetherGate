# NeoCortex-X Stage 30 Partial-Context Scale Retrieval Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage29-contextual-memory-scale-probe-plan.md`
- `research/neocortex_x/stage29_contextual_memory_scale_probe.py`

## Цель

Stage 30 проверяет retrieval при неполном context vector:

```text
partial context query -> indexed candidate set -> exact-context rerank
```

Реальная память редко получает все условия сразу. Например, система может знать
`subject/property` и `weather=cloudy`, но не знать season/location/time. Stage
30 проверяет, что indexed query store умеет быстро вернуть candidate set без
JSON scan и затем выбрать точный slot, если полный context доступен для rerank.

## Probe Contract

- сгенерировать thousands-of-records synthetic conditional slots;
- записать их через append-only evidence log и deterministic compaction;
- создать sampled partial-context queries;
- вернуть candidate set через `context_dimensions` index;
- подтвердить target recall и exact-context rerank accuracy;
- сравнить p95 latency с JSON scan baseline.

## Pass/Fail

- `protocol_evidence.valid = true`;
- record count >= `min_records`;
- event/materialized counts match records;
- append-only sequence and event hashes are valid;
- deterministic compaction = `true`;
- partial-context target recall >= configured minimum;
- exact-context rerank accuracy >= configured minimum;
- multi-candidate queries are present;
- indexed partial lookup p95 latency <= configured budget;
- context index usage verified;
- JSON scan baseline recalls target;
- indexed lookup beats JSON scan baseline.

## Files

- `research/neocortex_x/stage30_partial_context_scale_retrieval.py`
- `Makefile`
  - add `neocortex-x-stage30-partial-context-smoke`
- `research/neocortex_x/README.md`

## Artifacts

- `artifacts/neocortex_x/stage30_partial_context.sqlite`
- `artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage30_partial_context_scale_retrieval.py
make neocortex-x-stage30-partial-context-smoke
```
