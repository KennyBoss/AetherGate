# NeoCortex-X Stage 25 Indexed Contextual Memory Store Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage16-contextual-memory-plan.md`
- `research/neocortex_x/stage16_contextual_memory.py`

## Цель

Stage 25 начинает practical scale track для context-aware memory:

```text
conditional slots JSON smoke -> indexed durable store preflight
```

JSON остается только отчетом smoke. Сами conditional slots загружаются в SQLite
artifact с явной схемой и индексами. Это не финальный 10M+ benchmark, но первый
проверяемый шаг от игрушечного JSON к indexed storage.

## Storage Contract

SQLite artifact содержит таблицы:

- `memory_slots`;
- `context_dimensions`;
- `provenance_refs`;
- `store_metadata`.

Индексы:

- `(subject, property)`;
- `(subject, property, value)`;
- `(context_key, context_value)`;
- `(slot_id)`;
- `(provenance_ref)`.

## Query Contract

Smoke queries должны покрывать:

- exact context lookup;
- partial context lookup;
- subject/property lookup;
- missing context candidate lookup;
- provenance lookup.

## Scale Guardrail

Output обязан явно зафиксировать:

- JSON role: `report_only`;
- store role: `indexed_preflight`;
- 10M+ target остается future benchmark;
- recommended next storage candidates: SQLite/DuckDB/Postgres/Parquet/RocksDB
  plus hybrid symbolic+vector index.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 16 artifact valid and passed;
- SQLite DB artifact created;
- slot count equals Stage 16 conditional slots;
- required tables and indexes exist;
- exact context lookup accuracy >= `0.95`;
- partial lookup returns candidates;
- provenance lookup returns slots;
- query plan uses indexes for indexed lookup;
- deterministic rebuild = `true`.

## Files

- `research/neocortex_x/stage25_indexed_contextual_memory_store.py`
- `Makefile`
  - add `neocortex-x-stage25-indexed-contextual-store-smoke`
- `research/neocortex_x/README.md`

## Artifacts

- `artifacts/neocortex_x/stage25_contextual_memory.sqlite`
- `artifacts/neocortex_x/stage25_indexed_contextual_memory_store.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage25_indexed_contextual_memory_store.py
make neocortex-x-stage25-indexed-contextual-store-smoke
```
