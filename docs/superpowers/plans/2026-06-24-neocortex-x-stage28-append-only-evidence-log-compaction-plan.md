# NeoCortex-X Stage 28 Append-Only Evidence Log Compaction Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage16-contextual-memory-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage25-indexed-contextual-memory-store-plan.md`
- `research/neocortex_x/stage16_contextual_memory.py`
- `research/neocortex_x/stage25_indexed_contextual_memory_store.py`

## Цель

Stage 28 делает следующий practical scale step для context-aware memory:

```text
conditional slots -> append-only evidence log -> deterministic compaction -> indexed query store
```

Это закрывает важный architectural gap: JSON не должен быть памятью. Источник
истины становится append-only evidence log, а быстрый retrieval работает по
compacted indexed store.

## Storage Contract

SQLite artifact содержит:

- `evidence_events`: append-only события с `seq`, `event_hash`, `payload_json`;
- `memory_slots`: материализованные conditional slots;
- `context_dimensions`: индексируемые context key/value;
- `provenance_refs`: ссылки на evidence/provenance;
- `store_metadata`: роли log/store/json и compaction hash.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 16 artifact valid and passed;
- event count equals conditional slot count;
- append-only sequence is contiguous;
- event hashes validate payload integrity;
- compaction is deterministic;
- materialized slot count equals event count;
- required tables and indexes exist;
- event lookup uses index;
- materialized subject/context lookup uses indexes;
- Stage 26 retrieval policy remains compatible with compacted store;
- JSON role is `report_only`;
- evidence log role is `append_only_source_of_truth`;
- query store role is `compacted_indexed_view`.

## Files

- `research/neocortex_x/stage28_append_only_evidence_log_compaction.py`
- `Makefile`
  - add `neocortex-x-stage28-evidence-log-compaction-smoke`
- `research/neocortex_x/README.md`

## Artifacts

- `artifacts/neocortex_x/stage28_evidence_log_compacted_store.sqlite`
- `artifacts/neocortex_x/stage28_append_only_evidence_log_compaction.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage28_append_only_evidence_log_compaction.py
make neocortex-x-stage28-evidence-log-compaction-smoke
```
