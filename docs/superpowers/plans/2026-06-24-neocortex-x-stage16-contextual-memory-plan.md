# NeoCortex-X Stage 16 Context-Aware Memory Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage6-evidence-packets-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage8-temporal-freshness-plan.md`
- `research/neocortex_x/stage6_evidence_packets.py`

## Цель

Stage 16 исправляет ключевую слабость factual memory: память не должна хранить
безусловное `subject = value`, если значение зависит от условий.

Вместо:

```text
sky = blue
```

Stage 16 хранит conditional knowledge:

```text
subject = sky
property = expected_color
context = {time_of_day: daytime, weather: clear}
value = blue
```

Это мост от Stage 6 evidence packets к Stage 4B world model: система начинает
различать настоящий конфликт и тот же subject в другом контексте.

## Контракт

Каждый conditional memory slot обязан содержать:

- stable `slot_id`;
- compact `memory_key`;
- `subject`;
- `property`;
- structured `context`;
- `value`;
- confidence;
- provenance refs;
- replay hash.

## Decisions

Stage 16 проверяет четыре типа решений:

- `context_match`: наблюдение совпадает с известным contextual slot;
- `same_subject_different_context`: тот же subject/property, но другой
  валидный context, поэтому это не конфликт;
- `request_context`: context отсутствует или недостаточен для безопасного
  выбора slot;
- `flag_conflict`: context совпал, но значение несовместимо с memory slot.

Пример:

```text
sky black + night -> same_subject_different_context
sky black + missing context -> request_context
sky green + daytime clear -> flag_conflict
```

## Scale Track, Not Current Implementation

JSON остается только smoke artifact. Для 10M+ conditional records нужен
отдельный future research track:

- schema and indexes по `subject`, `property`, context dimensions, time,
  confidence, provenance;
- append-only evidence log плюс compacted memory snapshots;
- deterministic replay через event log/snapshot hash;
- storage candidates: SQLite/DuckDB для локальных smoke, Postgres для durable
  relational memory, Parquet/Arrow для columnar replay, RocksDB/LSM для
  large key-value memory log, hybrid symbolic+vector index для retrieval;
- graph DB только если появится доказанная польза для traversals, не как
  default storage.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 6 packet artifact valid;
- conditional slot count > 0;
- all four decision types are present;
- decision accuracy >= `0.95`;
- false contextual conflict rate = `0.0`;
- missing context detection rate = `1.0`;
- deterministic replay = `true`;
- scale research track included in output.

## Files

- `research/neocortex_x/stage16_contextual_memory.py`
- `Makefile`
  - add `neocortex-x-stage16-contextual-memory-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage16_contextual_memory.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage16_contextual_memory.py
make neocortex-x-stage16-contextual-memory-smoke
```
