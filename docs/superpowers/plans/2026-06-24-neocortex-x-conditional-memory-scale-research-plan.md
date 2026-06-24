# NeoCortex-X Conditional Memory Scale Research Plan

Дата: 2026-06-24.

## Цель

Перед масштабированием context-aware memory до 10M+ conditional records нужно
отдельно проверить storage/index слой. JSON остается форматом reports,
fixtures и deterministic summaries, но не primary memory.

Ключевая идея: память хранит не `X = Y`, а conditional slot:

```text
subject + property + value + context_vector + confidence + provenance
```

Пример:

```text
sky.expected_color = blue
  при time_of_day=daytime, weather=clear, location=earth
```

Наблюдение `sky.expected_color = dark` ночью не конфликтует с дневным слотом.
Если контекст неполный, retrieval должен вернуть candidate set и запросить
недостающий context key.

## Research Contract

Проверить несколько вариантов primary memory/index backend:

- append-only evidence log как source of truth;
- deterministic compaction в query store;
- exact lookup по `subject/property/context`;
- partial-context lookup по известным context keys;
- ambiguity detection и выбор уточняющего key;
- provenance lookup без раздувания горячих индексов;
- deterministic replay после rebuild;
- storage footprint на больших synthetic datasets.

Кандидаты:

- SQLite как embedded baseline;
- DuckDB/Parquet/Arrow для аналитического bulk scan;
- Postgres для concurrent/service mode;
- RocksDB/LSM для heavy ingest и key-value access;
- hybrid symbolic + vector index для будущего semantic retrieval.

## Benchmarks

Минимальный future benchmark:

- 10M+ conditional slots;
- configurable context dimensions;
- controlled ambiguity density;
- ingest throughput;
- compaction time;
- exact lookup p50/p95/p99;
- partial lookup p50/p95/p99;
- disk size and index overhead;
- replay rebuild time;
- correctness against deterministic oracle;
- JSON scan only as slow baseline, not as primary design.

## Pass/Fail

- deterministic replay returns the same materialized slots;
- exact lookup accuracy >= `0.999`;
- partial-context target recall >= configured threshold;
- ambiguity detection accuracy >= configured threshold;
- p95 lookup latency stays under the selected backend budget;
- storage backend beats JSON scan by a meaningful margin;
- primary memory path never depends on loading all records into one JSON object.

## Placement In Roadmap

Этот research gate должен идти после small-scale contextual memory stages
и до любого утверждения о production-scale memory. Практически он продолжает
Stage 29/30, но меняет вопрос с “работает ли indexed lookup на thousands of
records” на “какой backend выдерживает 10M+ conditional records”.

## Artifacts

Будущий stage должен сохранять:

- backend comparison report;
- schema/index definitions;
- benchmark manifest;
- deterministic replay hash;
- sampled correctness evidence;
- latency and storage metrics.
