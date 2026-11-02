# NeoCortex-X Stage 7 Memory Retrieval Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage6-evidence-packets-plan.md`
- `research/neocortex_x/stage6_evidence_packets.py`

## Цель

Stage 7 делает следующий маленький, но важный шаг: evidence packets становятся
retrievable long-term memory.

Stage 6 уже строит компактные factual packets. Stage 7 индексирует их и
проверяет, что memory lookup возвращает правильный factual value и provenance
для held-out query strings.

```text
evidence packets -> memory index -> query -> recalled fact + provenance
```

## Query Contract

Поддерживаемые smoke-запросы:

- exact subject query: `entity_alpha`;
- natural compact query: `entity_alpha current_value`;
- memory-key query: `nx6:entity_alpha:...`.

## Controls

- `packet_memory`: индекс по subject и memory_key.
- `latest_string_control`: выбирает последний packet без semantic lookup.
- `no_memory_control`: не использует packet store.

## Pass/Fail

- `protocol_evidence.valid = true`;
- packet artifact valid and non-empty;
- memory lookup accuracy >= `0.95`;
- provenance coverage = `1.0`;
- packet-memory beats no-memory by >= `0.50`;
- packet-memory beats latest-string control by >= `0.25`.

## Files

- `research/neocortex_x/stage7_memory_retrieval.py`
- `Makefile`
  - add `neocortex-x-stage7-memory-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage7_memory_retrieval.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage7_memory_retrieval.py
make neocortex-x-stage7-memory-smoke
```
