# NeoCortex-X Stage 26 Store-Backed Contextual Retrieval Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage16-contextual-memory-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage25-indexed-contextual-memory-store-plan.md`
- `research/neocortex_x/stage16_contextual_memory.py`
- `research/neocortex_x/stage25_indexed_contextual_memory_store.py`

## Цель

Stage 26 проверяет, что context-aware memory decisions можно делать через
indexed SQLite store, а не через in-memory JSON slots:

```text
SQLite contextual store + observations -> contextual decisions
```

Это следующий шаг scale track: Stage 25 построил indexed store, Stage 26
использует его как источник retrieval.

## Retrieval Contract

Store-backed retrieval должен поддерживать:

- exact context match;
- value alias under same context;
- value known under different context;
- missing context candidate lookup;
- same-context incompatible value conflict.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 25 store report valid and passed;
- SQLite DB artifact exists;
- store-backed decision accuracy >= `0.95`;
- all four decision types are present;
- false contextual conflict rate = `0.0`;
- missing context detection rate = `1.0`;
- every non-missing decision uses a DB-backed candidate;
- indexed query plans are used;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage26_store_backed_contextual_retrieval.py`
- `Makefile`
  - add `neocortex-x-stage26-store-backed-contextual-retrieval-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage26_store_backed_contextual_retrieval.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage26_store_backed_contextual_retrieval.py
make neocortex-x-stage26-store-backed-contextual-retrieval-smoke
```
