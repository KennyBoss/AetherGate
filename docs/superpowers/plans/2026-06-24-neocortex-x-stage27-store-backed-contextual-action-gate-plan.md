# NeoCortex-X Stage 27 Store-Backed Contextual Action Gate Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage26-store-backed-contextual-retrieval-plan.md`
- `research/neocortex_x/stage26_store_backed_contextual_retrieval.py`

## Цель

Stage 27 превращает store-backed contextual decisions в bounded agent actions:

```text
SQLite-backed contextual retrieval -> agent action -> output payload
```

Это store-backed аналог Stage 17. После Stage 26 система уже принимает
contextual decisions через SQLite lookup; Stage 27 проверяет, что action gate
тоже работает от DB-backed decisions.

Главный design point: память NeoCortex-X не должна хранить голые факты вида
`subject = sky; fact = blue`. Это ложное обобщение. Рабочая единица памяти:

```text
subject + property + value + context_vector + confidence + provenance
```

То есть `sky.expected_color = blue` валидно только при условиях вроде
`time_of_day=daytime`, `weather=clear`, `location=earth`. Если система видит
`sky=black`, это не конфликт само по себе: при `time_of_day=night` это другой
контекст; при missing context надо спросить контекст; при same-context
incompatible value надо запросить evidence.

## Action Contract

- `context_match` -> `answer_from_context_memory`;
- `same_subject_different_context` -> `answer_with_context_scope`;
- `request_context` -> `ask_context_question`;
- `flag_conflict` -> `request_more_evidence`.

## Scale Research Backlog

JSON не должен становиться основной памятью. В этом stage JSON остается только
читабельным smoke report; retrieval уже питается от indexed store. До 10M+
conditional records нужно вынести отдельный research/benchmark plan:

- сравнить SQLite, DuckDB, Postgres, Parquet/Arrow snapshots, RocksDB/LSM и
  hybrid symbolic+vector index;
- измерять ingest throughput, lookup latency, storage size, index usage,
  partial-context lookup и deterministic replay;
- сохранить append-only evidence log как источник истины;
- строить compacted query stores из evidence log;
- проверять, что action gate не деградирует при миллионах conditional slots.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 26 artifact valid and passed;
- all four contextual actions are present;
- gate action accuracy >= `0.95`;
- no false evidence request for `same_subject_different_context`;
- missing context produces `ask_context_question`;
- conflict produces `request_more_evidence`;
- DB-backed candidate usage is preserved for non-missing decisions;
- contextual gate beats unconditional control by >= `0.25`;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage27_store_backed_contextual_action_gate.py`
- `Makefile`
  - add `neocortex-x-stage27-store-backed-contextual-action-gate-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage27_store_backed_contextual_action_gate.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage27_store_backed_contextual_action_gate.py
make neocortex-x-stage27-store-backed-contextual-action-gate-smoke
```
