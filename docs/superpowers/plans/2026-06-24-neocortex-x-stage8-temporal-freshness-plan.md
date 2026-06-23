# NeoCortex-X Stage 8 Temporal Freshness Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage7-memory-retrieval-plan.md`
- `research/neocortex_x/stage7_memory_retrieval.py`

## Цель

Stage 8 добавляет TimeAxis-политику для долговременной памяти:

```text
past packet + present observation -> future memory action
```

Stage 7 доказал retrieval из evidence packets. Stage 8 проверяет, что система
может решить, что делать с памятью при новом наблюдении:

- `keep`: новое наблюдение подтверждает текущий packet;
- `refresh`: новое trusted наблюдение обновляет значение;
- `flag_conflict`: новое наблюдение спорит с packet, но доверия недостаточно
  для безопасного overwrite.

## Inputs

- Stage 6 evidence packets.
- Cached present observations derived from those packets.

No live web search is used in this smoke. The output schema is intended to be
compatible with a future live-search updater.

## Pass/Fail

- `protocol_evidence.valid = true`;
- packet artifact valid and non-empty;
- all three actions are present: `keep`, `refresh`, `flag_conflict`;
- policy action accuracy >= `0.95`;
- replay consistency = `1.0`;
- updated packet count equals input packet count.

## Files

- `research/neocortex_x/stage8_temporal_freshness.py`
- `Makefile`
  - add `neocortex-x-stage8-freshness-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage8_temporal_freshness.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage8_temporal_freshness.py
make neocortex-x-stage8-freshness-smoke
```
