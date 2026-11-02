# NeoCortex-X Stage 29 Contextual Memory Scale Probe Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage28-append-only-evidence-log-compaction-plan.md`
- `research/neocortex_x/stage28_append_only_evidence_log_compaction.py`

## Цель

Stage 29 делает быстрый scale-probe для context-aware memory:

```text
synthetic conditional records -> append-only evidence log -> compacted indexed store -> lookup benchmark
```

Это не финальный 10M+ benchmark. Это проверяемый шаг между маленьким fixture
и будущим большим масштабированием: thousands-of-records probe, измеряющий
ingest, compaction, indexed lookup и JSON scan baseline.

## Probe Contract

- сгенерировать минимум 1000 synthetic conditional slots;
- записать их как append-only evidence events;
- детерминированно compact в indexed SQLite query store;
- выполнить sampled exact contextual lookups;
- проверить, что event lookup и materialized lookup используют indexes;
- сравнить indexed lookup с JSON scan baseline, где JSON разрешен только как
  baseline, не как primary memory.

## Pass/Fail

- `protocol_evidence.valid = true`;
- record count >= `min_records`;
- event count equals record count;
- materialized slot count equals record count;
- append-only sequence is valid;
- event hashes validate payload integrity;
- deterministic compaction = `true`;
- indexed lookup accuracy >= `0.999`;
- indexed p95 lookup latency <= configured budget;
- index usage verified;
- JSON scan baseline is correct;
- indexed lookup beats JSON scan baseline by configured speedup.

## Files

- `research/neocortex_x/stage29_contextual_memory_scale_probe.py`
- `Makefile`
  - add `neocortex-x-stage29-scale-probe-smoke`
- `research/neocortex_x/README.md`

## Artifacts

- `artifacts/neocortex_x/stage29_scale_probe.sqlite`
- `artifacts/neocortex_x/stage29_contextual_memory_scale_probe.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage29_contextual_memory_scale_probe.py
make neocortex-x-stage29-scale-probe-smoke
```
