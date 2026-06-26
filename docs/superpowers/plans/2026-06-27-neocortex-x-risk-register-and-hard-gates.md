# NeoCortex-X Risk Register And Hard Gates

Дата: 2026-06-27.
Builds on:
- `research/neocortex_x/README.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage40-clarification-monitor-feedback-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-conditional-memory-scale-research-plan.md`

## Цель

Перед продолжением Stage 41+ нужно зафиксировать не только что система умеет,
но и где она может красиво ошибаться. Stage 30-40 уже дают controlled
clarification loop:

```text
partial context -> clarification policy -> runtime execution
  -> self-monitor -> feedback events
```

Новый план добавляет risk-first gates, чтобы feedback loop не закрепил
хрупкую synthetic policy.

## Risk Register

1. Synthetic overfit.
   Policy может выучить distribution генератора Stage 30, а не общий принцип
   выбора context key. Симптом: perfect smoke на synthetic queries и провал на
   noisy/rare contexts.

2. Context explosion.
   Добавление новых context dimensions может раздувать candidate set, индексы
   и число уточняющих вопросов. Симптом: p95/p99 latency и mean clarification
   steps растут быстрее, чем accuracy.

3. Wrong-key reinforcement.
   Stage 40 может усиливать ключи `location/season`, потому что они полезны в
   текущей синтетике, хотя в другой задаче главным станет `time_of_day`,
   source, temporal validity или observer.

4. Provenance poisoning.
   Live/search integration может добавить ложные, устаревшие или malicious
   evidence packets. Симптом: confidence растет у неправильного conditional
   slot.

5. Replay/live gap.
   Deterministic replay доказывает воспроизводимость, но не доказывает, что
   система устойчива к меняющимся источникам, temporal drift и partial
   observations.

6. Self-confirmation loop.
   Runtime monitor и feedback могут подтверждать собственную policy без
   внешнего evidence audit. Симптом: no critical alerts, но incorrect answer
   under adversarial evidence.

7. Storage cliff.
   SQLite thousands-of-records smoke может скрыть проблемы 10M+ records:
   index bloat, compaction time, concurrent writes, replay rebuild cost.

8. Claim drift.
   Phi, AGI, consciousness, energy, 86B neurons, neuromorphic и quantum claims
   могут звучать как доказанные, хотя текущие stages доказывают только bounded
   deterministic harness behavior.

9. Context/causality confusion.
   Conditional memory может правильно хранить `при C: X = Y`, но ошибочно
   принять коррелирующий key за причинный. Симптом: система спрашивает
   `season/location`, хотя реальная причина в `time_of_day`, source validity,
   observer или физическом состоянии среды.

10. Over-clarification.
   Context-aware память может стать слишком осторожной и превращать каждый
   вопрос в длинный диалог уточнений. Симптом: target slot сохраняется, но
   mean clarification steps и user-friction растут быстрее пользы.

11. Context schema drift.
   Новые предметные области требуют новых context keys, alias groups и
   validity windows. Симптом: старый `CONTEXT_KEYS` contract выглядит чисто в
   replay, но не выражает важные условия реального мира.

## Hard Gates Before Stage 41+

Stage 41+ не должен просто промотить feedback из Stage 40. Перед policy update
нужен review gate:

- reject promotion if any critical alert exists;
- require deterministic replay of feedback events;
- require held-out or adversarial context cases;
- compare current policy with fixed-order, entropy and previous-runtime
  controls;
- require rollback metadata before runtime promotion.

## Adversarial Context Suite

Минимальный набор проверок:

- missing context: слишком мало ключей, но есть правильный target slot;
- noisy context: один контекстный ключ ошибочный;
- correlated keys: `location` и `season` коррелируют, но не являются причиной;
- rare context: правильный slot встречается редко;
- alias context: разные имена одной локации/сезона/источника;
- temporal drift: старый slot был верным, но validity window истек;
- same-context contradiction: настоящий conflict, не другой context;
- different-context non-conflict: то же subject/property, но другой context.

## Provenance Poisoning Suite

Перед live internet/search:

- stale source: источник устарел;
- low-trust source: источник слабее уже сохраненного evidence;
- malicious contradiction: источник пытается перезаписать верный slot;
- duplicate source: много копий одной ошибки не должны выглядеть как
  независимое подтверждение;
- source drift: раньше надежный источник изменил утверждение.

## Scale Cliff Gate

До production-scale memory:

- 10M+ conditional records;
- exact lookup p50/p95/p99;
- partial lookup p50/p95/p99;
- ambiguity-key selection latency;
- ingest throughput;
- compaction time;
- replay rebuild time;
- disk size and index overhead;
- concurrent read/write behavior;
- JSON scan only as baseline, never as primary memory.

JSON остается форматом отчетов, fixtures и deterministic summaries. Он не
должен становиться primary memory store: на 10M+ conditional slots нужны
append-only evidence log, deterministic compaction, индексированные query
views и benchmark p50/p95/p99. Иначе система может пройти smoke tests, но
сломаться на частичных обновлениях, index rebuild, latency tail и конкуренции
чтения/записи.

Минимальный storage research перед 10M+:

- сравнить SQLite, DuckDB/Postgres, Parquet/Arrow snapshots, RocksDB/LSM и
  hybrid symbolic+vector index;
- измерить exact lookup, partial-context retrieval и ambiguity-key selection;
- разделить hot indexes и cold provenance, чтобы refs не раздували быстрый
  путь;
- проверить deterministic rebuild из evidence log;
- проверить alias/context-schema migration без потери replay.

## Claim Guardrails

Разделить язык claims:

- Current evidence: deterministic harness, indexed retrieval smoke,
  clarification loop, runtime monitor, feedback ledger.
- Roadmap only: AGI, consciousness, Phi thresholds, 86B neurons, quantum
  advantage, neuromorphic energy claims.
- Forbidden wording until gated: "proved AGI", "proved consciousness",
  "production memory", "live reliable internet memory", "0.1 W AGI".

## Recommended Next Plan

Перед Stage 41 policy update лучше сделать Stage 41A risk audit:

```text
Stage 40 feedback
  -> Stage 41A adversarial/risk audit
  -> Stage 41B guarded policy review
  -> Stage 42 replay against clean + adversarial suites
  -> Stage 42C provenance/conflict guard if critical risks remain
  -> Stage 43 promotion with rollback metadata only after risks clear
```

Так мы не просто делаем систему умнее, а не даем ей уверенно закрепить
неправильные привычки.

Stage 42 pass не должен автоматически означать promotion. Если replay всё еще
показывает critical same-context contradiction, noisy-context unsafe answer или
target loss, следующий шаг должен быть защитным: evidence/conflict guard,
policy revision или новый adversarial suite, а не runtime activation.

Stage 42C реализует защитный вариант этого шага:

- same-context contradiction -> `request_evidence_on_value_conflict`;
- noisy context -> `request_context_integrity_check`;
- temporal drift -> `request_validity_window_check`;
- clean unique answer -> сохранить answer route;
- runtime promotion -> всё еще blocked до Stage 43 rollback review.
