# NeoCortex-X Core Experiment

Этот подпроект реализует первые шаги утвержденного плана:

- Stage 0: deterministic tick-by-tick instrumentation с replay из JSONL trace.
- Stage 1: recall-to-action smoke, где `kv_memory` должен превратить отложенное
  bind/query воспоминание в корректное действие, а `context_window` служит
  короткоконтекстным baseline.
- Stage 1B: trained SSM/KV wrapper, который запускает существующий
  `compare_long_context_recall.py` и мапит held-out value recall в `act(value)`.
- Stage 2: stateful CodePy-like tape world с `tape`, `ptr`, `pc`, регистрами,
  bounded execution и baseline comparison.
- Stage 2B: learned trace policy для tape world, обученная imitation learning
  from oracle traces и проверенная на held-out tasks.
- Stage 3: procedure reuse smoke с bounded `sum_segment`, Axis-S/Axis-C задачами
  и контролями exact-cache/random-routing/inline-loop.
- Stage 3B: learned procedure promotion из inline traces с held-out
  certification перед reuse.
- Stage 4: short lookahead planning smoke, где planner использует predicted
  state deltas и бьет greedy action baseline.
- Stage 4B: learned world-model planner, где transition table обучается из
  опыта и используется для lookahead.
- Stage 5: deterministic open-world adapter smoke с provenance, trust-weighted
  factual updates и contradiction handling.
- Stage 5B: cached live-shaped provenance adapter, который прогоняет тот же
  provenance contract на replayable fixture schema перед подключением live
  search backend.
- Stage 6: evidence packets для долговременной памяти, где replayable
  provenance state превращается в compact memory key + factual packet.
- Stage 7: memory retrieval smoke, где evidence packets индексируются и
  достаются по subject, compact query и memory key.
- Stage 8: temporal freshness policy, где past packet + present observation
  дают future memory action: keep, refresh или flag_conflict.
- Stage 9: memory action gate, где freshness action превращается в действие
  агента: answer_from_memory, update_memory_and_answer или request_more_evidence.
- Stage 10: feedback loop, где action outcome превращается в reward и
  feedback event для будущего усиления/ревью политики.
- Stage 11: policy update, где feedback events меняют веса действий и создают
  следующий policy snapshot.
- Stage 12: policy-guided replay, где Stage 11 policy snapshot применяется в
  следующем deterministic decision cycle.
- Stage 13: policy promotion, где replay-validated policy получает runtime
  manifest с rollback metadata.
- Stage 14: runtime policy loader, где promoted manifest загружается как
  рабочий runtime snapshot с smoke decisions.
- Stage 15: runtime execution trace, где загруженная policy выполняет cached
  temporal episodes и пишет action outputs + active compute.
- Stage 16: context-aware memory, где один subject/property может иметь разные
  истинные значения при разных условиях, а missing context приводит к
  `request_context` вместо ложного `flag_conflict`.
- Stage 17: contextual action gate, где решения Stage 16 превращаются в
  bounded agent actions: ответ из контекстной памяти, ответ с context scope,
  уточняющий вопрос или запрос evidence.
- Stage 18: contextual feedback loop, где результаты Stage 17 превращаются в
  reward-bearing feedback events без ложного штрафа за другой контекст.
- Stage 19: contextual policy update, где contextual feedback обновляет
  policy snapshot по таблице `context_decision -> action`.
- Stage 20: contextual policy replay, где Stage 19 snapshot применяется к
  Stage 17 context decisions и проверяется против global-prior control.
- Stage 21: contextual policy promotion, где replay-validated contextual
  policy получает runtime manifest с rollback metadata.
- Stage 22: contextual runtime loader, где promoted contextual manifest
  загружается как runtime snapshot с per-context action defaults.
- Stage 23: contextual runtime execution, где загруженная contextual policy
  выполняет cached context episodes и пишет action outputs + active compute.
- Stage 24: contextual self-monitor, где runtime traces превращаются в
  monitor records с реконструкцией action, context decision, output и compute.
- Stage 25: indexed contextual memory store, где Stage 16 conditional slots
  загружаются в SQLite artifact со схемой и индексами; JSON остается только
  smoke report.
- Stage 26: store-backed contextual retrieval, где contextual decisions
  принимаются через SQLite lookup вместо in-memory JSON slots.
- Stage 27: store-backed contextual action gate, где SQLite-backed
  contextual decisions превращаются в bounded agent actions без возврата к
  unconditional fact memory.
- Stage 28: append-only evidence log compaction, где conditional slots
  записываются как immutable evidence events и детерминированно компактифицируются
  в indexed query store.
- Stage 29: contextual memory scale probe, где thousands-of-records synthetic
  dataset измеряет ingest, compaction, indexed lookup и JSON scan baseline.
- Stage 30: partial-context scale retrieval, где неполный context vector
  возвращает indexed candidate set и exact-context rerank выбирает нужный slot.
- Stage 31: partial-context disambiguation gate, где candidate ambiguity
  превращается в answer-or-clarify action с выбором недостающего context key.
- Stage 32: clarification resolution loop, где ответ на уточняющий context key
  сужает candidate set до правильного conditional slot.
- Stage 33: clarification feedback ledger, где шаги уточнения получают reward
  за сужение candidates и сохранение target slot.
- Stage 34: clarification policy update, где feedback по context keys
  превращается в normalized policy snapshot для выбора уточняющего ключа.
- Stage 35: clarification policy replay, где Stage 34 key policy применяется
  к partial-context episodes и сравнивается с fixed/static controls.
- Stage 36: clarification policy promotion, где replay-validated
  clarification-key policy получает runtime manifest с rollback metadata.
- Stage 37: clarification runtime loader, где promoted clarification manifest
  загружается как runtime snapshot с smoke decisions по context keys.
- Stage 38: clarification runtime execution, где Stage 37 runtime policy
  выполняет bounded clarification loop на Stage 30 partial-context episodes и
  пишет answer outputs + active compute.
- Stage 39: clarification runtime self-monitor, где Stage 38 traces
  превращаются в monitor records с реконструкцией runtime path, output,
  selected keys, target preservation и active compute.
- Stage 40: clarification monitor feedback, где Stage 39 monitor records
  становятся deterministic feedback events, key rewards и alert summary для
  будущего runtime policy review.
- Stage 41A: adversarial risk audit, где Stage 40 feedback проверяется на
  missing/noisy/correlated/rare/contradictory/drift context cases и блокирует
  policy promotion до guarded replay.
- Stage 41B: guarded policy review, где Stage 40 rewards превращаются только
  в quarantined candidate policy для Stage 42 replay, а runtime promotion
  остается blocked из-за Stage 41A risks.
- Stage 42: clean + adversarial replay, где quarantined Stage 41B candidate
  сравнивается с current runtime policy, fixed-order и entropy controls на
  чистых Stage 30 episodes и Stage 41A risk cases; runtime promotion остается
  blocked, если critical risks остаются.
- Stage 42C: evidence/conflict guard, где critical unsafe answers из Stage 42
  переводятся в evidence, context-integrity или validity-window checks; clean
  ответы сохраняются, а runtime promotion остается blocked до Stage 43 review.
- Stage 43: guarded policy promotion, где Stage 41B candidate + Stage 42 replay
  + Stage 42C guard превращаются в guarded cached-replay runtime candidate
  manifest с rollback metadata и строгими claim guardrails.
- Planned risk-first gates: перед Stage 41+ нужно добавить adversarial
  context tests, provenance poisoning checks, policy rollback audit и scale
  cliff checks, чтобы feedback loop не закреплял красивую, но хрупкую policy.
- Planned storage research gate: перед 10M+ conditional records JSON остается
  только форматом отчетов/fixtures; primary memory должна пройти benchmark
  append-only evidence log + deterministic compaction + indexed query store.

## Context-Aware Memory Principle

NeoCortex-X не должен хранить глобальные факты вида `X = Y`, если значение
зависит от условий. Базовая единица долговременной памяти:

```text
subject + property + value + context_vector + confidence + provenance
```

Пример `sky -> blue` является плохой памятью. Правильная форма:
`sky.expected_color = blue` при `time_of_day=daytime`,
`weather=clear`, `location=earth`. Наблюдение `sky = dark` ночью не является
конфликтом; это другой conditional slot. Конфликт возникает только когда
контекст совпадает, а value/provenance расходятся.

Практический contract для stages 30+:

- partial context возвращает candidate set, а не преждевременный global answer;
- ambiguity превращается в уточнение недостающего context key;
- clarification policy учится выбирать самый информативный ключ;
- JSON reports не считаются primary memory и используются только для smoke,
  fixtures и deterministic summaries;
- scale work должен проверять p95/p99 lookup, ingest, compaction, disk size,
  replay safety и index usage на больших conditional datasets.

Важно: Stage 1/2 пока используют hand-coded agents. Это проверка harness,
артефактов, метрик и pass/fail контракта, а не claim про обученный SSM или AGI.
Следующий шаг - заменить `kv_memory` на trained SSM/KV core без изменения
формата задач и evidence artifact, затем подключать live-search backend только
после deterministic provenance gate.

## Risk Register And Hard Gates

Текущая цепочка Stage 30-40 проверяет context-aware retrieval, clarification
policy, runtime execution, self-monitor и feedback на controlled synthetic
episodes. Это сильный harness, но не доказательство robust open-world memory.

Главные риски перед продолжением:

- synthetic overfit: policy может выучить distribution генератора, а не общий
  принцип выбора context key;
- context explosion: новые dimensions могут дать слишком широкий candidate set
  и слишком много уточняющих вопросов;
- wrong-key reinforcement: Stage 40 rewards могут усиливать ключи, которые
  полезны только на текущем наборе задач;
- provenance poisoning: live sources могут занести ложные или устаревшие
  evidence packets;
- replay/live gap: deterministic replay не равен устойчивости к изменяющимся
  источникам и temporal drift;
- context/causality confusion: conditional memory может хранить правильные
  условия, но принять коррелирующий context key за причинный;
- over-clarification: система может сохранять target slot, но задавать слишком
  много уточняющих вопросов вместо полезного ответа;
- context schema drift: старый набор keys может не выражать важные условия в
  новой предметной области;
- storage cliff: thousands-of-records SQLite smoke не доказывает 10M+ latency,
  compaction и concurrency;
- claim drift: Phi, AGI, energy, 86B neurons и quantum claims остаются roadmap
  language, пока нет отдельного evidence gate.

Следующие gates должны идти перед новыми крупными claims:

- adversarial context suite: missing/noisy/conflicting context, aliasing,
  correlated keys, rare contexts;
- provenance corruption suite: stale source, malicious source, contradictory
  source, low-trust source;
- policy rollback audit: плохой Stage 40 feedback не должен промотиться в
  runtime policy без replay evidence;
- Stage 41A реализует первый risk gate: promotion должен быть blocked, пока
  adversarial audit не пройдет guarded replay/review;
- Stage 41B разрешает только quarantined candidate policy для Stage 42 replay,
  но не runtime promotion;
- Stage 42 должен сравнить candidate с current runtime и controls на clean +
  adversarial suites, но pass Stage 42 не означает automatic promotion;
- Stage 42C/provenance-conflict guard нужен, если остаются critical
  same-context/noisy-context риски;
- Stage 42C должен обнулять critical unsafe answers после guard, но всё равно
  не является runtime promotion;
- Stage 43 может промотить только guarded cached-replay runtime candidate с
  rollback metadata; live-source trust и production memory остаются отдельными
  gates;
- scale cliff benchmark: p95/p99 lookup, compaction time, disk/index overhead,
  deterministic rebuild;
- live-source trust gate: internet integration только через cached,
  replayable, source-scored evidence packets.

## Команды

```bash
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
make neocortex-x-stage1b-trained-smoke
make neocortex-x-stage2-smoke
make neocortex-x-stage2b-learned-smoke
make neocortex-x-stage3-smoke
make neocortex-x-stage3b-promotion-smoke
make neocortex-x-stage4-smoke
make neocortex-x-stage4b-world-model-smoke
make neocortex-x-stage5-smoke
make neocortex-x-stage5b-cached-smoke
make neocortex-x-stage6-packets-smoke
make neocortex-x-stage7-memory-smoke
make neocortex-x-stage8-freshness-smoke
make neocortex-x-stage9-action-gate-smoke
make neocortex-x-stage10-feedback-smoke
make neocortex-x-stage11-policy-smoke
make neocortex-x-stage12-policy-replay-smoke
make neocortex-x-stage13-promote-policy-smoke
make neocortex-x-stage14-runtime-loader-smoke
make neocortex-x-stage15-runtime-execution-smoke
make neocortex-x-stage16-contextual-memory-smoke
make neocortex-x-stage17-contextual-action-gate-smoke
make neocortex-x-stage18-contextual-feedback-smoke
make neocortex-x-stage19-contextual-policy-smoke
make neocortex-x-stage20-contextual-policy-replay-smoke
make neocortex-x-stage21-contextual-policy-promotion-smoke
make neocortex-x-stage22-contextual-runtime-loader-smoke
make neocortex-x-stage23-contextual-runtime-execution-smoke
make neocortex-x-stage24-contextual-self-monitor-smoke
make neocortex-x-stage25-indexed-contextual-store-smoke
make neocortex-x-stage26-store-backed-contextual-retrieval-smoke
make neocortex-x-stage27-store-backed-contextual-action-gate-smoke
make neocortex-x-stage28-evidence-log-compaction-smoke
make neocortex-x-stage29-scale-probe-smoke
make neocortex-x-stage30-partial-context-smoke
make neocortex-x-stage31-disambiguation-gate-smoke
make neocortex-x-stage32-clarification-resolution-smoke
make neocortex-x-stage33-clarification-feedback-smoke
make neocortex-x-stage34-clarification-policy-smoke
make neocortex-x-stage35-clarification-policy-replay-smoke
make neocortex-x-stage36-clarification-policy-promotion-smoke
make neocortex-x-stage37-clarification-runtime-loader-smoke
make neocortex-x-stage38-clarification-runtime-execution-smoke
make neocortex-x-stage39-clarification-runtime-self-monitor-smoke
make neocortex-x-stage40-clarification-monitor-feedback-smoke
make neocortex-x-stage41a-adversarial-risk-audit-smoke
make neocortex-x-stage41b-guarded-policy-review-smoke
make neocortex-x-stage42-clean-adversarial-replay-smoke
make neocortex-x-stage42c-evidence-conflict-guard-smoke
make neocortex-x-stage43-guarded-policy-promotion-smoke
```

Прямой запуск:

```bash
python3 research/neocortex_x/core_experiment.py \
  --stage stage0 \
  --output-json artifacts/neocortex_x/stage0_smoke.json \
  --output-trace artifacts/neocortex_x/stage0_trace.jsonl

python3 research/neocortex_x/core_experiment.py \
  --stage stage1 \
  --seeds 11,17,23,29,31,37 \
  --episodes 128 \
  --key-count 16 \
  --value-count 16 \
  --binding-count 4 \
  --delay 24 \
  --baseline-window 8 \
  --output-json artifacts/neocortex_x/stage1_recall_to_action.json

python3 research/neocortex_x/stage1b_trained_recall_to_action.py \
  --output-json artifacts/neocortex_x/stage1b_trained_recall_to_action.json \
  --inner-json artifacts/neocortex_x/stage1b_inner_long_context_recall.json

python3 research/neocortex_x/core_experiment.py \
  --stage stage2 \
  --seeds 11,17,23,29,31,37 \
  --episodes 128 \
  --tape-len 8 \
  --input-len-min 4 \
  --input-len-max 8 \
  --max-steps 64 \
  --output-json artifacts/neocortex_x/stage2_stateful_tape_world.json \
  --output-trace artifacts/neocortex_x/stage2_trace.jsonl

python3 research/neocortex_x/stage2b_learned_tape_policy.py \
  --output-json artifacts/neocortex_x/stage2b_learned_tape_policy.json

python3 research/neocortex_x/core_experiment.py \
  --stage stage3 \
  --seeds 11,17,23,29,31,37 \
  --episodes 128 \
  --stage3-tape-len 12 \
  --stage3-segment-len 3 \
  --stage3-call-overhead 1 \
  --output-json artifacts/neocortex_x/stage3_procedure_reuse.json

python3 research/neocortex_x/stage3b_learned_procedure_promotion.py \
  --output-json artifacts/neocortex_x/stage3b_learned_procedure_promotion.json

python3 research/neocortex_x/core_experiment.py \
  --stage stage4 \
  --seeds 11,17,23,29,31,37 \
  --episodes 96 \
  --stage4-corridor-len 9 \
  --stage4-horizon 4 \
  --output-json artifacts/neocortex_x/stage4_planning.json

python3 research/neocortex_x/stage4b_learned_world_model.py \
  --output-json artifacts/neocortex_x/stage4b_learned_world_model.json

python3 research/neocortex_x/core_experiment.py \
  --stage stage5 \
  --seeds 11,17,23,29,31,37 \
  --episodes 96 \
  --stage5-contradiction-margin 0.20 \
  --output-json artifacts/neocortex_x/stage5_open_world_adapter.json

python3 research/neocortex_x/stage5b_live_cached_adapter.py \
  --mode cached \
  --fixture-json research/neocortex_x/fixtures/stage5b_cached_observations.json \
  --output-json artifacts/neocortex_x/stage5b_live_cached_adapter.json

python3 research/neocortex_x/stage6_evidence_packets.py \
  --fixture-json research/neocortex_x/fixtures/stage5b_cached_observations.json \
  --output-json artifacts/neocortex_x/stage6_evidence_packets.json

python3 research/neocortex_x/stage7_memory_retrieval.py \
  --packet-json artifacts/neocortex_x/stage6_evidence_packets.json \
  --output-json artifacts/neocortex_x/stage7_memory_retrieval.json

python3 research/neocortex_x/stage8_temporal_freshness.py \
  --packet-json artifacts/neocortex_x/stage6_evidence_packets.json \
  --output-json artifacts/neocortex_x/stage8_temporal_freshness.json

python3 research/neocortex_x/stage9_action_gate.py \
  --temporal-json artifacts/neocortex_x/stage8_temporal_freshness.json \
  --output-json artifacts/neocortex_x/stage9_action_gate.json

python3 research/neocortex_x/stage10_feedback_loop.py \
  --action-gate-json artifacts/neocortex_x/stage9_action_gate.json \
  --output-json artifacts/neocortex_x/stage10_feedback_loop.json

python3 research/neocortex_x/stage11_policy_update.py \
  --feedback-json artifacts/neocortex_x/stage10_feedback_loop.json \
  --output-json artifacts/neocortex_x/stage11_policy_update.json

python3 research/neocortex_x/stage12_policy_replay.py \
  --temporal-json artifacts/neocortex_x/stage8_temporal_freshness.json \
  --policy-json artifacts/neocortex_x/stage11_policy_update.json \
  --output-json artifacts/neocortex_x/stage12_policy_replay.json

python3 research/neocortex_x/stage13_policy_promotion.py \
  --replay-json artifacts/neocortex_x/stage12_policy_replay.json \
  --policy-json artifacts/neocortex_x/stage11_policy_update.json \
  --output-json artifacts/neocortex_x/stage13_policy_promotion.json

python3 research/neocortex_x/stage14_runtime_loader.py \
  --promotion-json artifacts/neocortex_x/stage13_policy_promotion.json \
  --output-json artifacts/neocortex_x/stage14_runtime_loader.json

python3 research/neocortex_x/stage15_runtime_execution.py \
  --runtime-json artifacts/neocortex_x/stage14_runtime_loader.json \
  --temporal-json artifacts/neocortex_x/stage8_temporal_freshness.json \
  --output-json artifacts/neocortex_x/stage15_runtime_execution.json

python3 research/neocortex_x/stage16_contextual_memory.py \
  --packet-json artifacts/neocortex_x/stage6_evidence_packets.json \
  --output-json artifacts/neocortex_x/stage16_contextual_memory.json

python3 research/neocortex_x/stage17_contextual_action_gate.py \
  --contextual-json artifacts/neocortex_x/stage16_contextual_memory.json \
  --output-json artifacts/neocortex_x/stage17_contextual_action_gate.json

python3 research/neocortex_x/stage18_contextual_feedback.py \
  --contextual-action-json artifacts/neocortex_x/stage17_contextual_action_gate.json \
  --output-json artifacts/neocortex_x/stage18_contextual_feedback.json

python3 research/neocortex_x/stage19_contextual_policy_update.py \
  --feedback-json artifacts/neocortex_x/stage18_contextual_feedback.json \
  --output-json artifacts/neocortex_x/stage19_contextual_policy_update.json

python3 research/neocortex_x/stage20_contextual_policy_replay.py \
  --contextual-action-json artifacts/neocortex_x/stage17_contextual_action_gate.json \
  --policy-json artifacts/neocortex_x/stage19_contextual_policy_update.json \
  --output-json artifacts/neocortex_x/stage20_contextual_policy_replay.json

python3 research/neocortex_x/stage21_contextual_policy_promotion.py \
  --replay-json artifacts/neocortex_x/stage20_contextual_policy_replay.json \
  --policy-json artifacts/neocortex_x/stage19_contextual_policy_update.json \
  --output-json artifacts/neocortex_x/stage21_contextual_policy_promotion.json

python3 research/neocortex_x/stage22_contextual_runtime_loader.py \
  --promotion-json artifacts/neocortex_x/stage21_contextual_policy_promotion.json \
  --output-json artifacts/neocortex_x/stage22_contextual_runtime_loader.json

python3 research/neocortex_x/stage23_contextual_runtime_execution.py \
  --runtime-json artifacts/neocortex_x/stage22_contextual_runtime_loader.json \
  --contextual-json artifacts/neocortex_x/stage16_contextual_memory.json \
  --output-json artifacts/neocortex_x/stage23_contextual_runtime_execution.json

python3 research/neocortex_x/stage24_contextual_self_monitor.py \
  --execution-json artifacts/neocortex_x/stage23_contextual_runtime_execution.json \
  --output-json artifacts/neocortex_x/stage24_contextual_self_monitor.json

python3 research/neocortex_x/stage25_indexed_contextual_memory_store.py \
  --contextual-json artifacts/neocortex_x/stage16_contextual_memory.json \
  --output-db artifacts/neocortex_x/stage25_contextual_memory.sqlite \
  --output-json artifacts/neocortex_x/stage25_indexed_contextual_memory_store.json

python3 research/neocortex_x/stage26_store_backed_contextual_retrieval.py \
  --store-json artifacts/neocortex_x/stage25_indexed_contextual_memory_store.json \
  --store-db artifacts/neocortex_x/stage25_contextual_memory.sqlite \
  --output-json artifacts/neocortex_x/stage26_store_backed_contextual_retrieval.json

python3 research/neocortex_x/stage27_store_backed_contextual_action_gate.py \
  --retrieval-json artifacts/neocortex_x/stage26_store_backed_contextual_retrieval.json \
  --output-json artifacts/neocortex_x/stage27_store_backed_contextual_action_gate.json

python3 research/neocortex_x/stage28_append_only_evidence_log_compaction.py \
  --contextual-json artifacts/neocortex_x/stage16_contextual_memory.json \
  --output-db artifacts/neocortex_x/stage28_evidence_log_compacted_store.sqlite \
  --output-json artifacts/neocortex_x/stage28_append_only_evidence_log_compaction.json

python3 research/neocortex_x/stage29_contextual_memory_scale_probe.py \
  --records 5000 \
  --queries 256 \
  --json-scan-queries 64 \
  --output-db artifacts/neocortex_x/stage29_scale_probe.sqlite \
  --output-json artifacts/neocortex_x/stage29_contextual_memory_scale_probe.json

python3 research/neocortex_x/stage30_partial_context_scale_retrieval.py \
  --records 5000 \
  --queries 256 \
  --json-scan-queries 64 \
  --candidate-limit 64 \
  --output-db artifacts/neocortex_x/stage30_partial_context.sqlite \
  --output-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json

python3 research/neocortex_x/stage31_partial_context_disambiguation_gate.py \
  --partial-context-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json \
  --output-json artifacts/neocortex_x/stage31_partial_context_disambiguation_gate.json

python3 research/neocortex_x/stage32_clarification_resolution_loop.py \
  --partial-context-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json \
  --disambiguation-json artifacts/neocortex_x/stage31_partial_context_disambiguation_gate.json \
  --output-json artifacts/neocortex_x/stage32_clarification_resolution_loop.json

python3 research/neocortex_x/stage33_clarification_feedback_ledger.py \
  --resolution-json artifacts/neocortex_x/stage32_clarification_resolution_loop.json \
  --output-json artifacts/neocortex_x/stage33_clarification_feedback_ledger.json

python3 research/neocortex_x/stage34_clarification_policy_update.py \
  --feedback-json artifacts/neocortex_x/stage33_clarification_feedback_ledger.json \
  --output-json artifacts/neocortex_x/stage34_clarification_policy_update.json

python3 research/neocortex_x/stage35_clarification_policy_replay.py \
  --partial-context-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json \
  --policy-json artifacts/neocortex_x/stage34_clarification_policy_update.json \
  --output-json artifacts/neocortex_x/stage35_clarification_policy_replay.json

python3 research/neocortex_x/stage36_clarification_policy_promotion.py \
  --replay-json artifacts/neocortex_x/stage35_clarification_policy_replay.json \
  --policy-json artifacts/neocortex_x/stage34_clarification_policy_update.json \
  --output-json artifacts/neocortex_x/stage36_clarification_policy_promotion.json

python3 research/neocortex_x/stage37_clarification_runtime_loader.py \
  --promotion-json artifacts/neocortex_x/stage36_clarification_policy_promotion.json \
  --output-json artifacts/neocortex_x/stage37_clarification_runtime_loader.json

python3 research/neocortex_x/stage38_clarification_runtime_execution.py \
  --runtime-json artifacts/neocortex_x/stage37_clarification_runtime_loader.json \
  --partial-context-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json \
  --output-json artifacts/neocortex_x/stage38_clarification_runtime_execution.json

python3 research/neocortex_x/stage39_clarification_runtime_self_monitor.py \
  --execution-json artifacts/neocortex_x/stage38_clarification_runtime_execution.json \
  --output-json artifacts/neocortex_x/stage39_clarification_runtime_self_monitor.json

python3 research/neocortex_x/stage40_clarification_monitor_feedback.py \
  --monitor-json artifacts/neocortex_x/stage39_clarification_runtime_self_monitor.json \
  --output-json artifacts/neocortex_x/stage40_clarification_monitor_feedback.json

python3 research/neocortex_x/stage41a_adversarial_risk_audit.py \
  --partial-context-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json \
  --runtime-json artifacts/neocortex_x/stage37_clarification_runtime_loader.json \
  --feedback-json artifacts/neocortex_x/stage40_clarification_monitor_feedback.json \
  --output-json artifacts/neocortex_x/stage41a_adversarial_risk_audit.json

python3 research/neocortex_x/stage41b_guarded_policy_review.py \
  --feedback-json artifacts/neocortex_x/stage40_clarification_monitor_feedback.json \
  --risk-audit-json artifacts/neocortex_x/stage41a_adversarial_risk_audit.json \
  --output-json artifacts/neocortex_x/stage41b_guarded_policy_review.json

python3 research/neocortex_x/stage42_clean_adversarial_replay.py \
  --partial-context-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json \
  --runtime-json artifacts/neocortex_x/stage37_clarification_runtime_loader.json \
  --risk-audit-json artifacts/neocortex_x/stage41a_adversarial_risk_audit.json \
  --guarded-review-json artifacts/neocortex_x/stage41b_guarded_policy_review.json \
  --output-json artifacts/neocortex_x/stage42_clean_adversarial_replay.json

python3 research/neocortex_x/stage42c_evidence_conflict_guard.py \
  --stage42-json artifacts/neocortex_x/stage42_clean_adversarial_replay.json \
  --output-json artifacts/neocortex_x/stage42c_evidence_conflict_guard.json

python3 research/neocortex_x/stage43_guarded_policy_promotion.py \
  --stage42-json artifacts/neocortex_x/stage42_clean_adversarial_replay.json \
  --stage42c-json artifacts/neocortex_x/stage42c_evidence_conflict_guard.json \
  --guarded-review-json artifacts/neocortex_x/stage41b_guarded_policy_review.json \
  --output-json artifacts/neocortex_x/stage43_guarded_policy_promotion.json
```

## Артефакты

- `artifacts/neocortex_x/stage0_smoke.json`
- `artifacts/neocortex_x/stage0_trace.jsonl`
- `artifacts/neocortex_x/stage1_recall_to_action.json`
- `artifacts/neocortex_x/stage1b_trained_recall_to_action.json`
- `artifacts/neocortex_x/stage1b_inner_long_context_recall.json`
- `artifacts/neocortex_x/stage2_stateful_tape_world.json`
- `artifacts/neocortex_x/stage2_trace.jsonl`
- `artifacts/neocortex_x/stage2b_learned_tape_policy.json`
- `artifacts/neocortex_x/stage3_procedure_reuse.json`
- `artifacts/neocortex_x/stage3b_learned_procedure_promotion.json`
- `artifacts/neocortex_x/stage4_planning.json`
- `artifacts/neocortex_x/stage4b_learned_world_model.json`
- `artifacts/neocortex_x/stage5_open_world_adapter.json`
- `artifacts/neocortex_x/stage5b_live_cached_adapter.json`
- `artifacts/neocortex_x/stage6_evidence_packets.json`
- `artifacts/neocortex_x/stage7_memory_retrieval.json`
- `artifacts/neocortex_x/stage8_temporal_freshness.json`
- `artifacts/neocortex_x/stage9_action_gate.json`
- `artifacts/neocortex_x/stage10_feedback_loop.json`
- `artifacts/neocortex_x/stage11_policy_update.json`
- `artifacts/neocortex_x/stage12_policy_replay.json`
- `artifacts/neocortex_x/stage13_policy_promotion.json`
- `artifacts/neocortex_x/stage14_runtime_loader.json`
- `artifacts/neocortex_x/stage15_runtime_execution.json`
- `artifacts/neocortex_x/stage16_contextual_memory.json`
- `artifacts/neocortex_x/stage17_contextual_action_gate.json`
- `artifacts/neocortex_x/stage18_contextual_feedback.json`
- `artifacts/neocortex_x/stage19_contextual_policy_update.json`
- `artifacts/neocortex_x/stage20_contextual_policy_replay.json`
- `artifacts/neocortex_x/stage21_contextual_policy_promotion.json`
- `artifacts/neocortex_x/stage22_contextual_runtime_loader.json`
- `artifacts/neocortex_x/stage23_contextual_runtime_execution.json`
- `artifacts/neocortex_x/stage24_contextual_self_monitor.json`
- `artifacts/neocortex_x/stage25_contextual_memory.sqlite`
- `artifacts/neocortex_x/stage25_indexed_contextual_memory_store.json`
- `artifacts/neocortex_x/stage26_store_backed_contextual_retrieval.json`
- `artifacts/neocortex_x/stage27_store_backed_contextual_action_gate.json`
- `artifacts/neocortex_x/stage28_evidence_log_compacted_store.sqlite`
- `artifacts/neocortex_x/stage28_append_only_evidence_log_compaction.json`
- `artifacts/neocortex_x/stage29_scale_probe.sqlite`
- `artifacts/neocortex_x/stage29_contextual_memory_scale_probe.json`
- `artifacts/neocortex_x/stage30_partial_context.sqlite`
- `artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json`
- `artifacts/neocortex_x/stage31_partial_context_disambiguation_gate.json`
- `artifacts/neocortex_x/stage32_clarification_resolution_loop.json`
- `artifacts/neocortex_x/stage33_clarification_feedback_ledger.json`
- `artifacts/neocortex_x/stage34_clarification_policy_update.json`
- `artifacts/neocortex_x/stage35_clarification_policy_replay.json`
- `artifacts/neocortex_x/stage36_clarification_policy_promotion.json`
- `artifacts/neocortex_x/stage37_clarification_runtime_loader.json`
- `artifacts/neocortex_x/stage38_clarification_runtime_execution.json`
- `artifacts/neocortex_x/stage39_clarification_runtime_self_monitor.json`
- `artifacts/neocortex_x/stage40_clarification_monitor_feedback.json`
- `artifacts/neocortex_x/stage41a_adversarial_risk_audit.json`
- `artifacts/neocortex_x/stage41b_guarded_policy_review.json`
- `artifacts/neocortex_x/stage42_clean_adversarial_replay.json`
- `artifacts/neocortex_x/stage42c_evidence_conflict_guard.json`
- `artifacts/neocortex_x/stage43_guarded_policy_promotion.json`

`artifacts/` игнорируется git и регенерируется командами выше.

## Context-Aware Memory Principle

NeoCortex-X memory must not store unconditional facts such as `sky = blue`.
That shape loses the conditions that made the observation true. The stable
unit is a conditional memory slot:

```text
subject + property + value + context_vector + confidence + provenance
```

Example:

```text
subject = sky
property = expected_color
value = blue
context = {time_of_day: daytime, weather: clear, location: earth}
```

If the system later sees `sky = black`, that is not automatically a conflict.
With `time_of_day = night`, it is a different context. With missing context,
the correct action is to ask for context. With same-context incompatible value,
the correct action is to request more evidence. This is the bridge from fact
memory into world-model memory: reality is usually `if context C, then X`,
not global `X = Y`.

Brainstorm guardrail: conditional slots solve the bad graph edge
`sky --color--> blue`, but they do not automatically solve causality,
provenance, or scale. The memory layer must distinguish:

- different context from real conflict;
- missing context from false contradiction;
- correlated key from causal key;
- repeated evidence from duplicated source;
- useful clarification from over-clarification.

## Scale Track

JSON is not the memory database. JSON is acceptable for deterministic smoke
reports, small fixtures, and replay summaries. At 10M+ records it becomes the
wrong substrate for primary memory: poor partial updates, expensive scans,
weak indexing, and awkward concurrency.

Scale research is part of the roadmap before larger memory benchmarks:

- schema for conditional slots, context dimensions, provenance, confidence,
  temporal validity, and alias groups;
- indexed symbolic lookup by `(subject, property)`, exact/partial context, and
  provenance;
- append-only evidence log plus compacted snapshots;
- deterministic replay from evidence log into a queryable store;
- Stage 28 preflights that contract: append-only evidence events are the
  source of truth, while the SQLite query store is a compacted indexed view;
- Stage 29 adds a fast thousands-of-records probe for ingest, compaction,
  indexed lookup latency, and JSON scan baseline speedup before the 10M+ run;
- Stage 30 adds partial-context retrieval over that scale shape, because real
  observations often include only some context dimensions;
- Stage 31 maps partial-context ambiguity to bounded actions: answer when the
  candidate set is unique, otherwise ask for the most informative missing
  context dimension;
- Stage 32 closes that loop by applying the clarification answer, narrowing
  candidates, and resolving to the target conditional slot;
- Stage 33 records feedback over clarification keys, making the next policy
  update measurable instead of heuristic;
- Stage 34 turns that feedback into a deterministic clarification-key policy
  snapshot with normalized probabilities;
- Stage 38 executes the promoted Stage 37 clarification runtime policy over
  Stage 30 partial-context episodes, records bounded clarification steps,
  answer outputs, target preservation, and active compute;
- Stage 39 monitors those runtime traces by reconstructing direct-answer vs
  clarify-then-answer paths, selected context keys, output consistency, target
  preservation, and active compute;
- Stage 40 turns Stage 39 monitor records into deterministic feedback and
  alert events with key rewards, runtime path rewards, and critical/info alert
  accounting;
- Stage 41A adds the first risk-first gate by replaying adversarial context
  variants and blocking policy promotion until guarded review/replay exists;
- Stage 41B creates a quarantined candidate policy from Stage 40 rewards while
  keeping runtime promotion blocked until Stage 42 clean+adversarial replay;
- Stage 42 replays that quarantined candidate on clean Stage 30 episodes plus
  Stage 41A adversarial cases and compares it with current runtime, fixed-order
  and entropy controls while keeping promotion blocked if critical risks remain;
- Stage 42C adds a deterministic evidence/conflict guard over Stage 42 records:
  same-context contradictions require evidence review, noisy context requires
  context integrity check, and temporal drift requires validity-window check;
- Stage 43 promotes only a guarded cached-replay runtime candidate with rollback
  metadata and explicit claim guardrails; Stage 44 must load it before runtime
  use, while live-source trust and 10M+ memory scale remain separate gates;
- risk-first gates should test adversarial context, provenance poisoning,
  policy rollback, scale cliffs, and claim drift before adding stronger
  runtime-policy or live-world claims;
- storage candidates: SQLite for preflight, DuckDB/Postgres for analytical and
  relational scale, Parquet/Arrow for columnar snapshots, RocksDB/LSM for
  high-write local stores, and hybrid symbolic+vector indexes for semantic
  candidate retrieval;
- benchmark target: 10M+ conditional records with measured ingest throughput,
  lookup latency, storage size, index usage, and replay determinism.

Before 10M+ scale work, the plan should include a storage research stage that
benchmarks exact lookup, partial-context retrieval, ambiguity-key selection,
append-only replay rebuild, compaction cost, index size, alias migration and
concurrent reads/writes. JSON scan must stay as a baseline to beat, not the
memory substrate.
