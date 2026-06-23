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

Важно: Stage 1/2 пока используют hand-coded agents. Это проверка harness,
артефактов, метрик и pass/fail контракта, а не claim про обученный SSM или AGI.
Следующий шаг - заменить `kv_memory` на trained SSM/KV core без изменения
формата задач и evidence artifact, затем подключать live-search backend только
после deterministic provenance gate.

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

`artifacts/` игнорируется git и регенерируется командами выше.

Stage 16 intentionally keeps JSON as a small deterministic smoke artifact only.
The 10M+ record path is tracked as future storage/index research: indexed
subject/property/context schemas, append-only evidence logs, compacted
snapshots, deterministic replay, and candidates such as SQLite/DuckDB,
Postgres, Parquet/Arrow, RocksDB/LSM, and hybrid symbolic+vector indexes.
