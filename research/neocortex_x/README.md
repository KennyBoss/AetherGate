# NeoCortex-X Core Experiment

Этот подпроект реализует первые шаги утвержденного плана:

- Stage 0: deterministic tick-by-tick instrumentation с replay из JSONL trace.
- Stage 1: recall-to-action smoke, где `kv_memory` должен превратить отложенное
  bind/query воспоминание в корректное действие, а `context_window` служит
  короткоконтекстным baseline.
- Stage 2: stateful CodePy-like tape world с `tape`, `ptr`, `pc`, регистрами,
  bounded execution и baseline comparison.
- Stage 3: procedure reuse smoke с bounded `sum_segment`, Axis-S/Axis-C задачами
  и контролями exact-cache/random-routing/inline-loop.

Важно: Stage 1/2 пока используют hand-coded agents. Это проверка harness,
артефактов, метрик и pass/fail контракта, а не claim про обученный SSM или AGI.
Следующий шаг - заменить `kv_memory` на trained SSM/KV core без изменения
формата задач и evidence artifact, затем перейти к planning/lookahead.

## Команды

```bash
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
make neocortex-x-stage2-smoke
make neocortex-x-stage3-smoke
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

python3 research/neocortex_x/core_experiment.py \
  --stage stage3 \
  --seeds 11,17,23,29,31,37 \
  --episodes 128 \
  --stage3-tape-len 12 \
  --stage3-segment-len 3 \
  --stage3-call-overhead 1 \
  --output-json artifacts/neocortex_x/stage3_procedure_reuse.json
```

## Артефакты

- `artifacts/neocortex_x/stage0_smoke.json`
- `artifacts/neocortex_x/stage0_trace.jsonl`
- `artifacts/neocortex_x/stage1_recall_to_action.json`
- `artifacts/neocortex_x/stage2_stateful_tape_world.json`
- `artifacts/neocortex_x/stage2_trace.jsonl`
- `artifacts/neocortex_x/stage3_procedure_reuse.json`

`artifacts/` игнорируется git и регенерируется командами выше.
