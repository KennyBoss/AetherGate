# NeoCortex-X Core Experiment

Этот подпроект реализует первые два шага утвержденного плана:

- Stage 0: deterministic tick-by-tick instrumentation с replay из JSONL trace.
- Stage 1: recall-to-action smoke, где `kv_memory` должен превратить отложенное
  bind/query воспоминание в корректное действие, а `context_window` служит
  короткоконтекстным baseline.

Важно: Stage 1 пока использует hand-coded agents. Это проверка harness,
артефактов, метрик и pass/fail контракта, а не claim про обученный SSM или AGI.
Следующий шаг - заменить `kv_memory` на trained SSM/KV core без изменения
формата задач и evidence artifact.

## Команды

```bash
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
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
```

## Артефакты

- `artifacts/neocortex_x/stage0_smoke.json`
- `artifacts/neocortex_x/stage0_trace.jsonl`
- `artifacts/neocortex_x/stage1_recall_to_action.json`

`artifacts/` игнорируется git и регенерируется командами выше.
