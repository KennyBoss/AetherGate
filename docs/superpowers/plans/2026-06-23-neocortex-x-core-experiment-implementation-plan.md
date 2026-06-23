# NeoCortex-X Core Experiment Implementation Plan

Статус: план реализации для утвержденного design spec.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.
Дата: 2026-06-23.

## Принцип Порядка

Идем строго по ladder из spec:

1. Stage 0: instrumentation.
2. Stage 1: recall-to-action.
3. Stage 2+ не начинаем, пока Stage 0/1 не имеют проверяемых артефактов.

Этот план не пытается сразу строить весь NeoCortex-X. Его задача — дать первый
исполняемый и воспроизводимый каркас, который можно расширять без подмены
научного claim.

## Skill Status

Spec требовал следующий planning artifact. Отдельный `writing-plans` skill в
доступных навыках и локальных skill директориях не найден, поэтому этот документ
является fallback planning artifact. Он сохраняет требуемую структуру: файлы,
команды, артефакты, thresholds и pass/fail tests.

## Stage 0: Instrumentation

### Цель

Сделать deterministic tick-by-tick harness, который логирует:

- tick id;
- phase/event type;
- observation;
- gates: read/write/env_action/call_codepy/stop;
- predicted_delta;
- actual_delta;
- state_before/state_after digest;
- reward;
- active_compute counters;
- replay check.

### Файлы

- `research/neocortex_x/core_experiment.py`
  - CLI для Stage 0 и Stage 1.
  - Dataclass/state model для простой controlled environment.
  - `run_stage0_smoke(...)`.
  - `replay_trace(...)`.
  - JSON/JSONL artifact writers.
- `research/neocortex_x/README.md`
  - Назначение подпроекта, команды запуска, статус.
- `Makefile`
  - `neocortex-x-stage0-smoke`.
  - `neocortex-x-stage1-smoke`.

### Артефакты

- `artifacts/neocortex_x/stage0_smoke.json`
- `artifacts/neocortex_x/stage0_trace.jsonl`

### Stage 0 Pass/Fail

PASS если:

1. trace содержит хотя бы bind, filler, query, act и stop ticks;
2. каждый tick содержит required fields;
3. deterministic replay из JSONL восстанавливает все `state_after`;
4. `protocol_evidence.valid = true`;
5. `active_compute.total_ticks > 0`;
6. `world_delta_exact_rate = 1.0` для deterministic hand-coded predictor.

FAIL если:

- replay расходится;
- отсутствуют required trace fields;
- artifact не содержит `protocol_evidence`;
- command возвращает non-zero exit.

### Команда

```bash
python3 research/neocortex_x/core_experiment.py \
  --stage stage0 \
  --output-json artifacts/neocortex_x/stage0_smoke.json \
  --output-trace artifacts/neocortex_x/stage0_trace.jsonl
```

## Stage 1: Recall-To-Action

### Цель

Перенести delayed recall из token prediction в действие: агент должен не просто
вспомнить value, а выбрать correct environment action на query.

### Минимальный Эксперимент

Task stream:

```text
bind(key_i, value_i)
filler * delay
query(key_j)
act(value_j)
stop
```

Agents:

- `kv_memory`: explicit key->value memory, gates write at bind and read at query.
- `context_window`: baseline, который видит только последние `baseline_window`
  событий и должен проваливаться при `delay > baseline_window`.

Это Stage 1 smoke, не финальный trained SSM. Его задача — доказать, что
instrumentation, replay, protocol evidence и recall-to-action metric работают.
Позже этот же harness заменит hand-coded KV agent на trained SSM/KV core.

### Метрики

- `action_accuracy`: доля query/action эпизодов с правильным действием.
- `state_consistency`: replay exact match.
- `write_read_fidelity`: доля query, где прочитан правильный value.
- `trajectory_horizon`: ticks до первой ошибки, либо полная длина если ошибок нет.
- `active_compute`: reads + writes + actions + total_ticks.
- `kv_minus_context_action_accuracy`.

### Thresholds Для Smoke

Для `--delay > --baseline-window`, `--episodes >= 64`, `--seeds >= 3`:

- `kv_memory.action_accuracy_mean >= 0.95`;
- `context_window.action_accuracy_mean <= chance + 0.20`;
- `kv_minus_context_action_accuracy_mean >= 0.50`;
- `state_consistency_mean = 1.0`;
- `protocol_evidence.valid = true`.

Chance для baseline равен `1 / value_count`.

### Артефакты

- `artifacts/neocortex_x/stage1_recall_to_action.json`
- optional per-seed traces under `artifacts/neocortex_x/stage1_traces/`.

### Команда

```bash
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

## Test And Verification Plan

Минимальная проверка после реализации:

```bash
python3 -m py_compile research/neocortex_x/core_experiment.py
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
```

Дополнительная проверка:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/neocortex_x/stage1_recall_to_action.json")
d = json.loads(p.read_text())
assert d["protocol_evidence"]["valid"] is True
assert d["aggregate"]["comparison"]["kv_minus_context_action_accuracy"]["mean"] >= 0.5
print("neo cortex stage1 artifact ok")
PY
```

## Commit Plan

1. Commit this implementation plan.
2. Implement Stage 0/1 harness.
3. Run compile + smoke commands.
4. Commit code + Makefile + README only after smoke artifacts pass.

## Out Of Scope For This Plan

- trained differentiable SSM/KV agent;
- CodePy procedure promotion;
- Axis-S / Axis-C controls;
- dense+curriculum and random-routing controls;
- Phi-like probes beyond minimal metrics-schema fields;
- internet/search;
- neuromorphic/quantum/spiking backends.

Эти пункты начинаются после Stage 0/1 smoke, чтобы не смешивать
instrumentation, task semantics и learning claims в один непроверяемый шаг.
