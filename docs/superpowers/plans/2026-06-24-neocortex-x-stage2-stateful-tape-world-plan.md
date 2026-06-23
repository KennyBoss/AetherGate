# NeoCortex-X Stage 2 Stateful Tape World Plan

Статус: continuation plan после завершенных Stage 0/1 smoke.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.
Дата: 2026-06-24.

## Цель

Реализовать следующий rung ladder:

> Stage 2: использовать CodePy tape/register/pointer/pc environment как первый
> world. Exit: state consistency и trajectory horizon бьют baselines на held-out
> stateful tasks.

Это все еще harness smoke, не training claim. Главная задача Stage 2 - добавить
stateful world с исполняемыми действиями, deterministic replay и baseline
comparison, чтобы Stage 3 мог уже проверять procedure reuse.

## Environment

Минимальный CodePy-like world:

- `tape[tape_len]`: фиксированная целочисленная память;
- `input_len`: сколько первых ячеек являются входом;
- `ptr`: указатель на tape;
- `pc`: program counter;
- `registers.r0`: accumulator / result;
- `registers.r1`: current tape value;
- `predicate`: branch condition;
- `halted`: terminal flag;
- `max_steps`: bounded execution budget.

Базовая held-out задача Stage 2: `sum_first_n`.

Reference program:

```text
pc0: r1 = tape[ptr]
pc1: r0 += r1
pc2: ptr += 1
pc3: predicate = ptr < input_len
pc4: pc = 0 if predicate else 5
pc5: stop
```

## Agents And Baselines

- `codepy_loop`: hand-coded bounded pc loop. Это положительный harness oracle,
  который должен пройти replay/state tests.
- `single_cell_baseline`: читает и складывает только первую ячейку, затем
  останавливается. Он изолирует пользу stateful pointer/pc loop.

Оба агента получают одни и те же held-out generated tasks.

## Metrics

- `state_consistency`: final semantic state equals reference final state.
- `task_success`: `halted` и `registers.r0 == target_sum`.
- `trajectory_horizon`: ticks until first semantic divergence from reference,
  либо полная reference trajectory length.
- `replay_consistency`: deterministic replay exact state digest match.
- `world_delta_exact_rate`: predicted delta equals actual delta.
- `active_compute`: total ticks, tape reads, register writes, branches, stops.

## Pass/Fail Thresholds

Для `--seeds 11,17,23,29,31,37`, `--episodes >= 96`:

- `protocol_evidence.valid = true`;
- `codepy_loop.state_consistency_mean >= 0.95`;
- `single_cell_baseline.state_consistency_mean <= 0.40`;
- `codepy_minus_baseline_state_consistency_mean >= 0.50`;
- `codepy_loop.trajectory_horizon_mean >= single_cell_baseline + 4.0`;
- `codepy_loop.replay_consistency_mean = 1.0`;
- `codepy_loop.world_delta_exact_rate_mean = 1.0`.

## Files

- `research/neocortex_x/core_experiment.py`
  - add `--stage stage2`;
  - tape world state, transition, trace, replay, aggregation.
- `research/neocortex_x/README.md`
  - document Stage 2 command and artifact.
- `Makefile`
  - add `neocortex-x-stage2-smoke`.

## Artifacts

- `artifacts/neocortex_x/stage2_stateful_tape_world.json`
- `artifacts/neocortex_x/stage2_trace.jsonl`

`artifacts/` remains ignored and regenerable.

## Verification

```bash
python3 -m py_compile research/neocortex_x/core_experiment.py
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
make neocortex-x-stage2-smoke
```

Additional JSON assertion:

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("artifacts/neocortex_x/stage2_stateful_tape_world.json").read_text())
assert d["protocol_evidence"]["valid"] is True
assert d["pass"] is True
assert d["aggregate"]["comparison"]["codepy_minus_baseline_state_consistency"]["mean"] >= 0.5
print("neo cortex stage2 artifact ok")
PY
```

## Out Of Scope

- learned SSM policy for tape actions;
- procedure promotion and procedure reuse;
- planning/lookahead;
- open-world internet adapter;
- Phi-like diagnostics as evidence.

Эти пункты начинаются после Stage 2 smoke, чтобы не смешивать stateful world
semantics с reuse/planning claims.
