# NeoCortex-X Stage 4 Planning Plan

Статус: continuation plan после Stage 3 procedure reuse smoke.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.
Дата: 2026-06-24.

## Цель

Реализовать следующий rung ladder:

> Stage 4: добавить short lookahead по predicted state deltas. Exit: planned
> action selection бьет greedy action selection при matched compute или
> показывает явный compute/quality frontier.

Это smoke для planning contract, не learned world model. Predictor deterministic
и exact, чтобы проверить logging, scoring, compute accounting и greedy baseline.

## Environment

Grid-world corridor with traps:

- state: position, goal, trap cells, budget, halted;
- actions: left/right/stay;
- immediate reward: tempting local reward near trap;
- terminal reward: goal reward;
- penalty: trap and step cost.

Greedy chooses action by one-step reward. Planner evaluates bounded action
sequences using predicted deltas and chooses the best first action.

## Agents

- `lookahead_planner`: horizon-limited search over action sequences.
- `greedy_action`: one-step reward maximizer.

Both use the same deterministic transition function. Planner pays planning
compute proportional to expanded nodes.

## Metrics

- `task_success`: reaches goal without trap.
- `return`: cumulative reward.
- `planning_nodes`: expanded predicted states.
- `quality_frontier`: planner return minus greedy return vs compute overhead.
- `world_delta_exact_rate`: predicted transition matches actual transition.
- `trajectory_horizon`: steps until terminal or trap.

## Pass/Fail Thresholds

For `--seeds 11,17,23,29,31,37`, `--episodes >= 96`:

- `protocol_evidence.valid = true`;
- `lookahead_planner.task_success_mean >= 0.90`;
- `greedy_action.task_success_mean <= 0.60`;
- `planner_minus_greedy_return_mean >= 2.0`;
- `planner_minus_greedy_success_mean >= 0.30`;
- `lookahead_planner.world_delta_exact_rate_mean = 1.0`;
- `lookahead_planner.planning_nodes_mean > greedy_action.planning_nodes_mean`.

## Files

- `research/neocortex_x/core_experiment.py`
  - add `--stage stage4`;
  - planning task generator, exact predictor, greedy/planner agents.
- `research/neocortex_x/README.md`
  - document Stage 4 command and artifact.
- `Makefile`
  - add `neocortex-x-stage4-smoke`.

## Artifacts

- `artifacts/neocortex_x/stage4_planning.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/core_experiment.py
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
make neocortex-x-stage2-smoke
make neocortex-x-stage3-smoke
make neocortex-x-stage4-smoke
```

Additional JSON assertion:

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("artifacts/neocortex_x/stage4_planning.json").read_text())
assert d["protocol_evidence"]["valid"] is True
assert d["pass"] is True
assert d["aggregate"]["comparison"]["planner_minus_greedy_return"]["mean"] >= 2.0
print("neo cortex stage4 artifact ok")
PY
```

## Out Of Scope

- learned predictor;
- neural policy optimization;
- open-world internet/search;
- Phi-like diagnostics as evidence.

Эти пункты начинаются после Stage 4 smoke, when the planning frontier is
measurable and reproducible.
