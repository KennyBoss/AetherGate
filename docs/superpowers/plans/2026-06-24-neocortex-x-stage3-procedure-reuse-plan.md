# NeoCortex-X Stage 3 Procedure Reuse Plan

Статус: continuation plan после завершенного Stage 2 smoke.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.
Дата: 2026-06-24.

## Цель

Реализовать следующий rung ladder:

> Stage 3: promote useful CodePy traces в bounded procedures и проверить reuse.
> Exit: active compute падает на held-out Axis-S и Axis-C tasks с контролями.

Это smoke для procedure-reuse контракта, не learned procedure discovery.
Процедуры hand-coded/certified, чтобы проверить artifact schema, controls,
Axis-S/Axis-C splits и active-compute accounting до learned promotion.

## Procedure Memory

Минимальная bounded procedure:

```text
sum_segment(start, length):
  acc = 0
  for offset in range(length):
    acc += tape[start + offset]
  return acc
```

Procedure metadata:

- typed signature: `start:int, length:int -> int`;
- bounded budget: `length + call_overhead`;
- certification: exact on generated train and held-out tasks;
- accounting: `codepy_calls`, `procedure_steps`, `active_compute_saved`.

## Tasks

- `Axis-S`: surface remap. Tape values are permuted/re-signed but latent task
  remains "sum two segments". Procedure should survive because it depends on
  structure, not surface tokens.
- `Axis-C`: composition. Task target is `sum_segment(A) + sum_segment(B)`;
  procedure is called twice as a subprocedure inside a novel composition.

## Agents And Controls

- `procedure_reuse`: calls `sum_segment` for each segment.
- `inline_loop_control`: performs equivalent tape loop inline with no procedure
  memory and higher active-step cost.
- `exact_cache_control`: exact task-id cache. It is cold on held-out tasks and
  therefore should not explain generalization.
- `random_routing_control`: same marginal call count as procedure_reuse, but
  chooses wrong/irrelevant segment routes.

## Metrics

- `task_success`: final answer equals target.
- `state_consistency`: semantic state matches reference.
- `axis_s_success`, `axis_c_success`.
- `active_compute_total`: ticks + tape reads + procedure steps + call overhead.
- `active_compute_drop_vs_inline`: inline total minus procedure total.
- `reuse_count`: procedure calls.
- `cache_hit_rate`: exact cache hits on held-out tasks.
- `random_routing_success`.
- `protocol_evidence.valid`.

## Pass/Fail Thresholds

For `--seeds 11,17,23,29,31,37`, `--episodes >= 96`:

- `protocol_evidence.valid = true`;
- `procedure_reuse.task_success_mean >= 0.95`;
- `procedure_reuse.axis_s_success_mean >= 0.95`;
- `procedure_reuse.axis_c_success_mean >= 0.95`;
- `active_compute_drop_vs_inline_mean >= 4.0`;
- `exact_cache_control.cache_hit_rate_mean = 0.0`;
- `random_routing_control.task_success_mean <= 0.40`;
- `procedure_reuse.reuse_count_mean >= 2.0`.

## Files

- `research/neocortex_x/core_experiment.py`
  - add `--stage stage3`;
  - procedure task generator, agents/controls, aggregation.
- `research/neocortex_x/README.md`
  - document Stage 3 command and artifact.
- `Makefile`
  - add `neocortex-x-stage3-smoke`.

## Artifacts

- `artifacts/neocortex_x/stage3_procedure_reuse.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/core_experiment.py
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
make neocortex-x-stage2-smoke
make neocortex-x-stage3-smoke
```

Additional JSON assertion:

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("artifacts/neocortex_x/stage3_procedure_reuse.json").read_text())
assert d["protocol_evidence"]["valid"] is True
assert d["pass"] is True
assert d["aggregate"]["comparison"]["active_compute_drop_vs_inline"]["mean"] >= 4.0
print("neo cortex stage3 artifact ok")
PY
```

## Out Of Scope

- learned promotion from raw traces;
- pruning noisy procedure libraries;
- planning/lookahead;
- open-world adapter;
- Phi-like diagnostics as evidence.

Эти пункты начинаются после Stage 3 smoke, когда reuse accounting and controls
are already deterministic and reproducible.
