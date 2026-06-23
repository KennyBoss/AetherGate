# NeoCortex-X Stage 5 Open-World Adapter Plan

Статус: continuation plan после Stage 4 planning smoke.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.
Дата: 2026-06-24.

## Цель

Реализовать последний rung текущего ladder:

> Stage 5: добавить internet/search observations с provenance и contradiction
> handling. Exit: external observations улучшают factual state updates, не
> ломая controlled replay или provenance.

Smoke использует deterministic synthetic external observations вместо живого
интернета. Это сохраняет replay и проверяет adapter contract. Реальная сеть
может быть отдельным backend после прохождения deterministic gate.

## Adapter Contract

External observation:

```text
claim_id
subject
value
source_id
source_trust
observed_at
provenance_url
```

State update:

- группировать observations по subject;
- выбирать value по суммарному trust;
- если competing value имеет близкий trust, пометить contradiction;
- сохранять provenance для выбранного value и rejected candidates;
- replay должен exact-match reconstructed factual state.

## Agents And Controls

- `provenance_adapter`: trust-weighted update with contradiction handling.
- `no_external_control`: игнорирует external observations.
- `naive_latest_control`: берет последний observation без trust/provenance
  reasoning.

## Metrics

- `factual_accuracy`: selected value equals ground truth.
- `contradiction_precision`: contradictions flagged only when competing evidence
  is close enough.
- `provenance_coverage`: every selected value has source/provenance.
- `replay_consistency`: deterministic reconstruction from observations.
- `adapter_minus_no_external_accuracy`.
- `adapter_minus_naive_accuracy`.

## Pass/Fail Thresholds

For `--seeds 11,17,23,29,31,37`, `--episodes >= 96`:

- `protocol_evidence.valid = true`;
- `provenance_adapter.factual_accuracy_mean >= 0.95`;
- `provenance_adapter.provenance_coverage_mean = 1.0`;
- `provenance_adapter.replay_consistency_mean = 1.0`;
- `adapter_minus_no_external_accuracy_mean >= 0.50`;
- `adapter_minus_naive_accuracy_mean >= 0.25`;
- `naive_latest_control.factual_accuracy_mean <= 0.75`.

## Files

- `research/neocortex_x/core_experiment.py`
  - add `--stage stage5`;
  - synthetic observation generator, adapter/control agents, replay.
- `research/neocortex_x/README.md`
  - document Stage 5 command and artifact.
- `Makefile`
  - add `neocortex-x-stage5-smoke`.

## Artifacts

- `artifacts/neocortex_x/stage5_open_world_adapter.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/core_experiment.py
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
make neocortex-x-stage2-smoke
make neocortex-x-stage3-smoke
make neocortex-x-stage4-smoke
make neocortex-x-stage5-smoke
```

Additional JSON assertion:

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("artifacts/neocortex_x/stage5_open_world_adapter.json").read_text())
assert d["protocol_evidence"]["valid"] is True
assert d["pass"] is True
assert d["aggregate"]["comparison"]["adapter_minus_no_external_accuracy"]["mean"] >= 0.5
print("neo cortex stage5 artifact ok")
PY
```

## Out Of Scope

- live web search;
- search-result ranking model;
- natural-language extraction;
- Phi-like diagnostics as evidence.

The live internet backend can be attached after deterministic provenance and
contradiction handling are reproducible.
