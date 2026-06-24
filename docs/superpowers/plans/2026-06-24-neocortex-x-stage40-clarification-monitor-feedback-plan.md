# NeoCortex-X Stage 40 Clarification Monitor Feedback Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage39-clarification-runtime-self-monitor-plan.md`
- `research/neocortex_x/stage39_clarification_runtime_self_monitor.py`

## Цель

Stage 40 превращает clarification runtime monitor records в deterministic
feedback ledger:

```text
monitor records -> feedback events -> key rewards + alert summary
```

Это следующий шаг после self-monitor. Stage 39 доказывает, что runtime trace
восстанавливается; Stage 40 делает это пригодным для будущего policy review:
каждый monitor record получает feedback по runtime path, output consistency,
active compute и selected clarification keys.

## Feedback Contract

Stage 40 должен создать:

- `runtime_path_feedback` для каждого monitor record;
- `output_consistency_feedback` для каждого monitor record;
- `active_compute_feedback` для каждого monitor record;
- `selected_key_feedback` для records с clarification steps;
- `clarification_efficiency_diagnostic` как info alert, если не каждый step
  уменьшил candidates, но весь trace успешно дошел до unique answer;
- `key_rewards` по reinforced clarification keys;
- `runtime_path_rewards` по reinforced runtime paths;
- `alert_summary` с critical/warning/info counts.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 39 monitor artifact valid and passed;
- feedback events are present;
- runtime path, output and compute feedback counts match monitor record count;
- selected-key feedback is present for clarification traces;
- efficiency diagnostic count matches monitor records that need info diagnostics;
- success rate >= `0.95`;
- mean reward >= `0.75`;
- no runtime review needed;
- no target loss penalties;
- no critical alerts;
- key rewards and runtime path rewards are present;
- replay hashes are present;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage40_clarification_monitor_feedback.py`
- `Makefile`
  - add `neocortex-x-stage40-clarification-monitor-feedback-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage40_clarification_monitor_feedback.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage40_clarification_monitor_feedback.py
make neocortex-x-stage40-clarification-monitor-feedback-smoke
make compile
```

## Guardrail

Stage 40 is a deterministic feedback ledger over cached runtime monitor
records. It is not RL training, live deployment, or a production memory-scale
claim.
