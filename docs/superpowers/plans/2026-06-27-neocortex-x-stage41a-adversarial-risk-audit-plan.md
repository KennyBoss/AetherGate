# NeoCortex-X Stage 41A Adversarial Risk Audit Plan

Дата: 2026-06-27.
Builds on:
- `docs/superpowers/plans/2026-06-27-neocortex-x-risk-register-and-hard-gates.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage40-clarification-monitor-feedback-plan.md`
- `research/neocortex_x/stage30_partial_context_scale_retrieval.py`
- `research/neocortex_x/stage37_clarification_runtime_loader.py`
- `research/neocortex_x/stage40_clarification_monitor_feedback.py`

## Цель

Stage 41A реализует первый risk-first gate перед любым Stage 41 policy update:

```text
Stage 40 feedback + runtime policy + partial-context queries
  -> adversarial cases
  -> risk audit
  -> block promotion until guarded replay
```

Pass для Stage 41A означает не "policy можно промотить", а наоборот:
adversarial risks измерены, critical cases найдены, и небезопасный policy
update заблокирован до guarded review/replay.

## Audit Contract

Stage 41A должен:

- загрузить Stage 30 evaluated partial-context queries;
- загрузить Stage 37 runtime clarification policy;
- загрузить Stage 40 monitor feedback ledger;
- создать adversarial variants:
  - `missing_context`;
  - `noisy_context`;
  - `correlated_keys`;
  - `rare_context`;
  - `same_context_contradiction`;
  - `temporal_drift`;
- выполнить bounded clarification loop на этих variants;
- проверить target preservation для safe-required cases;
- обнаружить risky cases, где promotion должен быть blocked;
- сохранить `promotion_decision.promotion_allowed = false`;
- требовать rollback metadata и deterministic replay.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 30 artifact valid and passed;
- Stage 37 runtime artifact valid and passed;
- Stage 40 feedback artifact valid and passed;
- adversarial cases are present;
- all required risk types are present;
- safe-required cases preserve target;
- risky cases are detected;
- critical risks block promotion;
- promotion is blocked;
- rollback metadata present;
- Stage 40 replay valid;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage41a_adversarial_risk_audit.py`
- `Makefile`
  - add `neocortex-x-stage41a-adversarial-risk-audit-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage41a_adversarial_risk_audit.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage41a_adversarial_risk_audit.py
make neocortex-x-stage41a-adversarial-risk-audit-smoke
make compile
```

## Guardrail

Stage 41A is an adversarial audit gate. It is not policy training, not policy
promotion, not live-world robustness, and not a production memory-scale claim.
