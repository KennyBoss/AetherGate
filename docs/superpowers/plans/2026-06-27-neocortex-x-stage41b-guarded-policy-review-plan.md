# NeoCortex-X Stage 41B Guarded Policy Review Plan

Дата: 2026-06-27.
Builds on:
- `docs/superpowers/plans/2026-06-27-neocortex-x-stage41a-adversarial-risk-audit-plan.md`
- `research/neocortex_x/stage40_clarification_monitor_feedback.py`
- `research/neocortex_x/stage41a_adversarial_risk_audit.py`

## Цель

Stage 41B делает guarded review Stage 40 feedback после Stage 41A risk audit:

```text
Stage 40 rewards + Stage 41A blockers
  -> quarantined candidate policy
  -> Stage 42 replay only
```

Это не runtime promotion. Candidate policy можно передать в Stage 42 replay,
но нельзя делать active runtime policy, пока clean+adversarial replay не
докажет отсутствие regressions и не появится rollback metadata.

## Review Contract

Stage 41B должен:

- загрузить Stage 40 feedback ledger;
- загрузить Stage 41A adversarial risk audit;
- пересчитать candidate key weights из Stage 40 `key_rewards`;
- создать `candidate_policy.review_state = quarantined_until_stage42_replay`;
- оставить `runtime_promotion_allowed = false`;
- сохранить список promotion requirements для Stage 42/43;
- требовать, что Stage 41A действительно blocked promotion;
- быть deterministic replayable.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 40 artifact valid and passed;
- Stage 41A artifact valid and passed;
- candidate policy present;
- candidate policy quarantined;
- reinforced key gains positive;
- non-reinforced keys unchanged;
- probabilities sum to `1.0`;
- runtime promotion blocked;
- candidate allowed only for Stage 42 replay;
- blocked by Stage 41A;
- rollback required;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage41b_guarded_policy_review.py`
- `Makefile`
  - add `neocortex-x-stage41b-guarded-policy-review-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage41b_guarded_policy_review.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage41b_guarded_policy_review.py
make neocortex-x-stage41b-guarded-policy-review-smoke
make compile
```

## Guardrail

Stage 41B creates a quarantined candidate policy for replay only. It is not
policy promotion, not runtime activation, not live-world robustness, and not a
production memory-scale claim.
