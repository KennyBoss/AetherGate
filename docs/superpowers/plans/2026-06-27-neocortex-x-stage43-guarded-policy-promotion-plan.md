# NeoCortex-X Stage 43 Guarded Policy Promotion Plan

Дата: 2026-06-27.
Builds on:
- `docs/superpowers/plans/2026-06-27-neocortex-x-stage42-clean-adversarial-replay-plan.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-stage42c-evidence-conflict-guard-plan.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-risk-register-and-hard-gates.md`
- `research/neocortex_x/stage42_clean_adversarial_replay.py`
- `research/neocortex_x/stage42c_evidence_conflict_guard.py`

## Цель

Stage 43 делает promotion только для guarded cached-replay runtime candidate:

```text
Stage 41B candidate policy
  + Stage 42 clean/adversarial replay
  + Stage 42C evidence/conflict guard
  + rollback metadata
  -> guarded promotion manifest
```

Это не production deployment и не live-source trust. Stage 43 лишь создает
manifest, который Stage 44 runtime loader сможет загрузить и проверить.

## Promotion Contract

Stage 43 должен:

- загрузить Stage 41B guarded review;
- загрузить Stage 42 clean + adversarial replay;
- загрузить Stage 42C evidence/conflict guard;
- проверить, что Stage 42 blocked unguarded promotion;
- проверить, что Stage 42C убрал critical unsafe answers;
- проверить, что candidate ids и ranked keys совпадают;
- создать guarded promotion manifest;
- включить runtime contract с guard rules;
- включить rollback metadata;
- ограничить scope до `guarded_cached_replay_runtime_candidate`;
- явно запретить claims про live search memory, production scale memory, AGI,
  consciousness и neuromorphic energy.

## Pass/Fail

- Stage 42 artifact valid and deterministic;
- Stage 42C artifact valid and deterministic;
- Stage 41B artifact valid;
- candidate IDs match;
- ranked keys match;
- Stage 42 blocked unguarded promotion;
- Stage 42C allowed Stage 43 review;
- critical unsafe answers after guard = `0`;
- clean answers preserved;
- adversarial safe after guard = `1.0`;
- guard rules present;
- rollback metadata present;
- promotion scope limited;
- claim guardrails present;
- deterministic replay = `true`.

## Files

- `research/neocortex_x/stage43_guarded_policy_promotion.py`
- `Makefile`
  - add `neocortex-x-stage43-guarded-policy-promotion-smoke`
- `research/neocortex_x/README.md`
- `docs/superpowers/plans/2026-06-27-neocortex-x-risk-register-and-hard-gates.md`
- `🧠 NEOCORTEX-X.md`

## Artifact

- `artifacts/neocortex_x/stage43_guarded_policy_promotion.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage43_guarded_policy_promotion.py
make neocortex-x-stage43-guarded-policy-promotion-smoke
make compile
```

## Guardrail

Stage 43 promotes only a guarded cached-replay runtime candidate with rollback
metadata. It is not live-source trust, not production memory, not AGI, and not
consciousness evidence.
