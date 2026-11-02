# NeoCortex-X Stage 21 Contextual Policy Promotion Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage19-contextual-policy-update-plan.md`
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage20-contextual-policy-replay-plan.md`
- `research/neocortex_x/stage19_contextual_policy_update.py`
- `research/neocortex_x/stage20_contextual_policy_replay.py`

## Цель

Stage 21 продвигает replay-validated contextual policy snapshot в runtime
candidate manifest:

```text
contextual replay + contextual policy -> promoted contextual runtime manifest
```

Это контекстный аналог Stage 13. Manifest должен сохранить bucketed policy:

```text
context_decision -> action -> probability
```

и rollback metadata, чтобы future runtime loader мог безопасно откатиться.

## Promotion Contract

Manifest содержит:

- stable `manifest_id`;
- `policy_id`;
- contextual `runtime_probabilities`;
- expected top actions per context decision;
- replay evidence;
- promotion thresholds;
- rollback metadata.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 20 replay artifact valid and passed;
- Stage 19 policy artifact valid and passed;
- policy ids match;
- guided action accuracy >= `0.95`;
- guided replay beats global prior by >= `0.25`;
- selected actions cover all contextual gate actions;
- no false evidence request for contextual alias;
- rollback metadata present.

## Files

- `research/neocortex_x/stage21_contextual_policy_promotion.py`
- `Makefile`
  - add `neocortex-x-stage21-contextual-policy-promotion-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage21_contextual_policy_promotion.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage21_contextual_policy_promotion.py
make neocortex-x-stage21-contextual-policy-promotion-smoke
```
