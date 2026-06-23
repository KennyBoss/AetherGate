# NeoCortex-X Stage 12 Policy-Guided Replay Plan

Дата: 2026-06-24.
Builds on:
- `docs/superpowers/plans/2026-06-24-neocortex-x-stage11-policy-update-plan.md`
- `research/neocortex_x/stage11_policy_update.py`

## Цель

Stage 12 применяет обновленный policy snapshot в следующем decision cycle:

```text
policy snapshot + temporal state -> selected action -> replay score
```

Это первый replay после feedback-driven update. Он проверяет не только наличие
новых weights, но и способность использовать их при следующем проходе.

## Agents

- `policy_guided_gate`: uses temporal `future_action` as state constraint and
  Stage 11 probabilities as action scores.
- `flat_prior_control`: uses the same state constraints with uniform weights.
- `state_blind_control`: ignores temporal state and chooses the globally highest
  policy action.

## Pass/Fail

- `protocol_evidence.valid = true`;
- Stage 8 and Stage 11 artifacts valid and non-empty;
- policy-guided accuracy >= `0.95`;
- policy-guided beats state-blind by >= `0.25`;
- deterministic replay = `true`;
- selected actions cover all three executive actions.

## Files

- `research/neocortex_x/stage12_policy_replay.py`
- `Makefile`
  - add `neocortex-x-stage12-policy-replay-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage12_policy_replay.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage12_policy_replay.py
make neocortex-x-stage12-policy-replay-smoke
```
