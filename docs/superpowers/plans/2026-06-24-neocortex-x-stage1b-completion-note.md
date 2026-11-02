# NeoCortex-X Stage 1B Completion Note

Дата: 2026-06-24.

Stage 1B trained recall-to-action is now implemented and verified.

## Evidence

- Runner: `research/neocortex_x/stage1b_trained_recall_to_action.py`
- Command: `make neocortex-x-stage1b-trained-smoke`
- Artifact: `artifacts/neocortex_x/stage1b_trained_recall_to_action.json`
- Inner trained benchmark artifact:
  `artifacts/neocortex_x/stage1b_inner_long_context_recall.json`

Passing smoke metrics:

```text
protocol_valid: true
pass: true
ssm_action_accuracy: 1.0000
transformer_action_accuracy: 0.0625
ssm_minus_transformer_action_accuracy: 0.9375
```

## Updated Gap List

The next ordered work is now:

1. Stage 2B: learned tape-world action policy against `single_cell_baseline` and
   dense/control agents.
2. Stage 3B: learned procedure promotion from successful traces, with
   certification and pruning before reuse claims.
3. Stage 4B: learned world-model predictor inside the planning smoke.
4. Stage 5B: live external search adapter with deterministic cached fixtures for
   replay.

The broader NeoCortex-X objective remains active because these stronger learned
and live-backend claims are not completed yet.
