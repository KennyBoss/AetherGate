# NeoCortex-X Stage 1B Trained Recall-To-Action Plan

Дата: 2026-06-24.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.
Audit source: `docs/superpowers/plans/2026-06-24-neocortex-x-completion-audit-and-next-gaps.md`.

## Цель

Replace the Stage 1 hand-coded `kv_memory` smoke with a trained SSM/KV evidence
artifact while keeping the recall-to-action semantics:

```text
bind(key_i, value_i)
filler * delay
query(key_j)
act(value_j)
```

The existing `compare_long_context_recall.py` already trains SSM/KV against a
context-capped Transformer on `multi-assignment`. Stage 1B wraps that trained
benchmark and maps held-out `recall_accuracy` to `action_accuracy`: predicting
the correct value token is the policy choosing `act(value_j)`.

## Files

- `research/neocortex_x/stage1b_trained_recall_to_action.py`
  - calls `compare_long_context_recall.py` as a subprocess;
  - reads the trained benchmark JSON;
  - emits NeoCortex-X action-level artifact.
- `Makefile`
  - add `neocortex-x-stage1b-trained-smoke`.
- `research/neocortex_x/README.md`
  - document command and artifact.

## Artifact

- `artifacts/neocortex_x/stage1b_trained_recall_to_action.json`
- inner benchmark:
  `artifacts/neocortex_x/stage1b_inner_long_context_recall.json`

## Pass/Fail

Smoke threshold is deliberately modest because this is a fast training gate:

- `protocol_evidence.valid = true`;
- `ssm_action_accuracy >= 0.70`;
- `ssm_minus_transformer_action_accuracy >= 0.30`;
- `delay > transformer_context`;
- inner artifact exists and records trained SSM/KV protocol.

The full research version should later raise this to repeated seeds and stronger
accuracy thresholds.

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage1b_trained_recall_to_action.py
make neocortex-x-stage1b-trained-smoke
```
