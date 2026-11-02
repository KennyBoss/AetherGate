# NeoCortex-X Stage 2B Learned Tape Policy Plan

Дата: 2026-06-24.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.
Previous gap note: `docs/superpowers/plans/2026-06-24-neocortex-x-stage1b-completion-note.md`.

## Цель

Replace the Stage 2 `codepy_loop` oracle with a learned tape-world action policy.

Stage 2 already proves the deterministic tape/register/pointer/pc world and
baseline comparison. Stage 2B must show that a policy learned from traces can
execute the same stateful world on held-out tasks and beat simple controls.

## Approach

Use imitation learning from oracle traces:

1. Generate training tasks with the Stage 2 tape world.
2. Execute `codepy_loop` to collect `(state_features -> action)` examples.
3. Train a compact table policy keyed by semantic state features:
   `pc`, `ptr_relation_to_input_len`, and `halted`.
4. Evaluate on held-out seeds and input lengths.

This is deliberately small and deterministic. It is a learned policy because
the action table is induced from traces rather than hand-coded in the evaluator.

## Agents And Controls

- `learned_trace_policy`: table policy learned from oracle traces.
- `single_cell_baseline`: existing Stage 2 baseline.
- `dense_majority_control`: always emits the most frequent training action.

## Metrics

- `action_accuracy`: learned action vs oracle action on held-out states.
- `state_consistency`: final semantic state equals oracle final state.
- `task_success`: final `r0 == target_sum` and halted.
- `trajectory_horizon`: ticks until semantic divergence from oracle.
- `learned_minus_baseline_state_consistency`.
- `learned_minus_dense_state_consistency`.

## Pass/Fail

For smoke:

- `protocol_evidence.valid = true`;
- `learned_trace_policy.action_accuracy_mean >= 0.95`;
- `learned_trace_policy.state_consistency_mean >= 0.95`;
- `learned_trace_policy.task_success_mean >= 0.95`;
- `learned_minus_baseline_state_consistency_mean >= 0.50`;
- `learned_minus_dense_state_consistency_mean >= 0.50`;
- `heldout_seed_count >= 3`;
- `train_example_count > 0`.

## Files

- `research/neocortex_x/stage2b_learned_tape_policy.py`
- `Makefile`
  - add `neocortex-x-stage2b-learned-smoke`
- `research/neocortex_x/README.md`

## Artifacts

- `artifacts/neocortex_x/stage2b_learned_tape_policy.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage2b_learned_tape_policy.py
make neocortex-x-stage2b-learned-smoke
```
