# NeoCortex-X Completion Audit

Дата: 2026-06-24.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.

## Completion Verdict

The NeoCortex-X Core Experiment scope is complete and verified.

This means the controlled research ladder from the design spec now has
replayable artifacts for Stage 0 through Stage 5, plus the learned/cached
follow-up ladder Stage 1B through Stage 5B. The verified claim is intentionally
narrow: a controlled TimeAxis + SSM/KV + CodePy-like world-model harness can
connect memory, action, procedure reuse, planning and replayable provenance
under explicit baselines and pass/fail checks.

This is not evidence for AGI, consciousness, IIT Phi thresholds, 86B-scale
biological realism, quantum advantage or neuromorphic energy claims. Those
remain roadmap ideas outside this completed core experiment.

## Verified Artifacts

| Stage | Artifact | Evidence |
|---|---|---|
| Stage 0 Instrumentation | `artifacts/neocortex_x/stage0_smoke.json` | `protocol_evidence.valid = true`, `world_delta_exact_rate = 1.0` |
| Stage 1 Recall-To-Action | `artifacts/neocortex_x/stage1_recall_to_action.json` | `pass = true`, `kv_minus_context_action_accuracy.mean = 0.9375` |
| Stage 1B Trained Recall-To-Action | `artifacts/neocortex_x/stage1b_trained_recall_to_action.json` | `pass = true`, `ssm_minus_transformer_action_accuracy = 0.9375` |
| Stage 2 Stateful Tape World | `artifacts/neocortex_x/stage2_stateful_tape_world.json` | `pass = true`, `codepy_minus_baseline_state_consistency.mean = 1.0` |
| Stage 2B Learned Tape Policy | `artifacts/neocortex_x/stage2b_learned_tape_policy.json` | `pass = true`, `learned.task_success = 1.0` |
| Stage 3 Procedure Reuse | `artifacts/neocortex_x/stage3_procedure_reuse.json` | `pass = true`, `active_compute_drop_vs_inline.mean = 18.0` |
| Stage 3B Learned Procedure Promotion | `artifacts/neocortex_x/stage3b_learned_procedure_promotion.json` | `pass = true`, `promoted = true`, `certified = true` |
| Stage 4 Planning | `artifacts/neocortex_x/stage4_planning.json` | `pass = true`, `planner_minus_greedy_success.mean = 1.0` |
| Stage 4B Learned World Model | `artifacts/neocortex_x/stage4b_learned_world_model.json` | `pass = true`, `prediction_exact_rate = 1.0`, `learned_to_exact_return_ratio = 1.0` |
| Stage 5 Open-World Adapter | `artifacts/neocortex_x/stage5_open_world_adapter.json` | `pass = true`, `adapter_minus_no_external_accuracy.mean = 1.0` |
| Stage 5B Cached Live-Shaped Adapter | `artifacts/neocortex_x/stage5b_live_cached_adapter.json` | `pass = true`, `protocol_evidence.valid = true`, `adapter_minus_no_external_accuracy = 1.0` |

## Verification Commands

The full suite passed in one run:

```bash
make compile
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
make neocortex-x-stage1b-trained-smoke
make neocortex-x-stage2-smoke
make neocortex-x-stage2b-learned-smoke
make neocortex-x-stage3-smoke
make neocortex-x-stage3b-promotion-smoke
make neocortex-x-stage4-smoke
make neocortex-x-stage4b-world-model-smoke
make neocortex-x-stage5-smoke
make neocortex-x-stage5b-cached-smoke
```

Additional JSON assertions passed for all listed artifacts:

```text
stage0: ok
stage1: ok
stage1b: ok
stage2: ok
stage2b: ok
stage3: ok
stage3b: ok
stage4: ok
stage4b: ok
stage5: ok
stage5b: ok
all neocortex_x completion artifacts ok
```

## What Is Demonstrated

- Tick-level trace/replay instrumentation works for the controlled environment.
- Delayed associative recall can be converted into an action, including a
  trained SSM/KV wrapper beating a context-capped Transformer baseline.
- The bounded CodePy-like tape/register/pointer/pc world supports stateful
  action evaluation against baselines.
- A learned trace policy solves held-out tape-world tasks and beats baseline and
  dense-majority controls.
- Procedure reuse reduces active compute on held-out Axis-S and Axis-C tasks,
  with exact-cache and random-routing controls.
- Procedure promotion from successful traces can be certified before reuse.
- Planning over predicted state deltas beats greedy action selection.
- A learned transition world model supports lookahead planning and matches the
  exact planner on this deterministic corridor world.
- Deterministic external observations can update factual state with provenance,
  contradiction handling and replay consistency.
- Cached live-shaped fixtures preserve the Stage 5 provenance contract before a
  future live-search backend is allowed to provide observations.

## Remaining Roadmap, Not Current Scope

The following are deliberately not claimed by this completion audit:

1. Live internet/search ingestion with real sources and persistent caches.
2. Broader stochastic or adversarial worlds beyond the current deterministic
   harnesses.
3. Scaling curves over model size, active compute and dataset complexity.
4. Phi-like diagnostics as anything stronger than secondary probes.
5. Spiking, neuromorphic or quantum-classical backends.
6. Biological-scale neuron counts, energy claims, AGI or consciousness claims.

Recommended next work is a new, separately scoped spec for live search or
scaling curves, not more additions to this completed core experiment.
