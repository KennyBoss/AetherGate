# NeoCortex-X Completion Audit And Next Gaps

Дата: 2026-06-24.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.

## Аудит Текущего Состояния

Текущий deterministic smoke ladder реализован до Stage 5 включительно:

| Stage | Artifact | Evidence |
|---|---|---|
| Stage 0 Instrumentation | `artifacts/neocortex_x/stage0_smoke.json` | `protocol_evidence.valid = true`, `world_delta_exact_rate = 1.0` |
| Stage 1 Recall-To-Action | `artifacts/neocortex_x/stage1_recall_to_action.json` | `pass = true`, `kv_minus_context_action_accuracy.mean = 0.9375` |
| Stage 2 Stateful Tape World | `artifacts/neocortex_x/stage2_stateful_tape_world.json` | `pass = true`, `codepy_minus_baseline_state_consistency.mean = 1.0` |
| Stage 3 Procedure Reuse | `artifacts/neocortex_x/stage3_procedure_reuse.json` | `pass = true`, `active_compute_drop_vs_inline.mean = 18.0` |
| Stage 4 Planning | `artifacts/neocortex_x/stage4_planning.json` | `pass = true`, `planner_minus_greedy_return.mean = 6.4119` |
| Stage 5 Open-World Adapter | `artifacts/neocortex_x/stage5_open_world_adapter.json` | `pass = true`, `adapter_minus_no_external_accuracy.mean = 1.0` |

Verification command that passed:

```bash
make compile
make neocortex-x-stage0-smoke
make neocortex-x-stage1-smoke
make neocortex-x-stage2-smoke
make neocortex-x-stage3-smoke
make neocortex-x-stage4-smoke
make neocortex-x-stage5-smoke
```

Additional JSON assertion passed for all Stage 0-5 artifacts.

## Что Доказано

- The tick-level trace/replay contract works for Stage 0.
- Delayed recall can be converted into environment action in the harness.
- A bounded CodePy-like tape/register/pointer/pc world exists and has baseline
  comparison.
- Procedure reuse accounting has Axis-S/Axis-C tasks and exact-cache/random-
  routing/inline controls.
- Short lookahead over predicted state deltas beats a greedy action baseline in
  the deterministic planning smoke.
- External observations can update factual state with provenance and replay
  consistency in a deterministic adapter.

## Что Еще Не Доказано

These are explicit stronger claims from the design spec that are not completed
by deterministic hand-coded smoke tests:

1. Replace Stage 1 hand-coded `kv_memory` with a trained SSM/KV policy and compare
   against a matched dense baseline on held-out recall-to-action tasks.
2. Replace Stage 2 `codepy_loop` oracle with a learned action policy in the
   tape world and measure trajectory horizon against baselines.
3. Replace Stage 3 hand-coded `sum_segment` procedure with learned trace
   promotion, certification, pruning, Axis-S and Axis-C reuse.
4. Replace Stage 4 exact deterministic predictor with a learned world-model
   predictor and measure the compute/quality frontier.
5. Add a live-search backend behind Stage 5 only after deterministic provenance
   handling remains replayable.
6. Integrate these stages into the existing stronger benchmark style:
   repeated seeds, saved protocol evidence, controls, negative-result logging.

## Next Ordered Work

1. Stage 1B: trained SSM/KV recall-to-action policy, keeping the Stage 1 task and
   artifact schema unchanged.
2. Stage 2B: learned tape-world action policy against `single_cell_baseline` and
   dense/control agents.
3. Stage 3B: procedure promotion from successful traces, with certification and
   pruning before reuse claims.
4. Stage 4B: learned world-model predictor inside the planning smoke.
5. Stage 5B: live external search adapter with deterministic cached fixtures for
   replay.

## Completion Verdict

The deterministic Stage 0-5 harness ladder is complete and verified.

The full NeoCortex-X research objective is not complete yet, because the design
spec asks for learned SSM/KV, learned procedure reuse, learned world-model
planning and eventually live open-world adapter evidence. The next work should
continue with Stage 1B rather than inventing new architecture layers.
