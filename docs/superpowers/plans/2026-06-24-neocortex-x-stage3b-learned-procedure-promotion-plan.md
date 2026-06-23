# NeoCortex-X Stage 3B Learned Procedure Promotion Plan

Дата: 2026-06-24.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.

## Цель

Replace the Stage 3 hand-coded `sum_segment` procedure with a promoted procedure
learned from successful traces.

Stage 3 already validates procedure-reuse accounting and controls. Stage 3B
must add a promotion step:

1. collect successful inline-loop traces;
2. detect the repeated read/add/move segment pattern;
3. promote a bounded `sum_segment` candidate;
4. certify it on held-out tasks;
5. reuse it on Axis-S and Axis-C tasks with controls.

## Promotion Rule

The promoter scans traces for repeated blocks matching:

```text
read tape[start + offset]
accumulate into result
advance pointer
```

If the block appears at least `min_support` times with consistent typed
signature, promote:

```text
sum_segment(start:int, length:int) -> int
```

The promoted procedure is accepted only after held-out certification.

## Agents And Controls

- `promoted_procedure`: uses the learned/certified procedure.
- `inline_loop_control`: no procedure memory.
- `exact_cache_control`: task-id cache, cold on held-out.
- `random_routing_control`: same call count, wrong segment route.

## Pass/Fail

- `protocol_evidence.valid = true`;
- `promotion_evidence.promoted = true`;
- `promotion_evidence.certified = true`;
- `promoted_procedure.task_success_mean >= 0.95`;
- `promoted_procedure.axis_s_success_mean >= 0.95`;
- `promoted_procedure.axis_c_success_mean >= 0.95`;
- `active_compute_drop_vs_inline_mean >= 4.0`;
- `exact_cache_control.cache_hit_rate_mean = 0.0`;
- `random_routing_control.task_success_mean <= 0.40`.

## Files

- `research/neocortex_x/stage3b_learned_procedure_promotion.py`
- `Makefile`
  - add `neocortex-x-stage3b-promotion-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage3b_learned_procedure_promotion.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage3b_learned_procedure_promotion.py
make neocortex-x-stage3b-promotion-smoke
```
