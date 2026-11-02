# ECS Stage 3: Observable State-Porting Falsification

Date: 2026-07-13

## Goal

Test the pre-registered claim in `research/MODEL_THESIS.md`: local fast-weight
skills fail on novel composition because no state is carried from one skill
application into the next.

This experiment tests one narrow instantiation. A positive result earns an
external benchmark. A negative result rejects this additive recurrent
state-porting instantiation, not every possible composition mechanism.

## Non-goals

- No integration with `inversion-driven-growth`.
- No larger language model or new dataset.
- No composition-aware selector: Stage 2A already excluded selector blindness.
- No oracle plan, true skill boundary, remaining depth, or hidden subroutine ID
  as model input.
- No Stage 4 work unless this experiment passes.

## Reuse

Reuse task generation, primitive actions, policy training, best-first search,
seed set, and accounting from:

- `research/skill_compression/ecs_stage2_fastweight.py`
- `research/skill_compression/ecs_stage2a_composition.py`

The implementation lives in one new experiment file:
`research/skill_compression/ecs_stage3_state_porting.py`. The stale CodePy stub
`research/codepy/ecs_stage3_composition.py` describes a different experiment
and is not repurposed.

## Mechanism

Each search node carries a bounded recurrent context `c`. The context is
updated only from information available on the explored path:

```text
c_next = tanh(Wc [c, h(state, target), one_hot(chosen_primitive)] + bc)
logits = base_logits(h) + Delta_k + U_k c
```

The existing learned gate still chooses `k`; its inputs and decision rule do
not change. Context is stored with the search node, so separate paths do not
share state. Search dominance is keyed by `(environment_state,
float32_context_bytes)`, not by environment state alone; distinct recurrent
contexts therefore remain distinct nodes and their expansion cost is fully
counted. The action set remains the same five primitives.

Training examples are ordered primitive trajectories rather than shuffled
independent states. Training compositions have depth at most 2. Evaluation
uses fresh depth-1, depth-2, and depth-3 tasks; depth 3 is the held-out Axis-C
condition.

## Arms

1. `ported`: recurrent context is carried along each explored path.
2. `reset`: the same trained weights are used, but `c` is reset before every
   decision. This is the parameter-matched state-porting ablation.
3. `dense_context`: same recurrent context width and per-node operations, but
   no skill-conditioned `Delta_k` or `U_k`; trained on the same trajectories.
4. `local`: the existing Stage-2A local fast-weight arm, retained as the
   historical control.

The output records parameter count and estimated multiply-adds per forward for
every arm. `ported`, `reset`, and `dense_context` must have matched forward cost;
otherwise the protocol is invalid rather than a pass.

## Data and leakage controls

- Fixed seeds: `11,17,23,41,73,101`.
- Identical evaluation tasks for every arm within a seed.
- Training/evaluation `(start, target, plan)` triples must be disjoint.
- The model receives only current observable state, target, its own recurrent
  context, and its own selected primitive.
- True plan, composition depth, true subroutine ID, and true boundaries may
  appear only in artifact diagnostics, never in policy input or context update.
- Solve-rate and compute are reported per task, per seed, and in aggregate.

## Primary metrics

- solve rate;
- mean best-first forward passes;
- seed-wise wins at depth 3;
- parameter count and estimated multiply-adds per forward;
- protocol validity.

Wall-clock time is diagnostic only.

## Pre-registered verdict

`PASS` requires all of the following at depth 3:

- `ported` solve rate is within 1 percentage point of `dense_context` and at
  least 0.98;
- `ported` uses at least 10% fewer mean forward passes than both
  `dense_context` and `reset`;
- `ported` beats each of those arms on at least 5 of 6 seeds;
- matched-forward-cost and leakage checks pass.

Otherwise the result is `FAIL`. No threshold is changed after observing seed
results. A protocol violation produces `INVALID`, not `FAIL` or `PASS`.

## Artifact and failure handling

Write `artifacts/ecs_stage3_state_porting.json` with:

```text
config
protocol_evidence
parameter_accounting
aggregate_by_depth
seed_runs
verdict
```

Numerical failure, non-finite loss, empty evaluation split, seed count below 6,
task overlap, hidden-plan input, or unmatched forward cost marks the artifact
`INVALID` and exits non-zero. Ordinary inability to solve remains a valid
negative result, is written as `FAIL`, and exits zero.

## Minimal verification

Add one runnable check that uses two seeds and small task counts to assert:

- deterministic repeated output;
- identical task IDs across arms;
- context changes in `ported` and remains zero in `reset`;
- no oracle-only field enters a policy feature vector;
- JSON contains a verdict and protocol evidence.

The two-seed JSON is expected to be `INVALID` because it does not meet the
research seed count. The full six-seed run is the research result; the small
check is only a wiring regression.
