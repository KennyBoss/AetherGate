# KV Scale Law 1 Implementation Plan

**Date:** 2026-07-13

**Design:** `/Users/makbuk/inversion-driven-growth/docs/superpowers/specs/2026-07-13-dual-track-90-day-research-roadmap-design.md`

## Goal

Build one reproducible runner that tests whether the existing KV-memory SSM keeps recall plus resource advantages against a full-context, parameter-matched Transformer as task difficulty and capacity increase.

This gate does not claim ordinary language-model superiority. Additive recurrent state-porting remains closed.

## Reuse

Do not add another trainer or model implementation.

- `compare_long_context_recall.py` remains the single-seed trainer/evaluator.
- `run_long_context_recall_multiseed.py` remains the repeated-seed runner.
- `compare_text_architectures.py` remains the separate ordinary-LM diagnostic.
- The new file only registers points, launches those programs, validates their JSON, and aggregates the gate.

## Registered Axes

### Phase 1: task difficulty first

Hold `ssm_input_dim=64`, `ssm_state_dim=96`, optimizer, records, and seeds fixed. Run canonical marker-free MQAR at:

| Point | Vocabulary | Bindings | Queries | Delay |
|---|---:|---:|---:|---:|
| `task-0` | 16 | 4 | 2 | 64 |
| `task-1` | 32 | 6 | 3 | 96 |
| `task-2` | 64 | 8 | 4 | 128 |
| `task-3` | 128 | 12 | 6 | 192 |

Vocabulary-dependent embeddings may change the exact parameter count; the dimensions are intentionally fixed so this phase does not deliberately increase capacity.

Capacity promotion is closed until at least three adjacent Phase-1 points are valid and Pareto-positive.

### Phase 2: capacity

Freeze MQAR at `vocabulary=128`, `bindings=12`, `queries=6`, `delay=192`. Use these approximate KV-memory SSM parameter tiers:

| Point | Target | Input dim | State dim | Batch size |
|---|---:|---:|---:|---:|
| `cap-10k` | 10k | 32 | 36 | 128 |
| `cap-100k` | 100k | 128 | 580 | 32 |
| `cap-1m` | 1M | 256 | 3,000 | 4 |
| `cap-10m` | 10M | 512 | 15,400 | 1 |

The runner calculates the exact initialized parameter count from produced artifacts and rejects a point outside `0.8x..1.25x` of its target. Large tiers are registered but are not run locally until the preceding point passes and the explicit resource preflight fits the selected machine.

## Controls

Every promoted point requires:

1. full-context Transformer with context equal to the complete MQAR input length;
2. context-capped Transformer at 32 tokens as a diagnostic only;
3. at least three fixed seeds: `11,17,23`;
4. parameter ratio between Transformer and KV-memory SSM in `0.8x..1.25x`;
5. identical generated data, epochs, optimizer, batch size, and seed within each comparison;
6. a separate ordinary-LM run for capacity points.

An official or validated Mamba implementation is not installed in this JAX/CPU environment. The artifact records that baseline as missing. No local selective-SSM variant may be mislabeled Mamba. A final A1 `PASS` is impossible until an equivalent-budget Mamba baseline runs or a later registered amendment explicitly removes it.

## Metrics

Read measured values from the existing artifacts:

- recall accuracy and loss;
- all-token loss;
- training seconds;
- eval records/second;
- parameter count and bytes.

Add transparent estimates per sequence:

- KV-memory SSM runtime state: `4 * (state_dim + input_dim + input_dim²)` bytes;
- autoregressive Transformer KV cache: `4 * 2 * layers * context * d_model` bytes.

These are estimates, labeled as such, and are not mixed with measured peak process memory.

## Registered Gate

A point is `Pareto-positive` only when, against the full-context Transformer:

- mean KV-memory recall is no more than `0.01` below mean Transformer recall; and
- KV-memory has either at least `10%` higher measured eval throughput or at least `10%` lower estimated runtime state bytes.

`PASS` requires three adjacent valid Pareto-positive task points, three adjacent valid Pareto-positive capacity points, all required ordinary-LM reports, and the required Mamba baseline. `FAIL` occurs when two successive valid promoted points are not Pareto-positive. Anything missing or mismatched is `INCOMPLETE` or `INVALID`, never silently promoted.

## Files

1. Add `run_kv_scale_law.py` with the registry, command construction, validation, aggregation, dry-run mode, and smoke mode.
2. Add `test_kv_scale_law.py` for pure registry/gate/resource checks.
3. Save smoke output under `artifacts/kv_scale_law_1/smoke/`.
4. Add a result note after the smoke run with the exact command and claim boundary.

Do not modify the dirty `Makefile` or unrelated NeoCortex-X/CodePy files.

## Verification

```bash
/opt/homebrew/bin/python3 -m unittest -v test_kv_scale_law.py
/opt/homebrew/bin/python3 run_kv_scale_law.py --dry-run
/opt/homebrew/bin/python3 run_kv_scale_law.py --smoke --output-dir artifacts/kv_scale_law_1/smoke
```

The smoke run proves wiring only and must report `INCOMPLETE`.
