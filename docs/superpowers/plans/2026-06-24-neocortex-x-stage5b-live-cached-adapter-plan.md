# NeoCortex-X Stage 5B Live/Cached Search Adapter Plan

Дата: 2026-06-24.
Design spec: `docs/superpowers/specs/2026-06-23-neocortex-x-core-experiment-design.md`.

## Цель

Add a live-search-capable adapter behind Stage 5 while preserving deterministic
cached replay fixtures.

Stage 5 already validates synthetic external observations with provenance.
Stage 5B adds the production-facing adapter contract:

- cached fixture mode for deterministic tests;
- optional live mode behind the same schema;
- replay from cache must match the normalized factual state.

## Modes

- `cached`: read fixture observations from local JSON and run provenance update.
- `live`: future backend that may call web/search, then writes the same fixture
  schema before evaluation.

Smoke uses `cached` mode only.

## Fixture Schema

```text
query
ground_truth
observations[]:
  subject
  value
  source_id
  source_trust
  observed_at
  provenance_url
```

## Metrics

- `factual_accuracy`;
- `provenance_coverage`;
- `replay_consistency`;
- `cache_fixture_valid`;
- `adapter_minus_naive_accuracy`;
- `adapter_minus_no_external_accuracy`.

## Pass/Fail

- `protocol_evidence.valid = true`;
- `mode = cached`;
- `cache_fixture_valid = true`;
- `provenance_adapter.factual_accuracy_mean >= 0.95`;
- `provenance_adapter.provenance_coverage_mean = 1.0`;
- `provenance_adapter.replay_consistency_mean = 1.0`;
- `adapter_minus_naive_accuracy_mean >= 0.25`;
- `adapter_minus_no_external_accuracy_mean >= 0.50`.

## Files

- `research/neocortex_x/stage5b_live_cached_adapter.py`
- `research/neocortex_x/fixtures/stage5b_cached_observations.json`
- `Makefile`
  - add `neocortex-x-stage5b-cached-smoke`
- `research/neocortex_x/README.md`

## Artifact

- `artifacts/neocortex_x/stage5b_live_cached_adapter.json`

## Verification

```bash
python3 -m py_compile research/neocortex_x/stage5b_live_cached_adapter.py
make neocortex-x-stage5b-cached-smoke
```
