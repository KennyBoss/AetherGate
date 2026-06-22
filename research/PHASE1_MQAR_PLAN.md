# Phase 1 plan — MQAR external validation (make-or-break)

Goal: run KV-Memory SSM vs parameter-matched Transformer on **canonical MQAR**
(Multi-Query Associative Recall, Zoology/"Based"), a benchmark we did **not**
design, and write the verdict — positive *or* negative — into `RESULTS.md`.

## What the recall harness actually is (from code reading)

- `compare_long_context_recall.py` — one file. Per-task `make_*` data generators →
  shared `train_model` / `eval_model` loop → JSON payload. SSM and Transformer
  are parameter-matched via `choose_transformer_d_model`.
- `run_long_context_recall_multiseed.py` — repeats seeds, aggregates
  `summary.json`, and gates on `protocol_evidence.valid` (seed count, query
  positions, delay-beyond-context).
- **Honesty check passed:** headline runs do **not** pass
  `--init-assignment-kv-memory`. The win is architectural (`--ssm-variant
  kv-memory` + query-margin gate), not hand-wired weights.

## The two crutches canonical MQAR must remove

1. **`QUERY_TOKEN` marker.** Scoring uses `query_mask = (input_ids ==
   QUERY_TOKEN)`. Canonical MQAR has no marker — the query is a *repeated key*
   and the value is predicted at the next position.
2. **Structured tokens + split vocab.** Our tasks use `ASSIGN`/`SEP` and disjoint
   name/value ranges (`ASSIGN_NAME_OFFSET`, `value_offset`). Canonical MQAR uses
   a single shared vocabulary and bare `k v k v …` with random gaps.

Stripping these is the whole point: it tests whether the mechanism *learns*
binding recall under a standard formulation rather than under our token layout.

## Design

### 1. New generator `make_mqar_recall`
- Single shared vocab of size `V` for both keys and values.
- `N` key→value pairs, keys unique within a sequence, values sampled from `V`.
- Pairs scattered across a length-`L` sequence with **random gaps** (canonical
  MQAR), not packed contiguously.
- `Q` queries: a previously-seen key reappears; target = its bound value.
- Held-out: train/eval use disjoint seeds (as today); keys/values resampled.

### 2. Honest scoring (remove the marker dependency)
The harness scores at `QUERY_TOKEN`. Two options:

- **(A) Minimal marker** — keep one neutral "query" position token so the
  existing `query_mask` path works unchanged. Fast, but a mild deviation from
  canonical MQAR (still far more external than our tasks).
- **(B) Position-mask scoring** — generalize `recall_loss_and_accuracy` to accept
  an explicit loss/target position mask (the post-query positions), so **no
  marker token exists at all**. Fully faithful to canonical MQAR. More code:
  thread a `loss_mask` through `make_task_data` → train/eval → metrics.

Recommendation: **(B)** for a maximally credible result; (A) only if we want a
same-day smoke first.

### 3. Fair models (no privileged init)
- Both start fair, parameter-matched, exactly as the current headline.
- SSM uses `--ssm-variant kv-memory` + the same query-margin gate training; **no**
  `--init-assignment-kv-memory`.
- Transformer keeps the context cap (`bypass_mode`) so the binding sits beyond
  its window — plus a full-context control run, as we already do.

### 4. Wire into the honesty machinery
- Add `mqar` to the `--task` choices, `vocab_size_for_task`,
  `target_tokens_for_task`, and the `run()` validation switch.
- Add a `compare-mqar-recall-multiseed` Makefile target mirroring
  `compare-mixed-code-recall-multiseed` (3 seeds, `--require-valid-protocol`).
- Report **quality** (recall acc/loss) **and** the cost story (throughput,
  parameter bytes, effective context) the RESULTS sell depends on.

### 5. Run + verdict
- Smoke: 1 seed, small `L`, to confirm wiring and protocol_valid.
- Full: 3 seeds. Write the result into `RESULTS.md` §Phase 1 — **including if the
  SSM loses.** A negative result is a successful crash test, not a failure.

## MQAR config (CPU-scale, matched to our regime)
Starting point, to be tuned in the smoke run:
- `V` (vocab) ≈ 64; `N` (pairs) ≈ 8; `Q` (queries) ≈ 4; `L` (seq len) ≈ 128
  with random gaps; transformer context cap 32 (binding beyond window).
- Same optimizer settings as `mixed-code` (epochs 40, lr 0.006, batch 128).

## Exit criteria (decide explicitly, no fudging)
- **SSM holds** (recall margin survives on canonical MQAR, protocol_valid) →
  result has legs → Roadmap Phase 2a (scale / niche tool).
- **SSM breaks** → wins were a task-design artifact → document honestly in
  RESULTS, pivot to `research/codepy/` (Roadmap Phase 2b).

## Risks / honest caveats
- Stripping the marker may genuinely hurt the kv-memory variant — that is the
  test, not a bug to patch around.
- Random gaps + shared vocab raise difficulty; small `L` first, scale `L`/`N`
  only after a clean signal.
- If we keep option (A)'s marker, say so in RESULTS — do not imply full canonical
  fidelity.
