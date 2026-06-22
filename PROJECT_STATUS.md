# Project Status

**Product:** KV-Memory SSM long-delay recall benchmark (see `README.md` and
`RESULTS.md`). The repo was converged from a ~92-script sandbox to a focused
23-script product surface plus a `research/` archive. Prior state is in the git
freeze commit.

## Status summary

- **KV-memory recall benchmark: PASS, reproducible.** Explicit recurrent KV
  memory beats a parameter-matched, context-capped Transformer on long-delay
  binding recall across synthetic, generated-code, and real-AST tasks, on
  repeated seeds, with protocol gating. Full numbers in `RESULTS.md`.
- **Sparse KV read routing: PASS.** A query-margin gate drives reads near-binary
  and supports deterministic hard-threshold inference at `1.0000` recall over 6
  seeds.
- **Natural-language LM loss: Transformer still wins (measured trade-off).** The
  SSM wins throughput, recurrent-state memory footprint, and effective context,
  but loses held-out LM loss on the million-token Gutenberg/BPE protocol.
- **General algorithmic synthesis (research/): NOT solved.** Documented negative
  result; see `research/codepy/`.

## Product components (repo root)

**Core runtime**
- `agent_ssm_core.py` — shared SoA agent-world math (observation, state update,
  reward, init) used by the runtime and replay.
- `soa_ssm_agents.py` — hand-coded flat-array SSM over many agents; JAX/XLA
  recurrent update demo (`h_t = tanh(A h_{t-1} + x_t B)`).
- `train_ssm_text.py` — word/BPE next-token SSM trainer with JAX autodiff + Adam,
  truncated BPTT, checkpoint save/load, tokenizer stored in the checkpoint.
- `text_tokenization.py` — shared word + lightweight BPE tokenizer contract.

**The benchmark**
- `compare_long_context_recall.py` — the core harness. Generates/extracts
  delayed-recall records (assignment, multi-assignment, distractor, alias,
  update, mixed-code, generated-python-code, real-python-code) and compares the
  KV/sparse-KV SSM against a parameter-matched causal Transformer on held-out
  recall.
- `run_long_context_recall_multiseed.py` — runs the harness across seeds,
  aggregates per-seed JSON, and enforces `protocol_evidence` (delay beyond
  context, seed count, query-position count) before accepting a result.

**SSM vs Transformer baseline**
- `compare_text_architectures.py` — trains both models on one split/tokenizer,
  matches parameter budget, reports loss/perplexity/accuracy/throughput/memory/
  effective-context; also hosts SSM variants (selective, conv-skip, token-memory,
  kv-memory, sparse-kv, aux-recall).
- `run_text_architecture_multiseed.py` — repeated-seed million-token aggregation
  with a bounded-benchmark guardrail in every artifact.

**Data + text tooling**
- `ingest_text_data.py` — builds Project Gutenberg / local corpora with
  SHA256/token-count manifests; sharding and arbitrary-ID ingestion.
- `materialize_text_corpus.py` — rebuilds a bounded train file from a sharded
  manifest.
- `profile_text_corpus.py` — token/vocabulary/memory profiling and checkpoint OOV
  analysis; recommends run shape.
- `run_text_checkpoint.py` — evaluates a checkpoint/manifest, generates samples,
  emits a JSON run report.
- `compare_text_checkpoints.py` — two-checkpoint loss/accuracy/throughput/OOV
  comparison with CI-style regression gating.
- `run_text_experiment.py` / `sweep_text_experiments.py` / `list_text_experiments.py`
  — held-out profile→train→compare pipeline, grid sweeps, and leaderboards.
- `promote_text_checkpoint.py` / `inspect_text_checkpoint.py` — promote a winning
  checkpoint into a manifest artifact; build JSON/Markdown model cards.
- `plan_text_sweep.py` / `run_text_sweep_plan.py` — resumable large-sweep planning
  and batch execution.

**Quality and measurement**
- `verify_system.py` — end-to-end smoke verifier (22 steps): compile, agent
  dynamics, text checkpoint save/load/run, profiling, comparison, architecture
  comparison + multiseed, the recall benchmarks (long-context, assignment,
  multi-assignment, multi-query, generated-/real-python-code), and the
  experiment/sweep/promote/inspect pipeline.
- `benchmark_system.py` / `compare_benchmarks.py` — quick/standard throughput
  benchmarks with regression comparison.

## research/ (not the product)

- `research/codepy/` — safe program synthesis over a constrained opcode DSL.
  PASS on synthesis/compression/skill-promotion/hierarchical-reuse and an early
  Stage-4B scaling curve; **NOT solved** on autonomous stateful synthesis
  (`argmax_index4`) — a documented negative result with trace priors, trajectory
  beams, prefix-value rollouts, survival policies, and a late-game re-ranker.
- `research/evolution/` — agent/formula evolution, self-play coevolution, and an
  LLM-style formula-hypothesis feedback loop.

Run research scripts from the repo root with `PYTHONPATH=.` (see
`research/README.md`).
