# Project Status

Current state: working SoA/SSM prototype stack with verification, benchmarking,
checkpointing, real-data ingestion, BPE subword tokenization, corpus profiling, checkpoint comparison, architecture
baselines against a tiny causal Transformer, text experiment
pipelines, experiment sweeps, experiment leaderboards, promoted checkpoints,
checkpoint inspection, package benchmarking, package validation, package
bundling, package restore, package comparison, release comparison, release
registries with memory diagnostics, release leaderboards with memory summaries,
release selection, active release execution, one-command release pipelines,
hidden-memory probes, hidden-memory comparison, prompt-suite memory matrices,
memory-suite validation, release-pipeline memory gates, static memory viewing,
static memory-matrix viewing, static release leaderboard viewing, static
release dashboard viewing, static release evidence ledger viewing, release gate
summaries, tamper-tested and replay-tested release evidence ledgers,
release evidence audit rollups, optional one-command release evidence audit
pipelines, release evidence audit validation, static release evidence audit
validation viewing, release evidence audit validation comparisons, static
release pipeline proof cockpit viewing,
formula-level SSM architecture search, self-play coevolution, safe CodePy block
evolution, LLM-compatible hypothesis feedback loops, and replayable evolved
policies.

## Runtime Layers

0. `agent_ssm_core.py`
   - Shared agent-world constants, SoA observation builder, state update,
     reward, entropy, and world initialization.
   - Used by the hand-coded runtime, evolutionary search, and replay harness.

1. `soa_ssm_agents.py`
   - Hand-coded SSM rules for many independent logical agents.
   - Demonstrates flat SoA rails and JAX/XLA recurrent updates.

2. `train_ssm_text.py`
   - Word-level or BPE-subword next-token SSM.
   - Trains `A/B/C`, embeddings, and biases with JAX autodiff plus built-in Adam.
   - Supports `--save-checkpoint`, `--load-checkpoint`, and `--eval-only`.
   - Stores tokenizer configuration inside checkpoints so run, compare,
     benchmark, probe, and release commands use the same text encoding.

2a. `text_tokenization.py`
   - Shared tokenizer contract for word and lightweight BPE modes.
   - Learns merge tables, encodes/decodes subword tokens, builds vocabularies,
     and preserves backward compatibility for existing word-level artifacts.

2b. `ingest_text_data.py`
   - Builds real text corpora for experiments.
   - Supports a Project Gutenberg public-domain book preset and local text
     ingestion.
   - Writes cleaned corpus text plus SHA256/token-count/source metadata.
   - Supports cached batch ingestion with built-in Gutenberg IDs and
     `--target-bytes` for larger repeatable corpora.
   - Can write sharded corpora via `--shard-bytes` for larger-than-memory
     dataset management while preserving a manifest.
   - Can ingest arbitrary Project Gutenberg IDs with `--gutenberg-ids` or
     `--id-range`, continuing past missing books with `--continue-on-error`.

2c. `materialize_text_corpus.py`
   - Rebuilds a bounded `--text-file` from a sharded ingestion manifest.
   - Lets the current trainer consume a selected slice of a larger corpus while
     the storage layer can keep growing in shards.

2d. `plan_text_sweep.py`
   - Produces JSON plans for large resumable sweeps.
   - Emits batch commands using `sweep_text_experiments.py --resume`,
     `--start-index`, and `--max-runs`.

2e. `run_text_sweep_plan.py`
   - Executes the next incomplete batch from a sweep plan.
   - Writes progress JSON with completed run counts, batch status, and command
     tails so hundred-plus sweeps can advance safely over repeated sessions.

3. `run_text_checkpoint.py`
   - Loads a saved text SSM checkpoint.
   - Evaluates loss/accuracy on text and generates samples from a prompt.
   - Can export a JSON run report.
   - Can run a promoted model package directly via `--manifest`.
   - Can run the selected active package release directly via `--release`.

4. `probe_text_memory.py`
   - Loads a checkpoint, promoted manifest, or active release pointer.
   - Traces `STATES_H` across prompt tokens and exports JSON/CSV timelines.
   - Reports state norms, state deltas, state statistics, and top predictions.
   - Can include hidden-state vectors for cosine/L2 memory geometry.

5. `compare_text_memory.py`
   - Compares two hidden-memory probe JSON reports.
   - Reports token match rate, top prediction overlap, norm drift, state-delta
     drift, probability drift, final top-token agreement, and optional
     vector-level L2/cosine geometry.
   - Can fail on memory-drift thresholds for regression gating.

6. `run_text_memory_suite.py`
   - Runs a batch of memory probes for multiple prompts against one checkpoint,
     manifest, or active release.
   - Writes one probe JSON/CSV per prompt plus pairwise prompt-matrix JSON/CSV
     summaries, including vector geometry when enabled.

7. `validate_text_memory_suite.py`
   - Validates `memory_suite.json` artifacts before they become release
     diagnostics.
   - Checks prompt/pair counts, CSV row counts, per-prompt probe artifacts,
     metric ranges, optional vector geometry, and optional drift/overlap
     thresholds.

8. `benchmark_text_package.py`
   - Benchmarks a saved checkpoint or promoted package manifest.
   - Verifies checkpoint bytes/SHA256, warms up JAX, and reports best/mean
     throughput, loss, accuracy, OOV rate, and sample text.

9. `profile_text_corpus.py`
   - Profiles an external or built-in text corpus before training.
   - Reports token classes, vocabulary shape, top tokens, SoA memory estimates,
     and recommended `streams/tokens-per-stream/seq-len`.
   - Can compare a corpus against a saved text checkpoint vocabulary and report
     OOV rates.
   - Can profile either word tokens or BPE tokens; checkpoint OOV analysis uses
     the tokenizer stored in the checkpoint.

10. `compare_text_checkpoints.py`
   - Compares two saved text SSM checkpoints on the same corpus and eval shape.
   - Reports loss, accuracy, warmed-up throughput, OOV rates, and samples.
   - Can fail on loss or throughput regressions for CI-style gating.

10a. `compare_text_architectures.py`
   - Compares the text SSM against a tiny causal Transformer on one train/eval
     split, tokenizer, and evaluation shape.
   - Chooses a Transformer width close to the SSM parameter count unless an
     explicit width is provided.
   - Reports loss, perplexity, accuracy, warmed-up throughput, parameter
     memory, optimizer-memory estimate, and effective context length.
   - Writes a JSON guardrail that smoke runs are protocol checks, not evidence
     that either architecture is generally better.

10b. `run_text_architecture_multiseed.py`
   - Runs `compare_text_architectures.py` across repeated seeds and aggregates
     the per-seed JSON reports.
   - Enforces protocol evidence for repeated seeds, million-token train/eval
     positions, and minimum real-corpus train/eval token counts.
   - Reports mean/stdev/min/max loss, accuracy, throughput, parameter counts,
     attention-score memory vs SSM state memory, context ratios, and winner
     counts.
   - Keeps the same bounded-benchmark guardrail: repeated seeds are stronger
     evidence than smoke tests, not universal architecture proof.

11. `compare_text_packages.py`
   - Compares two promoted text packages through their manifests.
   - Revalidates package metadata and benchmark reports before comparing.
   - Can fail on release-level loss, accuracy, or throughput regressions.

12. `compare_text_releases.py`
   - Compares two active release pointers or ranked registry entries.
   - Reports loss, accuracy, throughput, checkpoint/archive SHA agreement, and
     memory diagnostic deltas.
   - Can fail on package regressions and memory overlap/cosine/L2/norm drift.

13. `run_text_experiment.py`
   - Orchestrates profile -> train baseline -> train candidate -> compare.
   - Splits text into train/eval files, trains on train, compares on held-out
     eval, and writes profile, checkpoints, comparison, and run metadata.
   - Accepts `--tokenizer bpe` and BPE vocabulary settings for real-corpus runs.

14. `register_text_package.py`
   - Registers validated text packages as release artifacts.
   - Stores archive/checkpoint SHA256, benchmark metrics, selected run metadata,
     optional package comparison verdicts, and optional memory-suite validation
     summaries.
   - Sorts registry entries by benchmark loss and throughput.

15. `list_text_experiments.py`
   - Reads saved text experiment directories and ranks them by metric.
   - Prints a leaderboard with eval size and can export JSON/CSV summaries.

16. `list_text_releases.py`
   - Reads package release registries.
   - Prints ranked release tables and exports JSON/CSV leaderboards.
   - Flattens benchmark, SHA256, package path, comparison verdict, and memory
     diagnostic fields.

17. `sweep_text_experiments.py`
   - Runs a grid of held-out text experiments over dimensions, learning rates,
     and epoch counts.
   - Writes per-run experiment directories plus sweep JSON/CSV leaderboards.
   - Can reuse a saved tokenizer config across runs so bulk BPE sweeps do not
     relearn the merge table for every configuration.

18. `select_text_release.py`
   - Selects the active package release from a registry by rank, name, or SHA.
   - Verifies archive SHA256 and package metadata before writing the active
     release pointer.
   - Carries release memory diagnostics into the active pointer when present.

19. `build_text_release_dashboard.py`
   - Builds a single dashboard JSON from release leaderboard, active release,
     memory suite, validation, run report, and optional pipeline report.
   - Audits that the active release matches the rank-1 leaderboard row and that
     memory summaries agree across current pointer, suite, and validation.
   - Can fail with `--require-valid` when release dashboard invariants do not
     hold.

19a. `validate_text_release_dashboard.py`
   - Validates one-file release dashboard JSON artifacts after they are built.
   - Checks embedded leaderboard/current/suite/validation/run/pipeline payloads,
     audit check names, metric consistency, and referenced artifact existence.
   - Can enforce expected prompt/pair counts and minimum memory overlap.

19b. `compare_text_release_dashboards.py`
   - Compares two validated release dashboard JSON artifacts.
   - Reports loss, accuracy, throughput, audit, and memory-geometry deltas.
   - Can fail on quality, throughput, audit, or memory regressions.

19c. `build_text_release_history.py`
   - Builds a compact release history index from validated dashboard JSON
     artifacts.
   - Records latest/best loss, throughput, memory marks, and per-release deltas.
   - Accepts repeated dashboard paths or glob patterns for dashboard histories.

19d. `validate_text_release_history.py`
   - Validates release history JSON artifacts after they are built.
   - Checks summary best/mean metrics, per-release deltas, ordering, uniqueness,
     and optional referenced dashboard artifact existence.

19e. `analyze_text_release_history.py`
   - Analyzes latest-release trend against previous, first, best-loss,
     best-throughput, or best-memory history baselines.
   - Reports loss, accuracy, throughput, audit, and memory-geometry deltas.
   - Can fail on quality, throughput, audit, or memory regressions.

19f. `summarize_text_release_gates.py`
   - Summarizes dashboard validation, dashboard comparison, history
     validation, and history trend analysis artifacts into one verdict JSON.
   - Reports failed gate counts plus release loss, throughput, memory overlap,
     and history-baseline deltas.

19g. `generate_text_release_report.py`
   - Generates a portable Markdown report from release gate, dashboard, and
     history analysis artifacts.
   - Records final verdict, gate table, history deltas, dashboard audit checks,
     source artifacts, and rebuild commands.

19h. `bundle_text_release_evidence.py`
   - Bundles release evidence artifacts into a tar.gz archive plus checksum
     manifest.
   - Includes gate summary, Markdown report, dashboard/history artifacts,
     active release pointer, run report, leaderboard, and memory-suite files.

19i. `validate_text_release_evidence.py`
   - Validates release evidence bundle archives and manifests.
   - Recomputes archive/file SHA256 values, rejects unsafe tar paths, checks
     the inner evidence manifest, and revalidates the bundled gate summary.

19j. `compare_text_release_evidence.py`
   - Compares two validated release evidence bundle manifests.
   - Reports archive SHA equality, file-label drift, per-file SHA/byte drift,
     archive-name drift, validity changes, and gate failure deltas.
   - Has a static evidence comparison viewer for archive/file hash drift,
     label-set changes, byte deltas, and per-label evidence status.

19k. `unbundle_text_release_evidence.py`
   - Safely restores release evidence archives into a clean output directory.
   - Verifies archive SHA256, rejects unsafe tar paths, checks restored file
     SHA/byte counts, and revalidates the restored inner evidence manifest.

19l. `register_text_release_evidence.py`
   - Registers validated release evidence bundles in a tamper-evident hash-chain
     ledger.
   - Records manifest/archive/report hashes, previous entry hash, canonical
     entry hash, and chain head for independent audit.

19m. `validate_text_release_evidence_ledger.py`
   - Validates release evidence ledger structure and hash-chain integrity.
   - Recomputes entry hashes, checks previous-hash links and chain head, and can
     verify linked artifact SHA256 values.

19n. `tamper_text_release_evidence_ledger.py`
   - Creates a modified copy of a valid release evidence ledger.
   - Requires the independent ledger validator to reject the tampered copy with
     the expected hash-chain mismatch.

19o. `replay_text_release_evidence_ledger.py`
   - Replays ledger registration with insert, idempotent update, and append
     operations.
   - Builds a second valid evidence bundle, verifies previous-hash chaining,
     and validates artifact snapshots across the two-entry ledger.

19p. `compare_text_release_evidence_ledgers.py`
   - Compares two validated release evidence ledger JSON artifacts.
   - Reports chain-head drift, entry-count deltas, added/missing entry hashes,
     archive/manifest SHA set drift, and changed fields for common entries.

19q. `summarize_text_release_evidence_audit.py`
   - Summarizes release evidence proof artifacts into one audit verdict.
   - Verifies bundle manifest, validation, restore, bundle comparison, source
     ledger, ledger validation, tamper detection, replay semantics,
     self-comparison, and expected one-entry ledger growth.
   - Writes machine-readable JSON plus a portable Markdown audit report.

19r. `compare_text_release_evidence_audits.py`
   - Compares two release evidence audit rollup JSON artifacts.
   - Reports final validity drift, check-set drift, check validity/message/
     metric/path drift, replay/tamper changes, entry-count deltas, source-path
     drift, archive SHA changes, chain-head changes, and artifact-check deltas.
   - Can fail on audit proof regressions for CI-style final release gates.

19s. `validate_text_release_evidence_audit.py`
   - Validates a release evidence audit rollup independently.
   - Enforces required audit checks, required source artifacts, summary
     invariants, full SHA256 chain-head fields, replay/tamper semantics, and
     cross-artifact consistency against the ten source JSON proof files.
   - Writes a compact validation JSON for CI-style release proof gates.

19t. `compare_text_release_evidence_audit_validations.py`
   - Compares two release evidence audit validation JSON artifacts.
   - Reports validity drift, audit path drift, release-name drift, check/
     failure/error-count deltas, artifact-check coverage deltas, source
     artifact coverage deltas, and validation error-set changes.
   - Can fail on validation proof regressions for CI-style release gates.

20. `release_text_pipeline.py`
   - Runs the full operator release path from experiment leaderboard to active
     release smoke run.
   - Promotes, inspects, benchmarks, validates, bundles, restores, compares,
     registers, ranks, selects, and executes the chosen package.
   - Runs and validates a hidden-memory prompt suite for the selected release.
   - Builds the audited one-file release dashboard JSON as the final operator
     artifact.
   - Can optionally continue through release gates, Markdown report generation,
     evidence bundling, evidence validation/restore/compare, ledger registration,
     ledger validation/tamper/replay/compare, evidence audit rollup, audit
     validation, and audit comparison with `--include-evidence-audit`.
   - Writes a stage-by-stage JSON report with command tails, release summary,
     memory geometry metrics, dashboard audit results, and optional evidence
     audit proof paths.

20a. `validate_text_release_pipeline.py`
   - Validates release pipeline JSON reports as standalone control artifacts.
   - Checks required stage records, release pointers, memory-suite geometry,
     embedded dashboard audit coverage, and optional evidence-audit proof paths.
   - Can require all linked artifacts and the full `--include-evidence-audit`
     stage chain for CI release gates.

20b. `compare_text_release_pipeline_validations.py`
   - Compares two release pipeline validation JSON artifacts.
   - Reports validity drift, pipeline/name drift, stage-count drift,
     memory-suite count deltas, dashboard/evidence audit coverage deltas,
     artifact-check coverage deltas, and validation error-set changes.
   - Can fail on pipeline validation proof regressions for CI release gates.

20c. `compare_text_release_pipelines.py`
   - Compares two full release pipeline JSON reports directly.
   - Reports stage-set drift, release/rank deltas, loss/accuracy/throughput
     deltas, memory-geometry drift, dashboard/evidence audit regressions,
     evidence archive SHA drift, and evidence ledger chain-head drift.
   - Can fail on full pipeline report regressions before looking at separate
     validation artifacts.

20d. `validate_text_release_pipeline_comparison.py`
   - Validates full release pipeline comparison JSON artifacts.
   - Recomputes stage-set drift, metric deltas, memory-geometry deltas,
     proof-hash drift, audit validity regressions, and shape-error propagation
     from the embedded baseline/candidate pipeline summaries.
   - Can require evidence-audit proof fields, expected stage counts, empty
     threshold failures, and empty shape-error lists for CI release gates.

20e. `compare_text_release_pipeline_comparison_validations.py`
   - Compares two full release pipeline comparison validation JSON artifacts.
   - Reports validation validity drift, compared-pipeline path drift,
     stage-count drift, loss/memory delta drift, proof flag changes, threshold
     failure regressions, and validation error-set changes.
   - Can fail on validation-proof drift for CI release gates.

21. `promote_text_checkpoint.py`
   - Promotes the best, candidate, or baseline checkpoint from a text
     leaderboard.
   - Copies the selected checkpoint, writes SHA256/size into a manifest, and can
     smoke-run it.

22. `inspect_text_checkpoint.py`
   - Inspects a saved text checkpoint or promoted manifest without JAX.
   - Reports config, metrics, vocab preview, parameter shapes, parameter count,
     and memory footprint as JSON and Markdown model cards.

23. `validate_text_package.py`
   - Validates promoted text package integrity.
   - Checks checkpoint bytes/SHA256, model cards, smoke-run metadata, and
     optional required benchmark metadata.
   - Resolves package-local checkpoint paths first so restored bundles are
     validated as self-contained artifacts.

24. `bundle_text_package.py`
   - Bundles a validated promoted text package into `.tar.gz`.
   - Includes manifest, checkpoint, model cards, smoke run, benchmark report,
     and validation report.
   - Writes bundle manifest with file checksums and archive SHA256.

25. `unbundle_text_package.py`
   - Safely restores bundled text packages.
   - Verifies archive SHA256, rejects unsafe paths, checks file checksums, and
     validates the restored package including its benchmark report.

26. `evolve_ssm_agents.py`
   - Population-level evolutionary search over agent SSM rule sets.
   - Supports `--save-best` for reusable `.npz` genomes.

26a. `evolve_ssm_formulas.py`
   - NAS-style formula search over the SSM hidden-state update rule.
   - Mutates a constrained math DSL instead of arbitrary Python code:
     memory/input unary ops, combine ops, and final activation ops.
   - Runs a short inner weight-evolution loop for each formula in the same
     agent world, then ranks formulas by reward and throughput.
   - Writes a JSON guardrail that this is constrained formula search, not
     autonomous AGI.

26b. `coevolve_ssm_selfplay.py`
   - Coevolves runner and blocker SSM populations in a small self-play world.
   - Evaluates a runner-vs-blocker payoff matrix each generation, selects
     elites on both sides, and mutates both populations.
   - Reports pressure deltas, diversity, throughput, payoff matrices, role
     descriptions, and a non-AGI guardrail.

26c. `evolve_code_blocks.py`
   - Evolves safe CodePy programs as fixed-length opcode arrays over cached
     command blocks instead of generating source text token by token.
   - Uses a constrained register DSL with `r0`, `r1`, predicate rail `p`,
     optional two-input `x/y` datasets, vectorized JAX execution over
     population/cases, and train/holdout MSE fitness.
   - Supports predicated control-flow blocks such as comparisons and
     `where(p, ...)`, enabling relu, sign, max(x,y), and piecewise targets.
   - Runs semantic peephole minimization after evolution, removing blocks only
     when the shorter program preserves train and holdout correctness; exposes
     `--length-penalty` for direct short-program pressure during search.
   - Reports search-vs-minimal complexity with `search_blocks`,
     `minimal_blocks`, and `compression_ratio`; includes demos for cubic,
     compact quadratic, abs, relu, sign, max(x,y), and a piecewise function.
   - Supports `--block-profile base|learned`; the learned profile promotes
     reusable minimized programs such as square, abs, relu, max, and sign into
     macro-blocks for downstream synthesis.
   - Also exposes Stage 4 hierarchical profiles that add promoted skills in
     order, so later tasks can be searched with only earlier macro-blocks.
   - Keeps an explicit guardrail that this is DSL search, not arbitrary
     generated Python execution.

26d. `compare_code_block_library.py`
   - Runs matched CodePy searches with base primitives vs learned macro-block
     library profiles.
   - Compares success rate, best generation, minimized block count, runtime,
     evaluated programs, and learned-vs-base deltas for targets such as
     `clamp01` and `max_abs_x_y`.

26e. `code_block_skill_ladder.py`
   - Runs a three-phase CodePy accumulation experiment: Phase A discovers
     skills from base primitives, Phase B promotes successful minimized
     programs into macro-blocks, and Phase C compares downstream synthesis
     with base vs learned profiles.
   - Records promoted blocks, source minimized blocks, `compression_ratio`,
     downstream `reuse_frequency`, and `fitness_gain`.

26f. `compare_code_block_hierarchy.py`
   - Runs the Stage 4 hierarchy ladder: square, abs, max, clamp,
     `max(abs(x), y)`, `clamp(abs(x),0,1)`, and
     `piecewise(max(abs(x),y))`.
   - Compares base primitives against hierarchical profiles containing only
     skills promoted by earlier stages, then records `generation_found`,
     active/search blocks, `compression_ratio`, search time, and deltas for
     plotting growth in base-vs-library separation.

26g. `compare_code_block_stage4b.py`
   - Runs the Stage 4B statistical scaling test over `max_abs_x_y`,
     `clamp_abs_01`, and `piecewise_max_abs_x_y`.
   - Sweeps hierarchy depths across many seeds and writes plot-ready
     `success_rate vs hierarchy_depth` rows plus generation-found, block-count,
     compression, and search-time statistics.

26h. `analyze_code_block_pruning.py`
   - Reads Stage 4B results and analyzes macro pruning/regularization with
     `success_rate - penalty * hierarchy_depth`.
   - Reports noisy depths, Pareto frontiers, and regularized depth choices for
     each target.

26i. `CODEPY_TURING_COMPLETENESS_PLAN.md`
   - Documents the bounded-loop/JUMP path for CodePy using fixed `MAX_STEPS`,
     a program-counter rail, and fixed-size memory under `jax.lax.scan`.
   - Explicitly keeps General Algorithmic Scaling unproven until bounded-loop
     programs solve stateful held-out tasks across seeds.

26j. `evolve_code_tape.py`
   - Implements the first bounded stateful CodePy executor with `r0`, `r1`,
     fixed `tape[8]`, pointer `ptr`, program counter `pc`, predicate rail, and
     bounded execution through `jax.lax.scan`.
   - Supports tape reads/writes, pointer moves, predicate checks, conditional
     increment, accumulation, and bounded loop-back via `pc = 0`.
   - The action space is append-only extended for `argmax_index4` with
     `tape[ptr] > r0/r1` comparisons, `r0/r1 = ptr`, conditional updates from
     `tape[ptr]` or `ptr`, register copy, `pc = 3`, conditional loop-back to
     `pc = 3`, and bounded `halt`.
   - Includes `sum4` and `count_positive4` vector tasks. Reference-injected
     demos validate state geometry with train/holdout MSE 0. Shaped clean
     evolution now autonomously discovers zero-error stateful loops for both
     `sum4` and the predicate/conditional `count_positive4` task without
     reference injection.
   - `argmax_index4` now passes the state-geometry gate with a zero-error
     reference program `[3, 24, 7, 20, 25, 28, 7, 15, 30, 32, 31]`; autonomous
     discovery remains pending.

26k. CodePy tape prior-search layer
   - `evolve_code_tape.py` can now write JSONL execution traces for every
     evaluated program via `--trace-jsonl --trace-all`, including program
     tokens, reward/MSE metrics, and normalized trace embeddings.
   - `train_code_tape_prior.py` trains a lightweight logistic prior over program
     tokens plus execution trace embeddings.
   - `search_code_tape_prior.py` uses that prior inside bounded beam search,
     while every candidate is still verified by the safe tape interpreter.
   - `promote_code_tape_macros.py` promotes top zero-error traces into explicit
     safe DSL macro manifests. `make code-tape-prior-demo` currently logs 16,384
     `sum4` traces, trains a prior with about 0.99 test AUC, finds a zero-error
     loop with 672 guided beam evaluations, and promotes five macro candidates.
   - `make code-tape-count-positive-prior-demo` logs 65,536 clean no-reference
     `count_positive4` traces, finds a zero-error predicate loop at generation 7,
     trains a weak warm-start prior from one positive trace, re-finds a compact
     zero-error loop with 2,320 guided beam evaluations, and promotes three
     macro candidates. A transfer attempt using the `sum4` prior did not solve
     `count_positive4`, which is evidence that branch-specific traces matter.
   - `search_code_tape_trajectory.py` implements the Phase 4
     trajectory-guided path: it samples execution signatures over `r0`, `r1`,
     `ptr`, `pc`, and predicate `p`, applies a beam diversity constraint against
     repeated signatures, writes a latent trace graph for reconstruction, and
     can export every scored beam candidate as JSONL.
   - `build_code_tape_trace_dataset.py` and `train_code_tape_trace_prior.py`
     close the contrastive Step 3 loop for `argmax_index4`: autonomous
     trajectory traces, seeded/reference successes, and near-misses are turned
     into a trace-only dataset for `P(trace | task)`. The current full artifact
     has 8,205 records: 3 success, 6,369 near-miss, and 1,833 failure. The
     strict trained trace prior omits program-token identity, train/holdout MSE,
     and the seeded/reference marker; it separates success from near-miss with a
     probability gap of about 0.93, but this remains diagnostic because the
     positive class is supplied by seeded/reference traces.
   - `compare_code_tape_trace_guidance.py` closes the Step 4 loop by comparing
     autonomous trajectory beam search against beam + the contrastive trace
     prior on equal candidate budgets. The current b24 run evaluates 8,435
     programs per variant: baseline remains a near-miss at `train_mse=0.4375`,
     while small trace-prior weights reach `train_mse=0.21875` and improve
     holdout to 1.15625 at weight 0.25. A b48 equal-budget run evaluates 16,380
     programs per variant and repeats the same train-MSE improvement. Trace
     guidance alone did not find autonomous zero-error `argmax_index4`; the
     successful bridge below adds explicit ordered prefix survival.
   - A wider b160 guided run evaluates 51,660 programs and still reaches only
     `train_mse=0.21875`. The failure is localized: reference-like
     initialization survives through depth 2, then the delayed-reward
     read/index/move path falls out before compare/update/loop behavior becomes
     useful. A suffix-repair sweep over close near-misses did not produce an
     exact solution, and the experimental multilane role-survival selector is
     currently diagnostic rather than a winning search policy.
   - `search_code_tape_trajectory.py` now has an optional prefix-value rollout
     layer (`--value-rollouts`) that estimates `V(prefix)` from deterministic
     safe-DSL suffix completions and logs `prefix_value_score`,
     `prefix_value_suffix`, and `value_rollout_evaluations`. The diagnostic b64
     value run confirms that the layer can see good completions for
     delayed-reward initialization prefixes, but the current rollout policy does
     not improve beyond `train_mse=0.21875`; a value-lane selector currently
     overprotects poor futures and worsens the result.
   - `search_code_tape_trajectory.py` also has an experimental survival policy:
     high `V(prefix)` candidates get `survival_ttl`, `survival_score`, and
     `survival_root`, and `select_diverse` can reserve a survival lane with
     optional root-capping. The b64 TTL=8 artifact
     `artifacts/code_tape_prior/argmax_index4_survival_b64_ttl8_uncapped_w025.json`
     performs 38,920 total evaluations, logs 27 survival activations and 266
     selected protected prefixes, and keeps delayed-reward argmax branches alive
     through the double conditional update. It remains a negative result for
     autonomous synthesis: best train/holdout MSE is `0.75`/`1.96875`, and
     `first_solution` is still false. Root-capped variants were also negative,
     so the localized bottleneck has moved from early prefix survival to ranking
     the protected loop/return continuation.
   - `analyze_code_tape_late_game.py` turns that bottleneck into an offline
     measurable report over the survival JSONL. The current
     `artifacts/code_tape_prior/argmax_index4_late_game_analysis.json` shows two
     distinct failure families: the best overall top-50 near-misses are a
     noncanonical reward exploit that usually misses the initial `r0 =
     tape[ptr]` read, while the survival-protected top-50 contain canonical
     argmax blocks `[3,7,15,20,25,28,32]` in at least 90% of rows but fail in
     late ordering and terminal control (`28` before `25` in 38/50, loop test
     before first move in 50/50, missing `halt` in 49/50). The next target is
     therefore a late-game loop/update/return/halt ranker rather than more blind
     beam width.
   - The autonomous `argmax_index4` trajectory run currently builds a
     2,948-node graph and reaches a near-miss; seeded trace-prefix
     reconstruction reaches zero train/holdout error and is explicitly marked
     as a diagnostic rather than autonomous discovery.
   - `code_tape_late_game_features.py`, `train_code_tape_late_game_ranker.py`,
     and `rank_code_tape_late_game.py` act on that H4 diagnostic. The ranker is
     the structural counterpart to the trace-only prior: rather than
     `P(trace | task)` it scores the opcode sequence's control-flow grammar with
     features derived generically from the registered reference — signed
     canonical-order relations, terminal-control flags, stage completeness, and
     ordered n-grams. Positives are verified zero-error programs; negatives are
     the structurally-broken final-depth near-misses plus three synthetic
     reference corruptions (drop `halt`, swap `25`/`28`, loop-test before first
     move). `make code-tape-late-game-ranker-demo` trains the logistic ranker
     (positive-vs-negative score gap about `0.98` on the survival JSONL) and
     proves the re-ranking: the canonical argmax program rises to rank 1 of 2,241
     above the best reward-exploit near-miss, with canonical-minus-broken family
     gaps near `0.98` for missing-halt, swapped update, and loop-test-before-move
     (`artifacts/code_tape_prior/argmax_index4_late_game_rerank.json`). This stays
     a negative result for autonomous synthesis: the reference is injected, not
     discovered (`present_in_candidates` false), and the live beam search is
     unchanged. Wiring the ranker into beam search (a `late_game` selection lane
     plus an equal-budget comparison) is the explicit next step.
  - That integration now exists. `search_code_tape_trajectory.py` accepts
     `--late-game-ranker-json`, `--late-game-weight`, `--late-game-lane-fraction`,
     and `--late-game-min-remaining`; the ranker only fires on the final stages
     (`depth >= program_length - late_game_min_remaining`), biasing `raw_score`
     and running a dedicated `late_game` selection lane so structurally-correct
     programs survive even when `train_mse` temporarily loses to reward exploits.
     The first full guided run was a useful negative: the canonical depth-8 prefix
     `[3,24,7,20,25,28,7,15]` never reached the gated depths, proving the
     bottleneck was upstream of the ranker.
   - The upstream bridge now exists: `--prefix-stage-lane-fraction` and
     `--prefix-stage-ignore-signature-limit` add a small order-aware structural
     prefix lane. It scores generated prefixes by generic argmax stage progress
     (`read -> index-init -> move -> compare -> conditional updates -> loop ->
     return/halt`) and penalizes premature loop/return/control ordering; it does
     not inject the reference program. With `--prefix-stage-lane-fraction 0.125`,
     `make code-tape-late-game-guided-demo` now writes
     `artifacts/code_tape_prior/late_game_guidance/argmax_index4_late_game_guided.json`
     with `first_solution=true`, `train_mse=0.0`, `holdout_mse=0.0`, 20,650 direct
     beam evaluations, 18,625 value-rollout evaluations, and 88 prefix-stage lane
     selections. The canonical chain is selected by `prefix_stage` through
     `[3,24,7,20,25,28,7,15,30,32,31]`; the first zero-error solution appears as
     `[3,24,7,20,25,28,7,15,30,32]`, and the halt-terminated canonical program is
     also selected at depth 11.
   - `compare_code_tape_late_game_guidance.py` now compares the shared
     `Beam + Survival + PrefixStage` setup against the same setup plus
     `Late-Game` on equal budget. The latest
     `artifacts/code_tape_prior/argmax_index4_late_game_guidance_comparison.json`
     shows `baseline_solved=true`; late-game weights 0.25/0.5/1.0 also solve with
     the same 39,275 total evaluations. The current conclusion is that hard
     ordered prefix immunity is the decisive bridge through delayed reward for
     `argmax_index4`; value rollouts remain useful diagnostics/TTL triggers, and
     the late-game ranker remains a terminal grammar inspector rather than the
     component that first makes this task solvable.

26l. `run_hypothesis_loop.py`
   - Accepts LLM-style JSON/JSONL formula hypotheses, validates them against
     the safe SSM formula DSL, benchmarks accepted proposals, and rejects
     invalid operations before evaluation.
   - Can call an external generator command by sending prompt-context JSON to
     stdin and reading JSON/JSONL proposals from stdout.
   - Writes prompt context, JSON feedback, JSONL feedback for the next
     generator round, baseline deltas, and a non-arbitrary-code guardrail.

27. `run_evolved_agent.py`
   - Loads a saved evolved genome and replays it on a fresh world seed.

28. `export_agent_trajectory.py`
   - Exports small hand-coded or evolved-agent trajectories to CSV plus JSON
     summary for inspection and visualization.

29. `trajectory_viewer.html`
   - Static browser viewer for exported `trajectory.csv` and `summary.json`.
   - Renders world bounds, goal, threat, trails, agent actions, and summary
     metrics without a build step.

30. `text_memory_viewer.html`
   - Static browser viewer for `text_memory_probe.json`.
   - Renders hidden-state norm, state delta, top probability, token timeline,
     and top-k predictions without a build step.

31. `text_memory_suite_viewer.html`
   - Static browser viewer for `memory_suite.json`.
   - Renders prompt-to-prompt heatmaps for cosine, L2, top-overlap, and norm
     drift plus prompt and pair detail panels.

32. `text_release_leaderboard_viewer.html`
   - Static browser viewer for release leaderboard JSON.
   - Renders ranked releases with loss, throughput, memory overlap, and memory
     cosine bars.

33. `text_release_dashboard.html`
   - Static browser dashboard for release leaderboard, active release pointer,
     and memory-suite JSON.
   - Combines active release metrics, package audit fields, ranked releases,
     memory summaries, and prompt matrix in one operator view.

34. `text_release_dashboard_compare.html`
   - Static browser viewer for release dashboard comparison JSON.
   - Renders baseline/candidate release cards plus loss, accuracy, throughput,
     audit, and memory-geometry deltas from the release gate.

35. `text_release_evidence_ledger_viewer.html`
   - Static browser viewer for release evidence ledger JSON.
   - Renders chain-head status, previous-hash links, entry metrics, snapshot
     artifact paths, and optional validation or replay reports.

36. `text_release_evidence_ledger_compare.html`
   - Static browser viewer for release evidence ledger comparison JSON.
   - Renders chain-head drift, entry-count deltas, added/missing entry hashes,
     archive/manifest SHA-set drift, changed common entries, and failures.

37. `text_release_evidence_audit_viewer.html`
   - Static browser viewer for release evidence audit JSON.
   - Renders final audit verdict, replay/tamper indicators, proof-check cards,
     source artifact paths, and failed proof-chain checks.

38. `text_release_evidence_audit_compare.html`
   - Static browser viewer for release evidence audit comparison JSON.
   - Renders baseline/candidate audit cards, final proof drift, check drift,
     source drift, and threshold failures from the audit comparison gate.

39. `text_release_evidence_audit_validation_viewer.html`
   - Static browser viewer for release evidence audit validation JSON.
   - Renders validator verdict, audit path, release name, check/source
     coverage, failure counts, and validation error ledger.

40. `text_release_evidence_audit_validation_compare.html`
   - Static browser viewer for release evidence audit validation comparison
     JSON.
   - Renders baseline/candidate validation cards, count deltas, source
     coverage deltas, identity drift, error-set drift, and threshold failures.

41. `text_release_history_viewer.html`
   - Static browser viewer for release dashboard history JSON.
   - Renders latest release, best marks, and a timeline of quality, speed, and
     memory deltas.

42. `text_release_history_analysis_viewer.html`
   - Static browser viewer for release history analysis JSON.
   - Renders latest-vs-baseline verdicts, regression deltas, and threshold
     failures from the history trend gate.

43. `text_release_gates_viewer.html`
   - Static browser viewer for release gate summary JSON.
   - Renders the final release verdict, four gate cards, summary metrics, and
     threshold failures.

44. `text_release_pipeline_validation_viewer.html`
   - Static browser viewer for release pipeline validation JSON.
   - Renders validator verdict, pipeline path, release name, stage count,
     memory-suite counts, dashboard/evidence audit checks, artifact coverage,
     and validation errors.

45. `text_release_pipeline_validation_compare.html`
   - Static browser viewer for release pipeline validation comparison JSON.
   - Renders baseline/candidate validation cards, stage/memory/audit deltas,
     identity drift, error-set drift, artifact coverage drift, and threshold
     failures.

46. `text_release_pipeline_compare.html`
   - Static browser viewer for full release pipeline comparison JSON.
   - Renders baseline/candidate pipeline cards, metric deltas, stage drift,
     proof-hash drift, audit validity drift, and threshold failures.

47. `text_release_pipeline_comparison_validation_viewer.html`
   - Static browser viewer for full release pipeline comparison validation JSON.
   - Renders validation verdict, compared pipeline paths, stage/loss/memory
     drift, proof-hash agreement, threshold failure count, and validation
     errors.

48. `text_release_pipeline_comparison_validation_compare.html`
   - Static browser viewer for full release pipeline comparison validation
     comparison JSON.
   - Renders baseline/candidate validation cards, validation drift deltas,
     path/proof/error drift, and threshold failures.

49. `text_release_pipeline_proof_cockpit.html`
   - Static browser cockpit for the full release pipeline proof chain.
   - Accepts the six pipeline proof JSON artifacts or a bundled proof object,
     then renders release, validation, comparison, proof-hash, and drift status
     as one operator surface.

## Quality And Measurement

- `verify_system.py`
  - End-to-end smoke verifier.
  - Checks syntax, agent dynamics, text checkpoint save/load, text checkpoint
    runner, corpus profiling, text checkpoint comparison, text experiment
    pipeline, experiment leaderboard, text experiment sweep, checkpoint
    promotion, checkpoint inspection, manifest-based text execution, package
    benchmarking, package validation, package bundling, package restore,
    package comparison, release registration, release leaderboard export,
    release memory metadata, release comparison, active release selection,
    active release execution, text hidden-memory probes, hidden-memory
    comparison, prompt-suite memory matrices, memory-suite validation,
    release-pipeline memory gates, embedded release-pipeline dashboard
    generation, release-pipeline report validation, release-pipeline validation
    comparisons, full release-pipeline report comparisons, full
    release-pipeline comparison validation, full release-pipeline comparison
    validation comparisons, one-file release dashboard audit, release dashboard
    validation, release dashboard comparison, release dashboard history, release history
    validation, release history trend analysis, release gate summaries,
    Markdown release reports, release evidence bundles, release evidence
    validation, release evidence restore, release evidence hash-chain ledger,
    release evidence ledger validation, release evidence ledger tamper tests,
    release evidence ledger replay tests, release evidence ledger comparisons,
    release evidence audit rollups, release evidence audit validation,
    release evidence audit validation comparisons,
    release evidence audit comparisons,
    release evidence comparison,
    one-command release pipeline execution, evolution save-best,
    formula-level architecture search, LLM-style
    hypothesis feedback, self-play coevolution, safe CodePy block evolution,
    replay, trajectory export, text memory viewer hooks, text memory-suite
    viewer hooks, release leaderboard viewer hooks, release dashboard hooks,
    release dashboard comparison viewer hooks, release evidence ledger viewer
    hooks, release evidence ledger comparison viewer hooks, release evidence
    audit viewer hooks, release evidence audit validation viewer hooks, release
    evidence audit validation comparison viewer hooks, release evidence audit
    comparison viewer hooks, release history viewer hooks, release history
    analysis viewer hooks, release gate summary viewer hooks, release pipeline
    validation viewer hooks, release pipeline validation comparison viewer
    hooks, release pipeline comparison viewer hooks, release pipeline
    comparison validation viewer hooks, release pipeline comparison validation
    comparison viewer hooks, release pipeline proof cockpit hooks, and static
    trajectory viewer hooks.

- `benchmark_system.py`
  - Runs quick or standard benchmark profiles.
  - Saves JSON reports to `artifacts/benchmarks/`.

- `compare_benchmarks.py`
  - Compares two benchmark reports.
  - Fails on regressions beyond a configurable threshold.

- `compare_text_checkpoints.py`
  - Compares two text checkpoints.
  - Fails on text-model loss or throughput regressions when requested.

- `compare_text_architectures.py`
  - Compares SSM and Transformer baselines under one protocol.
  - Reports parameter-budget match and refuses to frame smoke results as
    architecture superiority.

- `run_text_architecture_multiseed.py`
  - Runs repeated-seed SSM-vs-Transformer comparisons and aggregates the
    evidence into one JSON report.
  - Can require million-token train/eval positions and minimum real-corpus
    train/eval token counts before accepting the benchmark protocol.

- `evolve_ssm_formulas.py`
  - Evolves hidden-state update formulas over a constrained math DSL.
  - Benchmarks each formula with an inner weight-evolution loop and keeps an
    explicit non-AGI guardrail in the JSON artifact.

- `run_hypothesis_loop.py`
  - Validates LLM-style formula proposals, rejects invalid operations, and
    returns benchmark feedback plus prompt context without executing arbitrary
    generated Python.

- `coevolve_ssm_selfplay.py`
  - Coevolves runner/blocker populations with a payoff matrix and changing
    opponent pressure.
  - Emits role metadata, diversity/pressure summaries, throughput, and a
    non-AGI guardrail in the JSON artifact.

- `evolve_code_blocks.py`
  - Evolves safe register/predicate programs over cached CodePy blocks and
    reports train/holdout fitness, compression metrics, and rendered Python for
    inspection.

- `compare_code_block_library.py`
  - Tests whether accumulated CodePy macro-blocks speed up or shorten later
    program synthesis tasks.

- `code_block_skill_ladder.py`
  - Runs the find-compress-promote-reuse CodePy ladder and reports
    `reuse_frequency` plus downstream `fitness_gain`.

- `compare_code_block_hierarchy.py`
  - Measures the seven-task Stage 4 synthesis ladder and writes plot-ready
    base-vs-hierarchical metrics.

- `compare_code_block_stage4b.py`
  - Runs the multi-seed Stage 4B hierarchy-depth scaling test for success-rate
    curves.

- `analyze_code_block_pruning.py`
  - Analyzes Stage 4B macro-library noise and regularized pruning choices.

- `CODEPY_TURING_COMPLETENESS_PLAN.md`
  - Specifies the bounded loop/JUMP/memory path toward stateful algorithm
    synthesis.

- `evolve_code_tape.py`
  - Runs bounded-loop tape/pointer CodePy programs on stateful vector tasks.

- `compare_text_packages.py`
  - Compares two promoted text packages as release artifacts.
  - Fails on package-level loss, accuracy, or throughput regressions when
    requested.

- `compare_text_releases.py`
  - Compares active release pointers or registry entries, including memory
    diagnostics.
  - Fails on release-level loss, throughput, accuracy, or memory regressions
    when requested.

- `register_text_package.py`
  - Registers validated package releases in a JSON registry.
  - Records archive/checkpoint SHA256, benchmark metrics, and comparison
    verdicts.
  - Can attach validated prompt-suite memory diagnostics to each release entry.

- `list_text_releases.py`
  - Lists registered package releases and exports release leaderboards.
  - Includes memory overlap and cosine diagnostics when they are present.

- `select_text_release.py`
  - Selects and verifies the active package release pointer.
  - Preserves release memory diagnostics in the active pointer.

- `release_text_pipeline.py`
  - Orchestrates the complete release path and emits a stage-by-stage JSON
    report plus final active-release run metrics and selected-release memory
    suite metrics.
  - Supports `--include-evidence-audit` to emit a self-contained proof packet:
    release gates, Markdown report, evidence bundle, validated restore,
    tamper/replay-tested ledger, final evidence audit, audit validation, and
    audit comparisons.

- `run_text_checkpoint.py`
  - Evaluates saved checkpoints, promoted manifests, and active release
    pointers, emitting JSON run reports.

- `probe_text_memory.py`
  - Traces hidden memory over prompt tokens and exports JSON/CSV diagnostics for
    release inspection.

- `compare_text_memory.py`
  - Compares memory probe reports and can fail on hidden-state drift thresholds.

- `run_text_memory_suite.py`
  - Runs prompt batches and exports pairwise hidden-memory overlap/drift
    matrices.

- `validate_text_memory_suite.py`
  - Validates prompt-suite memory artifacts and can fail on missing prompt
    reports, malformed vector geometry, or configured drift/overlap thresholds.

- `benchmark_text_package.py`
  - Measures promoted package speed with warmed-up repeats.
  - Emits JSON with timing, throughput, integrity, quality, OOV, and sample
    fields.

- `run_text_experiment.py`
  - Runs a complete text experiment from corpus profile to checkpoint
    comparison.

- `list_text_experiments.py`
  - Builds JSON/CSV leaderboards from saved text experiment directories.

- `sweep_text_experiments.py`
  - Runs multiple held-out text experiments and builds sweep leaderboards.

- `promote_text_checkpoint.py`
  - Promotes a selected text checkpoint into a reusable artifact with manifest.

- `inspect_text_checkpoint.py`
  - Builds lightweight JSON and Markdown model cards for text checkpoints.

- `validate_text_package.py`
  - Validates promoted checkpoint integrity and required package metadata,
    including benchmark reports when requested.

- `bundle_text_package.py`
  - Creates portable `.tar.gz` archives of validated promoted text packages,
    including package benchmark reports.

- `unbundle_text_package.py`
  - Restores and validates package archives safely, including the restored
    benchmark report.

## Common Commands

```bash
make help
make verify
make benchmark
make compare
make profile-text-demo
make compare-text-demo
make compare-text-architectures-demo
make text-experiment-demo
make text-sweep-demo
make text-leaderboard
make promote-text-demo
make inspect-text-demo
make run-promoted-text-demo
make probe-text-memory-demo
make compare-text-memory-demo
make text-memory-suite-demo
make validate-text-memory-suite-demo
make benchmark-text-package-demo
make validate-text-package-demo
make bundle-text-package-demo
make unbundle-text-package-demo
make compare-text-package-demo
make register-text-package-demo
make text-release-leaderboard-demo
make select-text-release-demo
make run-current-text-release-demo
make build-release-dashboard-demo
make validate-release-dashboard-demo
make compare-release-dashboard-demo
make build-release-history-demo
make validate-release-history-demo
make analyze-release-history-demo
make summarize-release-gates-demo
make generate-release-report-demo
make bundle-release-evidence-demo
make validate-release-evidence-demo
make compare-release-evidence-demo
make unbundle-release-evidence-demo
make register-release-evidence-demo
make validate-release-evidence-ledger-demo
make tamper-release-evidence-ledger-demo
make replay-release-evidence-ledger-demo
make compare-release-evidence-ledger-demo
make summarize-release-evidence-audit-demo
make validate-release-evidence-audit-demo
make compare-release-evidence-audit-validation-demo
make compare-release-evidence-audit-demo
make compare-text-release-demo
make text-release-pipeline-demo
make validate-release-pipeline-demo
make compare-release-pipeline-validation-demo
make compare-release-pipeline-demo
make validate-release-pipeline-comparison-demo
make compare-release-pipeline-comparison-validation-demo
make text-run-demo
make export-demo
make memory-viewer-open
make memory-suite-viewer-open
make release-viewer-open
make release-dashboard-open
make release-dashboard-compare-open
make release-pipeline-validation-open
make release-pipeline-validation-compare-open
make release-pipeline-compare-open
make release-pipeline-comparison-validation-open
make release-pipeline-comparison-validation-compare-open
make release-pipeline-proof-cockpit-open
make release-evidence-compare-open
make release-evidence-ledger-open
make release-evidence-ledger-compare-open
make release-evidence-audit-open
make release-evidence-audit-validation-open
make release-evidence-audit-validation-compare-open
make release-evidence-audit-compare-open
make release-history-open
make release-history-analysis-open
make release-gates-open
make formula-evolution-demo
make code-block-evolution-demo
make hypothesis-loop-demo
make selfplay-demo
make viewer-open
```

## Program Synthesis Status

- Program Synthesis: PASS
- Program Compression: PASS
- Skill Promotion: PASS
- Hierarchical Reuse: PASS
- Hierarchical Scaling: EARLY PASS, supported by the Stage 4 one-seed ladder
  and Stage 4B 100-seed success-rate curves. Current Stage 4B results:
  `max_abs_x_y` 97% at depth 0 to 100% at depth 2+, `clamp_abs_01` 84% at
  depth 0 to 100% at depth 3+, and `piecewise_max_abs_x_y` 0% at depth 0 to
  51% at depth 5. The curve is not strictly monotonic, so more macros can also
  add search noise.
- General Algorithmic Scaling: not proven; the first bounded tape executor now
  validates LOOP/JUMP/memory state geometry on `sum4` and `count_positive4`, and
  shaped clean evolution autonomously found both `sum4` and `count_positive4`
  without reference injection. The vNext trace-prior path logs all evaluated
  tape traces, trains lightweight success priors, uses them in guided beam
  search, and promotes zero-error traces into macros. `argmax_index4` is now
  expressible and reference-verified in the expanded DSL. The Phase 4 trajectory
  layer computes execution signatures, enforces beam diversity, exports
  contrastive candidates, trains a strict trace-only prior that distinguishes
  exact argmax traces from value-like near-misses, runs equal-budget
  trace-guided beam comparisons, and reconstructs seeded target traces from a
  latent trace graph. Trace guidance alone improved the best autonomous
  `argmax_index4` MSE under b24/b48/b160 budgets but remained a near-miss.
  Prefix-value rollouts are implemented and measured, and the new ordered
  prefix-stage lane gives delayed-reward structural prefixes a small hard
  survival quota. With `--prefix-stage-lane-fraction 0.125`, autonomous
  `argmax_index4` now reaches zero train/holdout error without reference-prefix
  injection. Robust multi-seed branch discovery, calibrated trace-prior/value
  scoring, and broader scaling across harder stateful tasks remain pending.

## Latest Verified Snapshot

`python3 verify_system.py` passed with:

```text
PASS compile: 66 scripts
PASS agent-runtime: goal 1.1283->1.0847, threat 0.8230->0.8635
PASS text-checkpoint: loss 4.5040->3.0026, reload 3.0026
PASS text-runner: loss 3.0026, accuracy 0.2344
PASS text-profiler: tokens 140, vocab 90, seq_len 16
PASS text-ingest-bpe: tokens 115, merges 45, oov 0.0000
PASS text-compare: loss 3.8586->3.0026, delta -0.8560
PASS text-architecture-compare: winner ssm, ssm_loss 4.3288, transformer_loss 5.1558, param_ratio 0.964
PASS text-experiment: shape 4x96, eval 28, winner baseline
PASS text-leaderboard: runs 1, eval 28, winner baseline
PASS text-sweep: runs 2, best_loss 4.3538
PASS text-promote: role best, smoke_loss 3.7914
PASS text-inspect: params 7,837, bytes 30.61 KiB
PASS text-manifest-run: loss 3.7914, sample_len 31
PASS text-package-benchmark: best 580,498 tokens/s, loss 3.7914
PASS text-validate: bytes 92700, sha256 e83bbba1995f
PASS text-bundle: files 7, bytes 95504
PASS text-unbundle: files 7, sha256 ad5f6673ced8
PASS text-package-compare: loss_delta 0.0000, same_sha True
PASS text-package-register: releases 1, sha256 e83bbba1995f
PASS text-release-leaderboard: releases 1, best_loss 3.7914
PASS text-release-select: rank 1, sha256 e83bbba1995f
PASS text-release-run: loss 3.7914, sample_len 31
PASS text-memory-probe: steps 4, final_norm 1.2051, top river
PASS text-memory-compare: steps 4, drift 0.0000, overlap 1.0000
PASS text-memory-suite: prompts 3, pairs 3, mean_overlap 0.5250
PASS text-memory-suite-validate: prompts 3, pairs 3, mean_overlap 0.5250
PASS text-release-pipeline: rank 1, loss 3.7914, memory_overlap 0.5250, evidence a4ab271ce72d
PASS text-release-pipeline-validate: stages 35, artifacts 19, evidence_checks 10
PASS text-release-pipeline-validation-compare: same_valid True, artifacts_delta 0, errors 0
PASS text-release-pipeline-compare: stage_delta 0, loss_delta 0.0000, memory_overlap_delta 0.0000
PASS text-release-pipeline-compare-validate: stages 35->35, failures 0
PASS text-release-pipeline-comparison-validation-compare: same_valid True, failures 0, errors 0
PASS text-release-compare: loss_delta 0.0000, memory_overlap_delta 0.0000
PASS text-release-dashboard-json: rank 1, loss 3.7914, checks 11
PASS text-release-dashboard-validate: checks 11, artifacts 6, mean_overlap 0.5250
PASS text-release-dashboard-compare: loss_delta 0.0000, memory_overlap_delta 0.0000
PASS text-release-history-json: releases 1, best_loss 3.7914
PASS text-release-history-validate: releases 1, artifacts 1
PASS text-release-history-analyze: baseline previous, loss_delta 0.0000
PASS text-release-gates: gates 4, failed 0
PASS text-release-report: markdown text_release_report.md
PASS text-release-evidence: files 16, bytes 15061
PASS text-release-evidence-validate: files 16, members 17
PASS text-release-evidence-unbundle: files 16, members 17
PASS text-release-evidence-compare: same_sha True, files 16
PASS text-release-evidence-ledger: entries 1, head 05bc825e9214
PASS text-release-evidence-ledger-validate: entries 1, artifacts 5
PASS text-release-evidence-ledger-tamper: mode archive_sha256, detected True
PASS text-release-evidence-ledger-replay: actions inserted,updated,inserted, entries 2
PASS text-release-evidence-ledger-compare: self_same True, growth_delta 1
PASS text-release-evidence-audit: checks 10, replay 1->2
PASS text-release-evidence-audit-validate: checks 10, sources 10
PASS text-release-evidence-audit-validation-compare: same_valid True, errors 0
PASS text-release-evidence-audit-compare: same_checks True, failures 0
PASS evolution-replay: evolved reward 2.3096, replay reward 2.0696, goal 1.2012->0.9981
PASS formula-evolution: best_reward 2.9317, formulas 6, best mem=relu|inp=identity|combine=max|final=tanh
PASS hypothesis-loop: accepted 1, rejected 1, best valid_relu_max
PASS code-block-evolution: target square_minus_one, train_mse 0.00e+00, active 6
PASS selfplay-coevolution: runner_best 1.0211, blocker_best 3.4002, history 3
PASS trajectory-export: rows 209, goal 0.9062->0.8419
PASS trajectory-viewer: static HTML hooks present
PASS text-memory-viewer: static HTML hooks present
PASS text-memory-suite-viewer: static HTML hooks present
PASS text-release-viewer: static HTML hooks present
PASS text-release-dashboard: static HTML hooks present
PASS text-release-dashboard-compare-viewer: static HTML hooks present
PASS text-release-evidence-compare-viewer: static HTML hooks present
PASS text-release-evidence-ledger-viewer: static HTML hooks present
PASS text-release-evidence-ledger-compare-viewer: static HTML hooks present
PASS text-release-evidence-audit-viewer: static HTML hooks present
PASS text-release-evidence-audit-validation-viewer: static HTML hooks present
PASS text-release-evidence-audit-validation-compare-viewer: static HTML hooks present
PASS text-release-evidence-audit-compare-viewer: static HTML hooks present
PASS text-release-history-viewer: static HTML hooks present
PASS text-release-history-analysis-viewer: static HTML hooks present
PASS text-release-gates-viewer: static HTML hooks present
PASS text-release-pipeline-validation-viewer: static HTML hooks present
PASS text-release-pipeline-validation-compare-viewer: static HTML hooks present
PASS text-release-pipeline-compare-viewer: static HTML hooks present
PASS text-release-pipeline-comparison-validation-viewer: static HTML hooks present
PASS text-release-pipeline-comparison-validation-compare-viewer: static HTML hooks present
PASS text-release-pipeline-proof-cockpit-viewer: static HTML hooks present
SYSTEM OK
```

Latest quick benchmark artifacts include:

- `artifacts/benchmarks/quick_20260620T171809Z.json`
- `artifacts/benchmarks/quick_compare_candidate.json`

## Latest Architecture Benchmark

The first larger repeated-seed SSM-vs-Transformer run has been executed on the
50MB Gutenberg BPE corpus:

- Summary: `artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million/summary.json`
- Per-seed reports:
  - `artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million/seed_11.json`
  - `artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million/seed_17.json`
  - `artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million/seed_23.json`
- Protocol evidence:
  - `3` repeated seeds.
  - `1,032,192` train positions per epoch.
  - `1,032,192` eval positions per repeat.
  - `18,274,478` BPE train tokens and `4,389,724` BPE eval tokens.
  - Parameter budget ratio Transformer/SSM: `1.014`.
- Result:
  - Transformer won held-out loss on `3/3` seeds.
  - Mean SSM loss: `5.6696`; mean Transformer loss: `5.5810`.
  - Mean SSM accuracy: `0.0602`; mean Transformer accuracy: `0.0693`.
  - Mean SSM eval throughput: `895,146 tok/s`.
  - Mean Transformer eval throughput: `792,408 tok/s`.
  - Transformer attention-score memory was `341.3x` SSM state memory for this
    protocol.
  - SSM effective context was `8,192` tokens versus Transformer `128` tokens
    (`64x` context ratio).

Interpretation: this bounded benchmark supports a nuanced result, not a blanket
victory claim. Under this one-epoch, parameter-matched protocol the Transformer
learned the held-out distribution better, while the SSM retained the speed and
state-memory/context advantages that motivate the line of research.

Follow-up SSM architecture probes now cover:

- `selective`, `selective-lite`, and `write-gated` input-dependent memory
  updates.
- `skip`, `skip-write-gated`, `skip-selective-lite`, `tied-skip`,
  `lowrank-skip`, and `conv-skip` direct shortcut variants.
- `state-mix`, `skip-state-mix`, and `conv-state-mix` hidden-state mixing
  variants.
- `token-memory`, a token-gated slow memory rail in embedding space for
  associative/delayed recall.
- `conv-token-memory`, a transfer probe that combines the best local
  `conv-skip` shortcut with the token-memory rail.
- `kv-memory`, a key-value memory matrix for addressable `name -> value`
  binding.

Key result: `skip` reduced the 3-seed SSM mean loss from `5.6696` to `5.5591`,
but still lost the mean comparison (`5.5211` Transformer) and won only `1/3`
seeds. `conv-skip` improved absolute SSM loss further to `5.5403`, but the
matched Transformer also grew stronger and won `3/3` seeds with mean loss
`5.4885`. The best current conclusion is that local shortcut channels materially
improve SSM quality, while selective gates and simple state mixing did not yet
produce a robust natural-language LM win.

The first direct attempt to transfer learned token memory back into the
Gutenberg/BPE LM benchmark is `conv-token-memory`. It initializes both the
`conv-skip` local shortcut and the token-gated slow memory rail:

- Smoke artifact:
  `artifacts/text_architecture_comparison_conv-token-memory_smoke.json`
  - Built-in tiny text corpus.
  - SSM loss `4.2691`; Transformer loss `4.8084`.
  - This only proves the variant runs and can help on a toy corpus.
- BPE transfer probe:
  `artifacts/text_architecture_comparison_conv_token_memory_bpe_probe.json`
  - Gutenberg 50MB corpus, BPE tokenizer, seed `11`.
  - `64` streams, `4,096` tokens/stream, sequence length `128`.
  - SSM `conv-token-memory` held-out loss: `5.9292`.
  - Matched Transformer held-out loss: `5.7013`.
  - Diagnostics after training: mean write gate `0.5012`, mean read gate
    `0.5072`, memory logit scale `1.0292`.
- Same-protocol `conv-skip` control:
  `artifacts/text_architecture_comparison_conv_skip_bpe_probe.json`
  - SSM `conv-skip` held-out loss: `5.8554`.
  - Matched Transformer held-out loss: `5.6932`.
- Conservative closed-gate transfer probe:
  `artifacts/text_architecture_comparison_conv_token_memory_closed_bpe_probe.json`
  - SSM `conv-token-memory-closed` held-out loss: `5.8575`.
  - Matched Transformer held-out loss: `5.7013`.
  - Diagnostics after training: mean write gate `0.0185`, mean read gate
    `0.0187`, memory logit scale `0.3390`.
  - This mostly removes the naive memory regression but does not beat the
    simpler `conv-skip` baseline.
- Sparse/L1 gate transfer probe:
  `artifacts/text_architecture_comparison_conv_token_memory_sparse_bpe_probe.json`
  - SSM `conv-token-memory-sparse` held-out loss: `5.8575`.
  - Matched Transformer held-out loss: `5.7013`.
  - Training used `--memory-gate-l1 0.01`.
  - Diagnostics after training: mean write gate `0.0169`, mean read gate
    `0.0171`, memory logit scale `0.3387`.
  - The penalty keeps memory shut and gives no useful advantage over the
    closed-gate probe or `conv-skip`.

Interpretation: the naive memory rail does not transfer cleanly to
natural-language BPE LM loss. It worsens the SSM relative to the simpler
`conv-skip` baseline on this probe. Closed gates prevent most of that harm but
do not create a useful LM advantage; diagnostics show the closed variant mostly
keeps memory disabled. L1 sparsity alone reinforces that disabled state. The
next LM-quality route should therefore provide a positive utility signal for
opening memory, such as an auxiliary recall objective or token classes where
memory use is explicitly rewarded, rather than merely penalizing open gates.

That auxiliary route has now been probed inside the same Gutenberg/BPE
architecture harness. The held-out evaluation remains the ordinary
natural-language LM loss; the auxiliary updates affect only SSM training and are
therefore a memory-curriculum experiment, not an apples-to-apples pure-LM
objective comparison.

- Filler-dominated delayed-key probe:
  `artifacts/text_architecture_comparison_conv_token_memory_aux_bpe_probe.json`
  - SSM `conv-token-memory-sparse` held-out loss: `5.8749`.
  - Matched Transformer held-out loss: `5.7013`.
  - Auxiliary accuracy reported `0.9567`, but this was inflated by filler-token
    positions; gates stayed mostly shut (write `0.0180`, read `0.0183`).
- Masked delayed-key correction:
  `artifacts/text_architecture_comparison_conv_token_memory_aux_masked_bpe_probe.json`
  - Auxiliary loss is now measured only on query->key positions.
  - SSM held-out loss: `5.8966`; Transformer loss: `5.7013`.
  - Auxiliary recall accuracy: `0.0000`; gates stayed shut.
- KV delayed-key probe:
  `artifacts/text_architecture_comparison_kv_aux_masked_bpe_probe.json`
  - SSM `kv-memory` held-out loss: `5.8978`; matched Transformer loss:
    `5.7942`.
  - Auxiliary recall accuracy: `0.0469`.
- KV assignment-memory curriculum:
  `artifacts/text_architecture_comparison_kv_assignment_aux_init_bpe_probe.json`
  - Uses an assignment-style auxiliary task (`name = value ... name ? value`)
    with structural memory-gate initialization (`±8` gate logits and memory
    logit scale `32`).
  - Auxiliary recall reaches `1.0000` accuracy after `31` SSM-only updates.
  - SSM held-out Gutenberg/BPE LM loss: `5.8927`.
  - Matched Transformer held-out loss: `5.7942`.
  - Diagnostics show sparse, specialized KV gates: key/value/deref/erase means
    about `0.074`, query/write-context means about `0.0015`, and maximum gates
    near `0.9997`.

Interpretation: a positive memory curriculum can make the KV rail solve
binding recall inside the LM harness, but it still does not beat the matched
Transformer on ordinary natural-language BPE loss. This narrows the route
forward: the memory mechanism is useful for code-like bindings, while the
natural-language objective still needs either better routing from real tokens
to memory events, a mixed code/text corpus where bindings matter in-distribution,
or an architecture that lets the memory rail help local next-token prediction
instead of acting as an auxiliary skill.

The new long-context bypass harness is `compare_long_context_recall.py`.
It creates a synthetic delayed-recall task where a key token appears before a
filler span, then a later query must predict that key. This isolates the SSM
advantage when the key is outside a Transformer's configured attention cap:

- Bypass artifact:
  `artifacts/long_context_recall/static_delay32_ctx8_epochs60.json`
- Protocol:
  - `1,024` train records and `512` eval records.
  - `8` possible keys.
  - Delay `32`; Transformer context cap `8`.
  - Parameter budget ratio Transformer/SSM: `1.023`.
- Result:
  - SSM held-out recall accuracy: `1.0000`.
  - Transformer held-out recall accuracy: `0.1328`.
  - SSM eval speed: `176,742 records/s`.
  - Transformer eval speed: `102,272 records/s`.

Control artifact:
`artifacts/long_context_recall/static_delay32_ctx64_epochs60_control.json`.
With Transformer context `64`, the same Transformer reaches `1.0000` held-out
recall, so the bypass result is specifically about being outside the attention
window.

The plain tanh SSM does not reliably scale this recall trick to delay `64`/`96`.
The new `token-memory` variant is the first working learned associative-memory
route:

- Main artifact:
  `artifacts/long_context_recall/demo.json`
- Protocol:
  - `1,024` train records and `512` eval records.
  - `16` possible keys.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `0.995`.
  - `40` epochs.
  - No structural key/query memory initialization.
- Result:
  - SSM held-out recall accuracy: `1.0000`.
  - Transformer held-out recall accuracy: `0.0469`.
  - SSM eval speed: `49,716 records/s`.
  - Transformer eval speed: `23,323 records/s`.

Repeated-seed learned-gate artifacts:

- `artifacts/long_context_recall/token_memory_uninit_delay96_ctx32_seed11_epochs120.json`
- `artifacts/long_context_recall/token_memory_uninit_delay96_ctx32_seed17_epochs120.json`
- `artifacts/long_context_recall/token_memory_uninit_delay96_ctx32_seed23_epochs120.json`

All three reached `1.0000` SSM held-out recall without `--init-recall-memory`.
The shorter `40`-epoch probes for seeds `11`, `17`, and `23` also reached
`1.0000` recall:

- `artifacts/long_context_recall/token_memory_uninit_delay96_ctx32_epochs40_probe.json`
- `artifacts/long_context_recall/token_memory_uninit_delay96_ctx32_seed17_epochs40.json`
- `artifacts/long_context_recall/token_memory_uninit_delay96_ctx32_seed23_epochs40.json`

Harder artifact:
`artifacts/long_context_recall/token_memory_delay192_ctx64_epochs1.json`.
With structural recall-memory initialization, delay `192`, and Transformer
context cap `64`, `token-memory` reaches `1.0000` held-out recall while the
capped Transformer reaches `0.0859`. This remains a diagnostic stress case; the
learned-gate repeated-seed result above is the stronger current claim.

Control artifact:
`artifacts/long_context_recall/token_memory_uninit_delay96_ctx128_control_epochs120.json`.
With Transformer context `128`, the Transformer also reaches `1.0000`, proving
again that the bypass win is caused by the key being outside the attention cap.
The next research step is to transfer the learned token-memory behavior from
synthetic delayed recall into natural text/code structure.

The first transfer probe is a code-like assignment recall task in the same
harness (`--task assignment`). Records look like `name = value ; other = value ;
... name ?`, and the target is the original value. This is harder than delayed
key copy because the model must preserve a name-to-value binding rather than a
single token:

- Artifact:
  `artifacts/long_context_recall/assignment_token_memory_delay96_ctx32_seed11_epochs80.json`
- Protocol:
  - `2,048` train records and `1,024` eval records.
  - `16` names and `16` values.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `1.040`.
- Result:
  - SSM held-out recall accuracy: `0.5176`.
  - Transformer held-out recall accuracy: `0.0791`.
  - SSM eval speed: `38,936 records/s`.
  - Transformer eval speed: `25,223 records/s`.

Longer training (`4,096` train records, `200` epochs) reached `0.5293` SSM
held-out recall, so the current token-memory rail transfers beyond pure key
copy but plateaus well below perfect binding. The next architecture should add
explicit key-value/addressed memory rather than only a single slow token trace.

That next layer now exists as `kv-memory`:

- Main artifact:
  `artifacts/long_context_recall/assignment_demo.json`
- Protocol:
  - `2,048` train records and `1,024` eval records.
  - `16` names and `16` values.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `1.037`.
  - `40` epochs.
  - No structural key/value/query initialization.
- Result:
  - SSM held-out recall accuracy: `0.9990` in the current demo.
  - Transformer held-out recall accuracy: `0.0742`.
  - SSM eval speed: `4,547 records/s`.
  - Transformer eval speed: `28,376 records/s`.

Repeated-seed KV artifacts:

- `artifacts/long_context_recall/assignment_kv_memory_uninit_delay96_ctx32_epochs40.json`
- `artifacts/long_context_recall/assignment_kv_memory_uninit_delay96_ctx32_seed17_epochs40.json`
- `artifacts/long_context_recall/assignment_kv_memory_uninit_delay96_ctx32_seed23_epochs40.json`

All three reached `1.0000` SSM held-out recall. Control artifact
`artifacts/long_context_recall/assignment_kv_memory_uninit_delay96_ctx128_control_epochs40.json`
shows that with Transformer context `128`, the Transformer also reaches
`1.0000`; the assignment win is therefore specifically caused by the key/value
binding being outside the capped attention window.

The harder multi-binding probe is now implemented as `--task multi-assignment`.
Each record contains four independent `name = value ;` bindings and later asks
for one queried name. This checks whether the memory can preserve several
addressable bindings at once, not just a single pair:

- Main artifacts:
  - `artifacts/long_context_recall/multi_assignment_kv_memory_uninit_bind4_delay96_ctx32_seed11_epochs40.json`
  - `artifacts/long_context_recall/multi_assignment_kv_memory_uninit_bind4_delay96_ctx32_seed17_epochs40.json`
  - `artifacts/long_context_recall/multi_assignment_kv_memory_uninit_bind4_delay96_ctx32_seed23_epochs40.json`
- Protocol:
  - `2,048` train records and `1,024` eval records.
  - `16` names, `16` values, and `4` bindings per record.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `1.037`.
  - `40` epochs.
- Result:
  - SSM held-out recall accuracy: `1.0000` on all three seeds.
  - Transformer held-out recall accuracy: `0.0615`, `0.0586`, and `0.0508`.

This is stronger than the one-binding assignment probe, but it is not the same
kind of clean window-only control. Full-window controls at context `128` did not
make the small matched Transformer generalize on held-out records:

- One-layer, `120` epochs:
  `artifacts/long_context_recall/multi_assignment_kv_memory_uninit_bind4_delay96_ctx128_control_epochs120.json`
  reached Transformer train recall `0.9453` but held-out recall `0.2695`.
- Two-layer, matched-parameter, `120` epochs:
  `artifacts/long_context_recall/multi_assignment_kv_memory_uninit_bind4_delay96_ctx128_layers2_control_epochs120.json`
  reached Transformer train recall `0.8750` but held-out recall `0.2617`.

Interpretation: explicit recurrent KV memory is now solving a small
algorithmic multi-binding task that the matched small Transformer overfits or
fails to generalize under this protocol. This is promising route-around evidence
for structured long-context/code-like memory, not a blanket natural-language
Transformer defeat.

The System-1/System-2 routing probe is now implemented as `sparse-kv-memory`.
It adds a dynamic scalar KV read gate from the fast SSM state and current token
for KV reads. `compare_long_context_recall.py` now supports three smooth
regularizers for that gate: L1 read cost, quadratic target sparsity, and entropy
sharpening. This tests whether the model can reduce expensive memory reads while
preserving binding recall.

- Main smoke artifact:
  `artifacts/long_context_recall/sparse_kv_multi_assignment_curriculum_target_entropy_smoke.json`
- Query-margin hardening artifact:
  `artifacts/long_context_recall/sparse_kv_multi_assignment_query_margin_smoke.json`
- Hard-inference smoke artifact:
  `artifacts/long_context_recall/sparse_kv_multi_assignment_query_margin_hard_eval_smoke.json`
- Six-seed hard-inference summary:
  `artifacts/long_context_recall_multiseed/sparse_kv_margin_hard_seed6/summary.json`
- Protocol:
  - `512` train records and `256` eval records.
  - `16` names, `16` values, `4` bindings, and `2` query reads per record.
  - Delay `96`; Transformer context cap `32`.
  - `sparse-kv-memory` with structural assignment KV initialization.
  - `--kv-read-gate-l1-start 0.0`, linearly ramped to
    `--kv-read-gate-l1 0.5` over `6` epochs.
  - `--kv-read-gate-target 0.02`, `--kv-read-gate-target-weight 5.0`.
  - `--kv-read-gate-entropy-weight 0.01`.
  - `12` epochs.
- Result:
  - SSM held-out recall accuracy: `1.0000`.
  - Transformer held-out recall accuracy: `0.0742`.
  - KV read gate mean: `0.0117`.
  - KV read gate on query tokens: `0.3325`.
  - KV read gate on non-query tokens: `0.0084`.
  - Query/non-query gate ratio: about `39.7x`.
  - KV read gate entropy mean: `0.0505`.
  - Fraction of all KV reads below `0.01`: `0.9159`.
  - Fraction of non-query KV reads below `0.01`: `0.9254`.
  - Fraction of query KV reads above `0.5`: `0.0000`.
  - Fraction of query KV reads above `0.9`: `0.0000`.

Query-margin hardening result:

- Adds `--kv-read-gate-query-margin 0.9` and
  `--kv-read-gate-query-margin-weight 2.0` on top of the same L1 curriculum,
  target sparsity, and entropy penalty.
- SSM held-out recall accuracy: `1.0000`.
- Transformer held-out recall accuracy: `0.0742`.
- KV read gate mean: `0.0149`.
- KV read gate on query tokens: `0.9358`.
- KV read gate on non-query tokens: `0.0053`.
- Query/non-query gate ratio: about `175.5x`.
- KV read gate entropy mean: `0.0251`.
- Fraction of all KV reads below `0.01`: `0.9561`.
- Fraction of non-query KV reads below `0.01`: `0.9660`.
- Fraction of query KV reads above `0.5`: `1.0000`.
- Fraction of query KV reads above `0.9`: `1.0000`.

Hard-inference result:

- Uses the same continuous margin-trained model, then evaluates an additional
  path with `--kv-read-gate-hard-eval-threshold 0.5`, so KV reads are actually
  thresholded to `0` or `1` inside the forward pass.
- Single-seed hard eval:
  - SSM hard held-out recall accuracy: `1.0000`.
  - Hard KV read gate mean: `0.0109`.
  - Hard KV read gate on query tokens: `1.0000`.
  - Hard KV read gate on non-query tokens: `0.0006`.
  - Fraction of hard non-query KV reads below `0.01`: `0.9994`.
- Six-seed stability run, seeds `11,17,23,41,73,101`:
  - Protocol valid: `true`; minimum query positions per seed: `256`.
  - Soft SSM held-out recall accuracy: mean/min/max `1.0000/1.0000/1.0000`.
  - Hard-inference SSM held-out recall accuracy:
    mean/min/max `1.0000/1.0000/1.0000`.
  - Transformer held-out recall accuracy: mean `0.0592`.
  - Soft KV read gate query mean: mean `0.9320`, min `0.9258`.
  - Soft fraction of query reads above `0.5`: mean/min `1.0000/1.0000`.
  - Soft fraction of query reads above `0.9`: mean `0.9798`, min `0.8789`.
  - Hard KV read gate query mean: mean/min `1.0000/1.0000`.
  - Hard KV read gate non-query mean: mean `0.0002`, max `0.0007`.
  - Hard fraction of non-query reads below `0.01`: mean `0.9998`, min `0.9993`.

Controls:

- With L1 curriculum only, recall remains `1.0000`, mean gate is `0.0192`,
  query gate is `0.3649`, non-query gate is `0.0156`, for about `23.4x`
  query/non-query separation.
- Target sparsity alone hits the target region but does not improve separation:
  target `0.02`, weight `10.0` gives mean `0.0221`, query `0.3035`,
  non-query `0.0192`, about `15.8x`.
- Entropy sharpening alone on top of target sparsity lowers the mean but also
  lowers the query signal: entropy weight `0.05` gives mean `0.0184`, query
  `0.2845`, non-query `0.0157`, about `18.2x`.
- Constant L1 controls keep recall `1.0000` but are less efficient:
  - `--kv-read-gate-l1 0.3`: mean `0.0333`, query `0.3534`, non-query `0.0300`.
  - Constant `--kv-read-gate-l1 0.5`: mean `0.0244`, query `0.3349`,
    non-query `0.0211`.
  - `--kv-read-gate-l1 1.0`: mean `0.0148`, query `0.3172`, non-query `0.0116`.
- With the previous token-gate/product formulation, query/non-query separation
  looked much sharper, but that evidence mostly reflected the hand-initialized
  token-level query gate rather than a learned dynamic state gate.
- Without structural assignment KV initialization and no L1, the gate opens
  broadly (mean `0.4882`, query `0.6081`, non-query `0.4870`) but recall stays
  low (`0.0859`).
- Without structural initialization and with `--kv-read-gate-l1 0.1`, the gate
  closes almost everywhere (mean `0.0017`, query `0.0007`) and recall stays low
  (`0.0781`).

Interpretation: the combination of L1 curriculum, target sparsity, and a small
entropy penalty currently gives the best smooth routing tradeoff: perfect
recall, lower background reads, and much stronger signal/noise than L1
curriculum alone. The new threshold diagnostics show that this comes mainly
from pushing background reads almost to zero; the query read remains soft rather
than crossing a hard open threshold. The dynamic sparse read gate is therefore
not yet a clean near-binary "open only on ?" mechanism.

The query-margin probe changes that specific failure mode: a direct squared
hinge penalty on query-token gates pushes every held-out query read above `0.5`
across six seeds, and near/above `0.9` on average, while keeping background
reads low and preserving perfect recall. The hard-inference probe is stronger
than the soft threshold diagnostics: after continuous training, replacing the
read gate with `gate = (gate > 0.5)` at eval time preserves perfect recall on
all six seeds and makes the executed read policy effectively discrete. This is
the pragmatic hard-routing path, not yet the fundamental Straight-Through/Gumbel
path. The remaining unresolved problem is learning syntax-to-memory roles from
scratch without structural initialization, and then testing whether a true
discrete estimator can keep the same stability.

The harness now also supports `--task multi-query-assignment`, where one record
contains several bindings and several later `name ?` reads. Evaluation was
fixed to aggregate every query-token position, not only the final position:

- Main artifact:
  `artifacts/long_context_recall/multi_query_assignment_demo.json`
- Protocol:
  - `2,048` train records and `1,024` eval records.
  - `16` names, `16` values, `4` bindings, and `2` query reads per record.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `1.049`.
  - `40` epochs.
- Result:
  - SSM held-out recall accuracy across `2,048` query positions: `1.0000`.
  - Transformer held-out recall accuracy: `0.0698`.
  - SSM eval speed: `3,822 records/s`.
  - Transformer eval speed: `22,981 records/s`.

Verifier artifact:
`artifacts/verify/multi_query_assignment_recall_verify.json`
uses a smaller `512`/`256` protocol and requires SSM recall `>=0.99`, a large
margin, and exact query-position counting.

The KV writer now has a learned previous-token context gate
(`kv_write_context_token`) so value writes can depend on syntax. The first probe
for this is `--task assignment-distractor`: after a correct assignment and a
first read, an unrelated wrong value literal appears before a second read. The
literal must not overwrite the original binding:

- Main artifact:
  `artifacts/long_context_recall/assignment_distractor_demo.json`
- Protocol:
  - `2,048` train records and `1,024` eval records.
  - `16` names and `16` values.
  - Two query reads per record; the second follows a wrong non-assignment value
    literal.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `1.046`.
  - `40` epochs.
- Result:
  - SSM held-out recall accuracy across `2,048` query positions: `1.0000`.
  - Transformer held-out recall accuracy: `0.5317`.
  - SSM eval speed: `3,726 records/s`.
  - Transformer eval speed: `22,233 records/s`.

Interpretation: the recurrent KV path has moved from token-triggered binding
toward syntax-conditioned binding. This is important for code-like data because
not every value literal should mutate memory. A smaller `512`/`256` verifier
probe was unstable (`0.9023` SSM recall), so this remains a full demo artifact
rather than an always-on smoke check.

The next semantic step is alias/copy assignment, implemented as
`--task assignment-alias`: `source = value ; alias = source ; ... alias ?`.
`kv-memory` now has a learned dereference gate (`kv_deref_token`) for RHS
identifier tokens, so a name on the right-hand side can be read through the KV
matrix and the retrieved value written under the left-hand-side alias:

- Main artifact:
  `artifacts/long_context_recall/assignment_alias_demo.json`
- Protocol:
  - `2,048` train records and `1,024` eval records.
  - `16` names and `16` values.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `1.028`.
  - `40` epochs.
  - No structural key/value/query/deref initialization.
- Result:
  - SSM held-out recall accuracy: `1.0000`.
  - Transformer held-out recall accuracy: `0.0664`.
  - SSM eval speed: `4,118 records/s`.
  - Transformer eval speed: `25,347 records/s`.

Interpretation: this is the first synthetic-code probe where the recurrent KV
model performs a tiny semantic dereference rather than only storing literal
assignments. It is still synthetic and expensive on CPU, but it is meaningfully
closer to code execution than delayed copy.

The strongest synthetic-code transfer probe is now `--task mixed-code`. It
procedurally simulates a tiny variable environment containing literal
assignments, alias assignments, updates, a non-assignment value literal, and
two later reads. Targets are generated from the simulated environment rather
than from one fixed hand-written template:

- Main artifact:
  `artifacts/long_context_recall/mixed_code_demo.json`
- Protocol:
  - `2,048` train records and `1,024` eval records.
  - `16` names and `16` values.
  - `2` query reads per record.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `1.043`.
  - `40` epochs.
  - No structural KV initialization.
- Result:
  - SSM held-out recall accuracy across `2,048` query positions: `0.9995`.
  - Transformer held-out recall accuracy: `0.0889`.
  - SSM eval speed: `4,024 records/s`.
  - Transformer eval speed: `22,662 records/s`.

Interpretation: this is the best current route-around evidence for structured
code-like long-context memory. It combines syntax-conditioned writes, alias
dereference, updates, and multiple reads in one record. It is still synthetic
and does not overturn the Gutenberg natural-language result; the next milestone
is a small generated or real code corpus with the same semantics.

Repeated-seed mixed-code evidence is now available:

- Summary:
  `artifacts/long_context_recall_multiseed/mixed_code_query2_delay96_ctx32/summary.json`
- Seed artifacts:
  - `artifacts/long_context_recall_multiseed/mixed_code_query2_delay96_ctx32/seed_11.json`
  - `artifacts/long_context_recall_multiseed/mixed_code_query2_delay96_ctx32/seed_17.json`
  - `artifacts/long_context_recall_multiseed/mixed_code_query2_delay96_ctx32/seed_23.json`
- Result:
  - Winner count by held-out recall: SSM `3/3`.
  - Mean SSM held-out recall: `0.9998`.
  - Mean Transformer held-out recall: `0.0822`.
  - Mean SSM minus Transformer recall: `0.9176`.
  - Protocol evidence valid: `True`.

This upgrades mixed-code from a single demo to repeated-seed evidence while
preserving the same guardrail: it is strong synthetic-code evidence, not a
general natural-language victory claim.

Full-context mixed-code controls clarify the mechanism. On seed `11`, the
Transformer was given context `128`, which covers the full sequence length
(`103`) and therefore removes the short-window limitation:

- `artifacts/long_context_recall/mixed_code_kv_memory_uninit_query2_delay96_ctx128_control_seed11_epochs40.json`
  - Transformer train recall: `0.7070`.
  - Transformer held-out recall: `0.4116`.
  - SSM held-out recall: `0.9995`.
- `artifacts/long_context_recall/mixed_code_kv_memory_uninit_query2_delay96_ctx128_control_seed11_epochs120.json`
  - Transformer train recall: `0.8398`.
  - Transformer held-out recall: `0.4229`.
  - SSM held-out recall: `0.9995`.

Interpretation: for mixed-code, the capped result is not merely a short-attention
window artifact. Even with full context and longer training, the matched tiny
Transformer overfits or fails to generalize the mini-executor rule, while the
explicit recurrent KV path generalizes it. This remains a bounded synthetic
algorithmic-memory claim, not a broad natural-language LM win.

The next data step is `--task generated-python-code`, a generated Python-like
corpus with explicit syntax tokens around the same interpreter-checked
environment. Records include a `def f():` header, literal assignments, alias
assignments, updates, `print(...)` and `return(...)` literal noise, and two
later `print(name ? value)` reads. Targets are still computed from the simulated
environment rather than from a fixed answer template:

- Main artifact:
  `artifacts/long_context_recall/generated_python_code_demo.json`
- Protocol:
  - `2,048` train records and `1,024` eval records.
  - `16` names, `16` values, and `7` Python-like syntax tokens.
  - `2` query reads per record.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `0.991`.
  - `40` epochs.
- Result:
  - SSM held-out recall accuracy across `2,048` query positions: `1.0000`.
  - Transformer held-out recall accuracy: `0.0693`.
  - SSM eval speed: `4,288 records/s`.
  - Transformer eval speed: `26,866 records/s`.

Repeated-seed generated Python-like evidence is now available:

- Summary:
  `artifacts/long_context_recall_multiseed/generated_python_code_query2_delay96_ctx32/summary.json`
- Seed artifacts:
  - `artifacts/long_context_recall_multiseed/generated_python_code_query2_delay96_ctx32/seed_11.json`
  - `artifacts/long_context_recall_multiseed/generated_python_code_query2_delay96_ctx32/seed_17.json`
  - `artifacts/long_context_recall_multiseed/generated_python_code_query2_delay96_ctx32/seed_23.json`
- Result:
  - Winner count by held-out recall: SSM `3/3`.
  - Mean SSM held-out recall: `0.9998`.
  - Mean Transformer held-out recall: `0.0700`.
  - Mean SSM minus Transformer recall: `0.9299`.
  - Protocol evidence valid: `True`.

Interpretation: this is the strongest current generated-code route-around
result. It moves beyond compact symbolic records toward Python-shaped data while
keeping strict semantic ground truth. It is still synthetic generated code and
does not overturn the Gutenberg natural-language benchmark.

Full-context generated Python-like controls test whether this is only a
short-attention-window artifact. On seed `11`, the Transformer was given context
`128`, which covers the full sequence length (`106`):

- `artifacts/long_context_recall/generated_python_code_full_context_control.json`
  - Epochs: `40`.
  - Transformer train recall: `0.5117`.
  - Transformer held-out recall: `0.2832`.
  - SSM held-out recall: `1.0000`.
- `artifacts/long_context_recall/generated_python_code_full_context_control_seed11_epochs120.json`
  - Epochs: `120`.
  - Transformer train recall: `0.6172`.
  - Transformer held-out recall: `0.2798`.
  - SSM held-out recall: `1.0000`.

Interpretation: even when the tiny matched Transformer can attend over the whole
generated Python-like sequence and trains longer, it does not generalize the
mini-executor rule on held-out records. This strengthens the result beyond a
pure context-window bypass, while remaining a bounded generated-code claim.

Capacity probe:
`artifacts/long_context_recall/multi_assignment_kv_memory_uninit_bind8_delay96_ctx32_seed11_epochs40.json`
keeps `8` bindings in the same delay-`96` record and still reaches `1.0000`
SSM held-out recall on seed `11`; treat this as a promising single-seed probe
until repeated.

The next code-like stressor is `--task assignment-update`, where a name is bound
to an old value, then rebound to a new value, and the query must return the
latest value. `kv-memory` now includes a learned value-write erase gate so the
same key can be overwritten instead of only accumulating values:

- Artifact:
  `artifacts/long_context_recall/assignment_update_demo.json`
- Protocol:
  - `2,048` train records and `1,024` eval records.
  - `16` names and `16` values.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `1.034`.
  - `40` epochs.
  - No structural key/value/query/erase initialization.
- Result:
  - SSM held-out recall accuracy: `1.0000`.
  - Transformer held-out recall accuracy: `0.0566`.
  - SSM eval speed: `4,759 records/s`.
  - Transformer eval speed: `27,035 records/s`.

Interpretation: the explicit KV path now handles both multi-binding lookup and
last-write-wins update semantics on synthetic code-like records. The erase
operation is much slower on the current CPU implementation, so this is a quality
and capability result first, not a throughput win.

The first real-repository code-memory probe is now implemented as
`--task real-python-code`. It parses Python files in this repository with `ast`,
never executes source files, extracts assignment/alias/update sequences from
functions, and statically simulates a compact variable environment. Train and
eval use opposite stable hash-parity folds:

- Main artifact:
  `artifacts/long_context_recall/real_python_code_demo.json`
- Protocol:
  - `1,024` sampled train records and `512` sampled eval records.
  - `186` extracted train-fold records and `186` extracted eval-fold records.
  - `16` compact name buckets and `16` compact value buckets.
  - `4` assignment bindings and `2` query reads per record.
  - Delay `96`; Transformer context cap `32`.
  - Parameter budget ratio Transformer/SSM: `1.043`.
  - `40` epochs.
- Result:
  - SSM held-out recall accuracy across `1,024` query positions: `0.9951`.
  - Transformer held-out recall accuracy: `0.2617`.

Repeated-seed real-repo evidence is now available:

- Summary:
  `artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx32/summary.json`
- Seed artifacts:
  - `artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx32/seed_11.json`
  - `artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx32/seed_17.json`
  - `artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx32/seed_23.json`
- Result:
  - Winner count by held-out recall: SSM `3/3`.
  - Mean SSM held-out recall: `0.9932`.
  - Mean Transformer held-out recall: `0.2035`.
  - Mean SSM minus Transformer recall: `0.7897`.
  - Protocol evidence valid: `True`.

Full-context real-repo control:

- Artifact:
  `artifacts/long_context_recall/real_python_code_full_context_control.json`
- Protocol:
  - Same seed and sampled records.
  - Transformer context `128`, which covers the full sequence length (`103`).
  - `--allow-visible-key`, so this is a control run rather than bypass evidence.
- Result:
  - Transformer train recall: `0.9922`.
  - Transformer held-out recall: `0.5889`.
  - SSM held-out recall: `0.9951`.

Repeated-seed full-context real-repo control:

- Summary:
  `artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx128_full_context/summary.json`
- Seed artifacts:
  - `artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx128_full_context/seed_11.json`
  - `artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx128_full_context/seed_17.json`
  - `artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx128_full_context/seed_23.json`
- Result:
  - Winner count by held-out recall: SSM `3/3`.
  - Mean SSM held-out recall: `0.9932`.
  - Mean Transformer held-out recall: `0.5830`.
  - Mean SSM minus Transformer recall: `0.4102`.
  - Transformer train recall by seed: `0.9922`, `1.0000`, `0.9961`.
  - Protocol evidence valid as a full-context control: `True`.

Interpretation: this is the strongest move so far from synthetic generated code
toward actual project code. The tiny matched Transformer can memorize/train the
full-context real-repo records but generalizes worse on held-out folds, while
explicit KV memory preserves the binding rule. This is still a compact
static-AST probe with bucketized values, not a broad Python language-model or
natural-language win.

## Next Natural Layers

- Learned associative/key-value memory gates for delayed recall in natural text
  and code, beyond the current structurally initialized synthetic protocol.
- Transfer `kv-memory` from synthetic assignment recall into real code/text
  corpora and mixed natural-language objectives.
- Architecture sweeps that vary both SSM dimensions and Transformer depth/width.
- Memory-science comparisons between SSM hidden-state geometry and Transformer
  attention/activation diagnostics.
