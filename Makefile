PYTHON ?= python3
BENCHMARK_BASELINE ?= artifacts/benchmarks/quick_20260620T171809Z.json
BENCHMARK_CANDIDATE ?= artifacts/benchmarks/quick_compare_candidate.json
MAX_REGRESSION ?= 0.25
STAGE4B_SEED_COUNT ?= 20
ARCH_SEEDS ?= 11,17,23
STAGE4B_JOBS ?= 4
STAGE4B_FLAGS ?= --resume --quiet

.PHONY: help verify benchmark benchmark-standard compare compile ingest-gutenberg-demo profile-text-demo profile-text-bpe-demo compare-text-demo compare-text-architectures-demo compare-text-architectures-million compare-sparse-kv-routing-smoke text-experiment-demo text-experiment-bpe-demo text-sweep-demo text-sweep-bpe-demo text-leaderboard promote-text-demo inspect-text-demo run-promoted-text-demo probe-text-memory-demo compare-text-memory-demo text-memory-suite-demo validate-text-memory-suite-demo memory-viewer-open memory-suite-viewer-open release-viewer-open release-dashboard-open release-dashboard-compare-open release-pipeline-validation-open release-pipeline-validation-compare-open release-pipeline-compare-open release-pipeline-comparison-validation-open release-pipeline-comparison-validation-compare-open release-pipeline-proof-cockpit-open release-evidence-compare-open release-evidence-ledger-open release-evidence-ledger-compare-open release-evidence-audit-open release-evidence-audit-validation-open release-evidence-audit-validation-compare-open release-evidence-audit-compare-open release-history-open release-history-analysis-open release-gates-open formula-evolution-demo code-block-evolution-demo code-block-quadratic-demo code-block-abs-demo code-block-relu-demo code-block-sign-demo code-block-max-xy-demo code-block-piecewise-demo code-block-library-demo code-block-skill-ladder-demo code-block-hierarchy-demo code-block-stage4b-demo code-block-pruning-demo code-tape-sum-demo code-tape-sum-clean-demo code-tape-prior-demo code-tape-count-positive-demo code-tape-count-positive-prior-demo code-tape-argmax-demo code-tape-trajectory-demo code-tape-trajectory-seeded-demo code-tape-trace-prior-demo code-tape-trace-guidance-demo code-tape-prefix-value-demo code-tape-survival-demo code-tape-late-game-demo code-tape-late-game-ranker-demo code-tape-late-game-guided-demo code-tape-late-game-guidance-demo hypothesis-loop-demo selfplay-demo benchmark-text-package-demo validate-text-package-demo bundle-text-package-demo unbundle-text-package-demo compare-text-package-demo register-text-package-demo text-release-leaderboard-demo select-text-release-demo run-current-text-release-demo build-release-dashboard-demo validate-release-dashboard-demo compare-release-dashboard-demo build-release-history-demo validate-release-history-demo analyze-release-history-demo summarize-release-gates-demo generate-release-report-demo bundle-release-evidence-demo validate-release-evidence-demo compare-release-evidence-demo unbundle-release-evidence-demo register-release-evidence-demo validate-release-evidence-ledger-demo tamper-release-evidence-ledger-demo replay-release-evidence-ledger-demo compare-release-evidence-ledger-demo summarize-release-evidence-audit-demo validate-release-evidence-audit-demo compare-release-evidence-audit-validation-demo compare-release-evidence-audit-demo compare-text-release-demo text-release-pipeline-demo validate-release-pipeline-demo compare-release-pipeline-validation-demo compare-release-pipeline-demo validate-release-pipeline-comparison-demo compare-release-pipeline-comparison-validation-demo clean-artifacts

help:
	@echo "SoA/SSM prototype commands"
	@echo "  make compile             Syntax-check all scripts"
	@echo "  make verify              Run end-to-end smoke verifier"
	@echo "  make benchmark           Run quick benchmark and save JSON report"
	@echo "  make benchmark-standard  Run larger standard benchmark"
	@echo "  make compare             Compare benchmark JSON reports"
	@echo "  make ingest-gutenberg-demo Download public-domain book corpus"
	@echo "  make ingest-gutenberg-bulk Build a larger cached Gutenberg corpus"
	@echo "  make profile-text-demo   Profile the built-in text corpus"
	@echo "  make profile-text-bpe-demo Profile the book corpus with BPE"
	@echo "  make profile-text-bulk-bpe Profile bulk corpus and save BPE tokenizer"
	@echo "  make compare-text-demo   Train two tiny text checkpoints and compare them"
	@echo "  make compare-text-architectures-demo Compare SSM vs tiny Transformer"
	@echo "  make compare-text-architectures-million Run million-token repeated-seed architecture benchmark"
	@echo "  make compare-sparse-kv-routing-smoke Probe dynamic sparse KV read routing"
	@echo "  make compare-sparse-kv-routing-margin-smoke Probe query-margin hardening of sparse KV routing"
	@echo "  make compare-sparse-kv-routing-hard-smoke Probe hard inference after margin training"
	@echo "  make compare-sparse-kv-routing-margin-multiseed Check margin routing across seeds"
	@echo "  make text-experiment-demo Run profile/train/compare text pipeline"
	@echo "  make text-experiment-bpe-demo Run BPE experiment on book corpus"
	@echo "  make text-sweep-demo     Run a tiny text experiment hyperparameter sweep"
	@echo "  make text-sweep-bpe-demo Run a small BPE sweep on book corpus"
	@echo "  make text-sweep-bulk-bpe Run a larger BPE sweep on bulk book corpus"
	@echo "  make plan-text-sweep-bulk Plan resumable large BPE sweep batches"
	@echo "  make run-text-sweep-plan-bulk Run next batch from the large sweep plan"
	@echo "  make text-leaderboard    List saved text experiment runs"
	@echo "  make promote-text-demo   Promote best checkpoint from demo sweep"
	@echo "  make inspect-text-demo   Inspect promoted text checkpoint"
	@echo "  make run-promoted-text-demo Run promoted text package from manifest"
	@echo "  make probe-text-memory-demo Probe hidden-state memory of selected release"
	@echo "  make compare-text-memory-demo Compare two hidden-memory probes"
	@echo "  make text-memory-suite-demo Run prompt-suite hidden-memory matrix"
	@echo "  make validate-text-memory-suite-demo Validate prompt-suite memory artifacts"
	@echo "  make memory-viewer-open Open the static text memory viewer"
	@echo "  make memory-suite-viewer-open Open the static text memory suite matrix"
	@echo "  make release-viewer-open Open the static release leaderboard viewer"
	@echo "  make release-dashboard-open Open the static release control dashboard"
	@echo "  make release-dashboard-compare-open Open the static release dashboard compare viewer"
	@echo "  make release-pipeline-validation-open Open the static release pipeline validation viewer"
	@echo "  make release-pipeline-validation-compare-open Open the static release pipeline validation compare viewer"
	@echo "  make release-pipeline-compare-open Open the static release pipeline run compare viewer"
	@echo "  make release-pipeline-comparison-validation-open Open the static release pipeline comparison validation viewer"
	@echo "  make release-pipeline-comparison-validation-compare-open Open the static release pipeline comparison validation compare viewer"
	@echo "  make release-pipeline-proof-cockpit-open Open the static release pipeline proof cockpit"
	@echo "  make release-evidence-compare-open Open the static release evidence compare viewer"
	@echo "  make release-evidence-ledger-open Open the static release evidence ledger viewer"
	@echo "  make release-evidence-ledger-compare-open Open the static release evidence ledger compare viewer"
	@echo "  make release-evidence-audit-open Open the static release evidence audit viewer"
	@echo "  make release-evidence-audit-validation-open Open the static release evidence audit validation viewer"
	@echo "  make release-evidence-audit-validation-compare-open Open the static release evidence audit validation compare viewer"
	@echo "  make release-evidence-audit-compare-open Open the static release evidence audit compare viewer"
	@echo "  make release-history-open Open the static release history viewer"
	@echo "  make release-history-analysis-open Open the static release history analysis viewer"
	@echo "  make release-gates-open Open the static release gate summary viewer"
	@echo "  make formula-evolution-demo Search over SSM update formulas"
	@echo "  make code-block-evolution-demo Search over cached CodePy command blocks"
	@echo "  make code-block-quadratic-demo Search for compact quadratic form"
	@echo "  make code-block-abs-demo Search for abs(x) with conditional primitive"
	@echo "  make code-block-relu-demo Search for relu(x) with predicate blocks"
	@echo "  make code-block-sign-demo Search for sign(x) with predicate blocks"
	@echo "  make code-block-max-xy-demo Search for max(x,y) with predicate blocks"
	@echo "  make code-block-piecewise-demo Search for a piecewise function"
	@echo "  make code-block-library-demo Compare base vs learned CodePy block library"
	@echo "  make code-block-skill-ladder-demo Run CodePy find-compress-promote-reuse ladder"
	@echo "  make code-block-hierarchy-demo Compare base vs hierarchical CodePy task ladder"
	@echo "  make code-block-stage4b-demo Measure success_rate vs hierarchy_depth"
	@echo "  make code-block-pruning-demo Analyze macro pruning from Stage 4B"
	@echo "  make code-tape-sum-demo Run bounded tape DSL on sum(vector[4])"
	@echo "  make code-tape-sum-clean-demo Evolve sum(vector[4]) without reference injection"
	@echo "  make code-tape-prior-demo Build traces, train prior, run guided beam, promote macros"
	@echo "  make code-tape-count-positive-demo Run bounded tape DSL on count_positive(vector[4])"
	@echo "  make code-tape-count-positive-prior-demo Run clean branch discovery plus prior-guided beam"
	@echo "  make code-tape-argmax-demo Verify expanded tape DSL can express argmax_index(vector[4])"
	@echo "  make code-tape-trajectory-demo Run trajectory-guided beam on argmax_index(vector[4])"
	@echo "  make code-tape-trajectory-seeded-demo Verify seeded trace reconstruction for argmax_index(vector[4])"
	@echo "  make code-tape-trace-prior-demo Build contrastive argmax traces and train P(trace | task)"
	@echo "  make code-tape-trace-guidance-demo Compare autonomous beam vs beam + trace-prior"
	@echo "  make code-tape-prefix-value-demo Run prefix-value rollout diagnostic for argmax_index(vector[4])"
	@echo "  make code-tape-survival-demo Run prefix-value survival-policy search for argmax_index(vector[4])"
	@echo "  make code-tape-late-game-demo Analyze top late-game argmax near-misses"
	@echo "  make code-tape-late-game-ranker-demo Train+prove the late-game structural re-ranker"
	@echo "  make code-tape-late-game-guided-demo Run survival beam search with the late-game ranker"
	@echo "  make code-tape-late-game-guidance-demo Compare Beam+Survival vs +Late-Game on equal budget"
	@echo "  make hypothesis-loop-demo Evaluate LLM-style formula hypotheses"
	@echo "  make selfplay-demo       Coevolve runner/blocker SSM populations"
	@echo "  make benchmark-text-package-demo Benchmark promoted text package"
	@echo "  make validate-text-package-demo Validate promoted text package integrity"
	@echo "  make bundle-text-package-demo Bundle promoted text package"
	@echo "  make unbundle-text-package-demo Restore bundled text package"
	@echo "  make compare-text-package-demo Compare promoted and restored text packages"
	@echo "  make register-text-package-demo Register promoted package release"
	@echo "  make text-release-leaderboard-demo List registered text package releases"
	@echo "  make select-text-release-demo Select active text package release"
	@echo "  make run-current-text-release-demo Run selected active text release"
	@echo "  make build-release-dashboard-demo Build one-file release dashboard JSON"
	@echo "  make validate-release-dashboard-demo Validate release dashboard JSON"
	@echo "  make compare-release-dashboard-demo Compare release dashboard JSON against itself"
	@echo "  make build-release-history-demo Build release dashboard history JSON"
	@echo "  make validate-release-history-demo Validate release dashboard history JSON"
	@echo "  make analyze-release-history-demo Analyze release history trend"
	@echo "  make summarize-release-gates-demo Summarize release gate verdicts"
	@echo "  make generate-release-report-demo Generate Markdown release report"
	@echo "  make bundle-release-evidence-demo Bundle release evidence artifacts"
	@echo "  make validate-release-evidence-demo Validate release evidence bundle"
	@echo "  make compare-release-evidence-demo Compare release evidence bundles"
	@echo "  make unbundle-release-evidence-demo Restore release evidence bundle"
	@echo "  make register-release-evidence-demo Register release evidence ledger"
	@echo "  make validate-release-evidence-ledger-demo Validate release evidence ledger"
	@echo "  make tamper-release-evidence-ledger-demo Confirm ledger tamper detection"
	@echo "  make replay-release-evidence-ledger-demo Confirm ledger append/update replay"
	@echo "  make compare-release-evidence-ledger-demo Compare release evidence ledgers"
	@echo "  make summarize-release-evidence-audit-demo Summarize release evidence audit"
	@echo "  make validate-release-evidence-audit-demo Validate release evidence audit"
	@echo "  make compare-release-evidence-audit-validation-demo Compare release evidence audit validations"
	@echo "  make compare-release-evidence-audit-demo Compare release evidence audits"
	@echo "  make compare-text-release-demo Compare selected release against itself"
	@echo "  make text-release-pipeline-demo Promote, package, register, select, and run release"
	@echo "  make validate-release-pipeline-demo Validate the release pipeline report"
	@echo "  make compare-release-pipeline-validation-demo Compare release pipeline validations"
	@echo "  make compare-release-pipeline-demo Compare full release pipeline reports"
	@echo "  make validate-release-pipeline-comparison-demo Validate full pipeline comparison JSON"
	@echo "  make compare-release-pipeline-comparison-validation-demo Compare full pipeline comparison validations"
	@echo "  make text-release-bulk-bpe Package the bulk BPE sweep winner"
	@echo "  make text-run-demo       Train then run a small text checkpoint"
	@echo "  make export-demo         Export a small hand-coded trajectory"
	@echo "  make viewer-open         Open the static trajectory viewer"
	@echo "  make clean-artifacts     Remove generated artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  BENCHMARK_BASELINE=$(BENCHMARK_BASELINE)"
	@echo "  BENCHMARK_CANDIDATE=$(BENCHMARK_CANDIDATE)"
	@echo "  MAX_REGRESSION=$(MAX_REGRESSION)"

compile:
	$(PYTHON) -m py_compile \
		agent_ssm_core.py \
		analyze_text_release_history.py \
		text_tokenization.py \
		soa_ssm_agents.py \
		train_ssm_text.py \
		ingest_text_data.py \
		materialize_text_corpus.py \
		plan_text_sweep.py \
		run_text_sweep_plan.py \
		profile_text_corpus.py \
		run_text_checkpoint.py \
		probe_text_memory.py \
		compare_text_memory.py \
		run_text_memory_suite.py \
		validate_text_memory_suite.py \
		benchmark_text_package.py \
		compare_text_checkpoints.py \
		compare_text_architectures.py \
		run_text_architecture_multiseed.py \
		compare_long_context_recall.py \
		run_long_context_recall_multiseed.py \
		compare_text_packages.py \
		compare_text_releases.py \
		compare_text_release_dashboards.py \
		summarize_text_release_gates.py \
		generate_text_release_report.py \
		bundle_text_release_evidence.py \
		validate_text_release_evidence.py \
		compare_text_release_evidence.py \
		unbundle_text_release_evidence.py \
		register_text_release_evidence.py \
		validate_text_release_evidence_ledger.py \
		tamper_text_release_evidence_ledger.py \
		replay_text_release_evidence_ledger.py \
		compare_text_release_evidence_ledgers.py \
		summarize_text_release_evidence_audit.py \
		validate_text_release_evidence_audit.py \
		compare_text_release_evidence_audit_validations.py \
		compare_text_release_evidence_audits.py \
		register_text_package.py \
		list_text_releases.py \
		select_text_release.py \
		build_text_release_dashboard.py \
		build_text_release_history.py \
		validate_text_release_dashboard.py \
		validate_text_release_history.py \
		validate_text_release_pipeline.py \
		compare_text_release_pipeline_validations.py \
		compare_text_release_pipelines.py \
		validate_text_release_pipeline_comparison.py \
		compare_text_release_pipeline_comparison_validations.py \
		release_text_pipeline.py \
		run_text_experiment.py \
		list_text_experiments.py \
		sweep_text_experiments.py \
		promote_text_checkpoint.py \
		inspect_text_checkpoint.py \
		validate_text_package.py \
		bundle_text_package.py \
		unbundle_text_package.py \
		evolve_ssm_agents.py \
		evolve_ssm_formulas.py \
		evolve_code_blocks.py \
		evolve_code_tape.py \
		code_tape_prior.py \
		code_tape_trace_prior.py \
		train_code_tape_prior.py \
		train_code_tape_trace_prior.py \
		build_code_tape_trace_dataset.py \
		search_code_tape_prior.py \
		search_code_tape_trajectory.py \
		compare_code_tape_trace_guidance.py \
		promote_code_tape_macros.py \
		compare_code_block_library.py \
		code_block_skill_ladder.py \
		compare_code_block_hierarchy.py \
		compare_code_block_stage4b.py \
		analyze_code_block_pruning.py \
		analyze_code_tape_late_game.py \
		run_hypothesis_loop.py \
		coevolve_ssm_selfplay.py \
		run_evolved_agent.py \
		export_agent_trajectory.py \
		verify_system.py \
		benchmark_system.py \
		compare_benchmarks.py

verify:
	$(PYTHON) verify_system.py

benchmark:
	$(PYTHON) benchmark_system.py --profile quick

benchmark-standard:
	$(PYTHON) benchmark_system.py --profile standard

compare:
	$(PYTHON) compare_benchmarks.py \
		--baseline "$(BENCHMARK_BASELINE)" \
		--candidate "$(BENCHMARK_CANDIDATE)" \
		--max-regression "$(MAX_REGRESSION)" \
		--require-all

ingest-gutenberg-demo:
	$(PYTHON) ingest_text_data.py \
		--source gutenberg \
		--output-dir artifacts/corpora/gutenberg_demo \
		--output-text artifacts/corpora/gutenberg_demo/corpus.txt \
		--output-json artifacts/corpora/gutenberg_demo/corpus_manifest.json

ingest-gutenberg-bulk:
	$(PYTHON) ingest_text_data.py \
		--source gutenberg \
		--output-dir artifacts/corpora/gutenberg_bulk \
		--output-text artifacts/corpora/gutenberg_bulk/corpus.txt \
		--output-json artifacts/corpora/gutenberg_bulk/corpus_manifest.json \
		--target-bytes 1500000 \
		--max-books 8

profile-text-demo:
	$(PYTHON) profile_text_corpus.py \
		--output-json artifacts/text_profile_demo.json

profile-text-bpe-demo: ingest-gutenberg-demo
	$(PYTHON) profile_text_corpus.py \
		--text-file artifacts/corpora/gutenberg_demo/corpus.txt \
		--tokenizer bpe \
		--bpe-vocab-size 512 \
		--bpe-min-frequency 2 \
		--output-json artifacts/corpora/gutenberg_demo/profile_bpe.json

profile-text-bulk-bpe: ingest-gutenberg-bulk
	$(PYTHON) profile_text_corpus.py \
		--text-file artifacts/corpora/gutenberg_bulk/corpus.txt \
		--tokenizer bpe \
		--bpe-vocab-size 768 \
		--bpe-min-frequency 3 \
		--bpe-train-chars 1000000 \
		--target-streams 16 \
		--min-seq-len 32 \
		--max-seq-len 128 \
		--input-dim 64 \
		--state-dim 96 \
		--output-json artifacts/corpora/gutenberg_bulk/profile_bpe.json \
		--output-tokenizer-json artifacts/corpora/gutenberg_bulk/tokenizer_bpe_768.json

compare-text-demo:
	$(PYTHON) train_ssm_text.py \
		--streams 4 \
		--tokens-per-stream 96 \
		--seq-len 16 \
		--epochs 1 \
		--sample-steps 4 \
		--save-checkpoint artifacts/text_compare_baseline.npz
	$(PYTHON) train_ssm_text.py \
		--streams 4 \
		--tokens-per-stream 96 \
		--seq-len 16 \
		--epochs 2 \
		--sample-steps 4 \
		--save-checkpoint artifacts/text_compare_candidate.npz
	$(PYTHON) compare_text_checkpoints.py \
		--baseline artifacts/text_compare_baseline.npz \
		--candidate artifacts/text_compare_candidate.npz \
		--streams 4 \
		--tokens-per-stream 96 \
		--seq-len 16 \
		--sample-steps 6 \
		--fail-on-loss-regression \
		--output-json artifacts/text_compare_demo.json

compare-text-architectures-demo:
	$(PYTHON) compare_text_architectures.py \
		--output-json artifacts/text_architecture_comparison_demo.json \
		--epochs 1 \
		--repeat 1 \
		--sample-steps 4

compare-text-architectures-million:
	$(PYTHON) run_text_architecture_multiseed.py \
		--text-file artifacts/corpora/gutenberg_50mb/corpus.txt \
		--tokenizer bpe \
		--tokenizer-config artifacts/corpora/gutenberg_50mb/tokenizer_bpe_768.json \
		--output-dir artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million \
		--output-json artifacts/text_architecture_multiseed/gutenberg_50mb_bpe_million/summary.json \
		--seeds $(ARCH_SEEDS) \
		--streams 128 \
		--tokens-per-stream 8192 \
		--seq-len 128 \
		--ssm-input-dim 64 \
		--ssm-state-dim 96 \
		--transformer-layers 1 \
		--transformer-heads 2 \
		--transformer-ff-mult 2 \
		--epochs 1 \
		--learning-rate 0.003 \
		--repeat 2 \
		--sample-steps 8 \
		--require-valid-protocol \
		--timeout 900

compare-long-context-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--output-json artifacts/long_context_recall/demo.json \
		--train-records 1024 \
		--eval-records 512 \
		--key-count 16 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant token-memory \
		--ssm-decay-init 8.0

compare-assignment-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--task assignment \
		--output-json artifacts/long_context_recall/assignment_demo.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-multi-assignment-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--task multi-assignment \
		--output-json artifacts/long_context_recall/multi_assignment_demo.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-sparse-kv-routing-smoke:
	$(PYTHON) compare_long_context_recall.py \
		--task multi-assignment \
		--output-json artifacts/long_context_recall/sparse_kv_multi_assignment_curriculum_target_entropy_smoke.json \
		--train-records 512 \
		--eval-records 256 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 12 \
		--batch-size 64 \
		--learning-rate 0.003 \
		--kv-read-gate-l1 0.5 \
		--kv-read-gate-l1-start 0.0 \
		--kv-read-gate-l1-ramp-epochs 6 \
		--kv-read-gate-target 0.02 \
		--kv-read-gate-target-weight 5.0 \
		--kv-read-gate-entropy-weight 0.01 \
		--ssm-input-dim 48 \
		--ssm-state-dim 64 \
		--ssm-variant sparse-kv-memory \
		--init-assignment-kv-memory

.PHONY: compare-sparse-kv-routing-margin-smoke
compare-sparse-kv-routing-margin-smoke:
	$(PYTHON) compare_long_context_recall.py \
		--task multi-assignment \
		--output-json artifacts/long_context_recall/sparse_kv_multi_assignment_query_margin_smoke.json \
		--train-records 512 \
		--eval-records 256 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 12 \
		--batch-size 64 \
		--learning-rate 0.003 \
		--kv-read-gate-l1 0.5 \
		--kv-read-gate-l1-start 0.0 \
		--kv-read-gate-l1-ramp-epochs 6 \
		--kv-read-gate-target 0.02 \
		--kv-read-gate-target-weight 5.0 \
		--kv-read-gate-entropy-weight 0.01 \
		--kv-read-gate-query-margin 0.9 \
		--kv-read-gate-query-margin-weight 2.0 \
		--ssm-input-dim 48 \
		--ssm-state-dim 64 \
		--ssm-variant sparse-kv-memory \
		--init-assignment-kv-memory

.PHONY: compare-sparse-kv-routing-hard-smoke
compare-sparse-kv-routing-hard-smoke:
	$(PYTHON) compare_long_context_recall.py \
		--task multi-assignment \
		--output-json artifacts/long_context_recall/sparse_kv_multi_assignment_query_margin_hard_eval_smoke.json \
		--train-records 512 \
		--eval-records 256 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 12 \
		--batch-size 64 \
		--learning-rate 0.003 \
		--kv-read-gate-l1 0.5 \
		--kv-read-gate-l1-start 0.0 \
		--kv-read-gate-l1-ramp-epochs 6 \
		--kv-read-gate-target 0.02 \
		--kv-read-gate-target-weight 5.0 \
		--kv-read-gate-entropy-weight 0.01 \
		--kv-read-gate-query-margin 0.9 \
		--kv-read-gate-query-margin-weight 2.0 \
		--kv-read-gate-hard-eval-threshold 0.5 \
		--ssm-input-dim 48 \
		--ssm-state-dim 64 \
		--ssm-variant sparse-kv-memory \
		--init-assignment-kv-memory

.PHONY: compare-sparse-kv-routing-margin-multiseed
compare-sparse-kv-routing-margin-multiseed:
	$(PYTHON) run_long_context_recall_multiseed.py \
		--task multi-assignment \
		--output-dir artifacts/long_context_recall_multiseed/sparse_kv_margin_hard_seed6 \
		--output-json artifacts/long_context_recall_multiseed/sparse_kv_margin_hard_seed6/summary.json \
		--seeds 11,17,23,41,73,101 \
		--train-records 512 \
		--eval-records 256 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 12 \
		--batch-size 64 \
		--learning-rate 0.003 \
		--kv-read-gate-l1 0.5 \
		--kv-read-gate-l1-start 0.0 \
		--kv-read-gate-l1-ramp-epochs 6 \
		--kv-read-gate-target 0.02 \
		--kv-read-gate-target-weight 5.0 \
		--kv-read-gate-entropy-weight 0.01 \
		--kv-read-gate-query-margin 0.9 \
		--kv-read-gate-query-margin-weight 2.0 \
		--kv-read-gate-hard-eval-threshold 0.5 \
		--ssm-input-dim 48 \
		--ssm-state-dim 64 \
		--ssm-variant sparse-kv-memory \
		--init-assignment-kv-memory \
		--min-seeds 6 \
		--min-query-positions 256 \
		--require-valid-protocol

compare-multi-query-assignment-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--task multi-query-assignment \
		--output-json artifacts/long_context_recall/multi_query_assignment_demo.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-assignment-distractor-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--task assignment-distractor \
		--output-json artifacts/long_context_recall/assignment_distractor_demo.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-assignment-alias-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--task assignment-alias \
		--output-json artifacts/long_context_recall/assignment_alias_demo.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-mixed-code-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--task mixed-code \
		--output-json artifacts/long_context_recall/mixed_code_demo.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-mixed-code-recall-multiseed:
	$(PYTHON) run_long_context_recall_multiseed.py \
		--task mixed-code \
		--output-dir artifacts/long_context_recall_multiseed/mixed_code_query2_delay96_ctx32 \
		--output-json artifacts/long_context_recall_multiseed/mixed_code_query2_delay96_ctx32/summary.json \
		--seeds 11,17,23 \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0 \
		--min-seeds 3 \
		--min-query-positions 2048 \
		--require-valid-protocol \
		--timeout 420

compare-mixed-code-full-context-control:
	$(PYTHON) compare_long_context_recall.py \
		--task mixed-code \
		--output-json artifacts/long_context_recall/mixed_code_full_context_control.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 128 \
		--allow-visible-key \
		--epochs 120 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-generated-python-code-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--task generated-python-code \
		--output-json artifacts/long_context_recall/generated_python_code_demo.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-generated-python-code-recall-multiseed:
	$(PYTHON) run_long_context_recall_multiseed.py \
		--task generated-python-code \
		--output-dir artifacts/long_context_recall_multiseed/generated_python_code_query2_delay96_ctx32 \
		--output-json artifacts/long_context_recall_multiseed/generated_python_code_query2_delay96_ctx32/summary.json \
		--seeds 11,17,23 \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0 \
		--min-seeds 3 \
		--min-query-positions 2048 \
		--require-valid-protocol \
		--timeout 420

compare-generated-python-code-full-context-control:
	$(PYTHON) compare_long_context_recall.py \
		--task generated-python-code \
		--output-json artifacts/long_context_recall/generated_python_code_full_context_control.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 128 \
		--allow-visible-key \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-real-python-code-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--task real-python-code \
		--output-json artifacts/long_context_recall/real_python_code_demo.json \
		--train-records 1024 \
		--eval-records 512 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-real-python-code-recall-multiseed:
	$(PYTHON) run_long_context_recall_multiseed.py \
		--task real-python-code \
		--output-dir artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx32 \
		--output-json artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx32/summary.json \
		--seeds 11,17,23 \
		--train-records 1024 \
		--eval-records 512 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0 \
		--min-seeds 3 \
		--min-query-positions 1024 \
		--require-valid-protocol \
		--timeout 300

compare-real-python-code-full-context-control:
	$(PYTHON) compare_long_context_recall.py \
		--task real-python-code \
		--output-json artifacts/long_context_recall/real_python_code_full_context_control.json \
		--train-records 1024 \
		--eval-records 512 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 128 \
		--allow-visible-key \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

compare-real-python-code-full-context-multiseed:
	$(PYTHON) run_long_context_recall_multiseed.py \
		--task real-python-code \
		--output-dir artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx128_full_context \
		--output-json artifacts/long_context_recall_multiseed/real_python_code_query2_delay96_ctx128_full_context/summary.json \
		--seeds 11,17,23 \
		--train-records 1024 \
		--eval-records 512 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--query-count 2 \
		--delay 96 \
		--transformer-context 128 \
		--allow-visible-key \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0 \
		--min-seeds 3 \
		--min-query-positions 1024 \
		--require-valid-protocol \
		--timeout 300

compare-assignment-update-recall-demo:
	$(PYTHON) compare_long_context_recall.py \
		--task assignment-update \
		--output-json artifacts/long_context_recall/assignment_update_demo.json \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0

text-experiment-demo:
	rm -rf artifacts/text_experiments/demo
	$(PYTHON) run_text_experiment.py \
		--output-dir artifacts/text_experiments/demo \
		--streams 4 \
		--tokens-per-stream 96 \
		--seq-len 16 \
		--baseline-epochs 1 \
		--candidate-epochs 2

text-experiment-bpe-demo: ingest-gutenberg-demo
	rm -rf artifacts/text_experiments/gutenberg_bpe_demo
	$(PYTHON) run_text_experiment.py \
		--text-file artifacts/corpora/gutenberg_demo/corpus.txt \
		--tokenizer bpe \
		--bpe-vocab-size 512 \
		--bpe-min-frequency 2 \
		--bpe-train-chars 250000 \
		--output-dir artifacts/text_experiments/gutenberg_bpe_demo \
		--streams 8 \
		--tokens-per-stream 256 \
		--seq-len 32 \
		--baseline-epochs 1 \
		--candidate-epochs 2 \
		--timeout 240

text-leaderboard:
	$(PYTHON) list_text_experiments.py \
		--output-json artifacts/text_experiments/leaderboard.json \
		--output-csv artifacts/text_experiments/leaderboard.csv

text-sweep-demo:
	rm -rf artifacts/text_sweeps/demo
	$(PYTHON) sweep_text_experiments.py \
		--output-dir artifacts/text_sweeps/demo \
		--clean \
		--streams 4 \
		--tokens-per-stream 96 \
		--seq-len 16 \
		--state-dims 32,64 \
		--input-dims 48 \
		--learning-rates 0.012 \
		--baseline-epochs 1 \
		--candidate-epochs 2

text-sweep-bpe-demo: ingest-gutenberg-demo
	rm -rf artifacts/text_sweeps/gutenberg_bpe_demo
	$(PYTHON) sweep_text_experiments.py \
		--text-file artifacts/corpora/gutenberg_demo/corpus.txt \
		--tokenizer bpe \
		--bpe-vocab-size 512 \
		--bpe-min-frequency 2 \
		--bpe-train-chars 250000 \
		--output-dir artifacts/text_sweeps/gutenberg_bpe_demo \
		--clean \
		--streams 8 \
		--tokens-per-stream 256 \
		--seq-len 32 \
		--state-dims 32,64 \
		--input-dims 48 \
		--learning-rates 0.006,0.012 \
		--baseline-epochs 1 \
		--candidate-epochs 2 \
		--timeout 240

text-sweep-bulk-bpe: profile-text-bulk-bpe
	rm -rf artifacts/text_sweeps/gutenberg_bulk_bpe
	$(PYTHON) sweep_text_experiments.py \
		--text-file artifacts/corpora/gutenberg_bulk/corpus.txt \
		--tokenizer bpe \
		--tokenizer-config artifacts/corpora/gutenberg_bulk/tokenizer_bpe_768.json \
		--output-dir artifacts/text_sweeps/gutenberg_bulk_bpe \
		--clean \
		--streams 16 \
		--tokens-per-stream 2048 \
		--seq-len 128 \
		--state-dims 64,96,128 \
		--input-dims 64 \
		--learning-rates 0.003,0.006 \
		--baseline-epochs 3 \
		--candidate-epochs 5 \
		--compare-repeat 2 \
		--timeout 300

plan-text-sweep-bulk: profile-text-bulk-bpe
	$(PYTHON) plan_text_sweep.py \
		--text-file artifacts/corpora/gutenberg_bulk/corpus.txt \
		--corpus-manifest artifacts/corpora/gutenberg_bulk/corpus_manifest.json \
		--tokenizer bpe \
		--tokenizer-config artifacts/corpora/gutenberg_bulk/tokenizer_bpe_768.json \
		--output-dir artifacts/text_sweeps/gutenberg_bulk_bpe_large \
		--input-dims 48,64,96 \
		--state-dims 64,96,128,160 \
		--learning-rates 0.0015,0.003,0.006,0.009 \
		--baseline-epochs 3,5 \
		--candidate-epochs 5,8 \
		--streams 16 \
		--tokens-per-stream 2048 \
		--seq-len 128 \
		--batch-size 12 \
		--output-json artifacts/text_sweeps/gutenberg_bulk_bpe_large/plan.json

run-text-sweep-plan-bulk:
	$(PYTHON) run_text_sweep_plan.py \
		--plan artifacts/text_sweeps/gutenberg_bulk_bpe_large/plan.json \
		--max-batches 1 \
		--output-json artifacts/text_sweeps/gutenberg_bulk_bpe_large/plan_progress.json

formula-evolution-demo:
	$(PYTHON) evolve_ssm_formulas.py \
		--architecture-population 3 \
		--architecture-elites 1 \
		--architecture-generations 1 \
		--weight-population 8 \
		--weight-elites 2 \
		--weight-generations 1 \
		--agents 256 \
		--steps 16 \
		--output-json artifacts/formula_evolution_demo.json

code-block-evolution-demo:
	$(PYTHON) evolve_code_blocks.py \
		--target cubic_minus_x \
		--population 2048 \
		--generations 60 \
		--elites 96 \
		--program-length 9 \
		--cases 21 \
		--log-every 10 \
		--seed 4 \
		--output-json artifacts/code_block_evolution_demo.json

code-block-quadratic-demo:
	$(PYTHON) evolve_code_blocks.py \
		--target quadratic \
		--population 8192 \
		--generations 120 \
		--elites 256 \
		--program-length 7 \
		--cases 21 \
		--log-every 20 \
		--length-penalty 0.0001 \
		--stop-active-blocks 3 \
		--output-json artifacts/code_block_quadratic_demo.json

code-block-abs-demo:
	$(PYTHON) evolve_code_blocks.py \
		--target abs \
		--population 512 \
		--generations 20 \
		--elites 32 \
		--program-length 5 \
		--cases 21 \
		--log-every 5 \
		--output-json artifacts/code_block_abs_demo.json

code-block-relu-demo:
	$(PYTHON) evolve_code_blocks.py \
		--target relu \
		--population 4096 \
		--generations 80 \
		--elites 128 \
		--program-length 6 \
		--cases 21 \
		--log-every 20 \
		--stop-active-blocks 3 \
		--output-json artifacts/code_block_relu_demo.json

code-block-sign-demo:
	$(PYTHON) evolve_code_blocks.py \
		--target sign \
		--population 8192 \
		--generations 160 \
		--elites 256 \
		--program-length 10 \
		--cases 21 \
		--log-every 20 \
		--stop-active-blocks 6 \
		--output-json artifacts/code_block_sign_demo.json

code-block-max-xy-demo:
	$(PYTHON) evolve_code_blocks.py \
		--target max_xy \
		--population 8192 \
		--generations 100 \
		--elites 256 \
		--program-length 6 \
		--cases 49 \
		--log-every 20 \
		--stop-active-blocks 4 \
		--output-json artifacts/code_block_max_xy_demo.json

code-block-piecewise-demo:
	$(PYTHON) evolve_code_blocks.py \
		--target piecewise_square_neg \
		--population 8192 \
		--generations 120 \
		--elites 256 \
		--program-length 8 \
		--cases 21 \
		--log-every 20 \
		--output-json artifacts/code_block_piecewise_demo.json

code-block-library-demo:
	$(PYTHON) compare_code_block_library.py \
		--targets clamp01,max_abs_x_y \
		--profiles base,learned \
		--seeds 31 \
		--population 4096 \
		--generations 80 \
		--elites 128 \
		--program-length 8 \
		--cases 49 \
		--output-json artifacts/code_block_library/comparison.json

code-block-skill-ladder-demo:
	$(PYTHON) code_block_skill_ladder.py \
		--phase-a-targets square,abs,relu,max_xy,sign \
		--phase-c-targets clamp01,max_abs_x_y \
		--seeds 31 \
		--population 4096 \
		--generations 80 \
		--elites 128 \
		--phase-a-program-length 10 \
		--phase-c-program-length 8 \
		--phase-a-cases 49 \
		--phase-c-cases 49 \
		--output-json artifacts/code_block_skill_ladder/skill_ladder.json

code-block-hierarchy-demo:
	$(PYTHON) compare_code_block_hierarchy.py \
		--seeds 4 \
		--population 4096 \
		--generations 80 \
		--elites 128 \
		--cases 49 \
		--output-json artifacts/code_block_hierarchy/hierarchy.json

code-block-stage4b-demo:
	$(PYTHON) compare_code_block_stage4b.py \
		--seed-count $(STAGE4B_SEED_COUNT) \
		--jobs $(STAGE4B_JOBS) \
		$(STAGE4B_FLAGS) \
		--population 4096 \
		--generations 80 \
		--elites 128 \
		--cases 49 \
		--output-json artifacts/code_block_stage4b/stage4b.json \
		--output-svg artifacts/code_block_stage4b/success_rate_vs_depth.svg

code-block-pruning-demo:
	$(PYTHON) analyze_code_block_pruning.py \
		--stage4b-json artifacts/code_block_stage4b/stage4b.json \
		--output-json artifacts/code_block_stage4c/pruning.json \
		--output-md artifacts/code_block_stage4c/pruning.md

code-tape-sum-demo:
	$(PYTHON) evolve_code_tape.py \
		--target sum4 \
		--population 1024 \
		--generations 5 \
		--elites 64 \
		--program-length 6 \
		--max-steps 32 \
		--cases 16 \
		--inject-reference \
		--output-json artifacts/code_tape_sum4_demo.json

code-tape-sum-clean-demo:
	$(PYTHON) evolve_code_tape.py \
		--target sum4 \
		--population 8192 \
		--generations 120 \
		--elites 256 \
		--program-length 6 \
		--max-steps 32 \
		--cases 32 \
		--seed 7 \
		--mutation-rate 0.18 \
		--random-reset-fraction 0.08 \
		--shape-read 0.02 \
		--shape-move 0.02 \
		--shape-loop 0.02 \
		--shape-accumulate 0.03 \
		--shape-coverage 0.01 \
		--output-json artifacts/code_tape_sum4_clean_demo.json

code-tape-prior-demo:
	$(PYTHON) evolve_code_tape.py \
		--target sum4 \
		--population 8192 \
		--generations 120 \
		--elites 256 \
		--program-length 6 \
		--max-steps 32 \
		--cases 32 \
		--seed 7 \
		--mutation-rate 0.18 \
		--random-reset-fraction 0.08 \
		--shape-read 0.02 \
		--shape-move 0.02 \
		--shape-loop 0.02 \
		--shape-accumulate 0.03 \
		--shape-coverage 0.01 \
		--trace-jsonl artifacts/code_tape_prior/sum4_traces.jsonl \
		--trace-all \
		--output-json artifacts/code_tape_prior/sum4_trace_run.json
	$(PYTHON) train_code_tape_prior.py \
		--trace-jsonl artifacts/code_tape_prior/sum4_traces.jsonl \
		--output-json artifacts/code_tape_prior/sum4_prior.json \
		--program-length 6 \
		--vocab-size 35 \
		--epochs 160 \
		--seed 17
	$(PYTHON) search_code_tape_prior.py \
		--prior-json artifacts/code_tape_prior/sum4_prior.json \
		--target sum4 \
		--program-length 6 \
		--max-steps 32 \
		--cases 32 \
		--beam-width 12 \
		--expand-top-k 12 \
		--prior-weight 1.0 \
		--reward-weight 1.0 \
		--mse-weight 0.25 \
		--shape-read 0.02 \
		--shape-move 0.02 \
		--shape-loop 0.02 \
		--shape-accumulate 0.03 \
		--shape-coverage 0.01 \
		--stop-on-solution \
		--output-json artifacts/code_tape_prior/sum4_prior_beam.json
	$(PYTHON) promote_code_tape_macros.py \
		--trace-jsonl artifacts/code_tape_prior/sum4_traces.jsonl \
		--search-json artifacts/code_tape_prior/sum4_prior_beam.json \
		--top-k 8 \
		--output-json artifacts/code_tape_prior/sum4_macros.json

code-tape-count-positive-demo:
	$(PYTHON) evolve_code_tape.py \
		--target count_positive4 \
		--population 1024 \
		--generations 5 \
		--elites 64 \
		--program-length 7 \
		--max-steps 40 \
		--cases 16 \
		--inject-reference \
		--output-json artifacts/code_tape_count_positive4_demo.json

code-tape-count-positive-prior-demo:
	$(PYTHON) evolve_code_tape.py \
		--target count_positive4 \
		--population 8192 \
		--generations 60 \
		--elites 256 \
		--program-length 6 \
		--max-steps 32 \
		--cases 32 \
		--seed 11 \
		--mutation-rate 0.18 \
		--random-reset-fraction 0.08 \
		--shape-read 0.02 \
		--shape-move 0.02 \
		--shape-loop 0.02 \
		--shape-predicate 0.05 \
		--shape-conditional 0.05 \
		--shape-coverage 0.01 \
		--trace-jsonl artifacts/code_tape_prior/count_positive4_traces_no_reference.jsonl \
		--trace-all \
		--output-json artifacts/code_tape_prior/count_positive4_trace_run_no_reference.json
	$(PYTHON) train_code_tape_prior.py \
		--trace-jsonl artifacts/code_tape_prior/count_positive4_traces_no_reference.jsonl \
		--output-json artifacts/code_tape_prior/count_positive4_prior.json \
		--program-length 6 \
		--vocab-size 35 \
		--epochs 200 \
		--seed 23
	$(PYTHON) search_code_tape_prior.py \
		--prior-json artifacts/code_tape_prior/count_positive4_prior.json \
		--target count_positive4 \
		--program-length 6 \
		--max-steps 32 \
		--cases 32 \
		--beam-width 32 \
		--expand-top-k 20 \
		--prior-weight 1.0 \
		--reward-weight 1.0 \
		--mse-weight 0.25 \
		--shape-read 0.02 \
		--shape-move 0.02 \
		--shape-loop 0.02 \
		--shape-predicate 0.05 \
		--shape-conditional 0.05 \
		--shape-coverage 0.01 \
		--stop-on-solution \
		--output-json artifacts/code_tape_prior/count_positive4_prior_beam.json
	$(PYTHON) promote_code_tape_macros.py \
		--trace-jsonl artifacts/code_tape_prior/count_positive4_traces_no_reference.jsonl \
		--search-json artifacts/code_tape_prior/count_positive4_prior_beam.json \
		--top-k 8 \
		--output-json artifacts/code_tape_prior/count_positive4_macros.json

code-tape-argmax-demo:
	$(PYTHON) evolve_code_tape.py \
		--target argmax_index4 \
		--population 256 \
		--generations 1 \
		--elites 32 \
		--program-length 11 \
		--max-steps 64 \
		--cases 32 \
		--inject-reference \
		--output-json artifacts/code_tape_prior/argmax_index4_geometry_check.json

code-tape-trajectory-demo:
	$(PYTHON) search_code_tape_trajectory.py \
		--target argmax_index4 \
		--program-length 11 \
		--max-steps 64 \
		--cases 32 \
		--beam-width 24 \
		--signature-cases 8 \
		--signature-steps 18 \
		--diversity-penalty 0.65 \
		--max-same-signature 1 \
		--mse-weight 0.45 \
		--output-json artifacts/code_tape_prior/argmax_index4_trajectory_beam.json

code-tape-trajectory-seeded-demo:
	$(PYTHON) search_code_tape_trajectory.py \
		--target argmax_index4 \
		--program-length 11 \
		--max-steps 64 \
		--cases 32 \
		--beam-width 24 \
		--signature-cases 8 \
		--signature-steps 18 \
		--diversity-penalty 0.65 \
		--max-same-signature 1 \
		--mse-weight 0.45 \
		--seed-reference-prefixes \
		--stop-on-solution \
		--output-json artifacts/code_tape_prior/argmax_index4_trajectory_seeded.json

code-tape-trace-prior-demo:
	$(PYTHON) search_code_tape_trajectory.py \
		--target argmax_index4 \
		--program-length 11 \
		--max-steps 64 \
		--cases 32 \
		--beam-width 24 \
		--signature-cases 8 \
		--signature-steps 18 \
		--diversity-penalty 0.65 \
		--max-same-signature 1 \
		--mse-weight 0.45 \
		--candidate-jsonl artifacts/code_tape_prior/argmax_index4_trajectory_candidates.jsonl \
		--output-json artifacts/code_tape_prior/argmax_index4_trajectory_beam.json
	$(PYTHON) search_code_tape_trajectory.py \
		--target argmax_index4 \
		--program-length 11 \
		--max-steps 64 \
		--cases 32 \
		--beam-width 24 \
		--signature-cases 8 \
		--signature-steps 18 \
		--diversity-penalty 0.65 \
		--max-same-signature 1 \
		--mse-weight 0.45 \
		--seed-reference-prefixes \
		--stop-on-solution \
		--candidate-jsonl artifacts/code_tape_prior/argmax_index4_trajectory_seeded_candidates.jsonl \
		--output-json artifacts/code_tape_prior/argmax_index4_trajectory_seeded.json
	$(PYTHON) build_code_tape_trace_dataset.py \
		--input-jsonl artifacts/code_tape_prior/argmax_index4_trajectory_candidates.jsonl \
		--input-jsonl artifacts/code_tape_prior/argmax_index4_trajectory_seeded_candidates.jsonl \
		--reference-json artifacts/code_tape_prior/argmax_index4_trajectory_seeded.json \
		--output-jsonl artifacts/code_tape_prior/argmax_index4_contrastive_traces.jsonl \
		--summary-json artifacts/code_tape_prior/argmax_index4_contrastive_summary.json \
		--dedupe \
		--min-successes 1 \
		--min-near-misses 1
	$(PYTHON) train_code_tape_trace_prior.py \
		--trace-jsonl artifacts/code_tape_prior/argmax_index4_contrastive_traces.jsonl \
		--output-json artifacts/code_tape_prior/argmax_index4_trace_prior.json \
		--hash-buckets 64 \
		--epochs 240 \
		--seed 31

code-tape-trace-guidance-demo:
	$(PYTHON) compare_code_tape_trace_guidance.py \
		--trace-prior-json artifacts/code_tape_prior/argmax_index4_trace_prior.json \
		--output-dir artifacts/code_tape_prior/trace_guidance_argmax_b24 \
		--output-json artifacts/code_tape_prior/argmax_index4_trace_guidance_comparison.json \
		--cases 32 \
		--beam-width 24 \
		--signature-cases 8 \
		--signature-steps 18 \
		--trace-prior-weight 0.1 \
		--trace-prior-weight 0.25 \
		--trace-prior-weight 0.5 \
		--trace-prior-weight 1.0 \
		--trace-prior-weight 2.0

code-tape-prefix-value-demo:
	$(PYTHON) search_code_tape_trajectory.py \
		--target argmax_index4 \
		--program-length 11 \
		--max-steps 64 \
		--cases 32 \
		--beam-width 24 \
		--signature-cases 8 \
		--signature-steps 18 \
		--trace-prior-json artifacts/code_tape_prior/argmax_index4_trace_prior.json \
		--trace-prior-weight 0.25 \
		--value-rollouts 25 \
		--value-rollout-top-k 96 \
		--value-rollout-role-top-k 96 \
		--value-rollout-max-depth 4 \
		--value-weight 0.03 \
		--candidate-jsonl artifacts/code_tape_prior/argmax_index4_value_b24_w025_candidates.jsonl \
		--output-json artifacts/code_tape_prior/argmax_index4_value_b24_w025.json

code-tape-survival-demo:
	$(PYTHON) search_code_tape_trajectory.py \
		--target argmax_index4 \
		--program-length 11 \
		--max-steps 64 \
		--cases 32 \
		--beam-width 64 \
		--signature-cases 8 \
		--signature-steps 18 \
		--trace-prior-json artifacts/code_tape_prior/argmax_index4_trace_prior.json \
		--trace-prior-weight 0.25 \
		--value-rollouts 25 \
		--value-rollout-top-k 128 \
		--value-rollout-role-top-k 128 \
		--value-rollout-max-depth 4 \
		--value-weight 0.03 \
		--survival-lane-fraction 0.25 \
		--survival-ttl 8 \
		--survival-min-value 11.9 \
		--survival-ignore-signature-limit \
		--candidate-jsonl artifacts/code_tape_prior/argmax_index4_survival_b64_ttl8_uncapped_w025_candidates.jsonl \
		--output-json artifacts/code_tape_prior/argmax_index4_survival_b64_ttl8_uncapped_w025.json

code-tape-late-game-demo:
	$(PYTHON) analyze_code_tape_late_game.py \
		--input-jsonl artifacts/code_tape_prior/argmax_index4_survival_b64_ttl8_uncapped_w025_candidates.jsonl \
		--search-json artifacts/code_tape_prior/argmax_index4_survival_b64_ttl8_uncapped_w025.json \
		--target argmax_index4 \
		--top-n 50 \
		--min-depth 7 \
		--max-train-mse 3.0 \
		--output-json artifacts/code_tape_prior/argmax_index4_late_game_analysis.json \
		--output-md artifacts/code_tape_prior/argmax_index4_late_game_analysis.md

code-tape-late-game-ranker-demo:
	$(PYTHON) train_code_tape_late_game_ranker.py \
		--candidates-jsonl artifacts/code_tape_prior/argmax_index4_survival_b64_ttl8_uncapped_w025_candidates.jsonl \
		--search-json artifacts/code_tape_prior/argmax_index4_survival_b64_ttl8_uncapped_w025.json \
		--target argmax_index4 \
		--max-negatives 1500 \
		--output-json artifacts/code_tape_prior/argmax_index4_late_game_ranker.json
	$(PYTHON) rank_code_tape_late_game.py \
		--model-json artifacts/code_tape_prior/argmax_index4_late_game_ranker.json \
		--candidates-jsonl artifacts/code_tape_prior/argmax_index4_survival_b64_ttl8_uncapped_w025_candidates.jsonl \
		--target argmax_index4 \
		--top-n 25 \
		--output-json artifacts/code_tape_prior/argmax_index4_late_game_rerank.json \
		--output-md artifacts/code_tape_prior/argmax_index4_late_game_rerank.md

code-tape-late-game-guided-demo:
	$(PYTHON) search_code_tape_trajectory.py \
		--target argmax_index4 \
		--program-length 11 \
		--max-steps 64 \
		--cases 32 \
		--beam-width 64 \
		--signature-cases 8 \
		--signature-steps 18 \
		--trace-prior-json artifacts/code_tape_prior/argmax_index4_trace_prior.json \
		--trace-prior-weight 0.25 \
		--value-rollouts 25 \
		--value-rollout-top-k 128 \
		--value-rollout-role-top-k 128 \
		--value-rollout-max-depth 4 \
		--value-weight 0.03 \
		--prefix-stage-lane-fraction 0.125 \
		--prefix-stage-ignore-signature-limit \
		--survival-lane-fraction 0.25 \
		--survival-ttl 8 \
		--survival-min-value 11.9 \
		--survival-ignore-signature-limit \
		--late-game-ranker-json artifacts/code_tape_prior/argmax_index4_late_game_ranker.json \
		--late-game-weight 0.25 \
		--late-game-lane-fraction 0.25 \
		--late-game-min-remaining 2 \
		--candidate-jsonl artifacts/code_tape_prior/late_game_guidance/argmax_index4_late_game_guided_candidates.jsonl \
		--output-json artifacts/code_tape_prior/late_game_guidance/argmax_index4_late_game_guided.json

code-tape-late-game-guidance-demo:
	$(PYTHON) compare_code_tape_late_game_guidance.py \
		--late-game-ranker-json artifacts/code_tape_prior/argmax_index4_late_game_ranker.json \
		--trace-prior-json artifacts/code_tape_prior/argmax_index4_trace_prior.json \
		--trace-prior-weight 0.25 \
		--beam-width 64 \
		--value-rollouts 25 \
		--value-rollout-top-k 128 \
		--value-rollout-role-top-k 128 \
		--value-rollout-max-depth 4 \
		--value-weight 0.03 \
		--prefix-stage-lane-fraction 0.125 \
		--prefix-stage-ignore-signature-limit \
		--survival-lane-fraction 0.25 \
		--survival-ttl 8 \
		--survival-min-value 11.9 \
		--survival-ignore-signature-limit \
		--late-game-lane-fraction 0.25 \
		--late-game-weight 0.25 \
		--late-game-weight 0.5 \
		--late-game-weight 1.0 \
		--output-dir artifacts/code_tape_prior/late_game_guidance \
		--output-json artifacts/code_tape_prior/argmax_index4_late_game_guidance_comparison.json

hypothesis-loop-demo:
	$(PYTHON) run_hypothesis_loop.py \
		--max-proposals 3 \
		--weight-population 8 \
		--weight-elites 2 \
		--weight-generations 1 \
		--agents 256 \
		--steps 16 \
		--output-json artifacts/hypothesis_loop_demo.json \
		--feedback-jsonl artifacts/hypothesis_feedback_demo.jsonl \
		--prompt-json artifacts/hypothesis_prompt_demo.json

selfplay-demo:
	$(PYTHON) coevolve_ssm_selfplay.py \
		--runners 4 \
		--blockers 4 \
		--runner-elites 1 \
		--blocker-elites 1 \
		--agents 128 \
		--steps 12 \
		--generations 2 \
		--output-json artifacts/selfplay_demo.json

promote-text-demo:
	$(PYTHON) promote_text_checkpoint.py \
		--leaderboard artifacts/text_sweeps/demo/leaderboard.json \
		--output-dir artifacts/promoted/text_demo \
		--smoke-run

inspect-text-demo:
	$(PYTHON) inspect_text_checkpoint.py \
		--manifest artifacts/promoted/text_demo/manifest.json \
		--output-json artifacts/promoted/text_demo/model_card.json \
		--output-md artifacts/promoted/text_demo/MODEL_CARD.md

run-promoted-text-demo:
	$(PYTHON) run_text_checkpoint.py \
		--manifest artifacts/promoted/text_demo/manifest.json \
		--sample-steps 8 \
		--output-json artifacts/promoted/text_demo/run_from_manifest.json

probe-text-memory-demo: select-text-release-demo
	$(PYTHON) probe_text_memory.py \
		--release artifacts/releases/current_text_release.json \
		--prompt "memory is a river" \
		--top-k 3 \
		--include-state-vectors \
		--output-json artifacts/releases/text_memory_probe.json \
		--output-csv artifacts/releases/text_memory_probe.csv

compare-text-memory-demo: select-text-release-demo
	$(PYTHON) probe_text_memory.py \
		--release artifacts/releases/current_text_release.json \
		--prompt "memory is a river" \
		--top-k 3 \
		--include-state-vectors \
		--output-json artifacts/releases/text_memory_probe_baseline.json \
		--output-csv artifacts/releases/text_memory_probe_baseline.csv
	$(PYTHON) probe_text_memory.py \
		--release artifacts/releases/current_text_release.json \
		--prompt "memory is another world" \
		--top-k 3 \
		--include-state-vectors \
		--output-json artifacts/releases/text_memory_probe_candidate.json \
		--output-csv artifacts/releases/text_memory_probe_candidate.csv
	$(PYTHON) compare_text_memory.py \
		--baseline artifacts/releases/text_memory_probe_baseline.json \
		--candidate artifacts/releases/text_memory_probe_candidate.json \
		--output-json artifacts/releases/text_memory_comparison.json \
		--output-csv artifacts/releases/text_memory_comparison.csv \
		--max-state-norm-delta 2.0 \
		--max-state-delta-norm-delta 2.0 \
		--max-top-probability-delta 0.2 \
		--min-mean-top-overlap 0.0

text-memory-suite-demo: select-text-release-demo
	$(PYTHON) run_text_memory_suite.py \
		--release artifacts/releases/current_text_release.json \
		--prompt "memory is a river" \
		--prompt "memory is another world" \
		--prompt "the future is guessed" \
		--top-k 3 \
		--include-state-vectors \
		--output-dir artifacts/releases/text_memory_suite \
		--output-json artifacts/releases/text_memory_suite/memory_suite.json \
		--prompts-csv artifacts/releases/text_memory_suite/prompts.csv \
		--pairs-csv artifacts/releases/text_memory_suite/pairs.csv

validate-text-memory-suite-demo: text-memory-suite-demo
	$(PYTHON) validate_text_memory_suite.py \
		--suite artifacts/releases/text_memory_suite/memory_suite.json \
		--prompts-csv artifacts/releases/text_memory_suite/prompts.csv \
		--pairs-csv artifacts/releases/text_memory_suite/pairs.csv \
		--expected-prompts 3 \
		--require-state-vectors \
		--require-prompt-artifacts \
		--min-mean-pair-overlap 0.0 \
		--min-pair-overlap 0.0 \
		--output-json artifacts/releases/text_memory_suite/validation.json

run-current-text-release-demo: select-text-release-demo
	$(PYTHON) run_text_checkpoint.py \
		--release artifacts/releases/current_text_release.json \
		--sample-steps 8 \
		--output-json artifacts/releases/run_current_text_release.json

build-release-dashboard-demo: validate-text-memory-suite-demo run-current-text-release-demo
	$(PYTHON) build_text_release_dashboard.py \
		--leaderboard artifacts/releases/text_release_leaderboard.json \
		--current-release artifacts/releases/current_text_release.json \
		--memory-suite artifacts/releases/text_memory_suite/memory_suite.json \
		--memory-validation artifacts/releases/text_memory_suite/validation.json \
		--run-json artifacts/releases/run_current_text_release.json \
		--output-json artifacts/releases/text_release_dashboard.json \
		--require-valid

validate-release-dashboard-demo: build-release-dashboard-demo
	$(PYTHON) validate_text_release_dashboard.py \
		--dashboard artifacts/releases/text_release_dashboard.json \
		--require-artifacts \
		--expected-prompts 3 \
		--expected-pairs 3 \
		--min-mean-pair-overlap 0.0 \
		--output-json artifacts/releases/text_release_dashboard_validation.json

compare-release-dashboard-demo: validate-release-dashboard-demo
	$(PYTHON) compare_text_release_dashboards.py \
		--baseline-dashboard artifacts/releases/text_release_dashboard.json \
		--candidate-dashboard artifacts/releases/text_release_dashboard.json \
		--require-artifacts \
		--expected-prompts 3 \
		--expected-pairs 3 \
		--min-mean-pair-overlap 0.0 \
		--fail-on-loss-regression \
		--fail-on-throughput-regression \
		--fail-on-accuracy-regression \
		--fail-on-audit-regression \
		--fail-on-memory-overlap-regression \
		--fail-on-memory-cosine-regression \
		--fail-on-memory-l2-regression \
		--fail-on-memory-norm-regression \
		--output-json artifacts/releases/text_release_dashboard_comparison.json

build-release-history-demo: validate-release-dashboard-demo
	$(PYTHON) build_text_release_history.py \
		--dashboard artifacts/releases/text_release_dashboard.json \
		--require-artifacts \
		--expected-prompts 3 \
		--expected-pairs 3 \
		--min-mean-pair-overlap 0.0 \
		--output-json artifacts/releases/text_release_history.json

validate-release-history-demo: build-release-history-demo
	$(PYTHON) validate_text_release_history.py \
		--history artifacts/releases/text_release_history.json \
		--require-artifacts \
		--expected-releases 1 \
		--output-json artifacts/releases/text_release_history_validation.json

analyze-release-history-demo: validate-release-history-demo
	$(PYTHON) analyze_text_release_history.py \
		--history artifacts/releases/text_release_history.json \
		--baseline previous \
		--require-artifacts \
		--expected-releases 1 \
		--fail-on-loss-regression \
		--fail-on-throughput-regression \
		--fail-on-accuracy-regression \
		--fail-on-audit-regression \
		--fail-on-memory-overlap-regression \
		--fail-on-memory-cosine-regression \
		--fail-on-memory-l2-regression \
		--fail-on-memory-norm-regression \
		--output-json artifacts/releases/text_release_history_analysis.json

summarize-release-gates-demo: compare-release-dashboard-demo validate-release-history-demo analyze-release-history-demo
	$(PYTHON) summarize_text_release_gates.py \
		--dashboard-validation artifacts/releases/text_release_dashboard_validation.json \
		--dashboard-comparison artifacts/releases/text_release_dashboard_comparison.json \
		--history-validation artifacts/releases/text_release_history_validation.json \
		--history-analysis artifacts/releases/text_release_history_analysis.json \
		--output-json artifacts/releases/text_release_gates.json

generate-release-report-demo: summarize-release-gates-demo
	$(PYTHON) generate_text_release_report.py \
		--gates artifacts/releases/text_release_gates.json \
		--dashboard artifacts/releases/text_release_dashboard.json \
		--history-analysis artifacts/releases/text_release_history_analysis.json \
		--output-md artifacts/releases/text_release_report.md

bundle-release-evidence-demo: generate-release-report-demo
	$(PYTHON) bundle_text_release_evidence.py \
		--gates artifacts/releases/text_release_gates.json \
		--output artifacts/releases/text_release_evidence.tar.gz \
		--output-json artifacts/releases/text_release_evidence_manifest.json

validate-release-evidence-demo: bundle-release-evidence-demo
	$(PYTHON) validate_text_release_evidence.py \
		--manifest artifacts/releases/text_release_evidence_manifest.json \
		--output-json artifacts/releases/text_release_evidence_validation.json

compare-release-evidence-demo: validate-release-evidence-demo
	$(PYTHON) compare_text_release_evidence.py \
		--baseline-manifest artifacts/releases/text_release_evidence_manifest.json \
		--candidate-manifest artifacts/releases/text_release_evidence_manifest.json \
		--fail-on-archive-sha-change \
		--fail-on-file-set-change \
		--fail-on-file-sha-change \
		--fail-on-file-byte-change \
		--fail-on-archive-name-change \
		--fail-on-validity-change \
		--fail-on-gate-regression \
		--output-json artifacts/releases/text_release_evidence_comparison.json

unbundle-release-evidence-demo: validate-release-evidence-demo
	rm -rf artifacts/restored/text_release_evidence
	$(PYTHON) unbundle_text_release_evidence.py \
		--archive artifacts/releases/text_release_evidence.tar.gz \
		--manifest artifacts/releases/text_release_evidence_manifest.json \
		--output-dir artifacts/restored \
		--output-json artifacts/restored/text_release_evidence_restore.json \
		--clean

register-release-evidence-demo: compare-release-evidence-demo unbundle-release-evidence-demo
	rm -rf artifacts/releases/text_release_evidence_store
	rm -f artifacts/releases/text_release_evidence_ledger.json
	$(PYTHON) register_text_release_evidence.py \
		--manifest artifacts/releases/text_release_evidence_manifest.json \
		--archive artifacts/releases/text_release_evidence.tar.gz \
		--validation-json artifacts/releases/text_release_evidence_validation.json \
		--restore-json artifacts/restored/text_release_evidence_restore.json \
		--comparison-json artifacts/releases/text_release_evidence_comparison.json \
		--ledger artifacts/releases/text_release_evidence_ledger.json \
		--store-dir artifacts/releases/text_release_evidence_store \
		--name text_demo

validate-release-evidence-ledger-demo: register-release-evidence-demo
	$(PYTHON) validate_text_release_evidence_ledger.py \
		--ledger artifacts/releases/text_release_evidence_ledger.json \
		--output-json artifacts/releases/text_release_evidence_ledger_validation.json \
		--expected-entries 1 \
		--require-artifacts \
		--artifact-scope latest \
		--fail-on-failed-gates

tamper-release-evidence-ledger-demo: validate-release-evidence-ledger-demo
	$(PYTHON) tamper_text_release_evidence_ledger.py \
		--ledger artifacts/releases/text_release_evidence_ledger.json \
		--output-ledger artifacts/releases/text_release_evidence_ledger_tampered.json \
		--output-json artifacts/releases/text_release_evidence_ledger_tamper.json \
		--mode archive_sha256 \
		--expected-entries 1 \
		--require-artifacts \
		--artifact-scope latest \
		--fail-on-failed-gates

replay-release-evidence-ledger-demo: validate-release-evidence-ledger-demo
	$(PYTHON) replay_text_release_evidence_ledger.py \
		--manifest artifacts/releases/text_release_evidence_manifest.json \
		--archive artifacts/releases/text_release_evidence.tar.gz \
		--validation-json artifacts/releases/text_release_evidence_validation.json \
		--restore-json artifacts/restored/text_release_evidence_restore.json \
		--comparison-json artifacts/releases/text_release_evidence_comparison.json \
		--base-dir artifacts/releases \
		--ledger artifacts/releases/text_release_evidence_ledger_replay.json \
		--store-dir artifacts/releases/text_release_evidence_ledger_replay_store \
		--work-dir artifacts/releases/text_release_evidence_ledger_replay \
		--output-json artifacts/releases/text_release_evidence_ledger_replay_report.json \
		--clean \
		--fail-on-failed-gates

compare-release-evidence-ledger-demo: replay-release-evidence-ledger-demo
	$(PYTHON) compare_text_release_evidence_ledgers.py \
		--baseline-ledger artifacts/releases/text_release_evidence_ledger_replay.json \
		--candidate-ledger artifacts/releases/text_release_evidence_ledger_replay.json \
		--output-json artifacts/releases/text_release_evidence_ledger_comparison.json \
		--require-artifacts \
		--artifact-scope all \
		--fail-on-failed-gates \
		--fail-on-chain-head-change \
		--fail-on-entry-count-change \
		--fail-on-entry-set-change \
		--fail-on-entry-field-change \
		--fail-on-archive-set-change \
		--fail-on-manifest-set-change
	$(PYTHON) compare_text_release_evidence_ledgers.py \
		--baseline-ledger artifacts/releases/text_release_evidence_ledger.json \
		--candidate-ledger artifacts/releases/text_release_evidence_ledger_replay.json \
		--output-json artifacts/releases/text_release_evidence_ledger_growth_comparison.json \
		--require-artifacts \
		--artifact-scope all \
		--fail-on-failed-gates

summarize-release-evidence-audit-demo: compare-release-evidence-ledger-demo tamper-release-evidence-ledger-demo
	$(PYTHON) summarize_text_release_evidence_audit.py \
		--manifest artifacts/releases/text_release_evidence_manifest.json \
		--validation-json artifacts/releases/text_release_evidence_validation.json \
		--restore-json artifacts/restored/text_release_evidence_restore.json \
		--comparison-json artifacts/releases/text_release_evidence_comparison.json \
		--ledger artifacts/releases/text_release_evidence_ledger.json \
		--ledger-validation-json artifacts/releases/text_release_evidence_ledger_validation.json \
		--tamper-json artifacts/releases/text_release_evidence_ledger_tamper.json \
		--replay-json artifacts/releases/text_release_evidence_ledger_replay_report.json \
		--ledger-comparison-json artifacts/releases/text_release_evidence_ledger_comparison.json \
		--ledger-growth-comparison-json artifacts/releases/text_release_evidence_ledger_growth_comparison.json \
		--output-json artifacts/releases/text_release_evidence_audit.json \
		--output-md artifacts/releases/text_release_evidence_audit.md

validate-release-evidence-audit-demo: summarize-release-evidence-audit-demo
	$(PYTHON) validate_text_release_evidence_audit.py \
		--audit artifacts/releases/text_release_evidence_audit.json \
		--output-json artifacts/releases/text_release_evidence_audit_validation.json \
		--require-sources

compare-release-evidence-audit-validation-demo: validate-release-evidence-audit-demo
	$(PYTHON) compare_text_release_evidence_audit_validations.py \
		--baseline-validation artifacts/releases/text_release_evidence_audit_validation.json \
		--candidate-validation artifacts/releases/text_release_evidence_audit_validation.json \
		--output-json artifacts/releases/text_release_evidence_audit_validation_comparison.json \
		--fail-on-validity-change \
		--fail-on-audit-path-change \
		--fail-on-release-name-change \
		--fail-on-check-count-change \
		--fail-on-source-count-change \
		--fail-on-failed-check-regression \
		--fail-on-failure-regression \
		--fail-on-error-regression \
		--fail-on-artifact-check-regression \
		--fail-on-source-artifact-check-regression \
		--fail-on-error-set-change

compare-release-evidence-audit-demo: compare-release-evidence-audit-validation-demo
	$(PYTHON) compare_text_release_evidence_audits.py \
		--baseline-audit artifacts/releases/text_release_evidence_audit.json \
		--candidate-audit artifacts/releases/text_release_evidence_audit.json \
		--output-json artifacts/releases/text_release_evidence_audit_comparison.json \
		--fail-on-validity-change \
		--fail-on-release-name-change \
		--fail-on-archive-sha-change \
		--fail-on-chain-head-change \
		--fail-on-replay-action-change \
		--fail-on-tamper-change \
		--fail-on-entry-count-change \
		--fail-on-check-set-change \
		--fail-on-check-validity-change \
		--fail-on-check-message-change \
		--fail-on-check-metric-change \
		--fail-on-check-path-change \
		--fail-on-source-set-change \
		--fail-on-source-path-change \
		--fail-on-failed-check-regression \
		--fail-on-failure-regression \
		--fail-on-artifact-check-regression

compare-text-release-demo: text-release-pipeline-demo
	$(PYTHON) compare_text_releases.py \
		--baseline-release artifacts/releases/current_text_pipeline_release.json \
		--candidate-release artifacts/releases/current_text_pipeline_release.json \
		--require-memory \
		--fail-on-loss-regression \
		--fail-on-throughput-regression \
		--fail-on-accuracy-regression \
		--fail-on-memory-overlap-regression \
		--fail-on-memory-cosine-regression \
		--fail-on-memory-l2-regression \
		--fail-on-memory-norm-regression \
		--max-loss-regression 0.0 \
		--max-throughput-regression 0.0 \
		--max-accuracy-regression 0.0 \
		--max-memory-overlap-regression 0.0 \
		--max-memory-cosine-regression 0.0 \
		--max-memory-l2-regression 0.0 \
		--max-memory-norm-regression 0.0 \
		--output-json artifacts/releases/text_release_comparison.json

text-release-pipeline-demo: text-sweep-demo
	$(PYTHON) release_text_pipeline.py \
		--leaderboard artifacts/text_sweeps/demo/leaderboard.json \
		--promote-dir artifacts/promoted/text_pipeline_demo \
		--restore-dir artifacts/restored/text_pipeline_demo \
		--root-name text_pipeline_demo \
		--registry artifacts/releases/text_pipeline_packages.json \
		--release-leaderboard-json artifacts/releases/text_pipeline_leaderboard.json \
		--release-leaderboard-csv artifacts/releases/text_pipeline_leaderboard.csv \
		--current-release-json artifacts/releases/current_text_pipeline_release.json \
		--run-json artifacts/releases/run_current_text_pipeline_release.json \
		--dashboard-json artifacts/releases/text_pipeline_dashboard.json \
		--output-json artifacts/releases/text_release_pipeline_demo.json \
		--include-evidence-audit \
		--name text_pipeline_demo \
		--benchmark-repeat 2 \
		--sample-steps 6 \
		--clean

validate-release-pipeline-demo: text-release-pipeline-demo
	$(PYTHON) validate_text_release_pipeline.py \
		--pipeline artifacts/releases/text_release_pipeline_demo.json \
		--require-artifacts \
		--require-evidence-audit \
		--expected-stages 35 \
		--output-json artifacts/releases/text_release_pipeline_validation.json

compare-release-pipeline-validation-demo: validate-release-pipeline-demo
	$(PYTHON) compare_text_release_pipeline_validations.py \
		--baseline-validation artifacts/releases/text_release_pipeline_validation.json \
		--candidate-validation artifacts/releases/text_release_pipeline_validation.json \
		--output-json artifacts/releases/text_release_pipeline_validation_comparison.json \
		--fail-on-validity-change \
		--fail-on-pipeline-path-change \
		--fail-on-name-change \
		--fail-on-evidence-presence-change \
		--fail-on-stage-count-change \
		--fail-on-memory-prompt-count-change \
		--fail-on-memory-pair-count-change \
		--fail-on-dashboard-audit-check-regression \
		--fail-on-evidence-audit-check-regression \
		--fail-on-artifact-check-regression \
		--fail-on-error-regression \
		--fail-on-error-set-change

compare-release-pipeline-demo: text-release-pipeline-demo
	$(PYTHON) compare_text_release_pipelines.py \
		--baseline-pipeline artifacts/releases/text_release_pipeline_demo.json \
		--candidate-pipeline artifacts/releases/text_release_pipeline_demo.json \
		--output-json artifacts/releases/text_release_pipeline_comparison.json \
		--require-evidence-audit \
		--fail-on-stage-set-change \
		--fail-on-stage-count-change \
		--fail-on-name-change \
		--fail-on-release-count-change \
		--fail-on-rank-change \
		--fail-on-checkpoint-sha-change \
		--fail-on-archive-sha-change \
		--fail-on-evidence-presence-change \
		--fail-on-evidence-archive-sha-change \
		--fail-on-ledger-chain-head-change \
		--fail-on-dashboard-audit-regression \
		--fail-on-evidence-audit-regression \
		--fail-on-loss-regression \
		--fail-on-throughput-regression \
		--fail-on-accuracy-regression \
		--fail-on-memory-prompt-count-change \
		--fail-on-memory-pair-count-change \
		--fail-on-memory-overlap-regression \
		--fail-on-memory-min-overlap-regression \
		--fail-on-memory-cosine-regression \
		--fail-on-memory-l2-regression \
		--fail-on-memory-norm-regression \
		--max-loss-regression 0.0 \
		--max-throughput-regression 0.0 \
		--max-accuracy-regression 0.0

validate-release-pipeline-comparison-demo: compare-release-pipeline-demo
	$(PYTHON) validate_text_release_pipeline_comparison.py \
		--comparison artifacts/releases/text_release_pipeline_comparison.json \
		--require-evidence-audit \
		--expected-stages 35 \
		--output-json artifacts/releases/text_release_pipeline_comparison_validation.json

compare-release-pipeline-comparison-validation-demo: validate-release-pipeline-comparison-demo
	$(PYTHON) compare_text_release_pipeline_comparison_validations.py \
		--baseline-validation artifacts/releases/text_release_pipeline_comparison_validation.json \
		--candidate-validation artifacts/releases/text_release_pipeline_comparison_validation.json \
		--output-json artifacts/releases/text_release_pipeline_comparison_validation_comparison.json \
		--fail-on-validity-change \
		--fail-on-comparison-path-change \
		--fail-on-pipeline-path-change \
		--fail-on-stage-count-change \
		--fail-on-stage-set-change \
		--fail-on-loss-delta-change \
		--fail-on-memory-overlap-delta-change \
		--fail-on-proof-flag-change \
		--fail-on-failure-regression \
		--fail-on-error-regression \
		--fail-on-error-set-change

text-release-bulk-bpe: text-sweep-bulk-bpe
	$(PYTHON) release_text_pipeline.py \
		--leaderboard artifacts/text_sweeps/gutenberg_bulk_bpe/leaderboard.json \
		--text-file artifacts/corpora/gutenberg_bulk/corpus.txt \
		--promote-dir artifacts/promoted/gutenberg_bulk_bpe \
		--restore-dir artifacts/restored/gutenberg_bulk_bpe \
		--root-name gutenberg_bulk_bpe \
		--registry artifacts/releases/gutenberg_bulk_bpe_packages.json \
		--release-leaderboard-json artifacts/releases/gutenberg_bulk_bpe_leaderboard.json \
		--release-leaderboard-csv artifacts/releases/gutenberg_bulk_bpe_leaderboard.csv \
		--current-release-json artifacts/releases/current_gutenberg_bulk_bpe_release.json \
		--run-json artifacts/releases/run_current_gutenberg_bulk_bpe_release.json \
		--dashboard-json artifacts/releases/gutenberg_bulk_bpe_dashboard.json \
		--output-json artifacts/releases/gutenberg_bulk_bpe_release_pipeline.json \
		--name gutenberg_bulk_bpe \
		--benchmark-repeat 3 \
		--sample-steps 12 \
		--clean \
		--timeout 300

benchmark-text-package-demo:
	$(PYTHON) benchmark_text_package.py \
		--manifest artifacts/promoted/text_demo/manifest.json \
		--warmup 1 \
		--repeat 5 \
		--sample-steps 8 \
		--output-json artifacts/promoted/text_demo/benchmark.json

validate-text-package-demo: benchmark-text-package-demo
	$(PYTHON) validate_text_package.py \
		--manifest artifacts/promoted/text_demo/manifest.json \
		--require-model-card \
		--require-smoke \
		--require-benchmark \
		--output-json artifacts/promoted/text_demo/validation.json

bundle-text-package-demo: validate-text-package-demo
	$(PYTHON) bundle_text_package.py \
		--manifest artifacts/promoted/text_demo/manifest.json \
		--output artifacts/promoted/text_demo/text_package.tar.gz \
		--output-json artifacts/promoted/text_demo/bundle_manifest.json \
		--root-name text_demo

unbundle-text-package-demo: bundle-text-package-demo
	rm -rf artifacts/restored/text_demo
	$(PYTHON) unbundle_text_package.py \
		--archive artifacts/promoted/text_demo/text_package.tar.gz \
		--bundle-manifest artifacts/promoted/text_demo/bundle_manifest.json \
		--output-dir artifacts/restored \
		--output-json artifacts/restored/text_demo_restore.json

compare-text-package-demo: unbundle-text-package-demo
	$(PYTHON) compare_text_packages.py \
		--baseline-manifest artifacts/promoted/text_demo/manifest.json \
		--candidate-manifest artifacts/restored/text_demo/manifest.json \
		--fail-on-loss-regression \
		--fail-on-throughput-regression \
		--output-json artifacts/promoted/text_demo/package_comparison.json

register-text-package-demo: compare-text-package-demo
	$(PYTHON) register_text_package.py \
		--manifest artifacts/promoted/text_demo/manifest.json \
		--bundle-manifest artifacts/promoted/text_demo/bundle_manifest.json \
		--archive artifacts/promoted/text_demo/text_package.tar.gz \
		--comparison-json artifacts/promoted/text_demo/package_comparison.json \
		--registry artifacts/releases/text_packages.json \
		--name text_demo

text-release-leaderboard-demo: register-text-package-demo
	$(PYTHON) list_text_releases.py \
		--registry artifacts/releases/text_packages.json \
		--output-json artifacts/releases/text_release_leaderboard.json \
		--output-csv artifacts/releases/text_release_leaderboard.csv

select-text-release-demo: text-release-leaderboard-demo
	$(PYTHON) select_text_release.py \
		--registry artifacts/releases/text_packages.json \
		--rank 1 \
		--output-json artifacts/releases/current_text_release.json

text-run-demo:
	$(PYTHON) train_ssm_text.py \
		--streams 4 \
		--tokens-per-stream 96 \
		--seq-len 16 \
		--epochs 2 \
		--sample-steps 6 \
		--save-checkpoint artifacts/text_run_demo.npz
	$(PYTHON) run_text_checkpoint.py \
		--checkpoint artifacts/text_run_demo.npz \
		--streams 4 \
		--tokens-per-stream 96 \
		--seq-len 16 \
		--sample-steps 8 \
		--output-json artifacts/text_run_demo.json

export-demo:
	$(PYTHON) export_agent_trajectory.py \
		--mode handcoded \
		--agents 32 \
		--steps 48 \
		--output-dir artifacts/trajectories/handcoded_demo

viewer-open:
	open trajectory_viewer.html

memory-viewer-open:
	open text_memory_viewer.html

memory-suite-viewer-open:
	open text_memory_suite_viewer.html

release-viewer-open:
	open text_release_leaderboard_viewer.html

release-dashboard-open:
	open text_release_dashboard.html

release-dashboard-compare-open:
	open text_release_dashboard_compare.html

release-pipeline-validation-open:
	open text_release_pipeline_validation_viewer.html

release-pipeline-validation-compare-open:
	open text_release_pipeline_validation_compare.html

release-pipeline-compare-open:
	open text_release_pipeline_compare.html

release-pipeline-comparison-validation-open:
	open text_release_pipeline_comparison_validation_viewer.html

release-pipeline-comparison-validation-compare-open:
	open text_release_pipeline_comparison_validation_compare.html

release-pipeline-proof-cockpit-open:
	open text_release_pipeline_proof_cockpit.html

release-evidence-compare-open:
	open text_release_evidence_compare.html

release-evidence-ledger-open:
	open text_release_evidence_ledger_viewer.html

release-evidence-ledger-compare-open:
	open text_release_evidence_ledger_compare.html

release-evidence-audit-open:
	open text_release_evidence_audit_viewer.html

release-evidence-audit-validation-open:
	open text_release_evidence_audit_validation_viewer.html

release-evidence-audit-validation-compare-open:
	open text_release_evidence_audit_validation_compare.html

release-evidence-audit-compare-open:
	open text_release_evidence_audit_compare.html

release-history-open:
	open text_release_history_viewer.html

release-history-analysis-open:
	open text_release_history_analysis_viewer.html

release-gates-open:
	open text_release_gates_viewer.html

clean-artifacts:
	rm -rf artifacts/verify artifacts/benchmarks artifacts/trajectories
