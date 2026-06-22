PYTHON ?= python3
BENCHMARK_BASELINE ?= artifacts/benchmarks/quick_20260620T171809Z.json
BENCHMARK_CANDIDATE ?= artifacts/benchmarks/quick_compare_candidate.json
MAX_REGRESSION ?= 0.25
ARCH_SEEDS ?= 11,17,23

.PHONY: help compile verify benchmark benchmark-standard compare ingest-gutenberg-demo ingest-gutenberg-bulk profile-text-demo profile-text-bpe-demo profile-text-bulk-bpe compare-text-demo compare-text-architectures-demo compare-text-architectures-million compare-long-context-recall-demo compare-assignment-recall-demo compare-multi-assignment-recall-demo compare-sparse-kv-routing-smoke compare-sparse-kv-routing-margin-smoke compare-sparse-kv-routing-hard-smoke compare-sparse-kv-routing-margin-multiseed compare-multi-query-assignment-recall-demo compare-assignment-distractor-recall-demo compare-assignment-alias-recall-demo compare-mixed-code-recall-demo compare-mixed-code-recall-multiseed compare-mixed-code-full-context-control compare-generated-python-code-recall-demo compare-generated-python-code-recall-multiseed compare-generated-python-code-full-context-control compare-real-python-code-recall-demo compare-real-python-code-recall-multiseed compare-real-python-code-full-context-control compare-real-python-code-full-context-multiseed compare-assignment-update-recall-demo text-experiment-demo text-experiment-bpe-demo text-leaderboard text-sweep-demo text-sweep-bpe-demo text-sweep-bulk-bpe plan-text-sweep-bulk run-text-sweep-plan-bulk promote-text-demo inspect-text-demo run-promoted-text-demo text-run-demo clean-artifacts

help:
	@echo "KV-Memory SSM benchmark — operator commands"
	@echo ""
	@echo "Core:"
	@echo "  make compile   Syntax-check all product scripts"
	@echo "  make verify    Run end-to-end smoke verifier"
	@echo "  make benchmark Run quick benchmark, save JSON report"
	@echo "  make compare   Compare benchmark JSON reports"
	@echo ""
	@echo "Headline KV-memory vs Transformer recall benchmarks:"
	@echo "  make compare-long-context-recall-demo            delayed key recall"
	@echo "  make compare-multi-assignment-recall-demo        multi-binding recall"
	@echo "  make compare-sparse-kv-routing-margin-multiseed  near-binary read gate, multi-seed"
	@echo "  make compare-real-python-code-recall-multiseed   real-repo AST binding recall"
	@echo "  make compare-text-architectures-million          SSM vs Transformer, million-token"
	@echo ""
	@echo "Data + training:"
	@echo "  make ingest-gutenberg-demo  profile-text-bpe-demo  text-experiment-demo  text-sweep-demo"
	@echo ""
	@echo "Run 'grep -E \"^[a-z].*:\" Makefile' to list every target."

compile:
	$(PYTHON) -m py_compile \
		agent_ssm_core.py \
		soa_ssm_agents.py \
		text_tokenization.py \
		train_ssm_text.py \
		ingest_text_data.py \
		materialize_text_corpus.py \
		plan_text_sweep.py \
		run_text_sweep_plan.py \
		profile_text_corpus.py \
		run_text_checkpoint.py \
		compare_text_checkpoints.py \
		compare_text_architectures.py \
		run_text_architecture_multiseed.py \
		compare_long_context_recall.py \
		run_long_context_recall_multiseed.py \
		run_text_experiment.py \
		list_text_experiments.py \
		sweep_text_experiments.py \
		promote_text_checkpoint.py \
		inspect_text_checkpoint.py \
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

clean-artifacts:
	rm -rf artifacts/verify artifacts/benchmarks artifacts/trajectories

