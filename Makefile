PYTHON ?= python3
BENCHMARK_BASELINE ?= artifacts/benchmarks/quick_20260620T171809Z.json
BENCHMARK_CANDIDATE ?= artifacts/benchmarks/quick_compare_candidate.json
MAX_REGRESSION ?= 0.25
ARCH_SEEDS ?= 11,17,23

.PHONY: help compile verify benchmark benchmark-standard compare ingest-gutenberg-demo ingest-gutenberg-bulk profile-text-demo profile-text-bpe-demo profile-text-bulk-bpe compare-text-demo compare-text-architectures-demo compare-text-architectures-million compare-long-context-recall-demo compare-assignment-recall-demo compare-multi-assignment-recall-demo compare-sparse-kv-routing-smoke compare-sparse-kv-routing-margin-smoke compare-sparse-kv-routing-hard-smoke compare-sparse-kv-routing-margin-multiseed compare-multi-query-assignment-recall-demo compare-assignment-distractor-recall-demo compare-assignment-alias-recall-demo compare-mixed-code-recall-demo compare-mixed-code-recall-multiseed compare-mixed-code-full-context-control compare-generated-python-code-recall-demo compare-generated-python-code-recall-multiseed compare-generated-python-code-full-context-control compare-real-python-code-recall-demo compare-real-python-code-recall-multiseed compare-real-python-code-full-context-control compare-real-python-code-full-context-multiseed compare-assignment-update-recall-demo text-experiment-demo text-experiment-bpe-demo text-leaderboard text-sweep-demo text-sweep-bpe-demo text-sweep-bulk-bpe plan-text-sweep-bulk run-text-sweep-plan-bulk promote-text-demo inspect-text-demo run-promoted-text-demo text-run-demo neocortex-x-stage0-smoke neocortex-x-stage1-smoke neocortex-x-stage1b-trained-smoke neocortex-x-stage2-smoke neocortex-x-stage2b-learned-smoke neocortex-x-stage3-smoke neocortex-x-stage3b-promotion-smoke neocortex-x-stage4-smoke neocortex-x-stage4b-world-model-smoke neocortex-x-stage5-smoke clean-artifacts
.PHONY: neocortex-x-stage5b-cached-smoke
.PHONY: neocortex-x-stage6-packets-smoke
.PHONY: neocortex-x-stage7-memory-smoke
.PHONY: neocortex-x-stage8-freshness-smoke
.PHONY: neocortex-x-stage9-action-gate-smoke
.PHONY: neocortex-x-stage10-feedback-smoke
.PHONY: neocortex-x-stage11-policy-smoke
.PHONY: neocortex-x-stage12-policy-replay-smoke
.PHONY: neocortex-x-stage13-promote-policy-smoke
.PHONY: neocortex-x-stage14-runtime-loader-smoke
.PHONY: neocortex-x-stage15-runtime-execution-smoke
.PHONY: neocortex-x-stage16-contextual-memory-smoke
.PHONY: neocortex-x-stage17-contextual-action-gate-smoke
.PHONY: neocortex-x-stage18-contextual-feedback-smoke
.PHONY: neocortex-x-stage19-contextual-policy-smoke
.PHONY: neocortex-x-stage20-contextual-policy-replay-smoke
.PHONY: neocortex-x-stage21-contextual-policy-promotion-smoke
.PHONY: neocortex-x-stage22-contextual-runtime-loader-smoke
.PHONY: neocortex-x-stage23-contextual-runtime-execution-smoke
.PHONY: neocortex-x-stage24-contextual-self-monitor-smoke
.PHONY: neocortex-x-stage25-indexed-contextual-store-smoke
.PHONY: neocortex-x-stage26-store-backed-contextual-retrieval-smoke
.PHONY: neocortex-x-stage27-store-backed-contextual-action-gate-smoke
.PHONY: neocortex-x-stage28-evidence-log-compaction-smoke
.PHONY: neocortex-x-stage29-scale-probe-smoke
.PHONY: neocortex-x-stage30-partial-context-smoke
.PHONY: neocortex-x-stage31-disambiguation-gate-smoke
.PHONY: neocortex-x-stage32-clarification-resolution-smoke
.PHONY: neocortex-x-stage33-clarification-feedback-smoke
.PHONY: neocortex-x-stage34-clarification-policy-smoke

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
	@echo "NeoCortex-X:"
	@echo "  make neocortex-x-stage0-smoke                    trace/replay instrumentation"
	@echo "  make neocortex-x-stage1-smoke                    recall-to-action smoke"
	@echo "  make neocortex-x-stage1b-trained-smoke           trained SSM/KV action mapping"
	@echo "  make neocortex-x-stage2-smoke                    stateful tape world smoke"
	@echo "  make neocortex-x-stage2b-learned-smoke           learned tape policy smoke"
	@echo "  make neocortex-x-stage3-smoke                    procedure reuse smoke"
	@echo "  make neocortex-x-stage3b-promotion-smoke         learned procedure promotion smoke"
	@echo "  make neocortex-x-stage4-smoke                    short lookahead planning smoke"
	@echo "  make neocortex-x-stage4b-world-model-smoke       learned world-model planning smoke"
	@echo "  make neocortex-x-stage5-smoke                    open-world provenance smoke"
	@echo "  make neocortex-x-stage5b-cached-smoke            cached live-shaped provenance adapter"
	@echo "  make neocortex-x-stage6-packets-smoke            provenance memory evidence packets"
	@echo "  make neocortex-x-stage7-memory-smoke             long-term packet memory retrieval"
	@echo "  make neocortex-x-stage8-freshness-smoke          temporal memory freshness policy"
	@echo "  make neocortex-x-stage9-action-gate-smoke        memory freshness to action gate"
	@echo "  make neocortex-x-stage10-feedback-smoke          action outcome feedback ledger"
	@echo "  make neocortex-x-stage11-policy-smoke            feedback-driven policy update"
	@echo "  make neocortex-x-stage12-policy-replay-smoke     apply policy snapshot in replay"
	@echo "  make neocortex-x-stage13-promote-policy-smoke    promote replay-validated policy"
	@echo "  make neocortex-x-stage14-runtime-loader-smoke    load promoted policy runtime"
	@echo "  make neocortex-x-stage15-runtime-execution-smoke execute runtime policy traces"
	@echo "  make neocortex-x-stage16-contextual-memory-smoke conditional context-aware memory"
	@echo "  make neocortex-x-stage17-contextual-action-gate-smoke contextual decision to action gate"
	@echo "  make neocortex-x-stage18-contextual-feedback-smoke contextual action feedback ledger"
	@echo "  make neocortex-x-stage19-contextual-policy-smoke contextual feedback to policy snapshot"
	@echo "  make neocortex-x-stage20-contextual-policy-replay-smoke replay contextual policy snapshot"
	@echo "  make neocortex-x-stage21-contextual-policy-promotion-smoke promote contextual runtime policy"
	@echo "  make neocortex-x-stage22-contextual-runtime-loader-smoke load contextual runtime policy"
	@echo "  make neocortex-x-stage23-contextual-runtime-execution-smoke execute contextual runtime traces"
	@echo "  make neocortex-x-stage24-contextual-self-monitor-smoke monitor contextual runtime traces"
	@echo "  make neocortex-x-stage25-indexed-contextual-store-smoke build indexed contextual memory store"
	@echo "  make neocortex-x-stage26-store-backed-contextual-retrieval-smoke retrieve contextual memory from store"
	@echo "  make neocortex-x-stage27-store-backed-contextual-action-gate-smoke act on store-backed contextual retrieval"
	@echo "  make neocortex-x-stage28-evidence-log-compaction-smoke compact append-only evidence log"
	@echo "  make neocortex-x-stage29-scale-probe-smoke       probe contextual memory scale"
	@echo "  make neocortex-x-stage30-partial-context-smoke   probe partial-context retrieval"
	@echo "  make neocortex-x-stage31-disambiguation-gate-smoke answer-or-clarify partial context"
	@echo "  make neocortex-x-stage32-clarification-resolution-smoke resolve clarified context"
	@echo "  make neocortex-x-stage33-clarification-feedback-smoke score clarification feedback"
	@echo "  make neocortex-x-stage34-clarification-policy-smoke update clarification key policy"
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
		research/neocortex_x/core_experiment.py \
		research/neocortex_x/stage1b_trained_recall_to_action.py \
		research/neocortex_x/stage2b_learned_tape_policy.py \
		research/neocortex_x/stage3b_learned_procedure_promotion.py \
		research/neocortex_x/stage4b_learned_world_model.py \
		research/neocortex_x/stage5b_live_cached_adapter.py \
		research/neocortex_x/stage6_evidence_packets.py \
		research/neocortex_x/stage7_memory_retrieval.py \
		research/neocortex_x/stage8_temporal_freshness.py \
		research/neocortex_x/stage9_action_gate.py \
		research/neocortex_x/stage10_feedback_loop.py \
		research/neocortex_x/stage11_policy_update.py \
		research/neocortex_x/stage12_policy_replay.py \
		research/neocortex_x/stage13_policy_promotion.py \
		research/neocortex_x/stage14_runtime_loader.py \
		research/neocortex_x/stage15_runtime_execution.py \
		research/neocortex_x/stage16_contextual_memory.py \
		research/neocortex_x/stage17_contextual_action_gate.py \
		research/neocortex_x/stage18_contextual_feedback.py \
		research/neocortex_x/stage19_contextual_policy_update.py \
		research/neocortex_x/stage20_contextual_policy_replay.py \
		research/neocortex_x/stage21_contextual_policy_promotion.py \
		research/neocortex_x/stage22_contextual_runtime_loader.py \
		research/neocortex_x/stage23_contextual_runtime_execution.py \
		research/neocortex_x/stage24_contextual_self_monitor.py \
		research/neocortex_x/stage25_indexed_contextual_memory_store.py \
		research/neocortex_x/stage26_store_backed_contextual_retrieval.py \
		research/neocortex_x/stage27_store_backed_contextual_action_gate.py \
		research/neocortex_x/stage28_append_only_evidence_log_compaction.py \
		research/neocortex_x/stage29_contextual_memory_scale_probe.py \
		research/neocortex_x/stage30_partial_context_scale_retrieval.py \
		research/neocortex_x/stage31_partial_context_disambiguation_gate.py \
		research/neocortex_x/stage32_clarification_resolution_loop.py \
		research/neocortex_x/stage33_clarification_feedback_ledger.py \
		research/neocortex_x/stage34_clarification_policy_update.py \
		run_text_experiment.py \
		list_text_experiments.py \
		sweep_text_experiments.py \
		promote_text_checkpoint.py \
		inspect_text_checkpoint.py \
		verify_system.py \
		benchmark_system.py \
		compare_benchmarks.py

verify:
	$(PYTHON) verify_system.py --timeout 600

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

# Phase 1 external validation: canonical MQAR (shared vocab, no query-marker
# token, scattered pairs) against the parameter-matched context-capped
# Transformer, using the context-routed dynamic-kv-memory SSM. Make-or-break.
compare-mqar-recall-multiseed:
	$(PYTHON) run_long_context_recall_multiseed.py \
		--task mqar \
		--output-dir artifacts/long_context_recall_multiseed/mqar_query3_delay96_ctx32 \
		--output-json artifacts/long_context_recall_multiseed/mqar_query3_delay96_ctx32/summary.json \
		--seeds 11,17,23 \
		--train-records 2048 \
		--eval-records 1024 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 6 \
		--query-count 3 \
		--delay 96 \
		--transformer-context 32 \
		--epochs 40 \
		--batch-size 128 \
		--learning-rate 0.006 \
		--ssm-variant dynamic-kv-memory \
		--kv-logit-scale 64 \
		--ssm-decay-init 8.0 \
		--min-seeds 3 \
		--min-query-positions 2048 \
		--require-valid-protocol \
		--timeout 600

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

neocortex-x-stage0-smoke:
	$(PYTHON) research/neocortex_x/core_experiment.py \
		--stage stage0 \
		--seed 11 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--delay 24 \
		--baseline-window 8 \
		--output-json artifacts/neocortex_x/stage0_smoke.json \
		--output-trace artifacts/neocortex_x/stage0_trace.jsonl

neocortex-x-stage1-smoke:
	$(PYTHON) research/neocortex_x/core_experiment.py \
		--stage stage1 \
		--seeds 11,17,23,29,31,37 \
		--episodes 128 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--delay 24 \
		--baseline-window 8 \
		--output-json artifacts/neocortex_x/stage1_recall_to_action.json

neocortex-x-stage1b-trained-smoke:
	$(PYTHON) research/neocortex_x/stage1b_trained_recall_to_action.py \
		--train-records 512 \
		--eval-records 256 \
		--key-count 16 \
		--value-count 16 \
		--binding-count 4 \
		--delay 32 \
		--transformer-context 8 \
		--epochs 12 \
		--batch-size 64 \
		--learning-rate 0.003 \
		--ssm-variant sparse-kv-memory \
		--ssm-input-dim 48 \
		--ssm-state-dim 64 \
		--init-assignment-kv-memory \
		--kv-read-gate-l1 0.5 \
		--kv-read-gate-l1-start 0.0 \
		--kv-read-gate-l1-ramp-epochs 6 \
		--kv-read-gate-target 0.02 \
		--kv-read-gate-target-weight 5.0 \
		--kv-read-gate-entropy-weight 0.01 \
		--kv-read-gate-query-margin 0.9 \
		--kv-read-gate-query-margin-weight 2.0 \
		--output-json artifacts/neocortex_x/stage1b_trained_recall_to_action.json \
		--inner-json artifacts/neocortex_x/stage1b_inner_long_context_recall.json

neocortex-x-stage2-smoke:
	$(PYTHON) research/neocortex_x/core_experiment.py \
		--stage stage2 \
		--seeds 11,17,23,29,31,37 \
		--episodes 128 \
		--tape-len 8 \
		--input-len-min 4 \
		--input-len-max 8 \
		--max-steps 64 \
		--trace-episodes 2 \
		--output-json artifacts/neocortex_x/stage2_stateful_tape_world.json \
		--output-trace artifacts/neocortex_x/stage2_trace.jsonl

neocortex-x-stage2b-learned-smoke:
	$(PYTHON) research/neocortex_x/stage2b_learned_tape_policy.py \
		--train-seeds 101,103,107 \
		--eval-seeds 11,17,23,29,31,37 \
		--train-episodes 64 \
		--eval-episodes 64 \
		--tape-len 8 \
		--input-len-min 4 \
		--input-len-max 8 \
		--max-steps 64 \
		--output-json artifacts/neocortex_x/stage2b_learned_tape_policy.json

neocortex-x-stage3-smoke:
	$(PYTHON) research/neocortex_x/core_experiment.py \
		--stage stage3 \
		--seeds 11,17,23,29,31,37 \
		--episodes 128 \
		--stage3-tape-len 12 \
		--stage3-segment-len 3 \
		--stage3-call-overhead 1 \
		--output-json artifacts/neocortex_x/stage3_procedure_reuse.json

neocortex-x-stage3b-promotion-smoke:
	$(PYTHON) research/neocortex_x/stage3b_learned_procedure_promotion.py \
		--train-seeds 101,103,107 \
		--cert-seeds 109,113,127 \
		--eval-seeds 11,17,23,29,31,37 \
		--train-episodes 64 \
		--cert-episodes 64 \
		--eval-episodes 96 \
		--tape-len 12 \
		--segment-len 3 \
		--call-overhead 1 \
		--output-json artifacts/neocortex_x/stage3b_learned_procedure_promotion.json

neocortex-x-stage4-smoke:
	$(PYTHON) research/neocortex_x/core_experiment.py \
		--stage stage4 \
		--seeds 11,17,23,29,31,37 \
		--episodes 96 \
		--stage4-corridor-len 9 \
		--stage4-horizon 4 \
		--output-json artifacts/neocortex_x/stage4_planning.json

neocortex-x-stage4b-world-model-smoke:
	$(PYTHON) research/neocortex_x/stage4b_learned_world_model.py \
		--train-seeds 101,103,107 \
		--eval-seeds 11,17,23,29,31,37 \
		--train-episodes 64 \
		--eval-episodes 64 \
		--corridor-len 9 \
		--horizon 4 \
		--output-json artifacts/neocortex_x/stage4b_learned_world_model.json

neocortex-x-stage5-smoke:
	$(PYTHON) research/neocortex_x/core_experiment.py \
		--stage stage5 \
		--seeds 11,17,23,29,31,37 \
		--episodes 96 \
		--stage5-contradiction-margin 0.20 \
		--output-json artifacts/neocortex_x/stage5_open_world_adapter.json

neocortex-x-stage5b-cached-smoke:
	$(PYTHON) research/neocortex_x/stage5b_live_cached_adapter.py \
		--mode cached \
		--fixture-json research/neocortex_x/fixtures/stage5b_cached_observations.json \
		--output-json artifacts/neocortex_x/stage5b_live_cached_adapter.json

neocortex-x-stage6-packets-smoke:
	$(PYTHON) research/neocortex_x/stage6_evidence_packets.py \
		--fixture-json research/neocortex_x/fixtures/stage5b_cached_observations.json \
		--output-json artifacts/neocortex_x/stage6_evidence_packets.json

neocortex-x-stage7-memory-smoke: neocortex-x-stage6-packets-smoke
	$(PYTHON) research/neocortex_x/stage7_memory_retrieval.py \
		--packet-json artifacts/neocortex_x/stage6_evidence_packets.json \
		--output-json artifacts/neocortex_x/stage7_memory_retrieval.json

neocortex-x-stage8-freshness-smoke: neocortex-x-stage6-packets-smoke
	$(PYTHON) research/neocortex_x/stage8_temporal_freshness.py \
		--packet-json artifacts/neocortex_x/stage6_evidence_packets.json \
		--output-json artifacts/neocortex_x/stage8_temporal_freshness.json

neocortex-x-stage9-action-gate-smoke: neocortex-x-stage8-freshness-smoke
	$(PYTHON) research/neocortex_x/stage9_action_gate.py \
		--temporal-json artifacts/neocortex_x/stage8_temporal_freshness.json \
		--output-json artifacts/neocortex_x/stage9_action_gate.json

neocortex-x-stage10-feedback-smoke: neocortex-x-stage9-action-gate-smoke
	$(PYTHON) research/neocortex_x/stage10_feedback_loop.py \
		--action-gate-json artifacts/neocortex_x/stage9_action_gate.json \
		--output-json artifacts/neocortex_x/stage10_feedback_loop.json

neocortex-x-stage11-policy-smoke: neocortex-x-stage10-feedback-smoke
	$(PYTHON) research/neocortex_x/stage11_policy_update.py \
		--feedback-json artifacts/neocortex_x/stage10_feedback_loop.json \
		--output-json artifacts/neocortex_x/stage11_policy_update.json

neocortex-x-stage12-policy-replay-smoke: neocortex-x-stage11-policy-smoke
	$(PYTHON) research/neocortex_x/stage12_policy_replay.py \
		--temporal-json artifacts/neocortex_x/stage8_temporal_freshness.json \
		--policy-json artifacts/neocortex_x/stage11_policy_update.json \
		--output-json artifacts/neocortex_x/stage12_policy_replay.json

neocortex-x-stage13-promote-policy-smoke: neocortex-x-stage12-policy-replay-smoke
	$(PYTHON) research/neocortex_x/stage13_policy_promotion.py \
		--replay-json artifacts/neocortex_x/stage12_policy_replay.json \
		--policy-json artifacts/neocortex_x/stage11_policy_update.json \
		--output-json artifacts/neocortex_x/stage13_policy_promotion.json

neocortex-x-stage14-runtime-loader-smoke: neocortex-x-stage13-promote-policy-smoke
	$(PYTHON) research/neocortex_x/stage14_runtime_loader.py \
		--promotion-json artifacts/neocortex_x/stage13_policy_promotion.json \
		--output-json artifacts/neocortex_x/stage14_runtime_loader.json

neocortex-x-stage15-runtime-execution-smoke: neocortex-x-stage14-runtime-loader-smoke
	$(PYTHON) research/neocortex_x/stage15_runtime_execution.py \
		--runtime-json artifacts/neocortex_x/stage14_runtime_loader.json \
		--temporal-json artifacts/neocortex_x/stage8_temporal_freshness.json \
		--output-json artifacts/neocortex_x/stage15_runtime_execution.json

neocortex-x-stage16-contextual-memory-smoke: neocortex-x-stage6-packets-smoke
	$(PYTHON) research/neocortex_x/stage16_contextual_memory.py \
		--packet-json artifacts/neocortex_x/stage6_evidence_packets.json \
		--output-json artifacts/neocortex_x/stage16_contextual_memory.json

neocortex-x-stage17-contextual-action-gate-smoke: neocortex-x-stage16-contextual-memory-smoke
	$(PYTHON) research/neocortex_x/stage17_contextual_action_gate.py \
		--contextual-json artifacts/neocortex_x/stage16_contextual_memory.json \
		--output-json artifacts/neocortex_x/stage17_contextual_action_gate.json

neocortex-x-stage18-contextual-feedback-smoke: neocortex-x-stage17-contextual-action-gate-smoke
	$(PYTHON) research/neocortex_x/stage18_contextual_feedback.py \
		--contextual-action-json artifacts/neocortex_x/stage17_contextual_action_gate.json \
		--output-json artifacts/neocortex_x/stage18_contextual_feedback.json

neocortex-x-stage19-contextual-policy-smoke: neocortex-x-stage18-contextual-feedback-smoke
	$(PYTHON) research/neocortex_x/stage19_contextual_policy_update.py \
		--feedback-json artifacts/neocortex_x/stage18_contextual_feedback.json \
		--output-json artifacts/neocortex_x/stage19_contextual_policy_update.json

neocortex-x-stage20-contextual-policy-replay-smoke: neocortex-x-stage19-contextual-policy-smoke
	$(PYTHON) research/neocortex_x/stage20_contextual_policy_replay.py \
		--contextual-action-json artifacts/neocortex_x/stage17_contextual_action_gate.json \
		--policy-json artifacts/neocortex_x/stage19_contextual_policy_update.json \
		--output-json artifacts/neocortex_x/stage20_contextual_policy_replay.json

neocortex-x-stage21-contextual-policy-promotion-smoke: neocortex-x-stage20-contextual-policy-replay-smoke
	$(PYTHON) research/neocortex_x/stage21_contextual_policy_promotion.py \
		--replay-json artifacts/neocortex_x/stage20_contextual_policy_replay.json \
		--policy-json artifacts/neocortex_x/stage19_contextual_policy_update.json \
		--output-json artifacts/neocortex_x/stage21_contextual_policy_promotion.json

neocortex-x-stage22-contextual-runtime-loader-smoke: neocortex-x-stage21-contextual-policy-promotion-smoke
	$(PYTHON) research/neocortex_x/stage22_contextual_runtime_loader.py \
		--promotion-json artifacts/neocortex_x/stage21_contextual_policy_promotion.json \
		--output-json artifacts/neocortex_x/stage22_contextual_runtime_loader.json

neocortex-x-stage23-contextual-runtime-execution-smoke: neocortex-x-stage22-contextual-runtime-loader-smoke
	$(PYTHON) research/neocortex_x/stage23_contextual_runtime_execution.py \
		--runtime-json artifacts/neocortex_x/stage22_contextual_runtime_loader.json \
		--contextual-json artifacts/neocortex_x/stage16_contextual_memory.json \
		--output-json artifacts/neocortex_x/stage23_contextual_runtime_execution.json

neocortex-x-stage24-contextual-self-monitor-smoke: neocortex-x-stage23-contextual-runtime-execution-smoke
	$(PYTHON) research/neocortex_x/stage24_contextual_self_monitor.py \
		--execution-json artifacts/neocortex_x/stage23_contextual_runtime_execution.json \
		--output-json artifacts/neocortex_x/stage24_contextual_self_monitor.json

neocortex-x-stage25-indexed-contextual-store-smoke: neocortex-x-stage16-contextual-memory-smoke
	$(PYTHON) research/neocortex_x/stage25_indexed_contextual_memory_store.py \
		--contextual-json artifacts/neocortex_x/stage16_contextual_memory.json \
		--output-db artifacts/neocortex_x/stage25_contextual_memory.sqlite \
		--output-json artifacts/neocortex_x/stage25_indexed_contextual_memory_store.json

neocortex-x-stage26-store-backed-contextual-retrieval-smoke: neocortex-x-stage25-indexed-contextual-store-smoke
	$(PYTHON) research/neocortex_x/stage26_store_backed_contextual_retrieval.py \
		--store-json artifacts/neocortex_x/stage25_indexed_contextual_memory_store.json \
		--store-db artifacts/neocortex_x/stage25_contextual_memory.sqlite \
		--output-json artifacts/neocortex_x/stage26_store_backed_contextual_retrieval.json

neocortex-x-stage27-store-backed-contextual-action-gate-smoke: neocortex-x-stage26-store-backed-contextual-retrieval-smoke
	$(PYTHON) research/neocortex_x/stage27_store_backed_contextual_action_gate.py \
		--retrieval-json artifacts/neocortex_x/stage26_store_backed_contextual_retrieval.json \
		--output-json artifacts/neocortex_x/stage27_store_backed_contextual_action_gate.json

neocortex-x-stage28-evidence-log-compaction-smoke: neocortex-x-stage16-contextual-memory-smoke
	$(PYTHON) research/neocortex_x/stage28_append_only_evidence_log_compaction.py \
		--contextual-json artifacts/neocortex_x/stage16_contextual_memory.json \
		--output-db artifacts/neocortex_x/stage28_evidence_log_compacted_store.sqlite \
		--output-json artifacts/neocortex_x/stage28_append_only_evidence_log_compaction.json

neocortex-x-stage29-scale-probe-smoke:
	$(PYTHON) research/neocortex_x/stage29_contextual_memory_scale_probe.py \
		--records 5000 \
		--queries 256 \
		--json-scan-queries 64 \
		--output-db artifacts/neocortex_x/stage29_scale_probe.sqlite \
		--output-json artifacts/neocortex_x/stage29_contextual_memory_scale_probe.json

neocortex-x-stage30-partial-context-smoke:
	$(PYTHON) research/neocortex_x/stage30_partial_context_scale_retrieval.py \
		--records 5000 \
		--queries 256 \
		--json-scan-queries 64 \
		--candidate-limit 64 \
		--output-db artifacts/neocortex_x/stage30_partial_context.sqlite \
		--output-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json

neocortex-x-stage31-disambiguation-gate-smoke: neocortex-x-stage30-partial-context-smoke
	$(PYTHON) research/neocortex_x/stage31_partial_context_disambiguation_gate.py \
		--partial-context-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json \
		--output-json artifacts/neocortex_x/stage31_partial_context_disambiguation_gate.json

neocortex-x-stage32-clarification-resolution-smoke: neocortex-x-stage31-disambiguation-gate-smoke
	$(PYTHON) research/neocortex_x/stage32_clarification_resolution_loop.py \
		--partial-context-json artifacts/neocortex_x/stage30_partial_context_scale_retrieval.json \
		--disambiguation-json artifacts/neocortex_x/stage31_partial_context_disambiguation_gate.json \
		--output-json artifacts/neocortex_x/stage32_clarification_resolution_loop.json

neocortex-x-stage33-clarification-feedback-smoke: neocortex-x-stage32-clarification-resolution-smoke
	$(PYTHON) research/neocortex_x/stage33_clarification_feedback_ledger.py \
		--resolution-json artifacts/neocortex_x/stage32_clarification_resolution_loop.json \
		--output-json artifacts/neocortex_x/stage33_clarification_feedback_ledger.json

neocortex-x-stage34-clarification-policy-smoke: neocortex-x-stage33-clarification-feedback-smoke
	$(PYTHON) research/neocortex_x/stage34_clarification_policy_update.py \
		--feedback-json artifacts/neocortex_x/stage33_clarification_feedback_ledger.json \
		--output-json artifacts/neocortex_x/stage34_clarification_policy_update.json

clean-artifacts:
	rm -rf artifacts/verify artifacts/benchmarks artifacts/trajectories
