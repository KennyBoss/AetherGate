#!/usr/bin/env python3
"""End-to-end smoke verifier for the SoA/SSM prototype stack."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "verify"


class VerificationError(RuntimeError):
    pass


def run_cmd(args: list[str], timeout: int) -> str:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise VerificationError(
            f"Command failed with exit code {proc.returncode}: {' '.join(args)}\n{proc.stdout}"
        )
    return proc.stdout


def require(pattern: str, text: str, label: str) -> re.Match[str]:
    match = re.search(pattern, text)
    if not match:
        raise VerificationError(f"Missing {label}: pattern {pattern!r}\n{text}")
    return match


def parse_float(pattern: str, text: str, label: str) -> float:
    return float(require(pattern, text, label).group(1))


def parse_distance_pair(label: str, text: str) -> tuple[float, float]:
    match = require(rf"{label}:\s+([0-9.]+)\s+->\s+([0-9.]+)", text, label)
    return float(match.group(1)), float(match.group(2))


def pass_line(name: str, detail: str) -> None:
    print(f"PASS {name}: {detail}")


def verify_compile(timeout: int) -> None:
    scripts = [
        "agent_ssm_core.py",
        "analyze_text_release_history.py",
        "text_tokenization.py",
        "soa_ssm_agents.py",
        "train_ssm_text.py",
        "ingest_text_data.py",
        "materialize_text_corpus.py",
        "plan_text_sweep.py",
        "profile_text_corpus.py",
        "run_text_checkpoint.py",
        "probe_text_memory.py",
        "compare_text_memory.py",
        "run_text_memory_suite.py",
        "validate_text_memory_suite.py",
        "benchmark_text_package.py",
        "compare_text_checkpoints.py",
        "compare_text_architectures.py",
        "run_text_architecture_multiseed.py",
        "compare_long_context_recall.py",
        "compare_text_packages.py",
        "compare_text_releases.py",
        "compare_text_release_dashboards.py",
        "summarize_text_release_gates.py",
        "generate_text_release_report.py",
        "bundle_text_release_evidence.py",
        "validate_text_release_evidence.py",
        "compare_text_release_evidence.py",
        "unbundle_text_release_evidence.py",
        "register_text_release_evidence.py",
        "validate_text_release_evidence_ledger.py",
        "tamper_text_release_evidence_ledger.py",
        "replay_text_release_evidence_ledger.py",
        "compare_text_release_evidence_ledgers.py",
        "summarize_text_release_evidence_audit.py",
        "validate_text_release_evidence_audit.py",
        "compare_text_release_evidence_audit_validations.py",
        "compare_text_release_evidence_audits.py",
        "register_text_package.py",
        "list_text_releases.py",
        "select_text_release.py",
        "build_text_release_dashboard.py",
        "build_text_release_history.py",
        "validate_text_release_dashboard.py",
        "validate_text_release_history.py",
        "validate_text_release_pipeline.py",
        "compare_text_release_pipeline_validations.py",
        "compare_text_release_pipelines.py",
        "validate_text_release_pipeline_comparison.py",
        "compare_text_release_pipeline_comparison_validations.py",
        "release_text_pipeline.py",
        "run_text_experiment.py",
        "list_text_experiments.py",
        "sweep_text_experiments.py",
        "promote_text_checkpoint.py",
        "inspect_text_checkpoint.py",
        "validate_text_package.py",
        "bundle_text_package.py",
        "unbundle_text_package.py",
        "evolve_ssm_agents.py",
        "evolve_ssm_formulas.py",
        "evolve_code_blocks.py",
        "evolve_code_tape.py",
        "code_tape_prior.py",
        "train_code_tape_prior.py",
        "search_code_tape_prior.py",
        "promote_code_tape_macros.py",
        "compare_code_block_library.py",
        "code_block_skill_ladder.py",
        "compare_code_block_hierarchy.py",
        "compare_code_block_stage4b.py",
        "analyze_code_block_pruning.py",
        "analyze_code_tape_late_game.py",
        "code_tape_late_game_features.py",
        "train_code_tape_late_game_ranker.py",
        "rank_code_tape_late_game.py",
        "compare_code_tape_late_game_guidance.py",
        "search_code_tape_trajectory.py",
        "run_hypothesis_loop.py",
        "coevolve_ssm_selfplay.py",
        "run_evolved_agent.py",
        "export_agent_trajectory.py",
        "verify_system.py",
        "benchmark_system.py",
        "compare_benchmarks.py",
    ]
    run_cmd([sys.executable, "-m", "py_compile", *scripts], timeout)
    pass_line("compile", f"{len(scripts)} scripts")


def verify_agent_runtime(timeout: int) -> None:
    out = run_cmd(
        [
            sys.executable,
            "soa_ssm_agents.py",
            "--agents",
            "1024",
            "--steps",
            "16",
        ],
        timeout,
    )
    start_goal, end_goal = parse_distance_pair("mean_goal_distance", out)
    start_threat, end_threat = parse_distance_pair("mean_threat_distance", out)
    if end_goal >= start_goal:
        raise VerificationError(f"Agent runtime did not improve goal distance:\n{out}")
    if end_threat <= start_threat:
        raise VerificationError(f"Agent runtime did not improve threat distance:\n{out}")
    pass_line("agent-runtime", f"goal {start_goal:.4f}->{end_goal:.4f}, threat {start_threat:.4f}->{end_threat:.4f}")


def verify_text_checkpoint(timeout: int) -> None:
    checkpoint = ARTIFACT_DIR / "text_verify.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    out = run_cmd(
        [
            sys.executable,
            "train_ssm_text.py",
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--epochs",
            "2",
            "--log-every",
            "1",
            "--sample-steps",
            "6",
            "--save-checkpoint",
            str(checkpoint),
        ],
        timeout,
    )
    initial_loss = parse_float(r"initial_loss:\s+([0-9.]+)", out, "initial text loss")
    final_loss = parse_float(r"final_loss:\s+([0-9.]+)", out, "final text loss")
    if final_loss >= initial_loss:
        raise VerificationError(f"Text training did not reduce loss:\n{out}")
    if not checkpoint.exists():
        raise VerificationError(f"Checkpoint was not created: {checkpoint}")

    reload_out = run_cmd(
        [
            sys.executable,
            "train_ssm_text.py",
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--epochs",
            "99",
            "--eval-only",
            "--sample-steps",
            "6",
            "--load-checkpoint",
            str(checkpoint),
        ],
        timeout,
    )
    require(r"loaded_checkpoint:", reload_out, "checkpoint reload marker")
    require(r"eval_only: skipped training updates", reload_out, "eval-only marker")
    reloaded_loss = parse_float(r"final_loss:\s+([0-9.]+)", reload_out, "reloaded final loss")
    if abs(reloaded_loss - final_loss) > 1.0e-3:
        raise VerificationError(
            f"Reloaded checkpoint loss changed unexpectedly: saved {final_loss}, reload {reloaded_loss}"
        )
    pass_line("text-checkpoint", f"loss {initial_loss:.4f}->{final_loss:.4f}, reload {reloaded_loss:.4f}")


def verify_text_checkpoint_runner(timeout: int) -> None:
    checkpoint = ARTIFACT_DIR / "text_verify.npz"
    if not checkpoint.exists():
        verify_text_checkpoint(timeout)
    output_json = ARTIFACT_DIR / "text_run_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "run_text_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--sample-steps",
            "6",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text checkpoint runner", out, "text runner marker")
    loss = parse_float(r"loss:\s+([0-9.]+)", out, "text runner loss")
    accuracy = parse_float(r"accuracy:\s+([0-9.]+)", out, "text runner accuracy")
    if loss <= 0 or accuracy < 0:
        raise VerificationError(f"Unexpected text checkpoint runner metrics:\n{out}")
    if not output_json.exists():
        raise VerificationError(f"Text runner JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if not payload.get("sample"):
        raise VerificationError(f"Text runner JSON does not include a sample: {payload}")
    pass_line("text-runner", f"loss {loss:.4f}, accuracy {accuracy:.4f}")


def verify_text_corpus_profiler(timeout: int) -> None:
    checkpoint = ARTIFACT_DIR / "text_verify.npz"
    if not checkpoint.exists():
        verify_text_checkpoint(timeout)
    output_json = ARTIFACT_DIR / "text_profile_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "profile_text_corpus.py",
            "--checkpoint",
            str(checkpoint),
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA corpus profiler", out, "corpus profiler marker")
    require(r"Recommended training args", out, "training recommendation marker")
    if not output_json.exists():
        raise VerificationError(f"Corpus profiler JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if payload["tokens"] < 16:
        raise VerificationError(f"Corpus profiler reported too few tokens: {payload}")
    recommendation = payload["recommended_training_args"]
    if recommendation["seq_len"] + 1 > recommendation["tokens_per_stream"]:
        raise VerificationError(f"Invalid profiler recommendation: {recommendation}")
    if payload["checkpoint_oov"] is None:
        raise VerificationError(f"Profiler did not include checkpoint OOV report: {payload}")
    pass_line(
        "text-profiler",
        f"tokens {payload['tokens']:,}, vocab {payload['vocab_size_with_specials']:,}, seq_len {recommendation['seq_len']}",
    )


def verify_text_ingestion_and_bpe(timeout: int) -> None:
    input_text = ARTIFACT_DIR / "bpe_seed.txt"
    output_dir = ARTIFACT_DIR / "corpus_local"
    output_text = output_dir / "corpus.txt"
    output_json = output_dir / "corpus_manifest.json"
    input_text.write_text(
        "\n".join(
            [
                "State space memory systems read recurring fragments.",
                "Fragments become subword units, and subword units carry reusable structure.",
                "Verification keeps the experiment honest when the corpus changes.",
                "Memory systems read fragments, compare fragments, and compress fragments.",
            ]
        ),
        encoding="utf-8",
    )
    out = run_cmd(
        [
            sys.executable,
            "ingest_text_data.py",
            "--source",
            "local",
            "--input-text",
            str(input_text),
            "--output-dir",
            str(output_dir),
            "--output-text",
            str(output_text),
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA data ingestion", out, "data ingestion marker")
    for path in [output_text, output_json]:
        if not path.exists():
            raise VerificationError(f"Ingestion artifact was not created: {path}")
    ingest = json.loads(output_json.read_text(encoding="utf-8"))
    if ingest["stats"]["word_tokens"] < 24:
        raise VerificationError(f"Ingested corpus is unexpectedly small: {ingest}")

    checkpoint = ARTIFACT_DIR / "text_bpe_verify.npz"
    train_out = run_cmd(
        [
            sys.executable,
            "train_ssm_text.py",
            "--text-file",
            str(output_text),
            "--tokenizer",
            "bpe",
            "--bpe-vocab-size",
            "96",
            "--bpe-min-frequency",
            "2",
            "--bpe-train-chars",
            "10000",
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--epochs",
            "2",
            "--sample-steps",
            "6",
            "--save-checkpoint",
            str(checkpoint),
        ],
        timeout,
    )
    require(r"tokenizer:\s+bpe", train_out, "BPE train marker")
    require(r"bpe_merges:\s+[1-9]", train_out, "BPE merge marker")
    if not checkpoint.exists():
        raise VerificationError(f"BPE checkpoint was not created: {checkpoint}")

    profile_json = output_dir / "profile_bpe.json"
    profile_out = run_cmd(
        [
            sys.executable,
            "profile_text_corpus.py",
            "--text-file",
            str(output_text),
            "--checkpoint",
            str(checkpoint),
            "--output-json",
            str(profile_json),
        ],
        timeout,
    )
    require(r"tokenizer:\s+bpe", profile_out, "BPE profiler marker")
    profile = json.loads(profile_json.read_text(encoding="utf-8"))
    if profile["checkpoint_oov"]["oov_token_rate"] > 0.05:
        raise VerificationError(f"BPE checkpoint has unexpected OOV rate: {profile['checkpoint_oov']}")

    run_json = output_dir / "run_bpe.json"
    run_out = run_cmd(
        [
            sys.executable,
            "run_text_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--text-file",
            str(output_text),
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--sample-steps",
            "6",
            "--output-json",
            str(run_json),
        ],
        timeout,
    )
    require(r"tokenizer:\s+bpe", run_out, "BPE runner marker")
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    if payload["tokenizer"] != "bpe" or not payload.get("sample"):
        raise VerificationError(f"BPE runner JSON is invalid: {payload}")
    pass_line(
        "text-ingest-bpe",
        f"tokens {profile['tokens']:,}, merges {profile['tokenizer_summary']['merges']}, oov {profile['checkpoint_oov']['oov_token_rate']:.4f}",
    )


def train_text_checkpoint(path: Path, epochs: int, timeout: int) -> float:
    out = run_cmd(
        [
            sys.executable,
            "train_ssm_text.py",
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--epochs",
            str(epochs),
            "--log-every",
            "1",
            "--sample-steps",
            "4",
            "--save-checkpoint",
            str(path),
        ],
        timeout,
    )
    if not path.exists():
        raise VerificationError(f"Text checkpoint was not created: {path}")
    return parse_float(r"final_loss:\s+([0-9.]+)", out, f"text checkpoint loss {path}")


def verify_text_checkpoint_comparison(timeout: int) -> None:
    baseline = ARTIFACT_DIR / "text_compare_baseline.npz"
    candidate = ARTIFACT_DIR / "text_compare_candidate.npz"
    output_json = ARTIFACT_DIR / "text_compare_verify.json"
    baseline_loss = train_text_checkpoint(baseline, 1, timeout)
    candidate_loss = train_text_checkpoint(candidate, 2, timeout)
    if candidate_loss > baseline_loss:
        raise VerificationError(
            f"Candidate text checkpoint is worse before comparison: {candidate_loss} > {baseline_loss}"
        )

    out = run_cmd(
        [
            sys.executable,
            "compare_text_checkpoints.py",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--sample-steps",
            "4",
            "--fail-on-loss-regression",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text checkpoint comparison", out, "text comparison marker")
    require(r"winner_by_loss:\s+candidate", out, "candidate comparison winner")
    if not output_json.exists():
        raise VerificationError(f"Text comparison JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    if comparison["loss_delta"] > 0:
        raise VerificationError(f"Comparison reported a loss regression: {comparison}")
    pass_line(
        "text-compare",
        f"loss {baseline_loss:.4f}->{candidate_loss:.4f}, delta {comparison['loss_delta']:.4f}",
    )


def verify_text_architecture_comparison(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "text_architecture_compare_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_architectures.py",
            "--output-json",
            str(output_json),
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--epochs",
            "1",
            "--repeat",
            "1",
            "--sample-steps",
            "4",
        ],
        timeout,
    )
    require(r"TextPy/SoA architecture comparison", out, "architecture comparison marker")
    require(r"winner_by_loss:\s+(ssm|transformer|tie)", out, "architecture comparison winner")
    if not output_json.exists():
        raise VerificationError(f"Architecture comparison JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    ssm = payload["ssm"]
    transformer = payload["transformer"]
    comparison = payload["comparison"]
    if protocol["train_tokens"] <= 0 or protocol["eval_tokens"] <= 0:
        raise VerificationError(f"Invalid architecture comparison token split: {protocol}")
    if ssm["parameter_count"] <= 0 or transformer["parameter_count"] <= 0:
        raise VerificationError(f"Invalid architecture parameter counts: {ssm}, {transformer}")
    if comparison["winner_by_loss"] not in {"ssm", "transformer", "tie"}:
        raise VerificationError(f"Invalid architecture comparison winner: {comparison}")
    pass_line(
        "text-architecture-compare",
        (
            f"winner {comparison['winner_by_loss']}, "
            f"ssm_loss {ssm['eval']['loss']:.4f}, "
            f"transformer_loss {transformer['eval']['loss']:.4f}, "
            f"param_ratio {protocol['transformer_to_ssm_param_ratio']:.3f}"
        ),
    )


def verify_text_architecture_multiseed(timeout: int) -> None:
    output_dir = ARTIFACT_DIR / "text_architecture_multiseed"
    output_json = output_dir / "summary.json"
    out = run_cmd(
        [
            sys.executable,
            "run_text_architecture_multiseed.py",
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(output_json),
            "--seeds",
            "11,17",
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--epochs",
            "1",
            "--repeat",
            "1",
            "--sample-steps",
            "4",
            "--min-seeds",
            "2",
            "--min-train-positions",
            "256",
            "--min-eval-positions",
            "256",
            "--min-corpus-train-tokens",
            "64",
            "--min-corpus-eval-tokens",
            "16",
            "--require-valid-protocol",
        ],
        timeout,
    )
    require(r"TextPy/SoA multi-seed architecture comparison", out, "multi-seed architecture marker")
    require(r"protocol_valid:\s+True", out, "multi-seed protocol validity")
    if not output_json.exists():
        raise VerificationError(f"Architecture multi-seed JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol_evidence"]
    summary = payload["summary"]
    if protocol["valid"] is not True or protocol["repeated_seed_count"] != 2:
        raise VerificationError(f"Invalid multi-seed protocol evidence: {protocol}")
    winners = summary["comparison"]["winner_counts"]
    if sum(int(count) for count in winners.values()) != 2:
        raise VerificationError(f"Multi-seed winner counts do not match seeds: {winners}")
    if float(summary["memory_context"]["transformer_attention_to_ssm_state_memory_ratio"]["mean"]) <= 0:
        raise VerificationError(f"Invalid multi-seed memory summary: {summary['memory_context']}")
    pass_line(
        "text-architecture-multiseed",
        (
            f"seeds {protocol['repeated_seed_count']}, "
            f"positions {protocol['train_positions_per_epoch']}, "
            f"winners {winners}"
        ),
    )


def verify_long_context_recall(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "long_context_recall_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_long_context_recall.py",
            "--output-json",
            str(output_json),
            "--train-records",
            "256",
            "--eval-records",
            "128",
            "--key-count",
            "16",
            "--delay",
            "96",
            "--transformer-context",
            "32",
            "--epochs",
            "40",
            "--batch-size",
            "128",
            "--learning-rate",
            "0.006",
            "--ssm-variant",
            "token-memory",
            "--ssm-decay-init",
            "8.0",
        ],
        timeout,
    )
    require(r"TextPy/SoA long-context delayed-recall comparison", out, "long-context recall marker")
    require(r"winner_by_recall_accuracy:\s+(ssm|transformer|tie)", out, "long-context recall winner")
    if not output_json.exists():
        raise VerificationError(f"Long-context recall JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    comparison = payload["comparison"]
    if protocol["delay"] <= protocol["transformer_context"]:
        raise VerificationError(f"Delayed recall is not beyond the Transformer cap: {protocol}")
    for label in ["ssm", "transformer"]:
        acc = float(payload[label]["eval"]["recall_accuracy"])
        loss = float(payload[label]["eval"]["recall_loss"])
        if not (0.0 <= acc <= 1.0) or loss <= 0:
            raise VerificationError(f"Invalid {label} delayed-recall metrics: {payload[label]['eval']}")
    if comparison["winner_by_recall_accuracy"] not in {"ssm", "transformer", "tie"}:
        raise VerificationError(f"Invalid delayed-recall comparison: {comparison}")
    if comparison["winner_by_recall_accuracy"] != "ssm":
        raise VerificationError(f"Token-memory delayed recall should beat the capped Transformer: {comparison}")
    if float(payload["ssm"]["eval"]["recall_accuracy"]) < 0.99:
        raise VerificationError(f"Token-memory delayed recall did not reach near-perfect recall: {payload['ssm']['eval']}")
    pass_line(
        "long-context-recall",
        (
            f"delay {protocol['delay']}, cap {protocol['transformer_context']}, "
            f"winner {comparison['winner_by_recall_accuracy']}, "
            f"ssm_acc {payload['ssm']['eval']['recall_accuracy']:.4f}, "
            f"transformer_acc {payload['transformer']['eval']['recall_accuracy']:.4f}"
        ),
    )


def verify_assignment_recall(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "assignment_recall_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_long_context_recall.py",
            "--task",
            "assignment",
            "--output-json",
            str(output_json),
            "--train-records",
            "1024",
            "--eval-records",
            "512",
            "--key-count",
            "16",
            "--value-count",
            "16",
            "--delay",
            "96",
            "--transformer-context",
            "32",
            "--epochs",
            "40",
            "--batch-size",
            "128",
            "--learning-rate",
            "0.006",
            "--ssm-variant",
            "kv-memory",
            "--kv-logit-scale",
            "64",
            "--ssm-decay-init",
            "8.0",
        ],
        timeout,
    )
    require(r"TextPy/SoA long-context delayed-recall comparison", out, "assignment recall marker")
    if not output_json.exists():
        raise VerificationError(f"Assignment recall JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    comparison = payload["comparison"]
    ssm_acc = float(payload["ssm"]["eval"]["recall_accuracy"])
    transformer_acc = float(payload["transformer"]["eval"]["recall_accuracy"])
    if protocol["task"] != "assignment":
        raise VerificationError(f"Assignment recall verifier ran the wrong task: {protocol}")
    if protocol["delay"] <= protocol["transformer_context"]:
        raise VerificationError(f"Assignment recall is not beyond the Transformer cap: {protocol}")
    if comparison["winner_by_recall_accuracy"] != "ssm":
        raise VerificationError(f"Token-memory assignment recall should beat the capped Transformer: {comparison}")
    if ssm_acc < 0.99 or ssm_acc - transformer_acc < 0.80:
        raise VerificationError(
            f"Assignment recall transfer is too weak: ssm={ssm_acc:.4f}, transformer={transformer_acc:.4f}"
        )
    pass_line(
        "assignment-recall",
        (
            f"delay {protocol['delay']}, cap {protocol['transformer_context']}, "
            f"ssm_acc {ssm_acc:.4f}, transformer_acc {transformer_acc:.4f}"
        ),
    )


def verify_multi_assignment_recall(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "multi_assignment_recall_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_long_context_recall.py",
            "--task",
            "multi-assignment",
            "--output-json",
            str(output_json),
            "--train-records",
            "1024",
            "--eval-records",
            "512",
            "--key-count",
            "16",
            "--value-count",
            "16",
            "--binding-count",
            "4",
            "--delay",
            "96",
            "--transformer-context",
            "32",
            "--epochs",
            "40",
            "--batch-size",
            "128",
            "--learning-rate",
            "0.006",
            "--ssm-variant",
            "kv-memory",
            "--kv-logit-scale",
            "64",
            "--ssm-decay-init",
            "8.0",
        ],
        timeout,
    )
    require(r"TextPy/SoA long-context delayed-recall comparison", out, "multi-assignment recall marker")
    if not output_json.exists():
        raise VerificationError(f"Multi-assignment recall JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    comparison = payload["comparison"]
    ssm_acc = float(payload["ssm"]["eval"]["recall_accuracy"])
    transformer_acc = float(payload["transformer"]["eval"]["recall_accuracy"])
    if protocol["task"] != "multi-assignment":
        raise VerificationError(f"Multi-assignment verifier ran the wrong task: {protocol}")
    if int(protocol.get("binding_count", 0)) < 4:
        raise VerificationError(f"Multi-assignment verifier did not test multiple bindings: {protocol}")
    if protocol["delay"] <= protocol["transformer_context"]:
        raise VerificationError(f"Multi-assignment recall is not beyond the Transformer cap: {protocol}")
    if comparison["winner_by_recall_accuracy"] != "ssm":
        raise VerificationError(f"KV-memory multi-assignment recall should beat the capped Transformer: {comparison}")
    if ssm_acc < 0.99 or ssm_acc - transformer_acc < 0.80:
        raise VerificationError(
            f"Multi-assignment recall transfer is too weak: ssm={ssm_acc:.4f}, transformer={transformer_acc:.4f}"
        )
    pass_line(
        "multi-assignment-recall",
        (
            f"bindings {protocol['binding_count']}, delay {protocol['delay']}, "
            f"cap {protocol['transformer_context']}, "
            f"ssm_acc {ssm_acc:.4f}, transformer_acc {transformer_acc:.4f}"
        ),
    )


def verify_multi_query_assignment_recall(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "multi_query_assignment_recall_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_long_context_recall.py",
            "--task",
            "multi-query-assignment",
            "--output-json",
            str(output_json),
            "--train-records",
            "512",
            "--eval-records",
            "256",
            "--key-count",
            "16",
            "--value-count",
            "16",
            "--binding-count",
            "4",
            "--query-count",
            "2",
            "--delay",
            "96",
            "--transformer-context",
            "32",
            "--epochs",
            "40",
            "--batch-size",
            "128",
            "--learning-rate",
            "0.006",
            "--ssm-variant",
            "kv-memory",
            "--kv-logit-scale",
            "64",
            "--ssm-decay-init",
            "8.0",
        ],
        timeout,
    )
    require(r"TextPy/SoA long-context delayed-recall comparison", out, "multi-query assignment marker")
    if not output_json.exists():
        raise VerificationError(f"Multi-query assignment JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    comparison = payload["comparison"]
    ssm_acc = float(payload["ssm"]["eval"]["recall_accuracy"])
    transformer_acc = float(payload["transformer"]["eval"]["recall_accuracy"])
    query_count = int(payload["ssm"]["eval"]["query_count"])
    expected_queries = int(protocol["eval_records"]) * int(protocol["query_count"])
    if protocol["task"] != "multi-query-assignment":
        raise VerificationError(f"Multi-query assignment verifier ran the wrong task: {protocol}")
    if int(protocol.get("binding_count", 0)) < 4 or int(protocol.get("query_count", 0)) < 2:
        raise VerificationError(f"Multi-query assignment verifier did not test enough bindings/queries: {protocol}")
    if query_count != expected_queries:
        raise VerificationError(f"Multi-query assignment counted wrong query positions: {query_count} != {expected_queries}")
    if protocol["delay"] <= protocol["transformer_context"]:
        raise VerificationError(f"Multi-query assignment recall is not beyond the Transformer cap: {protocol}")
    if comparison["winner_by_recall_accuracy"] != "ssm":
        raise VerificationError(f"KV-memory multi-query recall should beat the capped Transformer: {comparison}")
    if ssm_acc < 0.99 or ssm_acc - transformer_acc < 0.80:
        raise VerificationError(
            f"Multi-query assignment recall is too weak: ssm={ssm_acc:.4f}, transformer={transformer_acc:.4f}"
        )
    pass_line(
        "multi-query-assignment-recall",
        (
            f"bindings {protocol['binding_count']}, queries {protocol['query_count']}, "
            f"delay {protocol['delay']}, cap {protocol['transformer_context']}, "
            f"ssm_acc {ssm_acc:.4f}, transformer_acc {transformer_acc:.4f}"
        ),
    )


def verify_long_context_recall_multiseed_runner(timeout: int) -> None:
    output_dir = ARTIFACT_DIR / "long_context_recall_multiseed"
    output_json = output_dir / "summary.json"
    out = run_cmd(
        [
            sys.executable,
            "run_long_context_recall_multiseed.py",
            "--task",
            "mixed-code",
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(output_json),
            "--seeds",
            "11",
            "--train-records",
            "128",
            "--eval-records",
            "64",
            "--key-count",
            "8",
            "--value-count",
            "8",
            "--query-count",
            "2",
            "--delay",
            "40",
            "--transformer-context",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--min-seeds",
            "1",
            "--min-query-positions",
            "128",
            "--require-valid-protocol",
            "--timeout",
            str(timeout),
        ],
        timeout,
    )
    require(r"TextPy/SoA multi-seed long-context recall comparison", out, "long-context multiseed marker")
    require(r"protocol_valid:\s+True", out, "long-context multiseed protocol validity")
    if not output_json.exists():
        raise VerificationError(f"Long-context multiseed JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    evidence = payload["protocol_evidence"]
    if payload["seed_count"] != 1 or payload["seeds"] != [11]:
        raise VerificationError(f"Long-context multiseed runner recorded wrong seeds: {payload['seeds']}")
    if not evidence["valid"] or evidence["min_query_positions"] < 128:
        raise VerificationError(f"Invalid long-context multiseed protocol evidence: {evidence}")
    pass_line(
        "long-context-recall-multiseed-runner",
        f"task {payload['task']}, seeds {payload['seed_count']}, protocol_valid {evidence['valid']}",
    )


def verify_generated_python_code_recall(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "generated_python_code_recall_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_long_context_recall.py",
            "--task",
            "generated-python-code",
            "--output-json",
            str(output_json),
            "--train-records",
            "128",
            "--eval-records",
            "64",
            "--key-count",
            "8",
            "--value-count",
            "8",
            "--query-count",
            "2",
            "--delay",
            "64",
            "--transformer-context",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--ssm-variant",
            "kv-memory",
            "--kv-logit-scale",
            "64",
            "--ssm-decay-init",
            "8.0",
        ],
        timeout,
    )
    require(r"TextPy/SoA long-context delayed-recall comparison", out, "generated Python code recall marker")
    if not output_json.exists():
        raise VerificationError(f"Generated Python code recall JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    if protocol["task"] != "generated-python-code":
        raise VerificationError(f"Generated Python code verifier ran the wrong task: {protocol}")
    if protocol["query_count"] != 2 or payload["ssm"]["eval"]["query_count"] != 128:
        raise VerificationError(f"Generated Python code query positions were counted incorrectly: {payload}")
    if not protocol["bypass_mode"] or protocol["delay"] <= protocol["transformer_context"]:
        raise VerificationError(f"Generated Python code verifier did not exercise the bypass protocol: {protocol}")
    pass_line(
        "generated-python-code-recall",
        f"queries {payload['ssm']['eval']['query_count']}, vocab {protocol['vocab_size']}",
    )


def verify_real_python_code_recall(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "real_python_code_recall_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_long_context_recall.py",
            "--task",
            "real-python-code",
            "--output-json",
            str(output_json),
            "--train-records",
            "128",
            "--eval-records",
            "64",
            "--key-count",
            "16",
            "--value-count",
            "16",
            "--binding-count",
            "4",
            "--query-count",
            "2",
            "--delay",
            "64",
            "--transformer-context",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--ssm-variant",
            "kv-memory",
            "--kv-logit-scale",
            "64",
            "--ssm-decay-init",
            "8.0",
        ],
        timeout,
    )
    require(r"TextPy/SoA long-context delayed-recall comparison", out, "real Python code recall marker")
    if not output_json.exists():
        raise VerificationError(f"Real Python code recall JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    if protocol["task"] != "real-python-code":
        raise VerificationError(f"Real Python code verifier ran the wrong task: {protocol}")
    if protocol["binding_count"] != 4 or protocol["query_count"] != 2:
        raise VerificationError(f"Real Python code protocol did not keep binding/query counts: {protocol}")
    if payload["ssm"]["eval"]["query_count"] != 128:
        raise VerificationError(f"Real Python code query positions were counted incorrectly: {payload}")
    if not protocol["bypass_mode"] or protocol["delay"] <= protocol["transformer_context"]:
        raise VerificationError(f"Real Python code verifier did not exercise the bypass protocol: {protocol}")
    pass_line(
        "real-python-code-recall",
        f"queries {payload['ssm']['eval']['query_count']}, vocab {protocol['vocab_size']}",
    )


def verify_text_experiment_pipeline(timeout: int) -> None:
    output_dir = ARTIFACT_DIR / "text_experiment"
    out = run_cmd(
        [
            sys.executable,
            "run_text_experiment.py",
            "--output-dir",
            str(output_dir),
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--baseline-epochs",
            "1",
            "--candidate-epochs",
            "2",
            "--timeout",
            str(timeout),
        ],
        timeout,
    )
    require(r"TextPy/SoA text experiment pipeline", out, "text experiment marker")
    experiment_json = output_dir / "experiment.json"
    profile_json = output_dir / "profile.json"
    comparison_json = output_dir / "comparison.json"
    baseline = output_dir / "baseline.npz"
    candidate = output_dir / "candidate.npz"
    for path in [experiment_json, profile_json, comparison_json, baseline, candidate]:
        if not path.exists():
            raise VerificationError(f"Text experiment artifact was not created: {path}")

    experiment = json.loads(experiment_json.read_text(encoding="utf-8"))
    comparison = experiment["comparison_run"]["comparison"]
    split = experiment["split"]
    if split["train_tokens"] <= 0 or split["eval_tokens"] < 16:
        raise VerificationError(f"Invalid text experiment split: {split}")
    if comparison["winner_by_loss"] not in {"baseline", "candidate", "tie"}:
        raise VerificationError(f"Invalid text experiment winner: {comparison}")
    shape = experiment["shape"]
    pass_line(
        "text-experiment",
        f"shape {shape['streams']}x{shape['tokens_per_stream']}, eval {split['eval_tokens']}, winner {comparison['winner_by_loss']}",
    )


def verify_text_experiment_leaderboard(timeout: int) -> None:
    root = ARTIFACT_DIR / "text_experiments_index"
    run_dir = root / "smoke"
    run_cmd(
        [
            sys.executable,
            "run_text_experiment.py",
            "--output-dir",
            str(run_dir),
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--baseline-epochs",
            "1",
            "--candidate-epochs",
            "2",
            "--timeout",
            str(timeout),
        ],
        timeout,
    )
    output_json = root / "leaderboard.json"
    output_csv = root / "leaderboard.csv"
    out = run_cmd(
        [
            sys.executable,
            "list_text_experiments.py",
            "--root",
            str(root),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        timeout,
    )
    require(r"TextPy/SoA text experiment leaderboard", out, "text leaderboard marker")
    require(r"runs:\s+1", out, "text leaderboard run count")
    for path in [output_json, output_csv]:
        if not path.exists():
            raise VerificationError(f"Text leaderboard artifact was not created: {path}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    rows = payload["experiments"]
    if len(rows) != 1:
        raise VerificationError(f"Unexpected leaderboard row count: {len(rows)}")
    row = rows[0]
    if row["winner_by_loss"] not in {"baseline", "candidate", "tie"} or row["eval_tokens"] < 16:
        raise VerificationError(f"Unexpected leaderboard result: {row}")
    pass_line(
        "text-leaderboard",
        f"runs {len(rows)}, eval {row['eval_tokens']}, winner {row['winner_by_loss']}",
    )


def verify_text_experiment_sweep(timeout: int) -> None:
    output_dir = ARTIFACT_DIR / "text_sweep"
    out = run_cmd(
        [
            sys.executable,
            "sweep_text_experiments.py",
            "--output-dir",
            str(output_dir),
            "--clean",
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--state-dims",
            "32,64",
            "--input-dims",
            "48",
            "--learning-rates",
            "0.012",
            "--baseline-epochs",
            "1",
            "--candidate-epochs",
            "2",
            "--timeout",
            str(timeout),
        ],
        timeout,
    )
    require(r"TextPy/SoA text experiment sweep", out, "text sweep marker")
    sweep_json = output_dir / "sweep.json"
    leaderboard_json = output_dir / "leaderboard.json"
    leaderboard_csv = output_dir / "leaderboard.csv"
    for path in [sweep_json, leaderboard_json, leaderboard_csv]:
        if not path.exists():
            raise VerificationError(f"Text sweep artifact was not created: {path}")
    sweep = json.loads(sweep_json.read_text(encoding="utf-8"))
    if sweep["run_count"] != 2:
        raise VerificationError(f"Unexpected sweep run count: {sweep['run_count']}")
    leaderboard = json.loads(leaderboard_json.read_text(encoding="utf-8"))
    rows = leaderboard["experiments"]
    if len(rows) != 2:
        raise VerificationError(f"Unexpected sweep leaderboard rows: {len(rows)}")
    pass_line(
        "text-sweep",
        f"runs {len(rows)}, best_loss {rows[0]['candidate_loss']:.4f}",
    )


def verify_text_checkpoint_promotion(timeout: int) -> None:
    sweep_dir = ARTIFACT_DIR / "text_promotion_sweep"
    promote_dir = ARTIFACT_DIR / "text_promoted"
    run_cmd(
        [
            sys.executable,
            "sweep_text_experiments.py",
            "--output-dir",
            str(sweep_dir),
            "--clean",
            "--streams",
            "4",
            "--tokens-per-stream",
            "96",
            "--seq-len",
            "16",
            "--state-dims",
            "32,64",
            "--input-dims",
            "48",
            "--learning-rates",
            "0.012",
            "--baseline-epochs",
            "1",
            "--candidate-epochs",
            "2",
            "--timeout",
            str(timeout),
        ],
        timeout,
    )
    out = run_cmd(
        [
            sys.executable,
            "promote_text_checkpoint.py",
            "--leaderboard",
            str(sweep_dir / "leaderboard.json"),
            "--output-dir",
            str(promote_dir),
            "--smoke-run",
            "--timeout",
            str(timeout),
        ],
        timeout,
    )
    require(r"TextPy/SoA promoted text checkpoint", out, "promotion marker")
    checkpoint = promote_dir / "text_ssm_promoted.npz"
    manifest_path = promote_dir / "manifest.json"
    smoke_path = promote_dir / "smoke_run.json"
    for path in [checkpoint, manifest_path, smoke_path]:
        if not path.exists():
            raise VerificationError(f"Promotion artifact was not created: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["checkpoint_role"] != "best":
        raise VerificationError(f"Unexpected promoted role: {manifest}")
    if manifest["smoke_run"]["result"]["loss"] <= 0:
        raise VerificationError(f"Invalid promoted smoke result: {manifest['smoke_run']}")
    pass_line(
        "text-promote",
        f"role {manifest['checkpoint_role']}, smoke_loss {manifest['smoke_run']['result']['loss']:.4f}",
    )


def verify_text_checkpoint_inspector(timeout: int) -> None:
    promote_dir = ARTIFACT_DIR / "text_promoted"
    manifest_path = promote_dir / "manifest.json"
    if not manifest_path.exists():
        verify_text_checkpoint_promotion(timeout)
    output_json = promote_dir / "model_card.json"
    output_md = promote_dir / "MODEL_CARD.md"
    out = run_cmd(
        [
            sys.executable,
            "inspect_text_checkpoint.py",
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        timeout,
    )
    require(r"TextPy/SoA text checkpoint inspector", out, "checkpoint inspector marker")
    if not output_json.exists():
        raise VerificationError(f"Checkpoint inspector JSON was not created: {output_json}")
    if not output_md.exists():
        raise VerificationError(f"Checkpoint inspector Markdown was not created: {output_md}")
    report = json.loads(output_json.read_text(encoding="utf-8"))
    if report["param_count"] <= 0:
        raise VerificationError(f"Invalid checkpoint parameter count: {report}")
    for name in ["token_embed", "input_b", "output_c"]:
        if name not in report["params"]:
            raise VerificationError(f"Missing parameter {name}: {report['params']}")
    markdown = output_md.read_text(encoding="utf-8")
    for marker in ["# TextPy/SoA Text SSM Model Card", "## Parameter Shapes", "run_text_checkpoint.py"]:
        if marker not in markdown:
            raise VerificationError(f"Model card Markdown missing {marker!r}: {markdown[:500]}")
    pass_line(
        "text-inspect",
        f"params {report['param_count']:,}, bytes {report['param_bytes_human']}",
    )


def verify_text_manifest_runner(timeout: int) -> None:
    promote_dir = ARTIFACT_DIR / "text_promoted"
    manifest_path = promote_dir / "manifest.json"
    if not manifest_path.exists():
        verify_text_checkpoint_promotion(timeout)
    output_json = promote_dir / "run_from_manifest.json"
    out = run_cmd(
        [
            sys.executable,
            "run_text_checkpoint.py",
            "--manifest",
            str(manifest_path),
            "--sample-steps",
            "6",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text checkpoint runner", out, "manifest runner marker")
    require(r"manifest:", out, "manifest runner manifest line")
    if not output_json.exists():
        raise VerificationError(f"Manifest runner JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if payload["manifest"] != str(manifest_path):
        raise VerificationError(f"Manifest runner did not record manifest path: {payload}")
    if not payload.get("sample"):
        raise VerificationError(f"Manifest runner did not generate sample: {payload}")
    pass_line(
        "text-manifest-run",
        f"loss {payload['loss']:.4f}, sample_len {len(payload['sample'])}",
    )


def verify_text_package_benchmark(timeout: int) -> None:
    promote_dir = ARTIFACT_DIR / "text_promoted"
    manifest_path = promote_dir / "manifest.json"
    if not manifest_path.exists():
        verify_text_checkpoint_promotion(timeout)
    output_json = promote_dir / "benchmark.json"
    out = run_cmd(
        [
            sys.executable,
            "benchmark_text_package.py",
            "--manifest",
            str(manifest_path),
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--sample-steps",
            "4",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text package benchmark", out, "package benchmark marker")
    require(r"matches_manifest:\s+True", out, "package benchmark integrity")
    if not output_json.exists():
        raise VerificationError(f"Package benchmark JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if payload["manifest"] != str(manifest_path):
        raise VerificationError(f"Package benchmark did not record manifest path: {payload}")
    if payload["best_throughput_tokens_s"] <= 0 or payload["loss"] <= 0:
        raise VerificationError(f"Invalid package benchmark metrics: {payload}")
    pass_line(
        "text-package-benchmark",
        f"best {payload['best_throughput_tokens_s']:,.0f} tokens/s, loss {payload['loss']:.4f}",
    )


def verify_text_package_validator(timeout: int) -> None:
    promote_dir = ARTIFACT_DIR / "text_promoted"
    manifest_path = promote_dir / "manifest.json"
    if not (promote_dir / "MODEL_CARD.md").exists():
        verify_text_checkpoint_inspector(timeout)
    output_json = promote_dir / "validation.json"
    out = run_cmd(
        [
            sys.executable,
            "validate_text_package.py",
            "--manifest",
            str(manifest_path),
            "--require-model-card",
            "--require-smoke",
            "--require-benchmark",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text package validator", out, "package validator marker")
    require(r"valid:\s+True", out, "package validator success")
    if not output_json.exists():
        raise VerificationError(f"Package validator JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if not result["valid"] or not result["checkpoint_sha256"]:
        raise VerificationError(f"Invalid package validation result: {result}")
    pass_line(
        "text-validate",
        f"bytes {result['checkpoint_bytes']}, sha256 {result['checkpoint_sha256'][:12]}",
    )


def verify_text_package_bundle(timeout: int) -> None:
    promote_dir = ARTIFACT_DIR / "text_promoted"
    manifest_path = promote_dir / "manifest.json"
    if not (promote_dir / "validation.json").exists():
        verify_text_package_validator(timeout)
    archive = promote_dir / "text_package.tar.gz"
    bundle_json = promote_dir / "bundle_manifest.json"
    out = run_cmd(
        [
            sys.executable,
            "bundle_text_package.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(archive),
            "--output-json",
            str(bundle_json),
            "--root-name",
            "text_promoted",
        ],
        timeout,
    )
    require(r"TextPy/SoA text package bundler", out, "package bundler marker")
    for path in [archive, bundle_json]:
        if not path.exists():
            raise VerificationError(f"Package bundle artifact was not created: {path}")
    bundle = json.loads(bundle_json.read_text(encoding="utf-8"))
    if not bundle["archive_sha256"] or bundle["archive_bytes"] <= 0:
        raise VerificationError(f"Invalid bundle manifest: {bundle}")
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    required = {
        "text_promoted/manifest.json",
        "text_promoted/text_ssm_promoted.npz",
        "text_promoted/model_card.json",
        "text_promoted/MODEL_CARD.md",
        "text_promoted/smoke_run.json",
        "text_promoted/benchmark.json",
        "text_promoted/validation.json",
    }
    missing = sorted(required - names)
    if missing:
        raise VerificationError(f"Bundle archive missing files: {missing}")
    pass_line(
        "text-bundle",
        f"files {len(bundle['files'])}, bytes {bundle['archive_bytes']}",
    )


def verify_text_package_unbundle(timeout: int) -> None:
    promote_dir = ARTIFACT_DIR / "text_promoted"
    archive = promote_dir / "text_package.tar.gz"
    bundle_json = promote_dir / "bundle_manifest.json"
    if not bundle_json.exists():
        verify_text_package_bundle(timeout)
    output_dir = ARTIFACT_DIR / "restored_text"
    report_json = ARTIFACT_DIR / "restored_text_report.json"
    out = run_cmd(
        [
            sys.executable,
            "unbundle_text_package.py",
            "--archive",
            str(archive),
            "--bundle-manifest",
            str(bundle_json),
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(report_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text package unbundler", out, "package unbundler marker")
    require(r"valid:\s+True", out, "package unbundler success")
    if not report_json.exists():
        raise VerificationError(f"Unbundle report was not created: {report_json}")
    report = json.loads(report_json.read_text(encoding="utf-8"))
    if not report["valid"] or len(report["files"]) != 7:
        raise VerificationError(f"Invalid unbundle report: {report}")
    restored_manifest = Path(report["package_dir"]) / "manifest.json"
    if not restored_manifest.exists():
        raise VerificationError(f"Restored manifest missing: {restored_manifest}")
    pass_line(
        "text-unbundle",
        f"files {len(report['files'])}, sha256 {report['archive_sha256'][:12]}",
    )


def verify_text_package_comparison(timeout: int) -> None:
    promote_dir = ARTIFACT_DIR / "text_promoted"
    restored_dir = ARTIFACT_DIR / "restored_text" / "text_promoted"
    if not (restored_dir / "manifest.json").exists():
        verify_text_package_unbundle(timeout)
    output_json = promote_dir / "package_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_packages.py",
            "--baseline-manifest",
            str(promote_dir / "manifest.json"),
            "--candidate-manifest",
            str(restored_dir / "manifest.json"),
            "--fail-on-loss-regression",
            "--fail-on-throughput-regression",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text package comparison", out, "package comparison marker")
    require(r"PACKAGE COMPARISON OK", out, "package comparison success")
    if not output_json.exists():
        raise VerificationError(f"Package comparison JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    if not comparison["same_checkpoint_sha256"]:
        raise VerificationError(f"Restored package checkpoint changed: {comparison}")
    if comparison["loss_delta"] != 0:
        raise VerificationError(f"Restored package loss changed: {comparison}")
    pass_line(
        "text-package-compare",
        f"loss_delta {comparison['loss_delta']:.4f}, same_sha {comparison['same_checkpoint_sha256']}",
    )


def verify_text_package_registry(timeout: int) -> None:
    promote_dir = ARTIFACT_DIR / "text_promoted"
    comparison_json = promote_dir / "package_comparison.json"
    if not comparison_json.exists():
        verify_text_package_comparison(timeout)
    registry_json = ARTIFACT_DIR / "text_release_registry.json"
    registry_json.unlink(missing_ok=True)
    out = run_cmd(
        [
            sys.executable,
            "register_text_package.py",
            "--manifest",
            str(promote_dir / "manifest.json"),
            "--bundle-manifest",
            str(promote_dir / "bundle_manifest.json"),
            "--archive",
            str(promote_dir / "text_package.tar.gz"),
            "--comparison-json",
            str(comparison_json),
            "--registry",
            str(registry_json),
            "--name",
            "verify_text_package",
        ],
        timeout,
    )
    require(r"TextPy/SoA text package release registry", out, "release registry marker")
    require(r"release_count:\s+1", out, "release registry count")
    if not registry_json.exists():
        raise VerificationError(f"Release registry JSON was not created: {registry_json}")
    registry = json.loads(registry_json.read_text(encoding="utf-8"))
    if registry["release_count"] != 1 or not registry.get("best_release"):
        raise VerificationError(f"Invalid release registry: {registry}")
    release = registry["best_release"]
    if release["checkpoint_sha256"] != registry["releases"][0]["checkpoint_sha256"]:
        raise VerificationError(f"Best release does not match first release: {registry}")
    if not release.get("comparison", {}).get("same_checkpoint_sha256"):
        raise VerificationError(f"Release registry did not capture comparison verdict: {release}")
    pass_line(
        "text-package-register",
        f"releases {registry['release_count']}, sha256 {release['checkpoint_sha256'][:12]}",
    )


def verify_text_release_leaderboard(timeout: int) -> None:
    registry_json = ARTIFACT_DIR / "text_release_registry.json"
    if not registry_json.exists():
        verify_text_package_registry(timeout)
    output_json = ARTIFACT_DIR / "text_release_leaderboard.json"
    output_csv = ARTIFACT_DIR / "text_release_leaderboard.csv"
    out = run_cmd(
        [
            sys.executable,
            "list_text_releases.py",
            "--registry",
            str(registry_json),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        timeout,
    )
    require(r"TextPy/SoA text release leaderboard", out, "release leaderboard marker")
    require(r"releases:\s+1", out, "release leaderboard count")
    for path in [output_json, output_csv]:
        if not path.exists():
            raise VerificationError(f"Release leaderboard artifact was not created: {path}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    rows = payload["releases"]
    if payload["release_count"] != 1 or len(rows) != 1:
        raise VerificationError(f"Invalid release leaderboard: {payload}")
    row = rows[0]
    if row["rank"] != 1 or not row["comparison_same_checkpoint_sha256"]:
        raise VerificationError(f"Invalid release leaderboard row: {row}")
    pass_line(
        "text-release-leaderboard",
        f"releases {payload['release_count']}, best_loss {row['benchmark_loss']:.4f}",
    )


def verify_text_release_selection(timeout: int) -> None:
    registry_json = ARTIFACT_DIR / "text_release_registry.json"
    if not registry_json.exists():
        verify_text_package_registry(timeout)
    output_json = ARTIFACT_DIR / "current_text_release.json"
    out = run_cmd(
        [
            sys.executable,
            "select_text_release.py",
            "--registry",
            str(registry_json),
            "--rank",
            "1",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA active text release selector", out, "release selector marker")
    require(r"selected_rank:\s+1", out, "release selector rank")
    if not output_json.exists():
        raise VerificationError(f"Current release JSON was not created: {output_json}")
    selected = json.loads(output_json.read_text(encoding="utf-8"))
    if selected["rank"] != 1 or selected["validation"]["valid"] is not True:
        raise VerificationError(f"Invalid selected release: {selected}")
    if not Path(selected["manifest"]).exists() or not Path(selected["archive"]).exists():
        raise VerificationError(f"Selected release paths do not exist: {selected}")
    pass_line(
        "text-release-select",
        f"rank {selected['rank']}, sha256 {selected['checkpoint_sha256'][:12]}",
    )


def verify_text_release_runner(timeout: int) -> None:
    release_json = ARTIFACT_DIR / "current_text_release.json"
    if not release_json.exists():
        verify_text_release_selection(timeout)
    output_json = ARTIFACT_DIR / "run_current_text_release.json"
    out = run_cmd(
        [
            sys.executable,
            "run_text_checkpoint.py",
            "--release",
            str(release_json),
            "--sample-steps",
            "6",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text checkpoint runner", out, "release runner marker")
    require(r"release:", out, "release runner release line")
    loss = parse_float(r"loss:\s+([0-9.]+)", out, "release runner loss")
    if loss <= 0:
        raise VerificationError(f"Unexpected active release runner loss:\n{out}")
    if not output_json.exists():
        raise VerificationError(f"Active release runner JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if payload["release"] != str(release_json):
        raise VerificationError(f"Release runner did not record release pointer: {payload}")
    if not payload.get("manifest") or not payload.get("checkpoint"):
        raise VerificationError(f"Release runner did not resolve package paths: {payload}")
    if not payload.get("sample"):
        raise VerificationError(f"Release runner did not generate sample: {payload}")
    pass_line(
        "text-release-run",
        f"loss {payload['loss']:.4f}, sample_len {len(payload['sample'])}",
    )


def verify_text_memory_probe(timeout: int) -> None:
    release_json = ARTIFACT_DIR / "current_text_release.json"
    if not release_json.exists():
        verify_text_release_selection(timeout)
    output_json = ARTIFACT_DIR / "text_memory_probe.json"
    output_csv = ARTIFACT_DIR / "text_memory_probe.csv"
    out = run_cmd(
        [
            sys.executable,
            "probe_text_memory.py",
            "--release",
            str(release_json),
            "--prompt",
            "memory is a river",
            "--top-k",
            "3",
            "--include-state-vectors",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        timeout,
    )
    require(r"TextPy/SoA text memory probe", out, "text memory probe marker")
    require(r"final_state_norm:", out, "text memory probe state norm")
    for path in [output_json, output_csv]:
        if not path.exists():
            raise VerificationError(f"Text memory probe artifact was not created: {path}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if payload["release"] != str(release_json):
        raise VerificationError(f"Memory probe did not record release pointer: {payload}")
    if summary["step_count"] != 4 or len(payload["steps"]) != 4:
        raise VerificationError(f"Memory probe step count mismatch: {payload}")
    if float(summary["final_state_norm"]) <= 0 or float(summary["max_state_delta_norm"]) <= 0:
        raise VerificationError(f"Memory probe state dynamics look invalid: {summary}")
    if summary.get("has_state_vectors") is not True:
        raise VerificationError(f"Memory probe did not export state vectors: {summary}")
    if len(summary.get("final_state_vector", [])) != int(payload["state_dim"]):
        raise VerificationError(f"Memory probe final vector shape mismatch: {summary}")
    if not summary["final_top_prediction"]:
        raise VerificationError(f"Memory probe missing final top prediction: {summary}")
    csv_rows = sum(1 for _ in output_csv.open("r", encoding="utf-8"))
    if csv_rows != 5:
        raise VerificationError(f"Memory probe CSV row count mismatch: {csv_rows}")
    pass_line(
        "text-memory-probe",
        f"steps {summary['step_count']}, final_norm {summary['final_state_norm']:.4f}, top {summary['final_top_prediction']}",
    )


def verify_text_memory_comparison(timeout: int) -> None:
    probe_json = ARTIFACT_DIR / "text_memory_probe.json"
    if not probe_json.exists():
        verify_text_memory_probe(timeout)
    output_json = ARTIFACT_DIR / "text_memory_comparison.json"
    output_csv = ARTIFACT_DIR / "text_memory_comparison.csv"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_memory.py",
            "--baseline",
            str(probe_json),
            "--candidate",
            str(probe_json),
            "--require-same-tokens",
            "--max-state-norm-delta",
            "0.0",
            "--max-state-delta-norm-delta",
            "0.0",
            "--max-top-probability-delta",
            "0.0",
            "--min-mean-top-overlap",
            "1.0",
            "--max-state-l2-distance",
            "1e-9",
            "--min-state-cosine-similarity",
            "0.999999",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        timeout,
    )
    require(r"TextPy/SoA text memory comparison", out, "text memory comparison marker")
    require(r"MEMORY COMPARISON OK", out, "text memory comparison success")
    for path in [output_json, output_csv]:
        if not path.exists():
            raise VerificationError(f"Text memory comparison artifact was not created: {path}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if summary["step_count"] != 4 or summary["token_match_rate"] != 1.0:
        raise VerificationError(f"Memory comparison shape mismatch: {summary}")
    if summary["max_abs_state_norm_delta"] != 0.0 or summary["mean_top_prediction_overlap"] != 1.0:
        raise VerificationError(f"Identical memory comparison should have no drift: {summary}")
    if summary.get("has_state_vectors") is not True:
        raise VerificationError(f"Memory comparison did not use state vectors: {summary}")
    if summary["max_state_l2_distance"] > 1.0e-9 or summary["min_state_cosine_similarity"] < 0.999999:
        raise VerificationError(f"Identical memory comparison vector geometry should match: {summary}")
    csv_rows = sum(1 for _ in output_csv.open("r", encoding="utf-8"))
    if csv_rows != 5:
        raise VerificationError(f"Memory comparison CSV row count mismatch: {csv_rows}")
    pass_line(
        "text-memory-compare",
        f"steps {summary['step_count']}, drift {summary['max_abs_state_norm_delta']:.4f}, overlap {summary['mean_top_prediction_overlap']:.4f}",
    )


def verify_text_memory_suite(timeout: int) -> None:
    release_json = ARTIFACT_DIR / "current_text_release.json"
    if not release_json.exists():
        verify_text_release_selection(timeout)
    output_dir = ARTIFACT_DIR / "text_memory_suite"
    output_json = output_dir / "memory_suite.json"
    prompts_csv = output_dir / "prompts.csv"
    pairs_csv = output_dir / "pairs.csv"
    out = run_cmd(
        [
            sys.executable,
            "run_text_memory_suite.py",
            "--release",
            str(release_json),
            "--prompt",
            "memory is a river",
            "--prompt",
            "memory is another world",
            "--prompt",
            "the future is guessed",
            "--top-k",
            "3",
            "--include-state-vectors",
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(output_json),
            "--prompts-csv",
            str(prompts_csv),
            "--pairs-csv",
            str(pairs_csv),
        ],
        timeout,
    )
    require(r"TextPy/SoA text memory prompt suite", out, "text memory suite marker")
    for path in [output_json, prompts_csv, pairs_csv]:
        if not path.exists():
            raise VerificationError(f"Text memory suite artifact was not created: {path}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if payload["prompt_count"] != 3 or payload["pair_count"] != 3:
        raise VerificationError(f"Memory suite count mismatch: {summary}")
    if summary["mean_pair_overlap"] < 0 or summary["max_pair_state_norm_delta"] <= 0:
        raise VerificationError(f"Memory suite summary is invalid: {summary}")
    if summary.get("has_state_vectors") is not True:
        raise VerificationError(f"Memory suite did not include vector geometry: {summary}")
    if summary["max_pair_state_l2_distance"] <= 0 or summary["min_pair_state_cosine_similarity"] > 1:
        raise VerificationError(f"Memory suite vector geometry is invalid: {summary}")
    prompt_rows = sum(1 for _ in prompts_csv.open("r", encoding="utf-8"))
    pair_rows = sum(1 for _ in pairs_csv.open("r", encoding="utf-8"))
    if prompt_rows != 4 or pair_rows != 4:
        raise VerificationError(f"Memory suite CSV row mismatch: prompts={prompt_rows}, pairs={pair_rows}")
    for row in payload["prompts"]:
        if not Path(row["json"]).exists() or not Path(row["csv"]).exists():
            raise VerificationError(f"Memory suite prompt artifact missing: {row}")
    pass_line(
        "text-memory-suite",
        f"prompts {payload['prompt_count']}, pairs {payload['pair_count']}, mean_overlap {summary['mean_pair_overlap']:.4f}",
    )


def verify_text_memory_suite_validator(timeout: int) -> None:
    output_dir = ARTIFACT_DIR / "text_memory_suite"
    output_json = output_dir / "memory_suite.json"
    if not output_json.exists():
        verify_text_memory_suite(timeout)
    validation_json = output_dir / "validation.json"
    out = run_cmd(
        [
            sys.executable,
            "validate_text_memory_suite.py",
            "--suite",
            str(output_json),
            "--prompts-csv",
            str(output_dir / "prompts.csv"),
            "--pairs-csv",
            str(output_dir / "pairs.csv"),
            "--expected-prompts",
            "3",
            "--require-state-vectors",
            "--require-prompt-artifacts",
            "--min-mean-pair-overlap",
            "0.0",
            "--min-pair-overlap",
            "0.0",
            "--output-json",
            str(validation_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text memory suite validator", out, "text memory suite validator marker")
    require(r"valid:\s+True", out, "text memory suite validator success")
    require(r"MEMORY SUITE VALIDATION OK", out, "text memory suite validator ok")
    if not validation_json.exists():
        raise VerificationError(f"Memory suite validation JSON was not created: {validation_json}")
    result = json.loads(validation_json.read_text(encoding="utf-8"))
    if not result["valid"] or result["prompt_count"] != 3 or result["pair_count"] != 3:
        raise VerificationError(f"Invalid memory suite validation result: {result}")
    if result["has_state_vectors"] is not True or result["prompt_artifacts_checked"] != 3:
        raise VerificationError(f"Memory suite validator did not check vector artifacts: {result}")
    pass_line(
        "text-memory-suite-validate",
        f"prompts {result['prompt_count']}, pairs {result['pair_count']}, mean_overlap {result['mean_pair_overlap']:.4f}",
    )


def verify_text_release_pipeline(timeout: int) -> None:
    sweep_leaderboard = ARTIFACT_DIR / "text_sweep" / "leaderboard.json"
    if not sweep_leaderboard.exists():
        verify_text_experiment_sweep(timeout)
    output_json = ARTIFACT_DIR / "text_release_pipeline.json"
    out = run_cmd(
        [
            sys.executable,
            "release_text_pipeline.py",
            "--leaderboard",
            str(sweep_leaderboard),
            "--promote-dir",
            str(ARTIFACT_DIR / "text_pipeline_promoted"),
            "--restore-dir",
            str(ARTIFACT_DIR / "text_pipeline_restored"),
            "--root-name",
            "text_pipeline_promoted",
            "--registry",
            str(ARTIFACT_DIR / "text_pipeline_registry.json"),
            "--release-leaderboard-json",
            str(ARTIFACT_DIR / "text_pipeline_leaderboard.json"),
            "--release-leaderboard-csv",
            str(ARTIFACT_DIR / "text_pipeline_leaderboard.csv"),
            "--current-release-json",
            str(ARTIFACT_DIR / "text_pipeline_current_release.json"),
            "--run-json",
            str(ARTIFACT_DIR / "text_pipeline_run_release.json"),
            "--dashboard-json",
            str(ARTIFACT_DIR / "text_pipeline_dashboard.json"),
            "--output-json",
            str(output_json),
            "--memory-suite-dir",
            str(ARTIFACT_DIR / "text_pipeline_memory_suite"),
            "--include-evidence-audit",
            "--name",
            "verify_text_pipeline",
            "--benchmark-repeat",
            "2",
            "--sample-steps",
            "6",
            "--clean",
            "--timeout",
            str(timeout),
        ],
        timeout,
    )
    require(r"TextPy/SoA text release pipeline", out, "release pipeline marker")
    require(r"RELEASE PIPELINE OK", out, "release pipeline success")
    if not output_json.exists():
        raise VerificationError(f"Release pipeline JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if summary["release_count"] != 1 or summary["selected_rank"] != 1:
        raise VerificationError(f"Release pipeline did not select a single rank-1 release: {summary}")
    if float(summary["loss"]) <= 0 or not summary.get("sample"):
        raise VerificationError(f"Release pipeline run result is invalid: {summary}")
    if int(summary["memory_prompt_count"]) != 3 or int(summary["memory_pair_count"]) != 3:
        raise VerificationError(f"Release pipeline memory suite count mismatch: {summary}")
    if float(summary["memory_mean_pair_overlap"]) < 0:
        raise VerificationError(f"Release pipeline memory overlap is invalid: {summary}")
    if not Path(summary["memory_suite"]).exists() or not Path(summary["memory_suite_validation"]).exists():
        raise VerificationError(f"Release pipeline memory artifacts missing: {summary}")
    if summary.get("release_dashboard_audit_valid") is not True:
        raise VerificationError(f"Release pipeline dashboard audit is invalid: {summary}")
    if int(summary.get("release_dashboard_audit_checks") or 0) < 1:
        raise VerificationError(f"Release pipeline dashboard audit did not run checks: {summary}")
    if not Path(summary.get("release_dashboard", "")).exists():
        raise VerificationError(f"Release pipeline dashboard artifact missing: {summary}")
    if summary.get("release_evidence_audit_valid") is not True:
        raise VerificationError(f"Release pipeline evidence audit is invalid: {summary}")
    if summary.get("release_evidence_audit_validation_valid") is not True:
        raise VerificationError(f"Release pipeline evidence audit validation is invalid: {summary}")
    if int(summary.get("release_evidence_audit_checks") or 0) < 10:
        raise VerificationError(f"Release pipeline evidence audit did not run checks: {summary}")
    for key in [
        "release_gates",
        "release_report",
        "release_evidence_archive",
        "release_evidence_manifest",
        "release_evidence_ledger",
        "release_evidence_audit",
        "release_evidence_audit_validation",
        "release_evidence_audit_validation_comparison",
        "release_evidence_audit_comparison",
    ]:
        if not Path(summary.get(key, "")).exists():
            raise VerificationError(f"Release pipeline evidence artifact missing for {key}: {summary}")
    if not summary.get("release_evidence_archive_sha256"):
        raise VerificationError(f"Release pipeline evidence archive sha missing: {summary}")
    if not summary.get("release_evidence_ledger_chain_head"):
        raise VerificationError(f"Release pipeline evidence ledger head missing: {summary}")
    audit_validation_comparison = json.loads(
        Path(summary["release_evidence_audit_validation_comparison"]).read_text(encoding="utf-8")
    )
    if audit_validation_comparison.get("failures"):
        raise VerificationError(
            f"Release pipeline audit validation comparison has failures: {audit_validation_comparison}"
        )
    audit_comparison = json.loads(Path(summary["release_evidence_audit_comparison"]).read_text(encoding="utf-8"))
    if audit_comparison.get("failures"):
        raise VerificationError(f"Release pipeline audit comparison has failures: {audit_comparison}")
    registry = json.loads(Path(summary["registry"]).read_text(encoding="utf-8"))
    release = registry["releases"][0]
    memory = release.get("memory") or {}
    if memory.get("validation") != summary["memory_suite_validation"]:
        raise VerificationError(f"Release registry did not capture memory validation path: {release}")
    if memory.get("mean_pair_overlap") != summary["memory_mean_pair_overlap"]:
        raise VerificationError(f"Release registry memory overlap mismatch: {release}")
    leaderboard = json.loads((ARTIFACT_DIR / "text_pipeline_leaderboard.json").read_text(encoding="utf-8"))
    leaderboard_row = leaderboard["releases"][0]
    if leaderboard_row.get("memory_mean_pair_overlap") != summary["memory_mean_pair_overlap"]:
        raise VerificationError(f"Release leaderboard memory overlap mismatch: {leaderboard_row}")
    current = json.loads((ARTIFACT_DIR / "text_pipeline_current_release.json").read_text(encoding="utf-8"))
    current_memory = current.get("memory") or {}
    if current_memory.get("validation") != summary["memory_suite_validation"]:
        raise VerificationError(f"Active release pointer did not carry memory validation: {current}")
    for name in [
        "promote",
        "bundle",
        "register",
        "select",
        "run_active_release",
        "memory_suite",
        "validate_memory_suite",
        "release_dashboard",
        "release_dashboard_validation",
        "release_dashboard_comparison",
        "release_history",
        "release_history_validation",
        "release_history_analysis",
        "release_gates",
        "release_report",
        "release_evidence_bundle",
        "release_evidence_validate",
        "release_evidence_compare",
        "release_evidence_unbundle",
        "release_evidence_register",
        "release_evidence_ledger_validate",
        "release_evidence_ledger_tamper",
        "release_evidence_ledger_replay",
        "release_evidence_ledger_compare",
        "release_evidence_ledger_growth_compare",
        "release_evidence_audit",
        "release_evidence_audit_validate",
        "release_evidence_audit_validation_compare",
        "release_evidence_audit_compare",
    ]:
        if name not in payload["stages"]:
            raise VerificationError(f"Release pipeline missing stage report {name}: {payload['stages'].keys()}")
    pass_line(
        "text-release-pipeline",
        f"rank {summary['selected_rank']}, loss {summary['loss']:.4f}, memory_overlap {summary['memory_mean_pair_overlap']:.4f}, evidence {summary['release_evidence_archive_sha256'][:12]}",
    )


def verify_text_release_pipeline_validation(timeout: int) -> None:
    pipeline_json = ARTIFACT_DIR / "text_release_pipeline.json"
    if not pipeline_json.exists():
        verify_text_release_pipeline(timeout)
    output_json = ARTIFACT_DIR / "text_release_pipeline_validation.json"
    out = run_cmd(
        [
            sys.executable,
            "validate_text_release_pipeline.py",
            "--pipeline",
            str(pipeline_json),
            "--require-artifacts",
            "--require-evidence-audit",
            "--expected-stages",
            "35",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release pipeline validator", out, "release pipeline validator marker")
    require(r"RELEASE PIPELINE VALIDATION OK", out, "release pipeline validation success")
    if not output_json.exists():
        raise VerificationError(f"Release pipeline validation JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("valid") is not True:
        raise VerificationError(f"Release pipeline validation failed: {result}")
    if result.get("has_evidence_audit") is not True or int(result.get("evidence_audit_checks") or 0) < 10:
        raise VerificationError(f"Release pipeline validation did not require evidence audit: {result}")
    if int(result.get("stage_count") or 0) != 35:
        raise VerificationError(f"Release pipeline validation stage count mismatch: {result}")
    if int(result.get("artifact_checks") or 0) < 16:
        raise VerificationError(f"Release pipeline validation artifact coverage is too low: {result}")
    pass_line(
        "text-release-pipeline-validate",
        f"stages {result['stage_count']}, artifacts {result['artifact_checks']}, evidence_checks {result['evidence_audit_checks']}",
    )


def verify_text_release_pipeline_validation_comparison(timeout: int) -> None:
    validation_json = ARTIFACT_DIR / "text_release_pipeline_validation.json"
    if not validation_json.exists():
        verify_text_release_pipeline_validation(timeout)
    output_json = ARTIFACT_DIR / "text_release_pipeline_validation_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_release_pipeline_validations.py",
            "--baseline-validation",
            str(validation_json),
            "--candidate-validation",
            str(validation_json),
            "--output-json",
            str(output_json),
            "--fail-on-validity-change",
            "--fail-on-pipeline-path-change",
            "--fail-on-name-change",
            "--fail-on-evidence-presence-change",
            "--fail-on-stage-count-change",
            "--fail-on-memory-prompt-count-change",
            "--fail-on-memory-pair-count-change",
            "--fail-on-dashboard-audit-check-regression",
            "--fail-on-evidence-audit-check-regression",
            "--fail-on-artifact-check-regression",
            "--fail-on-error-regression",
            "--fail-on-error-set-change",
        ],
        timeout,
    )
    require(
        r"TextPy/SoA release pipeline validation comparison",
        out,
        "release pipeline validation comparison marker",
    )
    require(r"RELEASE PIPELINE VALIDATION COMPARISON OK", out, "release pipeline validation comparison success")
    if not output_json.exists():
        raise VerificationError(f"Release pipeline validation comparison JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("failures"):
        raise VerificationError(f"Release pipeline validation comparison failed: {result}")
    comparison = result.get("comparison") or {}
    if comparison.get("same_valid") is not True or comparison.get("stage_count_delta") != 0:
        raise VerificationError(f"Release pipeline validation self-comparison drifted: {result}")
    if comparison.get("artifact_checks_delta") != 0 or comparison.get("error_count_delta") != 0:
        raise VerificationError(f"Release pipeline validation self-comparison changed coverage/errors: {result}")
    pass_line(
        "text-release-pipeline-validation-compare",
        f"same_valid {comparison['same_valid']}, artifacts_delta {comparison['artifact_checks_delta']}, errors {comparison['error_count_delta']}",
    )


def verify_text_release_pipeline_comparison(timeout: int) -> None:
    pipeline_json = ARTIFACT_DIR / "text_release_pipeline.json"
    if not pipeline_json.exists():
        verify_text_release_pipeline(timeout)
    output_json = ARTIFACT_DIR / "text_release_pipeline_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_release_pipelines.py",
            "--baseline-pipeline",
            str(pipeline_json),
            "--candidate-pipeline",
            str(pipeline_json),
            "--output-json",
            str(output_json),
            "--require-evidence-audit",
            "--fail-on-stage-set-change",
            "--fail-on-stage-count-change",
            "--fail-on-name-change",
            "--fail-on-release-count-change",
            "--fail-on-rank-change",
            "--fail-on-checkpoint-sha-change",
            "--fail-on-archive-sha-change",
            "--fail-on-evidence-presence-change",
            "--fail-on-evidence-archive-sha-change",
            "--fail-on-ledger-chain-head-change",
            "--fail-on-dashboard-audit-regression",
            "--fail-on-evidence-audit-regression",
            "--fail-on-loss-regression",
            "--fail-on-throughput-regression",
            "--fail-on-accuracy-regression",
            "--fail-on-memory-prompt-count-change",
            "--fail-on-memory-pair-count-change",
            "--fail-on-memory-overlap-regression",
            "--fail-on-memory-min-overlap-regression",
            "--fail-on-memory-cosine-regression",
            "--fail-on-memory-l2-regression",
            "--fail-on-memory-norm-regression",
            "--max-loss-regression",
            "0.0",
            "--max-throughput-regression",
            "0.0",
            "--max-accuracy-regression",
            "0.0",
        ],
        timeout,
    )
    require(r"TextPy/SoA release pipeline comparison", out, "release pipeline comparison marker")
    require(r"RELEASE PIPELINE COMPARISON OK", out, "release pipeline comparison success")
    if not output_json.exists():
        raise VerificationError(f"Release pipeline comparison JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("failures"):
        raise VerificationError(f"Release pipeline comparison failed: {result}")
    comparison = result.get("comparison") or {}
    if comparison.get("same_stage_set") is not True or comparison.get("stage_count_delta") != 0:
        raise VerificationError(f"Release pipeline self-comparison stage drifted: {result}")
    if comparison.get("loss_delta") != 0 or comparison.get("memory_mean_pair_overlap_delta") != 0:
        raise VerificationError(f"Release pipeline self-comparison metric drifted: {result}")
    if comparison.get("same_evidence_archive_sha256") is not True:
        raise VerificationError(f"Release pipeline self-comparison evidence archive changed: {result}")
    pass_line(
        "text-release-pipeline-compare",
        f"stage_delta {comparison['stage_count_delta']}, loss_delta {comparison['loss_delta']:.4f}, memory_overlap_delta {comparison['memory_mean_pair_overlap_delta']:.4f}",
    )


def verify_text_release_pipeline_comparison_validation(timeout: int) -> None:
    comparison_json = ARTIFACT_DIR / "text_release_pipeline_comparison.json"
    if not comparison_json.exists():
        verify_text_release_pipeline_comparison(timeout)
    output_json = ARTIFACT_DIR / "text_release_pipeline_comparison_validation.json"
    out = run_cmd(
        [
            sys.executable,
            "validate_text_release_pipeline_comparison.py",
            "--comparison",
            str(comparison_json),
            "--require-evidence-audit",
            "--expected-stages",
            "35",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(
        r"TextPy/SoA release pipeline comparison validator",
        out,
        "release pipeline comparison validator marker",
    )
    require(r"RELEASE PIPELINE COMPARISON VALIDATION OK", out, "release pipeline comparison validation success")
    if not output_json.exists():
        raise VerificationError(f"Release pipeline comparison validation JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("valid") is not True:
        raise VerificationError(f"Release pipeline comparison validation failed: {result}")
    if result.get("same_stage_set") is not True or result.get("stage_count_delta") != 0:
        raise VerificationError(f"Release pipeline comparison validation missed stage drift: {result}")
    if result.get("failure_count") != 0:
        raise VerificationError(f"Release pipeline comparison validation should have no failures: {result}")
    if result.get("same_evidence_archive_sha256") is not True:
        raise VerificationError(f"Release pipeline comparison validation missed evidence SHA drift: {result}")
    pass_line(
        "text-release-pipeline-compare-validate",
        f"stages {result['baseline_stage_count']}->{result['candidate_stage_count']}, failures {result['failure_count']}",
    )


def verify_text_release_pipeline_comparison_validation_comparison(timeout: int) -> None:
    validation_json = ARTIFACT_DIR / "text_release_pipeline_comparison_validation.json"
    if not validation_json.exists():
        verify_text_release_pipeline_comparison_validation(timeout)
    output_json = ARTIFACT_DIR / "text_release_pipeline_comparison_validation_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_release_pipeline_comparison_validations.py",
            "--baseline-validation",
            str(validation_json),
            "--candidate-validation",
            str(validation_json),
            "--output-json",
            str(output_json),
            "--fail-on-validity-change",
            "--fail-on-comparison-path-change",
            "--fail-on-pipeline-path-change",
            "--fail-on-stage-count-change",
            "--fail-on-stage-set-change",
            "--fail-on-loss-delta-change",
            "--fail-on-memory-overlap-delta-change",
            "--fail-on-proof-flag-change",
            "--fail-on-failure-regression",
            "--fail-on-error-regression",
            "--fail-on-error-set-change",
        ],
        timeout,
    )
    require(
        r"TextPy/SoA release pipeline comparison validation comparison",
        out,
        "release pipeline comparison validation comparison marker",
    )
    require(
        r"RELEASE PIPELINE COMPARISON VALIDATION COMPARISON OK",
        out,
        "release pipeline comparison validation comparison success",
    )
    if not output_json.exists():
        raise VerificationError(
            f"Release pipeline comparison validation comparison JSON was not created: {output_json}"
        )
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("failures"):
        raise VerificationError(f"Release pipeline comparison validation comparison failed: {result}")
    comparison = result.get("comparison") or {}
    if comparison.get("same_valid") is not True or comparison.get("stage_count_delta_delta") != 0:
        raise VerificationError(f"Release pipeline comparison validation self-comparison drifted: {result}")
    if comparison.get("failure_count_delta") != 0 or comparison.get("error_count_delta") != 0:
        raise VerificationError(f"Release pipeline comparison validation errors/failures drifted: {result}")
    pass_line(
        "text-release-pipeline-comparison-validation-compare",
        f"same_valid {comparison['same_valid']}, failures {comparison['failure_count_delta']}, errors {comparison['error_count_delta']}",
    )


def verify_text_release_comparison(timeout: int) -> None:
    release_json = ARTIFACT_DIR / "text_pipeline_current_release.json"
    if not release_json.exists():
        verify_text_release_pipeline(timeout)
    output_json = ARTIFACT_DIR / "text_release_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_releases.py",
            "--baseline-release",
            str(release_json),
            "--candidate-release",
            str(release_json),
            "--require-memory",
            "--fail-on-loss-regression",
            "--fail-on-throughput-regression",
            "--fail-on-accuracy-regression",
            "--fail-on-memory-overlap-regression",
            "--fail-on-memory-cosine-regression",
            "--fail-on-memory-l2-regression",
            "--fail-on-memory-norm-regression",
            "--max-loss-regression",
            "0.0",
            "--max-throughput-regression",
            "0.0",
            "--max-accuracy-regression",
            "0.0",
            "--max-memory-overlap-regression",
            "0.0",
            "--max-memory-cosine-regression",
            "0.0",
            "--max-memory-l2-regression",
            "0.0",
            "--max-memory-norm-regression",
            "0.0",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA text release comparison", out, "release comparison marker")
    require(r"RELEASE COMPARISON OK", out, "release comparison success")
    if not output_json.exists():
        raise VerificationError(f"Release comparison JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    if comparison["loss_delta"] != 0.0 or comparison["memory_mean_pair_overlap_delta"] != 0.0:
        raise VerificationError(f"Identical release comparison should have no deltas: {comparison}")
    if comparison["has_memory_diagnostics"] is not True:
        raise VerificationError(f"Release comparison did not use memory diagnostics: {comparison}")
    pass_line(
        "text-release-compare",
        f"loss_delta {comparison['loss_delta']:.4f}, memory_overlap_delta {comparison['memory_mean_pair_overlap_delta']:.4f}",
    )


def verify_text_release_dashboard_json(timeout: int) -> None:
    pipeline_json = ARTIFACT_DIR / "text_release_pipeline.json"
    if not pipeline_json.exists():
        verify_text_release_pipeline(timeout)
    pipeline = json.loads(pipeline_json.read_text(encoding="utf-8"))
    pipeline_summary = pipeline.get("summary") or {}
    pipeline_dashboard = Path(pipeline_summary.get("release_dashboard", ""))
    if not pipeline_dashboard.exists():
        raise VerificationError(f"Release pipeline did not create dashboard JSON: {pipeline_summary}")
    if pipeline_summary.get("release_dashboard_audit_valid") is not True:
        raise VerificationError(f"Release pipeline dashboard audit was not valid: {pipeline_summary}")
    output_json = ARTIFACT_DIR / "text_release_dashboard.json"
    out = run_cmd(
        [
            sys.executable,
            "build_text_release_dashboard.py",
            "--leaderboard",
            str(ARTIFACT_DIR / "text_pipeline_leaderboard.json"),
            "--current-release",
            str(ARTIFACT_DIR / "text_pipeline_current_release.json"),
            "--memory-suite",
            str(ARTIFACT_DIR / "text_pipeline_memory_suite" / "memory_suite.json"),
            "--memory-validation",
            str(ARTIFACT_DIR / "text_pipeline_memory_suite" / "validation.json"),
            "--run-json",
            str(ARTIFACT_DIR / "text_pipeline_run_release.json"),
            "--pipeline-json",
            str(pipeline_json),
            "--output-json",
            str(output_json),
            "--require-valid",
        ],
        timeout,
    )
    require(r"TextPy/SoA release dashboard builder", out, "release dashboard builder marker")
    require(r"audit_valid:\s+True", out, "release dashboard audit success")
    require(r"RELEASE DASHBOARD OK", out, "release dashboard ok")
    if not output_json.exists():
        raise VerificationError(f"Release dashboard JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if payload.get("kind") != "text_release_dashboard":
        raise VerificationError(f"Unexpected release dashboard kind: {payload.get('kind')}")
    audit = payload.get("audit") or {}
    dashboard = payload.get("dashboard") or {}
    if audit.get("valid") is not True or int(audit.get("failed_count", 1)) != 0:
        raise VerificationError(f"Release dashboard audit failed: {audit}")
    for key in ["leaderboard", "current", "suite", "memory_validation", "run", "pipeline"]:
        if payload.get(key) is None:
            raise VerificationError(f"Release dashboard missing embedded {key} payload.")
    memory = dashboard.get("memory") or {}
    if float(dashboard.get("benchmark_loss") or 0.0) <= 0.0:
        raise VerificationError(f"Release dashboard loss invalid: {dashboard}")
    if float(dashboard.get("benchmark_best_throughput_tokens_s") or 0.0) <= 0.0:
        raise VerificationError(f"Release dashboard throughput invalid: {dashboard}")
    if int(memory.get("prompt_count") or 0) != 3 or int(memory.get("pair_count") or 0) != 3:
        raise VerificationError(f"Release dashboard memory counts invalid: {memory}")
    pass_line(
        "text-release-dashboard-json",
        f"rank {dashboard['rank']}, loss {dashboard['benchmark_loss']:.4f}, checks {audit['check_count']}",
    )


def verify_text_release_dashboard_validator(timeout: int) -> None:
    dashboard_json = ARTIFACT_DIR / "text_release_dashboard.json"
    if not dashboard_json.exists():
        verify_text_release_dashboard_json(timeout)
    output_json = ARTIFACT_DIR / "text_release_dashboard_validation.json"
    out = run_cmd(
        [
            sys.executable,
            "validate_text_release_dashboard.py",
            "--dashboard",
            str(dashboard_json),
            "--require-artifacts",
            "--expected-prompts",
            "3",
            "--expected-pairs",
            "3",
            "--min-mean-pair-overlap",
            "0.0",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release dashboard validator", out, "release dashboard validator marker")
    require(r"valid:\s+True", out, "release dashboard validator success")
    require(r"RELEASE DASHBOARD VALIDATION OK", out, "release dashboard validator ok")
    if not output_json.exists():
        raise VerificationError(f"Release dashboard validation JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result["valid"] is not True or int(result["audit_check_count"]) != 11:
        raise VerificationError(f"Release dashboard validation failed: {result}")
    if int(result["prompt_count"]) != 3 or int(result["pair_count"]) != 3:
        raise VerificationError(f"Release dashboard validation memory counts mismatch: {result}")
    if int(result["artifact_checks"]) < 5:
        raise VerificationError(f"Release dashboard validator did not check referenced artifacts: {result}")
    pass_line(
        "text-release-dashboard-validate",
        f"checks {result['audit_check_count']}, artifacts {result['artifact_checks']}, mean_overlap {result['mean_pair_overlap']:.4f}",
    )


def verify_text_release_dashboard_comparison(timeout: int) -> None:
    dashboard_json = ARTIFACT_DIR / "text_release_dashboard.json"
    if not dashboard_json.exists():
        verify_text_release_dashboard_json(timeout)
    output_json = ARTIFACT_DIR / "text_release_dashboard_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_release_dashboards.py",
            "--baseline-dashboard",
            str(dashboard_json),
            "--candidate-dashboard",
            str(dashboard_json),
            "--require-artifacts",
            "--expected-prompts",
            "3",
            "--expected-pairs",
            "3",
            "--min-mean-pair-overlap",
            "0.0",
            "--fail-on-loss-regression",
            "--fail-on-throughput-regression",
            "--fail-on-accuracy-regression",
            "--fail-on-audit-regression",
            "--fail-on-memory-overlap-regression",
            "--fail-on-memory-cosine-regression",
            "--fail-on-memory-l2-regression",
            "--fail-on-memory-norm-regression",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release dashboard comparison", out, "release dashboard comparison marker")
    require(r"RELEASE DASHBOARD COMPARISON OK", out, "release dashboard comparison ok")
    if not output_json.exists():
        raise VerificationError(f"Release dashboard comparison JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    if comparison["loss_delta"] != 0.0 or comparison["memory_mean_pair_overlap_delta"] != 0.0:
        raise VerificationError(f"Identical dashboard comparison should have no deltas: {comparison}")
    if comparison["audit_failed_count_delta"] != 0:
        raise VerificationError(f"Identical dashboard comparison should not change audit failures: {comparison}")
    pass_line(
        "text-release-dashboard-compare",
        f"loss_delta {comparison['loss_delta']:.4f}, memory_overlap_delta {comparison['memory_mean_pair_overlap_delta']:.4f}",
    )


def verify_text_release_history_json(timeout: int) -> None:
    dashboard_json = ARTIFACT_DIR / "text_release_dashboard.json"
    if not dashboard_json.exists():
        verify_text_release_dashboard_json(timeout)
    output_json = ARTIFACT_DIR / "text_release_history.json"
    out = run_cmd(
        [
            sys.executable,
            "build_text_release_history.py",
            "--dashboard",
            str(dashboard_json),
            "--require-artifacts",
            "--expected-prompts",
            "3",
            "--expected-pairs",
            "3",
            "--min-mean-pair-overlap",
            "0.0",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release history builder", out, "release history builder marker")
    require(r"RELEASE HISTORY OK", out, "release history ok")
    if not output_json.exists():
        raise VerificationError(f"Release history JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if payload.get("kind") != "text_release_history":
        raise VerificationError(f"Unexpected release history kind: {payload.get('kind')}")
    dashboards = payload.get("dashboards")
    summary = payload.get("summary") or {}
    if not isinstance(dashboards, list) or len(dashboards) != 1:
        raise VerificationError(f"Release history should contain one dashboard in smoke test: {payload}")
    first = dashboards[0]
    if first.get("history_index") != 1:
        raise VerificationError(f"Release history index did not start at 1: {first}")
    if first.get("delta", {}).get("benchmark_loss") != 0.0:
        raise VerificationError(f"First release history delta should be zero: {first.get('delta')}")
    if int(summary.get("release_count") or 0) != 1 or int(summary.get("valid_count") or 0) != 1:
        raise VerificationError(f"Release history summary counts invalid: {summary}")
    pass_line(
        "text-release-history-json",
        f"releases {summary['release_count']}, best_loss {summary['best_loss']:.4f}",
    )


def verify_text_release_history_validator(timeout: int) -> None:
    history_json = ARTIFACT_DIR / "text_release_history.json"
    if not history_json.exists():
        verify_text_release_history_json(timeout)
    output_json = ARTIFACT_DIR / "text_release_history_validation.json"
    out = run_cmd(
        [
            sys.executable,
            "validate_text_release_history.py",
            "--history",
            str(history_json),
            "--require-artifacts",
            "--expected-releases",
            "1",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release history validator", out, "release history validator marker")
    require(r"valid:\s+True", out, "release history validator success")
    require(r"RELEASE HISTORY VALIDATION OK", out, "release history validation ok")
    if not output_json.exists():
        raise VerificationError(f"Release history validation JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("valid") is not True:
        raise VerificationError(f"Release history validation failed: {result}")
    if int(result.get("release_count") or 0) != 1:
        raise VerificationError(f"Release history validation release count mismatch: {result}")
    if int(result.get("artifact_count") or 0) < 1:
        raise VerificationError(f"Release history validator did not check referenced artifacts: {result}")
    pass_line(
        "text-release-history-validate",
        f"releases {result['release_count']}, artifacts {result['artifact_count']}",
    )


def verify_text_release_history_analysis(timeout: int) -> None:
    history_json = ARTIFACT_DIR / "text_release_history.json"
    if not history_json.exists():
        verify_text_release_history_json(timeout)
    output_json = ARTIFACT_DIR / "text_release_history_analysis.json"
    out = run_cmd(
        [
            sys.executable,
            "analyze_text_release_history.py",
            "--history",
            str(history_json),
            "--baseline",
            "previous",
            "--require-artifacts",
            "--expected-releases",
            "1",
            "--fail-on-loss-regression",
            "--fail-on-throughput-regression",
            "--fail-on-accuracy-regression",
            "--fail-on-audit-regression",
            "--fail-on-memory-overlap-regression",
            "--fail-on-memory-cosine-regression",
            "--fail-on-memory-l2-regression",
            "--fail-on-memory-norm-regression",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release history analysis", out, "release history analysis marker")
    require(r"RELEASE HISTORY ANALYSIS OK", out, "release history analysis ok")
    if not output_json.exists():
        raise VerificationError(f"Release history analysis JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    comparison = payload.get("comparison") or {}
    if payload.get("valid") is not True or payload.get("failures"):
        raise VerificationError(f"Release history analysis should be valid in smoke test: {payload}")
    if comparison.get("loss_delta") != 0.0 or comparison.get("memory_mean_pair_overlap_delta") != 0.0:
        raise VerificationError(f"Single-release history analysis should have zero deltas: {comparison}")
    pass_line(
        "text-release-history-analyze",
        f"baseline {payload['baseline_mode']}, loss_delta {comparison['loss_delta']:.4f}",
    )


def verify_text_release_gate_summary(timeout: int) -> None:
    dashboard_validation_json = ARTIFACT_DIR / "text_release_dashboard_validation.json"
    dashboard_comparison_json = ARTIFACT_DIR / "text_release_dashboard_comparison.json"
    history_validation_json = ARTIFACT_DIR / "text_release_history_validation.json"
    history_analysis_json = ARTIFACT_DIR / "text_release_history_analysis.json"
    if not dashboard_validation_json.exists():
        verify_text_release_dashboard_validator(timeout)
    if not dashboard_comparison_json.exists():
        verify_text_release_dashboard_comparison(timeout)
    if not history_validation_json.exists():
        verify_text_release_history_validator(timeout)
    if not history_analysis_json.exists():
        verify_text_release_history_analysis(timeout)
    output_json = ARTIFACT_DIR / "text_release_gates.json"
    out = run_cmd(
        [
            sys.executable,
            "summarize_text_release_gates.py",
            "--dashboard-validation",
            str(dashboard_validation_json),
            "--dashboard-comparison",
            str(dashboard_comparison_json),
            "--history-validation",
            str(history_validation_json),
            "--history-analysis",
            str(history_analysis_json),
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release gate summary", out, "release gate summary marker")
    require(r"RELEASE GATES OK", out, "release gates ok")
    if not output_json.exists():
        raise VerificationError(f"Release gate summary JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    gates = payload.get("gates")
    if payload.get("kind") != "text_release_gate_summary":
        raise VerificationError(f"Unexpected release gate summary kind: {payload.get('kind')}")
    if payload.get("valid") is not True:
        raise VerificationError(f"Release gate summary should be valid in smoke test: {payload}")
    if not isinstance(gates, list) or len(gates) != 4:
        raise VerificationError(f"Release gate summary should contain four gates: {payload}")
    if int(summary.get("failed_gate_count") or 0) != 0:
        raise VerificationError(f"Release gate summary reported failures: {payload}")
    pass_line(
        "text-release-gates",
        f"gates {len(gates)}, failed {summary['failed_gate_count']}",
    )


def verify_text_release_report(timeout: int) -> None:
    gates_json = ARTIFACT_DIR / "text_release_gates.json"
    dashboard_json = ARTIFACT_DIR / "text_release_dashboard.json"
    history_analysis_json = ARTIFACT_DIR / "text_release_history_analysis.json"
    if not gates_json.exists():
        verify_text_release_gate_summary(timeout)
    if not dashboard_json.exists():
        verify_text_release_dashboard_json(timeout)
    if not history_analysis_json.exists():
        verify_text_release_history_analysis(timeout)
    output_md = ARTIFACT_DIR / "text_release_report.md"
    out = run_cmd(
        [
            sys.executable,
            "generate_text_release_report.py",
            "--gates",
            str(gates_json),
            "--dashboard",
            str(dashboard_json),
            "--history-analysis",
            str(history_analysis_json),
            "--output-md",
            str(output_md),
        ],
        timeout,
    )
    require(r"TextPy/SoA release report", out, "release report marker")
    require(r"RELEASE REPORT OK", out, "release report ok")
    if not output_md.exists():
        raise VerificationError(f"Release report Markdown was not created: {output_md}")
    markdown = output_md.read_text(encoding="utf-8")
    for marker in [
        "# TextPy/SoA Release Report",
        "## Verdict",
        "## Gates",
        "## Dashboard Audit",
        "make summarize-release-gates-demo",
    ]:
        if marker not in markdown:
            raise VerificationError(f"Release report Markdown missing {marker!r}: {markdown[:500]}")
    if "| verdict | PASS |" not in markdown or "| dashboard_validation | PASS |" not in markdown:
        raise VerificationError(f"Release report did not record passing gates: {markdown[:1200]}")
    pass_line("text-release-report", f"markdown {output_md.name}")


def verify_text_release_evidence_bundle(timeout: int) -> None:
    gates_json = ARTIFACT_DIR / "text_release_gates.json"
    report_md = ARTIFACT_DIR / "text_release_report.md"
    if not gates_json.exists():
        verify_text_release_gate_summary(timeout)
    if not report_md.exists():
        verify_text_release_report(timeout)
    output_archive = ARTIFACT_DIR / "text_release_evidence.tar.gz"
    output_json = ARTIFACT_DIR / "text_release_evidence_manifest.json"
    out = run_cmd(
        [
            sys.executable,
            "bundle_text_release_evidence.py",
            "--base-dir",
            str(ARTIFACT_DIR),
            "--root-name",
            "verify_text_release_evidence",
            "--gates",
            str(gates_json),
            "--output",
            str(output_archive),
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence bundler", out, "release evidence bundler marker")
    require(r"RELEASE EVIDENCE BUNDLE OK", out, "release evidence bundle ok")
    for path in [output_archive, output_json]:
        if not path.exists():
            raise VerificationError(f"Release evidence bundle artifact was not created: {path}")
    manifest = json.loads(output_json.read_text(encoding="utf-8"))
    if manifest.get("kind") != "text_release_evidence_bundle":
        raise VerificationError(f"Unexpected release evidence manifest kind: {manifest.get('kind')}")
    if manifest.get("valid") is not True:
        raise VerificationError(f"Release evidence bundle should be valid in smoke test: {manifest}")
    if int(manifest.get("file_count") or 0) != 16:
        raise VerificationError(f"Release evidence bundle should contain 16 source files: {manifest}")
    if not manifest.get("archive_sha256") or int(manifest.get("archive_bytes") or 0) <= 0:
        raise VerificationError(f"Release evidence manifest missing archive checksum/size: {manifest}")
    with tarfile.open(output_archive, "r:gz") as tar:
        names = tar.getnames()
    expected = {
        "verify_text_release_evidence/text_release_gates.json",
        "verify_text_release_evidence/text_release_report.md",
        "verify_text_release_evidence/text_release_dashboard.json",
        "verify_text_release_evidence/text_memory_suite/memory_suite.json",
        "verify_text_release_evidence/evidence_manifest.json",
    }
    missing = sorted(expected - set(names))
    if missing:
        raise VerificationError(f"Release evidence archive missing files: {missing}")
    if len(names) != int(manifest["file_count"]) + 1:
        raise VerificationError(f"Release evidence archive member count mismatch: {len(names)} vs {manifest}")
    pass_line(
        "text-release-evidence",
        f"files {manifest['file_count']}, bytes {manifest['archive_bytes']}",
    )


def verify_text_release_evidence_validator(timeout: int) -> None:
    bundle_manifest = ARTIFACT_DIR / "text_release_evidence_manifest.json"
    if not bundle_manifest.exists():
        verify_text_release_evidence_bundle(timeout)
    output_json = ARTIFACT_DIR / "text_release_evidence_validation.json"
    out = run_cmd(
        [
            sys.executable,
            "validate_text_release_evidence.py",
            "--manifest",
            str(bundle_manifest),
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence validator", out, "release evidence validator marker")
    require(r"RELEASE EVIDENCE VALIDATION OK", out, "release evidence validation ok")
    if not output_json.exists():
        raise VerificationError(f"Release evidence validation JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("valid") is not True:
        raise VerificationError(f"Release evidence validation failed: {result}")
    if int(result.get("file_count") or 0) != 16 or int(result.get("checked_files") or 0) != 16:
        raise VerificationError(f"Release evidence validator did not check all source files: {result}")
    if int(result.get("member_count") or 0) != 17 or result.get("has_inner_manifest") is not True:
        raise VerificationError(f"Release evidence validator did not verify archive structure: {result}")
    pass_line(
        "text-release-evidence-validate",
        f"files {result['checked_files']}, members {result['member_count']}",
    )


def verify_text_release_evidence_unbundle(timeout: int) -> None:
    bundle_manifest = ARTIFACT_DIR / "text_release_evidence_manifest.json"
    output_archive = ARTIFACT_DIR / "text_release_evidence.tar.gz"
    if not bundle_manifest.exists() or not output_archive.exists():
        verify_text_release_evidence_bundle(timeout)
    output_dir = ARTIFACT_DIR / "restored_text_release_evidence"
    output_json = ARTIFACT_DIR / "text_release_evidence_restore.json"
    out = run_cmd(
        [
            sys.executable,
            "unbundle_text_release_evidence.py",
            "--archive",
            str(output_archive),
            "--manifest",
            str(bundle_manifest),
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(output_json),
            "--clean",
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence unbundler", out, "release evidence unbundler marker")
    require(r"RELEASE EVIDENCE UNBUNDLE OK", out, "release evidence unbundle ok")
    if not output_json.exists():
        raise VerificationError(f"Release evidence restore JSON was not created: {output_json}")
    report = json.loads(output_json.read_text(encoding="utf-8"))
    if report.get("valid") is not True:
        raise VerificationError(f"Release evidence restore report is not valid: {report}")
    if int(report.get("file_count") or 0) != 16 or int(report.get("member_count") or 0) != 17:
        raise VerificationError(f"Release evidence restore did not preserve expected file/member counts: {report}")
    restored_manifest = Path(report["restored_manifest"])
    if not restored_manifest.exists():
        raise VerificationError(f"Restored evidence manifest missing: {restored_manifest}")
    validation = report.get("validation") if isinstance(report.get("validation"), dict) else {}
    if validation.get("has_inner_manifest") is not True or validation.get("valid") is not True:
        raise VerificationError(f"Restored evidence validation did not pass: {validation}")
    labels = {item.get("label") for item in report.get("files", []) if isinstance(item, dict)}
    for label in ["gate_summary", "release_report", "dashboard", "memory_suite"]:
        if label not in labels:
            raise VerificationError(f"Restored evidence is missing label {label}: {labels}")
    pass_line(
        "text-release-evidence-unbundle",
        f"files {report['file_count']}, members {report['member_count']}",
    )


def verify_text_release_evidence_comparison(timeout: int) -> None:
    bundle_manifest = ARTIFACT_DIR / "text_release_evidence_manifest.json"
    if not bundle_manifest.exists():
        verify_text_release_evidence_bundle(timeout)
    output_json = ARTIFACT_DIR / "text_release_evidence_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_release_evidence.py",
            "--baseline-manifest",
            str(bundle_manifest),
            "--candidate-manifest",
            str(bundle_manifest),
            "--fail-on-archive-sha-change",
            "--fail-on-file-set-change",
            "--fail-on-file-sha-change",
            "--fail-on-file-byte-change",
            "--fail-on-archive-name-change",
            "--fail-on-validity-change",
            "--fail-on-gate-regression",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence comparison", out, "release evidence comparison marker")
    require(r"RELEASE EVIDENCE COMPARISON OK", out, "release evidence comparison ok")
    if not output_json.exists():
        raise VerificationError(f"Release evidence comparison JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    comparison = payload.get("comparison") or {}
    if payload.get("failures"):
        raise VerificationError(f"Release evidence comparison should not fail for identical manifests: {payload}")
    for key in ["same_archive_sha256", "same_file_set", "same_file_sha", "same_file_bytes", "same_archive_names"]:
        if comparison.get(key) is not True:
            raise VerificationError(f"Release evidence comparison mismatch {key}: {comparison}")
    if comparison.get("file_count_delta") != 0 or comparison.get("failed_gate_count_delta") != 0:
        raise VerificationError(f"Release evidence comparison should have zero deltas: {comparison}")
    pass_line(
        "text-release-evidence-compare",
        f"same_sha {comparison['same_archive_sha256']}, files {len(comparison['common_labels'])}",
    )


def verify_text_release_evidence_ledger(timeout: int) -> None:
    from register_text_release_evidence import verify_chain

    bundle_manifest = ARTIFACT_DIR / "text_release_evidence_manifest.json"
    output_archive = ARTIFACT_DIR / "text_release_evidence.tar.gz"
    validation_json = ARTIFACT_DIR / "text_release_evidence_validation.json"
    restore_json = ARTIFACT_DIR / "text_release_evidence_restore.json"
    comparison_json = ARTIFACT_DIR / "text_release_evidence_comparison.json"
    if not comparison_json.exists():
        verify_text_release_evidence_comparison(timeout)
    if not restore_json.exists():
        verify_text_release_evidence_unbundle(timeout)
    ledger_json = ARTIFACT_DIR / "text_release_evidence_ledger.json"
    store_dir = ARTIFACT_DIR / "text_release_evidence_store"
    ledger_json.unlink(missing_ok=True)
    if store_dir.exists():
        import shutil

        shutil.rmtree(store_dir)
    out = run_cmd(
        [
            sys.executable,
            "register_text_release_evidence.py",
            "--manifest",
            str(bundle_manifest),
            "--archive",
            str(output_archive),
            "--validation-json",
            str(validation_json),
            "--restore-json",
            str(restore_json),
            "--comparison-json",
            str(comparison_json),
            "--ledger",
            str(ledger_json),
            "--store-dir",
            str(store_dir),
            "--name",
            "verify_text_release_evidence",
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence ledger", out, "release evidence ledger marker")
    require(r"RELEASE EVIDENCE LEDGER OK", out, "release evidence ledger ok")
    if not ledger_json.exists():
        raise VerificationError(f"Release evidence ledger JSON was not created: {ledger_json}")
    ledger = json.loads(ledger_json.read_text(encoding="utf-8"))
    try:
        verify_chain(ledger)
    except RuntimeError as exc:
        raise VerificationError(f"Release evidence ledger chain did not verify: {exc}") from exc
    if ledger.get("entry_count") != 1 or not ledger.get("chain_head"):
        raise VerificationError(f"Release evidence ledger should contain one chained entry: {ledger}")
    entry = ledger["entries"][0]
    manifest = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    if entry.get("archive_sha256") != manifest.get("archive_sha256"):
        raise VerificationError(f"Release evidence ledger archive sha mismatch: {entry}")
    if entry.get("entry_hash") != ledger.get("chain_head"):
        raise VerificationError(f"Release evidence ledger head mismatch: {ledger}")
    pass_line(
        "text-release-evidence-ledger",
        f"entries {ledger['entry_count']}, head {ledger['chain_head'][:12]}",
    )


def verify_text_release_evidence_ledger_validator(timeout: int) -> None:
    ledger_json = ARTIFACT_DIR / "text_release_evidence_ledger.json"
    if not ledger_json.exists():
        verify_text_release_evidence_ledger(timeout)
    output_json = ARTIFACT_DIR / "text_release_evidence_ledger_validation.json"
    out = run_cmd(
        [
            sys.executable,
            "validate_text_release_evidence_ledger.py",
            "--ledger",
            str(ledger_json),
            "--output-json",
            str(output_json),
            "--expected-entries",
            "1",
            "--require-artifacts",
            "--artifact-scope",
            "latest",
            "--fail-on-failed-gates",
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence ledger validator", out, "release evidence ledger validator marker")
    require(r"RELEASE EVIDENCE LEDGER VALIDATION OK", out, "release evidence ledger validation ok")
    if not output_json.exists():
        raise VerificationError(f"Release evidence ledger validation JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("valid") is not True:
        raise VerificationError(f"Release evidence ledger validation failed: {result}")
    if int(result.get("entry_count") or 0) != 1 or not result.get("chain_head"):
        raise VerificationError(f"Release evidence ledger validation did not preserve chain summary: {result}")
    if result.get("last_entry_hash") != result.get("chain_head"):
        raise VerificationError(f"Release evidence ledger validation head mismatch: {result}")
    if int(result.get("artifact_checks") or 0) < 4:
        raise VerificationError(f"Release evidence ledger validation did not check linked artifacts: {result}")
    pass_line(
        "text-release-evidence-ledger-validate",
        f"entries {result['entry_count']}, artifacts {result['artifact_checks']}",
    )


def verify_text_release_evidence_ledger_tamper(timeout: int) -> None:
    ledger_json = ARTIFACT_DIR / "text_release_evidence_ledger.json"
    if not ledger_json.exists():
        verify_text_release_evidence_ledger(timeout)
    tampered_json = ARTIFACT_DIR / "text_release_evidence_ledger_tampered.json"
    output_json = ARTIFACT_DIR / "text_release_evidence_ledger_tamper.json"
    out = run_cmd(
        [
            sys.executable,
            "tamper_text_release_evidence_ledger.py",
            "--ledger",
            str(ledger_json),
            "--output-ledger",
            str(tampered_json),
            "--output-json",
            str(output_json),
            "--mode",
            "archive_sha256",
            "--expected-entries",
            "1",
            "--require-artifacts",
            "--artifact-scope",
            "latest",
            "--fail-on-failed-gates",
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence ledger tamper test", out, "release evidence ledger tamper marker")
    require(r"RELEASE EVIDENCE LEDGER TAMPER TEST OK", out, "release evidence ledger tamper ok")
    if not output_json.exists():
        raise VerificationError(f"Release evidence ledger tamper JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("valid") is not True:
        raise VerificationError(f"Release evidence ledger tamper test did not pass: {result}")
    if result.get("detected") is not True or result.get("expected_signal_found") is not True:
        raise VerificationError(f"Release evidence ledger tamper was not clearly detected: {result}")
    original = result.get("original_validation") if isinstance(result.get("original_validation"), dict) else {}
    tampered = result.get("tampered_validation") if isinstance(result.get("tampered_validation"), dict) else {}
    if original.get("valid") is not True or tampered.get("valid") is not False:
        raise VerificationError(f"Unexpected tamper validation polarity: {result}")
    errors = tampered.get("errors") if isinstance(tampered.get("errors"), list) else []
    if not any("Ledger entry hash mismatch" in str(error) for error in errors):
        raise VerificationError(f"Tamper report did not include hash mismatch: {result}")
    if not tampered_json.exists():
        raise VerificationError(f"Tampered ledger copy was not created: {tampered_json}")
    pass_line(
        "text-release-evidence-ledger-tamper",
        f"mode {result['tamper']['mode']}, detected {result['detected']}",
    )


def verify_text_release_evidence_ledger_replay(timeout: int) -> None:
    bundle_manifest = ARTIFACT_DIR / "text_release_evidence_manifest.json"
    output_archive = ARTIFACT_DIR / "text_release_evidence.tar.gz"
    validation_json = ARTIFACT_DIR / "text_release_evidence_validation.json"
    restore_json = ARTIFACT_DIR / "text_release_evidence_restore.json"
    comparison_json = ARTIFACT_DIR / "text_release_evidence_comparison.json"
    if not comparison_json.exists():
        verify_text_release_evidence_comparison(timeout)
    if not restore_json.exists():
        verify_text_release_evidence_unbundle(timeout)
    ledger_json = ARTIFACT_DIR / "text_release_evidence_ledger_replay.json"
    store_dir = ARTIFACT_DIR / "text_release_evidence_ledger_replay_store"
    work_dir = ARTIFACT_DIR / "text_release_evidence_ledger_replay"
    output_json = ARTIFACT_DIR / "text_release_evidence_ledger_replay_report.json"
    out = run_cmd(
        [
            sys.executable,
            "replay_text_release_evidence_ledger.py",
            "--manifest",
            str(bundle_manifest),
            "--archive",
            str(output_archive),
            "--validation-json",
            str(validation_json),
            "--restore-json",
            str(restore_json),
            "--comparison-json",
            str(comparison_json),
            "--base-dir",
            str(ARTIFACT_DIR),
            "--ledger",
            str(ledger_json),
            "--store-dir",
            str(store_dir),
            "--work-dir",
            str(work_dir),
            "--output-json",
            str(output_json),
            "--clean",
            "--fail-on-failed-gates",
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence ledger replay test", out, "release evidence ledger replay marker")
    require(r"RELEASE EVIDENCE LEDGER REPLAY TEST OK", out, "release evidence ledger replay ok")
    if not output_json.exists():
        raise VerificationError(f"Release evidence ledger replay JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("valid") is not True:
        raise VerificationError(f"Release evidence ledger replay test did not pass: {result}")
    actions = [item.get("action") for item in result.get("operations", []) if isinstance(item, dict)]
    if actions != ["inserted", "updated", "inserted"]:
        raise VerificationError(f"Unexpected release evidence ledger replay actions: {result}")
    validation = result.get("ledger_validation") if isinstance(result.get("ledger_validation"), dict) else {}
    if validation.get("valid") is not True or int(validation.get("entry_count") or 0) != 2:
        raise VerificationError(f"Replay ledger validation did not prove two valid entries: {result}")
    if result.get("previous_hash_linked") is not True or result.get("distinct_evidence") is not True:
        raise VerificationError(f"Replay ledger did not prove linked distinct evidence: {result}")
    if int(validation.get("artifact_checks") or 0) != int(result.get("expected_artifact_checks") or -1):
        raise VerificationError(f"Replay ledger artifact checks mismatch: {result}")
    pass_line(
        "text-release-evidence-ledger-replay",
        f"actions {','.join(actions)}, entries {validation['entry_count']}",
    )


def verify_text_release_evidence_ledger_comparison(timeout: int) -> None:
    ledger_json = ARTIFACT_DIR / "text_release_evidence_ledger.json"
    replay_ledger_json = ARTIFACT_DIR / "text_release_evidence_ledger_replay.json"
    if not replay_ledger_json.exists():
        verify_text_release_evidence_ledger_replay(timeout)
    output_json = ARTIFACT_DIR / "text_release_evidence_ledger_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_release_evidence_ledgers.py",
            "--baseline-ledger",
            str(replay_ledger_json),
            "--candidate-ledger",
            str(replay_ledger_json),
            "--output-json",
            str(output_json),
            "--require-artifacts",
            "--artifact-scope",
            "all",
            "--fail-on-failed-gates",
            "--fail-on-chain-head-change",
            "--fail-on-entry-count-change",
            "--fail-on-entry-set-change",
            "--fail-on-entry-field-change",
            "--fail-on-archive-set-change",
            "--fail-on-manifest-set-change",
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence ledger comparison", out, "release evidence ledger comparison marker")
    require(r"RELEASE EVIDENCE LEDGER COMPARISON OK", out, "release evidence ledger comparison ok")
    if not output_json.exists():
        raise VerificationError(f"Release evidence ledger comparison JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    comparison = result.get("comparison") if isinstance(result.get("comparison"), dict) else {}
    if result.get("failures"):
        raise VerificationError(f"Release evidence ledger self-comparison should not fail: {result}")
    for key in ["same_chain_head", "same_entry_count", "same_entry_hash_set", "same_archive_sha_set", "same_manifest_sha_set"]:
        if comparison.get(key) is not True:
            raise VerificationError(f"Release evidence ledger self-comparison mismatch {key}: {result}")
    changed_common = comparison.get("changed_common_entries") if isinstance(comparison.get("changed_common_entries"), list) else []
    if changed_common:
        raise VerificationError(f"Release evidence ledger self-comparison changed common entries: {result}")

    growth_json = ARTIFACT_DIR / "text_release_evidence_ledger_growth_comparison.json"
    growth_out = run_cmd(
        [
            sys.executable,
            "compare_text_release_evidence_ledgers.py",
            "--baseline-ledger",
            str(ledger_json),
            "--candidate-ledger",
            str(replay_ledger_json),
            "--output-json",
            str(growth_json),
            "--require-artifacts",
            "--artifact-scope",
            "all",
            "--fail-on-failed-gates",
        ],
        timeout,
    )
    require(r"RELEASE EVIDENCE LEDGER COMPARISON OK", growth_out, "release evidence ledger growth comparison ok")
    growth = json.loads(growth_json.read_text(encoding="utf-8"))
    growth_comparison = growth.get("comparison") if isinstance(growth.get("comparison"), dict) else {}
    if int(growth_comparison.get("entry_count_delta") or 0) != 1:
        raise VerificationError(f"Release evidence ledger growth comparison should show one added entry: {growth}")
    if growth_comparison.get("same_chain_head") is not False or growth_comparison.get("same_entry_count") is not False:
        raise VerificationError(f"Release evidence ledger growth comparison did not show ledger drift: {growth}")
    pass_line(
        "text-release-evidence-ledger-compare",
        f"self_same {comparison['same_chain_head']}, growth_delta {growth_comparison['entry_count_delta']}",
    )


def verify_text_release_evidence_audit(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "text_release_evidence_audit.json"
    output_md = ARTIFACT_DIR / "text_release_evidence_audit.md"
    growth_json = ARTIFACT_DIR / "text_release_evidence_ledger_growth_comparison.json"
    if not growth_json.exists():
        verify_text_release_evidence_ledger_comparison(timeout)
    out = run_cmd(
        [
            sys.executable,
            "summarize_text_release_evidence_audit.py",
            "--manifest",
            str(ARTIFACT_DIR / "text_release_evidence_manifest.json"),
            "--validation-json",
            str(ARTIFACT_DIR / "text_release_evidence_validation.json"),
            "--restore-json",
            str(ARTIFACT_DIR / "text_release_evidence_restore.json"),
            "--comparison-json",
            str(ARTIFACT_DIR / "text_release_evidence_comparison.json"),
            "--ledger",
            str(ARTIFACT_DIR / "text_release_evidence_ledger.json"),
            "--ledger-validation-json",
            str(ARTIFACT_DIR / "text_release_evidence_ledger_validation.json"),
            "--tamper-json",
            str(ARTIFACT_DIR / "text_release_evidence_ledger_tamper.json"),
            "--replay-json",
            str(ARTIFACT_DIR / "text_release_evidence_ledger_replay_report.json"),
            "--ledger-comparison-json",
            str(ARTIFACT_DIR / "text_release_evidence_ledger_comparison.json"),
            "--ledger-growth-comparison-json",
            str(growth_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence audit", out, "release evidence audit marker")
    require(r"RELEASE EVIDENCE AUDIT OK", out, "release evidence audit ok")
    for path in [output_json, output_md]:
        if not path.exists():
            raise VerificationError(f"Release evidence audit artifact was not created: {path}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    checks = result.get("checks") if isinstance(result.get("checks"), list) else []
    if result.get("kind") != "text_release_evidence_audit" or result.get("valid") is not True:
        raise VerificationError(f"Release evidence audit did not pass: {result}")
    check_count = int(summary.get("check_count") if summary.get("check_count") is not None else 0)
    failed_check_count = int(summary.get("failed_check_count") if summary.get("failed_check_count") is not None else -1)
    if check_count != 10 or failed_check_count != 0:
        raise VerificationError(f"Release evidence audit check counts mismatch: {result}")
    if len(checks) != 10 or any(not isinstance(check, dict) or check.get("valid") is not True for check in checks):
        raise VerificationError(f"Release evidence audit checks should all pass: {result}")
    if summary.get("replay_actions") != ["inserted", "updated", "inserted"]:
        raise VerificationError(f"Release evidence audit replay actions mismatch: {result}")
    if int(summary.get("source_entry_count") or 0) != 1 or int(summary.get("replay_entry_count") or 0) != 2:
        raise VerificationError(f"Release evidence audit ledger entry counts mismatch: {result}")
    if int(summary.get("growth_entry_count_delta") or 0) != 1 or summary.get("tamper_detected") is not True:
        raise VerificationError(f"Release evidence audit growth/tamper summary mismatch: {result}")
    markdown = output_md.read_text(encoding="utf-8")
    for marker in ["# TextPy/SoA Release Evidence Audit", "## Verdict", "## Checks", "make summarize-release-evidence-audit-demo"]:
        if marker not in markdown:
            raise VerificationError(f"Release evidence audit Markdown missing {marker!r}: {markdown[:600]}")
    pass_line(
        "text-release-evidence-audit",
        f"checks {check_count}, replay {summary['source_entry_count']}->{summary['replay_entry_count']}",
    )


def verify_text_release_evidence_audit_validator(timeout: int) -> None:
    audit_json = ARTIFACT_DIR / "text_release_evidence_audit.json"
    if not audit_json.exists():
        verify_text_release_evidence_audit(timeout)
    output_json = ARTIFACT_DIR / "text_release_evidence_audit_validation.json"
    out = run_cmd(
        [
            sys.executable,
            "validate_text_release_evidence_audit.py",
            "--audit",
            str(audit_json),
            "--output-json",
            str(output_json),
            "--require-sources",
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence audit validator", out, "release evidence audit validator marker")
    require(r"RELEASE EVIDENCE AUDIT VALIDATION OK", out, "release evidence audit validator ok")
    if not output_json.exists():
        raise VerificationError(f"Release evidence audit validation JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    expected = {
        "valid": True,
        "check_count": 10,
        "failed_check_count": 0,
        "failure_count": 0,
        "source_artifact_checks": 10,
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            raise VerificationError(f"Release evidence audit validation mismatch {key}: {result}")
    if result.get("errors"):
        raise VerificationError(f"Release evidence audit validation produced errors: {result}")
    pass_line(
        "text-release-evidence-audit-validate",
        f"checks {result['check_count']}, sources {result['source_artifact_checks']}",
    )


def verify_text_release_evidence_audit_validation_comparison(timeout: int) -> None:
    validation_json = ARTIFACT_DIR / "text_release_evidence_audit_validation.json"
    if not validation_json.exists():
        verify_text_release_evidence_audit_validator(timeout)
    output_json = ARTIFACT_DIR / "text_release_evidence_audit_validation_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_release_evidence_audit_validations.py",
            "--baseline-validation",
            str(validation_json),
            "--candidate-validation",
            str(validation_json),
            "--output-json",
            str(output_json),
            "--fail-on-validity-change",
            "--fail-on-audit-path-change",
            "--fail-on-release-name-change",
            "--fail-on-check-count-change",
            "--fail-on-source-count-change",
            "--fail-on-failed-check-regression",
            "--fail-on-failure-regression",
            "--fail-on-error-regression",
            "--fail-on-artifact-check-regression",
            "--fail-on-source-artifact-check-regression",
            "--fail-on-error-set-change",
        ],
        timeout,
    )
    require(
        r"TextPy/SoA release evidence audit validation comparison",
        out,
        "release evidence audit validation comparison marker",
    )
    require(
        r"RELEASE EVIDENCE AUDIT VALIDATION COMPARISON OK",
        out,
        "release evidence audit validation comparison ok",
    )
    if not output_json.exists():
        raise VerificationError(f"Release evidence audit validation comparison JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    comparison = result.get("comparison") if isinstance(result.get("comparison"), dict) else {}
    if result.get("failures"):
        raise VerificationError(f"Release evidence audit validation self-comparison should not fail: {result}")
    required_true = [
        "same_valid",
        "same_audit",
        "same_release_name",
        "same_errors",
    ]
    for key in required_true:
        if comparison.get(key) is not True:
            raise VerificationError(f"Release evidence audit validation comparison mismatch {key}: {result}")
    zero_deltas = [
        "check_count_delta",
        "failed_check_count_delta",
        "failure_count_delta",
        "artifact_check_count_delta",
        "source_artifact_checks_delta",
        "source_count_delta",
        "error_count_delta",
    ]
    for key in zero_deltas:
        if int(comparison.get(key) or 0) != 0:
            raise VerificationError(f"Release evidence audit validation comparison delta {key}: {result}")
    pass_line(
        "text-release-evidence-audit-validation-compare",
        f"same_valid {comparison['same_valid']}, errors {comparison['error_count_delta']}",
    )


def verify_text_release_evidence_audit_comparison(timeout: int) -> None:
    audit_json = ARTIFACT_DIR / "text_release_evidence_audit.json"
    if not audit_json.exists():
        verify_text_release_evidence_audit(timeout)
    output_json = ARTIFACT_DIR / "text_release_evidence_audit_comparison.json"
    out = run_cmd(
        [
            sys.executable,
            "compare_text_release_evidence_audits.py",
            "--baseline-audit",
            str(audit_json),
            "--candidate-audit",
            str(audit_json),
            "--output-json",
            str(output_json),
            "--fail-on-validity-change",
            "--fail-on-release-name-change",
            "--fail-on-archive-sha-change",
            "--fail-on-chain-head-change",
            "--fail-on-replay-action-change",
            "--fail-on-tamper-change",
            "--fail-on-entry-count-change",
            "--fail-on-check-set-change",
            "--fail-on-check-validity-change",
            "--fail-on-check-message-change",
            "--fail-on-check-metric-change",
            "--fail-on-check-path-change",
            "--fail-on-source-set-change",
            "--fail-on-source-path-change",
            "--fail-on-failed-check-regression",
            "--fail-on-failure-regression",
            "--fail-on-artifact-check-regression",
        ],
        timeout,
    )
    require(r"TextPy/SoA release evidence audit comparison", out, "release evidence audit comparison marker")
    require(r"RELEASE EVIDENCE AUDIT COMPARISON OK", out, "release evidence audit comparison ok")
    if not output_json.exists():
        raise VerificationError(f"Release evidence audit comparison JSON was not created: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    comparison = result.get("comparison") if isinstance(result.get("comparison"), dict) else {}
    if result.get("failures"):
        raise VerificationError(f"Release evidence audit self-comparison should not fail: {result}")
    required_true = [
        "same_valid",
        "same_release_name",
        "same_archive_sha256",
        "same_source_chain_head",
        "same_replay_chain_head",
        "same_replay_actions",
        "same_tamper_detected",
        "same_check_set",
        "same_check_validity",
        "same_check_messages",
        "same_check_metrics",
        "same_check_paths",
        "same_source_set",
        "same_source_paths",
    ]
    for key in required_true:
        if comparison.get(key) is not True:
            raise VerificationError(f"Release evidence audit self-comparison mismatch {key}: {result}")
    zero_deltas = [
        "source_entry_count_delta",
        "replay_entry_count_delta",
        "growth_entry_count_delta_change",
        "artifact_check_count_delta",
        "check_count_delta",
        "failed_check_count_delta",
        "failure_count_delta",
    ]
    for key in zero_deltas:
        if int(comparison.get(key) or 0) != 0:
            raise VerificationError(f"Release evidence audit self-comparison delta {key}: {result}")
    pass_line(
        "text-release-evidence-audit-compare",
        f"same_checks {comparison['same_check_set']}, failures {len(result.get('failures') or [])}",
    )


def verify_evolution_replay(timeout: int) -> None:
    genome = ARTIFACT_DIR / "best_genome_verify.npz"
    genome.parent.mkdir(parents=True, exist_ok=True)
    out = run_cmd(
        [
            sys.executable,
            "evolve_ssm_agents.py",
            "--population",
            "8",
            "--elites",
            "2",
            "--agents",
            "128",
            "--steps",
            "8",
            "--generations",
            "2",
            "--log-every",
            "1",
            "--save-best",
            str(genome),
        ],
        timeout,
    )
    require(r"saved_best:", out, "saved genome marker")
    if not genome.exists():
        raise VerificationError(f"Genome was not created: {genome}")
    best_reward = parse_float(r"best_reward_seen:\s+([0-9.]+)", out, "best reward")

    replay_out = run_cmd(
        [
            sys.executable,
            "run_evolved_agent.py",
            "--genome",
            str(genome),
            "--agents",
            "128",
            "--steps",
            "8",
            "--seed",
            "303",
        ],
        timeout,
    )
    require(r"stored_metrics:", replay_out, "stored metrics marker")
    start_goal, end_goal = parse_distance_pair("mean_goal_distance", replay_out)
    if end_goal >= start_goal:
        raise VerificationError(f"Replayed genome did not improve goal distance:\n{replay_out}")
    replay_reward = parse_float(r"reward:\s+([0-9.]+)", replay_out, "replay reward")
    pass_line(
        "evolution-replay",
        f"evolved reward {best_reward:.4f}, replay reward {replay_reward:.4f}, goal {start_goal:.4f}->{end_goal:.4f}",
    )


def verify_formula_evolution(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "formula_evolution_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "evolve_ssm_formulas.py",
            "--architecture-population",
            "3",
            "--architecture-elites",
            "1",
            "--architecture-generations",
            "1",
            "--weight-population",
            "8",
            "--weight-elites",
            "2",
            "--weight-generations",
            "1",
            "--agents",
            "256",
            "--steps",
            "16",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA formula evolution prototype", out, "formula evolution marker")
    require(r"Best formula search result", out, "formula evolution result marker")
    if not output_json.exists():
        raise VerificationError(f"Formula evolution JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    leaderboard = payload["leaderboard"]
    best = payload["best_record"]
    baseline = payload["baseline_record"]
    if protocol["search_space_size"] <= protocol["architecture_population"]:
        raise VerificationError(f"Formula search space is unexpectedly small: {protocol}")
    expected_records = (protocol["architecture_generations"] + 1) * protocol["architecture_population"]
    if len(leaderboard) != expected_records:
        raise VerificationError(f"Unexpected formula leaderboard length: {len(leaderboard)} != {expected_records}")
    if baseline is None:
        raise VerificationError("Formula evolution did not evaluate the baseline formula.")
    if float(best["best"]["reward"]) <= 0.0:
        raise VerificationError(f"Formula evolution best reward should be positive: {best}")
    if "autonomous AGI" not in payload["claim_guardrail"]:
        raise VerificationError("Formula evolution JSON is missing the AGI guardrail.")
    pass_line(
        "formula-evolution",
        f"best_reward {best['best']['reward']:.4f}, formulas {len(leaderboard)}, best {best['formula']['id']}",
    )


def verify_hypothesis_loop(timeout: int) -> None:
    generator_py = ARTIFACT_DIR / "hypothesis_generator.py"
    output_json = ARTIFACT_DIR / "hypothesis_loop_verify.json"
    feedback_jsonl = ARTIFACT_DIR / "hypothesis_feedback_verify.jsonl"
    prompt_json = ARTIFACT_DIR / "hypothesis_prompt_verify.json"
    generator_py.write_text(
        "\n".join(
            [
                "import json, sys",
                "json.loads(sys.stdin.read())",
                "items = [",
                "  {'id':'valid_relu_max','rationale':'generated valid probe','formula':{'memory_op':'relu','input_op':'identity','combine_op':'max','final_op':'tanh'}},",
                "  {'id':'invalid_arbitrary_code','rationale':'generated invalid probe','formula':{'memory_op':'eval','input_op':'identity','combine_op':'add','final_op':'tanh'}},",
                "]",
                "for item in items:",
                "    print(json.dumps(item))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = run_cmd(
        [
            sys.executable,
            "run_hypothesis_loop.py",
            "--generator-command",
            f"{sys.executable} {generator_py}",
            "--max-proposals",
            "4",
            "--weight-population",
            "8",
            "--weight-elites",
            "2",
            "--weight-generations",
            "1",
            "--agents",
            "256",
            "--steps",
            "16",
            "--output-json",
            str(output_json),
            "--feedback-jsonl",
            str(feedback_jsonl),
            "--prompt-json",
            str(prompt_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA hypothesis feedback loop", out, "hypothesis loop marker")
    require(r"Hypothesis loop result", out, "hypothesis loop result marker")
    for path in [output_json, feedback_jsonl, prompt_json]:
        if not path.exists():
            raise VerificationError(f"Hypothesis loop artifact was not created: {path}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    feedback = payload["feedback"]
    if protocol["accepted"] < 1 or protocol["rejected"] < 1:
        raise VerificationError(f"Hypothesis loop should accept and reject proposals: {protocol}")
    if payload["best_accepted"] is None:
        raise VerificationError("Hypothesis loop did not record a best accepted proposal.")
    if not str(payload["proposal_source"]).startswith("generator_command:"):
        raise VerificationError(f"Hypothesis loop did not use generator command: {payload['proposal_source']}")
    if "arbitrary code" not in payload["claim_guardrail"]:
        raise VerificationError("Hypothesis loop JSON is missing the arbitrary-code guardrail.")
    if "schema" not in json.loads(prompt_json.read_text(encoding="utf-8")):
        raise VerificationError("Hypothesis prompt context is missing schema.")
    feedback_lines = [json.loads(line) for line in feedback_jsonl.read_text(encoding="utf-8").splitlines() if line]
    if len(feedback_lines) != len(feedback):
        raise VerificationError("Hypothesis feedback JSONL does not match JSON feedback.")
    pass_line(
        "hypothesis-loop",
        f"accepted {protocol['accepted']}, rejected {protocol['rejected']}, best {payload['best_accepted']['id']}",
    )


def verify_code_block_evolution(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "code_block_evolution_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "evolve_code_blocks.py",
            "--target",
            "square_minus_one",
            "--population",
            "512",
            "--generations",
            "30",
            "--elites",
            "32",
            "--program-length",
            "7",
            "--cases",
            "17",
            "--log-every",
            "10",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA CodePy block evolution prototype", out, "code block marker")
    require(r"rendered_python:", out, "rendered code marker")
    if not output_json.exists():
        raise VerificationError(f"Code block evolution JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    best = payload["best"]
    if payload["target"]["name"] != "square_minus_one":
        raise VerificationError(f"Unexpected CodePy target: {payload['target']}")
    if len(payload["block_vocabulary"]) < 10:
        raise VerificationError("CodePy block vocabulary is too small.")
    if "arbitrary Python" not in payload["guardrail"]:
        raise VerificationError("CodePy JSON is missing the arbitrary-Python guardrail.")
    if float(best["train_mse"]) > 1.0e-4 or float(best["holdout_mse"]) > 1.0e-3:
        raise VerificationError(f"CodePy block evolution did not solve the smoke target: {best}")
    pass_line(
        "code-block-evolution",
        f"target {payload['target']['name']}, train_mse {best['train_mse']:.2e}, active {best['active_blocks']}",
    )


def verify_code_tape_prior_pipeline(timeout: int) -> None:
    trace_jsonl = ARTIFACT_DIR / "code_tape_prior_verify_traces.jsonl"
    trace_run_json = ARTIFACT_DIR / "code_tape_prior_verify_trace_run.json"
    prior_json = ARTIFACT_DIR / "code_tape_prior_verify_prior.json"
    search_json = ARTIFACT_DIR / "code_tape_prior_verify_search.json"
    macros_json = ARTIFACT_DIR / "code_tape_prior_verify_macros.json"
    out = run_cmd(
        [
            sys.executable,
            "evolve_code_tape.py",
            "--target",
            "sum4",
            "--population",
            "128",
            "--generations",
            "1",
            "--elites",
            "16",
            "--program-length",
            "6",
            "--max-steps",
            "32",
            "--cases",
            "16",
            "--seed",
            "7",
            "--inject-reference",
            "--trace-jsonl",
            str(trace_jsonl),
            "--trace-all",
            "--output-json",
            str(trace_run_json),
        ],
        timeout,
    )
    require(r"trace_records_written: 128", out, "tape trace count marker")
    trace_count = sum(1 for _ in trace_jsonl.open(encoding="utf-8"))
    if trace_count != 128:
        raise VerificationError(f"Unexpected CodePy tape trace count: {trace_count}")
    run_cmd(
        [
            sys.executable,
            "train_code_tape_prior.py",
            "--trace-jsonl",
            str(trace_jsonl),
            "--output-json",
            str(prior_json),
            "--program-length",
            "6",
            "--vocab-size",
            "35",
            "--epochs",
            "80",
        ],
        timeout,
    )
    prior = json.loads(prior_json.read_text(encoding="utf-8"))
    if int(prior["metrics"]["train"]["positives"]) < 1:
        raise VerificationError("CodePy tape prior verifier found no positive records.")
    if "coverage" not in prior or "insufficient_positive_coverage" not in prior["coverage"]:
        raise VerificationError("CodePy tape prior JSON is missing coverage diagnostics.")
    run_cmd(
        [
            sys.executable,
            "search_code_tape_prior.py",
            "--prior-json",
            str(prior_json),
            "--target",
            "sum4",
            "--program-length",
            "6",
            "--max-steps",
            "32",
            "--cases",
            "16",
            "--beam-width",
            "8",
            "--expand-top-k",
            "10",
            "--stop-on-solution",
            "--output-json",
            str(search_json),
        ],
        timeout,
    )
    search = json.loads(search_json.read_text(encoding="utf-8"))
    if search["first_solution"] is None:
        raise VerificationError("CodePy tape prior-guided search did not find a solution.")
    run_cmd(
        [
            sys.executable,
            "promote_code_tape_macros.py",
            "--trace-jsonl",
            str(trace_jsonl),
            "--search-json",
            str(search_json),
            "--top-k",
            "4",
            "--output-json",
            str(macros_json),
        ],
        timeout,
    )
    macros = json.loads(macros_json.read_text(encoding="utf-8"))
    if not macros["macros"]:
        raise VerificationError("CodePy tape macro promotion produced no macros.")
    pass_line(
        "code-tape-prior",
        f"traces {trace_count}, beam_eval {search['search']['evaluated_programs']}, macros {len(macros['macros'])}",
    )


def verify_code_tape_argmax_geometry(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "code_tape_argmax_geometry_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "evolve_code_tape.py",
            "--target",
            "argmax_index4",
            "--population",
            "128",
            "--generations",
            "1",
            "--elites",
            "16",
            "--program-length",
            "11",
            "--max-steps",
            "64",
            "--cases",
            "16",
            "--inject-reference",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"target: argmax_index4", out, "argmax target marker")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    best = payload["best"]
    reference = payload["reference_program"]["metrics"]
    if float(reference["train_mse"]) > 1.0e-8 or float(reference["holdout_mse"]) > 1.0e-8:
        raise VerificationError(f"Argmax reference program is not zero-error: {reference}")
    if float(best["train_mse"]) > 1.0e-8 or float(best["holdout_mse"]) > 1.0e-8:
        raise VerificationError(f"Argmax injected geometry did not surface as best: {best}")
    if len(payload["block_vocabulary"]) < 35:
        raise VerificationError("Expanded CodePy tape vocabulary is missing argmax blocks.")
    pass_line(
        "code-tape-argmax-geometry",
        f"reference train_mse {reference['train_mse']:.2e}, blocks {len(payload['block_vocabulary'])}",
    )


def verify_code_tape_trajectory_search(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "code_tape_trajectory_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "search_code_tape_trajectory.py",
            "--target",
            "argmax_index4",
            "--program-length",
            "11",
            "--max-steps",
            "64",
            "--cases",
            "16",
            "--beam-width",
            "16",
            "--signature-cases",
            "6",
            "--signature-steps",
            "18",
            "--diversity-penalty",
            "0.65",
            "--max-same-signature",
            "1",
            "--mse-weight",
            "0.45",
            "--seed-reference-prefixes",
            "--stop-on-solution",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"CodePy tape trajectory-guided search", out, "trajectory search marker")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    inference = payload["trajectory_inference"]
    graph = inference["latent_trace_graph"]
    reconstructed = payload["reconstructed_program"]
    if not inference["seeded_reference_prefixes"]:
        raise VerificationError("Trajectory verifier should run seeded reconstruction mode.")
    if inference["diversity_constraint"]["max_same_signature"] != 1:
        raise VerificationError("Trajectory verifier did not enforce max_same_signature=1.")
    if graph["nodes"] < 10 or graph["edges"] < 10:
        raise VerificationError(f"Trajectory graph is too small: {graph}")
    if float(graph["selected_signature_entropy"]) <= 0.0:
        raise VerificationError(f"Trajectory signature entropy did not register: {graph}")
    if float(reconstructed["train_mse"]) > 1.0e-8 or float(reconstructed["holdout_mse"]) > 1.0e-8:
        raise VerificationError(f"Seeded trajectory reconstruction is not zero-error: {reconstructed}")
    pass_line(
        "code-tape-trajectory",
        f"nodes {graph['nodes']}, edges {graph['edges']}, entropy {graph['selected_signature_entropy']:.2f}",
    )


def verify_code_tape_trace_prior(timeout: int) -> None:
    candidates_jsonl = ARTIFACT_DIR / "code_tape_trace_prior_candidates.jsonl"
    search_json = ARTIFACT_DIR / "code_tape_trace_prior_search.json"
    dataset_jsonl = ARTIFACT_DIR / "code_tape_trace_prior_dataset.jsonl"
    summary_json = ARTIFACT_DIR / "code_tape_trace_prior_summary.json"
    prior_json = ARTIFACT_DIR / "code_tape_trace_prior.json"
    out = run_cmd(
        [
            sys.executable,
            "search_code_tape_trajectory.py",
            "--target",
            "argmax_index4",
            "--program-length",
            "11",
            "--max-steps",
            "64",
            "--cases",
            "8",
            "--beam-width",
            "4",
            "--signature-cases",
            "4",
            "--signature-steps",
            "12",
            "--diversity-penalty",
            "0.65",
            "--max-same-signature",
            "1",
            "--mse-weight",
            "0.45",
            "--seed-reference-prefixes",
            "--stop-on-solution",
            "--candidate-jsonl",
            str(candidates_jsonl),
            "--output-json",
            str(search_json),
        ],
        timeout,
    )
    require(r"candidate_records_written:", out, "trace candidate export marker")
    candidate_count = sum(1 for _ in candidates_jsonl.open(encoding="utf-8"))
    if candidate_count < 100:
        raise VerificationError(f"Trace-prior verifier exported too few candidates: {candidate_count}")
    run_cmd(
        [
            sys.executable,
            "build_code_tape_trace_dataset.py",
            "--input-jsonl",
            str(candidates_jsonl),
            "--reference-json",
            str(search_json),
            "--output-jsonl",
            str(dataset_jsonl),
            "--summary-json",
            str(summary_json),
            "--dedupe",
            "--min-successes",
            "1",
            "--min-near-misses",
            "1",
        ],
        timeout,
    )
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    class_counts = summary["dataset"]["class_counts"]
    if int(class_counts.get("success", 0)) < 1 or int(class_counts.get("near_miss", 0)) < 1:
        raise VerificationError(f"Trace dataset lacks contrastive classes: {class_counts}")
    run_cmd(
        [
            sys.executable,
            "train_code_tape_trace_prior.py",
            "--trace-jsonl",
            str(dataset_jsonl),
            "--output-json",
            str(prior_json),
            "--hash-buckets",
            "32",
            "--epochs",
            "80",
            "--seed",
            "31",
        ],
        timeout,
    )
    prior = json.loads(prior_json.read_text(encoding="utf-8"))
    separation = prior["contrastive_separation"]
    if float(separation.get("success_vs_near_miss_gap", 0.0)) <= 0.1:
        raise VerificationError(f"Trace prior did not separate success from near-miss: {separation}")
    if not prior["coverage"]["has_near_misses"]:
        raise VerificationError("Trace prior coverage did not record near-miss support.")
    pass_line(
        "code-tape-trace-prior",
        f"records {summary['dataset']['records']}, gap {separation['success_vs_near_miss_gap']:.2f}",
    )


def verify_code_tape_trace_guidance_comparison(timeout: int) -> None:
    output_dir = ARTIFACT_DIR / "code_tape_trace_guidance"
    output_json = ARTIFACT_DIR / "code_tape_trace_guidance_comparison.json"
    prior_json = ROOT / "artifacts" / "code_tape_prior" / "argmax_index4_trace_prior.json"
    if not prior_json.exists():
        raise VerificationError(f"Trace-prior artifact is missing: {prior_json}")
    out = run_cmd(
        [
            sys.executable,
            "compare_code_tape_trace_guidance.py",
            "--trace-prior-json",
            str(prior_json),
            "--output-dir",
            str(output_dir),
            "--output-json",
            str(output_json),
            "--program-length",
            "6",
            "--max-steps",
            "32",
            "--cases",
            "8",
            "--beam-width",
            "2",
            "--signature-cases",
            "4",
            "--signature-steps",
            "10",
            "--trace-prior-weight",
            "0.25",
        ],
        timeout,
    )
    require(r"CodePy trace-guidance comparison", out, "trace-guidance comparison marker")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    runs = payload["runs"]
    comparisons = payload["comparisons"]
    if len(runs) != 2 or len(comparisons) != 1:
        raise VerificationError(f"Unexpected trace-guidance comparison shape: {payload}")
    baseline, guided = runs
    comparison = comparisons[0]
    if baseline["name"] != "baseline" or not guided["name"].startswith("trace_prior_w"):
        raise VerificationError(f"Unexpected run names: {[run['name'] for run in runs]}")
    if not comparison["same_budget"]:
        raise VerificationError(f"Trace-guidance comparison did not use equal budget: {comparison}")
    if guided.get("best_trace_prior_probability") is None:
        raise VerificationError("Guided trace-prior run did not report trace_prior_probability.")
    pass_line(
        "code-tape-trace-guidance",
        f"eval {baseline['evaluated_programs']}, guided_mse {guided['best_train_mse']:.2g}",
    )


def verify_code_tape_prefix_value_rollout(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "code_tape_prefix_value.json"
    candidates_jsonl = ARTIFACT_DIR / "code_tape_prefix_value_candidates.jsonl"
    prior_json = ROOT / "artifacts" / "code_tape_prior" / "argmax_index4_trace_prior.json"
    if not prior_json.exists():
        raise VerificationError(f"Trace-prior artifact is missing: {prior_json}")
    out = run_cmd(
        [
            sys.executable,
            "search_code_tape_trajectory.py",
            "--target",
            "argmax_index4",
            "--program-length",
            "11",
            "--max-steps",
            "64",
            "--cases",
            "8",
            "--beam-width",
            "4",
            "--signature-cases",
            "4",
            "--signature-steps",
            "10",
            "--trace-prior-json",
            str(prior_json),
            "--trace-prior-weight",
            "0.25",
            "--value-rollouts",
            "6",
            "--value-rollout-top-k",
            "6",
            "--value-rollout-role-top-k",
            "6",
            "--value-rollout-max-depth",
            "2",
            "--value-weight",
            "0.05",
            "--candidate-jsonl",
            str(candidates_jsonl),
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"value_rollout_evaluations:", out, "prefix-value rollout marker")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    search = payload["search"]
    if int(search.get("value_rollout_evaluations", 0)) <= 0:
        raise VerificationError(f"Prefix-value rollout did not evaluate suffixes: {search}")
    value_rows = 0
    with candidates_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("prefix_value_score") is not None:
                value_rows += 1
    if value_rows <= 0:
        raise VerificationError("Prefix-value JSONL contains no value-scored prefixes.")
    pass_line(
        "code-tape-prefix-value",
        f"rollouts {search['value_rollout_evaluations']}, value_rows {value_rows}",
    )


def verify_code_tape_survival_policy(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "code_tape_survival.json"
    candidates_jsonl = ARTIFACT_DIR / "code_tape_survival_candidates.jsonl"
    prior_json = ROOT / "artifacts" / "code_tape_prior" / "argmax_index4_trace_prior.json"
    if not prior_json.exists():
        raise VerificationError(f"Trace-prior artifact is missing: {prior_json}")
    out = run_cmd(
        [
            sys.executable,
            "search_code_tape_trajectory.py",
            "--target",
            "argmax_index4",
            "--program-length",
            "11",
            "--max-steps",
            "64",
            "--cases",
            "8",
            "--beam-width",
            "8",
            "--signature-cases",
            "4",
            "--signature-steps",
            "10",
            "--trace-prior-json",
            str(prior_json),
            "--trace-prior-weight",
            "0.25",
            "--value-rollouts",
            "12",
            "--value-rollout-top-k",
            "16",
            "--value-rollout-role-top-k",
            "16",
            "--value-rollout-max-depth",
            "3",
            "--value-weight",
            "0.03",
            "--survival-lane-fraction",
            "0.25",
            "--survival-ttl",
            "3",
            "--survival-min-value",
            "5.0",
            "--survival-max-per-root",
            "2",
            "--survival-ignore-signature-limit",
            "--candidate-jsonl",
            str(candidates_jsonl),
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"survival_activations:", out, "survival activation marker")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    search = payload["search"]
    if int(search.get("survival_activations", 0)) <= 0:
        raise VerificationError(f"Survival policy did not activate from prefix value: {search}")
    if int(search.get("survival_selected_total", 0)) <= 0:
        raise VerificationError(f"Survival policy did not select any protected prefixes: {search}")
    if float(search.get("survival_lane_fraction", 0.0)) <= 0.0 or int(search.get("survival_ttl", 0)) <= 0:
        raise VerificationError(f"Survival settings were not recorded: {search}")
    survival_rows = 0
    selected_survival_rows = 0
    survival_lane_rows = 0
    with candidates_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("survival_ttl", 0)) > 0:
                survival_rows += 1
                if row.get("selected_next_beam"):
                    selected_survival_rows += 1
                if row.get("selection_lane") == "survival":
                    survival_lane_rows += 1
    if survival_rows <= 0 or selected_survival_rows <= 0 or survival_lane_rows <= 0:
        raise VerificationError(
            "Survival JSONL did not record active, selected, and lane-selected prefixes: "
            f"{survival_rows}/{selected_survival_rows}/{survival_lane_rows}"
        )
    pass_line(
        "code-tape-survival",
        (
            f"activations {search['survival_activations']}, "
            f"selected {search['survival_selected_total']}, lane_rows {survival_lane_rows}"
        ),
    )


def verify_code_tape_late_game_analysis(timeout: int) -> None:
    candidates_jsonl = ROOT / "artifacts" / "code_tape_prior" / "argmax_index4_survival_b64_ttl8_uncapped_w025_candidates.jsonl"
    search_json = ROOT / "artifacts" / "code_tape_prior" / "argmax_index4_survival_b64_ttl8_uncapped_w025.json"
    output_json = ARTIFACT_DIR / "code_tape_late_game_analysis.json"
    output_md = ARTIFACT_DIR / "code_tape_late_game_analysis.md"
    if not candidates_jsonl.exists() or not search_json.exists():
        raise VerificationError(
            "Late-game analysis requires survival artifacts; run make code-tape-survival-demo first."
        )
    out = run_cmd(
        [
            sys.executable,
            "analyze_code_tape_late_game.py",
            "--input-jsonl",
            str(candidates_jsonl),
            "--search-json",
            str(search_json),
            "--target",
            "argmax_index4",
            "--top-n",
            "20",
            "--min-depth",
            "7",
            "--max-train-mse",
            "3.0",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        timeout,
    )
    require(r"CodePy late-game trace analysis", out, "late-game analysis marker")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    cohorts = {row["name"]: row for row in payload["cohorts"]}
    for name in ["top_overall", "top_final_depth", "top_selected", "top_survival_protected"]:
        if name not in cohorts or int(cohorts[name].get("count", 0)) <= 0:
            raise VerificationError(f"Late-game analysis missing cohort {name}: {payload}")
    overall_missing = cohorts["top_overall"]["stage_summary"]["first_missing_stage_counts"]
    protected = cohorts["top_survival_protected"]
    protected_blocks = {int(row["id"]) for row in protected.get("stable_blocks_90", [])}
    protected_order = protected["stage_summary"].get("order_issue_counts", {})
    if int(overall_missing.get("read_max", 0)) <= 0:
        raise VerificationError(f"Late-game analysis did not expose reward-exploit near-misses: {overall_missing}")
    if not {3, 7, 20, 25, 28}.issubset(protected_blocks):
        raise VerificationError(f"Survival-protected cohort lacks canonical argmax blocks: {protected_blocks}")
    if int(protected_order.get("missing_halt", 0)) <= 0:
        raise VerificationError(f"Late-game analysis did not expose terminal-control failure: {protected_order}")
    markdown = output_md.read_text(encoding="utf-8")
    for marker in ["CodePy Late-Game Trace Analysis", "top_survival_protected", "First missing stage distribution"]:
        if marker not in markdown:
            raise VerificationError(f"Late-game Markdown missing {marker!r}: {markdown[:800]}")
    pass_line(
        "code-tape-late-game",
        (
            f"overall_missing_read {overall_missing.get('read_max', 0)}, "
            f"protected_blocks {len(protected_blocks)}, missing_halt {protected_order.get('missing_halt', 0)}"
        ),
    )


def verify_code_tape_late_game_ranker(timeout: int) -> None:
    candidates_jsonl = ROOT / "artifacts" / "code_tape_prior" / "argmax_index4_survival_b64_ttl8_uncapped_w025_candidates.jsonl"
    search_json = ROOT / "artifacts" / "code_tape_prior" / "argmax_index4_survival_b64_ttl8_uncapped_w025.json"
    model_json = ARTIFACT_DIR / "code_tape_late_game_ranker.json"
    rerank_json = ARTIFACT_DIR / "code_tape_late_game_rerank.json"
    rerank_md = ARTIFACT_DIR / "code_tape_late_game_rerank.md"
    if not candidates_jsonl.exists() or not search_json.exists():
        raise VerificationError(
            "Late-game ranker requires survival artifacts; run make code-tape-survival-demo first."
        )
    train_out = run_cmd(
        [
            sys.executable,
            "train_code_tape_late_game_ranker.py",
            "--candidates-jsonl",
            str(candidates_jsonl),
            "--search-json",
            str(search_json),
            "--target",
            "argmax_index4",
            "--max-negatives",
            "400",
            "--output-json",
            str(model_json),
        ],
        timeout,
    )
    require(r"CodePy late-game structural ranker training", train_out, "late-game ranker training marker")
    model_payload = json.loads(model_json.read_text(encoding="utf-8"))
    model = model_payload["model"]
    if not model.get("weights") or not model.get("feature_names"):
        raise VerificationError(f"Late-game ranker model missing weights/feature_names: {model.keys()}")
    if len(model["weights"]) != len(model["feature_names"]):
        raise VerificationError("Late-game ranker weight/feature length mismatch.")
    separation = model_payload["contrastive_separation"]
    gap = float(separation.get("positive_vs_negative_gap", 0.0))
    if gap <= 0.0:
        raise VerificationError(f"Late-game ranker did not separate canonical from broken: gap {gap}")
    rank_out = run_cmd(
        [
            sys.executable,
            "rank_code_tape_late_game.py",
            "--model-json",
            str(model_json),
            "--candidates-jsonl",
            str(candidates_jsonl),
            "--target",
            "argmax_index4",
            "--top-n",
            "15",
            "--output-json",
            str(rerank_json),
            "--output-md",
            str(rerank_md),
        ],
        timeout,
    )
    require(r"CodePy late-game structural re-ranking proof", rank_out, "late-game rerank marker")
    rerank = json.loads(rerank_json.read_text(encoding="utf-8"))
    lift = rerank["lift"]
    if not lift.get("lift_achieved") or int(rerank["reference"]["rank"]) != 1:
        raise VerificationError(f"Late-game ranker did not lift the canonical program to rank 1: {lift}")
    if not lift.get("lift_over_reward_exploit"):
        raise VerificationError(f"Late-game ranker did not beat reward-exploit near-misses: {lift}")
    for flag in ["missing_halt", "swap_index_before_value_update", "loop_test_before_first_move"]:
        info = rerank["family_gaps"].get(flag, {})
        gap_value = info.get("gap_vs_canonical")
        if gap_value is None or float(gap_value) <= 0.0:
            raise VerificationError(f"Late-game ranker family gap not positive for {flag}: {info}")
    markdown = rerank_md.read_text(encoding="utf-8")
    for marker in ["CodePy Late-Game Structural Re-Ranking Proof", "Family gaps", "Top programs"]:
        if marker not in markdown:
            raise VerificationError(f"Late-game rerank Markdown missing {marker!r}: {markdown[:600]}")
    pass_line(
        "code-tape-late-game-ranker",
        (
            f"sep_gap {gap:.4f}, reference_rank {rerank['reference']['rank']}, "
            f"reward_exploit_best {lift.get('best_reward_exploit_structural_score')}"
        ),
    )


def verify_code_tape_late_game_guidance(timeout: int) -> None:
    candidates_jsonl = ROOT / "artifacts" / "code_tape_prior" / "argmax_index4_survival_b64_ttl8_uncapped_w025_candidates.jsonl"
    search_json = ROOT / "artifacts" / "code_tape_prior" / "argmax_index4_survival_b64_ttl8_uncapped_w025.json"
    model_json = ARTIFACT_DIR / "code_tape_late_game_guidance_ranker.json"
    output_json = ARTIFACT_DIR / "code_tape_late_game_guided_search.json"
    candidate_jsonl = ARTIFACT_DIR / "code_tape_late_game_guided_candidates.jsonl"
    if not candidates_jsonl.exists() or not search_json.exists():
        raise VerificationError(
            "Late-game guidance requires survival artifacts; run make code-tape-survival-demo first."
        )
    run_cmd(
        [
            sys.executable,
            "train_code_tape_late_game_ranker.py",
            "--candidates-jsonl",
            str(candidates_jsonl),
            "--search-json",
            str(search_json),
            "--target",
            "argmax_index4",
            "--max-negatives",
            "300",
            "--output-json",
            str(model_json),
        ],
        timeout,
    )
    program_length = 11
    min_remaining = 2
    out = run_cmd(
        [
            sys.executable,
            "search_code_tape_trajectory.py",
            "--target",
            "argmax_index4",
            "--program-length",
            str(program_length),
            "--max-steps",
            "64",
            "--cases",
            "32",
            "--beam-width",
            "12",
            "--signature-cases",
            "8",
            "--signature-steps",
            "18",
            "--late-game-ranker-json",
            str(model_json),
            "--late-game-weight",
            "0.25",
            "--late-game-lane-fraction",
            "0.34",
            "--late-game-min-remaining",
            str(min_remaining),
            "--candidate-jsonl",
            str(candidate_jsonl),
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"CodePy tape trajectory-guided search", out, "guided search marker")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    search = payload["search"]
    if int(search.get("late_game_gate_depth", -1)) != program_length - min_remaining:
        raise VerificationError(f"Late-game gate depth wrong: {search.get('late_game_gate_depth')}")
    if int(search.get("late_game_lane_selected_total", 0)) <= 0:
        raise VerificationError(f"Late-game lane never fired: {search}")
    gated_layers = [layer for layer in payload["layers"] if int(layer["depth"]) >= program_length - min_remaining]
    if not any(int(layer.get("late_game_lane_selected", 0)) > 0 for layer in gated_layers):
        raise VerificationError(f"Late-game lane did not select on gated depths: {gated_layers}")
    early_layers = [layer for layer in payload["layers"] if int(layer["depth"]) < program_length - min_remaining]
    if any(int(layer.get("late_game_lane_selected", 0)) > 0 for layer in early_layers):
        raise VerificationError("Late-game lane fired before the gate depth.")
    pass_line(
        "code-tape-late-game-guidance",
        (
            f"gate_depth {search['late_game_gate_depth']}, "
            f"late_game_lane_total {search['late_game_lane_selected_total']}, "
            f"best_mse {payload['best']['train_mse']:.4g}"
        ),
    )


def verify_selfplay_coevolution(timeout: int) -> None:
    output_json = ARTIFACT_DIR / "selfplay_verify.json"
    out = run_cmd(
        [
            sys.executable,
            "coevolve_ssm_selfplay.py",
            "--runners",
            "4",
            "--blockers",
            "4",
            "--runner-elites",
            "1",
            "--blocker-elites",
            "1",
            "--agents",
            "128",
            "--steps",
            "12",
            "--generations",
            "2",
            "--output-json",
            str(output_json),
        ],
        timeout,
    )
    require(r"TextPy/SoA SSM self-play coevolution prototype", out, "self-play marker")
    require(r"Self-play result", out, "self-play result marker")
    if not output_json.exists():
        raise VerificationError(f"Self-play JSON was not created: {output_json}")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    history = payload["history"]
    summary = payload["summary"]
    expected_history = protocol["generations"] + 1
    if len(history) != expected_history:
        raise VerificationError(f"Unexpected self-play history length: {len(history)} != {expected_history}")
    first_matrix = history[0]["payoff_matrix"]
    if len(first_matrix) != protocol["runners"] or len(first_matrix[0]) != protocol["blockers"]:
        raise VerificationError(f"Invalid self-play payoff matrix shape: {first_matrix}")
    if summary["evaluated_pair_agent_steps"] <= 0 or summary["throughput_pair_agent_steps_s"] <= 0:
        raise VerificationError(f"Invalid self-play throughput summary: {summary}")
    if "runner" not in payload["roles"] or "blocker" not in payload["roles"]:
        raise VerificationError(f"Self-play roles missing from JSON: {payload.get('roles')}")
    if "autonomous AGI" not in payload["claim_guardrail"]:
        raise VerificationError("Self-play JSON is missing the AGI guardrail.")
    pass_line(
        "selfplay-coevolution",
        (
            f"runner_best {summary['best_runner_reward']:.4f}, "
            f"blocker_best {summary['best_blocker_reward']:.4f}, "
            f"history {len(history)}"
        ),
    )


def verify_trajectory_export(timeout: int) -> None:
    output_dir = ARTIFACT_DIR / "trajectory_export"
    out = run_cmd(
        [
            sys.executable,
            "export_agent_trajectory.py",
            "--mode",
            "handcoded",
            "--agents",
            "16",
            "--steps",
            "12",
            "--output-dir",
            str(output_dir),
        ],
        timeout,
    )
    require(r"TextPy/SoA trajectory export", out, "trajectory export marker")

    csv_path = output_dir / "trajectory.csv"
    summary_path = output_dir / "summary.json"
    if not csv_path.exists():
        raise VerificationError(f"Trajectory CSV was not created: {csv_path}")
    if not summary_path.exists():
        raise VerificationError(f"Trajectory summary was not created: {summary_path}")

    row_count = sum(1 for _ in csv_path.open("r", encoding="utf-8"))
    expected_rows = 1 + (12 + 1) * 16
    if row_count != expected_rows:
        raise VerificationError(f"Unexpected trajectory row count: {row_count} != {expected_rows}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["final_goal_distance"] >= summary["initial_goal_distance"]:
        raise VerificationError(f"Trajectory export did not improve goal distance: {summary}")
    pass_line(
        "trajectory-export",
        f"rows {row_count}, goal {summary['initial_goal_distance']:.4f}->{summary['final_goal_distance']:.4f}",
    )


def verify_trajectory_viewer(_: int) -> None:
    html_path = ROOT / "trajectory_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Trajectory viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        '<canvas id="worldCanvas"',
        'id="csvFile"',
        'id="summaryFile"',
        "parseTrajectory",
        "drawAgents",
        "Trajectory<br>Workbench",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Trajectory viewer is missing expected hooks: {missing}")
    pass_line("trajectory-viewer", "static HTML hooks present")


def verify_text_memory_viewer(_: int) -> None:
    html_path = ROOT / "text_memory_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text memory viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        '<canvas id="memoryCanvas"',
        'id="probeFile"',
        'id="jsonPaste"',
        'id="timeline"',
        'id="topPredictions"',
        "validateProbe",
        "renderTimeline",
        "drawChart",
        "Memory<br>Probe",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text memory viewer is missing expected hooks: {missing}")
    pass_line("text-memory-viewer", "static HTML hooks present")


def verify_text_memory_suite_viewer(_: int) -> None:
    html_path = ROOT / "text_memory_suite_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text memory suite viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="suiteFile"',
        'id="metricSelect"',
        'id="matrixGrid"',
        'id="promptList"',
        'id="pairList"',
        "validateSuite",
        "renderMatrix",
        "metricValue",
        "Memory<br>Matrix",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text memory suite viewer is missing expected hooks: {missing}")
    pass_line("text-memory-suite-viewer", "static HTML hooks present")


def verify_text_release_leaderboard_viewer(_: int) -> None:
    html_path = ROOT / "text_release_leaderboard_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text release leaderboard viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="leaderboardFile"',
        'id="sortSelect"',
        'id="releaseList"',
        "validateLeaderboard",
        "renderBoard",
        "memory_mean_pair_overlap",
        "Release<br>Board",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release leaderboard viewer is missing expected hooks: {missing}")
    pass_line("text-release-viewer", "static HTML hooks present")


def verify_text_release_dashboard(_: int) -> None:
    html_path = ROOT / "text_release_dashboard.html"
    if not html_path.exists():
        raise VerificationError(f"Text release dashboard does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="dashboardJsonFile"',
        'id="dashboardLeaderboardFile"',
        'id="dashboardCurrentFile"',
        'id="dashboardSuiteFile"',
        'id="releaseDashboard"',
        'id="dashboardReleaseList"',
        'id="dashboardMatrix"',
        "validateDashboardInputs",
        "renderDashboard",
        "releaseMemorySummary",
        "Release<br>Deck",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release dashboard is missing expected hooks: {missing}")
    pass_line("text-release-dashboard", "static HTML hooks present")


def verify_text_release_dashboard_compare_viewer(_: int) -> None:
    html_path = ROOT / "text_release_dashboard_compare.html"
    if not html_path.exists():
        raise VerificationError(f"Text release dashboard compare viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="comparisonJsonFile"',
        'id="baselineDashboardFile"',
        'id="candidateDashboardFile"',
        'id="compareDashboard"',
        'id="deltaGrid"',
        'id="comparisonList"',
        "validateComparisonPayload",
        "compareDashboards",
        "renderCompare",
        "Release<br>Delta",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release dashboard compare viewer is missing expected hooks: {missing}")
    pass_line("text-release-dashboard-compare-viewer", "static HTML hooks present")


def verify_text_release_evidence_compare_viewer(_: int) -> None:
    html_path = ROOT / "text_release_evidence_compare.html"
    if not html_path.exists():
        raise VerificationError(f"Text release evidence compare viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="evidenceComparisonFile"',
        'id="baselineManifestFile"',
        'id="candidateManifestFile"',
        'id="evidenceDashboard"',
        'id="evidenceDeltaGrid"',
        'id="proofCheckList"',
        'id="evidenceFileMatrix"',
        "validateEvidenceComparison",
        "compareEvidenceManifests",
        "renderEvidenceCompare",
        "Release<br>Proof",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release evidence compare viewer is missing expected hooks: {missing}")
    pass_line("text-release-evidence-compare-viewer", "static HTML hooks present")


def verify_text_release_evidence_ledger_viewer(_: int) -> None:
    html_path = ROOT / "text_release_evidence_ledger_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text release evidence ledger viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="ledgerFile"',
        'id="reportFile"',
        'id="jsonPaste"',
        'id="chainList"',
        'id="artifactRows"',
        "validateLedger",
        "renderLedger",
        "artifactRecords",
        "Evidence<br>Ledger",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release evidence ledger viewer is missing expected hooks: {missing}")
    pass_line("text-release-evidence-ledger-viewer", "static HTML hooks present")


def verify_text_release_evidence_ledger_compare_viewer(_: int) -> None:
    html_path = ROOT / "text_release_evidence_ledger_compare.html"
    if not html_path.exists():
        raise VerificationError(f"Text release evidence ledger compare viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="ledgerComparisonFile"',
        'id="jsonPaste"',
        'id="entryDeltaRows"',
        'id="shaDeltaGrid"',
        'id="failureList"',
        "validateLedgerComparison",
        "renderLedgerComparison",
        "renderEntryDeltaRows",
        "Ledger<br>Compare",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release evidence ledger compare viewer is missing expected hooks: {missing}")
    pass_line("text-release-evidence-ledger-compare-viewer", "static HTML hooks present")


def verify_text_release_evidence_audit_viewer(_: int) -> None:
    html_path = ROOT / "text_release_evidence_audit_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text release evidence audit viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="auditFile"',
        'id="jsonPaste"',
        'id="auditDashboard"',
        'id="auditCheckGrid"',
        'id="auditSourceList"',
        'id="auditFailureList"',
        "validateAudit",
        "renderAudit",
        "text_release_evidence_audit",
        "Evidence<br>Audit",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release evidence audit viewer is missing expected hooks: {missing}")
    pass_line("text-release-evidence-audit-viewer", "static HTML hooks present")


def verify_text_release_evidence_audit_validation_viewer(_: int) -> None:
    html_path = ROOT / "text_release_evidence_audit_validation_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text release evidence audit validation viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="auditValidationFile"',
        'id="jsonPaste"',
        'id="validationDashboard"',
        'id="validationMetricGrid"',
        'id="validationDetailGrid"',
        'id="validationErrorList"',
        "validateAuditValidation",
        "renderAuditValidation",
        "text_release_evidence_audit_validation.json",
        "Audit<br>Validate",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release evidence audit validation viewer is missing expected hooks: {missing}")
    pass_line("text-release-evidence-audit-validation-viewer", "static HTML hooks present")


def verify_text_release_evidence_audit_validation_compare_viewer(_: int) -> None:
    html_path = ROOT / "text_release_evidence_audit_validation_compare.html"
    if not html_path.exists():
        raise VerificationError(f"Text release evidence audit validation compare viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="auditValidationComparisonFile"',
        'id="jsonPaste"',
        'id="auditValidationComparisonDashboard"',
        'id="auditValidationCards"',
        'id="auditValidationDeltaGrid"',
        'id="auditValidationDriftRows"',
        'id="auditValidationFailureList"',
        "validateAuditValidationComparison",
        "renderAuditValidationComparison",
        "text_release_evidence_audit_validation_comparison.json",
        "Validation<br>Compare",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release evidence audit validation compare viewer is missing expected hooks: {missing}")
    pass_line("text-release-evidence-audit-validation-compare-viewer", "static HTML hooks present")


def verify_text_release_evidence_audit_compare_viewer(_: int) -> None:
    html_path = ROOT / "text_release_evidence_audit_compare.html"
    if not html_path.exists():
        raise VerificationError(f"Text release evidence audit compare viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="auditComparisonFile"',
        'id="auditComparisonDashboard"',
        'id="auditDeltaGrid"',
        'id="auditDriftRows"',
        'id="auditCheckDeltaRows"',
        'id="auditSourceDeltaGrid"',
        'id="auditFailureList"',
        "validateAuditComparison",
        "renderAuditComparison",
        "text_release_evidence_audit_comparison.json",
        "Audit<br>Compare",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release evidence audit compare viewer is missing expected hooks: {missing}")
    pass_line("text-release-evidence-audit-compare-viewer", "static HTML hooks present")


def verify_text_release_history_viewer(_: int) -> None:
    html_path = ROOT / "text_release_history_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text release history viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="historyFile"',
        'id="historyDashboard"',
        'id="historyTimeline"',
        'id="bestList"',
        "validateHistory",
        "renderHistory",
        "text_release_history",
        "Release<br>History",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release history viewer is missing expected hooks: {missing}")
    pass_line("text-release-history-viewer", "static HTML hooks present")


def verify_text_release_history_analysis_viewer(_: int) -> None:
    html_path = ROOT / "text_release_history_analysis_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text release history analysis viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="analysisFile"',
        'id="analysisDashboard"',
        'id="verdictGrid"',
        'id="detailList"',
        "validateAnalysis",
        "renderAnalysis",
        "text_release_history_analysis.json",
        "Release<br>Verdict",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release history analysis viewer is missing expected hooks: {missing}")
    pass_line("text-release-history-analysis-viewer", "static HTML hooks present")


def verify_text_release_gates_viewer(_: int) -> None:
    html_path = ROOT / "text_release_gates_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text release gate summary viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="gatesFile"',
        'id="gateDashboard"',
        'id="gateGrid"',
        'id="failureList"',
        "validateGateSummary",
        "renderGateSummary",
        "text_release_gates.json",
        "Release<br>Gates",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release gate summary viewer is missing expected hooks: {missing}")
    pass_line("text-release-gates-viewer", "static HTML hooks present")


def verify_text_release_pipeline_validation_viewer(_: int) -> None:
    html_path = ROOT / "text_release_pipeline_validation_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text release pipeline validation viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="pipelineValidationFile"',
        'id="pipelineValidationDashboard"',
        'id="pipelineValidationMetricGrid"',
        'id="pipelineValidationErrorList"',
        "validatePipelineValidation",
        "renderPipelineValidation",
        "text_release_pipeline_validation.json",
        "Pipeline<br>Validate",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release pipeline validation viewer is missing expected hooks: {missing}")
    pass_line("text-release-pipeline-validation-viewer", "static HTML hooks present")


def verify_text_release_pipeline_validation_compare_viewer(_: int) -> None:
    html_path = ROOT / "text_release_pipeline_validation_compare.html"
    if not html_path.exists():
        raise VerificationError(f"Text release pipeline validation compare viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="pipelineValidationComparisonFile"',
        'id="pipelineValidationComparisonDashboard"',
        'id="pipelineValidationDeltaGrid"',
        'id="pipelineValidationDriftRows"',
        'id="pipelineValidationFailureList"',
        "validatePipelineValidationComparison",
        "renderPipelineValidationComparison",
        "text_release_pipeline_validation_comparison.json",
        "Pipeline<br>Compare",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release pipeline validation compare viewer is missing expected hooks: {missing}")
    pass_line("text-release-pipeline-validation-compare-viewer", "static HTML hooks present")


def verify_text_release_pipeline_compare_viewer(_: int) -> None:
    html_path = ROOT / "text_release_pipeline_compare.html"
    if not html_path.exists():
        raise VerificationError(f"Text release pipeline compare viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="pipelineComparisonFile"',
        'id="pipelineComparisonDashboard"',
        'id="pipelineComparisonDeltaGrid"',
        'id="pipelineComparisonDriftRows"',
        'id="pipelineComparisonFailureList"',
        "validatePipelineComparison",
        "renderPipelineComparison",
        "text_release_pipeline_comparison.json",
        "Pipeline<br>Run Compare",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release pipeline compare viewer is missing expected hooks: {missing}")
    pass_line("text-release-pipeline-compare-viewer", "static HTML hooks present")


def verify_text_release_pipeline_comparison_validation_viewer(_: int) -> None:
    html_path = ROOT / "text_release_pipeline_comparison_validation_viewer.html"
    if not html_path.exists():
        raise VerificationError(f"Text release pipeline comparison validation viewer does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="pipelineComparisonValidationFile"',
        'id="pipelineComparisonValidationDashboard"',
        'id="pipelineComparisonValidationMetricGrid"',
        'id="pipelineComparisonValidationErrorList"',
        "validatePipelineComparisonValidation",
        "renderPipelineComparisonValidation",
        "text_release_pipeline_comparison_validation.json",
        "Pipeline<br>Proof Check",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(
            f"Text release pipeline comparison validation viewer is missing expected hooks: {missing}"
        )
    pass_line("text-release-pipeline-comparison-validation-viewer", "static HTML hooks present")


def verify_text_release_pipeline_comparison_validation_compare_viewer(_: int) -> None:
    html_path = ROOT / "text_release_pipeline_comparison_validation_compare.html"
    if not html_path.exists():
        raise VerificationError(
            f"Text release pipeline comparison validation compare viewer does not exist: {html_path}"
        )
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="pipelineComparisonValidationComparisonFile"',
        'id="pipelineComparisonValidationComparisonDashboard"',
        'id="pipelineComparisonValidationDeltaGrid"',
        'id="pipelineComparisonValidationDriftRows"',
        'id="pipelineComparisonValidationFailureList"',
        "validatePipelineComparisonValidationComparison",
        "renderPipelineComparisonValidationComparison",
        "text_release_pipeline_comparison_validation_comparison.json",
        "Pipeline<br>Proof Compare",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(
            f"Text release pipeline comparison validation compare viewer is missing expected hooks: {missing}"
        )
    pass_line("text-release-pipeline-comparison-validation-compare-viewer", "static HTML hooks present")


def verify_text_release_pipeline_proof_cockpit_viewer(_: int) -> None:
    html_path = ROOT / "text_release_pipeline_proof_cockpit.html"
    if not html_path.exists():
        raise VerificationError(f"Text release pipeline proof cockpit does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    required = [
        'id="proofCockpitFile"',
        'id="proofCockpitDashboard"',
        'id="proofCockpitChain"',
        'id="proofCockpitMetricGrid"',
        'id="proofCockpitCheckRows"',
        'id="proofCockpitFailureList"',
        "validateProofCockpit",
        "renderProofCockpit",
        "text_release_pipeline_comparison_validation_comparison.json",
        "Pipeline<br>Proof Cockpit",
    ]
    missing = [item for item in required if item not in html]
    if missing:
        raise VerificationError(f"Text release pipeline proof cockpit is missing expected hooks: {missing}")
    pass_line("text-release-pipeline-proof-cockpit-viewer", "static HTML hooks present")


def run(args: argparse.Namespace) -> None:
    checks = [
        verify_compile,
        verify_agent_runtime,
        verify_text_checkpoint,
        verify_text_checkpoint_runner,
        verify_text_corpus_profiler,
        verify_text_ingestion_and_bpe,
        verify_text_checkpoint_comparison,
        verify_text_architecture_comparison,
        verify_text_architecture_multiseed,
        verify_long_context_recall,
        verify_assignment_recall,
        verify_multi_assignment_recall,
        verify_multi_query_assignment_recall,
        verify_long_context_recall_multiseed_runner,
        verify_generated_python_code_recall,
        verify_real_python_code_recall,
        verify_text_experiment_pipeline,
        verify_text_experiment_leaderboard,
        verify_text_experiment_sweep,
        verify_text_checkpoint_promotion,
        verify_text_checkpoint_inspector,
        verify_text_manifest_runner,
        verify_text_package_benchmark,
        verify_text_package_validator,
        verify_text_package_bundle,
        verify_text_package_unbundle,
        verify_text_package_comparison,
        verify_text_package_registry,
        verify_text_release_leaderboard,
        verify_text_release_selection,
        verify_text_release_runner,
        verify_text_memory_probe,
        verify_text_memory_comparison,
        verify_text_memory_suite,
        verify_text_memory_suite_validator,
        verify_text_release_pipeline,
        verify_text_release_pipeline_validation,
        verify_text_release_pipeline_validation_comparison,
        verify_text_release_pipeline_comparison,
        verify_text_release_pipeline_comparison_validation,
        verify_text_release_pipeline_comparison_validation_comparison,
        verify_text_release_comparison,
        verify_text_release_dashboard_json,
        verify_text_release_dashboard_validator,
        verify_text_release_dashboard_comparison,
        verify_text_release_history_json,
        verify_text_release_history_validator,
        verify_text_release_history_analysis,
        verify_text_release_gate_summary,
        verify_text_release_report,
        verify_text_release_evidence_bundle,
        verify_text_release_evidence_validator,
        verify_text_release_evidence_unbundle,
        verify_text_release_evidence_comparison,
        verify_text_release_evidence_ledger,
        verify_text_release_evidence_ledger_validator,
        verify_text_release_evidence_ledger_tamper,
        verify_text_release_evidence_ledger_replay,
        verify_text_release_evidence_ledger_comparison,
        verify_text_release_evidence_audit,
        verify_text_release_evidence_audit_validator,
        verify_text_release_evidence_audit_validation_comparison,
        verify_text_release_evidence_audit_comparison,
        verify_evolution_replay,
        verify_formula_evolution,
        verify_hypothesis_loop,
        verify_code_block_evolution,
        verify_code_tape_prior_pipeline,
        verify_code_tape_argmax_geometry,
        verify_code_tape_trajectory_search,
        verify_code_tape_trace_prior,
        verify_code_tape_trace_guidance_comparison,
        verify_code_tape_prefix_value_rollout,
        verify_code_tape_survival_policy,
        verify_code_tape_late_game_analysis,
        verify_code_tape_late_game_ranker,
        verify_code_tape_late_game_guidance,
        verify_selfplay_coevolution,
        verify_trajectory_export,
        verify_trajectory_viewer,
        verify_text_memory_viewer,
        verify_text_memory_suite_viewer,
        verify_text_release_leaderboard_viewer,
        verify_text_release_dashboard,
        verify_text_release_dashboard_compare_viewer,
        verify_text_release_evidence_compare_viewer,
        verify_text_release_evidence_ledger_viewer,
        verify_text_release_evidence_ledger_compare_viewer,
        verify_text_release_evidence_audit_viewer,
        verify_text_release_evidence_audit_validation_viewer,
        verify_text_release_evidence_audit_validation_compare_viewer,
        verify_text_release_evidence_audit_compare_viewer,
        verify_text_release_history_viewer,
        verify_text_release_history_analysis_viewer,
        verify_text_release_gates_viewer,
        verify_text_release_pipeline_validation_viewer,
        verify_text_release_pipeline_validation_compare_viewer,
        verify_text_release_pipeline_compare_viewer,
        verify_text_release_pipeline_comparison_validation_viewer,
        verify_text_release_pipeline_comparison_validation_compare_viewer,
        verify_text_release_pipeline_proof_cockpit_viewer,
    ]
    for check in checks:
        check(args.timeout)
    print("SYSTEM OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the SoA/SSM prototype stack.")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except VerificationError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
