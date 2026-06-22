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
        "soa_ssm_agents.py",
        "text_tokenization.py",
        "train_ssm_text.py",
        "ingest_text_data.py",
        "materialize_text_corpus.py",
        "plan_text_sweep.py",
        "run_text_sweep_plan.py",
        "profile_text_corpus.py",
        "run_text_checkpoint.py",
        "compare_text_checkpoints.py",
        "compare_text_architectures.py",
        "run_text_architecture_multiseed.py",
        "compare_long_context_recall.py",
        "run_long_context_recall_multiseed.py",
        "run_text_experiment.py",
        "list_text_experiments.py",
        "sweep_text_experiments.py",
        "promote_text_checkpoint.py",
        "inspect_text_checkpoint.py",
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
    ]
    for check in checks:
        check(args.timeout)
    print("SYSTEM OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the KV-Memory SSM benchmark stack.")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except VerificationError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
