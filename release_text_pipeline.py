#!/usr/bin/env python3
"""Run the full TextPy/SoA text release pipeline from leaderboard to active smoke run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MEMORY_PROMPTS = [
    "memory is a river",
    "memory is another world",
    "the future is guessed",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def copy_artifact(source: Path, target: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Missing release evidence source artifact: {source}")
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def run_cmd(args: list[str], timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    start = time.perf_counter()
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
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {proc.returncode}: {' '.join(args)}\n{proc.stdout}"
        )
    return {
        "command": args,
        "elapsed_s": elapsed,
        "stdout_tail": proc.stdout.splitlines()[-20:],
    }


def stage(report: dict[str, Any], name: str, cmd: list[str], timeout: int) -> dict[str, Any]:
    print(f"[{name}] {' '.join(cmd)}")
    result = run_cmd(cmd, timeout)
    report["stages"][name] = result
    return result


def maybe_clean(args: argparse.Namespace) -> None:
    if not args.clean:
        return
    memory_suite_dir, memory_suite_json, memory_prompts_csv, memory_pairs_csv, memory_validation_json = memory_paths(args)
    evidence = evidence_paths(args) if args.include_evidence_audit else {}
    paths = [
        Path(args.promote_dir),
        Path(args.restore_dir),
        Path(args.registry),
        Path(args.release_leaderboard_json),
        Path(args.release_leaderboard_csv),
        Path(args.current_release_json),
        Path(args.run_json),
        Path(args.dashboard_json),
        memory_suite_dir,
        memory_suite_json,
        memory_prompts_csv,
        memory_pairs_csv,
        memory_validation_json,
    ]
    paths.extend(path for key, path in evidence.items() if key != "base_dir")
    for path in paths:
        remove_path(path)


def memory_prompts(args: argparse.Namespace) -> list[str]:
    prompts = [prompt.strip() for prompt in (args.memory_prompt or []) if prompt.strip()]
    return prompts or list(DEFAULT_MEMORY_PROMPTS)


def memory_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    suite_dir = Path(args.memory_suite_dir)
    suite_json = Path(args.memory_suite_json) if args.memory_suite_json else suite_dir / "memory_suite.json"
    prompts_csv = Path(args.memory_prompts_csv) if args.memory_prompts_csv else suite_dir / "prompts.csv"
    pairs_csv = Path(args.memory_pairs_csv) if args.memory_pairs_csv else suite_dir / "pairs.csv"
    validation_json = (
        Path(args.memory_validation_json) if args.memory_validation_json else suite_dir / "validation.json"
    )
    return suite_dir, suite_json, prompts_csv, pairs_csv, validation_json


def output_base_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_json).parent


def evidence_path_arg(args: argparse.Namespace, attr: str, default_name: str) -> Path:
    value = getattr(args, attr)
    return Path(value) if value else output_base_dir(args) / default_name


def evidence_default(args: argparse.Namespace, suffix: str) -> str:
    return f"{Path(args.output_json).stem}_{suffix}"


def evidence_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "base_dir": output_base_dir(args),
        "dashboard_validation": evidence_path_arg(
            args,
            "release_dashboard_validation_json",
            evidence_default(args, "dashboard_validation.json"),
        ),
        "dashboard_comparison": evidence_path_arg(
            args,
            "release_dashboard_comparison_json",
            evidence_default(args, "dashboard_comparison.json"),
        ),
        "history": evidence_path_arg(args, "release_history_json", evidence_default(args, "history.json")),
        "history_validation": evidence_path_arg(
            args,
            "release_history_validation_json",
            evidence_default(args, "history_validation.json"),
        ),
        "history_analysis": evidence_path_arg(
            args,
            "release_history_analysis_json",
            evidence_default(args, "history_analysis.json"),
        ),
        "release_gates": evidence_path_arg(args, "release_gates_json", evidence_default(args, "gates.json")),
        "release_report_md": evidence_path_arg(args, "release_report_md", evidence_default(args, "report.md")),
        "evidence_archive": evidence_path_arg(args, "evidence_archive", evidence_default(args, "evidence.tar.gz")),
        "evidence_manifest": evidence_path_arg(
            args,
            "evidence_manifest_json",
            evidence_default(args, "evidence_manifest.json"),
        ),
        "evidence_validation": evidence_path_arg(
            args,
            "evidence_validation_json",
            evidence_default(args, "evidence_validation.json"),
        ),
        "evidence_restore_dir": Path(args.evidence_restore_dir)
        if args.evidence_restore_dir
        else output_base_dir(args) / evidence_default(args, "restored_evidence"),
        "evidence_restore": evidence_path_arg(
            args,
            "evidence_restore_json",
            evidence_default(args, "evidence_restore.json"),
        ),
        "evidence_comparison": evidence_path_arg(
            args,
            "evidence_comparison_json",
            evidence_default(args, "evidence_comparison.json"),
        ),
        "evidence_ledger": evidence_path_arg(
            args,
            "evidence_ledger_json",
            evidence_default(args, "evidence_ledger.json"),
        ),
        "evidence_ledger_store_dir": Path(args.evidence_ledger_store_dir)
        if args.evidence_ledger_store_dir
        else output_base_dir(args) / evidence_default(args, "evidence_store"),
        "evidence_ledger_validation": evidence_path_arg(
            args,
            "evidence_ledger_validation_json",
            evidence_default(args, "evidence_ledger_validation.json"),
        ),
        "evidence_ledger_tampered": evidence_path_arg(
            args,
            "evidence_ledger_tampered_json",
            evidence_default(args, "evidence_ledger_tampered.json"),
        ),
        "evidence_ledger_tamper": evidence_path_arg(
            args,
            "evidence_ledger_tamper_json",
            evidence_default(args, "evidence_ledger_tamper.json"),
        ),
        "evidence_ledger_replay": evidence_path_arg(
            args,
            "evidence_ledger_replay_json",
            evidence_default(args, "evidence_ledger_replay.json"),
        ),
        "evidence_ledger_replay_store_dir": Path(args.evidence_ledger_replay_store_dir)
        if args.evidence_ledger_replay_store_dir
        else output_base_dir(args) / evidence_default(args, "evidence_ledger_replay_store"),
        "evidence_ledger_replay_work_dir": Path(args.evidence_ledger_replay_work_dir)
        if args.evidence_ledger_replay_work_dir
        else output_base_dir(args) / evidence_default(args, "evidence_ledger_replay_work"),
        "evidence_ledger_replay_report": evidence_path_arg(
            args,
            "evidence_ledger_replay_report_json",
            evidence_default(args, "evidence_ledger_replay_report.json"),
        ),
        "evidence_ledger_comparison": evidence_path_arg(
            args,
            "evidence_ledger_comparison_json",
            evidence_default(args, "evidence_ledger_comparison.json"),
        ),
        "evidence_ledger_growth_comparison": evidence_path_arg(
            args,
            "evidence_ledger_growth_comparison_json",
            evidence_default(args, "evidence_ledger_growth_comparison.json"),
        ),
        "evidence_audit": evidence_path_arg(args, "evidence_audit_json", evidence_default(args, "evidence_audit.json")),
        "evidence_audit_md": evidence_path_arg(args, "evidence_audit_md", evidence_default(args, "evidence_audit.md")),
        "evidence_audit_validation": evidence_path_arg(
            args,
            "evidence_audit_validation_json",
            evidence_default(args, "evidence_audit_validation.json"),
        ),
        "evidence_audit_validation_comparison": evidence_path_arg(
            args,
            "evidence_audit_validation_comparison_json",
            evidence_default(args, "evidence_audit_validation_comparison.json"),
        ),
        "evidence_audit_comparison": evidence_path_arg(
            args,
            "evidence_audit_comparison_json",
            evidence_default(args, "evidence_audit_comparison.json"),
        ),
    }


def memory_threshold_args(args: argparse.Namespace) -> list[str]:
    thresholds: list[str] = []
    if args.memory_min_mean_pair_overlap is not None:
        thresholds.extend(["--min-mean-pair-overlap", str(args.memory_min_mean_pair_overlap)])
    if args.memory_min_pair_overlap is not None:
        thresholds.extend(["--min-pair-overlap", str(args.memory_min_pair_overlap)])
    if args.memory_max_pair_state_norm_delta is not None:
        thresholds.extend(["--max-pair-state-norm-delta", str(args.memory_max_pair_state_norm_delta)])
    if args.memory_max_pair_state_l2_distance is not None:
        thresholds.extend(["--max-pair-state-l2-distance", str(args.memory_max_pair_state_l2_distance)])
    if args.memory_min_pair_state_cosine_similarity is not None:
        thresholds.extend(["--min-pair-state-cosine-similarity", str(args.memory_min_pair_state_cosine_similarity)])
    return thresholds


def verify_pipeline_outputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.promote_dir) / "manifest.json"
    model_card = Path(args.promote_dir) / "model_card.json"
    benchmark = Path(args.promote_dir) / "benchmark.json"
    validation = Path(args.promote_dir) / "validation.json"
    bundle = Path(args.promote_dir) / "bundle_manifest.json"
    archive = Path(args.promote_dir) / "text_package.tar.gz"
    comparison = Path(args.promote_dir) / "package_comparison.json"
    registry = Path(args.registry)
    leaderboard = Path(args.release_leaderboard_json)
    current = Path(args.current_release_json)
    run_json = Path(args.run_json)
    _, memory_suite_json, memory_prompts_csv, memory_pairs_csv, memory_validation_json = memory_paths(args)
    prompts = memory_prompts(args)

    required = [
        manifest,
        model_card,
        benchmark,
        validation,
        bundle,
        archive,
        comparison,
        registry,
        leaderboard,
        current,
        run_json,
        memory_suite_json,
        memory_prompts_csv,
        memory_pairs_csv,
        memory_validation_json,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Release pipeline missed artifacts: {missing}")

    manifest_payload = load_json(manifest)
    model_card_payload = load_json(model_card)
    benchmark_payload = load_json(benchmark)
    validation_payload = load_json(validation)
    bundle_payload = load_json(bundle)
    comparison_payload = load_json(comparison)
    registry_payload = load_json(registry)
    leaderboard_payload = load_json(leaderboard)
    current_payload = load_json(current)
    run_payload = load_json(run_json)
    memory_suite_payload = load_json(memory_suite_json)
    memory_validation_payload = load_json(memory_validation_json)

    if validation_payload["valid"] is not True:
        raise SystemExit(f"Package validation failed: {validation_payload}")
    if int(model_card_payload["param_count"]) <= 0:
        raise SystemExit(f"Invalid model card param_count: {model_card_payload}")
    if float(benchmark_payload["loss"]) <= 0:
        raise SystemExit(f"Invalid benchmark loss: {benchmark_payload}")
    if float(benchmark_payload["best_throughput_tokens_s"]) <= 0:
        raise SystemExit(f"Invalid benchmark throughput: {benchmark_payload}")
    if int(bundle_payload["archive_bytes"]) <= 0:
        raise SystemExit(f"Invalid bundle manifest: {bundle_payload}")
    if not comparison_payload["comparison"]["same_checkpoint_sha256"]:
        raise SystemExit(f"Restored package checkpoint changed: {comparison_payload['comparison']}")
    if int(registry_payload["release_count"]) < 1:
        raise SystemExit(f"Registry did not record a release: {registry_payload}")
    if int(leaderboard_payload["shown_count"]) < 1:
        raise SystemExit(f"Release leaderboard is empty: {leaderboard_payload}")
    if int(current_payload["rank"]) != 1 or current_payload["validation"]["valid"] is not True:
        raise SystemExit(f"Active release pointer is invalid: {current_payload}")
    if run_payload["release"] != str(current):
        raise SystemExit(f"Run JSON did not record active release path: {run_payload}")
    if not run_payload.get("sample"):
        raise SystemExit(f"Run JSON has no sample: {run_payload}")
    if memory_suite_payload["manifest"] != str(manifest):
        raise SystemExit(f"Memory suite did not record release manifest path: {memory_suite_payload}")
    if memory_validation_payload["valid"] is not True:
        raise SystemExit(f"Memory suite validation failed: {memory_validation_payload}")
    if int(memory_validation_payload["prompt_count"]) != len(prompts):
        raise SystemExit(f"Memory suite prompt count mismatch: {memory_validation_payload}")
    if memory_validation_payload["has_state_vectors"] is not True:
        raise SystemExit(f"Memory suite validation did not include vector geometry: {memory_validation_payload}")
    release_memory = current_payload.get("memory") or {}
    if release_memory.get("validation") != str(memory_validation_json):
        raise SystemExit(f"Active release pointer did not capture memory validation: {current_payload}")
    if release_memory.get("mean_pair_overlap") != memory_validation_payload["mean_pair_overlap"]:
        raise SystemExit(f"Active release memory summary does not match validation: {current_payload}")

    return {
        "manifest": str(manifest),
        "checkpoint": manifest_payload["promoted_checkpoint"],
        "checkpoint_sha256": validation_payload["checkpoint_sha256"],
        "archive": str(archive),
        "archive_sha256": bundle_payload["archive_sha256"],
        "registry": str(registry),
        "current_release": str(current),
        "run_json": str(run_json),
        "release_count": registry_payload["release_count"],
        "selected_rank": current_payload["rank"],
        "loss": run_payload["loss"],
        "accuracy": run_payload["accuracy"],
        "best_throughput_tokens_s": benchmark_payload["best_throughput_tokens_s"],
        "sample": run_payload["sample"],
        "memory_suite": str(memory_suite_json),
        "memory_suite_validation": str(memory_validation_json),
        "memory_prompt_count": memory_validation_payload["prompt_count"],
        "memory_pair_count": memory_validation_payload["pair_count"],
        "memory_mean_pair_overlap": memory_validation_payload["mean_pair_overlap"],
        "memory_min_pair_overlap": memory_validation_payload["min_pair_overlap"],
        "memory_max_pair_state_norm_delta": memory_validation_payload["max_pair_state_norm_delta"],
        "memory_max_pair_state_l2_distance": memory_validation_payload["max_pair_state_l2_distance"],
        "memory_min_pair_state_cosine_similarity": memory_validation_payload[
            "min_pair_state_cosine_similarity"
        ],
    }


def run_evidence_audit_chain(
    args: argparse.Namespace,
    report: dict[str, Any],
    summary: dict[str, Any],
    *,
    dashboard_json: Path,
    memory_suite_json: Path,
    memory_prompts_csv: Path,
    memory_pairs_csv: Path,
    memory_validation_json: Path,
) -> dict[str, Any]:
    paths = evidence_paths(args)
    base_dir = paths["base_dir"]
    if base_dir == Path("."):
        base_dir = Path.cwd()
    source_files = {
        "text_release_gates.json": paths["release_gates"],
        "text_release_report.md": paths["release_report_md"],
        "text_release_dashboard.json": dashboard_json,
        "text_release_dashboard_validation.json": paths["dashboard_validation"],
        "text_release_dashboard_comparison.json": paths["dashboard_comparison"],
        "text_release_history.json": paths["history"],
        "text_release_history_validation.json": paths["history_validation"],
        "text_release_history_analysis.json": paths["history_analysis"],
        "text_release_leaderboard.json": Path(args.release_leaderboard_json),
        "text_release_leaderboard.csv": Path(args.release_leaderboard_csv),
        "current_text_release.json": Path(args.current_release_json),
        "run_current_text_release.json": Path(args.run_json),
        "text_memory_suite/memory_suite.json": memory_suite_json,
        "text_memory_suite/validation.json": memory_validation_json,
        "text_memory_suite/prompts.csv": memory_prompts_csv,
        "text_memory_suite/pairs.csv": memory_pairs_csv,
    }

    stage(
        report,
        "release_dashboard_validation",
        [
            sys.executable,
            "validate_text_release_dashboard.py",
            "--dashboard",
            str(dashboard_json),
            "--require-artifacts",
            "--expected-prompts",
            str(len(memory_prompts(args))),
            "--expected-pairs",
            str(max(0, len(memory_prompts(args)) * (len(memory_prompts(args)) - 1) // 2)),
            "--min-mean-pair-overlap",
            str(args.memory_min_mean_pair_overlap or 0.0),
            "--output-json",
            str(paths["dashboard_validation"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_dashboard_comparison",
        [
            sys.executable,
            "compare_text_release_dashboards.py",
            "--baseline-dashboard",
            str(dashboard_json),
            "--candidate-dashboard",
            str(dashboard_json),
            "--require-artifacts",
            "--expected-prompts",
            str(len(memory_prompts(args))),
            "--expected-pairs",
            str(max(0, len(memory_prompts(args)) * (len(memory_prompts(args)) - 1) // 2)),
            "--min-mean-pair-overlap",
            str(args.memory_min_mean_pair_overlap or 0.0),
            "--fail-on-loss-regression",
            "--fail-on-throughput-regression",
            "--fail-on-accuracy-regression",
            "--fail-on-audit-regression",
            "--fail-on-memory-overlap-regression",
            "--fail-on-memory-cosine-regression",
            "--fail-on-memory-l2-regression",
            "--fail-on-memory-norm-regression",
            "--output-json",
            str(paths["dashboard_comparison"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_history",
        [
            sys.executable,
            "build_text_release_history.py",
            "--dashboard",
            str(dashboard_json),
            "--require-artifacts",
            "--expected-prompts",
            str(len(memory_prompts(args))),
            "--expected-pairs",
            str(max(0, len(memory_prompts(args)) * (len(memory_prompts(args)) - 1) // 2)),
            "--min-mean-pair-overlap",
            str(args.memory_min_mean_pair_overlap or 0.0),
            "--output-json",
            str(paths["history"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_history_validation",
        [
            sys.executable,
            "validate_text_release_history.py",
            "--history",
            str(paths["history"]),
            "--require-artifacts",
            "--expected-releases",
            "1",
            "--output-json",
            str(paths["history_validation"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_history_analysis",
        [
            sys.executable,
            "analyze_text_release_history.py",
            "--history",
            str(paths["history"]),
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
            str(paths["history_analysis"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_gates",
        [
            sys.executable,
            "summarize_text_release_gates.py",
            "--dashboard-validation",
            str(paths["dashboard_validation"]),
            "--dashboard-comparison",
            str(paths["dashboard_comparison"]),
            "--history-validation",
            str(paths["history_validation"]),
            "--history-analysis",
            str(paths["history_analysis"]),
            "--output-json",
            str(paths["release_gates"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_report",
        [
            sys.executable,
            "generate_text_release_report.py",
            "--gates",
            str(paths["release_gates"]),
            "--dashboard",
            str(dashboard_json),
            "--history-analysis",
            str(paths["history_analysis"]),
            "--output-md",
            str(paths["release_report_md"]),
        ],
        args.timeout,
    )

    base_dir.mkdir(parents=True, exist_ok=True)
    for relative, source in source_files.items():
        copy_artifact(source, base_dir / relative)

    stage(
        report,
        "release_evidence_bundle",
        [
            sys.executable,
            "bundle_text_release_evidence.py",
            "--base-dir",
            str(base_dir),
            "--root-name",
            args.evidence_root_name,
            "--gates",
            str(paths["release_gates"]),
            "--report-md",
            str(paths["release_report_md"]),
            "--dashboard",
            str(dashboard_json),
            "--dashboard-validation",
            str(paths["dashboard_validation"]),
            "--dashboard-comparison",
            str(paths["dashboard_comparison"]),
            "--history",
            str(paths["history"]),
            "--history-validation",
            str(paths["history_validation"]),
            "--history-analysis",
            str(paths["history_analysis"]),
            "--leaderboard-json",
            str(Path(args.release_leaderboard_json)),
            "--leaderboard-csv",
            str(Path(args.release_leaderboard_csv)),
            "--current-release",
            str(Path(args.current_release_json)),
            "--run-json",
            str(Path(args.run_json)),
            "--memory-suite",
            str(memory_suite_json),
            "--memory-validation",
            str(memory_validation_json),
            "--memory-prompts-csv",
            str(memory_prompts_csv),
            "--memory-pairs-csv",
            str(memory_pairs_csv),
            "--output",
            str(paths["evidence_archive"]),
            "--output-json",
            str(paths["evidence_manifest"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_validate",
        [
            sys.executable,
            "validate_text_release_evidence.py",
            "--manifest",
            str(paths["evidence_manifest"]),
            "--archive",
            str(paths["evidence_archive"]),
            "--output-json",
            str(paths["evidence_validation"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_compare",
        [
            sys.executable,
            "compare_text_release_evidence.py",
            "--baseline-manifest",
            str(paths["evidence_manifest"]),
            "--candidate-manifest",
            str(paths["evidence_manifest"]),
            "--fail-on-archive-sha-change",
            "--fail-on-file-set-change",
            "--fail-on-file-sha-change",
            "--fail-on-file-byte-change",
            "--fail-on-archive-name-change",
            "--fail-on-validity-change",
            "--fail-on-gate-regression",
            "--output-json",
            str(paths["evidence_comparison"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_unbundle",
        [
            sys.executable,
            "unbundle_text_release_evidence.py",
            "--archive",
            str(paths["evidence_archive"]),
            "--manifest",
            str(paths["evidence_manifest"]),
            "--output-dir",
            str(paths["evidence_restore_dir"]),
            "--output-json",
            str(paths["evidence_restore"]),
            "--clean",
        ],
        args.timeout,
    )
    remove_path(paths["evidence_ledger"])
    remove_path(paths["evidence_ledger_store_dir"])
    stage(
        report,
        "release_evidence_register",
        [
            sys.executable,
            "register_text_release_evidence.py",
            "--manifest",
            str(paths["evidence_manifest"]),
            "--archive",
            str(paths["evidence_archive"]),
            "--validation-json",
            str(paths["evidence_validation"]),
            "--restore-json",
            str(paths["evidence_restore"]),
            "--comparison-json",
            str(paths["evidence_comparison"]),
            "--ledger",
            str(paths["evidence_ledger"]),
            "--store-dir",
            str(paths["evidence_ledger_store_dir"]),
            "--name",
            args.name,
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_ledger_validate",
        [
            sys.executable,
            "validate_text_release_evidence_ledger.py",
            "--ledger",
            str(paths["evidence_ledger"]),
            "--output-json",
            str(paths["evidence_ledger_validation"]),
            "--expected-entries",
            "1",
            "--require-artifacts",
            "--artifact-scope",
            "latest",
            "--fail-on-failed-gates",
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_ledger_tamper",
        [
            sys.executable,
            "tamper_text_release_evidence_ledger.py",
            "--ledger",
            str(paths["evidence_ledger"]),
            "--output-ledger",
            str(paths["evidence_ledger_tampered"]),
            "--output-json",
            str(paths["evidence_ledger_tamper"]),
            "--mode",
            "archive_sha256",
            "--expected-entries",
            "1",
            "--require-artifacts",
            "--artifact-scope",
            "latest",
            "--fail-on-failed-gates",
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_ledger_replay",
        [
            sys.executable,
            "replay_text_release_evidence_ledger.py",
            "--manifest",
            str(paths["evidence_manifest"]),
            "--archive",
            str(paths["evidence_archive"]),
            "--validation-json",
            str(paths["evidence_validation"]),
            "--restore-json",
            str(paths["evidence_restore"]),
            "--comparison-json",
            str(paths["evidence_comparison"]),
            "--base-dir",
            str(base_dir),
            "--ledger",
            str(paths["evidence_ledger_replay"]),
            "--store-dir",
            str(paths["evidence_ledger_replay_store_dir"]),
            "--work-dir",
            str(paths["evidence_ledger_replay_work_dir"]),
            "--output-json",
            str(paths["evidence_ledger_replay_report"]),
            "--clean",
            "--fail-on-failed-gates",
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_ledger_compare",
        [
            sys.executable,
            "compare_text_release_evidence_ledgers.py",
            "--baseline-ledger",
            str(paths["evidence_ledger_replay"]),
            "--candidate-ledger",
            str(paths["evidence_ledger_replay"]),
            "--output-json",
            str(paths["evidence_ledger_comparison"]),
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
        args.timeout,
    )
    stage(
        report,
        "release_evidence_ledger_growth_compare",
        [
            sys.executable,
            "compare_text_release_evidence_ledgers.py",
            "--baseline-ledger",
            str(paths["evidence_ledger"]),
            "--candidate-ledger",
            str(paths["evidence_ledger_replay"]),
            "--output-json",
            str(paths["evidence_ledger_growth_comparison"]),
            "--require-artifacts",
            "--artifact-scope",
            "all",
            "--fail-on-failed-gates",
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_audit",
        [
            sys.executable,
            "summarize_text_release_evidence_audit.py",
            "--manifest",
            str(paths["evidence_manifest"]),
            "--validation-json",
            str(paths["evidence_validation"]),
            "--restore-json",
            str(paths["evidence_restore"]),
            "--comparison-json",
            str(paths["evidence_comparison"]),
            "--ledger",
            str(paths["evidence_ledger"]),
            "--ledger-validation-json",
            str(paths["evidence_ledger_validation"]),
            "--tamper-json",
            str(paths["evidence_ledger_tamper"]),
            "--replay-json",
            str(paths["evidence_ledger_replay_report"]),
            "--ledger-comparison-json",
            str(paths["evidence_ledger_comparison"]),
            "--ledger-growth-comparison-json",
            str(paths["evidence_ledger_growth_comparison"]),
            "--output-json",
            str(paths["evidence_audit"]),
            "--output-md",
            str(paths["evidence_audit_md"]),
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_audit_validate",
        [
            sys.executable,
            "validate_text_release_evidence_audit.py",
            "--audit",
            str(paths["evidence_audit"]),
            "--output-json",
            str(paths["evidence_audit_validation"]),
            "--require-sources",
        ],
        args.timeout,
    )
    stage(
        report,
        "release_evidence_audit_validation_compare",
        [
            sys.executable,
            "compare_text_release_evidence_audit_validations.py",
            "--baseline-validation",
            str(paths["evidence_audit_validation"]),
            "--candidate-validation",
            str(paths["evidence_audit_validation"]),
            "--output-json",
            str(paths["evidence_audit_validation_comparison"]),
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
        args.timeout,
    )
    stage(
        report,
        "release_evidence_audit_compare",
        [
            sys.executable,
            "compare_text_release_evidence_audits.py",
            "--baseline-audit",
            str(paths["evidence_audit"]),
            "--candidate-audit",
            str(paths["evidence_audit"]),
            "--output-json",
            str(paths["evidence_audit_comparison"]),
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
        args.timeout,
    )

    manifest = load_json(paths["evidence_manifest"])
    ledger = load_json(paths["evidence_ledger"])
    audit = load_json(paths["evidence_audit"])
    audit_validation = load_json(paths["evidence_audit_validation"])
    audit_validation_comparison = load_json(paths["evidence_audit_validation_comparison"])
    audit_comparison = load_json(paths["evidence_audit_comparison"])
    if audit.get("valid") is not True:
        raise SystemExit(f"Release evidence audit failed: {audit}")
    if audit_validation.get("valid") is not True:
        raise SystemExit(f"Release evidence audit validation failed: {audit_validation}")
    if audit_validation_comparison.get("failures"):
        raise SystemExit(f"Release evidence audit validation comparison failed: {audit_validation_comparison}")
    if audit_comparison.get("failures"):
        raise SystemExit(f"Release evidence audit comparison failed: {audit_comparison}")

    summary.update(
        {
            "release_gates": str(paths["release_gates"]),
            "release_report": str(paths["release_report_md"]),
            "release_evidence_archive": str(paths["evidence_archive"]),
            "release_evidence_manifest": str(paths["evidence_manifest"]),
            "release_evidence_archive_sha256": manifest.get("archive_sha256"),
            "release_evidence_archive_bytes": manifest.get("archive_bytes"),
            "release_evidence_ledger": str(paths["evidence_ledger"]),
            "release_evidence_ledger_chain_head": ledger.get("chain_head"),
            "release_evidence_audit": str(paths["evidence_audit"]),
            "release_evidence_audit_md": str(paths["evidence_audit_md"]),
            "release_evidence_audit_valid": audit.get("valid"),
            "release_evidence_audit_checks": (audit.get("summary") or {}).get("check_count"),
            "release_evidence_audit_validation": str(paths["evidence_audit_validation"]),
            "release_evidence_audit_validation_valid": audit_validation.get("valid"),
            "release_evidence_audit_validation_comparison": str(
                paths["evidence_audit_validation_comparison"]
            ),
            "release_evidence_audit_comparison": str(paths["evidence_audit_comparison"]),
        }
    )
    return summary


def run(args: argparse.Namespace) -> None:
    maybe_clean(args)
    promote_dir = Path(args.promote_dir)
    restore_dir = Path(args.restore_dir)
    manifest = promote_dir / "manifest.json"
    model_card_json = promote_dir / "model_card.json"
    model_card_md = promote_dir / "MODEL_CARD.md"
    benchmark_json = promote_dir / "benchmark.json"
    validation_json = promote_dir / "validation.json"
    archive = promote_dir / "text_package.tar.gz"
    bundle_json = promote_dir / "bundle_manifest.json"
    restore_report = restore_dir / "restore_report.json"
    restored_manifest = restore_dir / args.root_name / "manifest.json"
    comparison_json = promote_dir / "package_comparison.json"
    dashboard_json = Path(args.dashboard_json)
    memory_suite_dir, memory_suite_json, memory_prompts_csv, memory_pairs_csv, memory_validation_json = memory_paths(args)
    prompts = memory_prompts(args)

    report: dict[str, Any] = {
        "created_at": timestamp(),
        "name": args.name,
        "leaderboard": args.leaderboard,
        "promote_dir": args.promote_dir,
        "registry": args.registry,
        "stages": {},
    }

    print("TextPy/SoA text release pipeline")
    print(f"name: {args.name}")
    print(f"leaderboard: {args.leaderboard}")
    print(f"promote_dir: {promote_dir}")
    print("")

    promote_cmd = [
        sys.executable,
        "promote_text_checkpoint.py",
        "--leaderboard",
        args.leaderboard,
        "--output-dir",
        str(promote_dir),
        "--checkpoint-role",
        args.checkpoint_role,
        "--sample-steps",
        str(args.sample_steps),
        "--timeout",
        str(args.timeout),
        "--smoke-run",
    ]
    if args.text_file:
        promote_cmd.extend(["--text-file", args.text_file])
    stage(report, "promote", promote_cmd, args.timeout)

    stage(
        report,
        "inspect",
        [
            sys.executable,
            "inspect_text_checkpoint.py",
            "--manifest",
            str(manifest),
            "--output-json",
            str(model_card_json),
            "--output-md",
            str(model_card_md),
        ],
        args.timeout,
    )
    benchmark_cmd = [
        sys.executable,
        "benchmark_text_package.py",
        "--manifest",
        str(manifest),
        "--warmup",
        str(args.benchmark_warmup),
        "--repeat",
        str(args.benchmark_repeat),
        "--sample-steps",
        str(args.sample_steps),
        "--output-json",
        str(benchmark_json),
    ]
    if args.text_file:
        benchmark_cmd.extend(["--text-file", args.text_file])
    stage(report, "benchmark", benchmark_cmd, args.timeout)
    stage(
        report,
        "validate",
        [
            sys.executable,
            "validate_text_package.py",
            "--manifest",
            str(manifest),
            "--require-model-card",
            "--require-smoke",
            "--require-benchmark",
            "--output-json",
            str(validation_json),
        ],
        args.timeout,
    )
    stage(
        report,
        "bundle",
        [
            sys.executable,
            "bundle_text_package.py",
            "--manifest",
            str(manifest),
            "--output",
            str(archive),
            "--output-json",
            str(bundle_json),
            "--root-name",
            args.root_name,
        ],
        args.timeout,
    )
    if restore_dir.exists():
        shutil.rmtree(restore_dir)
    stage(
        report,
        "unbundle",
        [
            sys.executable,
            "unbundle_text_package.py",
            "--archive",
            str(archive),
            "--bundle-manifest",
            str(bundle_json),
            "--output-dir",
            str(restore_dir),
            "--output-json",
            str(restore_report),
        ],
        args.timeout,
    )
    stage(
        report,
        "compare",
        [
            sys.executable,
            "compare_text_packages.py",
            "--baseline-manifest",
            str(manifest),
            "--candidate-manifest",
            str(restored_manifest),
            "--fail-on-loss-regression",
            "--fail-on-throughput-regression",
            "--output-json",
            str(comparison_json),
        ],
        args.timeout,
    )
    memory_suite_cmd = [
        sys.executable,
        "run_text_memory_suite.py",
        "--manifest",
        str(manifest),
        "--top-k",
        str(args.memory_top_k),
        "--max-tokens",
        str(args.memory_max_tokens),
        "--include-state-vectors",
        "--output-dir",
        str(memory_suite_dir),
        "--output-json",
        str(memory_suite_json),
        "--prompts-csv",
        str(memory_prompts_csv),
        "--pairs-csv",
        str(memory_pairs_csv),
    ]
    for prompt in prompts:
        memory_suite_cmd.extend(["--prompt", prompt])
    stage(report, "memory_suite", memory_suite_cmd, args.timeout)

    validate_memory_cmd = [
        sys.executable,
        "validate_text_memory_suite.py",
        "--suite",
        str(memory_suite_json),
        "--prompts-csv",
        str(memory_prompts_csv),
        "--pairs-csv",
        str(memory_pairs_csv),
        "--expected-prompts",
        str(len(prompts)),
        "--require-state-vectors",
        "--require-prompt-artifacts",
        "--output-json",
        str(memory_validation_json),
    ]
    validate_memory_cmd.extend(memory_threshold_args(args))
    stage(report, "validate_memory_suite", validate_memory_cmd, args.timeout)

    stage(
        report,
        "register",
        [
            sys.executable,
            "register_text_package.py",
            "--manifest",
            str(manifest),
            "--bundle-manifest",
            str(bundle_json),
            "--archive",
            str(archive),
            "--comparison-json",
            str(comparison_json),
            "--memory-validation-json",
            str(memory_validation_json),
            "--registry",
            args.registry,
            "--name",
            args.name,
        ],
        args.timeout,
    )
    stage(
        report,
        "release_leaderboard",
        [
            sys.executable,
            "list_text_releases.py",
            "--registry",
            args.registry,
            "--output-json",
            args.release_leaderboard_json,
            "--output-csv",
            args.release_leaderboard_csv,
        ],
        args.timeout,
    )
    stage(
        report,
        "select",
        [
            sys.executable,
            "select_text_release.py",
            "--registry",
            args.registry,
            "--rank",
            "1",
            "--output-json",
            args.current_release_json,
        ],
        args.timeout,
    )
    run_active_cmd = [
        sys.executable,
        "run_text_checkpoint.py",
        "--release",
        args.current_release_json,
        "--sample-steps",
        str(args.sample_steps),
        "--output-json",
        args.run_json,
    ]
    if args.text_file:
        run_active_cmd.extend(["--text-file", args.text_file])
    stage(report, "run_active_release", run_active_cmd, args.timeout)

    summary = verify_pipeline_outputs(args)
    report["summary"] = summary
    write_json(Path(args.output_json), report)
    stage(
        report,
        "release_dashboard",
        [
            sys.executable,
            "build_text_release_dashboard.py",
            "--leaderboard",
            args.release_leaderboard_json,
            "--current-release",
            args.current_release_json,
            "--memory-suite",
            str(memory_suite_json),
            "--memory-validation",
            str(memory_validation_json),
            "--run-json",
            args.run_json,
            "--pipeline-json",
            args.output_json,
            "--output-json",
            str(dashboard_json),
            "--require-valid",
        ],
        args.timeout,
    )
    dashboard_payload = load_json(dashboard_json)
    dashboard_audit = dashboard_payload.get("audit") or {}
    if dashboard_payload.get("kind") != "text_release_dashboard" or dashboard_audit.get("valid") is not True:
        raise SystemExit(f"Release dashboard artifact is invalid: {dashboard_payload}")
    summary["release_dashboard"] = str(dashboard_json)
    summary["release_dashboard_audit_valid"] = dashboard_audit["valid"]
    summary["release_dashboard_audit_checks"] = dashboard_audit["check_count"]
    report["summary"] = summary
    report["completed_at"] = timestamp()
    write_json(Path(args.output_json), report)

    if args.include_evidence_audit:
        summary = run_evidence_audit_chain(
            args,
            report,
            summary,
            dashboard_json=dashboard_json,
            memory_suite_json=memory_suite_json,
            memory_prompts_csv=memory_prompts_csv,
            memory_pairs_csv=memory_pairs_csv,
            memory_validation_json=memory_validation_json,
        )
        report["summary"] = summary
        report["completed_at"] = timestamp()
        write_json(Path(args.output_json), report)

    print("")
    print("Release Summary")
    print(f"  manifest: {summary['manifest']}")
    print(f"  checkpoint_sha256: {summary['checkpoint_sha256'][:12]}")
    print(f"  archive_sha256: {summary['archive_sha256'][:12]}")
    print(f"  registry_releases: {summary['release_count']}")
    print(f"  active_release: {summary['current_release']}")
    print(f"  loss: {summary['loss']:.4f}")
    print(f"  accuracy: {summary['accuracy']:.4f}")
    print(f"  best_throughput_tokens_s: {summary['best_throughput_tokens_s']:,.0f}")
    print(f"  memory_suite: {summary['memory_suite']}")
    print(f"  memory_mean_pair_overlap: {summary['memory_mean_pair_overlap']:.4f}")
    print(f"  memory_min_pair_state_cosine_similarity: {summary['memory_min_pair_state_cosine_similarity']:.4f}")
    print(f"  release_dashboard: {summary['release_dashboard']}")
    print(f"  release_dashboard_audit_checks: {summary['release_dashboard_audit_checks']}")
    if args.include_evidence_audit:
        print(f"  release_evidence_archive_sha256: {summary['release_evidence_archive_sha256'][:12]}")
        print(f"  release_evidence_ledger_chain_head: {summary['release_evidence_ledger_chain_head'][:12]}")
        print(f"  release_evidence_audit: {summary['release_evidence_audit']}")
        print(f"  release_evidence_audit_checks: {summary['release_evidence_audit_checks']}")
    print(f"  sample: {summary['sample']}")
    print("")
    print(f"Saved pipeline JSON: {args.output_json}")
    print("RELEASE PIPELINE OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full text package release pipeline.")
    parser.add_argument("--leaderboard", default="artifacts/text_sweeps/demo/leaderboard.json")
    parser.add_argument("--promote-dir", default="artifacts/promoted/text_release")
    parser.add_argument("--restore-dir", default="artifacts/restored/text_release")
    parser.add_argument("--root-name", default="text_release")
    parser.add_argument("--registry", default="artifacts/releases/text_packages.json")
    parser.add_argument("--release-leaderboard-json", default="artifacts/releases/text_release_leaderboard.json")
    parser.add_argument("--release-leaderboard-csv", default="artifacts/releases/text_release_leaderboard.csv")
    parser.add_argument("--current-release-json", default="artifacts/releases/current_text_release.json")
    parser.add_argument("--run-json", default="artifacts/releases/run_current_text_release.json")
    parser.add_argument("--dashboard-json", default="artifacts/releases/text_release_dashboard.json")
    parser.add_argument("--output-json", default="artifacts/releases/text_release_pipeline.json")
    parser.add_argument("--memory-suite-dir", default="artifacts/releases/text_release_memory_suite")
    parser.add_argument("--memory-suite-json", default=None)
    parser.add_argument("--memory-prompts-csv", default=None)
    parser.add_argument("--memory-pairs-csv", default=None)
    parser.add_argument("--memory-validation-json", default=None)
    parser.add_argument("--include-evidence-audit", action="store_true")
    parser.add_argument("--release-dashboard-validation-json", default=None)
    parser.add_argument("--release-dashboard-comparison-json", default=None)
    parser.add_argument("--release-history-json", default=None)
    parser.add_argument("--release-history-validation-json", default=None)
    parser.add_argument("--release-history-analysis-json", default=None)
    parser.add_argument("--release-gates-json", default=None)
    parser.add_argument("--release-report-md", default=None)
    parser.add_argument("--evidence-root-name", default="text_release_evidence")
    parser.add_argument("--evidence-archive", default=None)
    parser.add_argument("--evidence-manifest-json", default=None)
    parser.add_argument("--evidence-validation-json", default=None)
    parser.add_argument("--evidence-restore-dir", default=None)
    parser.add_argument("--evidence-restore-json", default=None)
    parser.add_argument("--evidence-comparison-json", default=None)
    parser.add_argument("--evidence-ledger-json", default=None)
    parser.add_argument("--evidence-ledger-store-dir", default=None)
    parser.add_argument("--evidence-ledger-validation-json", default=None)
    parser.add_argument("--evidence-ledger-tamper-json", default=None)
    parser.add_argument("--evidence-ledger-tampered-json", default=None)
    parser.add_argument("--evidence-ledger-replay-json", default=None)
    parser.add_argument("--evidence-ledger-replay-store-dir", default=None)
    parser.add_argument("--evidence-ledger-replay-work-dir", default=None)
    parser.add_argument("--evidence-ledger-replay-report-json", default=None)
    parser.add_argument("--evidence-ledger-comparison-json", default=None)
    parser.add_argument("--evidence-ledger-growth-comparison-json", default=None)
    parser.add_argument("--evidence-audit-json", default=None)
    parser.add_argument("--evidence-audit-md", default=None)
    parser.add_argument("--evidence-audit-validation-json", default=None)
    parser.add_argument("--evidence-audit-validation-comparison-json", default=None)
    parser.add_argument("--evidence-audit-comparison-json", default=None)
    parser.add_argument("--name", default="text_release")
    parser.add_argument("--checkpoint-role", choices=("best", "candidate", "baseline"), default="best")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeat", type=int, default=5)
    parser.add_argument("--memory-prompt", action="append", default=None)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--memory-max-tokens", type=int, default=32)
    parser.add_argument("--memory-min-mean-pair-overlap", type=float, default=0.0)
    parser.add_argument("--memory-min-pair-overlap", type=float, default=0.0)
    parser.add_argument("--memory-max-pair-state-norm-delta", type=float, default=None)
    parser.add_argument("--memory-max-pair-state-l2-distance", type=float, default=None)
    parser.add_argument("--memory-min-pair-state-cosine-similarity", type=float, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
