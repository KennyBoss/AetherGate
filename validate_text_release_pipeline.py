#!/usr/bin/env python3
"""Validate a TextPy/SoA release pipeline JSON report."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from validate_text_memory_suite import validate as validate_memory_suite
from validate_text_release_dashboard import validate as validate_dashboard
from validate_text_release_evidence_audit import validate as validate_evidence_audit


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

BASE_STAGES = [
    "promote",
    "inspect",
    "benchmark",
    "validate",
    "bundle",
    "unbundle",
    "compare",
    "memory_suite",
    "validate_memory_suite",
    "register",
    "release_leaderboard",
    "select",
    "run_active_release",
    "release_dashboard",
]

EVIDENCE_STAGES = [
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
]


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return payload


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def as_number(value: Any, label: str, errors: list[str], *, minimum: float | None = None) -> float | None:
    if not is_number(value):
        errors.append(f"{label} must be a finite number.")
        return None
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        errors.append(f"{label} is below minimum {minimum}: {numeric}")
    return numeric


def as_int(value: Any, label: str, errors: list[str], *, minimum: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer.")
        return None
    numeric = int(value)
    if minimum is not None and numeric < minimum:
        errors.append(f"{label} is below minimum {minimum}: {numeric}")
    return numeric


def close_enough(left: Any, right: Any, tolerance: float = 1.0e-9) -> bool:
    if is_number(left) and is_number(right):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def resolve_path(base: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    local = base / path
    if local.exists():
        return local
    local_by_name = base / path.name
    if local_by_name.exists():
        return local_by_name
    return path


def require_path(base: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    path = resolve_path(base, value)
    if path is None:
        errors.append(f"{label} is missing.")
        return None
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")
        return path
    return path


def check_sha(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        errors.append(f"{label} must be a 64-character lowercase SHA256 string.")


def validate_stage_record(stage: Any, name: str, errors: list[str]) -> None:
    if not isinstance(stage, dict):
        errors.append(f"stages.{name} must be an object.")
        return
    command = stage.get("command")
    if not isinstance(command, list) or not command:
        errors.append(f"stages.{name}.command must be a non-empty list.")
    elapsed = as_number(stage.get("elapsed_s"), f"stages.{name}.elapsed_s", errors, minimum=0.0)
    if elapsed is None:
        return
    stdout_tail = stage.get("stdout_tail")
    if not isinstance(stdout_tail, list):
        errors.append(f"stages.{name}.stdout_tail must be a list.")


def validate_stages(stages: Any, errors: list[str], *, require_evidence_audit: bool) -> int:
    if not isinstance(stages, dict):
        errors.append("stages must be an object.")
        return 0
    required = list(BASE_STAGES)
    if require_evidence_audit:
        required.extend(EVIDENCE_STAGES)
    missing = [name for name in required if name not in stages]
    if missing:
        errors.append(f"missing required stages: {missing}")
    for name, stage in stages.items():
        validate_stage_record(stage, name, errors)
    return len(stages)


def validate_summary_shape(summary: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(summary, dict):
        errors.append("summary must be an object.")
        return {}
    for key in [
        "manifest",
        "checkpoint",
        "archive",
        "registry",
        "current_release",
        "run_json",
        "memory_suite",
        "memory_suite_validation",
        "release_dashboard",
    ]:
        if not isinstance(summary.get(key), str) or not summary.get(key):
            errors.append(f"summary.{key} must be a non-empty string.")
    for key in ["checkpoint_sha256", "archive_sha256"]:
        check_sha(summary.get(key), f"summary.{key}", errors)
    as_int(summary.get("release_count"), "summary.release_count", errors, minimum=1)
    as_int(summary.get("selected_rank"), "summary.selected_rank", errors, minimum=1)
    as_int(summary.get("memory_prompt_count"), "summary.memory_prompt_count", errors, minimum=1)
    as_int(summary.get("memory_pair_count"), "summary.memory_pair_count", errors, minimum=0)
    for key in [
        "loss",
        "accuracy",
        "best_throughput_tokens_s",
        "memory_mean_pair_overlap",
        "memory_min_pair_overlap",
        "memory_max_pair_state_norm_delta",
        "memory_max_pair_state_l2_distance",
        "memory_min_pair_state_cosine_similarity",
    ]:
        as_number(summary.get(key), f"summary.{key}", errors)
    if not summary.get("sample"):
        errors.append("summary.sample must be present.")
    if summary.get("release_dashboard_audit_valid") is not True:
        errors.append("summary.release_dashboard_audit_valid must be true.")
    as_int(summary.get("release_dashboard_audit_checks"), "summary.release_dashboard_audit_checks", errors, minimum=1)
    return summary


def validate_release_artifacts(
    base: Path,
    summary: dict[str, Any],
    errors: list[str],
    *,
    require_artifacts: bool,
) -> dict[str, Any]:
    if not require_artifacts:
        return {}
    paths = {
        key: require_path(base, summary.get(key), f"summary.{key}", errors)
        for key in [
            "manifest",
            "archive",
            "registry",
            "current_release",
            "run_json",
            "memory_suite",
            "memory_suite_validation",
            "release_dashboard",
        ]
    }
    manifest_path = paths.get("manifest")
    if manifest_path and manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("checkpoint_sha256") and manifest.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
            errors.append("summary.checkpoint_sha256 must match manifest.checkpoint_sha256.")
    registry_path = paths.get("registry")
    if registry_path and registry_path.exists():
        registry = load_json(registry_path)
        if int(registry.get("release_count") or 0) < int(summary.get("release_count") or 0):
            errors.append("registry.release_count is lower than summary.release_count.")
    current_path = paths.get("current_release")
    if current_path and current_path.exists():
        current = load_json(current_path)
        if current.get("rank") != summary.get("selected_rank"):
            errors.append("summary.selected_rank must match current release rank.")
        if current.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
            errors.append("summary.checkpoint_sha256 must match current release checkpoint SHA.")
        if current.get("archive_sha256") != summary.get("archive_sha256"):
            errors.append("summary.archive_sha256 must match current release archive SHA.")
        memory = current.get("memory") if isinstance(current.get("memory"), dict) else {}
        if memory.get("validation") != summary.get("memory_suite_validation"):
            errors.append("current release memory validation must match summary.memory_suite_validation.")
    run_path = paths.get("run_json")
    if run_path and run_path.exists():
        run_report = load_json(run_path)
        if run_report.get("release") != summary.get("current_release"):
            errors.append("run_json.release must match summary.current_release.")
        if not close_enough(run_report.get("loss"), summary.get("loss")):
            errors.append("run_json.loss must match summary.loss.")
        if not close_enough(run_report.get("accuracy"), summary.get("accuracy")):
            errors.append("run_json.accuracy must match summary.accuracy.")
    return paths


def validate_memory_artifacts(base: Path, summary: dict[str, Any], errors: list[str], *, require_artifacts: bool) -> int:
    if not require_artifacts:
        return 0
    suite = require_path(base, summary.get("memory_suite"), "summary.memory_suite", errors)
    validation = require_path(base, summary.get("memory_suite_validation"), "summary.memory_suite_validation", errors)
    if not suite or not suite.exists() or not validation or not validation.exists():
        return 0
    suite_payload = load_json(suite)
    validation_payload = load_json(validation)
    prompts_csv = suite.parent / "prompts.csv"
    pairs_csv = suite.parent / "pairs.csv"
    expected_prompts = int(summary.get("memory_prompt_count") or 0)
    expected_pairs = int(summary.get("memory_pair_count") or 0)
    validation_args = argparse.Namespace(
        suite=str(suite),
        prompts_csv=str(prompts_csv) if prompts_csv.exists() else None,
        pairs_csv=str(pairs_csv) if pairs_csv.exists() else None,
        output_json=None,
        expected_prompts=expected_prompts,
        expected_pairs=expected_pairs,
        min_prompts=1,
        min_pairs=0,
        require_state_vectors=True,
        require_prompt_artifacts=True,
        min_mean_pair_overlap=None,
        min_pair_overlap=None,
        max_pair_state_norm_delta=None,
        max_pair_state_l2_distance=None,
        min_pair_state_cosine_similarity=None,
    )
    result = validate_memory_suite(validation_args)
    if result.get("valid") is not True:
        errors.append(f"memory suite validation failed: {result.get('errors')}")
    if validation_payload.get("valid") is not True:
        errors.append("summary.memory_suite_validation artifact must be valid.")
    for key in [
        "mean_pair_overlap",
        "min_pair_overlap",
        "max_pair_state_norm_delta",
        "max_pair_state_l2_distance",
        "min_pair_state_cosine_similarity",
    ]:
        summary_key = "memory_" + key
        if not close_enough(validation_payload.get(key), summary.get(summary_key)):
            errors.append(f"summary.{summary_key} must match memory validation {key}.")
    if suite_payload.get("prompt_count") != expected_prompts:
        errors.append("memory suite prompt_count must match summary.memory_prompt_count.")
    return int(result.get("checked_prompt_artifacts") or 0)


def validate_dashboard_artifact(base: Path, summary: dict[str, Any], errors: list[str], *, require_artifacts: bool) -> int:
    if not require_artifacts:
        return 0
    dashboard = require_path(base, summary.get("release_dashboard"), "summary.release_dashboard", errors)
    if not dashboard or not dashboard.exists():
        return 0
    result = validate_dashboard(
        argparse.Namespace(
            dashboard=str(dashboard),
            output_json=None,
            require_artifacts=True,
            expected_checks=summary.get("release_dashboard_audit_checks"),
            expected_prompts=summary.get("memory_prompt_count"),
            expected_pairs=summary.get("memory_pair_count"),
            min_mean_pair_overlap=0.0,
        )
    )
    if result.get("valid") is not True:
        errors.append(f"release dashboard validation failed: {result.get('errors')}")
    if result.get("audit_check_count") != summary.get("release_dashboard_audit_checks"):
        errors.append("dashboard audit check count must match pipeline summary.")
    return int(result.get("artifact_checks") or 0)


def validate_evidence_artifacts(base: Path, summary: dict[str, Any], errors: list[str], *, require_artifacts: bool) -> int:
    evidence_keys = [
        "release_gates",
        "release_report",
        "release_evidence_archive",
        "release_evidence_manifest",
        "release_evidence_ledger",
        "release_evidence_audit",
        "release_evidence_audit_md",
        "release_evidence_audit_validation",
        "release_evidence_audit_validation_comparison",
        "release_evidence_audit_comparison",
    ]
    missing_fields = [key for key in evidence_keys if not summary.get(key)]
    if missing_fields:
        errors.append(f"missing evidence summary fields: {missing_fields}")
        return 0
    check_sha(summary.get("release_evidence_archive_sha256"), "summary.release_evidence_archive_sha256", errors)
    check_sha(summary.get("release_evidence_ledger_chain_head"), "summary.release_evidence_ledger_chain_head", errors)
    if summary.get("release_evidence_audit_valid") is not True:
        errors.append("summary.release_evidence_audit_valid must be true.")
    if summary.get("release_evidence_audit_validation_valid") is not True:
        errors.append("summary.release_evidence_audit_validation_valid must be true.")
    as_int(summary.get("release_evidence_audit_checks"), "summary.release_evidence_audit_checks", errors, minimum=10)
    if not require_artifacts:
        return 0
    paths = {key: require_path(base, summary.get(key), f"summary.{key}", errors) for key in evidence_keys}
    manifest_path = paths.get("release_evidence_manifest")
    if manifest_path and manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("archive_sha256") != summary.get("release_evidence_archive_sha256"):
            errors.append("evidence manifest archive_sha256 must match pipeline summary.")
    ledger_path = paths.get("release_evidence_ledger")
    if ledger_path and ledger_path.exists():
        ledger = load_json(ledger_path)
        if ledger.get("chain_head") != summary.get("release_evidence_ledger_chain_head"):
            errors.append("evidence ledger chain_head must match pipeline summary.")
    audit_path = paths.get("release_evidence_audit")
    artifact_checks = 0
    if audit_path and audit_path.exists():
        audit_result = validate_evidence_audit(
            argparse.Namespace(audit=str(audit_path), output_json=None, require_sources=True)
        )
        if audit_result.get("valid") is not True:
            errors.append(f"release evidence audit validation failed: {audit_result.get('errors')}")
        if audit_result.get("check_count") != summary.get("release_evidence_audit_checks"):
            errors.append("evidence audit check count must match pipeline summary.")
        artifact_checks += int(audit_result.get("source_artifact_checks") or 0)
    for key in ["release_evidence_audit_validation_comparison", "release_evidence_audit_comparison"]:
        path = paths.get(key)
        if path and path.exists():
            payload = load_json(path)
            if payload.get("failures"):
                errors.append(f"{key} failures must be empty.")
            artifact_checks += 1
    validation_path = paths.get("release_evidence_audit_validation")
    if validation_path and validation_path.exists():
        validation = load_json(validation_path)
        if validation.get("valid") is not True:
            errors.append("release evidence audit validation artifact must be valid.")
        artifact_checks += 1
    return artifact_checks


def validate(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_path = Path(args.pipeline)
    base = pipeline_path.parent
    payload = load_json(pipeline_path)
    errors: list[str] = []
    for key in ["created_at", "completed_at", "name", "leaderboard", "promote_dir", "registry"]:
        if not isinstance(payload.get(key), str) or not payload.get(key):
            errors.append(f"{key} must be a non-empty string.")
    stages = payload.get("stages")
    stage_count = validate_stages(stages, errors, require_evidence_audit=args.require_evidence_audit)
    summary = validate_summary_shape(payload.get("summary"), errors)
    validate_release_artifacts(base, summary, errors, require_artifacts=args.require_artifacts)
    memory_artifacts = validate_memory_artifacts(base, summary, errors, require_artifacts=args.require_artifacts)
    dashboard_artifacts = validate_dashboard_artifact(base, summary, errors, require_artifacts=args.require_artifacts)
    evidence_artifacts = 0
    has_evidence = bool(summary.get("release_evidence_audit"))
    if args.require_evidence_audit or has_evidence:
        evidence_artifacts = validate_evidence_artifacts(
            base,
            summary,
            errors,
            require_artifacts=args.require_artifacts,
        )
    if args.expected_stages is not None and stage_count != args.expected_stages:
        errors.append(f"stage count mismatch: {stage_count} != {args.expected_stages}")
    return {
        "valid": not errors,
        "pipeline": str(pipeline_path),
        "name": payload.get("name"),
        "release_count": summary.get("release_count"),
        "selected_rank": summary.get("selected_rank"),
        "stage_count": stage_count,
        "memory_prompt_count": summary.get("memory_prompt_count"),
        "memory_pair_count": summary.get("memory_pair_count"),
        "dashboard_audit_checks": summary.get("release_dashboard_audit_checks"),
        "has_evidence_audit": has_evidence,
        "evidence_audit_checks": summary.get("release_evidence_audit_checks"),
        "artifact_checks": memory_artifacts + dashboard_artifacts + evidence_artifacts,
        "errors": errors,
    }


def run(args: argparse.Namespace) -> int:
    result = validate(args)
    if args.output_json:
        save_json(args.output_json, result)
    print("TextPy/SoA release pipeline validator")
    print(f"pipeline: {result['pipeline']}")
    print(f"valid: {result['valid']}")
    print(f"name: {result['name']}")
    print(f"stages: {result['stage_count']}")
    print(f"rank: {result['selected_rank']}")
    print(f"memory_prompts: {result['memory_prompt_count']}")
    print(f"memory_pairs: {result['memory_pair_count']}")
    print(f"dashboard_audit_checks: {result['dashboard_audit_checks']}")
    print(f"has_evidence_audit: {result['has_evidence_audit']}")
    print(f"evidence_audit_checks: {result['evidence_audit_checks']}")
    print(f"artifact_checks: {result['artifact_checks']}")
    if args.output_json:
        print(f"Saved JSON: {args.output_json}")
    if not result["valid"]:
        print("")
        print("FAIL")
        for error in result["errors"]:
            print(f"  {error}")
        return 1
    print("")
    print("RELEASE PIPELINE VALIDATION OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a release pipeline JSON report.")
    parser.add_argument("--pipeline", default="artifacts/releases/text_release_pipeline_demo.json")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--require-evidence-audit", action="store_true")
    parser.add_argument("--expected-stages", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
