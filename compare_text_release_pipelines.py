#!/usr/bin/env python3
"""Compare two TextPy/SoA release pipeline JSON reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class ReleasePipelineComparisonError(RuntimeError):
    pass


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleasePipelineComparisonError(f"Missing pipeline JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleasePipelineComparisonError(f"Invalid pipeline JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleasePipelineComparisonError(f"Expected JSON object: {path}")
    return payload


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def as_number(value: Any) -> float | None:
    return float(value) if is_number(value) else None


def as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def stage_names_from(stages: Any) -> list[str]:
    if isinstance(stages, dict):
        return [str(name) for name in stages.keys()]
    if isinstance(stages, list):
        names: list[str] = []
        for index, stage in enumerate(stages):
            if isinstance(stage, dict) and isinstance(stage.get("name"), str):
                names.append(stage["name"])
            else:
                names.append(str(index))
        return names
    return []


def stage_elapsed_s(stages: Any) -> float | None:
    records: list[Any]
    if isinstance(stages, dict):
        records = list(stages.values())
    elif isinstance(stages, list):
        records = stages
    else:
        return None
    total = 0.0
    found = False
    for record in records:
        if isinstance(record, dict) and is_number(record.get("elapsed_s")):
            total += float(record["elapsed_s"])
            found = True
    return total if found else None


def require_number(summary: dict[str, Any], key: str, errors: list[str]) -> None:
    if not is_number(summary.get(key)):
        errors.append(f"summary.{key} must be a finite number.")


def require_int(summary: dict[str, Any], key: str, errors: list[str]) -> None:
    if as_int(summary.get(key)) is None:
        errors.append(f"summary.{key} must be an integer.")


def require_string(summary: dict[str, Any], key: str, errors: list[str]) -> None:
    if not isinstance(summary.get(key), str) or not summary.get(key):
        errors.append(f"summary.{key} must be a non-empty string.")


def validate_shape(
    path: Path,
    payload: dict[str, Any],
    *,
    require_evidence_audit: bool,
    allow_failed_pipeline: bool,
) -> list[str]:
    errors: list[str] = []
    summary = payload.get("summary")
    stages = payload.get("stages")
    if not isinstance(summary, dict):
        errors.append("summary must be an object.")
        summary = {}
    if not stage_names_from(stages):
        errors.append("stages must be a non-empty object or list.")
    if payload.get("name") is not None and not isinstance(payload.get("name"), str):
        errors.append("name must be a string or null.")
    if allow_failed_pipeline:
        return errors

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
        require_number(summary, key, errors)
    for key in [
        "release_count",
        "selected_rank",
        "memory_prompt_count",
        "memory_pair_count",
        "release_dashboard_audit_checks",
    ]:
        require_int(summary, key, errors)
    for key in ["checkpoint_sha256", "archive_sha256"]:
        require_string(summary, key, errors)
    if summary.get("release_dashboard_audit_valid") is not True:
        errors.append("summary.release_dashboard_audit_valid must be true.")
    if require_evidence_audit:
        for key in ["release_evidence_archive_sha256", "release_evidence_ledger_chain_head"]:
            require_string(summary, key, errors)
        require_int(summary, "release_evidence_audit_checks", errors)
        if summary.get("release_evidence_audit_valid") is not True:
            errors.append("summary.release_evidence_audit_valid must be true.")
        if summary.get("release_evidence_audit_validation_valid") is not True:
            errors.append("summary.release_evidence_audit_validation_valid must be true.")
        if not summary.get("release_evidence_audit"):
            errors.append("summary.release_evidence_audit must be present.")
    if errors:
        errors.insert(0, f"pipeline report is incomplete: {path}")
    return errors


def load_pipeline(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(path)
    errors = validate_shape(
        path,
        payload,
        require_evidence_audit=args.require_evidence_audit,
        allow_failed_pipeline=args.allow_failed_pipeline,
    )
    if errors and not args.allow_failed_pipeline:
        raise ReleasePipelineComparisonError(f"Pipeline shape failed for {path}: {errors}")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    stages = payload.get("stages")
    stage_names = stage_names_from(stages)
    return {
        "path": str(path),
        "payload": payload,
        "shape_errors": errors,
        "name": payload.get("name"),
        "created_at": payload.get("created_at"),
        "completed_at": payload.get("completed_at"),
        "leaderboard": payload.get("leaderboard"),
        "promote_dir": payload.get("promote_dir"),
        "registry": payload.get("registry"),
        "stage_names": stage_names,
        "stage_count": len(stage_names),
        "stage_elapsed_s": stage_elapsed_s(stages),
        "release_count": as_int(summary.get("release_count")),
        "selected_rank": as_int(summary.get("selected_rank")),
        "loss": as_number(summary.get("loss")),
        "accuracy": as_number(summary.get("accuracy")),
        "best_throughput_tokens_s": as_number(summary.get("best_throughput_tokens_s")),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "archive_sha256": summary.get("archive_sha256"),
        "current_release": summary.get("current_release"),
        "release_dashboard": summary.get("release_dashboard"),
        "release_dashboard_audit_valid": summary.get("release_dashboard_audit_valid"),
        "release_dashboard_audit_checks": as_int(summary.get("release_dashboard_audit_checks")),
        "memory_prompt_count": as_int(summary.get("memory_prompt_count")),
        "memory_pair_count": as_int(summary.get("memory_pair_count")),
        "memory_mean_pair_overlap": as_number(summary.get("memory_mean_pair_overlap")),
        "memory_min_pair_overlap": as_number(summary.get("memory_min_pair_overlap")),
        "memory_max_pair_state_norm_delta": as_number(summary.get("memory_max_pair_state_norm_delta")),
        "memory_max_pair_state_l2_distance": as_number(summary.get("memory_max_pair_state_l2_distance")),
        "memory_min_pair_state_cosine_similarity": as_number(
            summary.get("memory_min_pair_state_cosine_similarity")
        ),
        "has_evidence_audit": bool(summary.get("release_evidence_audit")),
        "release_evidence_archive": summary.get("release_evidence_archive"),
        "release_evidence_archive_sha256": summary.get("release_evidence_archive_sha256"),
        "release_evidence_ledger": summary.get("release_evidence_ledger"),
        "release_evidence_ledger_chain_head": summary.get("release_evidence_ledger_chain_head"),
        "release_evidence_audit": summary.get("release_evidence_audit"),
        "release_evidence_audit_valid": summary.get("release_evidence_audit_valid"),
        "release_evidence_audit_checks": as_int(summary.get("release_evidence_audit_checks")),
        "release_evidence_audit_validation": summary.get("release_evidence_audit_validation"),
        "release_evidence_audit_validation_valid": summary.get("release_evidence_audit_validation_valid"),
    }


def compact_pipeline(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "shape_errors": row["shape_errors"],
        "name": row["name"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "leaderboard": row["leaderboard"],
        "promote_dir": row["promote_dir"],
        "registry": row["registry"],
        "stage_count": row["stage_count"],
        "stage_names": row["stage_names"],
        "stage_elapsed_s": row["stage_elapsed_s"],
        "release_count": row["release_count"],
        "selected_rank": row["selected_rank"],
        "loss": row["loss"],
        "accuracy": row["accuracy"],
        "best_throughput_tokens_s": row["best_throughput_tokens_s"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "archive_sha256": row["archive_sha256"],
        "current_release": row["current_release"],
        "release_dashboard": row["release_dashboard"],
        "release_dashboard_audit_valid": row["release_dashboard_audit_valid"],
        "release_dashboard_audit_checks": row["release_dashboard_audit_checks"],
        "memory_prompt_count": row["memory_prompt_count"],
        "memory_pair_count": row["memory_pair_count"],
        "memory_mean_pair_overlap": row["memory_mean_pair_overlap"],
        "memory_min_pair_overlap": row["memory_min_pair_overlap"],
        "memory_max_pair_state_norm_delta": row["memory_max_pair_state_norm_delta"],
        "memory_max_pair_state_l2_distance": row["memory_max_pair_state_l2_distance"],
        "memory_min_pair_state_cosine_similarity": row["memory_min_pair_state_cosine_similarity"],
        "has_evidence_audit": row["has_evidence_audit"],
        "release_evidence_archive": row["release_evidence_archive"],
        "release_evidence_archive_sha256": row["release_evidence_archive_sha256"],
        "release_evidence_ledger": row["release_evidence_ledger"],
        "release_evidence_ledger_chain_head": row["release_evidence_ledger_chain_head"],
        "release_evidence_audit": row["release_evidence_audit"],
        "release_evidence_audit_valid": row["release_evidence_audit_valid"],
        "release_evidence_audit_checks": row["release_evidence_audit_checks"],
        "release_evidence_audit_validation": row["release_evidence_audit_validation"],
        "release_evidence_audit_validation_valid": row["release_evidence_audit_validation_valid"],
    }


def delta(candidate: Any, baseline: Any) -> float | int | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def regression_rate(base: float | None, candidate: float | None, *, higher_is_better: bool) -> float | None:
    if base is None or candidate is None:
        return None
    if math.isclose(base, 0.0, abs_tol=1.0e-12):
        return base - candidate if higher_is_better else candidate - base
    if higher_is_better:
        return (base - candidate) / abs(base)
    return (candidate - base) / abs(base)


def positive_regression(base: float | int | None, candidate: float | int | None) -> float | int | None:
    if base is None or candidate is None:
        return None
    return max(0, base - candidate)


def valid_regressed(base: Any, candidate: Any) -> bool:
    return base is True and candidate is not True


def compare_pipelines(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_stages = set(baseline["stage_names"])
    candidate_stages = set(candidate["stage_names"])
    return {
        "same_name": baseline["name"] == candidate["name"],
        "same_stage_set": baseline_stages == candidate_stages,
        "missing_stages": sorted(baseline_stages - candidate_stages),
        "added_stages": sorted(candidate_stages - baseline_stages),
        "stage_count_delta": candidate["stage_count"] - baseline["stage_count"],
        "stage_elapsed_s_delta": delta(candidate["stage_elapsed_s"], baseline["stage_elapsed_s"]),
        "release_count_delta": delta(candidate["release_count"], baseline["release_count"]),
        "selected_rank_delta": delta(candidate["selected_rank"], baseline["selected_rank"]),
        "loss_delta": delta(candidate["loss"], baseline["loss"]),
        "accuracy_delta": delta(candidate["accuracy"], baseline["accuracy"]),
        "best_throughput_tokens_s_delta": delta(
            candidate["best_throughput_tokens_s"],
            baseline["best_throughput_tokens_s"],
        ),
        "loss_regression_rate": regression_rate(baseline["loss"], candidate["loss"], higher_is_better=False),
        "accuracy_regression": positive_regression(baseline["accuracy"], candidate["accuracy"]),
        "best_throughput_regression_rate": regression_rate(
            baseline["best_throughput_tokens_s"],
            candidate["best_throughput_tokens_s"],
            higher_is_better=True,
        ),
        "same_checkpoint_sha256": baseline["checkpoint_sha256"] == candidate["checkpoint_sha256"],
        "same_archive_sha256": baseline["archive_sha256"] == candidate["archive_sha256"],
        "dashboard_audit_valid_regressed": valid_regressed(
            baseline["release_dashboard_audit_valid"],
            candidate["release_dashboard_audit_valid"],
        ),
        "dashboard_audit_checks_delta": delta(
            candidate["release_dashboard_audit_checks"],
            baseline["release_dashboard_audit_checks"],
        ),
        "dashboard_audit_check_regression": positive_regression(
            baseline["release_dashboard_audit_checks"],
            candidate["release_dashboard_audit_checks"],
        ),
        "memory_prompt_count_delta": delta(candidate["memory_prompt_count"], baseline["memory_prompt_count"]),
        "memory_pair_count_delta": delta(candidate["memory_pair_count"], baseline["memory_pair_count"]),
        "memory_mean_pair_overlap_delta": delta(
            candidate["memory_mean_pair_overlap"],
            baseline["memory_mean_pair_overlap"],
        ),
        "memory_min_pair_overlap_delta": delta(
            candidate["memory_min_pair_overlap"],
            baseline["memory_min_pair_overlap"],
        ),
        "memory_min_pair_state_cosine_similarity_delta": delta(
            candidate["memory_min_pair_state_cosine_similarity"],
            baseline["memory_min_pair_state_cosine_similarity"],
        ),
        "memory_max_pair_state_l2_distance_delta": delta(
            candidate["memory_max_pair_state_l2_distance"],
            baseline["memory_max_pair_state_l2_distance"],
        ),
        "memory_max_pair_state_norm_delta_delta": delta(
            candidate["memory_max_pair_state_norm_delta"],
            baseline["memory_max_pair_state_norm_delta"],
        ),
        "same_has_evidence_audit": baseline["has_evidence_audit"] == candidate["has_evidence_audit"],
        "same_evidence_archive_sha256": (
            baseline["release_evidence_archive_sha256"] == candidate["release_evidence_archive_sha256"]
        ),
        "same_evidence_ledger_chain_head": (
            baseline["release_evidence_ledger_chain_head"] == candidate["release_evidence_ledger_chain_head"]
        ),
        "evidence_audit_valid_regressed": valid_regressed(
            baseline["release_evidence_audit_valid"],
            candidate["release_evidence_audit_valid"],
        ),
        "evidence_audit_validation_valid_regressed": valid_regressed(
            baseline["release_evidence_audit_validation_valid"],
            candidate["release_evidence_audit_validation_valid"],
        ),
        "evidence_audit_checks_delta": delta(
            candidate["release_evidence_audit_checks"],
            baseline["release_evidence_audit_checks"],
        ),
        "evidence_audit_check_regression": positive_regression(
            baseline["release_evidence_audit_checks"],
            candidate["release_evidence_audit_checks"],
        ),
        "baseline_shape_errors": baseline["shape_errors"],
        "candidate_shape_errors": candidate["shape_errors"],
    }


def fail_on_delta(
    failures: list[str],
    comparison: dict[str, Any],
    key: str,
    label: str,
    *,
    allow_missing: bool = False,
) -> None:
    value = comparison.get(key)
    if value is None:
        if not allow_missing:
            failures.append(f"{label} delta could not be computed")
        return
    if value != 0:
        failures.append(f"{label} changed by {value}")


def fail_on_regression(
    failures: list[str],
    comparison: dict[str, Any],
    key: str,
    label: str,
    maximum: float,
) -> None:
    value = comparison.get(key)
    if value is None:
        failures.append(f"{label} regression could not be computed")
    elif float(value) > maximum:
        failures.append(f"{label} regression {float(value):.6f} > {maximum:.6f}")


def threshold_failures(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.fail_on_name_change and not comparison["same_name"]:
        failures.append("pipeline name changed")
    if args.fail_on_stage_set_change and not comparison["same_stage_set"]:
        if comparison["missing_stages"]:
            failures.append(f"missing stages: {comparison['missing_stages']}")
        if comparison["added_stages"]:
            failures.append(f"added stages: {comparison['added_stages']}")
    if args.fail_on_stage_count_change and comparison["stage_count_delta"] != 0:
        failures.append(f"stage_count changed by {comparison['stage_count_delta']}")
    if args.fail_on_release_count_change:
        fail_on_delta(failures, comparison, "release_count_delta", "release_count")
    if args.fail_on_rank_change:
        fail_on_delta(failures, comparison, "selected_rank_delta", "selected_rank")
    if args.fail_on_checkpoint_sha_change and not comparison["same_checkpoint_sha256"]:
        failures.append("checkpoint_sha256 changed")
    if args.fail_on_archive_sha_change and not comparison["same_archive_sha256"]:
        failures.append("archive_sha256 changed")
    if args.fail_on_evidence_presence_change and not comparison["same_has_evidence_audit"]:
        failures.append("evidence audit presence changed")
    if args.fail_on_evidence_archive_sha_change and not comparison["same_evidence_archive_sha256"]:
        failures.append("evidence archive sha256 changed")
    if args.fail_on_ledger_chain_head_change and not comparison["same_evidence_ledger_chain_head"]:
        failures.append("evidence ledger chain head changed")
    if args.fail_on_dashboard_audit_regression:
        if comparison["dashboard_audit_valid_regressed"]:
            failures.append("dashboard audit validity regressed")
        fail_on_regression(
            failures,
            comparison,
            "dashboard_audit_check_regression",
            "dashboard audit check-count",
            args.max_dashboard_audit_check_regression,
        )
    if args.fail_on_evidence_audit_regression:
        if comparison["evidence_audit_valid_regressed"]:
            failures.append("evidence audit validity regressed")
        if comparison["evidence_audit_validation_valid_regressed"]:
            failures.append("evidence audit validation validity regressed")
        fail_on_regression(
            failures,
            comparison,
            "evidence_audit_check_regression",
            "evidence audit check-count",
            args.max_evidence_audit_check_regression,
        )
    if args.fail_on_loss_regression:
        fail_on_regression(failures, comparison, "loss_regression_rate", "loss", args.max_loss_regression)
    if args.fail_on_throughput_regression:
        fail_on_regression(
            failures,
            comparison,
            "best_throughput_regression_rate",
            "best throughput",
            args.max_throughput_regression,
        )
    if args.fail_on_accuracy_regression:
        fail_on_regression(
            failures,
            comparison,
            "accuracy_regression",
            "accuracy",
            args.max_accuracy_regression,
        )
    if args.fail_on_memory_prompt_count_change:
        fail_on_delta(failures, comparison, "memory_prompt_count_delta", "memory_prompt_count")
    if args.fail_on_memory_pair_count_change:
        fail_on_delta(failures, comparison, "memory_pair_count_delta", "memory_pair_count")
    if args.fail_on_memory_overlap_regression:
        overlap_delta = comparison.get("memory_mean_pair_overlap_delta")
        if overlap_delta is None:
            failures.append("memory mean pair overlap regression could not be computed")
        elif -float(overlap_delta) > args.max_memory_overlap_regression:
            failures.append(
                "memory mean pair overlap regression "
                f"{-float(overlap_delta):.6f} > {args.max_memory_overlap_regression:.6f}"
            )
    if args.fail_on_memory_min_overlap_regression:
        overlap_delta = comparison.get("memory_min_pair_overlap_delta")
        if overlap_delta is None:
            failures.append("memory min pair overlap regression could not be computed")
        elif -float(overlap_delta) > args.max_memory_min_overlap_regression:
            failures.append(
                "memory min pair overlap regression "
                f"{-float(overlap_delta):.6f} > {args.max_memory_min_overlap_regression:.6f}"
            )
    if args.fail_on_memory_cosine_regression:
        cosine_delta = comparison.get("memory_min_pair_state_cosine_similarity_delta")
        if cosine_delta is None:
            failures.append("memory min cosine regression could not be computed")
        elif -float(cosine_delta) > args.max_memory_cosine_regression:
            failures.append(
                "memory min cosine regression "
                f"{-float(cosine_delta):.6f} > {args.max_memory_cosine_regression:.6f}"
            )
    if args.fail_on_memory_l2_regression:
        l2_delta = comparison.get("memory_max_pair_state_l2_distance_delta")
        if l2_delta is None:
            failures.append("memory max L2 regression could not be computed")
        elif float(l2_delta) > args.max_memory_l2_regression:
            failures.append(f"memory max L2 regression {float(l2_delta):.6f} > {args.max_memory_l2_regression:.6f}")
    if args.fail_on_memory_norm_regression:
        norm_delta = comparison.get("memory_max_pair_state_norm_delta_delta")
        if norm_delta is None:
            failures.append("memory max norm-delta regression could not be computed")
        elif float(norm_delta) > args.max_memory_norm_regression:
            failures.append(
                "memory max norm-delta regression "
                f"{float(norm_delta):.6f} > {args.max_memory_norm_regression:.6f}"
            )
    if not args.allow_failed_pipeline:
        for label in ["baseline", "candidate"]:
            shape_errors = comparison.get(f"{label}_shape_errors") or []
            if shape_errors:
                failures.append(f"{label} pipeline shape errors: {shape_errors}")
    return failures


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if abs(value_f) >= 1000:
        return f"{value_f:,.0f}"
    return f"{value_f:.{digits}f}"


def print_pipeline(label: str, row: dict[str, Any]) -> None:
    print(label)
    print(f"  path: {row['path']}")
    print(f"  name: {row['name']}")
    print(f"  stages: {row['stage_count']}")
    print(f"  release_count: {row['release_count']}")
    print(f"  selected_rank: {row['selected_rank']}")
    print(f"  loss: {fmt(row['loss'])}")
    print(f"  accuracy: {fmt(row['accuracy'])}")
    print(f"  best_throughput_tokens_s: {fmt(row['best_throughput_tokens_s'], 0)}")
    print(f"  checkpoint_sha256: {str(row['checkpoint_sha256'] or '')[:12]}")
    print(f"  archive_sha256: {str(row['archive_sha256'] or '')[:12]}")
    print(f"  dashboard_audit: {row['release_dashboard_audit_valid']} checks={row['release_dashboard_audit_checks']}")
    print(f"  memory_mean_pair_overlap: {fmt(row['memory_mean_pair_overlap'])}")
    print(
        "  memory_min_pair_state_cosine_similarity: "
        f"{fmt(row['memory_min_pair_state_cosine_similarity'])}"
    )
    print(
        "  evidence_audit: "
        f"{row['release_evidence_audit_valid']} checks={row['release_evidence_audit_checks']}"
    )
    print(f"  evidence_archive_sha256: {str(row['release_evidence_archive_sha256'] or '')[:12]}")
    print(f"  ledger_chain_head: {str(row['release_evidence_ledger_chain_head'] or '')[:12]}")
    if row["shape_errors"]:
        print(f"  shape_errors: {len(row['shape_errors'])}")


def run(args: argparse.Namespace) -> int:
    baseline = load_pipeline(Path(args.baseline_pipeline), args)
    candidate = load_pipeline(Path(args.candidate_pipeline), args)
    comparison = compare_pipelines(baseline, candidate)
    failures = threshold_failures(comparison, args)
    payload = {
        "baseline": compact_pipeline(baseline),
        "candidate": compact_pipeline(candidate),
        "comparison": comparison,
        "failures": failures,
    }

    print("TextPy/SoA release pipeline comparison")
    print("")
    print_pipeline("baseline", baseline)
    print("")
    print_pipeline("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  same_name: {comparison['same_name']}")
    print(f"  same_stage_set: {comparison['same_stage_set']}")
    print(f"  stage_count_delta: {comparison['stage_count_delta']}")
    print(f"  release_count_delta: {comparison['release_count_delta']}")
    print(f"  selected_rank_delta: {comparison['selected_rank_delta']}")
    print(f"  loss_delta: {fmt(comparison['loss_delta'], 6)}")
    print(f"  accuracy_delta: {fmt(comparison['accuracy_delta'], 6)}")
    print(f"  best_throughput_tokens_s_delta: {fmt(comparison['best_throughput_tokens_s_delta'], 0)}")
    print(f"  memory_mean_pair_overlap_delta: {fmt(comparison['memory_mean_pair_overlap_delta'], 6)}")
    print(
        "  memory_min_pair_state_cosine_similarity_delta: "
        f"{fmt(comparison['memory_min_pair_state_cosine_similarity_delta'], 6)}"
    )
    print(f"  dashboard_audit_checks_delta: {comparison['dashboard_audit_checks_delta']}")
    print(f"  evidence_audit_checks_delta: {comparison['evidence_audit_checks_delta']}")
    print(f"  same_evidence_archive_sha256: {comparison['same_evidence_archive_sha256']}")
    print(f"  same_evidence_ledger_chain_head: {comparison['same_evidence_ledger_chain_head']}")

    if args.output_json:
        save_json(args.output_json, payload)
        print("")
        print(f"Saved JSON: {args.output_json}")

    if failures:
        print("")
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("")
    print("RELEASE PIPELINE COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two release pipeline JSON reports.")
    parser.add_argument("--baseline-pipeline", required=True)
    parser.add_argument("--candidate-pipeline", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--allow-failed-pipeline", action="store_true")
    parser.add_argument("--require-evidence-audit", action="store_true")
    parser.add_argument("--fail-on-stage-set-change", action="store_true")
    parser.add_argument("--fail-on-stage-count-change", action="store_true")
    parser.add_argument("--fail-on-name-change", action="store_true")
    parser.add_argument("--fail-on-release-count-change", action="store_true")
    parser.add_argument("--fail-on-rank-change", action="store_true")
    parser.add_argument("--fail-on-checkpoint-sha-change", action="store_true")
    parser.add_argument("--fail-on-archive-sha-change", action="store_true")
    parser.add_argument("--fail-on-evidence-presence-change", action="store_true")
    parser.add_argument("--fail-on-evidence-archive-sha-change", action="store_true")
    parser.add_argument("--fail-on-ledger-chain-head-change", action="store_true")
    parser.add_argument("--fail-on-dashboard-audit-regression", action="store_true")
    parser.add_argument("--max-dashboard-audit-check-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-evidence-audit-regression", action="store_true")
    parser.add_argument("--max-evidence-audit-check-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-loss-regression", action="store_true")
    parser.add_argument("--max-loss-regression", type=float, default=0.01)
    parser.add_argument("--fail-on-throughput-regression", action="store_true")
    parser.add_argument("--max-throughput-regression", type=float, default=0.05)
    parser.add_argument("--fail-on-accuracy-regression", action="store_true")
    parser.add_argument("--max-accuracy-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-memory-prompt-count-change", action="store_true")
    parser.add_argument("--fail-on-memory-pair-count-change", action="store_true")
    parser.add_argument("--fail-on-memory-overlap-regression", action="store_true")
    parser.add_argument("--max-memory-overlap-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-memory-min-overlap-regression", action="store_true")
    parser.add_argument("--max-memory-min-overlap-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-memory-cosine-regression", action="store_true")
    parser.add_argument("--max-memory-cosine-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-memory-l2-regression", action="store_true")
    parser.add_argument("--max-memory-l2-regression", type=float, default=0.0)
    parser.add_argument("--fail-on-memory-norm-regression", action="store_true")
    parser.add_argument("--max-memory-norm-regression", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleasePipelineComparisonError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
