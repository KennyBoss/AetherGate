#!/usr/bin/env python3
"""Compare two TextPy/SoA release evidence bundle manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_text_release_evidence import validate


class ReleaseEvidenceComparisonError(RuntimeError):
    pass


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_valid_manifest(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    validation_args = argparse.Namespace(
        manifest=str(path),
        archive=None,
        output_json=None,
        allow_failed_gates=args.allow_failed_gates,
    )
    validation = validate(validation_args)
    if not validation["valid"]:
        raise ReleaseEvidenceComparisonError(f"Evidence validation failed for {path}: {validation['errors']}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "payload": payload,
        "validation": validation,
        "release_name": payload.get("release_name"),
        "valid": payload.get("valid") is True,
        "root_name": payload.get("root_name"),
        "archive": payload.get("archive"),
        "archive_sha256": payload.get("archive_sha256"),
        "archive_bytes": int(payload.get("archive_bytes") or 0),
        "file_count": int(payload.get("file_count") or 0),
        "total_input_bytes": int(payload.get("total_input_bytes") or 0),
        "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
        "files": payload.get("files") if isinstance(payload.get("files"), list) else [],
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    summary = row["summary"]
    return {
        "path": row["path"],
        "release_name": row["release_name"],
        "valid": row["valid"],
        "root_name": row["root_name"],
        "archive": row["archive"],
        "archive_sha256": row["archive_sha256"],
        "archive_bytes": row["archive_bytes"],
        "file_count": row["file_count"],
        "total_input_bytes": row["total_input_bytes"],
        "failed_gate_count": summary.get("failed_gate_count"),
        "benchmark_loss": summary.get("benchmark_loss"),
        "best_throughput_tokens_s": summary.get("best_throughput_tokens_s"),
        "memory_mean_pair_overlap": summary.get("memory_mean_pair_overlap"),
    }


def file_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for item in row["files"]:
        if isinstance(item, dict) and isinstance(item.get("label"), str):
            mapped[item["label"]] = item
    return mapped


def compare_files(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_files = file_map(baseline)
    candidate_files = file_map(candidate)
    baseline_labels = set(baseline_files)
    candidate_labels = set(candidate_files)
    common = sorted(baseline_labels & candidate_labels)
    changed_sha = [
        label
        for label in common
        if baseline_files[label].get("sha256") != candidate_files[label].get("sha256")
    ]
    changed_bytes = [
        label
        for label in common
        if int(baseline_files[label].get("bytes") or 0) != int(candidate_files[label].get("bytes") or 0)
    ]
    changed_archive_names = [
        label
        for label in common
        if baseline_files[label].get("archive_name") != candidate_files[label].get("archive_name")
    ]
    return {
        "common_labels": common,
        "missing_labels": sorted(baseline_labels - candidate_labels),
        "added_labels": sorted(candidate_labels - baseline_labels),
        "changed_sha_labels": changed_sha,
        "changed_byte_labels": changed_bytes,
        "changed_archive_name_labels": changed_archive_names,
        "same_file_set": baseline_labels == candidate_labels,
        "same_file_sha": not changed_sha,
        "same_file_bytes": not changed_bytes,
        "same_archive_names": not changed_archive_names,
    }


def compare_manifests(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    file_comparison = compare_files(baseline, candidate)
    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    return {
        "same_release_name": baseline["release_name"] == candidate["release_name"],
        "same_valid": baseline["valid"] == candidate["valid"],
        "same_archive_sha256": baseline["archive_sha256"] == candidate["archive_sha256"],
        "archive_bytes_delta": candidate["archive_bytes"] - baseline["archive_bytes"],
        "file_count_delta": candidate["file_count"] - baseline["file_count"],
        "total_input_bytes_delta": candidate["total_input_bytes"] - baseline["total_input_bytes"],
        "failed_gate_count_delta": int(candidate_summary.get("failed_gate_count") or 0)
        - int(baseline_summary.get("failed_gate_count") or 0),
        "benchmark_loss_delta": float(candidate_summary.get("benchmark_loss") or 0.0)
        - float(baseline_summary.get("benchmark_loss") or 0.0),
        "best_throughput_tokens_s_delta": float(candidate_summary.get("best_throughput_tokens_s") or 0.0)
        - float(baseline_summary.get("best_throughput_tokens_s") or 0.0),
        "memory_mean_pair_overlap_delta": float(candidate_summary.get("memory_mean_pair_overlap") or 0.0)
        - float(baseline_summary.get("memory_mean_pair_overlap") or 0.0),
        **file_comparison,
    }


def threshold_failures(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.fail_on_archive_sha_change and not comparison["same_archive_sha256"]:
        failures.append("archive sha256 changed")
    if args.fail_on_file_set_change and not comparison["same_file_set"]:
        if comparison["missing_labels"]:
            failures.append(f"missing evidence labels: {comparison['missing_labels']}")
        if comparison["added_labels"]:
            failures.append(f"added evidence labels: {comparison['added_labels']}")
    if args.fail_on_file_sha_change and not comparison["same_file_sha"]:
        failures.append(f"changed evidence SHA labels: {comparison['changed_sha_labels']}")
    if args.fail_on_file_byte_change and not comparison["same_file_bytes"]:
        failures.append(f"changed evidence byte labels: {comparison['changed_byte_labels']}")
    if args.fail_on_archive_name_change and not comparison["same_archive_names"]:
        failures.append(f"changed archive-name labels: {comparison['changed_archive_name_labels']}")
    if args.fail_on_validity_change and not comparison["same_valid"]:
        failures.append("evidence validity changed")
    if args.fail_on_gate_regression and comparison["failed_gate_count_delta"] > args.max_failed_gate_regression:
        failures.append(
            f"failed gate count regression {comparison['failed_gate_count_delta']} > {args.max_failed_gate_regression}"
        )
    return failures


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    if abs(value_f) >= 1000:
        return f"{value_f:,.0f}"
    return f"{value_f:.{digits}f}"


def print_manifest(label: str, row: dict[str, Any]) -> None:
    print(label)
    print(f"  path: {row['path']}")
    print(f"  release_name: {row['release_name']}")
    print(f"  valid: {row['valid']}")
    print(f"  archive_sha256: {str(row['archive_sha256'] or '')[:12]}")
    print(f"  archive_bytes: {row['archive_bytes']:,}")
    print(f"  files: {row['file_count']}")
    print(f"  benchmark_loss: {fmt(row['summary'].get('benchmark_loss'))}")
    print(f"  best_throughput_tokens_s: {fmt(row['summary'].get('best_throughput_tokens_s'), 0)}")


def run(args: argparse.Namespace) -> int:
    baseline = load_valid_manifest(Path(args.baseline_manifest), args)
    candidate = load_valid_manifest(Path(args.candidate_manifest), args)
    comparison = compare_manifests(baseline, candidate)
    failures = threshold_failures(comparison, args)
    payload = {
        "baseline": compact(baseline),
        "candidate": compact(candidate),
        "comparison": comparison,
        "failures": failures,
    }

    print("TextPy/SoA release evidence comparison")
    print("")
    print_manifest("baseline", baseline)
    print("")
    print_manifest("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  same_archive_sha256: {comparison['same_archive_sha256']}")
    print(f"  same_file_set: {comparison['same_file_set']}")
    print(f"  same_file_sha: {comparison['same_file_sha']}")
    print(f"  same_file_bytes: {comparison['same_file_bytes']}")
    print(f"  file_count_delta: {comparison['file_count_delta']}")
    print(f"  archive_bytes_delta: {comparison['archive_bytes_delta']}")
    print(f"  failed_gate_count_delta: {comparison['failed_gate_count_delta']}")
    print(f"  benchmark_loss_delta: {comparison['benchmark_loss_delta']:.6f}")

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
    print("RELEASE EVIDENCE COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two TextPy/SoA release evidence manifests.")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--allow-failed-gates", action="store_true")
    parser.add_argument("--fail-on-archive-sha-change", action="store_true")
    parser.add_argument("--fail-on-file-set-change", action="store_true")
    parser.add_argument("--fail-on-file-sha-change", action="store_true")
    parser.add_argument("--fail-on-file-byte-change", action="store_true")
    parser.add_argument("--fail-on-archive-name-change", action="store_true")
    parser.add_argument("--fail-on-validity-change", action="store_true")
    parser.add_argument("--fail-on-gate-regression", action="store_true")
    parser.add_argument("--max-failed-gate-regression", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceComparisonError as exc:
        raise SystemExit(str(exc)) from exc
