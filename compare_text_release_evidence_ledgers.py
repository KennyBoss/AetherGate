#!/usr/bin/env python3
"""Compare two TextPy/SoA release evidence ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from register_text_release_evidence import ReleaseEvidenceRegistryError
from validate_text_release_evidence_ledger import validate


class ReleaseEvidenceLedgerComparisonError(RuntimeError):
    pass


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_valid_ledger(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    validation_args = argparse.Namespace(
        ledger=str(path),
        output_json=None,
        expected_entries=None,
        require_artifacts=args.require_artifacts,
        artifact_scope=args.artifact_scope,
        fail_on_failed_gates=args.fail_on_failed_gates,
    )
    validation = validate(validation_args)
    if validation.get("valid") is not True:
        raise ReleaseEvidenceLedgerComparisonError(f"Ledger validation failed for {path}: {validation.get('errors')}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
    return {
        "path": str(path),
        "payload": payload,
        "validation": validation,
        "entry_count": len(entries),
        "chain_head": payload.get("chain_head"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "entries": entries,
    }


def compact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": entry.get("index"),
        "name": entry.get("name"),
        "release_name": entry.get("release_name"),
        "entry_hash": entry.get("entry_hash"),
        "previous_hash": entry.get("previous_hash"),
        "archive_sha256": entry.get("archive_sha256"),
        "manifest_sha256": entry.get("manifest_sha256"),
        "file_count": entry.get("file_count"),
        "failed_gate_count": entry.get("failed_gate_count"),
        "benchmark_loss": entry.get("benchmark_loss"),
        "best_throughput_tokens_s": entry.get("best_throughput_tokens_s"),
        "memory_mean_pair_overlap": entry.get("memory_mean_pair_overlap"),
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "entry_count": row["entry_count"],
        "chain_head": row["chain_head"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "artifact_checks": row["validation"].get("artifact_checks"),
        "entries": [compact_entry(entry) for entry in row["entries"]],
    }


def entry_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for entry in row["entries"]:
        if isinstance(entry, dict) and isinstance(entry.get("entry_hash"), str):
            mapped[entry["entry_hash"]] = entry
    return mapped


def artifact_fields(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest": entry.get("manifest"),
        "manifest_sha256": entry.get("manifest_sha256"),
        "archive": entry.get("archive"),
        "archive_sha256": entry.get("archive_sha256"),
        "validation_json": entry.get("validation_json"),
        "validation_sha256": entry.get("validation_sha256"),
        "restore_json": entry.get("restore_json"),
        "restore_sha256": entry.get("restore_sha256"),
        "comparison_json": entry.get("comparison_json"),
        "comparison_sha256": entry.get("comparison_sha256"),
    }


def changed_entry_fields(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    fields = [
        "name",
        "release_name",
        "archive_sha256",
        "manifest_sha256",
        "archive_bytes",
        "file_count",
        "total_input_bytes",
        "failed_gate_count",
        "benchmark_loss",
        "best_throughput_tokens_s",
        "memory_mean_pair_overlap",
        "previous_hash",
    ]
    changed = [field for field in fields if left.get(field) != right.get(field)]
    left_artifacts = artifact_fields(left)
    right_artifacts = artifact_fields(right)
    changed.extend([field for field in left_artifacts if left_artifacts[field] != right_artifacts[field]])
    return sorted(set(changed))


def compare_ledgers(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_entries = entry_map(baseline)
    candidate_entries = entry_map(candidate)
    baseline_hashes = set(baseline_entries)
    candidate_hashes = set(candidate_entries)
    common_hashes = sorted(baseline_hashes & candidate_hashes)
    changed_common = [
        {"entry_hash": entry_hash, "changed_fields": changed_entry_fields(baseline_entries[entry_hash], candidate_entries[entry_hash])}
        for entry_hash in common_hashes
        if changed_entry_fields(baseline_entries[entry_hash], candidate_entries[entry_hash])
    ]
    baseline_archives = {entry.get("archive_sha256") for entry in baseline["entries"] if isinstance(entry, dict)}
    candidate_archives = {entry.get("archive_sha256") for entry in candidate["entries"] if isinstance(entry, dict)}
    baseline_manifests = {entry.get("manifest_sha256") for entry in baseline["entries"] if isinstance(entry, dict)}
    candidate_manifests = {entry.get("manifest_sha256") for entry in candidate["entries"] if isinstance(entry, dict)}
    return {
        "same_chain_head": baseline["chain_head"] == candidate["chain_head"],
        "same_entry_count": baseline["entry_count"] == candidate["entry_count"],
        "entry_count_delta": candidate["entry_count"] - baseline["entry_count"],
        "same_entry_hash_set": baseline_hashes == candidate_hashes,
        "common_entry_hashes": common_hashes,
        "missing_entry_hashes": sorted(baseline_hashes - candidate_hashes),
        "added_entry_hashes": sorted(candidate_hashes - baseline_hashes),
        "changed_common_entries": changed_common,
        "same_archive_sha_set": baseline_archives == candidate_archives,
        "missing_archive_sha256": sorted(sha for sha in baseline_archives - candidate_archives if sha),
        "added_archive_sha256": sorted(sha for sha in candidate_archives - baseline_archives if sha),
        "same_manifest_sha_set": baseline_manifests == candidate_manifests,
        "missing_manifest_sha256": sorted(sha for sha in baseline_manifests - candidate_manifests if sha),
        "added_manifest_sha256": sorted(sha for sha in candidate_manifests - baseline_manifests if sha),
        "baseline_artifact_checks": baseline["validation"].get("artifact_checks"),
        "candidate_artifact_checks": candidate["validation"].get("artifact_checks"),
    }


def threshold_failures(comparison: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if args.fail_on_chain_head_change and not comparison["same_chain_head"]:
        failures.append("chain_head changed")
    if args.fail_on_entry_count_change and not comparison["same_entry_count"]:
        failures.append(f"entry_count changed by {comparison['entry_count_delta']}")
    if args.fail_on_entry_set_change and not comparison["same_entry_hash_set"]:
        if comparison["missing_entry_hashes"]:
            failures.append(f"missing entry hashes: {comparison['missing_entry_hashes']}")
        if comparison["added_entry_hashes"]:
            failures.append(f"added entry hashes: {comparison['added_entry_hashes']}")
    if args.fail_on_entry_field_change and comparison["changed_common_entries"]:
        failures.append(f"changed common entries: {comparison['changed_common_entries']}")
    if args.fail_on_archive_set_change and not comparison["same_archive_sha_set"]:
        failures.append("archive SHA set changed")
    if args.fail_on_manifest_set_change and not comparison["same_manifest_sha_set"]:
        failures.append("manifest SHA set changed")
    return failures


def short(value: Any) -> str:
    if not value:
        return "none"
    return str(value)[:12]


def print_ledger(label: str, row: dict[str, Any]) -> None:
    print(label)
    print(f"  path: {row['path']}")
    print(f"  entries: {row['entry_count']}")
    print(f"  chain_head: {short(row['chain_head'])}")
    print(f"  artifact_checks: {row['validation'].get('artifact_checks')}")


def run(args: argparse.Namespace) -> int:
    baseline = load_valid_ledger(Path(args.baseline_ledger), args)
    candidate = load_valid_ledger(Path(args.candidate_ledger), args)
    comparison = compare_ledgers(baseline, candidate)
    failures = threshold_failures(comparison, args)
    payload = {
        "baseline": compact(baseline),
        "candidate": compact(candidate),
        "comparison": comparison,
        "failures": failures,
    }

    print("TextPy/SoA release evidence ledger comparison")
    print("")
    print_ledger("baseline", baseline)
    print("")
    print_ledger("candidate", candidate)
    print("")
    print("Comparison")
    print(f"  same_chain_head: {comparison['same_chain_head']}")
    print(f"  same_entry_count: {comparison['same_entry_count']}")
    print(f"  entry_count_delta: {comparison['entry_count_delta']}")
    print(f"  same_entry_hash_set: {comparison['same_entry_hash_set']}")
    print(f"  added_entries: {len(comparison['added_entry_hashes'])}")
    print(f"  missing_entries: {len(comparison['missing_entry_hashes'])}")
    print(f"  changed_common_entries: {len(comparison['changed_common_entries'])}")
    print(f"  same_archive_sha_set: {comparison['same_archive_sha_set']}")
    print(f"  same_manifest_sha_set: {comparison['same_manifest_sha_set']}")

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
    print("RELEASE EVIDENCE LEDGER COMPARISON OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two TextPy/SoA release evidence ledgers.")
    parser.add_argument("--baseline-ledger", required=True)
    parser.add_argument("--candidate-ledger", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--artifact-scope", choices=["all", "latest", "none"], default="all")
    parser.add_argument("--fail-on-failed-gates", action="store_true")
    parser.add_argument("--fail-on-chain-head-change", action="store_true")
    parser.add_argument("--fail-on-entry-count-change", action="store_true")
    parser.add_argument("--fail-on-entry-set-change", action="store_true")
    parser.add_argument("--fail-on-entry-field-change", action="store_true")
    parser.add_argument("--fail-on-archive-set-change", action="store_true")
    parser.add_argument("--fail-on-manifest-set-change", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (ReleaseEvidenceLedgerComparisonError, ReleaseEvidenceRegistryError) as exc:
        raise SystemExit(str(exc)) from exc
