#!/usr/bin/env python3
"""Validate a TextPy/SoA release evidence hash-chain ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from register_text_release_evidence import ReleaseEvidenceRegistryError, sha256_file, verify_chain


REQUIRED_ENTRY_KEYS = {
    "index",
    "registered_at",
    "name",
    "release_name",
    "manifest",
    "manifest_sha256",
    "archive",
    "archive_sha256",
    "archive_bytes",
    "root_name",
    "valid",
    "file_count",
    "total_input_bytes",
    "failed_gate_count",
    "benchmark_loss",
    "best_throughput_tokens_s",
    "memory_mean_pair_overlap",
    "previous_hash",
    "entry_hash",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseEvidenceRegistryError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceRegistryError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseEvidenceRegistryError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_path(base: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    local_by_name = base / path.name
    if local_by_name.exists():
        return local_by_name
    return path


def check_file_sha(path: Path, expected_sha: Any, errors: list[str], label: str) -> bool:
    if not isinstance(expected_sha, str) or not expected_sha:
        errors.append(f"{label} expected SHA must be a non-empty string.")
        return False
    if not path.exists():
        errors.append(f"{label} is missing: {path}")
        return False
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        errors.append(f"{label} SHA mismatch: {actual_sha} != {expected_sha}")
        return False
    return True


def validate_entry_shape(entry: Any, index: int, errors: list[str]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        errors.append(f"entries[{index}] must be an object.")
        return {}
    missing = sorted(REQUIRED_ENTRY_KEYS - set(entry))
    if missing:
        errors.append(f"entries[{index}] missing keys: {missing}")
    if entry.get("index") != index + 1:
        errors.append(f"entries[{index}].index must be {index + 1}.")
    for key in ["registered_at", "name", "release_name", "root_name", "entry_hash"]:
        if not isinstance(entry.get(key), str) or not entry.get(key):
            errors.append(f"entries[{index}].{key} must be a non-empty string.")
    for key in ["manifest_sha256", "archive_sha256"]:
        value = entry.get(key)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"entries[{index}].{key} must be a full SHA256 string.")
    for key in ["archive_bytes", "file_count", "total_input_bytes", "failed_gate_count"]:
        if isinstance(entry.get(key), bool) or not isinstance(entry.get(key), int):
            errors.append(f"entries[{index}].{key} must be an integer.")
    if entry.get("valid") is not True:
        errors.append(f"entries[{index}].valid must be true.")
    return entry


def validate_entry_artifacts(base: Path, entry: dict[str, Any], index: int, errors: list[str]) -> int:
    checked = 0
    manifest = resolve_path(base, entry.get("manifest"))
    archive = resolve_path(base, entry.get("archive"))
    if manifest is None:
        errors.append(f"entries[{index}].manifest path is missing.")
    else:
        checked += int(check_file_sha(manifest, entry.get("manifest_sha256"), errors, f"entries[{index}].manifest"))
        if manifest and manifest.exists():
            payload = load_json(manifest)
            if payload.get("kind") != "text_release_evidence_bundle":
                errors.append(f"entries[{index}].manifest kind must be text_release_evidence_bundle.")
            if payload.get("archive_sha256") != entry.get("archive_sha256"):
                errors.append(f"entries[{index}].manifest archive_sha256 does not match ledger entry.")
            if int(payload.get("file_count") or -1) != int(entry.get("file_count") or -2):
                errors.append(f"entries[{index}].manifest file_count does not match ledger entry.")
    if archive is None:
        errors.append(f"entries[{index}].archive path is missing.")
    else:
        checked += int(check_file_sha(archive, entry.get("archive_sha256"), errors, f"entries[{index}].archive"))
        if archive.exists() and archive.stat().st_size != int(entry.get("archive_bytes") or -1):
            errors.append(f"entries[{index}].archive byte size does not match ledger entry.")
    optional_files = [
        ("validation_json", "validation_sha256"),
        ("restore_json", "restore_sha256"),
        ("comparison_json", "comparison_sha256"),
    ]
    for path_key, sha_key in optional_files:
        value = entry.get(path_key)
        if value is None:
            continue
        path = resolve_path(base, value)
        if path is None:
            errors.append(f"entries[{index}].{path_key} path is missing.")
            continue
        checked += int(check_file_sha(path, entry.get(sha_key), errors, f"entries[{index}].{path_key}"))
    return checked


def validate(args: argparse.Namespace) -> dict[str, Any]:
    ledger_path = Path(args.ledger)
    base = ledger_path.parent
    ledger = load_json(ledger_path)
    errors: list[str] = []
    artifact_checks = 0

    if ledger.get("kind") != "text_release_evidence_ledger":
        errors.append("kind must be text_release_evidence_ledger.")
    entries = ledger.get("entries")
    rows = entries if isinstance(entries, list) else []
    if not isinstance(entries, list):
        errors.append("entries must be a list.")
    if int(ledger.get("entry_count") or -1) != len(rows):
        errors.append("entry_count must match entries length.")
    if args.expected_entries is not None and len(rows) != args.expected_entries:
        errors.append(f"entry count {len(rows)} does not match expected {args.expected_entries}.")
    try:
        verify_chain(ledger)
    except ReleaseEvidenceRegistryError as exc:
        errors.append(str(exc))

    checked_entries: list[dict[str, Any]] = []
    artifact_indexes: set[int] = set()
    if args.require_artifacts:
        if args.artifact_scope == "all":
            artifact_indexes = set(range(len(rows)))
        elif args.artifact_scope == "latest" and rows:
            artifact_indexes = {len(rows) - 1}

    for index, raw_entry in enumerate(rows):
        entry = validate_entry_shape(raw_entry, index, errors)
        if not entry:
            continue
        checked_entries.append(entry)
        if args.fail_on_failed_gates and int(entry.get("failed_gate_count") or 0) > 0:
            errors.append(f"entries[{index}].failed_gate_count is greater than zero.")
        if index in artifact_indexes:
            artifact_checks += validate_entry_artifacts(base, entry, index, errors)

    valid = not errors
    return {
        "valid": valid,
        "ledger": str(ledger_path),
        "entry_count": len(rows),
        "chain_head": ledger.get("chain_head"),
        "first_entry_hash": rows[0].get("entry_hash") if rows and isinstance(rows[0], dict) else None,
        "last_entry_hash": rows[-1].get("entry_hash") if rows and isinstance(rows[-1], dict) else None,
        "artifact_checks": artifact_checks,
        "errors": errors,
    }


def run(args: argparse.Namespace) -> int:
    result = validate(args)
    if args.output_json:
        write_json(Path(args.output_json), result)
    print("TextPy/SoA release evidence ledger validator")
    print(f"ledger: {result['ledger']}")
    print(f"valid: {result['valid']}")
    print(f"entries: {result['entry_count']}")
    print(f"chain_head: {result['chain_head']}")
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
    print("RELEASE EVIDENCE LEDGER VALIDATION OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a TextPy/SoA release evidence hash-chain ledger.")
    parser.add_argument("--ledger", default="artifacts/releases/text_release_evidence_ledger.json")
    parser.add_argument("--output-json", default="artifacts/releases/text_release_evidence_ledger_validation.json")
    parser.add_argument("--expected-entries", type=int, default=None)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--artifact-scope", choices=["all", "latest", "none"], default="all")
    parser.add_argument("--fail-on-failed-gates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceRegistryError as exc:
        raise SystemExit(str(exc)) from exc
