#!/usr/bin/env python3
"""Register validated TextPy/SoA release evidence bundles in a hash-chain ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_text_release_evidence import validate


class ReleaseEvidenceRegistryError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "format_version": 1,
            "kind": "text_release_evidence_ledger",
            "created_at": timestamp(),
            "entries": [],
            "entry_count": 0,
            "chain_head": None,
        }
    ledger = load_json(path)
    ledger.setdefault("format_version", 1)
    ledger.setdefault("kind", "text_release_evidence_ledger")
    ledger.setdefault("created_at", timestamp())
    ledger.setdefault("entries", [])
    ledger.setdefault("entry_count", len(ledger["entries"]))
    ledger.setdefault("chain_head", ledger["entries"][-1]["entry_hash"] if ledger["entries"] else None)
    return ledger


def verify_chain(ledger: dict[str, Any]) -> None:
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ReleaseEvidenceRegistryError("Ledger entries must be a list.")
    previous_hash: str | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ReleaseEvidenceRegistryError("Ledger entry must be an object.")
        expected_previous = previous_hash
        if entry.get("index") != index + 1:
            raise ReleaseEvidenceRegistryError(f"Ledger entry index mismatch at {index}: {entry.get('index')}")
        if entry.get("previous_hash") != expected_previous:
            raise ReleaseEvidenceRegistryError(
                f"Ledger previous_hash mismatch at entry {index + 1}: {entry.get('previous_hash')} != {expected_previous}"
            )
        entry_without_hash = dict(entry)
        entry_hash = entry_without_hash.pop("entry_hash", None)
        calculated_hash = canonical_sha(entry_without_hash)
        if entry_hash != calculated_hash:
            raise ReleaseEvidenceRegistryError(
                f"Ledger entry hash mismatch at entry {index + 1}: {entry_hash} != {calculated_hash}"
            )
        previous_hash = entry_hash
    if ledger.get("chain_head") != previous_hash:
        raise ReleaseEvidenceRegistryError(f"Ledger chain_head mismatch: {ledger.get('chain_head')} != {previous_hash}")
    if int(ledger.get("entry_count") or 0) != len(entries):
        raise ReleaseEvidenceRegistryError(f"Ledger entry_count mismatch: {ledger.get('entry_count')} != {len(entries)}")


def checked_optional_json(path: str | None, required_valid: bool = False) -> tuple[str | None, str | None, dict[str, Any] | None]:
    if not path:
        return None, None, None
    json_path = Path(path)
    payload = load_json(json_path)
    if required_valid and payload.get("valid") is not True:
        raise ReleaseEvidenceRegistryError(f"Report is not valid: {json_path}")
    return str(json_path), sha256_file(json_path), payload


def artifact_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".tar.gz"):
        return ".tar.gz"
    return path.suffix or ".bin"


def snapshot_artifact(source: Path, store_dir: Path, archive_sha: str, label: str) -> tuple[str, str]:
    digest = sha256_file(source)
    target_dir = store_dir / archive_sha[:16]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{label}_{digest[:12]}{artifact_suffix(source)}"
    if target.exists():
        if sha256_file(target) != digest:
            raise ReleaseEvidenceRegistryError(f"Snapshot path collision with different bytes: {target}")
    else:
        shutil.copy2(source, target)
    return str(target), digest


def file_sha(manifest: dict[str, Any], label: str) -> str | None:
    for item in manifest.get("files", []):
        if isinstance(item, dict) and item.get("label") == label:
            return item.get("sha256")
    return None


def build_entry(args: argparse.Namespace, ledger: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    archive = Path(args.archive)
    ledger_path = Path(args.ledger)
    store_dir = Path(args.store_dir) if args.store_dir else ledger_path.parent / "text_release_evidence_store"
    manifest = load_json(manifest_path)
    if manifest.get("kind") != "text_release_evidence_bundle":
        raise ReleaseEvidenceRegistryError("Evidence manifest kind must be text_release_evidence_bundle.")
    validation_args = argparse.Namespace(
        manifest=str(manifest_path),
        archive=str(archive),
        output_json=None,
        allow_failed_gates=args.allow_failed_gates,
    )
    validation = validate(validation_args)
    if validation.get("valid") is not True:
        raise ReleaseEvidenceRegistryError(f"Evidence validation failed: {validation.get('errors')}")
    actual_archive_sha = sha256_file(archive)
    if manifest.get("archive_sha256") != actual_archive_sha:
        raise ReleaseEvidenceRegistryError(
            f"Archive sha256 mismatch: {actual_archive_sha} != {manifest.get('archive_sha256')}"
        )
    if int(manifest.get("archive_bytes") or -1) != archive.stat().st_size:
        raise ReleaseEvidenceRegistryError("Archive byte size does not match evidence manifest.")

    validation_path, validation_sha, validation_payload = checked_optional_json(args.validation_json, required_valid=True)
    restore_path, restore_sha, restore_payload = checked_optional_json(args.restore_json, required_valid=True)
    comparison_path, comparison_sha, comparison_payload = checked_optional_json(args.comparison_json)
    if comparison_payload and comparison_payload.get("failures"):
        raise ReleaseEvidenceRegistryError(f"Evidence comparison has failures: {comparison_payload['failures']}")
    if restore_payload:
        if restore_payload.get("archive_sha256") != actual_archive_sha:
            raise ReleaseEvidenceRegistryError("Restore report archive_sha256 does not match evidence archive.")
        if int(restore_payload.get("file_count") or 0) != int(manifest.get("file_count") or 0):
            raise ReleaseEvidenceRegistryError("Restore report file_count does not match evidence manifest.")

    entries = ledger.get("entries", [])
    previous_hash = entries[-1]["entry_hash"] if entries else None
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    manifest_snapshot, manifest_sha = snapshot_artifact(manifest_path, store_dir, actual_archive_sha, "manifest")
    archive_snapshot, archive_sha = snapshot_artifact(archive, store_dir, actual_archive_sha, "archive")
    validation_snapshot, validation_snapshot_sha = (
        snapshot_artifact(Path(validation_path), store_dir, actual_archive_sha, "validation") if validation_path else (None, None)
    )
    restore_snapshot, restore_snapshot_sha = (
        snapshot_artifact(Path(restore_path), store_dir, actual_archive_sha, "restore") if restore_path else (None, None)
    )
    comparison_snapshot, comparison_snapshot_sha = (
        snapshot_artifact(Path(comparison_path), store_dir, actual_archive_sha, "comparison") if comparison_path else (None, None)
    )
    entry = {
        "index": len(entries) + 1,
        "registered_at": timestamp(),
        "name": args.name or manifest.get("release_name"),
        "release_name": manifest.get("release_name"),
        "source_manifest": str(manifest_path),
        "source_archive": str(archive),
        "store_dir": str(store_dir),
        "manifest": manifest_snapshot,
        "manifest_sha256": manifest_sha,
        "archive": archive_snapshot,
        "archive_sha256": archive_sha,
        "archive_bytes": archive.stat().st_size,
        "root_name": manifest.get("root_name"),
        "valid": manifest.get("valid") is True,
        "file_count": int(manifest.get("file_count") or 0),
        "total_input_bytes": int(manifest.get("total_input_bytes") or 0),
        "failed_gate_count": int(summary.get("failed_gate_count") or 0),
        "benchmark_loss": summary.get("benchmark_loss"),
        "best_throughput_tokens_s": summary.get("best_throughput_tokens_s"),
        "memory_mean_pair_overlap": summary.get("memory_mean_pair_overlap"),
        "gate_summary_sha256": file_sha(manifest, "gate_summary"),
        "release_report_sha256": file_sha(manifest, "release_report"),
        "dashboard_sha256": file_sha(manifest, "dashboard"),
        "memory_suite_sha256": file_sha(manifest, "memory_suite"),
        "source_validation_json": validation_path,
        "source_restore_json": restore_path,
        "source_comparison_json": comparison_path,
        "validation_json": validation_snapshot,
        "validation_sha256": validation_snapshot_sha or validation_sha,
        "restore_json": restore_snapshot,
        "restore_sha256": restore_snapshot_sha or restore_sha,
        "comparison_json": comparison_snapshot,
        "comparison_sha256": comparison_snapshot_sha or comparison_sha,
        "previous_hash": previous_hash,
    }
    entry["entry_hash"] = canonical_sha(entry)
    return entry


def same_evidence(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("archive_sha256") == right.get("archive_sha256")
        and left.get("manifest_sha256") == right.get("manifest_sha256")
    )


def register_entry(ledger: dict[str, Any], entry: dict[str, Any]) -> tuple[dict[str, Any], str]:
    verify_chain(ledger)
    entries = ledger["entries"]
    for index, existing in enumerate(entries):
        if same_evidence(existing, entry):
            replacement = dict(entry)
            replacement["index"] = existing["index"]
            replacement["previous_hash"] = existing.get("previous_hash")
            replacement["entry_hash"] = canonical_sha({k: v for k, v in replacement.items() if k != "entry_hash"})
            entries[index] = replacement
            for next_index in range(index + 1, len(entries)):
                entries[next_index]["previous_hash"] = entries[next_index - 1]["entry_hash"]
                entries[next_index]["entry_hash"] = canonical_sha(
                    {k: v for k, v in entries[next_index].items() if k != "entry_hash"}
                )
            ledger["updated_at"] = timestamp()
            ledger["entry_count"] = len(entries)
            ledger["chain_head"] = entries[-1]["entry_hash"] if entries else None
            verify_chain(ledger)
            return ledger, "updated"
    entries.append(entry)
    ledger["updated_at"] = timestamp()
    ledger["entry_count"] = len(entries)
    ledger["chain_head"] = entry["entry_hash"]
    verify_chain(ledger)
    return ledger, "inserted"


def run(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger)
    ledger = load_ledger(ledger_path)
    entry = build_entry(args, ledger)
    ledger, action = register_entry(ledger, entry)
    write_json(ledger_path, ledger)

    print("TextPy/SoA release evidence ledger")
    print(f"ledger: {ledger_path}")
    print(f"action: {action}")
    print(f"entry_count: {ledger['entry_count']}")
    print(f"name: {entry['name']}")
    print(f"archive_sha256: {entry['archive_sha256']}")
    print(f"entry_hash: {entry['entry_hash']}")
    print(f"chain_head: {ledger['chain_head']}")
    print("")
    print("RELEASE EVIDENCE LEDGER OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a validated release evidence bundle in a hash-chain ledger.")
    parser.add_argument("--manifest", default="artifacts/releases/text_release_evidence_manifest.json")
    parser.add_argument("--archive", default="artifacts/releases/text_release_evidence.tar.gz")
    parser.add_argument("--validation-json", default="artifacts/releases/text_release_evidence_validation.json")
    parser.add_argument("--restore-json", default=None)
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--ledger", default="artifacts/releases/text_release_evidence_ledger.json")
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--allow-failed-gates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceRegistryError as exc:
        raise SystemExit(str(exc)) from exc
