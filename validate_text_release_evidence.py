#!/usr/bin/env python3
"""Validate a TextPy/SoA release evidence bundle archive and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


class ReleaseEvidenceValidationError(RuntimeError):
    pass


REQUIRED_LABELS = {
    "gate_summary",
    "release_report",
    "dashboard",
    "dashboard_validation",
    "dashboard_comparison",
    "history",
    "history_validation",
    "history_analysis",
    "leaderboard_json",
    "leaderboard_csv",
    "current_release",
    "run_json",
    "memory_suite",
    "memory_validation",
    "memory_prompts_csv",
    "memory_pairs_csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseEvidenceValidationError(f"Missing evidence manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceValidationError(f"Invalid JSON evidence manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseEvidenceValidationError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_members(archive: Path, root_name: str) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    prefix = f"{root_name}/"
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ReleaseEvidenceValidationError(f"Unsafe archive member path: {name}")
            if not name.startswith(prefix):
                raise ReleaseEvidenceValidationError(f"Archive member outside expected root {root_name!r}: {name}")
            if not member.isfile():
                raise ReleaseEvidenceValidationError(f"Archive member is not a regular file: {name}")
            members.append(member)
    return members


def read_archive_json(archive: Path, member_name: str) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            extracted = tar.extractfile(member_name)
        except KeyError as exc:
            raise ReleaseEvidenceValidationError(f"Archive is missing {member_name}") from exc
        if extracted is None:
            raise ReleaseEvidenceValidationError(f"Archive member is not readable: {member_name}")
        payload = json.loads(extracted.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseEvidenceValidationError(f"Archive member must be JSON object: {member_name}")
    return payload


def read_archive_bytes(archive: Path, member_name: str) -> bytes:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            extracted = tar.extractfile(member_name)
        except KeyError as exc:
            raise ReleaseEvidenceValidationError(f"Archive is missing {member_name}") from exc
        if extracted is None:
            raise ReleaseEvidenceValidationError(f"Archive member is not readable: {member_name}")
        return extracted.read()


def compare_manifest_without_archive_fields(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_clean = dict(left)
    right_clean = dict(right)
    for payload in [left_clean, right_clean]:
        payload["archive_sha256"] = None
        payload["archive_bytes"] = None
    return left_clean == right_clean


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    archive = Path(args.archive or manifest.get("archive", ""))
    if not archive.exists():
        raise ReleaseEvidenceValidationError(f"Missing evidence archive: {archive}")
    errors: list[str] = []

    if manifest.get("kind") != "text_release_evidence_bundle":
        errors.append("kind must be text_release_evidence_bundle.")
    root_name = manifest.get("root_name")
    if not isinstance(root_name, str) or not root_name:
        errors.append("root_name must be a non-empty string.")
        root_name = "text_release_evidence"
    files = manifest.get("files")
    file_records = files if isinstance(files, list) else []
    if not isinstance(files, list):
        errors.append("files must be a list.")
    if int(manifest.get("file_count") or -1) != len(file_records):
        errors.append("file_count must match files length.")
    labels = {item.get("label") for item in file_records if isinstance(item, dict)}
    missing_labels = sorted(REQUIRED_LABELS - labels)
    if missing_labels:
        errors.append(f"missing required labels: {missing_labels}")
    duplicate_archive_names = [
        name
        for name in {item.get("archive_name") for item in file_records if isinstance(item, dict)}
        if sum(1 for item in file_records if isinstance(item, dict) and item.get("archive_name") == name) > 1
    ]
    if duplicate_archive_names:
        errors.append(f"duplicate archive names: {sorted(duplicate_archive_names)}")

    actual_archive_sha = sha256_file(archive)
    actual_archive_bytes = archive.stat().st_size
    if manifest.get("archive_sha256") != actual_archive_sha:
        errors.append("archive_sha256 does not match archive bytes.")
    if int(manifest.get("archive_bytes") or -1) != actual_archive_bytes:
        errors.append("archive_bytes does not match archive size.")

    try:
        members = safe_members(archive, root_name)
    except ReleaseEvidenceValidationError as exc:
        errors.append(str(exc))
        members = []
    member_names = {member.name for member in members}
    expected_names = {str(Path(root_name) / "evidence_manifest.json")}
    for item in file_records:
        if isinstance(item, dict) and isinstance(item.get("archive_name"), str):
            expected_names.add(item["archive_name"])
    missing_members = sorted(expected_names - member_names)
    extra_members = sorted(member_names - expected_names)
    if missing_members:
        errors.append(f"archive missing members: {missing_members}")
    if extra_members:
        errors.append(f"archive has unexpected members: {extra_members}")

    inner_manifest: dict[str, Any] | None = None
    if str(Path(root_name) / "evidence_manifest.json") in member_names:
        try:
            inner_manifest = read_archive_json(archive, str(Path(root_name) / "evidence_manifest.json"))
            if not compare_manifest_without_archive_fields(inner_manifest, manifest):
                errors.append("inner evidence_manifest.json does not match outer manifest source fields.")
        except (ReleaseEvidenceValidationError, json.JSONDecodeError) as exc:
            errors.append(str(exc))

    checked_files = 0
    for item in file_records:
        if not isinstance(item, dict):
            errors.append("file record must be an object.")
            continue
        archive_name = item.get("archive_name")
        if not isinstance(archive_name, str) or archive_name not in member_names:
            continue
        payload = read_archive_bytes(archive, archive_name)
        actual_size = len(payload)
        actual_sha = hashlib.sha256(payload).hexdigest()
        if int(item.get("bytes") or -1) != actual_size:
            errors.append(f"byte mismatch for {archive_name}: {actual_size} != {item.get('bytes')}")
        if item.get("sha256") != actual_sha:
            errors.append(f"sha256 mismatch for {archive_name}: {actual_sha} != {item.get('sha256')}")
        checked_files += 1
        if item.get("label") == "gate_summary":
            try:
                gates = json.loads(payload.decode("utf-8"))
                if gates.get("kind") != "text_release_gate_summary":
                    errors.append("gate_summary kind must be text_release_gate_summary.")
                if gates.get("valid") is not True and not args.allow_failed_gates:
                    errors.append("gate_summary is not valid.")
                summary = gates.get("summary") if isinstance(gates.get("summary"), dict) else {}
                if int(summary.get("failed_gate_count") or 0) != 0 and not args.allow_failed_gates:
                    errors.append("gate_summary has failed gates.")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"gate_summary is not valid JSON: {exc}")

    valid = not errors
    return {
        "valid": valid,
        "archive": str(archive),
        "archive_sha256": actual_archive_sha,
        "archive_bytes": actual_archive_bytes,
        "manifest": str(manifest_path),
        "root_name": root_name,
        "file_count": len(file_records),
        "checked_files": checked_files,
        "member_count": len(members),
        "has_inner_manifest": inner_manifest is not None,
        "errors": errors,
    }


def run(args: argparse.Namespace) -> int:
    result = validate(args)
    if args.output_json:
        write_json(Path(args.output_json), result)
    print("TextPy/SoA release evidence validator")
    print(f"archive: {result['archive']}")
    print(f"valid: {result['valid']}")
    print(f"files: {result['file_count']}")
    print(f"checked_files: {result['checked_files']}")
    print(f"members: {result['member_count']}")
    print(f"archive_sha256: {result['archive_sha256']}")
    if args.output_json:
        print(f"Saved JSON: {args.output_json}")
    if not result["valid"]:
        print("")
        print("FAIL")
        for error in result["errors"]:
            print(f"  {error}")
        return 1
    print("")
    print("RELEASE EVIDENCE VALIDATION OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a TextPy/SoA release evidence bundle.")
    parser.add_argument("--manifest", default="artifacts/releases/text_release_evidence_manifest.json")
    parser.add_argument("--archive", default=None)
    parser.add_argument("--output-json", default="artifacts/releases/text_release_evidence_validation.json")
    parser.add_argument("--allow-failed-gates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceValidationError as exc:
        raise SystemExit(str(exc)) from exc
