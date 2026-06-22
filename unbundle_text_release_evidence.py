#!/usr/bin/env python3
"""Safely restore and validate a bundled TextPy/SoA release evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from validate_text_release_evidence import validate


class ReleaseEvidenceUnbundleError(RuntimeError):
    pass


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
        raise ReleaseEvidenceUnbundleError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceUnbundleError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseEvidenceUnbundleError(f"Expected JSON object: {path}")
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
                raise ReleaseEvidenceUnbundleError(f"Unsafe archive member path: {name}")
            if not name.startswith(prefix):
                raise ReleaseEvidenceUnbundleError(f"Archive member outside expected root {root_name!r}: {name}")
            if not member.isfile():
                raise ReleaseEvidenceUnbundleError(f"Archive member is not a regular file: {name}")
            members.append(member)
    return members


def extract_archive(archive: Path, output_dir: Path, root_name: str, clean: bool) -> Path:
    members = safe_members(archive, root_name)
    evidence_dir = output_dir / root_name
    if clean and evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(output_dir, members=members)
    return evidence_dir


def expected_relative_path(root_name: str, archive_name: str) -> Path:
    path = Path(archive_name)
    expected_root = Path(root_name)
    try:
        return path.relative_to(expected_root)
    except ValueError as exc:
        raise ReleaseEvidenceUnbundleError(
            f"Evidence archive_name {archive_name!r} is outside root {root_name!r}"
        ) from exc


def verify_restored_files(evidence_dir: Path, manifest: dict[str, Any], root_name: str) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ReleaseEvidenceUnbundleError("Evidence manifest files must be a list.")
    results: list[dict[str, Any]] = []
    seen_archive_names: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ReleaseEvidenceUnbundleError("Evidence file record must be an object.")
        label = item.get("label")
        archive_name = item.get("archive_name")
        if not isinstance(label, str) or not isinstance(archive_name, str):
            raise ReleaseEvidenceUnbundleError(f"Evidence file record missing label/archive_name: {item}")
        if archive_name in seen_archive_names:
            raise ReleaseEvidenceUnbundleError(f"Duplicate archive_name in evidence manifest: {archive_name}")
        seen_archive_names.add(archive_name)
        restored_path = evidence_dir / expected_relative_path(root_name, archive_name)
        if not restored_path.exists():
            raise ReleaseEvidenceUnbundleError(f"Missing restored evidence file: {restored_path}")
        if not restored_path.is_file():
            raise ReleaseEvidenceUnbundleError(f"Restored evidence path is not a file: {restored_path}")
        actual_bytes = restored_path.stat().st_size
        actual_sha = sha256_file(restored_path)
        if actual_bytes != int(item.get("bytes") or -1):
            raise ReleaseEvidenceUnbundleError(
                f"Restored byte mismatch for {restored_path}: {actual_bytes} != {item.get('bytes')}"
            )
        if actual_sha != item.get("sha256"):
            raise ReleaseEvidenceUnbundleError(
                f"Restored sha256 mismatch for {restored_path}: {actual_sha} != {item.get('sha256')}"
            )
        results.append(
            {
                "label": label,
                "archive_name": archive_name,
                "path": str(restored_path),
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
        )
    return results


def validate_restored_manifest(manifest_path: Path, archive: Path, allow_failed_gates: bool) -> dict[str, Any]:
    restored_manifest = load_json(manifest_path)
    normalized_manifest = dict(restored_manifest)
    normalized_manifest["archive"] = str(archive)
    normalized_manifest["archive_sha256"] = sha256_file(archive)
    normalized_manifest["archive_bytes"] = archive.stat().st_size
    with TemporaryDirectory(prefix="text_release_evidence_validate_") as tmp:
        normalized_path = Path(tmp) / "evidence_manifest.json"
        write_json(normalized_path, normalized_manifest)
        validation_args = argparse.Namespace(
            manifest=str(normalized_path),
            archive=str(archive),
            output_json=None,
            allow_failed_gates=allow_failed_gates,
        )
        result = validate(validation_args)
    result["manifest"] = str(manifest_path)
    result["normalized_archive_fields"] = True
    result["inner_archive_sha256"] = restored_manifest.get("archive_sha256")
    result["inner_archive_bytes"] = restored_manifest.get("archive_bytes")
    if not result["valid"]:
        raise ReleaseEvidenceUnbundleError(f"Restored evidence validation failed: {result['errors']}")
    return result


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    archive = Path(args.archive)
    manifest = load_json(manifest_path)
    if manifest.get("kind") != "text_release_evidence_bundle":
        raise ReleaseEvidenceUnbundleError("Manifest kind must be text_release_evidence_bundle.")
    root_name = args.root_name or manifest.get("root_name")
    if not isinstance(root_name, str) or not root_name:
        raise ReleaseEvidenceUnbundleError("Manifest root_name must be a non-empty string.")
    expected_archive_sha = manifest.get("archive_sha256")
    actual_archive_sha = sha256_file(archive)
    if expected_archive_sha != actual_archive_sha:
        raise ReleaseEvidenceUnbundleError(f"Archive sha256 mismatch: {actual_archive_sha} != {expected_archive_sha}")

    evidence_dir = extract_archive(archive, Path(args.output_dir), root_name, args.clean)
    restored_manifest_path = evidence_dir / "evidence_manifest.json"
    if not restored_manifest_path.exists():
        raise ReleaseEvidenceUnbundleError(f"Restored evidence manifest is missing: {restored_manifest_path}")
    restored_manifest = load_json(restored_manifest_path)
    if restored_manifest.get("archive_sha256") not in {None, actual_archive_sha}:
        raise ReleaseEvidenceUnbundleError(
            "Restored inner evidence manifest archive_sha256 does not match archive bytes."
        )
    file_results = verify_restored_files(evidence_dir, manifest, root_name)
    validation = validate_restored_manifest(restored_manifest_path, archive, args.allow_failed_gates)

    report = {
        "valid": True,
        "archive": str(archive),
        "archive_sha256": actual_archive_sha,
        "manifest": str(manifest_path),
        "output_dir": str(args.output_dir),
        "evidence_dir": str(evidence_dir),
        "root_name": root_name,
        "file_count": len(file_results),
        "member_count": validation["member_count"],
        "restored_manifest": str(restored_manifest_path),
        "files": file_results,
        "validation": validation,
    }
    if args.output_json:
        write_json(Path(args.output_json), report)

    print("TextPy/SoA release evidence unbundler")
    print(f"archive: {archive}")
    print(f"archive_sha256: {actual_archive_sha}")
    print(f"evidence_dir: {evidence_dir}")
    print(f"files: {len(file_results)}")
    print(f"members: {validation['member_count']}")
    print(f"valid: {report['valid']}")
    if args.output_json:
        print(f"restore_report: {args.output_json}")
    print("")
    print("RELEASE EVIDENCE UNBUNDLE OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a bundled TextPy/SoA release evidence archive.")
    parser.add_argument("--archive", default="artifacts/releases/text_release_evidence.tar.gz")
    parser.add_argument("--manifest", default="artifacts/releases/text_release_evidence_manifest.json")
    parser.add_argument("--output-dir", default="artifacts/restored")
    parser.add_argument("--output-json", default="artifacts/restored/text_release_evidence_restore.json")
    parser.add_argument("--root-name", default=None)
    parser.add_argument("--allow-failed-gates", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceUnbundleError as exc:
        raise SystemExit(str(exc)) from exc
