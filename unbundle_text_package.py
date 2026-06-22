#!/usr/bin/env python3
"""Safely restore and validate a bundled TextPy/SoA text package."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from validate_text_package import validate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_members(archive: Path, root_name: str) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    prefix = f"{root_name}/"
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"Unsafe archive member path: {name}")
            if not name.startswith(prefix):
                raise SystemExit(f"Archive member outside expected root {root_name!r}: {name}")
            if not member.isfile():
                raise SystemExit(f"Archive member is not a regular file: {name}")
            members.append(member)
    return members


def extract_archive(archive: Path, output_dir: Path, root_name: str) -> Path:
    members = safe_members(archive, root_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(output_dir, members=members)
    return output_dir / root_name


def verify_files(package_dir: Path, bundle_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in bundle_manifest["files"]:
        path = package_dir / item["name"]
        if not path.exists():
            raise SystemExit(f"Missing restored file: {path}")
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_bytes != int(item["bytes"]):
            raise SystemExit(f"Restored byte mismatch for {path}: {actual_bytes} != {item['bytes']}")
        if actual_sha != item["sha256"]:
            raise SystemExit(f"Restored sha256 mismatch for {path}: {actual_sha} != {item['sha256']}")
        results.append(
            {
                "name": item["name"],
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
        )
    return results


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    bundle_manifest_path = Path(args.bundle_manifest)
    bundle_manifest = load_json(bundle_manifest_path)
    expected_archive_sha = bundle_manifest["archive_sha256"]
    actual_archive_sha = sha256_file(archive)
    if actual_archive_sha != expected_archive_sha:
        raise SystemExit(f"Archive sha256 mismatch: {actual_archive_sha} != {expected_archive_sha}")

    root_name = args.root_name or bundle_manifest["root_name"]
    package_dir = extract_archive(archive, Path(args.output_dir), root_name)
    file_results = verify_files(package_dir, bundle_manifest)

    validation_args = argparse.Namespace(
        manifest=str(package_dir / "manifest.json"),
        output_json=None,
        model_card_json="model_card.json",
        model_card_md="MODEL_CARD.md",
        smoke_json="smoke_run.json",
        benchmark_json="benchmark.json",
        require_model_card=True,
        require_smoke=True,
        require_benchmark=True,
    )
    validation = validate(validation_args)
    if not validation["valid"]:
        raise SystemExit(f"Restored package validation failed: {validation['errors']}")

    report = {
        "valid": True,
        "archive": str(archive),
        "archive_sha256": actual_archive_sha,
        "bundle_manifest": str(bundle_manifest_path),
        "output_dir": str(args.output_dir),
        "package_dir": str(package_dir),
        "files": file_results,
        "validation": validation,
    }
    if args.output_json:
        write_json(Path(args.output_json), report)

    print("TextPy/SoA text package unbundler")
    print(f"archive: {archive}")
    print(f"archive_sha256: {actual_archive_sha}")
    print(f"package_dir: {package_dir}")
    print(f"files: {len(file_results)}")
    print(f"valid: {report['valid']}")
    if args.output_json:
        print(f"restore_report: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a bundled text package.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--bundle-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--root-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
