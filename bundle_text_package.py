#!/usr/bin/env python3
"""Bundle a validated promoted TextPy/SoA text package into a tar.gz archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_text_package import validate


DEFAULT_FILES = [
    "manifest.json",
    "text_ssm_promoted.npz",
    "model_card.json",
    "MODEL_CARD.md",
    "smoke_run.json",
    "benchmark.json",
    "validation.json",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def package_files(package_dir: Path, names: list[str]) -> list[Path]:
    files: list[Path] = []
    for name in names:
        path = package_dir / name
        if not path.exists():
            raise SystemExit(f"Missing package file: {path}")
        files.append(path)
    return files


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    package_dir = manifest_path.parent
    validation_args = argparse.Namespace(
        manifest=str(manifest_path),
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
        raise SystemExit(f"Package validation failed: {validation['errors']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root_name = args.root_name or package_dir.name
    file_names = list(DEFAULT_FILES)
    files = package_files(package_dir, file_names)

    with tarfile.open(output, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=f"{root_name}/{path.name}")

    bundle_sha = sha256_file(output)
    bundle_manifest = {
        "created_at": timestamp(),
        "archive": str(output),
        "archive_sha256": bundle_sha,
        "archive_bytes": output.stat().st_size,
        "root_name": root_name,
        "source_manifest": str(manifest_path),
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "validation": validation,
    }
    if args.output_json:
        write_json(Path(args.output_json), bundle_manifest)

    print("TextPy/SoA text package bundler")
    print(f"archive: {output}")
    print(f"archive_sha256: {bundle_sha}")
    print(f"archive_bytes: {output.stat().st_size}")
    print(f"files: {len(files)}")
    if args.output_json:
        print(f"bundle_manifest: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle a promoted text package.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--root-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
