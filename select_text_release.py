#!/usr/bin/env python3
"""Select the active TextPy/SoA text package release from a registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bundle_text_package import sha256_file
from list_text_releases import sort_releases
from validate_text_package import validate


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def find_release(args: argparse.Namespace, releases: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    ranked = sort_releases(releases, args.sort)
    if args.checkpoint_sha256:
        for index, release in enumerate(ranked, 1):
            if str(release.get("checkpoint_sha256", "")).startswith(args.checkpoint_sha256):
                return index, release
        raise SystemExit(f"No release with checkpoint SHA prefix: {args.checkpoint_sha256}")
    if args.archive_sha256:
        for index, release in enumerate(ranked, 1):
            if str(release.get("archive_sha256", "")).startswith(args.archive_sha256):
                return index, release
        raise SystemExit(f"No release with archive SHA prefix: {args.archive_sha256}")
    if args.name:
        for index, release in enumerate(ranked, 1):
            if release.get("name") == args.name:
                return index, release
        raise SystemExit(f"No release named: {args.name}")
    if args.rank < 1 or args.rank > len(ranked):
        raise SystemExit(f"--rank must be between 1 and {len(ranked)}.")
    return args.rank, ranked[args.rank - 1]


def verify_release(release: dict[str, Any]) -> dict[str, Any]:
    archive = Path(release["archive"])
    if not archive.exists():
        raise SystemExit(f"Release archive does not exist: {archive}")
    actual_archive_sha = sha256_file(archive)
    if actual_archive_sha != release["archive_sha256"]:
        raise SystemExit(f"Archive sha256 mismatch: {actual_archive_sha} != {release['archive_sha256']}")
    actual_archive_bytes = archive.stat().st_size
    if actual_archive_bytes != int(release["archive_bytes"]):
        raise SystemExit(f"Archive byte mismatch: {actual_archive_bytes} != {release['archive_bytes']}")

    manifest = Path(release["manifest"])
    validation_args = argparse.Namespace(
        manifest=str(manifest),
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
        raise SystemExit(f"Selected package validation failed: {validation['errors']}")
    if validation["checkpoint_sha256"] != release["checkpoint_sha256"]:
        raise SystemExit(
            f"Checkpoint sha256 mismatch: {validation['checkpoint_sha256']} != {release['checkpoint_sha256']}"
        )
    return validation


def run(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    registry = load_json(registry_path)
    releases = list(registry.get("releases", []))
    if not releases:
        raise SystemExit(f"No releases in registry: {registry_path}")
    rank, release = find_release(args, releases)
    validation = verify_release(release)

    selected = {
        "selected_at": timestamp(),
        "registry": str(registry_path),
        "rank": rank,
        "sort": args.sort,
        "name": release.get("name"),
        "manifest": release["manifest"],
        "archive": release["archive"],
        "bundle_manifest": release.get("bundle_manifest"),
        "checkpoint": validation["checkpoint"],
        "checkpoint_sha256": release["checkpoint_sha256"],
        "archive_sha256": release["archive_sha256"],
        "benchmark_loss": release.get("benchmark_loss"),
        "benchmark_accuracy": release.get("benchmark_accuracy"),
        "benchmark_best_throughput_tokens_s": release.get("benchmark_best_throughput_tokens_s"),
        "benchmark_mean_throughput_tokens_s": release.get("benchmark_mean_throughput_tokens_s"),
        "param_count": release.get("param_count"),
        "comparison": release.get("comparison"),
        "memory": release.get("memory"),
        "validation": validation,
    }
    write_json(Path(args.output_json), selected)

    print("TextPy/SoA active text release selector")
    print(f"registry: {registry_path}")
    print(f"selected_rank: {rank}")
    print(f"name: {selected['name']}")
    print(f"manifest: {selected['manifest']}")
    print(f"archive_sha256: {selected['archive_sha256']}")
    print(f"checkpoint_sha256: {selected['checkpoint_sha256']}")
    print(f"benchmark_loss: {selected['benchmark_loss']:.4f}")
    print(f"benchmark_best_throughput_tokens_s: {selected['benchmark_best_throughput_tokens_s']:,.0f}")
    print(f"selected_release: {args.output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select an active text package release.")
    parser.add_argument("--registry", default="artifacts/releases/text_packages.json")
    parser.add_argument("--output-json", default="artifacts/releases/current_text_release.json")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sort", default="benchmark_loss")
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--archive-sha256", default=None)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
