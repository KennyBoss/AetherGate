#!/usr/bin/env python3
"""Register a validated TextPy/SoA text package release in a JSON registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bundle_text_package import sha256_file
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


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"format_version": 1, "created_at": timestamp(), "releases": []}
    registry = load_json(path)
    registry.setdefault("format_version", 1)
    registry.setdefault("created_at", timestamp())
    registry.setdefault("releases", [])
    return registry


def package_file_sha(bundle: dict[str, Any], name: str) -> str | None:
    for item in bundle.get("files", []):
        if item.get("name") == name:
            return item.get("sha256")
    return None


def build_entry(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    bundle_manifest_path = Path(args.bundle_manifest)
    manifest = load_json(manifest_path)
    bundle = load_json(bundle_manifest_path)

    validation_args = argparse.Namespace(
        manifest=str(manifest_path),
        output_json=None,
        model_card_json=args.model_card_json,
        model_card_md=args.model_card_md,
        smoke_json=args.smoke_json,
        benchmark_json=args.benchmark_json,
        require_model_card=True,
        require_smoke=True,
        require_benchmark=True,
    )
    validation = validate(validation_args)
    if not validation["valid"]:
        raise SystemExit(f"Package validation failed: {validation['errors']}")

    archive = Path(args.archive or bundle["archive"])
    if not archive.exists():
        raise SystemExit(f"Package archive does not exist: {archive}")
    archive_sha = sha256_file(archive)
    if archive_sha != bundle["archive_sha256"]:
        raise SystemExit(f"Archive sha256 mismatch: {archive_sha} != {bundle['archive_sha256']}")
    archive_bytes = archive.stat().st_size
    if archive_bytes != int(bundle["archive_bytes"]):
        raise SystemExit(f"Archive byte mismatch: {archive_bytes} != {bundle['archive_bytes']}")

    package_dir = manifest_path.parent
    benchmark = load_json(package_dir / args.benchmark_json)
    model_card = load_json(package_dir / args.model_card_json)
    comparison = load_json(Path(args.comparison_json)) if args.comparison_json else None
    memory_validation = load_json(Path(args.memory_validation_json)) if args.memory_validation_json else None
    if memory_validation and memory_validation.get("valid") is not True:
        raise SystemExit(f"Memory suite validation failed: {memory_validation}")
    memory = None
    if memory_validation:
        memory = {
            "suite": memory_validation.get("suite"),
            "validation": args.memory_validation_json,
            "prompt_count": memory_validation.get("prompt_count"),
            "pair_count": memory_validation.get("pair_count"),
            "has_state_vectors": memory_validation.get("has_state_vectors"),
            "mean_pair_overlap": memory_validation.get("mean_pair_overlap"),
            "min_pair_overlap": memory_validation.get("min_pair_overlap"),
            "max_pair_state_norm_delta": memory_validation.get("max_pair_state_norm_delta"),
            "max_pair_state_l2_distance": memory_validation.get("max_pair_state_l2_distance"),
            "min_pair_state_cosine_similarity": memory_validation.get("min_pair_state_cosine_similarity"),
        }

    selected_run = manifest.get("selected_run", {})
    return {
        "registered_at": timestamp(),
        "name": args.name,
        "manifest": str(manifest_path),
        "bundle_manifest": str(bundle_manifest_path),
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "checkpoint": validation["checkpoint"],
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "checkpoint_bytes": validation["checkpoint_bytes"],
        "checkpoint_role": manifest.get("checkpoint_role"),
        "promoted_loss": manifest.get("promoted_loss"),
        "promoted_accuracy": manifest.get("promoted_accuracy"),
        "benchmark_loss": benchmark.get("loss"),
        "benchmark_accuracy": benchmark.get("accuracy"),
        "benchmark_best_throughput_tokens_s": benchmark.get("best_throughput_tokens_s"),
        "benchmark_mean_throughput_tokens_s": benchmark.get("mean_throughput_tokens_s"),
        "benchmark_repeat": benchmark.get("repeat"),
        "streams": benchmark.get("streams"),
        "tokens_per_stream": benchmark.get("tokens_per_stream"),
        "seq_len": benchmark.get("seq_len"),
        "source_tokens": benchmark.get("source_tokens"),
        "vocab_size": benchmark.get("vocab_size"),
        "param_count": model_card.get("param_count"),
        "param_bytes": model_card.get("param_bytes"),
        "winner_by_loss": selected_run.get("winner_by_loss"),
        "selected_run_dir": selected_run.get("run_dir"),
        "selected_text_file": selected_run.get("text_file"),
        "manifest_sha256": package_file_sha(bundle, "manifest.json"),
        "benchmark_sha256": package_file_sha(bundle, args.benchmark_json),
        "validation_sha256": package_file_sha(bundle, "validation.json"),
        "comparison": comparison.get("comparison") if comparison else None,
        "memory": memory,
    }


def sort_releases(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        releases,
        key=lambda item: (
            float(item.get("benchmark_loss") if item.get("benchmark_loss") is not None else 1.0e30),
            -float(
                item.get("benchmark_best_throughput_tokens_s")
                if item.get("benchmark_best_throughput_tokens_s") is not None
                else 0.0
            ),
            item.get("registered_at", ""),
        ),
    )


def upsert_release(registry: dict[str, Any], entry: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    key = (entry["checkpoint_sha256"], entry["archive_sha256"])
    path_key = (entry.get("manifest"), entry.get("archive"))
    releases = registry["releases"]
    for index, existing in enumerate(releases):
        existing_key = (existing.get("checkpoint_sha256"), existing.get("archive_sha256"))
        existing_path_key = (existing.get("manifest"), existing.get("archive"))
        if existing_key == key or existing_path_key == path_key:
            releases[index] = entry
            registry["updated_at"] = timestamp()
            registry["releases"] = sort_releases(releases)
            return registry, False
    releases.append(entry)
    registry["updated_at"] = timestamp()
    registry["releases"] = sort_releases(releases)
    return registry, True


def run(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    entry = build_entry(args)
    registry, inserted = upsert_release(registry, entry)
    registry["release_count"] = len(registry["releases"])
    registry["best_release"] = registry["releases"][0] if registry["releases"] else None
    write_json(registry_path, registry)

    print("TextPy/SoA text package release registry")
    print(f"registry: {registry_path}")
    print(f"action: {'inserted' if inserted else 'updated'}")
    print(f"release_count: {registry['release_count']}")
    print(f"name: {entry['name']}")
    print(f"checkpoint_sha256: {entry['checkpoint_sha256']}")
    print(f"archive_sha256: {entry['archive_sha256']}")
    print(f"benchmark_loss: {entry['benchmark_loss']:.4f}")
    print(f"benchmark_best_throughput_tokens_s: {entry['benchmark_best_throughput_tokens_s']:,.0f}")
    if entry["comparison"]:
        print(f"comparison_winner_by_loss: {entry['comparison']['winner_by_loss']}")
        print(f"comparison_same_checkpoint_sha256: {entry['comparison']['same_checkpoint_sha256']}")
    if entry["memory"]:
        print(f"memory_mean_pair_overlap: {entry['memory']['mean_pair_overlap']:.4f}")
        print(f"memory_min_pair_state_cosine_similarity: {entry['memory']['min_pair_state_cosine_similarity']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a validated text package release.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bundle-manifest", required=True)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--memory-validation-json", default=None)
    parser.add_argument("--registry", default="artifacts/releases/text_packages.json")
    parser.add_argument("--name", default="text_package")
    parser.add_argument("--model-card-json", default="model_card.json")
    parser.add_argument("--model-card-md", default="MODEL_CARD.md")
    parser.add_argument("--smoke-json", default="smoke_run.json")
    parser.add_argument("--benchmark-json", default="benchmark.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
