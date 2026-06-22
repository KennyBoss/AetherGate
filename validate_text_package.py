#!/usr/bin/env python3
"""Validate integrity of a promoted TextPy/SoA text package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    local_by_name = base / path.name
    if local_by_name.exists():
        return local_by_name
    if path.is_absolute() or path.exists():
        return path
    return local_by_name


def same_resolved_path(base: Path, recorded: str, expected: Path) -> bool:
    return resolve_path(base, recorded).resolve() == expected.resolve()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    package_dir = manifest_path.parent
    manifest = load_json(manifest_path)
    checkpoint = resolve_path(package_dir, manifest["promoted_checkpoint"])
    errors: list[str] = []
    actual_bytes: int | None = None
    actual_sha: str | None = None

    if not checkpoint.exists():
        errors.append(f"Missing checkpoint: {checkpoint}")
    else:
        actual_bytes = checkpoint.stat().st_size
        expected_bytes = manifest.get("promoted_checkpoint_bytes")
        if expected_bytes is not None and int(expected_bytes) != actual_bytes:
            errors.append(f"Checkpoint byte size mismatch: {actual_bytes} != {expected_bytes}")
        expected_sha = manifest.get("promoted_checkpoint_sha256")
        if expected_sha:
            actual_sha = sha256_file(checkpoint)
            if actual_sha != expected_sha:
                errors.append(f"Checkpoint sha256 mismatch: {actual_sha} != {expected_sha}")
        else:
            actual_sha = sha256_file(checkpoint)

    model_card_json = package_dir / args.model_card_json
    model_card_md = package_dir / args.model_card_md
    smoke_json = package_dir / args.smoke_json
    benchmark_json = package_dir / args.benchmark_json

    model_card: dict[str, Any] | None = None
    if model_card_json.exists():
        model_card = load_json(model_card_json)
        if not same_resolved_path(package_dir, str(model_card.get("checkpoint", "")), checkpoint):
            errors.append("Model card checkpoint path does not match manifest checkpoint.")
        if int(model_card.get("param_count", 0)) <= 0:
            errors.append("Model card has invalid param_count.")
    elif args.require_model_card:
        errors.append(f"Missing model card JSON: {model_card_json}")

    if model_card_md.exists():
        text = model_card_md.read_text(encoding="utf-8")
        for marker in ["# TextPy/SoA Text SSM Model Card", "## Parameter Shapes"]:
            if marker not in text:
                errors.append(f"Model card Markdown missing marker: {marker}")
    elif args.require_model_card:
        errors.append(f"Missing model card Markdown: {model_card_md}")

    smoke: dict[str, Any] | None = None
    if smoke_json.exists():
        smoke = load_json(smoke_json)
        if not same_resolved_path(package_dir, str(smoke.get("checkpoint", "")), checkpoint):
            errors.append("Smoke run checkpoint path does not match manifest checkpoint.")
        if float(smoke.get("loss", 0.0)) <= 0:
            errors.append("Smoke run has invalid loss.")
    elif args.require_smoke:
        errors.append(f"Missing smoke run JSON: {smoke_json}")

    benchmark: dict[str, Any] | None = None
    if benchmark_json.exists():
        benchmark = load_json(benchmark_json)
        if not same_resolved_path(package_dir, str(benchmark.get("checkpoint", "")), checkpoint):
            errors.append("Benchmark checkpoint path does not match manifest checkpoint.")
        integrity = benchmark.get("checkpoint_integrity") or {}
        if integrity.get("matches_manifest") is not True:
            errors.append("Benchmark did not record a successful manifest integrity check.")
        if actual_bytes is not None and int(integrity.get("bytes", -1)) != actual_bytes:
            errors.append("Benchmark checkpoint byte size does not match package checkpoint.")
        if actual_sha is not None and integrity.get("sha256") != actual_sha:
            errors.append("Benchmark checkpoint sha256 does not match package checkpoint.")
        if float(benchmark.get("loss", 0.0)) <= 0:
            errors.append("Benchmark has invalid loss.")
        if float(benchmark.get("best_throughput_tokens_s", 0.0)) <= 0:
            errors.append("Benchmark has invalid throughput.")
        if int(benchmark.get("repeat", 0)) <= 0:
            errors.append("Benchmark has invalid repeat count.")
    elif args.require_benchmark:
        errors.append(f"Missing benchmark JSON: {benchmark_json}")

    return {
        "valid": not errors,
        "errors": errors,
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": actual_bytes,
        "checkpoint_sha256": actual_sha,
        "has_model_card_json": model_card_json.exists(),
        "has_model_card_md": model_card_md.exists(),
        "has_smoke_json": smoke_json.exists(),
        "has_benchmark_json": benchmark_json.exists(),
        "param_count": model_card.get("param_count") if model_card else None,
        "smoke_loss": smoke.get("loss") if smoke else None,
        "benchmark_loss": benchmark.get("loss") if benchmark else None,
        "benchmark_best_throughput_tokens_s": benchmark.get("best_throughput_tokens_s") if benchmark else None,
    }


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace) -> None:
    result = validate(args)
    print("TextPy/SoA text package validator")
    print(f"manifest: {result['manifest']}")
    print(f"checkpoint: {result['checkpoint']}")
    print(f"valid: {result['valid']}")
    print(f"checkpoint_sha256: {result['checkpoint_sha256']}")
    print(f"checkpoint_bytes: {result['checkpoint_bytes']}")
    if result["param_count"] is not None:
        print(f"param_count: {result['param_count']:,}")
    if result["smoke_loss"] is not None:
        print(f"smoke_loss: {result['smoke_loss']:.4f}")
    if result["benchmark_best_throughput_tokens_s"] is not None:
        print(f"benchmark_best_throughput_tokens_s: {result['benchmark_best_throughput_tokens_s']:,.0f}")
    if result["benchmark_loss"] is not None:
        print(f"benchmark_loss: {result['benchmark_loss']:.4f}")
    if result["errors"]:
        print("Errors")
        for error in result["errors"]:
            print(f"  {error}")
    if args.output_json:
        save_json(args.output_json, result)
        print("")
        print(f"Saved JSON: {args.output_json}")
    if not result["valid"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a promoted text package.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--model-card-json", default="model_card.json")
    parser.add_argument("--model-card-md", default="MODEL_CARD.md")
    parser.add_argument("--smoke-json", default="smoke_run.json")
    parser.add_argument("--benchmark-json", default="benchmark.json")
    parser.add_argument("--require-model-card", action="store_true")
    parser.add_argument("--require-smoke", action="store_true")
    parser.add_argument("--require-benchmark", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
