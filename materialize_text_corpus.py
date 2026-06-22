#!/usr/bin/env python3
"""Materialize a text corpus from an ingestion manifest.

This bridges sharded corpus storage with the current training scripts, which
still expect a single --text-file. It lets operators choose a bounded slice of a
larger corpus without rewriting the model runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ingest_text_data import corpus_stats


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_inputs(manifest: dict[str, Any]) -> list[Path]:
    shards = manifest.get("shards") or []
    if shards:
        return [Path(str(item["path"])) for item in shards]
    output_text = manifest.get("output_text")
    if output_text:
        return [Path(str(output_text))]
    raise SystemExit("Manifest has neither shards nor output_text.")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    inputs = resolve_inputs(manifest)
    pieces: list[str] = []
    source_rows: list[dict[str, Any]] = []
    total_bytes = 0
    for path in inputs:
        text = path.read_text(encoding="utf-8")
        raw_bytes = len(text.encode("utf-8"))
        if args.max_bytes and total_bytes + raw_bytes > args.max_bytes:
            remaining = max(0, args.max_bytes - total_bytes)
            if remaining <= 0:
                break
            text = text.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
            raw_bytes = len(text.encode("utf-8"))
        if not text:
            break
        pieces.append(text.strip())
        total_bytes += raw_bytes
        source_rows.append({"path": str(path), "bytes_used": raw_bytes})
        if args.max_bytes and total_bytes >= args.max_bytes:
            break

    if not pieces:
        raise SystemExit("No text was materialized from the manifest.")

    output_text = "\n\n\n".join(pieces).strip() + "\n"
    output_path = Path(args.output_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")
    report = {
        "manifest": str(manifest_path),
        "output_text": str(output_path),
        "max_bytes": args.max_bytes,
        "source_count": len(source_rows),
        "sources": source_rows,
        "stats": corpus_stats(output_text),
        "sha256": sha256_text(output_text),
    }
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_report(report: dict[str, Any]) -> None:
    stats = report["stats"]
    print("TextPy/SoA corpus materializer")
    print(f"manifest: {report['manifest']}")
    print(f"output_text: {report['output_text']}")
    print(f"sources: {report['source_count']}")
    print(f"bytes_utf8: {stats['bytes_utf8']:,}")
    print(f"word_tokens: {stats['word_tokens']:,}")
    print(f"sha256: {report['sha256']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize text from a corpus manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-text", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-bytes", type=int, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    print_report(materialize(args))


if __name__ == "__main__":
    run(parse_args())
