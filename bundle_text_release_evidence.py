#!/usr/bin/env python3
"""Bundle TextPy/SoA release evidence artifacts with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


class ReleaseEvidenceBundleError(RuntimeError):
    pass


DEFAULT_ARTIFACTS = [
    ("gate_summary", "text_release_gates.json"),
    ("release_report", "text_release_report.md"),
    ("dashboard", "text_release_dashboard.json"),
    ("dashboard_validation", "text_release_dashboard_validation.json"),
    ("dashboard_comparison", "text_release_dashboard_comparison.json"),
    ("history", "text_release_history.json"),
    ("history_validation", "text_release_history_validation.json"),
    ("history_analysis", "text_release_history_analysis.json"),
    ("leaderboard_json", "text_release_leaderboard.json"),
    ("leaderboard_csv", "text_release_leaderboard.csv"),
    ("current_release", "current_text_release.json"),
    ("run_json", "run_current_text_release.json"),
    ("memory_suite", "text_memory_suite/memory_suite.json"),
    ("memory_validation", "text_memory_suite/validation.json"),
    ("memory_prompts_csv", "text_memory_suite/prompts.csv"),
    ("memory_pairs_csv", "text_memory_suite/pairs.csv"),
]


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
        raise ReleaseEvidenceBundleError(f"Missing JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceBundleError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseEvidenceBundleError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_artifact(base_dir: Path, value: str | None, fallback: str) -> Path:
    raw = value or str(base_dir / fallback)
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / raw


def archive_name(root_name: str, base_dir: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        rel = Path(path.name)
    return str(Path(root_name) / rel)


def artifact_record(label: str, base_dir: Path, root_name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ReleaseEvidenceBundleError(f"Missing release evidence artifact {label}: {path}")
    if not path.is_file():
        raise ReleaseEvidenceBundleError(f"Release evidence artifact is not a file {label}: {path}")
    return {
        "label": label,
        "path": str(path),
        "archive_name": archive_name(root_name, base_dir, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def collect_artifacts(args: argparse.Namespace, base_dir: Path, root_name: str) -> list[dict[str, Any]]:
    overrides = {
        "gate_summary": args.gates,
        "release_report": args.report_md,
        "dashboard": args.dashboard,
        "dashboard_validation": args.dashboard_validation,
        "dashboard_comparison": args.dashboard_comparison,
        "history": args.history,
        "history_validation": args.history_validation,
        "history_analysis": args.history_analysis,
        "leaderboard_json": args.leaderboard_json,
        "leaderboard_csv": args.leaderboard_csv,
        "current_release": args.current_release,
        "run_json": args.run_json,
        "memory_suite": args.memory_suite,
        "memory_validation": args.memory_validation,
        "memory_prompts_csv": args.memory_prompts_csv,
        "memory_pairs_csv": args.memory_pairs_csv,
    }
    records: list[dict[str, Any]] = []
    seen_archive_names: set[str] = set()
    for label, fallback in DEFAULT_ARTIFACTS:
        path = resolve_artifact(base_dir, overrides[label], fallback)
        record = artifact_record(label, base_dir, root_name, path)
        if record["archive_name"] in seen_archive_names:
            raise ReleaseEvidenceBundleError(f"Duplicate archive path: {record['archive_name']}")
        seen_archive_names.add(record["archive_name"])
        records.append(record)
    return records


def assert_valid_gates(path: Path, allow_failed_gates: bool) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("kind") != "text_release_gate_summary":
        raise ReleaseEvidenceBundleError("Gate summary kind must be text_release_gate_summary.")
    summary = payload.get("summary")
    gates = payload.get("gates")
    if not isinstance(summary, dict) or not isinstance(gates, list):
        raise ReleaseEvidenceBundleError("Gate summary must contain summary and gates.")
    if payload.get("valid") is not True and not allow_failed_gates:
        raise ReleaseEvidenceBundleError("Gate summary is not valid. Use --allow-failed-gates to bundle failed evidence.")
    return payload


def build_manifest(
    *,
    args: argparse.Namespace,
    base_dir: Path,
    root_name: str,
    gates_payload: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = gates_payload.get("summary") or {}
    return {
        "created_at": timestamp(),
        "kind": "text_release_evidence_bundle",
        "valid": gates_payload.get("valid") is True,
        "release_name": summary.get("release_name"),
        "root_name": root_name,
        "base_dir": str(base_dir),
        "archive": str(args.output),
        "archive_sha256": None,
        "archive_bytes": None,
        "file_count": len(records),
        "total_input_bytes": sum(int(item["bytes"]) for item in records),
        "summary": {
            "failed_gate_count": summary.get("failed_gate_count"),
            "benchmark_loss": summary.get("benchmark_loss"),
            "best_throughput_tokens_s": summary.get("best_throughput_tokens_s"),
            "memory_mean_pair_overlap": summary.get("memory_mean_pair_overlap"),
            "history_baseline_mode": summary.get("history_baseline_mode"),
            "history_loss_delta": summary.get("history_loss_delta"),
            "history_memory_overlap_delta": summary.get("history_memory_overlap_delta"),
        },
        "files": records,
    }


def write_bundle(output: Path, root_name: str, records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="text_release_evidence_") as tmp:
        manifest_path = Path(tmp) / "evidence_manifest.json"
        write_json(manifest_path, manifest)
        with tarfile.open(output, "w:gz") as tar:
            for record in records:
                tar.add(Path(record["path"]), arcname=record["archive_name"])
            tar.add(manifest_path, arcname=str(Path(root_name) / "evidence_manifest.json"))


def run(args: argparse.Namespace) -> int:
    base_dir = Path(args.base_dir) if args.base_dir else Path(args.gates).parent
    root_name = args.root_name
    gate_path = resolve_artifact(base_dir, args.gates, "text_release_gates.json")
    gates_payload = assert_valid_gates(gate_path, args.allow_failed_gates)
    args.gates = str(gate_path)
    records = collect_artifacts(args, base_dir, root_name)
    manifest = build_manifest(
        args=args,
        base_dir=base_dir,
        root_name=root_name,
        gates_payload=gates_payload,
        records=records,
    )
    output = Path(args.output)
    write_bundle(output, root_name, records, manifest)
    manifest["archive_sha256"] = sha256_file(output)
    manifest["archive_bytes"] = output.stat().st_size
    write_json(Path(args.output_json), manifest)

    print("TextPy/SoA release evidence bundler")
    print(f"release_name: {manifest['release_name']}")
    print(f"valid: {manifest['valid']}")
    print(f"files: {manifest['file_count']}")
    print(f"archive: {output}")
    print(f"archive_sha256: {manifest['archive_sha256']}")
    print(f"archive_bytes: {manifest['archive_bytes']}")
    print(f"manifest: {args.output_json}")
    print("")
    print("RELEASE EVIDENCE BUNDLE OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle TextPy/SoA release evidence artifacts.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--root-name", default="text_release_evidence")
    parser.add_argument("--gates", default="artifacts/releases/text_release_gates.json")
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--dashboard", default=None)
    parser.add_argument("--dashboard-validation", default=None)
    parser.add_argument("--dashboard-comparison", default=None)
    parser.add_argument("--history", default=None)
    parser.add_argument("--history-validation", default=None)
    parser.add_argument("--history-analysis", default=None)
    parser.add_argument("--leaderboard-json", default=None)
    parser.add_argument("--leaderboard-csv", default=None)
    parser.add_argument("--current-release", default=None)
    parser.add_argument("--run-json", default=None)
    parser.add_argument("--memory-suite", default=None)
    parser.add_argument("--memory-validation", default=None)
    parser.add_argument("--memory-prompts-csv", default=None)
    parser.add_argument("--memory-pairs-csv", default=None)
    parser.add_argument("--output", default="artifacts/releases/text_release_evidence.tar.gz")
    parser.add_argument("--output-json", default="artifacts/releases/text_release_evidence_manifest.json")
    parser.add_argument("--allow-failed-gates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceBundleError as exc:
        raise SystemExit(str(exc)) from exc
