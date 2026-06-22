#!/usr/bin/env python3
"""Replay release evidence ledger registration semantics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from bundle_text_release_evidence import (
    ReleaseEvidenceBundleError,
    assert_valid_gates,
    build_manifest,
    collect_artifacts,
    resolve_artifact,
    sha256_file as bundle_sha256_file,
    write_bundle,
)
from register_text_release_evidence import (
    ReleaseEvidenceRegistryError,
    build_entry,
    load_ledger,
    register_entry,
    verify_chain,
    write_json as write_ledger_json,
)
from validate_text_release_evidence import ReleaseEvidenceValidationError, validate as validate_evidence
from validate_text_release_evidence_ledger import validate as validate_ledger


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_outputs(args: argparse.Namespace, store_dir: Path, work_dir: Path) -> None:
    targets = [Path(args.ledger), store_dir, work_dir]
    if args.output_json:
        targets.append(Path(args.output_json))
    for target in targets:
        remove_path(target)


def ensure_clean_outputs(args: argparse.Namespace, store_dir: Path, work_dir: Path) -> None:
    existing = [path for path in [Path(args.ledger), store_dir, work_dir] if path.exists()]
    if existing and not args.clean:
        raise ReleaseEvidenceRegistryError(f"Replay outputs already exist; pass --clean: {existing}")
    if args.clean:
        clean_outputs(args, store_dir, work_dir)


def bundle_args(args: argparse.Namespace, base_dir: Path, work_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        base_dir=str(base_dir),
        root_name=args.second_root_name,
        gates=str(base_dir / "text_release_gates.json"),
        report_md=None,
        dashboard=None,
        dashboard_validation=None,
        dashboard_comparison=None,
        history=None,
        history_validation=None,
        history_analysis=None,
        leaderboard_json=None,
        leaderboard_csv=None,
        current_release=None,
        run_json=None,
        memory_suite=None,
        memory_validation=None,
        memory_prompts_csv=None,
        memory_pairs_csv=None,
        output=str(work_dir / "second_text_release_evidence.tar.gz"),
        output_json=str(work_dir / "second_text_release_evidence_manifest.json"),
        allow_failed_gates=args.allow_failed_gates,
    )


def build_second_evidence(args: argparse.Namespace, base_dir: Path, work_dir: Path) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    bargs = bundle_args(args, base_dir, work_dir)
    gate_path = resolve_artifact(base_dir, bargs.gates, "text_release_gates.json")
    gates_payload = assert_valid_gates(gate_path, bargs.allow_failed_gates)
    bargs.gates = str(gate_path)
    records = collect_artifacts(bargs, base_dir, bargs.root_name)
    manifest = build_manifest(
        args=bargs,
        base_dir=base_dir,
        root_name=bargs.root_name,
        gates_payload=gates_payload,
        records=records,
    )
    archive_path = Path(bargs.output)
    manifest_path = Path(bargs.output_json)
    write_bundle(archive_path, bargs.root_name, records, manifest)
    manifest["archive_sha256"] = bundle_sha256_file(archive_path)
    manifest["archive_bytes"] = archive_path.stat().st_size
    save_json(manifest_path, manifest)

    validation_args = argparse.Namespace(
        manifest=str(manifest_path),
        archive=str(archive_path),
        output_json=None,
        allow_failed_gates=args.allow_failed_gates,
    )
    validation = validate_evidence(validation_args)
    if validation.get("valid") is not True:
        raise ReleaseEvidenceRegistryError(f"Second evidence bundle is not valid: {validation.get('errors')}")
    validation_path = work_dir / "second_text_release_evidence_validation.json"
    save_json(validation_path, validation)
    return {
        "manifest": manifest_path,
        "archive": archive_path,
        "validation": validation_path,
    }


def register_once(
    *,
    ledger_path: Path,
    store_dir: Path,
    manifest: Path,
    archive: Path,
    validation_json: Path | None,
    restore_json: Path | None,
    comparison_json: Path | None,
    name: str,
    allow_failed_gates: bool,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    ledger = load_ledger(ledger_path)
    entry_args = argparse.Namespace(
        manifest=str(manifest),
        archive=str(archive),
        validation_json=str(validation_json) if validation_json else None,
        restore_json=str(restore_json) if restore_json else None,
        comparison_json=str(comparison_json) if comparison_json else None,
        ledger=str(ledger_path),
        store_dir=str(store_dir),
        name=name,
        allow_failed_gates=allow_failed_gates,
    )
    entry = build_entry(entry_args, ledger)
    ledger, action = register_entry(ledger, entry)
    write_ledger_json(ledger_path, ledger)
    return action, ledger, entry


def optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def expected_artifact_checks(ledger: dict[str, Any]) -> int:
    total = 0
    for entry in ledger.get("entries", []):
        if not isinstance(entry, dict):
            continue
        total += 2
        for key in ["validation_json", "restore_json", "comparison_json"]:
            if entry.get(key):
                total += 1
    return total


def run(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger)
    store_dir = Path(args.store_dir) if args.store_dir else ledger_path.parent / "text_release_evidence_ledger_replay_store"
    work_dir = Path(args.work_dir)
    base_dir = Path(args.base_dir) if args.base_dir else Path(args.manifest).parent
    ensure_clean_outputs(args, store_dir, work_dir)

    manifest = Path(args.manifest)
    archive = Path(args.archive)
    validation_json = optional_path(args.validation_json)
    restore_json = optional_path(args.restore_json)
    comparison_json = optional_path(args.comparison_json)

    operations: list[dict[str, Any]] = []
    action, ledger, _entry = register_once(
        ledger_path=ledger_path,
        store_dir=store_dir,
        manifest=manifest,
        archive=archive,
        validation_json=validation_json,
        restore_json=restore_json,
        comparison_json=comparison_json,
        name=args.first_name,
        allow_failed_gates=args.allow_failed_gates,
    )
    operations.append({"name": args.first_name, "action": action, "entry_count": ledger["entry_count"], "chain_head": ledger["chain_head"]})

    first_head = ledger["chain_head"]
    action, ledger, _entry = register_once(
        ledger_path=ledger_path,
        store_dir=store_dir,
        manifest=manifest,
        archive=archive,
        validation_json=validation_json,
        restore_json=restore_json,
        comparison_json=comparison_json,
        name=args.update_name,
        allow_failed_gates=args.allow_failed_gates,
    )
    operations.append({"name": args.update_name, "action": action, "entry_count": ledger["entry_count"], "chain_head": ledger["chain_head"]})
    duplicate_update_preserved_count = ledger.get("entry_count") == 1

    second = build_second_evidence(args, base_dir, work_dir)
    action, ledger, _entry = register_once(
        ledger_path=ledger_path,
        store_dir=store_dir,
        manifest=second["manifest"],
        archive=second["archive"],
        validation_json=second["validation"],
        restore_json=None,
        comparison_json=None,
        name=args.second_name,
        allow_failed_gates=args.allow_failed_gates,
    )
    operations.append({"name": args.second_name, "action": action, "entry_count": ledger["entry_count"], "chain_head": ledger["chain_head"]})

    verify_chain(ledger)
    entries = ledger.get("entries") if isinstance(ledger.get("entries"), list) else []
    first_entry = entries[0] if len(entries) > 0 and isinstance(entries[0], dict) else {}
    second_entry = entries[1] if len(entries) > 1 and isinstance(entries[1], dict) else {}
    previous_hash_linked = (
        len(entries) == 2
        and first_entry.get("previous_hash") is None
        and second_entry.get("previous_hash") == first_entry.get("entry_hash")
        and ledger.get("chain_head") == second_entry.get("entry_hash")
    )
    distinct_evidence = (
        bool(first_entry.get("archive_sha256"))
        and first_entry.get("archive_sha256") != second_entry.get("archive_sha256")
        and first_entry.get("manifest_sha256") != second_entry.get("manifest_sha256")
    )
    replay_actions_ok = [op["action"] for op in operations] == ["inserted", "updated", "inserted"]

    ledger_validation_args = argparse.Namespace(
        ledger=str(ledger_path),
        expected_entries=2,
        require_artifacts=True,
        artifact_scope="all",
        fail_on_failed_gates=args.fail_on_failed_gates,
    )
    ledger_validation = validate_ledger(ledger_validation_args)
    expected_checks = expected_artifact_checks(ledger)
    artifact_checks_ok = int(ledger_validation.get("artifact_checks") or 0) == expected_checks

    report = {
        "valid": all(
            [
                replay_actions_ok,
                duplicate_update_preserved_count,
                previous_hash_linked,
                distinct_evidence,
                ledger_validation.get("valid") is True,
                artifact_checks_ok,
            ]
        ),
        "ledger": str(ledger_path),
        "store_dir": str(store_dir),
        "work_dir": str(work_dir),
        "operations": operations,
        "first_head_before_update": first_head,
        "duplicate_update_preserved_count": duplicate_update_preserved_count,
        "previous_hash_linked": previous_hash_linked,
        "distinct_evidence": distinct_evidence,
        "expected_artifact_checks": expected_checks,
        "ledger_validation": ledger_validation,
        "final_chain_head": ledger.get("chain_head"),
    }
    if args.output_json:
        save_json(Path(args.output_json), report)

    print("TextPy/SoA release evidence ledger replay test")
    print(f"ledger: {ledger_path}")
    print(f"entries: {ledger.get('entry_count')}")
    print(f"actions: {','.join(op['action'] for op in operations)}")
    print(f"duplicate_update_preserved_count: {duplicate_update_preserved_count}")
    print(f"previous_hash_linked: {previous_hash_linked}")
    print(f"distinct_evidence: {distinct_evidence}")
    print(f"artifact_checks: {ledger_validation.get('artifact_checks')} / {expected_checks}")
    print(f"chain_head: {ledger.get('chain_head')}")
    if args.output_json:
        print(f"Saved JSON: {args.output_json}")
    if not report["valid"]:
        print("")
        print("FAIL")
        if ledger_validation.get("errors"):
            for error in ledger_validation["errors"]:
                print(f"  {error}")
        return 1
    print("")
    print("RELEASE EVIDENCE LEDGER REPLAY TEST OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay TextPy/SoA release evidence ledger registration semantics.")
    parser.add_argument("--manifest", default="artifacts/releases/text_release_evidence_manifest.json")
    parser.add_argument("--archive", default="artifacts/releases/text_release_evidence.tar.gz")
    parser.add_argument("--validation-json", default="artifacts/releases/text_release_evidence_validation.json")
    parser.add_argument("--restore-json", default=None)
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--ledger", default="artifacts/releases/text_release_evidence_ledger_replay.json")
    parser.add_argument("--store-dir", default=None)
    parser.add_argument("--work-dir", default="artifacts/releases/text_release_evidence_ledger_replay")
    parser.add_argument("--output-json", default="artifacts/releases/text_release_evidence_ledger_replay_report.json")
    parser.add_argument("--second-root-name", default="text_release_evidence_replay_second")
    parser.add_argument("--first-name", default="replay_first")
    parser.add_argument("--update-name", default="replay_first_updated")
    parser.add_argument("--second-name", default="replay_second")
    parser.add_argument("--allow-failed-gates", action="store_true")
    parser.add_argument("--fail-on-failed-gates", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (ReleaseEvidenceBundleError, ReleaseEvidenceRegistryError, ReleaseEvidenceValidationError) as exc:
        raise SystemExit(str(exc)) from exc
