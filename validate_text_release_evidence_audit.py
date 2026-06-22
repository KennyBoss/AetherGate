#!/usr/bin/env python3
"""Validate a TextPy/SoA release evidence audit rollup JSON artifact."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


class ReleaseEvidenceAuditValidationError(RuntimeError):
    pass


REQUIRED_CHECKS = [
    "evidence_manifest",
    "evidence_validation",
    "evidence_restore",
    "evidence_comparison",
    "ledger",
    "ledger_validation",
    "ledger_tamper",
    "ledger_replay",
    "ledger_self_comparison",
    "ledger_growth_comparison",
]

REQUIRED_SOURCES = [
    "manifest",
    "validation",
    "restore",
    "comparison",
    "ledger",
    "ledger_validation",
    "tamper",
    "replay",
    "ledger_comparison",
    "ledger_growth_comparison",
]

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseEvidenceAuditValidationError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceAuditValidationError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseEvidenceAuditValidationError(f"Expected JSON object: {path}")
    return payload


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_int(value: Any, label: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer.")
        return None
    return int(value)


def resolve_path(base: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, base / path, base / path.name]
    if not path.is_absolute():
        candidates.extend(parent / path for parent in base.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def check_names(checks: list[Any]) -> list[str]:
    return [check.get("name") for check in checks if isinstance(check, dict) and isinstance(check.get("name"), str)]


def check_by_name(checks: list[Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for check in checks:
        if isinstance(check, dict) and isinstance(check.get("name"), str):
            mapped[check["name"]] = check
    return mapped


def validate_check_shape(check: Any, index: int, errors: list[str]) -> dict[str, Any]:
    if not isinstance(check, dict):
        errors.append(f"checks[{index}] must be an object.")
        return {}
    prefix = f"checks[{index}]"
    if not isinstance(check.get("name"), str) or not check.get("name"):
        errors.append(f"{prefix}.name must be a non-empty string.")
    if not isinstance(check.get("path"), str) or not check.get("path"):
        errors.append(f"{prefix}.path must be a non-empty string.")
    if not isinstance(check.get("valid"), bool):
        errors.append(f"{prefix}.valid must be boolean.")
    if not isinstance(check.get("message"), str):
        errors.append(f"{prefix}.message must be a string.")
    if "metrics" in check and not isinstance(check.get("metrics"), dict):
        errors.append(f"{prefix}.metrics must be an object when present.")
    return check


def validate_core(payload: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    if payload.get("kind") != "text_release_evidence_audit":
        errors.append("kind must be text_release_evidence_audit.")
    summary = payload.get("summary")
    checks_raw = payload.get("checks")
    failures = payload.get("failures")
    sources = payload.get("sources")

    if not isinstance(summary, dict):
        errors.append("summary must be an object.")
        summary = {}
    checks = checks_raw if isinstance(checks_raw, list) else []
    if not isinstance(checks_raw, list):
        errors.append("checks must be a list.")
    if not isinstance(failures, list):
        errors.append("failures must be a list.")
        failures = []
    if not isinstance(sources, dict):
        errors.append("sources must be an object.")
        sources = {}

    for index, check in enumerate(checks):
        validate_check_shape(check, index, errors)

    names = check_names(checks)
    if len(names) != len(set(names)):
        errors.append("check names must be unique.")
    missing_checks = sorted(set(REQUIRED_CHECKS) - set(names))
    extra_checks = sorted(set(names) - set(REQUIRED_CHECKS))
    if missing_checks:
        errors.append(f"missing required checks: {missing_checks}")
    if extra_checks:
        errors.append(f"unexpected checks: {extra_checks}")

    missing_sources = sorted(set(REQUIRED_SOURCES) - set(sources))
    extra_sources = sorted(set(sources) - set(REQUIRED_SOURCES))
    if missing_sources:
        errors.append(f"missing required sources: {missing_sources}")
    if extra_sources:
        errors.append(f"unexpected sources: {extra_sources}")
    failed_checks = [check for check in checks if isinstance(check, dict) and check.get("valid") is not True]
    check_count = as_int(summary.get("check_count"), "summary.check_count", errors)
    failed_check_count = as_int(summary.get("failed_check_count"), "summary.failed_check_count", errors)
    source_entries = as_int(summary.get("source_entry_count"), "summary.source_entry_count", errors)
    replay_entries = as_int(summary.get("replay_entry_count"), "summary.replay_entry_count", errors)
    growth_delta = as_int(summary.get("growth_entry_count_delta"), "summary.growth_entry_count_delta", errors)
    artifact_checks = as_int(summary.get("artifact_check_count"), "summary.artifact_check_count", errors)
    if check_count is not None and check_count != len(checks):
        errors.append(f"summary.check_count must match checks length: {check_count} != {len(checks)}")
    if failed_check_count is not None and failed_check_count != len(failed_checks):
        errors.append(f"summary.failed_check_count must match failed checks: {failed_check_count} != {len(failed_checks)}")
    if source_entries != 1:
        errors.append(f"summary.source_entry_count must be 1: {source_entries}")
    if replay_entries != 2:
        errors.append(f"summary.replay_entry_count must be 2: {replay_entries}")
    if growth_delta != 1:
        errors.append(f"summary.growth_entry_count_delta must be 1: {growth_delta}")
    if artifact_checks is not None and artifact_checks < 10:
        errors.append(f"summary.artifact_check_count must be at least 10: {artifact_checks}")
    if summary.get("replay_actions") != ["inserted", "updated", "inserted"]:
        errors.append("summary.replay_actions must be inserted,updated,inserted.")
    if summary.get("tamper_detected") is not True:
        errors.append("summary.tamper_detected must be true.")
    for key in ["archive_sha256", "source_chain_head", "replay_chain_head"]:
        value = summary.get(key)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            errors.append(f"summary.{key} must be a 64-character lowercase SHA256 string.")
    expected_valid = not failed_checks and not failures
    if payload.get("valid") is not expected_valid:
        errors.append("valid must match failed checks and failures.")
    return {
        "summary": summary,
        "checks": checks,
        "failures": failures,
        "sources": sources,
    }


def source_payloads(base: Path, sources: dict[str, Any], errors: list[str]) -> tuple[dict[str, dict[str, Any]], int]:
    payloads: dict[str, dict[str, Any]] = {}
    artifact_checks = 0
    for name in REQUIRED_SOURCES:
        path = resolve_path(base, sources.get(name))
        if path is None:
            errors.append(f"sources.{name} path is missing.")
            continue
        if not path.exists():
            errors.append(f"sources.{name} does not exist: {path}")
            continue
        try:
            payloads[name] = load_json(path)
            artifact_checks += 1
        except ReleaseEvidenceAuditValidationError as exc:
            errors.append(str(exc))
    return payloads, artifact_checks


def validate_sources(
    *,
    audit_path: Path,
    summary: dict[str, Any],
    checks: list[Any],
    sources: dict[str, Any],
    errors: list[str],
) -> int:
    payloads, artifact_checks = source_payloads(audit_path.parent, sources, errors)
    if not payloads:
        return artifact_checks
    checks_by_name = check_by_name(checks)

    manifest = as_dict(payloads.get("manifest"))
    if manifest:
        if manifest.get("kind") != "text_release_evidence_bundle":
            errors.append("sources.manifest kind must be text_release_evidence_bundle.")
        if manifest.get("valid") is not True:
            errors.append("sources.manifest.valid must be true.")
        if manifest.get("archive_sha256") != summary.get("archive_sha256"):
            errors.append("summary.archive_sha256 must match sources.manifest.archive_sha256.")
        if int(manifest.get("file_count") or 0) != 16:
            errors.append("sources.manifest.file_count must be 16.")

    validation = as_dict(payloads.get("validation"))
    if validation and validation.get("valid") is not True:
        errors.append("sources.validation.valid must be true.")

    restore = as_dict(payloads.get("restore"))
    if restore and restore.get("valid") is not True:
        errors.append("sources.restore.valid must be true.")

    comparison = as_dict(payloads.get("comparison"))
    if comparison and as_list(comparison.get("failures")):
        errors.append("sources.comparison.failures must be empty.")

    ledger = as_dict(payloads.get("ledger"))
    if ledger:
        if ledger.get("kind") != "text_release_evidence_ledger":
            errors.append("sources.ledger kind must be text_release_evidence_ledger.")
        if int(ledger.get("entry_count") or 0) != 1:
            errors.append("sources.ledger.entry_count must be 1.")
        if ledger.get("chain_head") != summary.get("source_chain_head"):
            errors.append("summary.source_chain_head must match sources.ledger.chain_head.")

    ledger_validation = as_dict(payloads.get("ledger_validation"))
    if ledger_validation and ledger_validation.get("valid") is not True:
        errors.append("sources.ledger_validation.valid must be true.")

    tamper = as_dict(payloads.get("tamper"))
    if tamper:
        if tamper.get("valid") is not True:
            errors.append("sources.tamper.valid must be true.")
        if tamper.get("detected") is not True or tamper.get("expected_signal_found") is not True:
            errors.append("sources.tamper must prove detection and expected signal.")

    replay = as_dict(payloads.get("replay"))
    if replay:
        replay_validation = as_dict(replay.get("ledger_validation"))
        if replay.get("valid") is not True:
            errors.append("sources.replay.valid must be true.")
        actions = [item.get("action") for item in as_list(replay.get("operations")) if isinstance(item, dict)]
        if actions != ["inserted", "updated", "inserted"]:
            errors.append("sources.replay actions must be inserted,updated,inserted.")
        if replay.get("final_chain_head") != summary.get("replay_chain_head"):
            errors.append("summary.replay_chain_head must match sources.replay.final_chain_head.")
        if replay_validation.get("entry_count") != summary.get("replay_entry_count"):
            errors.append("summary.replay_entry_count must match sources.replay ledger_validation.entry_count.")

    ledger_comparison = as_dict(payloads.get("ledger_comparison"))
    ledger_comparison_block = as_dict(ledger_comparison.get("comparison"))
    if ledger_comparison:
        if as_list(ledger_comparison.get("failures")):
            errors.append("sources.ledger_comparison.failures must be empty.")
        if ledger_comparison_block.get("same_chain_head") is not True:
            errors.append("sources.ledger_comparison must be a self-same comparison.")

    growth = as_dict(payloads.get("ledger_growth_comparison"))
    growth_block = as_dict(growth.get("comparison"))
    if growth:
        if as_list(growth.get("failures")):
            errors.append("sources.ledger_growth_comparison.failures must be empty.")
        if growth_block.get("entry_count_delta") != summary.get("growth_entry_count_delta"):
            errors.append("summary.growth_entry_count_delta must match growth comparison.")

    expected_valid_checks = {
        "evidence_manifest",
        "evidence_validation",
        "evidence_restore",
        "evidence_comparison",
        "ledger",
        "ledger_validation",
        "ledger_tamper",
        "ledger_replay",
        "ledger_self_comparison",
        "ledger_growth_comparison",
    }
    for name in expected_valid_checks:
        check = checks_by_name.get(name, {})
        if check.get("valid") is not True:
            errors.append(f"check {name} must be valid.")
    return artifact_checks


def validate(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = Path(args.audit)
    payload = load_json(audit_path)
    errors: list[str] = []
    core = validate_core(payload, errors)
    source_artifact_checks = 0
    if args.require_sources:
        source_artifact_checks = validate_sources(
            audit_path=audit_path,
            summary=as_dict(core.get("summary")),
            checks=as_list(core.get("checks")),
            sources=as_dict(core.get("sources")),
            errors=errors,
        )
    valid = not errors
    summary = as_dict(core.get("summary"))
    checks = as_list(core.get("checks"))
    return {
        "valid": valid,
        "audit": str(audit_path),
        "release_name": summary.get("release_name"),
        "check_count": len(checks),
        "failed_check_count": len([check for check in checks if isinstance(check, dict) and check.get("valid") is not True]),
        "failure_count": len(as_list(core.get("failures"))),
        "artifact_check_count": summary.get("artifact_check_count"),
        "source_artifact_checks": source_artifact_checks,
        "source_count": len(as_dict(core.get("sources"))),
        "errors": errors,
    }


def run(args: argparse.Namespace) -> int:
    result = validate(args)
    if args.output_json:
        save_json(args.output_json, result)
    print("TextPy/SoA release evidence audit validator")
    print(f"audit: {result['audit']}")
    print(f"valid: {result['valid']}")
    print(f"release_name: {result['release_name']}")
    print(f"checks: {result['check_count']}")
    print(f"failed_checks: {result['failed_check_count']}")
    print(f"failures: {result['failure_count']}")
    print(f"artifact_checks: {result['artifact_check_count']}")
    print(f"source_artifact_checks: {result['source_artifact_checks']}")
    if args.output_json:
        print(f"Saved JSON: {args.output_json}")
    if not result["valid"]:
        print("")
        print("FAIL")
        for error in result["errors"]:
            print(f"  {error}")
        return 1
    print("")
    print("RELEASE EVIDENCE AUDIT VALIDATION OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a release evidence audit JSON artifact.")
    parser.add_argument("--audit", default="artifacts/releases/text_release_evidence_audit.json")
    parser.add_argument("--output-json", default="artifacts/releases/text_release_evidence_audit_validation.json")
    parser.add_argument("--require-sources", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceAuditValidationError as exc:
        raise SystemExit(str(exc)) from exc
