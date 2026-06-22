#!/usr/bin/env python3
"""Prove that release evidence ledger tampering is detected."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from register_text_release_evidence import ReleaseEvidenceRegistryError
from validate_text_release_evidence_ledger import load_json, validate, write_json


TAMPER_MODES = ("archive_sha256", "file_count", "chain_head")


def flip_hex_digest(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseEvidenceRegistryError("Expected a non-empty digest string to tamper.")
    first = value[0]
    replacement = "0" if first != "0" else "1"
    return replacement + value[1:]


def tamper_ledger(ledger: dict[str, Any], mode: str) -> dict[str, Any]:
    entries = ledger.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ReleaseEvidenceRegistryError("Ledger must contain at least one entry for tamper testing.")
    if mode == "chain_head":
        old_value = ledger.get("chain_head")
        new_value = flip_hex_digest(old_value)
        ledger["chain_head"] = new_value
        return {
            "mode": mode,
            "path": "chain_head",
            "old_value": old_value,
            "new_value": new_value,
            "expected_signal": "Ledger chain_head mismatch",
        }

    entry = entries[-1]
    if not isinstance(entry, dict):
        raise ReleaseEvidenceRegistryError("Latest ledger entry must be an object.")
    if mode == "archive_sha256":
        old_value = entry.get("archive_sha256")
        new_value = flip_hex_digest(old_value)
        entry["archive_sha256"] = new_value
        return {
            "mode": mode,
            "path": f"entries[{len(entries) - 1}].archive_sha256",
            "old_value": old_value,
            "new_value": new_value,
            "expected_signal": "Ledger entry hash mismatch",
        }
    if mode == "file_count":
        old_value = entry.get("file_count")
        if isinstance(old_value, bool) or not isinstance(old_value, int):
            raise ReleaseEvidenceRegistryError("Latest ledger entry file_count must be an integer.")
        new_value = old_value + 1
        entry["file_count"] = new_value
        return {
            "mode": mode,
            "path": f"entries[{len(entries) - 1}].file_count",
            "old_value": old_value,
            "new_value": new_value,
            "expected_signal": "Ledger entry hash mismatch",
        }
    raise ReleaseEvidenceRegistryError(f"Unsupported tamper mode: {mode}")


def validation_args(ledger: Path, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        ledger=str(ledger),
        expected_entries=args.expected_entries,
        require_artifacts=args.require_artifacts,
        artifact_scope=args.artifact_scope,
        fail_on_failed_gates=args.fail_on_failed_gates,
    )


def run(args: argparse.Namespace) -> int:
    source_ledger = Path(args.ledger)
    tampered_ledger = Path(args.output_ledger)
    output_json = Path(args.output_json) if args.output_json else None
    if source_ledger.resolve() == tampered_ledger.resolve():
        raise ReleaseEvidenceRegistryError("--output-ledger must be different from --ledger.")

    original_validation = validate(validation_args(source_ledger, args))
    if original_validation.get("valid") is not True:
        raise ReleaseEvidenceRegistryError(f"Source ledger is not valid: {original_validation.get('errors')}")

    ledger = load_json(source_ledger)
    tamper = tamper_ledger(ledger, args.mode)
    write_json(tampered_ledger, ledger)

    tampered_validation = validate(validation_args(tampered_ledger, args))
    errors = tampered_validation.get("errors") if isinstance(tampered_validation.get("errors"), list) else []
    expected_signal = tamper["expected_signal"]
    detected = tampered_validation.get("valid") is False
    expected_signal_found = any(expected_signal in str(error) for error in errors)

    report = {
        "valid": detected and expected_signal_found,
        "source_ledger": str(source_ledger),
        "tampered_ledger": str(tampered_ledger),
        "tamper": tamper,
        "original_validation": original_validation,
        "tampered_validation": tampered_validation,
        "detected": detected,
        "expected_signal_found": expected_signal_found,
    }
    if output_json:
        write_json(output_json, report)

    print("TextPy/SoA release evidence ledger tamper test")
    print(f"ledger: {source_ledger}")
    print(f"tampered_ledger: {tampered_ledger}")
    print(f"tamper_mode: {tamper['mode']}")
    print(f"tamper_path: {tamper['path']}")
    print(f"original_valid: {original_validation['valid']}")
    print(f"tampered_valid: {tampered_validation['valid']}")
    print(f"detected: {detected}")
    print(f"expected_signal_found: {expected_signal_found}")
    if output_json:
        print(f"Saved JSON: {output_json}")
    if not report["valid"]:
        print("")
        print("FAIL")
        for error in errors:
            print(f"  {error}")
        return 1
    print("")
    print("RELEASE EVIDENCE LEDGER TAMPER TEST OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tamper-test a TextPy/SoA release evidence ledger.")
    parser.add_argument("--ledger", default="artifacts/releases/text_release_evidence_ledger.json")
    parser.add_argument("--output-ledger", default="artifacts/releases/text_release_evidence_ledger_tampered.json")
    parser.add_argument("--output-json", default="artifacts/releases/text_release_evidence_ledger_tamper.json")
    parser.add_argument("--mode", choices=TAMPER_MODES, default="archive_sha256")
    parser.add_argument("--expected-entries", type=int, default=None)
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--artifact-scope", choices=["all", "latest", "none"], default="all")
    parser.add_argument("--fail-on-failed-gates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceRegistryError as exc:
        raise SystemExit(str(exc)) from exc
