#!/usr/bin/env python3
"""Summarize TextPy/SoA release evidence proof artifacts into one audit verdict."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReleaseEvidenceAuditError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseEvidenceAuditError(f"Missing release evidence audit artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseEvidenceAuditError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseEvidenceAuditError(f"Expected a JSON object: {path}")
    return payload


def save_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(path: str, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def check_record(name: str, path: Path, ok: bool, message: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "valid": bool(ok),
        "message": message,
        "metrics": metrics or {},
    }


def short(value: Any) -> str | None:
    if not value:
        return None
    return str(value)[:12]


def evidence_comparison_ok(payload: dict[str, Any]) -> bool:
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), dict) else {}
    required_true = [
        "same_archive_sha256",
        "same_file_set",
        "same_file_sha",
        "same_file_bytes",
        "same_archive_names",
        "same_valid",
    ]
    return (
        not as_list(payload.get("failures"))
        and all(comparison.get(key) is True for key in required_true)
        and int(comparison.get("file_count_delta") or 0) == 0
        and int(comparison.get("failed_gate_count_delta") or 0) == 0
    )


def ledger_self_comparison_ok(payload: dict[str, Any]) -> bool:
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), dict) else {}
    required_true = [
        "same_chain_head",
        "same_entry_count",
        "same_entry_hash_set",
        "same_archive_sha_set",
        "same_manifest_sha_set",
    ]
    changed_common = comparison.get("changed_common_entries")
    return (
        not as_list(payload.get("failures"))
        and all(comparison.get(key) is True for key in required_true)
        and int(comparison.get("entry_count_delta") or 0) == 0
        and (not isinstance(changed_common, list) or not changed_common)
    )


def ledger_growth_comparison_ok(payload: dict[str, Any]) -> bool:
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), dict) else {}
    return (
        not as_list(payload.get("failures"))
        and comparison.get("same_chain_head") is False
        and comparison.get("same_entry_count") is False
        and comparison.get("same_entry_hash_set") is False
        and comparison.get("same_archive_sha_set") is False
        and comparison.get("same_manifest_sha_set") is False
        and int(comparison.get("entry_count_delta") or 0) == 1
    )


def replay_ok(payload: dict[str, Any]) -> bool:
    operations = payload.get("operations") if isinstance(payload.get("operations"), list) else []
    actions = [op.get("action") for op in operations if isinstance(op, dict)]
    ledger_validation = payload.get("ledger_validation") if isinstance(payload.get("ledger_validation"), dict) else {}
    return (
        payload.get("valid") is True
        and actions == ["inserted", "updated", "inserted"]
        and payload.get("duplicate_update_preserved_count") is True
        and payload.get("previous_hash_linked") is True
        and payload.get("distinct_evidence") is True
        and ledger_validation.get("valid") is True
        and int(ledger_validation.get("entry_count") or 0) == 2
        and int(ledger_validation.get("artifact_checks") or -1) == int(payload.get("expected_artifact_checks") or -2)
    )


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": Path(args.manifest),
        "validation": Path(args.validation_json),
        "restore": Path(args.restore_json),
        "comparison": Path(args.comparison_json),
        "ledger": Path(args.ledger),
        "ledger_validation": Path(args.ledger_validation_json),
        "tamper": Path(args.tamper_json),
        "replay": Path(args.replay_json),
        "ledger_comparison": Path(args.ledger_comparison_json),
        "ledger_growth_comparison": Path(args.ledger_growth_comparison_json),
    }
    manifest = load_json(paths["manifest"])
    validation = load_json(paths["validation"])
    restore = load_json(paths["restore"])
    comparison = load_json(paths["comparison"])
    ledger = load_json(paths["ledger"])
    ledger_validation = load_json(paths["ledger_validation"])
    tamper = load_json(paths["tamper"])
    replay = load_json(paths["replay"])
    ledger_comparison = load_json(paths["ledger_comparison"])
    ledger_growth = load_json(paths["ledger_growth_comparison"])

    replay_validation = replay.get("ledger_validation") if isinstance(replay.get("ledger_validation"), dict) else {}
    evidence_cmp = comparison.get("comparison") if isinstance(comparison.get("comparison"), dict) else {}
    ledger_cmp = ledger_comparison.get("comparison") if isinstance(ledger_comparison.get("comparison"), dict) else {}
    ledger_growth_cmp = ledger_growth.get("comparison") if isinstance(ledger_growth.get("comparison"), dict) else {}
    tamper_validation = tamper.get("tampered_validation") if isinstance(tamper.get("tampered_validation"), dict) else {}

    checks = [
        check_record(
            "evidence_manifest",
            paths["manifest"],
            manifest.get("kind") == "text_release_evidence_bundle"
            and manifest.get("valid") is True
            and int(manifest.get("file_count") or 0) == 16
            and bool(manifest.get("archive_sha256"))
            and int(manifest.get("archive_bytes") or 0) > 0,
            "Manifest is a valid evidence bundle with archive checksum and 16 source files.",
            {
                "release_name": manifest.get("release_name"),
                "archive_sha256": short(manifest.get("archive_sha256")),
                "archive_bytes": manifest.get("archive_bytes"),
                "file_count": manifest.get("file_count"),
            },
        ),
        check_record(
            "evidence_validation",
            paths["validation"],
            validation.get("valid") is True
            and int(validation.get("file_count") or 0) == 16
            and int(validation.get("checked_files") or 0) == 16
            and int(validation.get("member_count") or 0) == 17
            and validation.get("has_inner_manifest") is True
            and not as_list(validation.get("errors")),
            "Bundle validator recomputed all file/archive checks and found the inner manifest.",
            {
                "checked_files": validation.get("checked_files"),
                "member_count": validation.get("member_count"),
                "has_inner_manifest": validation.get("has_inner_manifest"),
            },
        ),
        check_record(
            "evidence_restore",
            paths["restore"],
            restore.get("valid") is True
            and int(restore.get("file_count") or 0) == 16
            and int(restore.get("member_count") or 0) == 17
            and (restore.get("validation") if isinstance(restore.get("validation"), dict) else {}).get("valid") is True,
            "Archive restores into a clean tree and validates after restore.",
            {
                "file_count": restore.get("file_count"),
                "member_count": restore.get("member_count"),
                "restored_manifest": restore.get("restored_manifest"),
            },
        ),
        check_record(
            "evidence_comparison",
            paths["comparison"],
            evidence_comparison_ok(comparison),
            "Evidence bundle self-comparison has no archive/file/gate drift.",
            {
                "same_archive_sha256": evidence_cmp.get("same_archive_sha256"),
                "file_count_delta": evidence_cmp.get("file_count_delta"),
                "failed_gate_count_delta": evidence_cmp.get("failed_gate_count_delta"),
                "common_labels": len(evidence_cmp.get("common_labels") if isinstance(evidence_cmp.get("common_labels"), list) else []),
            },
        ),
        check_record(
            "ledger",
            paths["ledger"],
            ledger.get("kind") == "text_release_evidence_ledger"
            and int(ledger.get("entry_count") or 0) == 1
            and bool(ledger.get("chain_head"))
            and isinstance(ledger.get("entries"), list)
            and len(ledger.get("entries")) == 1
            and isinstance(ledger["entries"][0], dict)
            and ledger["entries"][0].get("entry_hash") == ledger.get("chain_head"),
            "Source ledger has one entry and a chain head matching the entry hash.",
            {
                "entry_count": ledger.get("entry_count"),
                "chain_head": short(ledger.get("chain_head")),
            },
        ),
        check_record(
            "ledger_validation",
            paths["ledger_validation"],
            ledger_validation.get("valid") is True
            and int(ledger_validation.get("entry_count") or 0) == 1
            and ledger_validation.get("last_entry_hash") == ledger_validation.get("chain_head")
            and int(ledger_validation.get("artifact_checks") or 0) >= 4
            and not as_list(ledger_validation.get("errors")),
            "Ledger validator proves hash-chain structure and linked artifact hashes.",
            {
                "entry_count": ledger_validation.get("entry_count"),
                "artifact_checks": ledger_validation.get("artifact_checks"),
                "chain_head": short(ledger_validation.get("chain_head")),
            },
        ),
        check_record(
            "ledger_tamper",
            paths["tamper"],
            tamper.get("valid") is True
            and tamper.get("detected") is True
            and tamper.get("expected_signal_found") is True
            and tamper_validation.get("valid") is False,
            "A modified ledger copy is rejected with the expected hash-chain signal.",
            {
                "mode": (tamper.get("tamper") or {}).get("mode") if isinstance(tamper.get("tamper"), dict) else None,
                "detected": tamper.get("detected"),
                "expected_signal_found": tamper.get("expected_signal_found"),
            },
        ),
        check_record(
            "ledger_replay",
            paths["replay"],
            replay_ok(replay),
            "Replay proves insert/update/append semantics, chaining, distinct evidence, and artifact checks.",
            {
                "actions": [
                    op.get("action")
                    for op in (replay.get("operations") if isinstance(replay.get("operations"), list) else [])
                    if isinstance(op, dict)
                ],
                "entry_count": replay_validation.get("entry_count"),
                "artifact_checks": replay_validation.get("artifact_checks"),
                "final_chain_head": short(replay.get("final_chain_head")),
            },
        ),
        check_record(
            "ledger_self_comparison",
            paths["ledger_comparison"],
            ledger_self_comparison_ok(ledger_comparison),
            "Replay ledger compared with itself has no chain, entry, archive, or manifest drift.",
            {
                "same_chain_head": ledger_cmp.get("same_chain_head"),
                "same_entry_count": ledger_cmp.get("same_entry_count"),
                "entry_count_delta": ledger_cmp.get("entry_count_delta"),
            },
        ),
        check_record(
            "ledger_growth_comparison",
            paths["ledger_growth_comparison"],
            ledger_growth_comparison_ok(ledger_growth),
            "One-entry source ledger versus replay ledger reports one-entry growth and expected drift.",
            {
                "same_chain_head": ledger_growth_cmp.get("same_chain_head"),
                "same_entry_count": ledger_growth_cmp.get("same_entry_count"),
                "entry_count_delta": ledger_growth_cmp.get("entry_count_delta"),
                "added_entry_hashes": len(
                    ledger_growth_cmp.get("added_entry_hashes")
                    if isinstance(ledger_growth_cmp.get("added_entry_hashes"), list)
                    else []
                ),
            },
        ),
    ]
    failed = [check for check in checks if check["valid"] is not True]
    summary = {
        "release_name": manifest.get("release_name"),
        "archive_sha256": manifest.get("archive_sha256"),
        "source_entry_count": ledger.get("entry_count"),
        "replay_entry_count": replay_validation.get("entry_count"),
        "replay_actions": [
            op.get("action")
            for op in (replay.get("operations") if isinstance(replay.get("operations"), list) else [])
            if isinstance(op, dict)
        ],
        "source_chain_head": ledger.get("chain_head"),
        "replay_chain_head": replay.get("final_chain_head") or replay_validation.get("chain_head"),
        "growth_entry_count_delta": ledger_growth_cmp.get("entry_count_delta"),
        "tamper_detected": tamper.get("detected"),
        "artifact_check_count": int(validation.get("checked_files") or 0)
        + int(ledger_validation.get("artifact_checks") or 0)
        + int(replay_validation.get("artifact_checks") or 0),
        "check_count": len(checks),
        "failed_check_count": len(failed),
    }
    return {
        "created_at": timestamp(),
        "kind": "text_release_evidence_audit",
        "valid": not failed,
        "summary": summary,
        "checks": checks,
        "failures": [{"check": check["name"], "message": check["message"]} for check in failed],
        "sources": {name: str(path) for name, path in paths.items()},
    }


def cell(value: Any) -> str:
    text = "-" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    checks = payload["checks"]
    failures = payload["failures"]
    verdict = "PASS" if payload.get("valid") is True else "FAIL"
    lines = [
        "# TextPy/SoA Release Evidence Audit",
        "",
        "## Verdict",
        "",
        *table(
            ["Field", "Value"],
            [
                ["verdict", verdict],
                ["release_name", summary.get("release_name")],
                ["check_count", summary.get("check_count")],
                ["failed_check_count", summary.get("failed_check_count")],
                ["source_entry_count", summary.get("source_entry_count")],
                ["replay_entry_count", summary.get("replay_entry_count")],
                ["growth_entry_count_delta", summary.get("growth_entry_count_delta")],
                ["tamper_detected", summary.get("tamper_detected")],
                ["artifact_check_count", summary.get("artifact_check_count")],
                ["created_at", payload.get("created_at")],
            ],
        ),
        "",
        "## Checks",
        "",
        *table(
            ["Check", "Status", "Message", "Source"],
            [
                [
                    check.get("name"),
                    "PASS" if check.get("valid") is True else "FAIL",
                    check.get("message"),
                    check.get("path"),
                ]
                for check in checks
            ],
        ),
        "",
        "## Failures",
        "",
    ]
    if failures:
        lines.extend([f"- `{item.get('check')}`: {item.get('message')}" for item in failures])
    else:
        lines.append("- No release evidence audit failures.")
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "```bash",
            "make summarize-release-evidence-audit-demo",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    payload = summarize(args)
    if args.output_json:
        save_json(args.output_json, payload)
    if args.output_md:
        save_text(args.output_md, render_markdown(payload))

    summary = payload["summary"]
    print("TextPy/SoA release evidence audit")
    print(f"release_name: {summary['release_name']}")
    print(f"valid: {payload['valid']}")
    print(f"checks: {summary['check_count']}")
    print(f"failed_checks: {summary['failed_check_count']}")
    print(f"source_entries: {summary['source_entry_count']}")
    print(f"replay_entries: {summary['replay_entry_count']}")
    print(f"growth_entry_count_delta: {summary['growth_entry_count_delta']}")
    print(f"tamper_detected: {summary['tamper_detected']}")
    print(f"artifact_checks: {summary['artifact_check_count']}")
    if args.output_json:
        print(f"Saved JSON: {args.output_json}")
    if args.output_md:
        print(f"Saved Markdown: {args.output_md}")

    if not payload["valid"]:
        print("")
        print("FAIL")
        for failure in payload["failures"]:
            print(f"  {failure['check']}: {failure['message']}")
        return 1
    print("")
    print("RELEASE EVIDENCE AUDIT OK")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize release evidence proof artifacts into one audit verdict.")
    parser.add_argument("--manifest", default="artifacts/releases/text_release_evidence_manifest.json")
    parser.add_argument("--validation-json", default="artifacts/releases/text_release_evidence_validation.json")
    parser.add_argument("--restore-json", default="artifacts/restored/text_release_evidence_restore.json")
    parser.add_argument("--comparison-json", default="artifacts/releases/text_release_evidence_comparison.json")
    parser.add_argument("--ledger", default="artifacts/releases/text_release_evidence_ledger.json")
    parser.add_argument("--ledger-validation-json", default="artifacts/releases/text_release_evidence_ledger_validation.json")
    parser.add_argument("--tamper-json", default="artifacts/releases/text_release_evidence_ledger_tamper.json")
    parser.add_argument("--replay-json", default="artifacts/releases/text_release_evidence_ledger_replay_report.json")
    parser.add_argument("--ledger-comparison-json", default="artifacts/releases/text_release_evidence_ledger_comparison.json")
    parser.add_argument(
        "--ledger-growth-comparison-json",
        default="artifacts/releases/text_release_evidence_ledger_growth_comparison.json",
    )
    parser.add_argument("--output-json", default="artifacts/releases/text_release_evidence_audit.json")
    parser.add_argument("--output-md", default="artifacts/releases/text_release_evidence_audit.md")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except ReleaseEvidenceAuditError as exc:
        raise SystemExit(str(exc)) from exc
